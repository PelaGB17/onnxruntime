// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed_sm90.h"
#include "contrib_ops/cuda/moe/moe_packed.h"
#include "contrib_ops/cuda/quantization/gguf_sm90_wgmma.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Dynamic shared memory layout per 256-thread (2-warpgroup) SM90 WGMMA CTA:
//   [0 .. 1023]          : up to 1024B alignment padding so s_B starts on a 1024B boundary
//   s_B[2][2][128 * 8]   : 65,536B double-buffered K-major B128-swizzled [128 routes, 128 BF16]
//   s_lut[2048] (FC1)    : 16,384B pre-expanded bf16x2 codebook LUT (IQ2_S 16KB, IQ3_XXS 8KB; FC2 8KB)
//   s_routes[128]        : 512B sorted route indices
//   s_token_rows[128]    : 512B token row indices (FC1)
// FC1 Total = 83,968B (82.0 KB -> 200.7 KB Hopper carveout, preserving 55 KB L1 cache!), FC2 Total = 75,776B.
constexpr size_t kMoEPackedSm90Fc1SmemBytes = 65536 + 16384 + 1024 + 1024;
constexpr size_t kMoEPackedSm90Fc2SmemBytes = 65536 + 8192 + 1024 + 1024;

__device__ __forceinline__ char* AlignSmem1024(char* smem_raw) {
  const uint32_t raw_saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  const uint32_t aligned_saddr = (raw_saddr + 1023u) & ~1023u;
  return smem_raw + (aligned_saddr - raw_saddr);
}

__device__ __forceinline__ int SelectSm90WgmmaRouteN(int count) {
  if (count <= 16) return 16;
  if (count <= 32) return 32;
  if (count <= 64) return 64;
  if (count <= 96) return 96;
  return 128;
}

template <int kGateUpType, bool kAligned = false>
__global__ __launch_bounds__(256, 2) void MoEPackedFc1SwiGLUSm90WgmmaKernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    __nv_bfloat16* __restrict__ fc1_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= __ldg(d_num_tiles)) return;

  using namespace onnxruntime::contrib::gguf_quant;
  extern __shared__ char smem_raw[];
  char* smem_aligned = AlignSmem1024(smem_raw);
  uint4* s_B = reinterpret_cast<uint4*>(smem_aligned);
  uint64_t* s_lut = reinterpret_cast<uint64_t*>(smem_aligned + 65536);
  int* s_routes = reinterpret_cast<int*>(s_lut + 2048);
  int* s_token_rows = s_routes + 128;

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count;
  const int expert = tile.expert;
  const int n_inst = SelectSm90WgmmaRouteN(count);
  const int num_subtiles = (count + 7) >> 3;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 128) {
    const int r = (tid < count) ? __ldg(&sorted_routes[tile.start_route + tid]) : 0;
    s_routes[tid] = r;
    s_token_rows[tid] = r / p.k;
  }
  if constexpr (kGateUpType == kGgmlTypeIQ2_XXS) {
    if (tid < 256) s_lut[tid] = __ldg(iq2xxs_grid + tid);
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_XS) {
    for (int i = tid; i < 512; i += 256) s_lut[i] = __ldg(iq2xs_grid + i);
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_S) {
    if constexpr (kAligned) {
      uint4* s_lut_iq2s = reinterpret_cast<uint4*>(s_lut);
      for (int i = tid; i < 1024; i += 256) {
        s_lut_iq2s[i] = gguf_cuda::PackU8OctetToBf16x2QuadDev(__ldg(iq2s_grid + i));
      }
    } else {
      for (int i = tid; i < 1024; i += 256) s_lut[i] = __ldg(iq2s_grid + i);
    }
  } else if constexpr (kGateUpType == kGgmlTypeIQ3_XXS) {
    if constexpr (kAligned) {
      uint2* s_lut_iq3 = reinterpret_cast<uint2*>(s_lut);
      if (tid < 256) {
        const uint2 mag = gguf_cuda::PackU8QuadToBf16x2PairDev(__ldg(iq3xxs_grid + tid));
        s_lut_iq3[tid * 4 + 0] = mag;
        s_lut_iq3[tid * 4 + 1] = mag;
        s_lut_iq3[tid * 4 + 2] = mag;
        s_lut_iq3[tid * 4 + 3] = mag;
      }
    } else {
      uint32_t* s_lut32 = reinterpret_cast<uint32_t*>(s_lut);
      if (tid < 256) s_lut32[tid] = __ldg(iq3xxs_grid + tid);
    }
  }
  __syncthreads();

  const bool input_aligned_16b = ((reinterpret_cast<uintptr_t>(input) & 15u) == 0) && ((p.hidden_size & 7) == 0);
  const int stage_u4_total = n_inst * 16;
  auto stage_x_chunk = [&](int k_base, int stg) {
    uint4* s_tile0 = s_B + (stg * 2 + 0) * 1024;
    uint4* s_tile1 = s_B + (stg * 2 + 1) * 1024;
    for (int idx = tid; idx < stage_u4_total; idx += 256) {
      const int m = idx >> 4;
      const int u4_col16 = idx & 15;
      const bool pred = (m < count);
      const int token_row = pred ? s_token_rows[m] : 0;
      const __nv_bfloat16* src_ptr =
          input + static_cast<int64_t>(token_row) * p.hidden_size + k_base + (u4_col16 << 3);
      gguf_sm90::CpAsyncB128Store16B(
          (u4_col16 & 8) ? s_tile1 : s_tile0, m, u4_col16 & 7, src_ptr, pred, input_aligned_16b);
    }
    gguf_sm90::CpAsyncCommitGroup();
  };

  stage_x_chunk(0, 0);

  const int warp = tid >> 5;
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int wg_ch_base = (static_cast<int>(blockIdx.x) * 2 + wg) * 32;
  const bool wg_valid = (wg_ch_base < p.inter_size);
  const int j_out = (static_cast<int>(blockIdx.x) * 8 + warp) * 8 + g;
  const bool valid_row = (j_out < p.inter_size);

  const int64_t exp_row_idx = static_cast<int64_t>(expert) * p.inter_size + (valid_row ? j_out : 0);
  const uint8_t* gate_row_ptr = gate_weights + exp_row_idx * p.gate_row_bytes;
  const uint8_t* up_row_ptr = up_weights + exp_row_idx * p.up_row_bytes;
  const bool aligned_2b = ((reinterpret_cast<uintptr_t>(gate_weights) & 1u) == 0) &&
                          ((reinterpret_cast<uintptr_t>(up_weights) & 1u) == 0) &&
                          ((p.gate_row_bytes & 1) == 0) && ((p.up_row_bytes & 1) == 0);

  float acc[16][2][2];
#pragma unroll
  for (int st = 0; st < 16; ++st) {
    acc[st][0][0] = 0.0f;
    acc[st][0][1] = 0.0f;
    acc[st][1][0] = 0.0f;
    acc[st][1][1] = 0.0f;
  }

  int stage = 0;
  for (int k_base = 0; k_base < p.hidden_size; k_base += 128, stage ^= 1) {
    gguf_sm90::CpAsyncWaitGroup<0>();
    __syncthreads();
    gguf_sm90::FenceAsyncSharedProxy();
    if (k_base + 128 < p.hidden_size) {
      stage_x_chunk(k_base + 128, stage ^ 1);
    }

    if (wg_valid) {
      const uint64_t desc_b0 = gguf_sm90::MakeGmmaDescB128KMajor(s_B + (stage * 2 + 0) * 1024);
      const uint64_t desc_b1 = gguf_sm90::MakeGmmaDescB128KMajor(s_B + (stage * 2 + 1) * 1024);
      const int half_idx = k_base >> 7;
      const int sb256 = half_idx >> 1;
      const int ip = half_idx & 1;

      uint4 et_buf[2];
      uint4 ot_buf[2];
      gguf_sm90::FenceWgmmaAcc(acc, num_subtiles);

      if constexpr (kAligned &&
                    (kGateUpType == kGgmlTypeIQ3_XXS || kGateUpType == kGgmlTypeIQ2_S ||
                     kGateUpType == kGgmlTypeIQ4_XS)) {
        (void)aligned_2b;
        if constexpr (true) {
          if constexpr (kGateUpType == kGgmlTypeIQ3_XXS) {
            const uint2* s_lut_iq3 = reinterpret_cast<const uint2*>(s_lut);
            const uint8_t* g_blk = gate_row_ptr + static_cast<int64_t>(sb256) * 98;
            const uint8_t* u_blk = up_row_ptr + static_cast<int64_t>(sb256) * 98;
            const float dg = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(g_blk))) * 0.5f;
            const float du = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(u_blk))) * 0.5f;
            const int my_ib = ip * 4 + t;
            const auto* ag16 = reinterpret_cast<const uint16_t*>(g_blk + 66 + 4 * my_ib);
            const auto* au16 = reinterpret_cast<const uint16_t*>(u_blk + 66 + 4 * my_ib);
            const uint32_t my_aux_g =
                static_cast<uint32_t>(__ldg(ag16)) | (static_cast<uint32_t>(__ldg(ag16 + 1)) << 16);
            const uint32_t my_aux_u =
                static_cast<uint32_t>(__ldg(au16)) | (static_cast<uint32_t>(__ldg(au16 + 1)) << 16);
            const auto* gq16 = reinterpret_cast<const uint16_t*>(g_blk + 2 + 8 * my_ib);
            const auto* uq16 = reinterpret_cast<const uint16_t*>(u_blk + 2 + 8 * my_ib);
            const uint32_t my_gq_lo =
                static_cast<uint32_t>(__ldg(gq16 + 0)) | (static_cast<uint32_t>(__ldg(gq16 + 1)) << 16);
            const uint32_t my_gq_hi =
                static_cast<uint32_t>(__ldg(gq16 + 2)) | (static_cast<uint32_t>(__ldg(gq16 + 3)) << 16);
            const uint32_t my_uq_lo =
                static_cast<uint32_t>(__ldg(uq16 + 0)) | (static_cast<uint32_t>(__ldg(uq16 + 1)) << 16);
            const uint32_t my_uq_hi =
                static_cast<uint32_t>(__ldg(uq16 + 2)) | (static_cast<uint32_t>(__ldg(uq16 + 3)) << 16);
            const int q16_shift = 16 * (t & 1);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const uint32_t aux_g = __shfl_sync(0xFFFFFFFFu, my_aux_g, q, 4);
              const uint32_t aux_u = __shfl_sync(0xFFFFFFFFu, my_aux_u, q, 4);
              const uint32_t gq_lo = __shfl_sync(0xFFFFFFFFu, my_gq_lo, q, 4);
              const uint32_t gq_hi = __shfl_sync(0xFFFFFFFFu, my_gq_hi, q, 4);
              const uint32_t uq_lo = __shfl_sync(0xFFFFFFFFu, my_uq_lo, q, 4);
              const uint32_t uq_hi = __shfl_sync(0xFFFFFFFFu, my_uq_hi, q, 4);
              const float dbg = dg * __fadd_rn(__uint_as_float(0x4B000000u | (aux_g >> 28)), -8388607.5f);
              const float dbu = du * __fadd_rn(__uint_as_float(0x4B000000u | (aux_u >> 28)), -8388607.5f);
              const uint32_t sg = gguf_cuda::Iq2xsSignsDev(aux_g >> (7 * t));
              const uint32_t su = gguf_cuda::Iq2xsSignsDev(aux_u >> (7 * t));
              const uint32_t q2_g = (((t & 2) ? gq_hi : gq_lo) >> q16_shift) & 0xFFFFu;
              const uint32_t q2_u = (((t & 2) ? uq_hi : uq_lo) >> q16_shift) & 0xFFFFu;
              const uint2 g1 = s_lut_iq3[((q2_g & 0xFFu) << 2) | t];
              const uint2 g2 = s_lut_iq3[((q2_g >> 8) << 2) | t];
              const uint2 u1 = s_lut_iq3[((q2_u & 0xFFu) << 2) | t];
              const uint2 u2 = s_lut_iq3[((q2_u >> 8) << 2) | t];
              const uint4 e_raw = valid_row ? gguf_cuda::PackBf16Mag8Sign8ToBf16x2(make_uint4(g1.x, g1.y, g2.x, g2.y), sg, dbg)
                                            : make_uint4(0u, 0u, 0u, 0u);
              const uint4 o_raw = valid_row ? gguf_cuda::PackBf16Mag8Sign8ToBf16x2(make_uint4(u1.x, u1.y, u2.x, u2.y), su, dbu)
                                            : make_uint4(0u, 0u, 0u, 0u);
              const int slot = q & 1;
              if (q >= 2) {
                gguf_sm90::WgmmaWaitGroup<1>();
                gguf_sm90::FenceWgmmaU4(et_buf[slot]);
                gguf_sm90::FenceWgmmaU4(ot_buf[slot]);
              }
              et_buf[slot] = gguf_sm90::TransposeQuad4x4(e_raw, t);
              ot_buf[slot] = gguf_sm90::TransposeQuad4x4(o_raw, t);
              const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
              gguf_sm90::WgmmaFence();
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
              gguf_sm90::WgmmaCommitGroup();
            }
          } else if constexpr (kGateUpType == kGgmlTypeIQ4_XS) {
            const uint8_t* g_blk = gate_row_ptr + static_cast<int64_t>(sb256) * 136;
            const uint8_t* u_blk = up_row_ptr + static_cast<int64_t>(sb256) * 136;
            const float dg = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(g_blk)));
            const float du = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(u_blk)));
            const uint16_t g_sh = __ldg(reinterpret_cast<const uint16_t*>(g_blk + 2));
            const uint16_t u_sh = __ldg(reinterpret_cast<const uint16_t*>(u_blk + 2));
            const uint16_t g_sl2 = __ldg(reinterpret_cast<const uint16_t*>(g_blk + 4 + 2 * ip));
            const uint16_t u_sl2 = __ldg(reinterpret_cast<const uint16_t*>(u_blk + 4 + 2 * ip));
            const int shift = (t >> 1) * 4;
            const int off = (t & 1) * 8;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int ib = ip * 4 + q;
              const uint32_t g_ls_u = (((g_sl2 >> (4 * q)) & 0xFu) | (((g_sh >> (2 * ib)) & 3u) << 4));
              const uint32_t u_ls_u = (((u_sl2 >> (4 * q)) & 0xFu) | (((u_sh >> (2 * ib)) & 3u) << 4));
              const float g_ls_f = __fadd_rn(__uint_as_float(0x4B000000u | g_ls_u), -8388640.0f);
              const float u_ls_f = __fadd_rn(__uint_as_float(0x4B000000u | u_ls_u), -8388640.0f);
              const uint32_t gq0 = gguf_cuda::ReadU32Unaligned2BDev(g_blk + 8 + ib * 16 + off);
              const uint32_t gq1 = gguf_cuda::ReadU32Unaligned2BDev(g_blk + 12 + ib * 16 + off);
              const uint32_t uq0 = gguf_cuda::ReadU32Unaligned2BDev(u_blk + 8 + ib * 16 + off);
              const uint32_t uq1 = gguf_cuda::ReadU32Unaligned2BDev(u_blk + 12 + ib * 16 + off);
              const uint4 e_raw = valid_row ? gguf_cuda::PackIq4NlWordsToBf16x2(gq0, gq1, shift, dg * g_ls_f)
                                            : make_uint4(0u, 0u, 0u, 0u);
              const uint4 o_raw = valid_row ? gguf_cuda::PackIq4NlWordsToBf16x2(uq0, uq1, shift, du * u_ls_f)
                                            : make_uint4(0u, 0u, 0u, 0u);
              uint4 et = gguf_sm90::TransposeQuad4x4(e_raw, t);
              uint4 ot = gguf_sm90::TransposeQuad4x4(o_raw, t);
              const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
              const int sb_in_k64 = q & 1;
              gguf_sm90::WgmmaFence();
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et.x, ot.x, et.y, ot.y,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, sb_in_k64 * 2 + 0));
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et.z, ot.z, et.w, ot.w,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, sb_in_k64 * 2 + 1));
              gguf_sm90::WgmmaCommitGroup();
              gguf_sm90::WgmmaWaitGroup<0>();
              gguf_sm90::FenceWgmmaU4(et);
              gguf_sm90::FenceWgmmaU4(ot);
            }
          } else {
            const uint8_t* g_blk = gate_row_ptr + static_cast<int64_t>(sb256) * 82;
            const uint8_t* u_blk = up_row_ptr + static_cast<int64_t>(sb256) * 82;
            const float dg = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(g_blk))) * 0.25f;
            const float du = gguf_cuda::Fp16ToFloatDev(__ldg(reinterpret_cast<const uint16_t*>(u_blk))) * 0.25f;
            const auto* g_sc16 = reinterpret_cast<const uint16_t*>(g_blk + 74 + 4 * ip);
            const auto* u_sc16 = reinterpret_cast<const uint16_t*>(u_blk + 74 + 4 * ip);
            const auto* g_qh16 = reinterpret_cast<const uint16_t*>(g_blk + 66 + 4 * ip);
            const auto* u_qh16 = reinterpret_cast<const uint16_t*>(u_blk + 66 + 4 * ip);
            const uint32_t g_sc4 = static_cast<uint32_t>(__ldg(g_sc16)) | (static_cast<uint32_t>(__ldg(g_sc16 + 1)) << 16);
            const uint32_t u_sc4 = static_cast<uint32_t>(__ldg(u_sc16)) | (static_cast<uint32_t>(__ldg(u_sc16 + 1)) << 16);
            const uint32_t g_qh4 = static_cast<uint32_t>(__ldg(g_qh16)) | (static_cast<uint32_t>(__ldg(g_qh16 + 1)) << 16);
            const uint32_t u_qh4 = static_cast<uint32_t>(__ldg(u_qh16)) | (static_cast<uint32_t>(__ldg(u_qh16 + 1)) << 16);
            const auto* gq16 = reinterpret_cast<const uint16_t*>(g_blk + 2 + 16 * ip + 4 * t);
            const auto* uq16 = reinterpret_cast<const uint16_t*>(u_blk + 2 + 16 * ip + 4 * t);
            const auto* gs16 = reinterpret_cast<const uint16_t*>(g_blk + 34 + 16 * ip + 4 * t);
            const auto* us16 = reinterpret_cast<const uint16_t*>(u_blk + 34 + 16 * ip + 4 * t);
            const uint32_t my_gq4 = static_cast<uint32_t>(__ldg(gq16)) | (static_cast<uint32_t>(__ldg(gq16 + 1)) << 16);
            const uint32_t my_uq4 = static_cast<uint32_t>(__ldg(uq16)) | (static_cast<uint32_t>(__ldg(uq16 + 1)) << 16);
            const uint32_t my_gs4 = static_cast<uint32_t>(__ldg(gs16)) | (static_cast<uint32_t>(__ldg(gs16 + 1)) << 16);
            const uint32_t my_us4 = static_cast<uint32_t>(__ldg(us16)) | (static_cast<uint32_t>(__ldg(us16 + 1)) << 16);
            const int sc_nibble_shift = 4 * (t >> 1);
            const int qh_shift_l = 8 - 2 * t;
            const int byte_shift = 8 * t;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const uint8_t g_sc = static_cast<uint8_t>((g_sc4 >> (8 * q)) & 0xFFu);
              const uint8_t u_sc = static_cast<uint8_t>((u_sc4 >> (8 * q)) & 0xFFu);
              const uint8_t g_qh = static_cast<uint8_t>((g_qh4 >> (8 * q)) & 0xFFu);
              const uint8_t u_qh = static_cast<uint8_t>((u_qh4 >> (8 * q)) & 0xFFu);
              const float dbg = dg * __fadd_rn(__uint_as_float(0x4B000000u | ((g_sc >> sc_nibble_shift) & 0xFu)), -8388607.5f);
              const float dbu = du * __fadd_rn(__uint_as_float(0x4B000000u | ((u_sc >> sc_nibble_shift) & 0xFu)), -8388607.5f);
              const uint32_t gq_w = __shfl_sync(0xFFFFFFFFu, my_gq4, q, 4);
              const uint32_t uq_w = __shfl_sync(0xFFFFFFFFu, my_uq4, q, 4);
              const uint32_t gs_w = __shfl_sync(0xFFFFFFFFu, my_gs4, q, 4);
              const uint32_t us_w = __shfl_sync(0xFFFFFFFFu, my_us4, q, 4);
              const int g_idx = static_cast<int>((gq_w >> byte_shift) & 0xFFu) | ((static_cast<int>(g_qh) << qh_shift_l) & 0x300);
              const int u_idx = static_cast<int>((uq_w >> byte_shift) & 0xFFu) | ((static_cast<int>(u_qh) << qh_shift_l) & 0x300);
              const uint4* s_lut_iq2s = reinterpret_cast<const uint4*>(s_lut);
              const uint4 gg_mag = s_lut_iq2s[g_idx];
              const uint4 ug_mag = s_lut_iq2s[u_idx];
              const uint32_t sg = (gs_w >> byte_shift) & 0xFFu;
              const uint32_t su = (us_w >> byte_shift) & 0xFFu;
              const uint4 e_raw = valid_row ? gguf_cuda::PackBf16Mag8Sign8ToBf16x2(gg_mag, sg, dbg)
                                            : make_uint4(0u, 0u, 0u, 0u);
              const uint4 o_raw = valid_row ? gguf_cuda::PackBf16Mag8Sign8ToBf16x2(ug_mag, su, dbu)
                                            : make_uint4(0u, 0u, 0u, 0u);
              uint4 et = gguf_sm90::TransposeQuad4x4(e_raw, t);
              uint4 ot = gguf_sm90::TransposeQuad4x4(o_raw, t);
              const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
              const int sb_in_k64 = q & 1;
              gguf_sm90::WgmmaFence();
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et.x, ot.x, et.y, ot.y,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, sb_in_k64 * 2 + 0));
              gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                  acc, n_inst, et.z, ot.z, et.w, ot.w,
                  gguf_sm90::AdvanceGmmaDescK16(desc_k64, sb_in_k64 * 2 + 1));
              gguf_sm90::WgmmaCommitGroup();
              gguf_sm90::WgmmaWaitGroup<0>();
              gguf_sm90::FenceWgmmaU4(et);
              gguf_sm90::FenceWgmmaU4(ot);
            }
          }
        } else {
          const int sb32_base = k_base >> 5;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float g_vals[8] = {};
            float u_vals[8] = {};
            if (valid_row) {
              gguf_cuda::DecodeGgufSubBlock8Tpl<kGateUpType>(gate_row_ptr, p.gate_type, sb32_base + q, t, g_vals);
              gguf_cuda::DecodeGgufSubBlock8Tpl<kGateUpType>(up_row_ptr, p.up_type, sb32_base + q, t, u_vals);
            }
            const uint4 e_raw = gguf_cuda::PackFloat8ToBf16x2(g_vals);
            const uint4 o_raw = gguf_cuda::PackFloat8ToBf16x2(u_vals);
            const int slot = q & 1;
            if (q >= 2) {
              gguf_sm90::WgmmaWaitGroup<1>();
              gguf_sm90::FenceWgmmaU4(et_buf[slot]);
              gguf_sm90::FenceWgmmaU4(ot_buf[slot]);
            }
            et_buf[slot] = gguf_sm90::TransposeQuad4x4(e_raw, t);
            ot_buf[slot] = gguf_sm90::TransposeQuad4x4(o_raw, t);
            const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
            gguf_sm90::WgmmaFence();
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
            gguf_sm90::WgmmaCommitGroup();
          }
        }
      } else {
        const int sb32_base = k_base >> 5;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float g_vals[8] = {};
          float u_vals[8] = {};
          if (valid_row) {
            gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(gate_row_ptr, p.gate_type, sb32_base + q, t, s_lut, g_vals);
            gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(up_row_ptr, p.up_type, sb32_base + q, t, s_lut, u_vals);
          }
          const uint4 e_raw = gguf_cuda::PackFloat8ToBf16x2(g_vals);
          const uint4 o_raw = gguf_cuda::PackFloat8ToBf16x2(u_vals);
          const int slot = q & 1;
          if (q >= 2) {
            gguf_sm90::WgmmaWaitGroup<1>();
            gguf_sm90::FenceWgmmaU4(et_buf[slot]);
            gguf_sm90::FenceWgmmaU4(ot_buf[slot]);
          }
          et_buf[slot] = gguf_sm90::TransposeQuad4x4(e_raw, t);
          ot_buf[slot] = gguf_sm90::TransposeQuad4x4(o_raw, t);
          const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
          gguf_sm90::WgmmaFence();
          gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
              acc, n_inst, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
              gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
          gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
              acc, n_inst, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
              gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
          gguf_sm90::WgmmaCommitGroup();
        }
      }

      if constexpr (!(kAligned && (kGateUpType == kGgmlTypeIQ2_S || kGateUpType == kGgmlTypeIQ4_XS))) {
        gguf_sm90::WgmmaWaitGroup<0>();
        gguf_sm90::FenceWgmmaU4(et_buf[0]);
        gguf_sm90::FenceWgmmaU4(ot_buf[0]);
        gguf_sm90::FenceWgmmaU4(et_buf[1]);
        gguf_sm90::FenceWgmmaU4(ot_buf[1]);
      }
      gguf_sm90::FenceWgmmaAcc(acc, num_subtiles);
    }
  }

  if (valid_row) {
#pragma unroll
    for (int st = 0; st < 16; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = st * 8 + 2 * t + h;
          if (m < count) {
            const float g_val = __bfloat162float(__float2bfloat16_rn(acc[st][0][h]));
            const float u_val = __bfloat162float(__float2bfloat16_rn(acc[st][1][h]));
            const float activation = u_val * (g_val / (1.0f + expf(-g_val)));
            const int route = s_routes[m];
            fc1_out[static_cast<int64_t>(route) * p.inter_size + j_out] = __float2bfloat16_rn(activation);
          }
        }
      }
    }
  }
}

template <int kDownType, bool kAligned = false>
__global__ __launch_bounds__(256, 2) void MoEPackedFc2Sm90WgmmaKernel(
    const __nv_bfloat16* __restrict__ fc1_out,
    const uint8_t* __restrict__ down_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    __nv_bfloat16* __restrict__ fc2_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= __ldg(d_num_tiles)) return;

  using namespace onnxruntime::contrib::gguf_quant;
  extern __shared__ char smem_raw[];
  char* smem_aligned = AlignSmem1024(smem_raw);
  uint4* s_B = reinterpret_cast<uint4*>(smem_aligned);
  uint64_t* s_lut = reinterpret_cast<uint64_t*>(smem_aligned + 65536);
  int* s_routes = reinterpret_cast<int*>(s_lut + 1024);

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count;
  const int expert = tile.expert;
  const int n_inst = SelectSm90WgmmaRouteN(count);
  const int num_subtiles = (count + 7) >> 3;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 128) {
    s_routes[tid] = (tid < count) ? __ldg(&sorted_routes[tile.start_route + tid]) : 0;
  }
  if constexpr (kDownType == kGgmlTypeIQ4_NL) {
    float2* s_iq4_pair = reinterpret_cast<float2*>(s_lut);
    s_iq4_pair[tid] = make_float2(
        static_cast<float>(kvalues_iq4nl[tid & 0xF]),
        static_cast<float>(kvalues_iq4nl[tid >> 4]));
  } else if constexpr (kDownType == kGgmlTypeIQ2_XXS) {
    s_lut[tid] = __ldg(iq2xxs_grid + tid);
  } else if constexpr (kDownType == kGgmlTypeIQ3_XXS) {
    uint32_t* s_lut32 = reinterpret_cast<uint32_t*>(s_lut);
    s_lut32[tid] = __ldg(iq3xxs_grid + tid);
  }
  __syncthreads();

  const bool fc1_aligned_16b = ((reinterpret_cast<uintptr_t>(fc1_out) & 15u) == 0) && ((p.inter_size & 7) == 0);
  const int stage_u4_total = n_inst * 16;
  auto stage_fc1_chunk = [&](int k_base, int stg) {
    uint4* s_tile0 = s_B + (stg * 2 + 0) * 1024;
    uint4* s_tile1 = s_B + (stg * 2 + 1) * 1024;
    for (int idx = tid; idx < stage_u4_total; idx += 256) {
      const int m = idx >> 4;
      const int u4_col16 = idx & 15;
      const bool pred = (m < count);
      const int route = pred ? s_routes[m] : 0;
      const __nv_bfloat16* src_ptr =
          fc1_out + static_cast<int64_t>(route) * p.inter_size + k_base + (u4_col16 << 3);
      gguf_sm90::CpAsyncB128Store16B(
          (u4_col16 & 8) ? s_tile1 : s_tile0, m, u4_col16 & 7, src_ptr, pred, fc1_aligned_16b);
    }
    gguf_sm90::CpAsyncCommitGroup();
  };

  stage_fc1_chunk(0, 0);

  const int warp = tid >> 5;
  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int wg_out_base = (static_cast<int>(blockIdx.x) * 2 + wg) * 64;
  const bool wg_valid = (wg_out_base < p.hidden_size);
  const int n_out_base = (static_cast<int>(blockIdx.x) * 8 + warp) * 16;
  const int r_lo = n_out_base + 2 * g;
  const int r_hi = r_lo + 1;
  const bool valid_lo = (r_lo < p.hidden_size);
  const bool valid_hi = (r_hi < p.hidden_size);

  const int64_t exp_base = static_cast<int64_t>(expert) * p.hidden_size;
  const uint8_t* down_row_lo = down_weights + (exp_base + (valid_lo ? r_lo : 0)) * p.down_row_bytes;
  const uint8_t* down_row_hi = down_weights + (exp_base + (valid_hi ? r_hi : 0)) * p.down_row_bytes;
  const bool aligned_iq4 = ((reinterpret_cast<uintptr_t>(down_weights) & 3u) == 0) && ((p.down_row_bytes & 3) == 0);

  float acc[16][2][2];
#pragma unroll
  for (int st = 0; st < 16; ++st) {
    acc[st][0][0] = 0.0f;
    acc[st][0][1] = 0.0f;
    acc[st][1][0] = 0.0f;
    acc[st][1][1] = 0.0f;
  }

  int stage = 0;
  for (int k_base = 0; k_base < p.inter_size; k_base += 128, stage ^= 1) {
    gguf_sm90::CpAsyncWaitGroup<0>();
    __syncthreads();
    gguf_sm90::FenceAsyncSharedProxy();
    if (k_base + 128 < p.inter_size) {
      stage_fc1_chunk(k_base + 128, stage ^ 1);
    }

    if (wg_valid) {
      const uint64_t desc_b0 = gguf_sm90::MakeGmmaDescB128KMajor(s_B + (stage * 2 + 0) * 1024);
      const uint64_t desc_b1 = gguf_sm90::MakeGmmaDescB128KMajor(s_B + (stage * 2 + 1) * 1024);
      uint4 et_buf[2];
      uint4 ot_buf[2];
      gguf_sm90::FenceWgmmaAcc(acc, num_subtiles);

      if constexpr (kAligned && kDownType == kGgmlTypeIQ4_NL) {
        (void)aligned_iq4;
        if constexpr (true) {
          const float2* s_iq4_pair = reinterpret_cast<const float2*>(s_lut);
          const int pair_base = k_base >> 6;
#pragma unroll
          for (int pair_local = 0; pair_local < 2; ++pair_local) {
            const int pair_idx = pair_base + pair_local;
            uint4 e0, e1, o0, o1;
            gguf_cuda::DecodeIq4NlPair64QuadSplitBf16x2(down_row_lo + pair_idx * 36, t, s_iq4_pair, e0, e1);
            gguf_cuda::DecodeIq4NlPair64QuadSplitBf16x2(down_row_hi + pair_idx * 36, t, s_iq4_pair, o0, o1);
            if (!valid_lo) e0 = e1 = make_uint4(0u, 0u, 0u, 0u);
            if (!valid_hi) o0 = o1 = make_uint4(0u, 0u, 0u, 0u);

            const uint64_t desc_k64 = (pair_local == 0) ? desc_b0 : desc_b1;
            if (pair_local > 0) {
              gguf_sm90::WgmmaWaitGroup<1>();
              gguf_sm90::FenceWgmmaU4(et_buf[0]);
              gguf_sm90::FenceWgmmaU4(ot_buf[0]);
            }
            et_buf[0] = gguf_sm90::TransposeQuad4x4(e0, t);
            ot_buf[0] = gguf_sm90::TransposeQuad4x4(o0, t);
            gguf_sm90::WgmmaFence();
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[0].x, ot_buf[0].x, et_buf[0].y, ot_buf[0].y,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, 0));
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[0].z, ot_buf[0].z, et_buf[0].w, ot_buf[0].w,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, 1));
            gguf_sm90::WgmmaCommitGroup();

            if (pair_local > 0) {
              gguf_sm90::WgmmaWaitGroup<1>();
              gguf_sm90::FenceWgmmaU4(et_buf[1]);
              gguf_sm90::FenceWgmmaU4(ot_buf[1]);
            }
            et_buf[1] = gguf_sm90::TransposeQuad4x4(e1, t);
            ot_buf[1] = gguf_sm90::TransposeQuad4x4(o1, t);
            gguf_sm90::WgmmaFence();
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[1].x, ot_buf[1].x, et_buf[1].y, ot_buf[1].y,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, 2));
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[1].z, ot_buf[1].z, et_buf[1].w, ot_buf[1].w,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, 3));
            gguf_sm90::WgmmaCommitGroup();
          }
        } else {
          const int sb32_base = k_base >> 5;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float d_lo[8] = {};
            float d_hi[8] = {};
            if (valid_lo) gguf_cuda::DecodeGgufSubBlock8Impl<kGgmlTypeIQ4_NL>(down_row_lo, sb32_base + q, t, d_lo);
            if (valid_hi) gguf_cuda::DecodeGgufSubBlock8Impl<kGgmlTypeIQ4_NL>(down_row_hi, sb32_base + q, t, d_hi);
            const uint4 e_raw = gguf_cuda::PackFloat8ToBf16x2(d_lo);
            const uint4 o_raw = gguf_cuda::PackFloat8ToBf16x2(d_hi);
            const int slot = q & 1;
            if (q >= 2) {
              gguf_sm90::WgmmaWaitGroup<1>();
              gguf_sm90::FenceWgmmaU4(et_buf[slot]);
              gguf_sm90::FenceWgmmaU4(ot_buf[slot]);
            }
            et_buf[slot] = gguf_sm90::TransposeQuad4x4(e_raw, t);
            ot_buf[slot] = gguf_sm90::TransposeQuad4x4(o_raw, t);
            const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
            gguf_sm90::WgmmaFence();
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
            gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
                acc, n_inst, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
                gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
            gguf_sm90::WgmmaCommitGroup();
          }
        }
      } else {
        const int sb32_base = k_base >> 5;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float d_lo[8] = {};
          float d_hi[8] = {};
          if constexpr (kDownType == kGgmlTypeIQ2_XXS || kDownType == kGgmlTypeIQ3_XXS) {
            if (valid_lo) gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kDownType>(down_row_lo, p.down_type, sb32_base + q, t, s_lut, d_lo);
            if (valid_hi) gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kDownType>(down_row_hi, p.down_type, sb32_base + q, t, s_lut, d_hi);
          } else {
            if (valid_lo) gguf_cuda::DecodeGgufSubBlock8Tpl<kDownType>(down_row_lo, p.down_type, sb32_base + q, t, d_lo);
            if (valid_hi) gguf_cuda::DecodeGgufSubBlock8Tpl<kDownType>(down_row_hi, p.down_type, sb32_base + q, t, d_hi);
          }
          const uint4 e_raw = gguf_cuda::PackFloat8ToBf16x2(d_lo);
          const uint4 o_raw = gguf_cuda::PackFloat8ToBf16x2(d_hi);
          const int slot = q & 1;
          if (q >= 2) {
            gguf_sm90::WgmmaWaitGroup<1>();
            gguf_sm90::FenceWgmmaU4(et_buf[slot]);
            gguf_sm90::FenceWgmmaU4(ot_buf[slot]);
          }
          et_buf[slot] = gguf_sm90::TransposeQuad4x4(e_raw, t);
          ot_buf[slot] = gguf_sm90::TransposeQuad4x4(o_raw, t);
          const uint64_t desc_k64 = (q < 2) ? desc_b0 : desc_b1;
          gguf_sm90::WgmmaFence();
          gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
              acc, n_inst, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
              gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
          gguf_sm90::WgmmaM64AdaptiveN128K16Rs(
              acc, n_inst, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
              gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
          gguf_sm90::WgmmaCommitGroup();
        }
      }

      gguf_sm90::WgmmaWaitGroup<0>();
      gguf_sm90::FenceWgmmaU4(et_buf[0]);
      gguf_sm90::FenceWgmmaU4(ot_buf[0]);
      gguf_sm90::FenceWgmmaU4(et_buf[1]);
      gguf_sm90::FenceWgmmaU4(ot_buf[1]);
      gguf_sm90::FenceWgmmaAcc(acc, num_subtiles);
    }
  }

  if (valid_lo) {
#pragma unroll
    for (int st = 0; st < 16; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = st * 8 + 2 * t + h;
          if (m < count) {
            const int route = s_routes[m];
            const float v_lo = acc[st][0][h];
            const float v_hi = acc[st][1][h];
            __nv_bfloat16* dst = fc2_out + static_cast<int64_t>(route) * p.hidden_size + r_lo;
            if (valid_hi && ((p.hidden_size & 1) == 0)) {
              *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v_lo, v_hi);
            } else {
              dst[0] = __float2bfloat16_rn(v_lo);
              if (valid_hi) dst[1] = __float2bfloat16_rn(v_hi);
            }
          }
        }
      }
    }
  }
}

template <typename KernelT>
cudaError_t SetMaxDynamicSmem(KernelT kernel, size_t smem_bytes) {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
}

}  // namespace

cudaError_t InitMoEPackedSm90WgmmaKernels() {
  using namespace onnxruntime::contrib::gguf_quant;
  cudaError_t err = cudaSuccess;
  auto chk = [&](cudaError_t e) {
    if (err == cudaSuccess && e != cudaSuccess) err = e;
  };
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ2_0>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_NL>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ5_0>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_XXS>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_XS>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_S, false>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_S, true>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ3_XXS, false>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ3_XXS, true>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_XS, false>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_XS, true>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ8_0>, kMoEPackedSm90Fc1SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc1SwiGLUSm90WgmmaKernel<-1>, kMoEPackedSm90Fc1SmemBytes));

  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ2_0>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_NL, false>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_NL, true>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ5_0>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_XXS>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_XS>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_S>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ3_XXS>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_XS>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ8_0>, kMoEPackedSm90Fc2SmemBytes));
  chk(SetMaxDynamicSmem(MoEPackedFc2Sm90WgmmaKernel<-1>, kMoEPackedSm90Fc2SmemBytes));
  return err;
}

cudaError_t LaunchMoEPackedSm90WgmmaPrefillCuda(
    cudaStream_t stream,
    int launch_tiles,
    const __nv_bfloat16* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const int* sorted_routes,
    const MoEPackedTile* tiles,
    const int* d_num_tiles,
    const MoEPackedParams& params,
    __nv_bfloat16* fc1_out,
    __nv_bfloat16* fc2_out) {
  using namespace onnxruntime::contrib::gguf_quant;
  const dim3 fc1_grid(static_cast<unsigned int>((params.inter_size + 63) / 64),
                      static_cast<unsigned int>(launch_tiles));
  const int fc1_type = (params.gate_type == params.up_type) ? params.gate_type : -1;
  const bool fc1_aligned_2b = ((reinterpret_cast<uintptr_t>(gate_weights) & 1u) == 0) &&
                              ((reinterpret_cast<uintptr_t>(up_weights) & 1u) == 0) &&
                              ((params.gate_row_bytes & 1) == 0) && ((params.up_row_bytes & 1) == 0);
  switch (fc1_type) {
    case kGgmlTypeQ2_0:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ2_0><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    case kGgmlTypeIQ4_NL:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_NL><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    case kGgmlTypeQ5_0:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ5_0><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    case kGgmlTypeIQ2_XXS:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_XXS><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    case kGgmlTypeIQ2_XS:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_XS><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    case kGgmlTypeIQ2_S:
      if (fc1_aligned_2b) {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_S, true><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      } else {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ2_S, false><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      }
      break;
    case kGgmlTypeIQ3_XXS:
      if (fc1_aligned_2b) {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ3_XXS, true><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      } else {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ3_XXS, false><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      }
      break;
    case kGgmlTypeIQ4_XS:
      if (fc1_aligned_2b) {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_XS, true><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      } else {
        MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeIQ4_XS, false><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      }
      break;
    case kGgmlTypeQ8_0:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<kGgmlTypeQ8_0><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
    default:
      MoEPackedFc1SwiGLUSm90WgmmaKernel<-1><<<fc1_grid, 256, kMoEPackedSm90Fc1SmemBytes, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      break;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  const dim3 fc2_grid(static_cast<unsigned int>((params.hidden_size + 127) / 128),
                      static_cast<unsigned int>(launch_tiles));
  switch (params.down_type) {
    case kGgmlTypeQ2_0:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ2_0><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ4_NL:
      if (((reinterpret_cast<uintptr_t>(down_weights) & 3u) == 0) && ((params.down_row_bytes & 3) == 0)) {
        MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_NL, true><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
            fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      } else {
        MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_NL, false><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
            fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      }
      break;
    case kGgmlTypeQ5_0:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ5_0><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ2_XXS:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_XXS><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ2_XS:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_XS><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ2_S:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ2_S><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ3_XXS:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ3_XXS><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeIQ4_XS:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeIQ4_XS><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    case kGgmlTypeQ8_0:
      MoEPackedFc2Sm90WgmmaKernel<kGgmlTypeQ8_0><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
    default:
      MoEPackedFc2Sm90WgmmaKernel<-1><<<fc2_grid, 256, kMoEPackedSm90Fc2SmemBytes, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      break;
  }
  return cudaGetLastError();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
