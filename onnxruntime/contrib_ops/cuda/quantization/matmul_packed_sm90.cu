// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed_sm90.h"
#include "contrib_ops/cuda/quantization/matmul_packed.h"
#include "contrib_ops/cuda/quantization/gguf_sm90_wgmma.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

__device__ __forceinline__ __nv_bfloat16 ApplySm90ActivationBf16(__nv_bfloat16 val, int activation_type) {
  if (activation_type == 0) return val;
  const float v = __bfloat162float(val);
  if (activation_type == 1) {
    const float pos = v < 0.0f ? 0.0f : v;
    return __float2bfloat16_rn(pos * pos);
  }
  return __float2bfloat16_rn(v / (1.0f + expf(-v)));
}

struct Q8_0Half128Stage {
  float d[4];
  uint32_t w0[4];
  uint32_t w1[4];
};

template <bool kAligned2B>
__device__ __forceinline__ void LoadQ8_0Half128Stage(
    const uint8_t* row_ptr, int half_idx, int t, Q8_0Half128Stage& st) {
  const int sb32_base = half_idx * 4;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint8_t* blk = row_ptr + static_cast<int64_t>(sb32_base + q) * 34;
    st.d[q] = gguf_cuda::Fp16ToFloatDev(
        kAligned2B ? __ldg(reinterpret_cast<const uint16_t*>(blk)) : gguf_cuda::ReadU16Dev(blk));
    const uint8_t* qs = blk + 2 + t * 8;
    if constexpr (kAligned2B) {
      const auto* q16 = reinterpret_cast<const uint16_t*>(qs);
      st.w0[q] = static_cast<uint32_t>(__ldg(q16 + 0)) | (static_cast<uint32_t>(__ldg(q16 + 1)) << 16);
      st.w1[q] = static_cast<uint32_t>(__ldg(q16 + 2)) | (static_cast<uint32_t>(__ldg(q16 + 3)) << 16);
    } else {
      st.w0[q] = gguf_cuda::ReadU32Dev(qs);
      st.w1[q] = gguf_cuda::ReadU32Dev(qs + 4);
    }
  }
}

__device__ __forceinline__ uint4 DecodeQ8_0Half128GroupBf16x2(const Q8_0Half128Stage& st, int q) {
  const float d = st.d[q];
  const uint32_t u0 = st.w0[q] ^ 0x80808080u;
  const uint32_t u1 = st.w1[q] ^ 0x80808080u;
  float out[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[j] = __fmul_rn(d, __fadd_rn(__uint_as_float(__byte_perm(u0, 0x00004B00u, 0x5440u + j)), -8388736.0f));
    out[j + 4] = __fmul_rn(d, __fadd_rn(__uint_as_float(__byte_perm(u1, 0x00004B00u, 0x5440u + j)), -8388736.0f));
  }
  return gguf_cuda::PackFloat8ToBf16x2(out);
}

template <int kSubTiles>
__device__ __forceinline__ void WgmmaMatMulStepRs(
    float (&acc)[kSubTiles][2][2],
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint64_t desc_b) {
  if constexpr (kSubTiles == 2) {
    gguf_sm90::WgmmaM64N16K16Rs(acc, a0, a1, a2, a3, desc_b);
  } else if constexpr (kSubTiles == 4) {
    gguf_sm90::WgmmaM64N32K16Rs(acc, a0, a1, a2, a3, desc_b);
  } else {
    static_assert(kSubTiles == 8, "Unsupported kSubTiles for MatMulPacked SM90 WGMMA");
    gguf_sm90::WgmmaM64N64K16Rs(acc, a0, a1, a2, a3, desc_b);
  }
}

// SM90 Hopper WGMMA kernel for MatMulPacked BF16 multi-row decode (M in 9..64 -> kMaxM in {16, 32, 64}).
// Each 128-thread warpgroup CTA computes 64 output columns across kMaxM token rows over its K-slice
// [h_begin, h_end) using double-buffered 128-K `cp.async.cg` B128-swizzled s_A:
//   - split_k == 1 (N > 16384): writes BF16 directly to `y` with ZERO global scratch workspace.
//   - split_k > 1 (2560 <= N <= 16384, M >= 17): gridDim.y = split_k fills a single <= 528-CTA wave
//     across 132 Hopper SMs and writes L2-resident FP32 partials to `scratch_partials` (<= 7.5 MiB),
//     followed by `MatMulPackedSplitKReduceBf16Kernel`.
// Static shared memory is 8 KB (M<=16), 16 KB (M<=32), or 32 KB (M<=64) <= 48 KB (100% CUDA-graph safe).
template <int kMaxM, int kCompileType, bool kAligned, bool kSplitK>
__global__ __launch_bounds__(128, 4) void MatMulPackedBf16Sm90WgmmaKernel(
    const __nv_bfloat16* __restrict__ a,
    MatMulPackedParams p,
    float* __restrict__ scratch_partials,
    __nv_bfloat16* __restrict__ y,
    int m_rt,
    int split_k) {
  static_assert(kMaxM == 16 || kMaxM == 32 || kMaxM == 64, "kMaxM must be 16, 32, or 64");

  constexpr int kThreads = 128;
  constexpr int kSubTiles = kMaxM / 8;
  constexpr int kTile64U4 = kMaxM * 8;  // 128 (2KB), 256 (4KB), or 512 (8KB) — all multiples of 1024B!

  alignas(1024) __shared__ uint4 s_A[4 * kTile64U4];

  const int tid = static_cast<int>(threadIdx.x);
  const int wg_warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  const int cta_col_base = static_cast<int>(blockIdx.x) * 64;
  const int num_half128 = p.k >> 7;
  int sk_idx = 0;
  int h_begin = 0;
  int h_end = num_half128;
  if constexpr (kSplitK) {
    sk_idx = static_cast<int>(blockIdx.y);
    h_begin = (num_half128 * sk_idx) / split_k;
    h_end = (num_half128 * (sk_idx + 1)) / split_k;
  } else {
    (void)scratch_partials;
    (void)split_k;
  }

  const int r_lo = cta_col_base + wg_warp * 16 + 2 * g;
  const int r_hi = r_lo + 1;
  const bool valid_lo = (r_lo < p.n);
  const bool valid_hi = (r_hi < p.n);
  const uint8_t* row_lo = p.b_data + static_cast<int64_t>(valid_lo ? r_lo : 0) * p.row_bytes;
  const uint8_t* row_hi = p.b_data + static_cast<int64_t>(valid_hi ? r_hi : 0) * p.row_bytes;

  const bool a_aligned_16b = ((reinterpret_cast<uintptr_t>(a) & 15u) == 0) && ((p.k & 7) == 0);
  const int num_subtiles = (m_rt + 7) >> 3;

  float acc[kSubTiles][2][2];
#pragma unroll
  for (int st = 0; st < kSubTiles; ++st) {
    acc[st][0][0] = 0.0f;
    acc[st][0][1] = 0.0f;
    acc[st][1][0] = 0.0f;
    acc[st][1][1] = 0.0f;
  }

  auto run_half128_wgmma = [&](int half_idx, const uint4* s_half0, const uint4* s_half1) {
    const uint64_t desc_b0 = gguf_sm90::MakeGmmaDescB128KMajor(s_half0);
    const uint64_t desc_b1 = gguf_sm90::MakeGmmaDescB128KMajor(s_half1);
    uint4 et_buf[2];
    uint4 ot_buf[2];
    gguf_sm90::FenceWgmmaAcc(acc);

    if constexpr (kCompileType == gguf_quant::kGgmlTypeQ6_K) {
      gguf_cuda::Q6KHalf128Stage<kAligned> st_lo{};
      gguf_cuda::Q6KHalf128Stage<kAligned> st_hi{};
      if (valid_lo) gguf_cuda::LoadQ6KHalf128Stage<kAligned>(row_lo, half_idx, t, st_lo);
      if (valid_hi) gguf_cuda::LoadQ6KHalf128Stage<kAligned>(row_hi, half_idx, t, st_hi);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 e_raw = valid_lo ? gguf_cuda::DecodeQ6KHalf128GroupBf16x2<kAligned>(st_lo, q, t)
                                     : make_uint4(0u, 0u, 0u, 0u);
        const uint4 o_raw = valid_hi ? gguf_cuda::DecodeQ6KHalf128GroupBf16x2<kAligned>(st_hi, q, t)
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
        WgmmaMatMulStepRs(
            acc, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
            gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
        WgmmaMatMulStepRs(
            acc, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
            gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
        gguf_sm90::WgmmaCommitGroup();
      }
    } else {
      Q8_0Half128Stage st_lo{};
      Q8_0Half128Stage st_hi{};
      if (valid_lo) LoadQ8_0Half128Stage<kAligned>(row_lo, half_idx, t, st_lo);
      if (valid_hi) LoadQ8_0Half128Stage<kAligned>(row_hi, half_idx, t, st_hi);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 e_raw = valid_lo ? DecodeQ8_0Half128GroupBf16x2(st_lo, q)
                                     : make_uint4(0u, 0u, 0u, 0u);
        const uint4 o_raw = valid_hi ? DecodeQ8_0Half128GroupBf16x2(st_hi, q)
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
        WgmmaMatMulStepRs(
            acc, et_buf[slot].x, ot_buf[slot].x, et_buf[slot].y, ot_buf[slot].y,
            gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 0));
        WgmmaMatMulStepRs(
            acc, et_buf[slot].z, ot_buf[slot].z, et_buf[slot].w, ot_buf[slot].w,
            gguf_sm90::AdvanceGmmaDescK16(desc_k64, slot * 2 + 1));
        gguf_sm90::WgmmaCommitGroup();
      }
    }

    gguf_sm90::WgmmaWaitGroup<0>();
    gguf_sm90::FenceWgmmaU4(et_buf[0]);
    gguf_sm90::FenceWgmmaU4(ot_buf[0]);
    gguf_sm90::FenceWgmmaU4(et_buf[1]);
    gguf_sm90::FenceWgmmaU4(ot_buf[1]);
    gguf_sm90::FenceWgmmaAcc(acc);
  };

  constexpr int kStageU4 = kMaxM * 16;
  auto stage_a_128 = [&](int k_base, int stg) {
    uint4* s_tile0 = s_A + (stg * 2 + 0) * kTile64U4;
    uint4* s_tile1 = s_A + (stg * 2 + 1) * kTile64U4;
    for (int idx = tid; idx < kStageU4; idx += kThreads) {
      const int m = idx >> 4;
      const int u4_col16 = idx & 15;
      const bool pred = (m < m_rt);
      const __nv_bfloat16* src_ptr =
          a + static_cast<int64_t>(pred ? m : 0) * p.k + k_base + (u4_col16 << 3);
      gguf_sm90::CpAsyncB128Store16B(
          (u4_col16 & 8) ? s_tile1 : s_tile0, m, u4_col16 & 7, src_ptr, pred, a_aligned_16b);
    }
    gguf_sm90::CpAsyncCommitGroup();
  };

  if (h_begin < h_end) {
    stage_a_128(h_begin << 7, 0);
    int stage = 0;
    for (int h = h_begin; h < h_end; ++h, stage ^= 1) {
      gguf_sm90::CpAsyncWaitGroup<0>();
      __syncthreads();
      gguf_sm90::FenceAsyncSharedProxy();
      if (h + 1 < h_end) {
        stage_a_128((h + 1) << 7, stage ^ 1);
      }
      run_half128_wgmma(
          h,
          s_A + (stage * 2 + 0) * kTile64U4,
          s_A + (stage * 2 + 1) * kTile64U4);
    }
  }

  if (!valid_lo) return;

  if constexpr (kSplitK) {
    float* sk_base = scratch_partials + static_cast<int64_t>(sk_idx) * m_rt * p.n;
#pragma unroll
    for (int st = 0; st < kSubTiles; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m_tok = st * 8 + 2 * t + h;
          if (m_tok < m_rt) {
            float* dst = sk_base + static_cast<int64_t>(m_tok) * p.n + r_lo;
            if (valid_hi && ((p.n & 1) == 0)) {
              *reinterpret_cast<float2*>(dst) = make_float2(acc[st][0][h], acc[st][1][h]);
            } else {
              dst[0] = acc[st][0][h];
              if (valid_hi) {
                dst[1] = acc[st][1][h];
              }
            }
          }
        }
      }
    }
  } else {
    const float scale = p.global_scale ? *p.global_scale : 1.0f;
    const bool has_scale = (p.global_scale != nullptr);
#pragma unroll
    for (int st = 0; st < kSubTiles; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m_tok = st * 8 + 2 * t + h;
          if (m_tok < m_rt) {
            const float s_lo = has_scale ? acc[st][0][h] * scale : acc[st][0][h];
            const float s_hi = has_scale ? acc[st][1][h] * scale : acc[st][1][h];
            __nv_bfloat16* dst = y + static_cast<int64_t>(m_tok) * p.n + r_lo;
            if (valid_hi && (reinterpret_cast<uintptr_t>(dst) & 3u) == 0 && p.activation_type == 0) {
              *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(s_lo, s_hi);
            } else {
              dst[0] = ApplySm90ActivationBf16(__float2bfloat16_rn(s_lo), p.activation_type);
              if (valid_hi) {
                dst[1] = ApplySm90ActivationBf16(__float2bfloat16_rn(s_hi), p.activation_type);
              }
            }
          }
        }
      }
    }
  }
}

__global__ __launch_bounds__(256) void MatMulPackedSplitKReduceBf16Vec4Kernel(
    const float4* __restrict__ partials4,
    __nv_bfloat16* __restrict__ y,
    const float* __restrict__ global_scale,
    int64_t stride4,
    int split_k) {
  const int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (vec_idx >= stride4) return;

  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float4* src = partials4 + vec_idx;
  for (int sk = 0; sk < split_k; ++sk) {
    const float4 v = src[static_cast<int64_t>(sk) * stride4];
    sum.x += v.x;
    sum.y += v.y;
    sum.z += v.z;
    sum.w += v.w;
  }
  if (global_scale != nullptr) {
    const float sc = *global_scale;
    sum.x *= sc;
    sum.y *= sc;
    sum.z *= sc;
    sum.w *= sc;
  }
  const __nv_bfloat162 b01 = __floats2bfloat162_rn(sum.x, sum.y);
  const __nv_bfloat162 b23 = __floats2bfloat162_rn(sum.z, sum.w);
  uint32_t u01, u23;
  asm volatile("mov.b32 %0, %1;" : "=r"(u01) : "r"(*reinterpret_cast<const uint32_t*>(&b01)));
  asm volatile("mov.b32 %0, %1;" : "=r"(u23) : "r"(*reinterpret_cast<const uint32_t*>(&b23)));
  reinterpret_cast<uint2*>(y)[vec_idx] = make_uint2(u01, u23);
}

__global__ __launch_bounds__(256) void MatMulPackedSplitKReduceBf16ScalarKernel(
    const float* __restrict__ partials,
    __nv_bfloat16* __restrict__ y,
    const float* __restrict__ global_scale,
    int64_t total_elems,
    int split_k,
    int activation_type) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (idx >= total_elems) return;

  float sum = 0.0f;
  const float* src = partials + idx;
  for (int sk = 0; sk < split_k; ++sk) {
    sum += src[static_cast<int64_t>(sk) * total_elems];
  }
  if (global_scale != nullptr) {
    sum *= *global_scale;
  }
  y[idx] = ApplySm90ActivationBf16(__float2bfloat16_rn(sum), activation_type);
}

template <int kMaxM, bool kSplitK>
void DispatchMatMulPackedSm90ByType(
    cudaStream_t stream,
    dim3 grid,
    const __nv_bfloat16* a,
    const MatMulPackedParams& params,
    float* scratch_partials,
    __nv_bfloat16* y,
    int m,
    int split_k) {
  using namespace onnxruntime::contrib::gguf_quant;
  if (params.ggml_type == kGgmlTypeQ6_K) {
    if ((reinterpret_cast<uintptr_t>(params.b_data) & 3u) == 0 && (params.row_bytes & 3) == 0) {
      MatMulPackedBf16Sm90WgmmaKernel<kMaxM, kGgmlTypeQ6_K, true, kSplitK>
          <<<grid, 128, 0, stream>>>(a, params, scratch_partials, y, m, split_k);
    } else {
      MatMulPackedBf16Sm90WgmmaKernel<kMaxM, kGgmlTypeQ6_K, false, kSplitK>
          <<<grid, 128, 0, stream>>>(a, params, scratch_partials, y, m, split_k);
    }
  } else {
    if ((reinterpret_cast<uintptr_t>(params.b_data) & 1u) == 0 && (params.row_bytes & 1) == 0) {
      MatMulPackedBf16Sm90WgmmaKernel<kMaxM, kGgmlTypeQ8_0, true, kSplitK>
          <<<grid, 128, 0, stream>>>(a, params, scratch_partials, y, m, split_k);
    } else {
      MatMulPackedBf16Sm90WgmmaKernel<kMaxM, kGgmlTypeQ8_0, false, kSplitK>
          <<<grid, 128, 0, stream>>>(a, params, scratch_partials, y, m, split_k);
    }
  }
}

}  // namespace

cudaError_t LaunchMatMulPackedBf16Sm90WgmmaCuda(
    cudaStream_t stream,
    const __nv_bfloat16* a,
    const MatMulPackedParams& params,
    float* scratch_partials,
    __nv_bfloat16* y,
    int m) {
  int split_k = GetMatMulPackedSm90SplitK(params.k, params.n);
  if (split_k > 1 && scratch_partials == nullptr) {
    split_k = 1;
  }
  const dim3 grid(static_cast<unsigned int>((params.n + 63) / 64), static_cast<unsigned int>(split_k));
  if (split_k > 1) {
    if (m <= 16) {
      DispatchMatMulPackedSm90ByType<16, true>(stream, grid, a, params, scratch_partials, y, m, split_k);
    } else if (m <= 32) {
      DispatchMatMulPackedSm90ByType<32, true>(stream, grid, a, params, scratch_partials, y, m, split_k);
    } else {
      DispatchMatMulPackedSm90ByType<64, true>(stream, grid, a, params, scratch_partials, y, m, split_k);
    }
  } else {
    if (m <= 16) {
      DispatchMatMulPackedSm90ByType<16, false>(stream, grid, a, params, nullptr, y, m, 1);
    } else if (m <= 32) {
      DispatchMatMulPackedSm90ByType<32, false>(stream, grid, a, params, nullptr, y, m, 1);
    } else {
      DispatchMatMulPackedSm90ByType<64, false>(stream, grid, a, params, nullptr, y, m, 1);
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess || split_k <= 1) {
    return err;
  }

  const int64_t total_elems = static_cast<int64_t>(m) * params.n;
  if ((total_elems & 3) == 0 && params.activation_type == 0 &&
      ((reinterpret_cast<uintptr_t>(scratch_partials) & 15u) == 0) &&
      ((reinterpret_cast<uintptr_t>(y) & 7u) == 0)) {
    const int64_t stride4 = total_elems >> 2;
    const unsigned int red_blocks = static_cast<unsigned int>((stride4 + 255) / 256);
    MatMulPackedSplitKReduceBf16Vec4Kernel<<<red_blocks, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(scratch_partials), y, params.global_scale, stride4, split_k);
  } else {
    const unsigned int red_blocks = static_cast<unsigned int>((total_elems + 255) / 256);
    MatMulPackedSplitKReduceBf16ScalarKernel<<<red_blocks, 256, 0, stream>>>(
        scratch_partials, y, params.global_scale, total_elems, split_k, params.activation_type);
  }
  return cudaGetLastError();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
