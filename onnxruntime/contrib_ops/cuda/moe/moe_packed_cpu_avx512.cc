// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX-512BW + AVX-512VL + AVX-512VNNI (`vpdpbusd`) 256-bit EVEX Q8 INT8 cold-expert kernels for MoEPacked offload.
//
// All hot loops intentionally use 256-bit EVEX (`ymm0..ymm31` + `__mmask32`) rather than 512-bit ZMM registers so
// Intel Skylake-X / Cascade Lake-X CPUs retain full AVX2 turbo clock frequencies while benefiting from 32 vector
// registers (zero accumulator stack spills in 4-row x 4-token blocking), masked byte negation, and 1-instruction
// `vpdpbusd` (`_mm256_dpbusd_epi32`). Callers dispatch here only after `CpuHasAvx512Vnni()` returns true.

#include "contrib_ops/cuda/moe/moe_packed_cpu.h"

#if ORT_MOE_PACKED_CPU_X86_64

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,bmi2,fma,f16c")
#define MOE_PACKED_POP_AVX512_GCC_TARGET 1
#endif

#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,bmi2,fma,f16c"))), apply_to = function)
#define MOE_PACKED_POP_AVX512_CLANG_TARGET 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define MOE_PACKED_FORCEINLINE __forceinline
#define MOE_PACKED_LAMBDA_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define MOE_PACKED_FORCEINLINE __attribute__((always_inline)) inline
#define MOE_PACKED_LAMBDA_INLINE __attribute__((always_inline))
#else
#define MOE_PACKED_FORCEINLINE inline
#define MOE_PACKED_LAMBDA_INLINE
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace moe_packed_cpu {
namespace {

using namespace onnxruntime::contrib::gguf_quant;

MOE_PACKED_FORCEINLINE uint16_t LoadU16Le(const uint8_t* p) noexcept {
  uint16_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

MOE_PACKED_FORCEINLINE uint32_t LoadU32Le(const uint8_t* p) noexcept {
  uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

MOE_PACKED_FORCEINLINE float Fp16ToFloatFast(uint16_t bits) noexcept {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
}

MOE_PACKED_FORCEINLINE float HsumAvx512Ymm(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 s = _mm_add_ps(lo, hi);
  const __m128 shuf = _mm_movehdup_ps(s);
  const __m128 sums = _mm_add_ps(s, shuf);
  return _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(shuf, sums)));
}

MOE_PACKED_FORCEINLINE float HsumI32Ymm(__m256i v) noexcept {
  const __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  const __m128i s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
  const __m128i s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
  return static_cast<float>(_mm_cvtsi128_si32(s32));
}

MOE_PACKED_FORCEINLINE void Hsum4Ymm(__m256 a0, __m256 a1, __m256 a2, __m256 a3, float* out4) noexcept {
  const __m256 t0 = _mm256_hadd_ps(a0, a1);
  const __m256 t1 = _mm256_hadd_ps(a2, a3);
  const __m256 t2 = _mm256_hadd_ps(t0, t1);
  _mm_storeu_ps(out4, _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1)));
}

MOE_PACKED_FORCEINLINE __m256 RoundBf16Vec8(__m256 x) noexcept {
  const __m256i bits = _mm256_castps_si256(x);
  const __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
  const __mmask8 is_nan = _mm256_cmpgt_epi32_mask(abs_bits, _mm256_set1_epi32(0x7F800000));
  const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
  const __m256i rounded = _mm256_and_si256(
      _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb)),
      _mm256_set1_epi32(static_cast<int>(0xFFFF0000u)));
  const __m256i quiet_nan = _mm256_or_si256(
      _mm256_and_si256(bits, _mm256_set1_epi32(static_cast<int>(0xFFFF0000u))),
      _mm256_set1_epi32(0x00400000));
  return _mm256_castsi256_ps(_mm256_mask_blend_epi32(is_nan, rounded, quiet_nan));
}

MOE_PACKED_FORCEINLINE __m256 RoundHalfVec8(__m256 x, bool is_bf16) noexcept {
  if (is_bf16) {
    return RoundBf16Vec8(x);
  }
  return _mm256_cvtph_ps(_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT));
}

// Cody-Waite degree-6 minimax exp(t) for SwiGLU: g / (1 + exp(-g)).
MOE_PACKED_FORCEINLINE __m256 SiluVec8(__m256 g) noexcept {
  const __mmask8 underflow_mask = _mm256_cmp_ps_mask(g, _mm256_set1_ps(-88.0f), _CMP_LE_OQ);
  const __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
  const __m256 t = _mm256_min_ps(_mm256_max_ps(neg_g, _mm256_set1_ps(-88.0f)), _mm256_set1_ps(88.0f));
  const __m256 inv_ln2 = _mm256_set1_ps(1.44269504088896341f);
  const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
  const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187045e-6f);

  const __m256 kf = _mm256_roundscale_ps(_mm256_mul_ps(t, inv_ln2), _MM_FROUND_TO_NEAREST_INT);
  const __m256 r = _mm256_fnmadd_ps(kf, ln2_lo, _mm256_fnmadd_ps(kf, ln2_hi, t));

  __m256 p = _mm256_set1_ps(1.3888888888888889e-3f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3333333333333333e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1666666666666664e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666666666666666e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0f - 4.5f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

  const __m256 exp_neg_g = _mm256_scalef_ps(p, kf);
  const __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_g);
  const __m256 silu = _mm256_div_ps(g, denom);
  return _mm256_mask_blend_ps(underflow_mask, silu, _mm256_setzero_ps());
}

MOE_PACKED_FORCEINLINE void QuantizeBlockQ8FromVecs(
    __m256 x0, __m256 x1, __m256 x2, __m256 x3, GgufQ8Block* blk) noexcept {
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 m01 = _mm256_max_ps(_mm256_and_ps(x0, sign_mask), _mm256_and_ps(x1, sign_mask));
  const __m256 m23 = _mm256_max_ps(_mm256_and_ps(x2, sign_mask), _mm256_and_ps(x3, sign_mask));
  const __m256 mv = _mm256_max_ps(m01, m23);
  __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(mv), _mm256_extractf128_ps(mv, 1));
  m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
  m128 = _mm_max_ss(m128, _mm_movehdup_ps(m128));
  const float max_abs = _mm_cvtss_f32(m128);
  if (!(max_abs > 0.0f)) {
    blk->d = 0.0f;
    blk->bsum = 0;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(blk->qs), _mm256_setzero_si256());
    return;
  }

  const float d = max_abs / 127.0f;
  const float id = 127.0f / max_abs;
  __m256 s0, s1, s2, s3;
  if (std::isfinite(id)) {
    const __m256 id_v = _mm256_set1_ps(id);
    s0 = _mm256_mul_ps(x0, id_v);
    s1 = _mm256_mul_ps(x1, id_v);
    s2 = _mm256_mul_ps(x2, id_v);
    s3 = _mm256_mul_ps(x3, id_v);
  } else {
    const __m256 max_v = _mm256_set1_ps(max_abs);
    const __m256 c127_v = _mm256_set1_ps(127.0f);
    s0 = _mm256_mul_ps(_mm256_div_ps(x0, max_v), c127_v);
    s1 = _mm256_mul_ps(_mm256_div_ps(x1, max_v), c127_v);
    s2 = _mm256_mul_ps(_mm256_div_ps(x2, max_v), c127_v);
    s3 = _mm256_mul_ps(_mm256_div_ps(x3, max_v), c127_v);
  }
  const __m256i q0 = _mm256_cvtps_epi32(s0);
  const __m256i q1 = _mm256_cvtps_epi32(s1);
  const __m256i q2 = _mm256_cvtps_epi32(s2);
  const __m256i q3 = _mm256_cvtps_epi32(s3);
  const __m256i p01 = _mm256_packs_epi32(q0, q1);
  const __m256i p23 = _mm256_packs_epi32(q2, q3);
  const __m256i p8_raw = _mm256_max_epi8(_mm256_packs_epi16(p01, p23), _mm256_set1_epi8(-127));
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256i qs_i8 = _mm256_permutevar8x32_epi32(p8_raw, perm);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(blk->qs), qs_i8);

  const __m256i sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_set1_epi8(1), qs_i8);
  const __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(sum32), _mm256_extracti128_si256(sum32, 1));
  const __m128i s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
  const __m128i s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
  blk->d = d;
  blk->bsum = _mm_cvtsi128_si32(s32);
}

// Computes all 4 bytes of ksigns_iq2xs[(aux32 >> (7*l)) & 127] in 7 ALU instructions with zero L1D loads.
MOE_PACKED_FORCEINLINE __mmask32 Signs7BitToMask32(uint32_t aux32) noexcept {
  const uint32_t s7 = _pdep_u32(aux32, 0x7F7F7F7Fu);
  uint32_t p = s7 ^ (s7 >> 4);
  p ^= (p >> 2);
  p ^= (p >> 1);
  return static_cast<__mmask32>(s7 | ((p & 0x01010101u) << 7));
}

MOE_PACKED_FORCEINLINE __m256i Pack4xU64(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) noexcept {
  const __m128i lo = _mm_unpacklo_epi64(_mm_cvtsi64_si128(static_cast<int64_t>(a0)),
                                        _mm_cvtsi64_si128(static_cast<int64_t>(a1)));
  const __m128i hi = _mm_unpacklo_epi64(_mm_cvtsi64_si128(static_cast<int64_t>(a2)),
                                        _mm_cvtsi64_si128(static_cast<int64_t>(a3)));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// Avoids vpgatherdd / vpgatherdq, which incur severe microcode penalties on Cascade Lake-X under GDS/Downfall mitigations.
MOE_PACKED_FORCEINLINE __m256i DecodeIq3XxsGrid32(const uint8_t* qs_ib) noexcept {
  const uint64_t g01 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[0]]) |
                       (static_cast<uint64_t>(iq3xxs_grid[qs_ib[1]]) << 32);
  const uint64_t g23 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[2]]) |
                       (static_cast<uint64_t>(iq3xxs_grid[qs_ib[3]]) << 32);
  const uint64_t g45 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[4]]) |
                       (static_cast<uint64_t>(iq3xxs_grid[qs_ib[5]]) << 32);
  const uint64_t g67 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[6]]) |
                       (static_cast<uint64_t>(iq3xxs_grid[qs_ib[7]]) << 32);
  return Pack4xU64(g01, g23, g45, g67);
}

MOE_PACKED_FORCEINLINE __m256i DecodeIq2XxsGrid32(const uint8_t* aux8) noexcept {
  return Pack4xU64(iq2xxs_grid[aux8[0]], iq2xxs_grid[aux8[1]],
                   iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[3]]);
}

MOE_PACKED_FORCEINLINE __m256i DecodeIq2SGrid32(const uint8_t* qs_ib, uint32_t qh_b) noexcept {
  return Pack4xU64(
      iq2s_grid[qs_ib[0] | ((qh_b << 8) & 0x300u)],
      iq2s_grid[qs_ib[1] | ((qh_b << 6) & 0x300u)],
      iq2s_grid[qs_ib[2] | ((qh_b << 4) & 0x300u)],
      iq2s_grid[qs_ib[3] | ((qh_b << 2) & 0x300u)]);
}

MOE_PACKED_FORCEINLINE void DecodeIq2XsGridAndSigns32(const uint8_t* qs_ib, __m256i& grid_u8,
                                                      __mmask32& smask) noexcept {
  const uint16_t q0 = LoadU16Le(qs_ib + 0);
  const uint16_t q1 = LoadU16Le(qs_ib + 2);
  const uint16_t q2 = LoadU16Le(qs_ib + 4);
  const uint16_t q3 = LoadU16Le(qs_ib + 6);
  grid_u8 = Pack4xU64(iq2xs_grid[q0 & 511u], iq2xs_grid[q1 & 511u],
                      iq2xs_grid[q2 & 511u], iq2xs_grid[q3 & 511u]);
  const uint32_t s0 = ksigns_iq2xs[q0 >> 9];
  const uint32_t s1 = ksigns_iq2xs[q1 >> 9];
  const uint32_t s2 = ksigns_iq2xs[q2 >> 9];
  const uint32_t s3 = ksigns_iq2xs[q3 >> 9];
  smask = static_cast<__mmask32>(s0 | (s1 << 8) | (s2 << 16) | (s3 << 24));
}

MOE_PACKED_FORCEINLINE __m256 MakeHalfPairScaleVec(float s_lo, float s_hi) noexcept {
  return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(s_lo)), _mm_set1_ps(s_hi), 1);
}

MOE_PACKED_FORCEINLINE __m256i DecodeIq4NlWeights32(const uint8_t* qs16, __m256i lut256, __m256i mask0f) noexcept {
  const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs16));
  const __m256i nibbles = _mm256_and_si256(
      _mm256_inserti128_si256(_mm256_castsi128_si256(raw16), _mm_srli_epi16(raw16, 4), 1), mask0f);
  return _mm256_shuffle_epi8(lut256, nibbles);
}

MOE_PACKED_FORCEINLINE __m256i UnpackQ20Unsigned32(const uint8_t* qs8, __m128i mask03) noexcept {
  const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs8));
  const __m128i u0 = _mm_and_si128(raw, mask03);
  const __m128i u1 = _mm_and_si128(_mm_srli_epi16(raw, 2), mask03);
  const __m128i u2 = _mm_and_si128(_mm_srli_epi16(raw, 4), mask03);
  const __m128i u3 = _mm_and_si128(_mm_srli_epi16(raw, 6), mask03);
  const __m128i p01 = _mm_unpacklo_epi8(u0, u1);
  const __m128i p23 = _mm_unpacklo_epi8(u2, u3);
  const __m128i lo16 = _mm_unpacklo_epi16(p01, p23);
  const __m128i hi16 = _mm_unpackhi_epi16(p01, p23);
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo16), hi16, 1);
}

template <int RT>
void DualDot2ChQ8VnniBatch(
    int type,
    const uint8_t* g0, const uint8_t* g1, const uint8_t* u0, const uint8_t* u1,
    const GgufQ8Block* const* x_rows, int K,
    float* gd0, float* ud0, float* gd1, float* ud1) noexcept {
  const __m256i zero = _mm256_setzero_si256();
  __m256 ag0[RT], au0[RT], ag1[RT], au1[RT];
  for (int r = 0; r < RT; ++r) {
    ag0[r] = _mm256_setzero_ps();
    au0[r] = _mm256_setzero_ps();
    ag1[r] = _mm256_setzero_ps();
    au1[r] = _mm256_setzero_ps();
  }

  if (type == kGgmlTypeIQ3_XXS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = g0 + i * 98;
      const uint8_t* bg1 = g1 + i * 98;
      const uint8_t* bu0 = u0 + i * 98;
      const uint8_t* bu1 = u1 + i * 98;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.25f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.25f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.25f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.25f;
      for (int ib = 0; ib < 8; ++ib) {
        const int blk_idx = i * 8 + ib;
        __m256i xq[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d025, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
          const uint32_t aux32 = LoadU32Le(blk + 66 + 4 * ib);
          const float scale_w = d025 * static_cast<float>(2 * (aux32 >> 28) + 1);
          const __m256i grid = DecodeIq3XxsGrid32(blk + 2 + 8 * ib);
          const __mmask32 smask = Signs7BitToMask32(aux32);
          for (int r = 0; r < RT; ++r) {
            const __m256i prod = _mm256_dpbusd_epi32(zero, grid, _mm256_mask_sub_epi8(xq[r], smask, zero, xq[r]));
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), _mm256_set1_ps(scale_w * xd[r]), acc_arr[r]);
          }
        };
        step(bg0, dg0, ag0);
        step(bg1, dg1, ag1);
        step(bu0, du0, au0);
        step(bu1, du1, au1);
      }
    }
  } else if (type == kGgmlTypeIQ2_S) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = g0 + i * 82;
      const uint8_t* bg1 = g1 + i * 82;
      const uint8_t* bu0 = u0 + i * 82;
      const uint8_t* bu1 = u1 + i * 82;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.125f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.125f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.125f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.125f;
      for (int ib = 0; ib < 8; ++ib) {
        const int blk_idx = i * 8 + ib;
        __m256i xq[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d0125, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
          const uint8_t sc = blk[74 + ib];
          const float w_lo = d0125 * static_cast<float>(2 * (sc & 0xF) + 1);
          const float w_hi = d0125 * static_cast<float>(2 * (sc >> 4) + 1);
          const __m256i grid = DecodeIq2SGrid32(blk + 2 + 4 * ib, blk[66 + ib]);
          const __mmask32 smask = static_cast<__mmask32>(LoadU32Le(blk + 34 + 4 * ib));
          for (int r = 0; r < RT; ++r) {
            const __m256i prod = _mm256_dpbusd_epi32(zero, grid, _mm256_mask_sub_epi8(xq[r], smask, zero, xq[r]));
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod),
                                         MakeHalfPairScaleVec(w_lo * xd[r], w_hi * xd[r]), acc_arr[r]);
          }
        };
        step(bg0, dg0, ag0);
        step(bg1, dg1, ag1);
        step(bu0, du0, au0);
        step(bu1, du1, au1);
      }
    }
  } else if (type == kGgmlTypeIQ2_XXS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = g0 + i * 66;
      const uint8_t* bg1 = g1 + i * 66;
      const uint8_t* bu0 = u0 + i * 66;
      const uint8_t* bu1 = u1 + i * 66;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.125f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.125f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.125f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.125f;
      for (int ib = 0; ib < 8; ++ib) {
        const int blk_idx = i * 8 + ib;
        __m256i xq[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d0125, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
          const uint8_t* aux8 = blk + 2 + 8 * ib;
          const uint32_t aux32 = LoadU32Le(aux8 + 4);
          const float scale_w = d0125 * static_cast<float>(2 * (aux32 >> 28) + 1);
          const __m256i grid = DecodeIq2XxsGrid32(aux8);
          const __mmask32 smask = Signs7BitToMask32(aux32);
          for (int r = 0; r < RT; ++r) {
            const __m256i prod = _mm256_dpbusd_epi32(zero, grid, _mm256_mask_sub_epi8(xq[r], smask, zero, xq[r]));
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), _mm256_set1_ps(scale_w * xd[r]), acc_arr[r]);
          }
        };
        step(bg0, dg0, ag0);
        step(bg1, dg1, ag1);
        step(bu0, du0, au0);
        step(bu1, du1, au1);
      }
    }
  } else if (type == kGgmlTypeIQ2_XS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = g0 + i * 74;
      const uint8_t* bg1 = g1 + i * 74;
      const uint8_t* bu0 = u0 + i * 74;
      const uint8_t* bu1 = u1 + i * 74;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.125f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.125f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.125f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.125f;
      for (int ib = 0; ib < 8; ++ib) {
        const int blk_idx = i * 8 + ib;
        __m256i xq[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d0125, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
          const uint8_t sc = blk[66 + ib];
          const float w_lo = d0125 * static_cast<float>(2 * (sc & 0xF) + 1);
          const float w_hi = d0125 * static_cast<float>(2 * (sc >> 4) + 1);
          __m256i grid;
          __mmask32 smask;
          DecodeIq2XsGridAndSigns32(blk + 2 + 8 * ib, grid, smask);
          for (int r = 0; r < RT; ++r) {
            const __m256i prod = _mm256_dpbusd_epi32(zero, grid, _mm256_mask_sub_epi8(xq[r], smask, zero, xq[r]));
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod),
                                         MakeHalfPairScaleVec(w_lo * xd[r], w_hi * xd[r]), acc_arr[r]);
          }
        };
        step(bg0, dg0, ag0);
        step(bg1, dg1, ag1);
        step(bu0, du0, au0);
        step(bu1, du1, au1);
      }
    }
  } else {
    // kGgmlTypeIQ4_XS
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = g0 + i * 136;
      const uint8_t* bg1 = g1 + i * 136;
      const uint8_t* bu0 = u0 + i * 136;
      const uint8_t* bu1 = u1 + i * 136;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0));
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1));
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0));
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1));
      const uint16_t sh_g0 = LoadU16Le(bg0 + 2);
      const uint16_t sh_g1 = LoadU16Le(bg1 + 2);
      const uint16_t sh_u0 = LoadU16Le(bu0 + 2);
      const uint16_t sh_u1 = LoadU16Le(bu1 + 2);
      for (int ib = 0; ib < 8; ++ib) {
        const int blk_idx = i * 8 + ib;
        __m256i ax[RT];
        __mmask32 xneg[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          ax[r] = _mm256_abs_epi8(xq);
          xneg[r] = _mm256_movepi8_mask(xq);
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d, uint16_t sh, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
          const int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
          const float scale_w = d * static_cast<float>(ls - 32);
          const __m256i w_i8 = DecodeIq4NlWeights32(blk + 8 + ib * 16, lut256, mask0f);
          for (int r = 0; r < RT; ++r) {
            const __m256i prod = _mm256_dpbusd_epi32(zero, ax[r], _mm256_mask_sub_epi8(w_i8, xneg[r], zero, w_i8));
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), _mm256_set1_ps(scale_w * xd[r]), acc_arr[r]);
          }
        };
        step(bg0, dg0, sh_g0, ag0);
        step(bg1, dg1, sh_g1, ag1);
        step(bu0, du0, sh_u0, au0);
        step(bu1, du1, sh_u1, au1);
      }
    }
  }

  for (int r = 0; r < RT; ++r) {
    float tmp[4];
    Hsum4Ymm(ag0[r], au0[r], ag1[r], au1[r], tmp);
    gd0[r] = tmp[0];
    ud0[r] = tmp[1];
    gd1[r] = tmp[2];
    ud1[r] = tmp[3];
  }
}

template <int RT>
void DotRowsBy4Q8VnniBatch(
    int type,
    const uint8_t* dr0, const uint8_t* dr1, const uint8_t* dr2, const uint8_t* dr3,
    const GgufQ8Block* x_base, size_t q8_stride, int K,
    float* dots, int dots_ld, int col_offset) noexcept {
  const __m256i zero = _mm256_setzero_si256();
  __m256 a0[RT], a1[RT], a2[RT], a3[RT];
  for (int r = 0; r < RT; ++r) {
    a0[r] = _mm256_setzero_ps();
    a1[r] = _mm256_setzero_ps();
    a2[r] = _mm256_setzero_ps();
    a3[r] = _mm256_setzero_ps();
  }

  if (type == kGgmlTypeIQ4_NL) {
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const int nb = K / 32;
    for (int i = 0; i < nb; ++i) {
      __m256i ax[RT];
      __mmask32 xneg[RT];
      float xd[RT];
      for (int r = 0; r < RT; ++r) {
        const GgufQ8Block& xb = x_base[static_cast<size_t>(r) * q8_stride + i];
        const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
        ax[r] = _mm256_abs_epi8(xq);
        xneg[r] = _mm256_movepi8_mask(xq);
        xd[r] = xb.d;
      }
      auto step = [&](const uint8_t* blk, __m256* acc_arr) MOE_PACKED_LAMBDA_INLINE {
        const float wd = Fp16ToFloatFast(LoadU16Le(blk));
        const __m256i w_i8 = DecodeIq4NlWeights32(blk + 2, lut256, mask0f);
        for (int r = 0; r < RT; ++r) {
          const __m256i prod = _mm256_dpbusd_epi32(zero, ax[r], _mm256_mask_sub_epi8(w_i8, xneg[r], zero, w_i8));
          acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod), _mm256_set1_ps(wd * xd[r]), acc_arr[r]);
        }
      };
      step(dr0 + i * 18, a0);
      step(dr1 + i * 18, a1);
      step(dr2 + i * 18, a2);
      step(dr3 + i * 18, a3);
    }
    for (int r = 0; r < RT; ++r) {
      Hsum4Ymm(a0[r], a1[r], a2[r], a3[r], dots + r * dots_ld + col_offset);
    }
    return;
  }

  if (type == kGgmlTypeQ2_0) {
    const __m128i mask03 = _mm_set1_epi8(0x03);
    const int nb = K / 64;
    float s0[RT]{}, s1[RT]{}, s2[RT]{}, s3[RT]{};
    for (int i = 0; i < nb; ++i) {
      const uint8_t* b0 = dr0 + i * 18;
      const uint8_t* b1 = dr1 + i * 18;
      const uint8_t* b2 = dr2 + i * 18;
      const uint8_t* b3 = dr3 + i * 18;
      const float wd0 = Fp16ToFloatFast(LoadU16Le(b0));
      const float wd1 = Fp16ToFloatFast(LoadU16Le(b1));
      const float wd2 = Fp16ToFloatFast(LoadU16Le(b2));
      const float wd3 = Fp16ToFloatFast(LoadU16Le(b3));
      for (int h = 0; h < 2; ++h) {
        const int blk_idx = i * 2 + h;
        __m256i xq[RT];
        float xd[RT];
        int32_t xbsum[RT];
        for (int r = 0; r < RT; ++r) {
          const GgufQ8Block& xb = x_base[static_cast<size_t>(r) * q8_stride + blk_idx];
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
          xd[r] = xb.d;
          xbsum[r] = xb.bsum;
        }
        const __m256i q0 = UnpackQ20Unsigned32(b0 + 2 + h * 8, mask03);
        const __m256i q1 = UnpackQ20Unsigned32(b1 + 2 + h * 8, mask03);
        const __m256i q2 = UnpackQ20Unsigned32(b2 + 2 + h * 8, mask03);
        const __m256i q3 = UnpackQ20Unsigned32(b3 + 2 + h * 8, mask03);
        for (int r = 0; r < RT; ++r) {
          const float bs = static_cast<float>(xbsum[r]);
          const float p0 = HsumI32Ymm(_mm256_dpbusd_epi32(zero, q0, xq[r])) - bs;
          const float p1 = HsumI32Ymm(_mm256_dpbusd_epi32(zero, q1, xq[r])) - bs;
          const float p2 = HsumI32Ymm(_mm256_dpbusd_epi32(zero, q2, xq[r])) - bs;
          const float p3 = HsumI32Ymm(_mm256_dpbusd_epi32(zero, q3, xq[r])) - bs;
          s0[r] = std::fma(p0, wd0 * xd[r], s0[r]);
          s1[r] = std::fma(p1, wd1 * xd[r], s1[r]);
          s2[r] = std::fma(p2, wd2 * xd[r], s2[r]);
          s3[r] = std::fma(p3, wd3 * xd[r], s3[r]);
        }
      }
    }
    for (int r = 0; r < RT; ++r) {
      dots[r * dots_ld + col_offset + 0] = s0[r];
      dots[r * dots_ld + col_offset + 1] = s1[r];
      dots[r * dots_ld + col_offset + 2] = s2[r];
      dots[r * dots_ld + col_offset + 3] = s3[r];
    }
    return;
  }

  // Fallback for IQ4_XS / IQ3_XXS / IQ2_S / IQ2_XXS / IQ2_XS down_weights via two DualDot2Ch calls
  const GgufQ8Block* x_ptrs[RT];
  for (int r = 0; r < RT; ++r) {
    x_ptrs[r] = x_base + static_cast<size_t>(r) * q8_stride;
  }
  float d0[RT], d1[RT], d2[RT], d3[RT];
  DualDot2ChQ8VnniBatch<RT>(type, dr0, dr1, dr2, dr3, x_ptrs, K, d0, d2, d1, d3);
  for (int r = 0; r < RT; ++r) {
    dots[r * dots_ld + col_offset + 0] = d0[r];
    dots[r * dots_ld + col_offset + 1] = d1[r];
    dots[r * dots_ld + col_offset + 2] = d2[r];
    dots[r * dots_ld + col_offset + 3] = d3[r];
  }
}

}  // namespace

void ConvertAndQuantizeRowQ8Avx512(const uint16_t* src, bool is_bf16, float* dst_f32, int K,
                                   GgufQ8Block* dst_q8) noexcept {
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const uint16_t* sb = src + b * 32;
    __m256 x0, x1, x2, x3;
    if (is_bf16) {
      x0 = _mm256_castsi256_ps(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 0))), 16));
      x1 = _mm256_castsi256_ps(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 8))), 16));
      x2 = _mm256_castsi256_ps(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 16))), 16));
      x3 = _mm256_castsi256_ps(_mm256_slli_epi32(
          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 24))), 16));
    } else {
      x0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 0)));
      x1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 8)));
      x2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 16)));
      x3 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sb + 24)));
    }
    if (dst_f32 != nullptr) {
      float* fb = dst_f32 + b * 32;
      _mm256_storeu_ps(fb + 0, x0);
      _mm256_storeu_ps(fb + 8, x1);
      _mm256_storeu_ps(fb + 16, x2);
      _mm256_storeu_ps(fb + 24, x3);
    }
    if (dst_q8 != nullptr) {
      QuantizeBlockQ8FromVecs(x0, x1, x2, x3, dst_q8 + b);
    }
  }
}

void QuantizeRowQ8Avx512(const float* src, int K, GgufQ8Block* dst) noexcept {
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    const float* xb = src + b * 32;
    QuantizeBlockQ8FromVecs(
        _mm256_loadu_ps(xb + 0),
        _mm256_loadu_ps(xb + 8),
        _mm256_loadu_ps(xb + 16),
        _mm256_loadu_ps(xb + 24),
        dst + b);
  }
}

void SwiGLURoundAndQuantize32Avx512(const float* gate32, const float* up32, bool is_bf16,
                                    float* dst_inter32, GgufQ8Block* dst_q8) noexcept {
  __m256 sw[4];
  for (int s = 0; s < 4; ++s) {
    const __m256 gv = RoundHalfVec8(_mm256_loadu_ps(gate32 + s * 8), is_bf16);
    const __m256 uv = RoundHalfVec8(_mm256_loadu_ps(up32 + s * 8), is_bf16);
    sw[s] = RoundHalfVec8(_mm256_mul_ps(uv, SiluVec8(gv)), is_bf16);
    if (dst_inter32 != nullptr) {
      _mm256_storeu_ps(dst_inter32 + s * 8, sw[s]);
    }
  }
  if (dst_q8 != nullptr) {
    QuantizeBlockQ8FromVecs(sw[0], sw[1], sw[2], sw[3], dst_q8);
  }
}

bool GgufFc1Tile32SwiGLUQ8Avx512Vnni(int type, const uint8_t* gate_tile0, size_t gate_row_bytes,
                                     const uint8_t* up_tile0, size_t up_row_bytes,
                                     const GgufQ8Block* const* x_q8_rows,
                                     int R, int K, bool is_bf16,
                                     GgufQ8Block* dst_inter_q8,
                                     size_t inter_q8_stride) noexcept {
  if (type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS &&
      type != kGgmlTypeIQ2_XS && type != kGgmlTypeIQ4_XS) {
    return false;
  }
  alignas(32) float g_tile[16][32];
  alignas(32) float u_tile[16][32];
  for (int j_rel = 0; j_rel < 32; j_rel += 2) {
    const uint8_t* g0 = gate_tile0 + static_cast<size_t>(j_rel) * gate_row_bytes;
    const uint8_t* g1 = g0 + gate_row_bytes;
    const uint8_t* u0 = up_tile0 + static_cast<size_t>(j_rel) * up_row_bytes;
    const uint8_t* u1 = u0 + up_row_bytes;
    int r = 0;
    for (; r + 4 <= R; r += 4) {
      float gd0[4], ud0[4], gd1[4], ud1[4];
      DualDot2ChQ8VnniBatch<4>(type, g0, g1, u0, u1, x_q8_rows + r, K, gd0, ud0, gd1, ud1);
      for (int sr = 0; sr < 4; ++sr) {
        g_tile[r + sr][j_rel + 0] = gd0[sr];
        g_tile[r + sr][j_rel + 1] = gd1[sr];
        u_tile[r + sr][j_rel + 0] = ud0[sr];
        u_tile[r + sr][j_rel + 1] = ud1[sr];
      }
    }
    for (; r + 2 <= R; r += 2) {
      float gd0[2], ud0[2], gd1[2], ud1[2];
      DualDot2ChQ8VnniBatch<2>(type, g0, g1, u0, u1, x_q8_rows + r, K, gd0, ud0, gd1, ud1);
      for (int sr = 0; sr < 2; ++sr) {
        g_tile[r + sr][j_rel + 0] = gd0[sr];
        g_tile[r + sr][j_rel + 1] = gd1[sr];
        u_tile[r + sr][j_rel + 0] = ud0[sr];
        u_tile[r + sr][j_rel + 1] = ud1[sr];
      }
    }
    if (r < R) {
      float gd0[1], ud0[1], gd1[1], ud1[1];
      DualDot2ChQ8VnniBatch<1>(type, g0, g1, u0, u1, x_q8_rows + r, K, gd0, ud0, gd1, ud1);
      g_tile[r][j_rel + 0] = gd0[0];
      g_tile[r][j_rel + 1] = gd1[0];
      u_tile[r][j_rel + 0] = ud0[0];
      u_tile[r][j_rel + 1] = ud1[0];
    }
  }
  for (int r = 0; r < R; ++r) {
    SwiGLURoundAndQuantize32Avx512(g_tile[r], u_tile[r], is_bf16, nullptr,
                                   dst_inter_q8 + static_cast<size_t>(r) * inter_q8_stride);
  }
  return true;
}

bool GgufDualDot2ChQ8Avx512Vnni(int type, const uint8_t* g0, const uint8_t* g1, const uint8_t* u0, const uint8_t* u1,
                                const GgufQ8Block* const* x_q8_rows, int R, int K,
                                float* g_dots0, float* u_dots0, float* g_dots1, float* u_dots1) noexcept {
  if (type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS &&
      type != kGgmlTypeIQ2_XS && type != kGgmlTypeIQ4_XS) {
    return false;
  }
  int r = 0;
  for (; r + 4 <= R; r += 4) {
    DualDot2ChQ8VnniBatch<4>(type, g0, g1, u0, u1, x_q8_rows + r, K,
                             g_dots0 + r, u_dots0 + r, g_dots1 + r, u_dots1 + r);
  }
  for (; r + 2 <= R; r += 2) {
    DualDot2ChQ8VnniBatch<2>(type, g0, g1, u0, u1, x_q8_rows + r, K,
                             g_dots0 + r, u_dots0 + r, g_dots1 + r, u_dots1 + r);
  }
  for (; r < R; ++r) {
    DualDot2ChQ8VnniBatch<1>(type, g0, g1, u0, u1, x_q8_rows + r, K,
                             g_dots0 + r, u_dots0 + r, g_dots1 + r, u_dots1 + r);
  }
  return true;
}

int GgufDotRowsBy4Q8Avx512Vnni(int type, const uint8_t* row0, size_t row_bytes, int n_rows,
                               const GgufQ8Block* x_q8_base, size_t q8_stride,
                               int R, int K, float* dots, int dots_ld) noexcept {
  if (type != kGgmlTypeIQ4_NL && type != kGgmlTypeQ2_0 && type != kGgmlTypeIQ4_XS &&
      type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS && type != kGgmlTypeIQ2_XS) {
    return 0;
  }
  int c = 0;
  for (; c + 4 <= n_rows; c += 4) {
    const uint8_t* dr0 = row0 + static_cast<size_t>(c) * row_bytes;
    const uint8_t* dr1 = dr0 + row_bytes;
    const uint8_t* dr2 = dr1 + row_bytes;
    const uint8_t* dr3 = dr2 + row_bytes;
    int r = 0;
    for (; r + 4 <= R; r += 4) {
      DotRowsBy4Q8VnniBatch<4>(type, dr0, dr1, dr2, dr3, x_q8_base + static_cast<size_t>(r) * q8_stride,
                               q8_stride, K, dots + r * dots_ld, dots_ld, c);
    }
    for (; r + 2 <= R; r += 2) {
      DotRowsBy4Q8VnniBatch<2>(type, dr0, dr1, dr2, dr3, x_q8_base + static_cast<size_t>(r) * q8_stride,
                               q8_stride, K, dots + r * dots_ld, dots_ld, c);
    }
    for (; r < R; ++r) {
      DotRowsBy4Q8VnniBatch<1>(type, dr0, dr1, dr2, dr3, x_q8_base + static_cast<size_t>(r) * q8_stride,
                               q8_stride, K, dots + r * dots_ld, dots_ld, c);
    }
  }
  return c;
}

}  // namespace moe_packed_cpu
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#if defined(MOE_PACKED_POP_AVX512_CLANG_TARGET)
#pragma clang attribute pop
#undef MOE_PACKED_POP_AVX512_CLANG_TARGET
#endif

#if defined(MOE_PACKED_POP_AVX512_GCC_TARGET)
#pragma GCC pop_options
#undef MOE_PACKED_POP_AVX512_GCC_TARGET
#endif

#endif  // ORT_MOE_PACKED_CPU_X86_64
