// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AVX2 + FMA + F16C GGUF cold-expert kernels for the MoEPacked CPU/GPU offload decode path.
//
// Callers dispatch here only after a runtime CPU check. The instruction set is enabled per function (GCC/Clang target
// pragmas, or /arch:AVX2 on MSVC for this file only), and everything except the entry points has internal linkage, so
// no AVX-encoded copy of a shared inline function can be chosen by the linker for baseline code.

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
#pragma GCC target("avx2,fma,f16c")
#define MOE_PACKED_POP_AVX2_GCC_TARGET 1
#endif

#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#define MOE_PACKED_POP_AVX2_CLANG_TARGET 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define MOE_PACKED_AVX2_LAMBDA_INLINE [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define MOE_PACKED_AVX2_LAMBDA_INLINE __attribute__((always_inline))
#else
#define MOE_PACKED_AVX2_LAMBDA_INLINE
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace moe_packed_cpu {
namespace {

using namespace onnxruntime::contrib::gguf_quant;

inline uint16_t LoadU16Le(const uint8_t* p) {
  uint16_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline uint32_t LoadU32Le(const uint8_t* p) {
  uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline float Fp16ToFloatFast(uint16_t bits) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(bits))));
}

inline float HorizontalSumAvx2(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

struct SignXorLutAvx2 {
  alignas(32) uint32_t table[256][8];
  constexpr SignXorLutAvx2() : table{} {
    for (int s = 0; s < 256; ++s) {
      for (int j = 0; j < 8; ++j) {
        table[s][j] = (s & (1 << j)) ? 0x80000000u : 0u;
      }
    }
  }
};
static constexpr SignXorLutAvx2 kSignXorLutAvx2{};

inline __m256 SignXorAvx2(uint8_t signs) {
  return _mm256_load_ps(reinterpret_cast<const float*>(kSignXorLutAvx2.table[signs]));
}

template <int RT>
void GgufDotRowAvx2Tpl(
    const uint8_t* down_row,
    int down_type,
    int K,
    const float* const* inter_rows,
    float* down_dots) {
  using namespace onnxruntime::contrib::gguf_quant;
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  const float* xr0 = inter_rows[0];
  const float* xr1 = (RT >= 2) ? inter_rows[1] : nullptr;

  if (down_type == kGgmlTypeIQ4_NL) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const int nb = K / 32;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 18;
      const __m256 vd = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(blk)));
      const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
      const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
      const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
      const __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)));
      const __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))));
      const __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)));
      const __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))));
      const int k_off = i * 32;
      acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr0 + k_off + 0), acc0);
      acc0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr0 + k_off + 8), acc0);
      acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr0 + k_off + 16), acc0);
      acc0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr0 + k_off + 24), acc0);
      if constexpr (RT >= 2) {
        acc1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr1 + k_off + 0), acc1);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr1 + k_off + 8), acc1);
        acc1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr1 + k_off + 16), acc1);
        acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr1 + k_off + 24), acc1);
      }
    }
  } else if (down_type == kGgmlTypeQ2_0) {
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256i ones = _mm256_set1_epi32(1);
    const int nb = K / 64;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 18;
      const __m256 vd = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(blk)));
      for (int w = 0; w < 8; ++w) {
        const uint16_t q16 = LoadU16Le(blk + 2 + w * 2);
        const __m256i qv = _mm256_sub_epi32(
            _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(q16), shifts), mask3), ones);
        const __m256 w_vec = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(qv));
        const int k_off = i * 64 + w * 8;
        acc0 = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(xr0 + k_off), acc0);
        if constexpr (RT >= 2) {
          acc1 = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(xr1 + k_off), acc1);
        }
      }
    }
  } else if (down_type == kGgmlTypeIQ2_S) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 82;
      const float d = Fp16ToFloatFast(LoadU16Le(blk)) * 0.25f;
      const uint8_t* qs = blk + 2;
      const uint8_t* signs = qs + 32;
      const uint8_t* qh = blk + 66;
      const uint8_t* scales = blk + 74;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xF));
        const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4));
        const uint8_t qh_byte = qh[ib32];
        for (int l = 0; l < 4; ++l) {
          const __m256 v_dl = _mm256_set1_ps((l < 2) ? db0 : db1);
          const int idx = qs[l] | ((static_cast<int>(qh_byte) << (8 - 2 * l)) & 0x300);
          const __m128i g8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx));
          const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
          const __m256 w = _mm256_mul_ps(v_dl, _mm256_xor_ps(vf, SignXorAvx2(signs[l])));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          acc0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr0 + k_off), acc0);
          if constexpr (RT >= 2) {
            acc1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr1 + k_off), acc1);
          }
        }
        qs += 4;
        signs += 4;
      }
    }
  } else if (down_type == kGgmlTypeIQ2_XXS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 66;
      const float d = Fp16ToFloatFast(LoadU16Le(blk)) * 0.25f;
      const uint8_t* qs = blk + 2;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t* aux8 = qs + 8 * ib32;
        const uint32_t aux32 = LoadU32Le(aux8 + 4);
        const __m256 v_db = _mm256_set1_ps(d * (0.5f + static_cast<float>(aux32 >> 28)));
        for (int l = 0; l < 4; ++l) {
          const __m128i g8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + aux8[l]));
          const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
          const uint8_t s = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
          const __m256 w = _mm256_mul_ps(v_db, _mm256_xor_ps(vf, SignXorAvx2(s)));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          acc0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr0 + k_off), acc0);
          if constexpr (RT >= 2) {
            acc1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr1 + k_off), acc1);
          }
        }
      }
    }
  } else if (down_type == kGgmlTypeIQ2_XS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 74;
      const float d = Fp16ToFloatFast(LoadU16Le(blk)) * 0.25f;
      const uint8_t* qs = blk + 2;
      const uint8_t* scales = blk + 66;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xF));
        const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4));
        for (int l = 0; l < 4; ++l) {
          const uint16_t qval = LoadU16Le(qs + (4 * ib32 + l) * 2);
          const __m128i g8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xs_grid + (qval & 511)));
          const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
          const __m256 v_dl = _mm256_set1_ps((l < 2) ? db0 : db1);
          const __m256 w = _mm256_mul_ps(v_dl, _mm256_xor_ps(vf, SignXorAvx2(ksigns_iq2xs[qval >> 9])));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          acc0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr0 + k_off), acc0);
          if constexpr (RT >= 2) {
            acc1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr1 + k_off), acc1);
          }
        }
      }
    }
  } else if (down_type == kGgmlTypeIQ3_XXS) {
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 98;
      const float d = Fp16ToFloatFast(LoadU16Le(blk)) * 0.5f;
      const uint8_t* qs = blk + 2;
      const uint8_t* scales_and_signs = qs + 64;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint32_t aux32 = LoadU32Le(scales_and_signs + 4 * ib32);
        const __m256 v_db = _mm256_set1_ps(d * (0.5f + static_cast<float>(aux32 >> 28)));
        for (int l = 0; l < 4; ++l) {
          const uint64_t g64 = static_cast<uint64_t>(iq3xxs_grid[qs[2 * l + 0]]) |
                               (static_cast<uint64_t>(iq3xxs_grid[qs[2 * l + 1]]) << 32);
          const __m128i g8 = _mm_cvtsi64_si128(static_cast<int64_t>(g64));
          const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
          const uint8_t s = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
          const __m256 w = _mm256_mul_ps(v_db, _mm256_xor_ps(vf, SignXorAvx2(s)));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          acc0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr0 + k_off), acc0);
          if constexpr (RT >= 2) {
            acc1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(xr1 + k_off), acc1);
          }
        }
        qs += 8;
      }
    }
  } else if (down_type == kGgmlTypeIQ4_XS) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const int nb = K / 256;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 136;
      const float d = Fp16ToFloatFast(LoadU16Le(blk));
      const uint16_t scales_h = LoadU16Le(blk + 2);
      const uint8_t* scales_l = blk + 4;
      const uint8_t* qs = blk + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const __m256 vd = _mm256_set1_ps(d * static_cast<float>(ls - 32));
        const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + ib * 16));
        const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
        const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
        const __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)));
        const __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))));
        const __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)));
        const __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))));
        const int k_off = i * 256 + ib * 32;
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr0 + k_off + 0), acc0);
        acc0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr0 + k_off + 8), acc0);
        acc0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr0 + k_off + 16), acc0);
        acc0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr0 + k_off + 24), acc0);
        if constexpr (RT >= 2) {
          acc1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr1 + k_off + 0), acc1);
          acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr1 + k_off + 8), acc1);
          acc1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xr1 + k_off + 16), acc1);
          acc1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xr1 + k_off + 24), acc1);
        }
      }
    }
  } else if (down_type == kGgmlTypeQ5_0) {
    const int nb = K / 32;
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk = down_row + i * 22;
      const __m256 vd = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(blk)));
      const uint32_t qh = LoadU32Le(blk + 2);
      const uint8_t* qs = blk + 6;
      alignas(32) float w32[32];
      for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
        w32[j] = static_cast<float>(((qs[j] & 0x0F) | xh_0) - 16);
        w32[j + 16] = static_cast<float>(((qs[j] >> 4) | xh_1) - 16);
      }
      const int k_off = i * 32;
      for (int s = 0; s < 32; s += 8) {
        const __m256 w_vec = _mm256_mul_ps(vd, _mm256_load_ps(w32 + s));
        acc0 = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(xr0 + k_off + s), acc0);
        if constexpr (RT >= 2) {
          acc1 = _mm256_fmadd_ps(w_vec, _mm256_loadu_ps(xr1 + k_off + s), acc1);
        }
      }
    }
  }
  down_dots[0] = HorizontalSumAvx2(acc0);
  if constexpr (RT >= 2) {
    down_dots[1] = HorizontalSumAvx2(acc1);
  }
}

template <int RT>
void GgufDualDotRowAvx2Tpl(
    const uint8_t* gate_row,
    const uint8_t* up_row,
    int gate_type,
    int K,
    const float* const* x_rows,
    float* gate_dots,
    float* up_dots) {
  using namespace onnxruntime::contrib::gguf_quant;
  __m256 ag0 = _mm256_setzero_ps();
  __m256 au0 = _mm256_setzero_ps();
  __m256 ag1 = _mm256_setzero_ps();
  __m256 au1 = _mm256_setzero_ps();
  const float* xr0 = x_rows[0];
  const float* xr1 = (RT >= 2) ? x_rows[1] : nullptr;
  const int nb = K / 256;

  if (gate_type == kGgmlTypeIQ2_S) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk_g = gate_row + i * 82;
      const uint8_t* blk_u = up_row + i * 82;
      const float d_g = Fp16ToFloatFast(LoadU16Le(blk_g)) * 0.25f;
      const float d_u = Fp16ToFloatFast(LoadU16Le(blk_u)) * 0.25f;
      const uint8_t* qs_g = blk_g + 2;
      const uint8_t* signs_g = qs_g + 32;
      const uint8_t* qh_g = blk_g + 66;
      const uint8_t* scales_g = blk_g + 74;
      const uint8_t* qs_u = blk_u + 2;
      const uint8_t* signs_u = qs_u + 32;
      const uint8_t* qh_u = blk_u + 66;
      const uint8_t* scales_u = blk_u + 74;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] & 0xF));
        const float db1_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] >> 4));
        const float db0_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] & 0xF));
        const float db1_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] >> 4));
        const uint8_t qh_byte_g = qh_g[ib32];
        const uint8_t qh_byte_u = qh_u[ib32];
        for (int l = 0; l < 4; ++l) {
          const __m256 v_dl_g = _mm256_set1_ps((l < 2) ? db0_g : db1_g);
          const __m256 v_dl_u = _mm256_set1_ps((l < 2) ? db0_u : db1_u);
          const int idx_g = qs_g[l] | ((static_cast<int>(qh_byte_g) << (8 - 2 * l)) & 0x300);
          const int idx_u = qs_u[l] | ((static_cast<int>(qh_byte_u) << (8 - 2 * l)) & 0x300);
          const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx_g));
          const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx_u));
          const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
          const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
          const __m256 wg = _mm256_mul_ps(v_dl_g, _mm256_xor_ps(vf_g, SignXorAvx2(signs_g[l])));
          const __m256 wu = _mm256_mul_ps(v_dl_u, _mm256_xor_ps(vf_u, SignXorAvx2(signs_u[l])));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          ag0 = _mm256_fmadd_ps(wg, xv0, ag0);
          au0 = _mm256_fmadd_ps(wu, xv0, au0);
          if constexpr (RT >= 2) {
            const __m256 xv1 = _mm256_loadu_ps(xr1 + k_off);
            ag1 = _mm256_fmadd_ps(wg, xv1, ag1);
            au1 = _mm256_fmadd_ps(wu, xv1, au1);
          }
        }
        qs_g += 4;
        signs_g += 4;
        qs_u += 4;
        signs_u += 4;
      }
    }
  } else if (gate_type == kGgmlTypeIQ2_XXS) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk_g = gate_row + i * 66;
      const uint8_t* blk_u = up_row + i * 66;
      const float d_g = Fp16ToFloatFast(LoadU16Le(blk_g)) * 0.25f;
      const float d_u = Fp16ToFloatFast(LoadU16Le(blk_u)) * 0.25f;
      const uint8_t* qs_g = blk_g + 2;
      const uint8_t* qs_u = blk_u + 2;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t* aux8_g = qs_g + 8 * ib32;
        const uint8_t* aux8_u = qs_u + 8 * ib32;
        const uint32_t aux32_g = LoadU32Le(aux8_g + 4);
        const uint32_t aux32_u = LoadU32Le(aux8_u + 4);
        const __m256 v_db_g = _mm256_set1_ps(d_g * (0.5f + static_cast<float>(aux32_g >> 28)));
        const __m256 v_db_u = _mm256_set1_ps(d_u * (0.5f + static_cast<float>(aux32_u >> 28)));
        for (int l = 0; l < 4; ++l) {
          const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + aux8_g[l]));
          const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + aux8_u[l]));
          const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
          const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
          const uint8_t sg = ksigns_iq2xs[(aux32_g >> (7 * l)) & 127];
          const uint8_t su = ksigns_iq2xs[(aux32_u >> (7 * l)) & 127];
          const __m256 wg = _mm256_mul_ps(v_db_g, _mm256_xor_ps(vf_g, SignXorAvx2(sg)));
          const __m256 wu = _mm256_mul_ps(v_db_u, _mm256_xor_ps(vf_u, SignXorAvx2(su)));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          ag0 = _mm256_fmadd_ps(wg, xv0, ag0);
          au0 = _mm256_fmadd_ps(wu, xv0, au0);
          if constexpr (RT >= 2) {
            const __m256 xv1 = _mm256_loadu_ps(xr1 + k_off);
            ag1 = _mm256_fmadd_ps(wg, xv1, ag1);
            au1 = _mm256_fmadd_ps(wu, xv1, au1);
          }
        }
      }
    }
  } else if (gate_type == kGgmlTypeIQ3_XXS) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk_g = gate_row + i * 98;
      const uint8_t* blk_u = up_row + i * 98;
      const float d_g = Fp16ToFloatFast(LoadU16Le(blk_g)) * 0.5f;
      const float d_u = Fp16ToFloatFast(LoadU16Le(blk_u)) * 0.5f;
      const uint8_t* qs_g = blk_g + 2;
      const uint8_t* sas_g = qs_g + 64;
      const uint8_t* qs_u = blk_u + 2;
      const uint8_t* sas_u = qs_u + 64;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint32_t aux32_g = LoadU32Le(sas_g + 4 * ib32);
        const uint32_t aux32_u = LoadU32Le(sas_u + 4 * ib32);
        const __m256 v_db_g = _mm256_set1_ps(d_g * (0.5f + static_cast<float>(aux32_g >> 28)));
        const __m256 v_db_u = _mm256_set1_ps(d_u * (0.5f + static_cast<float>(aux32_u >> 28)));
        for (int l = 0; l < 4; ++l) {
          const uint64_t g64_g = static_cast<uint64_t>(iq3xxs_grid[qs_g[2 * l + 0]]) |
                                 (static_cast<uint64_t>(iq3xxs_grid[qs_g[2 * l + 1]]) << 32);
          const uint64_t g64_u = static_cast<uint64_t>(iq3xxs_grid[qs_u[2 * l + 0]]) |
                                 (static_cast<uint64_t>(iq3xxs_grid[qs_u[2 * l + 1]]) << 32);
          const __m128i g8_g = _mm_cvtsi64_si128(static_cast<int64_t>(g64_g));
          const __m128i g8_u = _mm_cvtsi64_si128(static_cast<int64_t>(g64_u));
          const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
          const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
          const uint8_t sg = ksigns_iq2xs[(aux32_g >> (7 * l)) & 127];
          const uint8_t su = ksigns_iq2xs[(aux32_u >> (7 * l)) & 127];
          const __m256 wg = _mm256_mul_ps(v_db_g, _mm256_xor_ps(vf_g, SignXorAvx2(sg)));
          const __m256 wu = _mm256_mul_ps(v_db_u, _mm256_xor_ps(vf_u, SignXorAvx2(su)));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          ag0 = _mm256_fmadd_ps(wg, xv0, ag0);
          au0 = _mm256_fmadd_ps(wu, xv0, au0);
          if constexpr (RT >= 2) {
            const __m256 xv1 = _mm256_loadu_ps(xr1 + k_off);
            ag1 = _mm256_fmadd_ps(wg, xv1, ag1);
            au1 = _mm256_fmadd_ps(wu, xv1, au1);
          }
        }
        qs_g += 8;
        qs_u += 8;
      }
    }
  } else if (gate_type == kGgmlTypeIQ4_XS) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk_g = gate_row + i * 136;
      const uint8_t* blk_u = up_row + i * 136;
      const float d_g = Fp16ToFloatFast(LoadU16Le(blk_g));
      const float d_u = Fp16ToFloatFast(LoadU16Le(blk_u));
      const uint16_t scales_h_g = LoadU16Le(blk_g + 2);
      const uint16_t scales_h_u = LoadU16Le(blk_u + 2);
      const uint8_t* scales_l_g = blk_g + 4;
      const uint8_t* scales_l_u = blk_u + 4;
      const uint8_t* qs_g = blk_g + 8;
      const uint8_t* qs_u = blk_u + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls_g = ((scales_l_g[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h_g >> (2 * ib)) & 3) << 4);
        const int ls_u = ((scales_l_u[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h_u >> (2 * ib)) & 3) << 4);
        const __m256 vd_g = _mm256_set1_ps(d_g * static_cast<float>(ls_g - 32));
        const __m256 vd_u = _mm256_set1_ps(d_u * static_cast<float>(ls_u - 32));
        const __m128i raw16_g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_g + ib * 16));
        const __m128i raw16_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_u + ib * 16));
        const __m128i v_lo_g = _mm_shuffle_epi8(lut, _mm_and_si128(raw16_g, mask0f));
        const __m128i v_hi_g = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16_g, 4), mask0f));
        const __m128i v_lo_u = _mm_shuffle_epi8(lut, _mm_and_si128(raw16_u, mask0f));
        const __m128i v_hi_u = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16_u, 4), mask0f));
        const __m256 wg0 = _mm256_mul_ps(vd_g, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo_g)));
        const __m256 wg1 = _mm256_mul_ps(vd_g, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo_g, 8))));
        const __m256 wg2 = _mm256_mul_ps(vd_g, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi_g)));
        const __m256 wg3 = _mm256_mul_ps(vd_g, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi_g, 8))));
        const __m256 wu0 = _mm256_mul_ps(vd_u, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo_u)));
        const __m256 wu1 = _mm256_mul_ps(vd_u, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo_u, 8))));
        const __m256 wu2 = _mm256_mul_ps(vd_u, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi_u)));
        const __m256 wu3 = _mm256_mul_ps(vd_u, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi_u, 8))));
        const int k_off = i * 256 + ib * 32;
        const __m256 x0 = _mm256_loadu_ps(xr0 + k_off + 0);
        const __m256 x1 = _mm256_loadu_ps(xr0 + k_off + 8);
        const __m256 x2 = _mm256_loadu_ps(xr0 + k_off + 16);
        const __m256 x3 = _mm256_loadu_ps(xr0 + k_off + 24);
        ag0 = _mm256_fmadd_ps(wg0, x0, ag0);
        ag0 = _mm256_fmadd_ps(wg1, x1, ag0);
        ag0 = _mm256_fmadd_ps(wg2, x2, ag0);
        ag0 = _mm256_fmadd_ps(wg3, x3, ag0);
        au0 = _mm256_fmadd_ps(wu0, x0, au0);
        au0 = _mm256_fmadd_ps(wu1, x1, au0);
        au0 = _mm256_fmadd_ps(wu2, x2, au0);
        au0 = _mm256_fmadd_ps(wu3, x3, au0);
        if constexpr (RT >= 2) {
          const __m256 y0 = _mm256_loadu_ps(xr1 + k_off + 0);
          const __m256 y1 = _mm256_loadu_ps(xr1 + k_off + 8);
          const __m256 y2 = _mm256_loadu_ps(xr1 + k_off + 16);
          const __m256 y3 = _mm256_loadu_ps(xr1 + k_off + 24);
          ag1 = _mm256_fmadd_ps(wg0, y0, ag1);
          ag1 = _mm256_fmadd_ps(wg1, y1, ag1);
          ag1 = _mm256_fmadd_ps(wg2, y2, ag1);
          ag1 = _mm256_fmadd_ps(wg3, y3, ag1);
          au1 = _mm256_fmadd_ps(wu0, y0, au1);
          au1 = _mm256_fmadd_ps(wu1, y1, au1);
          au1 = _mm256_fmadd_ps(wu2, y2, au1);
          au1 = _mm256_fmadd_ps(wu3, y3, au1);
        }
      }
    }
  } else {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* blk_g = gate_row + i * 74;
      const uint8_t* blk_u = up_row + i * 74;
      const float d_g = Fp16ToFloatFast(LoadU16Le(blk_g)) * 0.25f;
      const float d_u = Fp16ToFloatFast(LoadU16Le(blk_u)) * 0.25f;
      const uint8_t* qs_g = blk_g + 2;
      const uint8_t* scales_g = blk_g + 66;
      const uint8_t* qs_u = blk_u + 2;
      const uint8_t* scales_u = blk_u + 66;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] & 0xF));
        const float db1_g = d_g * (0.5f + static_cast<float>(scales_g[ib32] >> 4));
        const float db0_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] & 0xF));
        const float db1_u = d_u * (0.5f + static_cast<float>(scales_u[ib32] >> 4));
        for (int l = 0; l < 4; ++l) {
          const uint16_t qg = LoadU16Le(qs_g + (4 * ib32 + l) * 2);
          const uint16_t qu = LoadU16Le(qs_u + (4 * ib32 + l) * 2);
          const __m128i g8_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xs_grid + (qg & 511)));
          const __m128i g8_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xs_grid + (qu & 511)));
          const __m256 vf_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_g));
          const __m256 vf_u = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8_u));
          const __m256 v_dl_g = _mm256_set1_ps((l < 2) ? db0_g : db1_g);
          const __m256 v_dl_u = _mm256_set1_ps((l < 2) ? db0_u : db1_u);
          const __m256 wg = _mm256_mul_ps(v_dl_g, _mm256_xor_ps(vf_g, SignXorAvx2(ksigns_iq2xs[qg >> 9])));
          const __m256 wu = _mm256_mul_ps(v_dl_u, _mm256_xor_ps(vf_u, SignXorAvx2(ksigns_iq2xs[qu >> 9])));
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          ag0 = _mm256_fmadd_ps(wg, xv0, ag0);
          au0 = _mm256_fmadd_ps(wu, xv0, au0);
          if constexpr (RT >= 2) {
            const __m256 xv1 = _mm256_loadu_ps(xr1 + k_off);
            ag1 = _mm256_fmadd_ps(wg, xv1, ag1);
            au1 = _mm256_fmadd_ps(wu, xv1, au1);
          }
        }
      }
    }
  }
  gate_dots[0] = HorizontalSumAvx2(ag0);
  up_dots[0] = HorizontalSumAvx2(au0);
  if constexpr (RT >= 2) {
    gate_dots[1] = HorizontalSumAvx2(ag1);
    up_dots[1] = HorizontalSumAvx2(au1);
  }
}

template <int RT>
void GgufDualDot2ChAvx2Tpl(
    const uint8_t* gate_row0,
    const uint8_t* up_row0,
    const uint8_t* gate_row1,
    const uint8_t* up_row1,
    int gate_type,
    int K,
    const float* const* x_rows,
    float* gate_dots0,
    float* up_dots0,
    float* gate_dots1,
    float* up_dots1) {
  using namespace onnxruntime::contrib::gguf_quant;
  const int nb = K / 256;
  const float* xr0 = x_rows[0];
  const float* xr1 = (RT >= 2) ? x_rows[1] : nullptr;
  __m256 ag0_0 = _mm256_setzero_ps();
  __m256 au0_0 = _mm256_setzero_ps();
  __m256 ag1_0 = _mm256_setzero_ps();
  __m256 au1_0 = _mm256_setzero_ps();
  __m256 ag0_1 = _mm256_setzero_ps();
  __m256 au0_1 = _mm256_setzero_ps();
  __m256 ag1_1 = _mm256_setzero_ps();
  __m256 au1_1 = _mm256_setzero_ps();

  if (gate_type == kGgmlTypeIQ2_S) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = gate_row0 + i * 82;
      const uint8_t* bu0 = up_row0 + i * 82;
      const uint8_t* bg1 = gate_row1 + i * 82;
      const uint8_t* bu1 = up_row1 + i * 82;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.25f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.25f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.25f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.25f;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0_g0 = dg0 * (0.5f + static_cast<float>(bg0[74 + ib32] & 0xF));
        const float db1_g0 = dg0 * (0.5f + static_cast<float>(bg0[74 + ib32] >> 4));
        const float db0_u0 = du0 * (0.5f + static_cast<float>(bu0[74 + ib32] & 0xF));
        const float db1_u0 = du0 * (0.5f + static_cast<float>(bu0[74 + ib32] >> 4));
        const float db0_g1 = dg1 * (0.5f + static_cast<float>(bg1[74 + ib32] & 0xF));
        const float db1_g1 = dg1 * (0.5f + static_cast<float>(bg1[74 + ib32] >> 4));
        const float db0_u1 = du1 * (0.5f + static_cast<float>(bu1[74 + ib32] & 0xF));
        const float db1_u1 = du1 * (0.5f + static_cast<float>(bu1[74 + ib32] >> 4));
        const uint8_t qhg0 = bg0[66 + ib32];
        const uint8_t qhu0 = bu0[66 + ib32];
        const uint8_t qhg1 = bg1[66 + ib32];
        const uint8_t qhu1 = bu1[66 + ib32];
        const int q_off = 2 + ib32 * 4;
        const int s_off = 34 + ib32 * 4;
        for (int l = 0; l < 4; ++l) {
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          __m256 xv1;
          if constexpr (RT >= 2) xv1 = _mm256_loadu_ps(xr1 + k_off);
          auto fma_iq2s = [&](const uint8_t* blk, uint8_t qh_b, float db, __m256& a0, __m256& a1) {
            const int idx = blk[q_off + l] | ((static_cast<int>(qh_b) << (8 - 2 * l)) & 0x300);
            const __m128i g8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2s_grid + idx));
            const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
            const __m256 w = _mm256_mul_ps(_mm256_set1_ps(db), _mm256_xor_ps(vf, SignXorAvx2(blk[s_off + l])));
            a0 = _mm256_fmadd_ps(w, xv0, a0);
            if constexpr (RT >= 2) a1 = _mm256_fmadd_ps(w, xv1, a1);
          };
          fma_iq2s(bg0, qhg0, (l < 2) ? db0_g0 : db1_g0, ag0_0, ag0_1);
          fma_iq2s(bu0, qhu0, (l < 2) ? db0_u0 : db1_u0, au0_0, au0_1);
          fma_iq2s(bg1, qhg1, (l < 2) ? db0_g1 : db1_g1, ag1_0, ag1_1);
          fma_iq2s(bu1, qhu1, (l < 2) ? db0_u1 : db1_u1, au1_0, au1_1);
        }
      }
    }
  } else if (gate_type == kGgmlTypeIQ2_XXS) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = gate_row0 + i * 66;
      const uint8_t* bu0 = up_row0 + i * 66;
      const uint8_t* bg1 = gate_row1 + i * 66;
      const uint8_t* bu1 = up_row1 + i * 66;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.25f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.25f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.25f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.25f;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const int a_off = 2 + 8 * ib32;
        const uint32_t aux_g0 = LoadU32Le(bg0 + a_off + 4);
        const uint32_t aux_u0 = LoadU32Le(bu0 + a_off + 4);
        const uint32_t aux_g1 = LoadU32Le(bg1 + a_off + 4);
        const uint32_t aux_u1 = LoadU32Le(bu1 + a_off + 4);
        const __m256 vdb_g0 = _mm256_set1_ps(dg0 * (0.5f + static_cast<float>(aux_g0 >> 28)));
        const __m256 vdb_u0 = _mm256_set1_ps(du0 * (0.5f + static_cast<float>(aux_u0 >> 28)));
        const __m256 vdb_g1 = _mm256_set1_ps(dg1 * (0.5f + static_cast<float>(aux_g1 >> 28)));
        const __m256 vdb_u1 = _mm256_set1_ps(du1 * (0.5f + static_cast<float>(aux_u1 >> 28)));
        for (int l = 0; l < 4; ++l) {
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          __m256 xv1;
          if constexpr (RT >= 2) xv1 = _mm256_loadu_ps(xr1 + k_off);
          auto fma_iq2xxs = [&](const uint8_t* blk, uint32_t aux32, __m256 v_db, __m256& a0, __m256& a1) {
            const __m128i g8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(iq2xxs_grid + blk[a_off + l]));
            const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
            const uint8_t s = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const __m256 w = _mm256_mul_ps(v_db, _mm256_xor_ps(vf, SignXorAvx2(s)));
            a0 = _mm256_fmadd_ps(w, xv0, a0);
            if constexpr (RT >= 2) a1 = _mm256_fmadd_ps(w, xv1, a1);
          };
          fma_iq2xxs(bg0, aux_g0, vdb_g0, ag0_0, ag0_1);
          fma_iq2xxs(bu0, aux_u0, vdb_u0, au0_0, au0_1);
          fma_iq2xxs(bg1, aux_g1, vdb_g1, ag1_0, ag1_1);
          fma_iq2xxs(bu1, aux_u1, vdb_u1, au1_0, au1_1);
        }
      }
    }
  } else if (gate_type == kGgmlTypeIQ3_XXS) {
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = gate_row0 + i * 98;
      const uint8_t* bu0 = up_row0 + i * 98;
      const uint8_t* bg1 = gate_row1 + i * 98;
      const uint8_t* bu1 = up_row1 + i * 98;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0)) * 0.5f;
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0)) * 0.5f;
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1)) * 0.5f;
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1)) * 0.5f;
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        const int q_off = 2 + 8 * ib32;
        const int s_off = 66 + 4 * ib32;
        const uint32_t aux_g0 = LoadU32Le(bg0 + s_off);
        const uint32_t aux_u0 = LoadU32Le(bu0 + s_off);
        const uint32_t aux_g1 = LoadU32Le(bg1 + s_off);
        const uint32_t aux_u1 = LoadU32Le(bu1 + s_off);
        const __m256 vdb_g0 = _mm256_set1_ps(dg0 * (0.5f + static_cast<float>(aux_g0 >> 28)));
        const __m256 vdb_u0 = _mm256_set1_ps(du0 * (0.5f + static_cast<float>(aux_u0 >> 28)));
        const __m256 vdb_g1 = _mm256_set1_ps(dg1 * (0.5f + static_cast<float>(aux_g1 >> 28)));
        const __m256 vdb_u1 = _mm256_set1_ps(du1 * (0.5f + static_cast<float>(aux_u1 >> 28)));
        for (int l = 0; l < 4; ++l) {
          const int k_off = i * 256 + ib32 * 32 + l * 8;
          const __m256 xv0 = _mm256_loadu_ps(xr0 + k_off);
          __m256 xv1;
          if constexpr (RT >= 2) xv1 = _mm256_loadu_ps(xr1 + k_off);
          auto fma_iq3xxs = [&](const uint8_t* blk, uint32_t aux32, __m256 v_db, __m256& a0, __m256& a1) {
            const uint64_t g64 = static_cast<uint64_t>(iq3xxs_grid[blk[q_off + 2 * l + 0]]) |
                                 (static_cast<uint64_t>(iq3xxs_grid[blk[q_off + 2 * l + 1]]) << 32);
            const __m128i g8 = _mm_cvtsi64_si128(static_cast<int64_t>(g64));
            const __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8));
            const uint8_t s = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const __m256 w = _mm256_mul_ps(v_db, _mm256_xor_ps(vf, SignXorAvx2(s)));
            a0 = _mm256_fmadd_ps(w, xv0, a0);
            if constexpr (RT >= 2) a1 = _mm256_fmadd_ps(w, xv1, a1);
          };
          fma_iq3xxs(bg0, aux_g0, vdb_g0, ag0_0, ag0_1);
          fma_iq3xxs(bu0, aux_u0, vdb_u0, au0_0, au0_1);
          fma_iq3xxs(bg1, aux_g1, vdb_g1, ag1_0, ag1_1);
          fma_iq3xxs(bu1, aux_u1, vdb_u1, au1_0, au1_1);
        }
      }
    }
  } else {
    // kGgmlTypeIQ4_XS
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    for (int i = 0; i < nb; ++i) {
      const uint8_t* bg0 = gate_row0 + i * 136;
      const uint8_t* bu0 = up_row0 + i * 136;
      const uint8_t* bg1 = gate_row1 + i * 136;
      const uint8_t* bu1 = up_row1 + i * 136;
      const float dg0 = Fp16ToFloatFast(LoadU16Le(bg0));
      const float du0 = Fp16ToFloatFast(LoadU16Le(bu0));
      const float dg1 = Fp16ToFloatFast(LoadU16Le(bg1));
      const float du1 = Fp16ToFloatFast(LoadU16Le(bu1));
      const uint16_t sh_g0 = LoadU16Le(bg0 + 2);
      const uint16_t sh_u0 = LoadU16Le(bu0 + 2);
      const uint16_t sh_g1 = LoadU16Le(bg1 + 2);
      const uint16_t sh_u1 = LoadU16Le(bu1 + 2);
      for (int ib = 0; ib < 8; ++ib) {
        const int k_off = i * 256 + ib * 32;
        const __m256 x0 = _mm256_loadu_ps(xr0 + k_off + 0), x1 = _mm256_loadu_ps(xr0 + k_off + 8);
        const __m256 x2 = _mm256_loadu_ps(xr0 + k_off + 16), x3 = _mm256_loadu_ps(xr0 + k_off + 24);
        __m256 y0, y1, y2, y3;
        if constexpr (RT >= 2) {
          y0 = _mm256_loadu_ps(xr1 + k_off + 0);
          y1 = _mm256_loadu_ps(xr1 + k_off + 8);
          y2 = _mm256_loadu_ps(xr1 + k_off + 16);
          y3 = _mm256_loadu_ps(xr1 + k_off + 24);
        }
        auto fma_iq4xs = [&](const uint8_t* blk, float d, uint16_t sh, __m256& a0, __m256& a1) {
          const int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
          const __m256 vd = _mm256_set1_ps(d * static_cast<float>(ls - 32));
          const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 8 + ib * 16));
          const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
          const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
          const __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)));
          const __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))));
          const __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)));
          const __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))));
          a0 = _mm256_fmadd_ps(w0, x0, _mm256_fmadd_ps(w1, x1, _mm256_fmadd_ps(w2, x2, _mm256_fmadd_ps(w3, x3, a0))));
          if constexpr (RT >= 2) {
            a1 = _mm256_fmadd_ps(w0, y0, _mm256_fmadd_ps(w1, y1, _mm256_fmadd_ps(w2, y2, _mm256_fmadd_ps(w3, y3, a1))));
          }
        };
        fma_iq4xs(bg0, dg0, sh_g0, ag0_0, ag0_1);
        fma_iq4xs(bu0, du0, sh_u0, au0_0, au0_1);
        fma_iq4xs(bg1, dg1, sh_g1, ag1_0, ag1_1);
        fma_iq4xs(bu1, du1, sh_u1, au1_0, au1_1);
      }
    }
  }
  gate_dots0[0] = HorizontalSumAvx2(ag0_0);
  up_dots0[0] = HorizontalSumAvx2(au0_0);
  gate_dots1[0] = HorizontalSumAvx2(ag1_0);
  up_dots1[0] = HorizontalSumAvx2(au1_0);
  if constexpr (RT >= 2) {
    gate_dots0[1] = HorizontalSumAvx2(ag0_1);
    up_dots0[1] = HorizontalSumAvx2(au0_1);
    gate_dots1[1] = HorizontalSumAvx2(ag1_1);
    up_dots1[1] = HorizontalSumAvx2(au1_1);
  }
}

}  // namespace

bool GgufDotRowAvx2(const uint8_t* row, int type, int K, const float* const* x_rows, int R, float* dots) {
  if (type != kGgmlTypeIQ4_NL && type != kGgmlTypeQ2_0 && type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS &&
      type != kGgmlTypeIQ2_XS && type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ4_XS && type != kGgmlTypeQ5_0) {
    return false;
  }
  int r = 0;
  for (; r + 2 <= R; r += 2) {
    GgufDotRowAvx2Tpl<2>(row, type, K, x_rows + r, dots + r);
  }
  if (r < R) {
    GgufDotRowAvx2Tpl<1>(row, type, K, x_rows + r, dots + r);
  }
  return true;
}

bool GgufDualDotRowAvx2(const uint8_t* gate_row, const uint8_t* up_row, int type, int K,
                        const float* const* x_rows, int R, float* gate_dots, float* up_dots) {
  if (type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS && type != kGgmlTypeIQ2_XS && type != kGgmlTypeIQ3_XXS &&
      type != kGgmlTypeIQ4_XS) {
    return false;
  }
  int r = 0;
  for (; r + 2 <= R; r += 2) {
    GgufDualDotRowAvx2Tpl<2>(gate_row, up_row, type, K, x_rows + r, gate_dots + r, up_dots + r);
  }
  if (r < R) {
    GgufDualDotRowAvx2Tpl<1>(gate_row, up_row, type, K, x_rows + r, gate_dots + r, up_dots + r);
  }
  return true;
}

bool GgufDualDot2ChAvx2(const uint8_t* gate_row0, const uint8_t* up_row0, const uint8_t* gate_row1,
                        const uint8_t* up_row1, int type, int K, const float* const* x_rows, int R,
                        float* gate_dots0, float* up_dots0, float* gate_dots1, float* up_dots1) {
  if (type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS && type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ4_XS) {
    return false;
  }
  int r = 0;
  for (; r + 2 <= R; r += 2) {
    GgufDualDot2ChAvx2Tpl<2>(gate_row0, up_row0, gate_row1, up_row1, type, K,
                             x_rows + r, gate_dots0 + r, up_dots0 + r, gate_dots1 + r, up_dots1 + r);
  }
  if (r < R) {
    GgufDualDot2ChAvx2Tpl<1>(gate_row0, up_row0, gate_row1, up_row1, type, K,
                             x_rows + r, gate_dots0 + r, up_dots0 + r, gate_dots1 + r, up_dots1 + r);
  }
  return true;
}

int GgufDotRowsBy4Avx2(const uint8_t* row0, size_t row_bytes, int n_rows, int type, int K,
                       const float* const* x_rows, int R, float* dots, int dots_ld) {
  int c = 0;
  if (type == kGgmlTypeIQ4_NL) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const int nb = K / 32;
    for (; c + 4 <= n_rows; c += 4) {
      const uint8_t* dr0 = row0 + static_cast<size_t>(c) * row_bytes;
      const uint8_t* dr1 = dr0 + row_bytes;
      const uint8_t* dr2 = dr1 + row_bytes;
      const uint8_t* dr3 = dr2 + row_bytes;
      int r = 0;
      for (; r + 2 <= R; r += 2) {
        const float* xr0 = x_rows[r + 0];
        const float* xr1 = x_rows[r + 1];
        __m256 a0_0 = _mm256_setzero_ps(), a1_0 = _mm256_setzero_ps(), a2_0 = _mm256_setzero_ps(), a3_0 = _mm256_setzero_ps();
        __m256 a0_1 = _mm256_setzero_ps(), a1_1 = _mm256_setzero_ps(), a2_1 = _mm256_setzero_ps(), a3_1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const float* xb0 = xr0 + i * 32;
          const float* xb1 = xr1 + i * 32;
          const __m256 x0 = _mm256_loadu_ps(xb0 + 0), x1 = _mm256_loadu_ps(xb0 + 8), x2 = _mm256_loadu_ps(xb0 + 16), x3 = _mm256_loadu_ps(xb0 + 24);
          const __m256 y0 = _mm256_loadu_ps(xb1 + 0), y1 = _mm256_loadu_ps(xb1 + 8), y2 = _mm256_loadu_ps(xb1 + 16), y3 = _mm256_loadu_ps(xb1 + 24);
          auto step_row2 = [&](const uint8_t* blk, __m256& acc0, __m256& acc1) MOE_PACKED_AVX2_LAMBDA_INLINE {
            const __m256 vd = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(blk)));
            const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
            const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
            const __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)));
            const __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))));
            const __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)));
            const __m256 w3 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))));
            acc0 = _mm256_fmadd_ps(w0, x0, _mm256_fmadd_ps(w1, x1, _mm256_fmadd_ps(w2, x2, _mm256_fmadd_ps(w3, x3, acc0))));
            acc1 = _mm256_fmadd_ps(w0, y0, _mm256_fmadd_ps(w1, y1, _mm256_fmadd_ps(w2, y2, _mm256_fmadd_ps(w3, y3, acc1))));
          };
          step_row2(dr0 + i * 18, a0_0, a0_1);
          step_row2(dr1 + i * 18, a1_0, a1_1);
          step_row2(dr2 + i * 18, a2_0, a2_1);
          step_row2(dr3 + i * 18, a3_0, a3_1);
        }
        const float d0_0 = HorizontalSumAvx2(a0_0), d1_0 = HorizontalSumAvx2(a1_0), d2_0 = HorizontalSumAvx2(a2_0), d3_0 = HorizontalSumAvx2(a3_0);
        const float d0_1 = HorizontalSumAvx2(a0_1), d1_1 = HorizontalSumAvx2(a1_1), d2_1 = HorizontalSumAvx2(a2_1), d3_1 = HorizontalSumAvx2(a3_1);
        dots[(r + 0) * dots_ld + c + 0] = d0_0;
        dots[(r + 0) * dots_ld + c + 1] = d1_0;
        dots[(r + 0) * dots_ld + c + 2] = d2_0;
        dots[(r + 0) * dots_ld + c + 3] = d3_0;
        dots[(r + 1) * dots_ld + c + 0] = d0_1;
        dots[(r + 1) * dots_ld + c + 1] = d1_1;
        dots[(r + 1) * dots_ld + c + 2] = d2_1;
        dots[(r + 1) * dots_ld + c + 3] = d3_1;
      }
      for (; r < R; ++r) {
        const float* xr = x_rows[r];
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const float* x_blk = xr + i * 32;
          const __m256 x0 = _mm256_loadu_ps(x_blk + 0);
          const __m256 x1 = _mm256_loadu_ps(x_blk + 8);
          const __m256 x2 = _mm256_loadu_ps(x_blk + 16);
          const __m256 x3 = _mm256_loadu_ps(x_blk + 24);
          auto step_row = [&](const uint8_t* blk, __m256& acc) MOE_PACKED_AVX2_LAMBDA_INLINE {
            const __m256 vd = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(blk)));
            const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m128i v_lo = _mm_shuffle_epi8(lut, _mm_and_si128(raw16, mask0f));
            const __m128i v_hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f));
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo))), x0, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8)))), x1, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi))), x2, acc);
            acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8)))), x3, acc);
          };
          step_row(dr0 + i * 18, a0);
          step_row(dr1 + i * 18, a1);
          step_row(dr2 + i * 18, a2);
          step_row(dr3 + i * 18, a3);
        }
        const float d0 = HorizontalSumAvx2(a0);
        const float d1 = HorizontalSumAvx2(a1);
        const float d2 = HorizontalSumAvx2(a2);
        const float d3 = HorizontalSumAvx2(a3);
        dots[r * dots_ld + c + 0] = d0;
        dots[r * dots_ld + c + 1] = d1;
        dots[r * dots_ld + c + 2] = d2;
        dots[r * dots_ld + c + 3] = d3;
      }
    }
  } else if (type == kGgmlTypeQ2_0) {
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256i ones = _mm256_set1_epi32(1);
    const int nb = K / 64;
    for (; c + 4 <= n_rows; c += 4) {
      const uint8_t* dr0 = row0 + static_cast<size_t>(c) * row_bytes;
      const uint8_t* dr1 = dr0 + row_bytes;
      const uint8_t* dr2 = dr1 + row_bytes;
      const uint8_t* dr3 = dr2 + row_bytes;
      int r = 0;
      for (; r + 2 <= R; r += 2) {
        const float* xr0 = x_rows[r + 0];
        const float* xr1 = x_rows[r + 1];
        __m256 a0_0 = _mm256_setzero_ps(), a1_0 = _mm256_setzero_ps(), a2_0 = _mm256_setzero_ps(), a3_0 = _mm256_setzero_ps();
        __m256 a0_1 = _mm256_setzero_ps(), a1_1 = _mm256_setzero_ps(), a2_1 = _mm256_setzero_ps(), a3_1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const uint8_t* b0 = dr0 + i * 18;
          const uint8_t* b1 = dr1 + i * 18;
          const uint8_t* b2 = dr2 + i * 18;
          const uint8_t* b3 = dr3 + i * 18;
          const __m256 vd0 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b0)));
          const __m256 vd1 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b1)));
          const __m256 vd2 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b2)));
          const __m256 vd3 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b3)));
          for (int w = 0; w < 8; ++w) {
            const __m256 xv0 = _mm256_loadu_ps(xr0 + i * 64 + w * 8);
            const __m256 xv1 = _mm256_loadu_ps(xr1 + i * 64 + w * 8);
            auto q_f32 = [&](uint16_t q16, __m256 vd) MOE_PACKED_AVX2_LAMBDA_INLINE {
              const __m256i qv = _mm256_sub_epi32(
                  _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(q16), shifts), mask3), ones);
              return _mm256_mul_ps(vd, _mm256_cvtepi32_ps(qv));
            };
            const __m256 w0 = q_f32(LoadU16Le(b0 + 2 + w * 2), vd0);
            const __m256 w1 = q_f32(LoadU16Le(b1 + 2 + w * 2), vd1);
            const __m256 w2 = q_f32(LoadU16Le(b2 + 2 + w * 2), vd2);
            const __m256 w3 = q_f32(LoadU16Le(b3 + 2 + w * 2), vd3);
            a0_0 = _mm256_fmadd_ps(w0, xv0, a0_0);
            a1_0 = _mm256_fmadd_ps(w1, xv0, a1_0);
            a2_0 = _mm256_fmadd_ps(w2, xv0, a2_0);
            a3_0 = _mm256_fmadd_ps(w3, xv0, a3_0);
            a0_1 = _mm256_fmadd_ps(w0, xv1, a0_1);
            a1_1 = _mm256_fmadd_ps(w1, xv1, a1_1);
            a2_1 = _mm256_fmadd_ps(w2, xv1, a2_1);
            a3_1 = _mm256_fmadd_ps(w3, xv1, a3_1);
          }
        }
        const float d0_0 = HorizontalSumAvx2(a0_0), d1_0 = HorizontalSumAvx2(a1_0), d2_0 = HorizontalSumAvx2(a2_0), d3_0 = HorizontalSumAvx2(a3_0);
        const float d0_1 = HorizontalSumAvx2(a0_1), d1_1 = HorizontalSumAvx2(a1_1), d2_1 = HorizontalSumAvx2(a2_1), d3_1 = HorizontalSumAvx2(a3_1);
        dots[(r + 0) * dots_ld + c + 0] = d0_0;
        dots[(r + 0) * dots_ld + c + 1] = d1_0;
        dots[(r + 0) * dots_ld + c + 2] = d2_0;
        dots[(r + 0) * dots_ld + c + 3] = d3_0;
        dots[(r + 1) * dots_ld + c + 0] = d0_1;
        dots[(r + 1) * dots_ld + c + 1] = d1_1;
        dots[(r + 1) * dots_ld + c + 2] = d2_1;
        dots[(r + 1) * dots_ld + c + 3] = d3_1;
      }
      for (; r < R; ++r) {
        const float* xr = x_rows[r];
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const uint8_t* b0 = dr0 + i * 18;
          const uint8_t* b1 = dr1 + i * 18;
          const uint8_t* b2 = dr2 + i * 18;
          const uint8_t* b3 = dr3 + i * 18;
          const __m256 vd0 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b0)));
          const __m256 vd1 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b1)));
          const __m256 vd2 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b2)));
          const __m256 vd3 = _mm256_set1_ps(Fp16ToFloatFast(LoadU16Le(b3)));
          for (int w = 0; w < 8; ++w) {
            const __m256 xv = _mm256_loadu_ps(xr + i * 64 + w * 8);
            auto q_f32 = [&](uint16_t q16, __m256 vd) MOE_PACKED_AVX2_LAMBDA_INLINE {
              const __m256i qv = _mm256_sub_epi32(
                  _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(q16), shifts), mask3), ones);
              return _mm256_mul_ps(vd, _mm256_cvtepi32_ps(qv));
            };
            a0 = _mm256_fmadd_ps(q_f32(LoadU16Le(b0 + 2 + w * 2), vd0), xv, a0);
            a1 = _mm256_fmadd_ps(q_f32(LoadU16Le(b1 + 2 + w * 2), vd1), xv, a1);
            a2 = _mm256_fmadd_ps(q_f32(LoadU16Le(b2 + 2 + w * 2), vd2), xv, a2);
            a3 = _mm256_fmadd_ps(q_f32(LoadU16Le(b3 + 2 + w * 2), vd3), xv, a3);
          }
        }
        const float d0 = HorizontalSumAvx2(a0);
        const float d1 = HorizontalSumAvx2(a1);
        const float d2 = HorizontalSumAvx2(a2);
        const float d3 = HorizontalSumAvx2(a3);
        dots[r * dots_ld + c + 0] = d0;
        dots[r * dots_ld + c + 1] = d1;
        dots[r * dots_ld + c + 2] = d2;
        dots[r * dots_ld + c + 3] = d3;
      }
    }
  }
  return c;
}

namespace {

inline __m256i MakeU256From4xU64Avx2(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) noexcept {
  const __m128i lo = _mm_unpacklo_epi64(_mm_cvtsi64_si128(static_cast<int64_t>(a0)),
                                        _mm_cvtsi64_si128(static_cast<int64_t>(a1)));
  const __m128i hi = _mm_unpacklo_epi64(_mm_cvtsi64_si128(static_cast<int64_t>(a2)),
                                        _mm_cvtsi64_si128(static_cast<int64_t>(a3)));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

inline __m256i MakeHalfPairI16Avx2(int16_t lo, int16_t hi) noexcept {
  return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(lo)), _mm_set1_epi16(hi), 1);
}

inline void Hsum4Avx2Local(__m256 a0, __m256 a1, __m256 a2, __m256 a3, float* out4) noexcept {
  const __m256 t0 = _mm256_hadd_ps(a0, a1);
  const __m256 t1 = _mm256_hadd_ps(a2, a3);
  const __m256 t2 = _mm256_hadd_ps(t0, t1);
  _mm_storeu_ps(out4, _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1)));
}

inline void QuantizeBlockQ8Avx2Local(const float* x, GgufQ8Block* blk) noexcept {
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256 x0 = _mm256_loadu_ps(x + 0);
  const __m256 x1 = _mm256_loadu_ps(x + 8);
  const __m256 x2 = _mm256_loadu_ps(x + 16);
  const __m256 x3 = _mm256_loadu_ps(x + 24);
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

  const __m256i sad = _mm256_sad_epu8(_mm256_xor_si256(qs_i8, _mm256_set1_epi8(-128)), _mm256_setzero_si256());
  const __m128i sad128 = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
  const int64_t u_sum = _mm_cvtsi128_si64(sad128) + _mm_extract_epi64(sad128, 1);
  blk->d = d;
  blk->bsum = static_cast<int32_t>(u_sum - 4096);
}

template <int RT>
void DualDot2ChQ8Avx2Batch(
    int type, const uint8_t* g0, const uint8_t* g1, const uint8_t* u0, const uint8_t* u1,
    const GgufQ8Block* const* x_rows, int K,
    float* gd0, float* ud0, float* gd1, float* ud1) noexcept {
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
        auto step = [&](const uint8_t* blk, float d025, __m256* acc_arr) MOE_PACKED_AVX2_LAMBDA_INLINE {
          const uint8_t* qs_ib = blk + 2 + 8 * ib;
          const uint32_t aux32 = LoadU32Le(blk + 66 + 4 * ib);
          const uint64_t g01 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[0]]) |
                               (static_cast<uint64_t>(iq3xxs_grid[qs_ib[1]]) << 32);
          const uint64_t g23 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[2]]) |
                               (static_cast<uint64_t>(iq3xxs_grid[qs_ib[3]]) << 32);
          const uint64_t g45 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[4]]) |
                               (static_cast<uint64_t>(iq3xxs_grid[qs_ib[5]]) << 32);
          const uint64_t g67 = static_cast<uint64_t>(iq3xxs_grid[qs_ib[6]]) |
                               (static_cast<uint64_t>(iq3xxs_grid[qs_ib[7]]) << 32);
          const __m256i grid = MakeU256From4xU64Avx2(g01, g23, g45, g67);
          const __m256i signs = MakeU256From4xU64Avx2(
              kSignBytes7Bit.mask[(aux32 >> 0) & 127u],
              kSignBytes7Bit.mask[(aux32 >> 7) & 127u],
              kSignBytes7Bit.mask[(aux32 >> 14) & 127u],
              kSignBytes7Bit.mask[(aux32 >> 21) & 127u]);
          const __m256i ls_v = _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1));
          for (int r = 0; r < RT; ++r) {
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(grid, _mm256_sign_epi8(xq[r], signs)), ls_v);
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d025 * xd[r]), acc_arr[r]);
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
        auto step = [&](const uint8_t* blk, float d0125, __m256* acc_arr) MOE_PACKED_AVX2_LAMBDA_INLINE {
          const uint8_t* qs_ib = blk + 2 + 4 * ib;
          const uint8_t* signs_ib = blk + 34 + 4 * ib;
          const uint32_t qh_b = blk[66 + ib];
          const uint8_t sc = blk[74 + ib];
          const __m256i grid = MakeU256From4xU64Avx2(
              iq2s_grid[qs_ib[0] | ((qh_b << 8) & 0x300u)],
              iq2s_grid[qs_ib[1] | ((qh_b << 6) & 0x300u)],
              iq2s_grid[qs_ib[2] | ((qh_b << 4) & 0x300u)],
              iq2s_grid[qs_ib[3] | ((qh_b << 2) & 0x300u)]);
          const __m256i signs = MakeU256From4xU64Avx2(
              kSignBytes8Bit.mask[signs_ib[0]], kSignBytes8Bit.mask[signs_ib[1]],
              kSignBytes8Bit.mask[signs_ib[2]], kSignBytes8Bit.mask[signs_ib[3]]);
          const __m256i ls_v = MakeHalfPairI16Avx2(static_cast<int16_t>(2 * (sc & 0xF) + 1),
                                                   static_cast<int16_t>(2 * (sc >> 4) + 1));
          for (int r = 0; r < RT; ++r) {
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(grid, _mm256_sign_epi8(xq[r], signs)), ls_v);
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d0125 * xd[r]), acc_arr[r]);
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
        auto step = [&](const uint8_t* blk, float d0125, __m256* acc_arr) MOE_PACKED_AVX2_LAMBDA_INLINE {
          const uint8_t* aux8 = blk + 2 + 8 * ib;
          const uint32_t aux32 = LoadU32Le(aux8 + 4);
          const __m256i grid = MakeU256From4xU64Avx2(
              iq2xxs_grid[aux8[0]], iq2xxs_grid[aux8[1]],
              iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[3]]);
          const __m256i signs = MakeU256From4xU64Avx2(
              kSignBytes7Bit.mask[(aux32 >> 0) & 127u], kSignBytes7Bit.mask[(aux32 >> 7) & 127u],
              kSignBytes7Bit.mask[(aux32 >> 14) & 127u], kSignBytes7Bit.mask[(aux32 >> 21) & 127u]);
          const __m256i ls_v = _mm256_set1_epi16(static_cast<int16_t>(2 * (aux32 >> 28) + 1));
          for (int r = 0; r < RT; ++r) {
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(grid, _mm256_sign_epi8(xq[r], signs)), ls_v);
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(d0125 * xd[r]), acc_arr[r]);
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
    const __m256i ones16 = _mm256_set1_epi16(1);
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
        __m256i xq[RT], ax[RT];
        float xd[RT];
        for (int r = 0; r < RT; ++r) {
          xq[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_rows[r][blk_idx].qs));
          ax[r] = _mm256_abs_epi8(xq[r]);
          xd[r] = x_rows[r][blk_idx].d;
        }
        auto step = [&](const uint8_t* blk, float d, uint16_t sh, __m256* acc_arr) MOE_PACKED_AVX2_LAMBDA_INLINE {
          const int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
          const float scale_w = d * static_cast<float>(ls - 32);
          const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 8 + ib * 16));
          const __m256i nibbles = _mm256_and_si256(
              _mm256_inserti128_si256(_mm256_castsi128_si256(raw16), _mm_srli_epi16(raw16, 4), 1), mask0f);
          const __m256i w_i8 = _mm256_shuffle_epi8(lut256, nibbles);
          for (int r = 0; r < RT; ++r) {
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax[r], _mm256_sign_epi8(w_i8, xq[r])), ones16);
            acc_arr[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(scale_w * xd[r]), acc_arr[r]);
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
    Hsum4Avx2Local(ag0[r], au0[r], ag1[r], au1[r], tmp);
    gd0[r] = tmp[0];
    ud0[r] = tmp[1];
    gd1[r] = tmp[2];
    ud1[r] = tmp[3];
  }
}

}  // namespace

void ConvertHalfRowAvx2(const uint16_t* src, bool is_bf16, float* dst, int K) noexcept {
  int c = 0;
  for (; c + 8 <= K; c += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
    const __m256 f = is_bf16 ? _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16))
                             : _mm256_cvtph_ps(h);
    _mm256_storeu_ps(dst + c, f);
  }
  for (; c < K; ++c) {
    dst[c] = is_bf16 ? Bf16BitsToFloatCpu(src[c]) : Fp16BitsToFloatCpu(src[c]);
  }
}

void QuantizeRowQ8Avx2(const float* src, int K, GgufQ8Block* dst) noexcept {
  const int nb = K / 32;
  for (int b = 0; b < nb; ++b) {
    QuantizeBlockQ8Avx2Local(src + b * 32, dst + b);
  }
}

bool GgufDualDot2ChQ8Avx2(int type, const uint8_t* g0, const uint8_t* g1, const uint8_t* u0, const uint8_t* u1,
                          const GgufQ8Block* const* x_q8_rows, int R, int K,
                          float* g_dots0, float* u_dots0, float* g_dots1, float* u_dots1) noexcept {
  if (type != kGgmlTypeIQ3_XXS && type != kGgmlTypeIQ2_S && type != kGgmlTypeIQ2_XXS && type != kGgmlTypeIQ4_XS) {
    return false;
  }
  int r = 0;
  for (; r + 2 <= R; r += 2) {
    DualDot2ChQ8Avx2Batch<2>(type, g0, g1, u0, u1, x_q8_rows + r, K,
                             g_dots0 + r, u_dots0 + r, g_dots1 + r, u_dots1 + r);
  }
  for (; r < R; ++r) {
    DualDot2ChQ8Avx2Batch<1>(type, g0, g1, u0, u1, x_q8_rows + r, K,
                             g_dots0 + r, u_dots0 + r, g_dots1 + r, u_dots1 + r);
  }
  return true;
}

int GgufDotRowsBy4Q8Avx2(int type, const uint8_t* row0, size_t row_bytes, int n_rows,
                         const GgufQ8Block* x_q8_base, size_t q8_stride,
                         int R, int K, float* dots, int dots_ld) noexcept {
  if (type != kGgmlTypeIQ4_NL && type != kGgmlTypeQ2_0) {
    return 0;
  }
  int c = 0;
  if (type == kGgmlTypeIQ4_NL) {
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m256i lut256 = _mm256_broadcastsi128_si256(lut128);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int nb = K / 32;
    for (; c + 4 <= n_rows; c += 4) {
      const uint8_t* dr0 = row0 + static_cast<size_t>(c) * row_bytes;
      const uint8_t* dr1 = dr0 + row_bytes;
      const uint8_t* dr2 = dr1 + row_bytes;
      const uint8_t* dr3 = dr2 + row_bytes;
      int r = 0;
      for (; r + 2 <= R; r += 2) {
        __m256 a0_0 = _mm256_setzero_ps(), a1_0 = _mm256_setzero_ps(), a2_0 = _mm256_setzero_ps(), a3_0 = _mm256_setzero_ps();
        __m256 a0_1 = _mm256_setzero_ps(), a1_1 = _mm256_setzero_ps(), a2_1 = _mm256_setzero_ps(), a3_1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const GgufQ8Block& xb0 = x_q8_base[static_cast<size_t>(r + 0) * q8_stride + i];
          const GgufQ8Block& xb1 = x_q8_base[static_cast<size_t>(r + 1) * q8_stride + i];
          const __m256i xq0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb0.qs));
          const __m256i xq1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb1.qs));
          const __m256i ax0 = _mm256_abs_epi8(xq0);
          const __m256i ax1 = _mm256_abs_epi8(xq1);
          auto step = [&](const uint8_t* blk, __m256& acc0, __m256& acc1) MOE_PACKED_AVX2_LAMBDA_INLINE {
            const float wd = Fp16ToFloatFast(LoadU16Le(blk));
            const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m256i nibbles = _mm256_and_si256(
                _mm256_inserti128_si256(_mm256_castsi128_si256(raw16), _mm_srli_epi16(raw16, 4), 1), mask0f);
            const __m256i w_i8 = _mm256_shuffle_epi8(lut256, nibbles);
            const __m256i p32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax0, _mm256_sign_epi8(w_i8, xq0)), ones16);
            const __m256i p32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax1, _mm256_sign_epi8(w_i8, xq1)), ones16);
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_0), _mm256_set1_ps(wd * xb0.d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32_1), _mm256_set1_ps(wd * xb1.d), acc1);
          };
          step(dr0 + i * 18, a0_0, a0_1);
          step(dr1 + i * 18, a1_0, a1_1);
          step(dr2 + i * 18, a2_0, a2_1);
          step(dr3 + i * 18, a3_0, a3_1);
        }
        Hsum4Avx2Local(a0_0, a1_0, a2_0, a3_0, dots + (r + 0) * dots_ld + c);
        Hsum4Avx2Local(a0_1, a1_1, a2_1, a3_1, dots + (r + 1) * dots_ld + c);
      }
      for (; r < R; ++r) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
          const GgufQ8Block& xb = x_q8_base[static_cast<size_t>(r) * q8_stride + i];
          const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
          const __m256i ax = _mm256_abs_epi8(xq);
          auto step = [&](const uint8_t* blk, __m256& acc) MOE_PACKED_AVX2_LAMBDA_INLINE {
            const float wd = Fp16ToFloatFast(LoadU16Le(blk));
            const __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 2));
            const __m256i nibbles = _mm256_and_si256(
                _mm256_inserti128_si256(_mm256_castsi128_si256(raw16), _mm_srli_epi16(raw16, 4), 1), mask0f);
            const __m256i w_i8 = _mm256_shuffle_epi8(lut256, nibbles);
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(w_i8, xq)), ones16);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(wd * xb.d), acc);
          };
          step(dr0 + i * 18, a0);
          step(dr1 + i * 18, a1);
          step(dr2 + i * 18, a2);
          step(dr3 + i * 18, a3);
        }
        Hsum4Avx2Local(a0, a1, a2, a3, dots + r * dots_ld + c);
      }
    }
  } else {
    // kGgmlTypeQ2_0
    const __m128i mask03 = _mm_set1_epi8(0x03);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int nb = K / 64;
    auto unpack_q20 = [&](const uint8_t* qs8) MOE_PACKED_AVX2_LAMBDA_INLINE {
      const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs8));
      const __m128i u0 = _mm_and_si128(raw, mask03);
      const __m128i u1 = _mm_and_si128(_mm_srli_epi16(raw, 2), mask03);
      const __m128i u2 = _mm_and_si128(_mm_srli_epi16(raw, 4), mask03);
      const __m128i u3 = _mm_and_si128(_mm_srli_epi16(raw, 6), mask03);
      const __m128i p01 = _mm_unpacklo_epi8(u0, u1);
      const __m128i p23 = _mm_unpacklo_epi8(u2, u3);
      return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi16(p01, p23)),
                                     _mm_unpackhi_epi16(p01, p23), 1);
    };
    auto hsum_i32 = [](__m256i v) MOE_PACKED_AVX2_LAMBDA_INLINE {
      const __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
      const __m128i s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
      return static_cast<float>(_mm_cvtsi128_si32(_mm_add_epi32(s64, _mm_srli_si128(s64, 4))));
    };
    for (; c + 4 <= n_rows; c += 4) {
      const uint8_t* dr0 = row0 + static_cast<size_t>(c) * row_bytes;
      const uint8_t* dr1 = dr0 + row_bytes;
      const uint8_t* dr2 = dr1 + row_bytes;
      const uint8_t* dr3 = dr2 + row_bytes;
      for (int r = 0; r < R; ++r) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < nb; ++i) {
          const uint8_t *b0 = dr0 + i * 18, *b1 = dr1 + i * 18, *b2 = dr2 + i * 18, *b3 = dr3 + i * 18;
          const float wd0 = Fp16ToFloatFast(LoadU16Le(b0)), wd1 = Fp16ToFloatFast(LoadU16Le(b1));
          const float wd2 = Fp16ToFloatFast(LoadU16Le(b2)), wd3 = Fp16ToFloatFast(LoadU16Le(b3));
          for (int h = 0; h < 2; ++h) {
            const GgufQ8Block& xb = x_q8_base[static_cast<size_t>(r) * q8_stride + i * 2 + h];
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
            const float bs = static_cast<float>(xb.bsum);
            const float p0 = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(unpack_q20(b0 + 2 + h * 8), xq), ones16)) - bs;
            const float p1 = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(unpack_q20(b1 + 2 + h * 8), xq), ones16)) - bs;
            const float p2 = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(unpack_q20(b2 + 2 + h * 8), xq), ones16)) - bs;
            const float p3 = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(unpack_q20(b3 + 2 + h * 8), xq), ones16)) - bs;
            s0 = std::fma(p0, wd0 * xb.d, s0);
            s1 = std::fma(p1, wd1 * xb.d, s1);
            s2 = std::fma(p2, wd2 * xb.d, s2);
            s3 = std::fma(p3, wd3 * xb.d, s3);
          }
        }
        dots[r * dots_ld + c + 0] = s0;
        dots[r * dots_ld + c + 1] = s1;
        dots[r * dots_ld + c + 2] = s2;
        dots[r * dots_ld + c + 3] = s3;
      }
    }
  }
  return c;
}

}  // namespace moe_packed_cpu
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#if defined(MOE_PACKED_POP_AVX2_CLANG_TARGET)
#pragma clang attribute pop
#undef MOE_PACKED_POP_AVX2_CLANG_TARGET
#endif

#if defined(MOE_PACKED_POP_AVX2_GCC_TARGET)
#pragma GCC pop_options
#undef MOE_PACKED_POP_AVX2_GCC_TARGET
#endif

#endif  // ORT_MOE_PACKED_CPU_X86_64
