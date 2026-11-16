// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Leave room for the largest (32-column) decode tile.
inline constexpr int64_t kMatMulPackedMaxDimension = std::numeric_limits<int>::max() - 31;
inline constexpr int kMatMulPackedPrefillNTile = 4096;
inline constexpr int kMatMulPackedPrefillMTile = 128;
inline constexpr int kMatMulPackedBf16TcMinN = 65536;
inline constexpr int kMatMulPackedMaxBf16TcBodyRows = 16;
inline constexpr int kMatMulPackedMaxBf16TcLargeNRows = 32;
inline constexpr int kMatMulPackedMaxSm90WgmmaRows = 64;
inline constexpr int kMatMulPackedMinSm90WgmmaN = 2560;
inline constexpr int kMatMulPackedLargeNTcThreshold = 16384;
inline constexpr int kMatMulPackedMinSm90SplitKRows = 9;
inline constexpr int kMatMulPackedMinSm90SplitKCtas = 320;

inline constexpr int GetMatMulPackedSm90SplitK(int64_t k, int64_t n) {
  if (n > kMatMulPackedLargeNTcThreshold) return 1;
  const int64_t num_half128 = k / 128;
  const int64_t n_tiles = (n + 63) / 64;
  if (num_half128 < 8 || n_tiles <= 0) return 1;
  // Target a single resident wave of <= 528 CTAs (4 CTAs/SM * 132 Hopper SMs)
  // while keeping at least 4 x 128-K chunks (K >= 512) per split.
  const int64_t max_sk_by_wave = 528 / n_tiles;
  const int64_t max_sk_by_k = num_half128 / 4;
  int64_t sk = max_sk_by_wave < max_sk_by_k ? max_sk_by_wave : max_sk_by_k;
  if (sk > 16) sk = 16;
  if (sk < 1) sk = 1;
  return static_cast<int>(sk);
}

inline constexpr bool UseMatMulPackedBf16Sm90Wgmma(
    bool sm90_wgmma_supported, bool fp32_prefill, int ggml_type, int64_t k, int64_t n, int64_t m) {
  if (!sm90_wgmma_supported || fp32_prefill || m <= 8 || m > kMatMulPackedMaxSm90WgmmaRows) return false;
  if ((k % 128) != 0 || n < kMatMulPackedMinSm90WgmmaN) return false;
  if (n <= kMatMulPackedLargeNTcThreshold) {
    if (m < kMatMulPackedMinSm90SplitKRows) return false;
    const int64_t n_tiles = (n + 63) / 64;
    if (n_tiles * GetMatMulPackedSm90SplitK(k, n) < kMatMulPackedMinSm90SplitKCtas) return false;
    // At M > 32 on <= 31.5 MB weights (K <= 2560, N <= 6144), L2-resident DequantFast + cuBLAS is faster.
    if (m > 32 && k <= 2560 && n <= 6144) return false;
  }
  // ggml_type 14 == kGgmlTypeQ6_K, 8 == kGgmlTypeQ8_0
  return ggml_type == 14 || ggml_type == 8;
}

inline constexpr bool UseMatMulPackedBf16MultiRowTc(
    bool bf16_tc_supported, bool fp32_prefill, int64_t k, int64_t n, int64_t m) {
  (void)k;
  if (!bf16_tc_supported || fp32_prefill || m <= 8) return false;
  return (m <= kMatMulPackedMaxBf16TcBodyRows) ||
         (n > kMatMulPackedLargeNTcThreshold && m <= kMatMulPackedMaxBf16TcLargeNRows);
}

struct MatMulPackedParams {
  const uint8_t* b_data{nullptr};
  const float* global_scale{nullptr};
  bool allow_unrounded_gemv{false};
  int k{0};
  int n{0};
  int64_t row_bytes{0};
  int ggml_type{0};
  int activation_type{0};
  bool bf16_tc_supported{false};
  bool sm90_wgmma_supported{false};
  bool fp32_prefill{false};
};

Status GetMatMulPackedBf16KernelSupport(int device_id, bool& supported, bool* sm90_wgmma_supported = nullptr);

Status GetMatMulPackedWorkspaceBytes(const MatMulPackedParams& params, int64_t m, size_t& workspace_bytes);

Status LaunchMatMulPackedCuda(
    cublasHandle_t cublas,
    cudaStream_t stream,
    bool is_bf16,
    const void* a,
    const MatMulPackedParams& params,
    void* scratch_w,
    void* y,
    int64_t m);

template <typename T>
class MatMulPacked final : public CudaKernel {
 public:
  explicit MatMulPacked(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t k_{0};
  int64_t n_{0};
  int64_t ggml_type_{-1};
  int activation_type_{0};
  bool bf16_tc_supported_{false};
  bool sm90_wgmma_supported_{false};
  bool allow_unrounded_gemv_{false};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
