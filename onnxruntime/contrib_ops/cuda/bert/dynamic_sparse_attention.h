// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/cuda_session_options.h"
#include "contrib_ops/cpu/bert/dynamic_sparse_attention_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct DsaKvHotCacheState;

template <typename T>
class DynamicSparseAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit DynamicSparseAttention(const OpKernelInfo& info);
  ~DynamicSparseAttention() override;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  int rotary_offset_;
  float scale_;
  bool has_scale_;
  float qk_norm_epsilon_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool use_smooth_softmax_;
  bool auxiliary_kv_shared_;
  bool strict_validation_;
  bool prefill_kernel_;
  bool prefill_int8_tc_;
  bool decode_int8_tc_;
  int decode_int8_tc_tiles_per_split_;
  int kv_hot_cache_tokens_;
  bool kv_stream_log_;
  DynamicSparseAttentionMode attention_mode_;
  DynamicSparseAttentionKvSource selected_kv_source_;
  mutable std::mutex hot_cache_mutex_;
  mutable std::unique_ptr<DsaKvHotCacheState> hot_cache_state_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
