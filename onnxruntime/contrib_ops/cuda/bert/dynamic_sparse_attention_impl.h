// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_parameters.h"
#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum DynamicSparseAttentionValidationError : int32_t {
  kDynamicSparseAttentionValidationOk = 0,
  kDynamicSparseAttentionInvalidCount = 1,
  kDynamicSparseAttentionInvalidPadding = 2,
  kDynamicSparseAttentionNegativeIndex = 3,
  kDynamicSparseAttentionIndexOutOfBounds = 4,
  kDynamicSparseAttentionDuplicateIndex = 5,
  kDynamicSparseAttentionNonCausalIndex = 6,
  kDynamicSparseAttentionInvalidSequenceLength = 7,
  kDynamicSparseAttentionInvalidPosition = 8,
  kDynamicSparseAttentionInvalidCommittedLength = 9,
  kDynamicSparseAttentionInvalidCacheScale = 10,
};

inline constexpr int kDsaKvHotPageTokens = 4;
inline constexpr int kDsaKvHotCtlInts = 16;

// Fixed-address device residency map and hot block slots for a streamed QSA layer whose authoritative K/V lives in
// device-mapped host memory. All pointers stay constant across runs so decode steps capture cleanly in CUDA graphs.
struct DsaKvHotCacheMap {
  void* hot_key = nullptr;        // [1, kv_num_heads, hot_capacity, head_size]
  void* hot_value = nullptr;      // [1, kv_num_heads, hot_capacity, head_size]
  int32_t* page_table = nullptr;  // [n_blocks] logical 4-token block -> physical slot in [0, n_slots), or -1
  int32_t* slot_block = nullptr;  // [n_slots] physical slot -> logical block, or -1
  int32_t* slot_stamp = nullptr;  // [n_slots] resolve epoch that last touched this slot
  int32_t* slot_ref = nullptr;    // [n_slots] CLOCK second-chance reference bit
  int32_t* miss_block = nullptr;  // [n_slots] scratch for missed logical blocks
  int32_t* miss_slot = nullptr;   // [n_slots] scratch for assigned physical slots
  int32_t* ctl = nullptr;         // [kDsaKvHotCtlInts] device control & 64-bit telemetry words
  int32_t n_blocks = 0;           // (cache_capacity + 3) / 4
  int32_t n_slots = 0;            // hot_capacity / 4
  int32_t hot_capacity = 0;       // n_slots * 4
};

// Shared single-layer VRAM staging buffers used by eager prefill chunks when K/V lives in mapped host memory.
struct DsaKvPrefillStage {
  void* stage_key = nullptr;         // [1, kv_num_heads, cache_capacity, head_size]
  void* stage_value = nullptr;       // [1, kv_num_heads, cache_capacity, head_size]
  uint32_t* block_bitmap = nullptr;  // [(n_blocks + 31) / 32] selected prefix blocks
  int32_t* stage_list = nullptr;     // [n_blocks + 1]: [0] = count, [1..count] = block indices
  int32_t n_blocks = 0;
};

struct DsaKvHotCacheCounters {
  uint64_t h2d_miss_blocks = 0;
  uint64_t block_lookups = 0;
  uint64_t decode_calls = 0;
  uint64_t prefill_staged_blocks = 0;
  bool overflow = false;
};

Status ResetDsaKvHotCacheMap(cudaStream_t stream, const DsaKvHotCacheMap& map);
DsaKvHotCacheCounters ReadDsaKvHotCacheCounters(const DsaKvHotCacheMap& map);

template <typename T>
struct DynamicSparseAttentionData {
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const T* auxiliary_key = nullptr;
  const T* auxiliary_value = nullptr;
  const int32_t* selected_indices = nullptr;
  const int32_t* selected_counts = nullptr;
  const int32_t* seqlens_k = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  const int64_t* position_ids = nullptr;
  const T* q_norm_weight = nullptr;
  const T* k_norm_weight = nullptr;
  const T* head_sink = nullptr;
  T* prepared_query = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
  T* output = nullptr;
  float* attention_workspace = nullptr;
  bool quantized_cache = false;
  bool fp16_cache = false;
  bool query_fp16 = false;
  bool key_fp16 = false;
  bool value_fp16 = false;
  bool prefill_kernel = true;
  bool prefill_int8_tc = false;
  bool decode_int8_tc = false;
  int decode_int8_tc_tiles_per_split = 0;
  const float* key_scale = nullptr;
  const float* value_scale = nullptr;
  const DsaKvHotCacheMap* hot_cache = nullptr;
  const DsaKvPrefillStage* prefill_stage = nullptr;
  bool use_hot_cache_decode = false;
};

size_t GetDynamicSparseAttentionValidationWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t max_shared_memory_per_block);

size_t GetDynamicSparseAttentionWorkspaceSize(
    const DynamicSparseAttentionParameters& parameters,
    size_t element_size,
    size_t max_shared_memory_per_block,
    bool quantized_cache = false,
    bool fp16_cache = false);

Status ValidateDynamicSparseAttentionOnDevice(
    cudaStream_t stream,
    const int32_t* selected_indices,
    const int32_t* selected_counts,
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    const DynamicSparseAttentionParameters& parameters,
    int32_t* error_flag,
    uint32_t* validation_bitmap,
    size_t max_shared_memory_per_block,
    bool copy_result_to_host,
    const float* key_scale = nullptr,
    const float* value_scale = nullptr,
    const int32_t* committed_length = nullptr,
    int committed_length_stride = 0);

template <typename T>
Status LaunchDynamicSparseAttention(
    cudaStream_t stream,
    const DynamicSparseAttentionParameters& parameters,
    const DynamicSparseAttentionData<T>& data,
    bool initialize_key_cache,
    bool initialize_value_cache,
    const cudaDeviceProp& device_prop);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
