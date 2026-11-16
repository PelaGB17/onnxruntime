// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/dynamic_sparse_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <type_traits>

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_impl.h"
#include "core/common/safeint.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T, ...)                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      DynamicSparseAttention,                                                  \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("TQuery", BuildKernelDefConstraints<T, MLFloat16>()) \
          .TypeConstraint("TValue", BuildKernelDefConstraints<T, MLFloat16>()) \
          .TypeConstraint("TCache", BuildKernelDefConstraints<__VA_ARGS__>())  \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())           \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())         \
          .MayInplace(3, 1)                                                    \
          .MayInplace(4, 2)                                                    \
          .InputMemoryType(OrtMemTypeCPUInput, 10),                            \
      DynamicSparseAttention<T>);

REGISTER_KERNEL_TYPED(float, float, MLFloat16, int8_t)
REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16, int8_t)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16, MLFloat16, int8_t)

#undef REGISTER_KERNEL_TYPED

namespace {

// Enables synchronous device metadata checks for diagnosing malformed standalone inputs.
constexpr const char* kStrictValidation = "ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION";

DynamicSparseAttentionMode ParseAttentionMode(const std::string& value) {
  ORT_ENFORCE(value == "selected_only" || value == "local_plus_selected",
              "DynamicSparseAttention: attention_mode must be selected_only or local_plus_selected.");
  return value == "selected_only" ? DynamicSparseAttentionMode::kSelectedOnly
                                  : DynamicSparseAttentionMode::kLocalPlusSelected;
}

DynamicSparseAttentionKvSource ParseKvSource(const std::string& value) {
  ORT_ENFORCE(value == "main" || value == "auxiliary",
              "DynamicSparseAttention: selected_kv_source must be main or auxiliary.");
  return value == "main" ? DynamicSparseAttentionKvSource::kMain
                         : DynamicSparseAttentionKvSource::kAuxiliary;
}

bool ParseBoolAttribute(const OpKernelInfo& info, const char* name, int64_t default_value) {
  const int64_t value = info.GetAttrOrDefault<int64_t>(name, default_value);
  ORT_ENFORCE(value == 0 || value == 1, "DynamicSparseAttention: ", name, " must be 0 or 1.");
  return value != 0;
}

}  // namespace

struct DsaKvHotCacheState {
  int device_id = -1;
  int allocated_blocks = 0;
  int allocated_slots = 0;
  size_t allocated_kv_bytes = 0;
  const void* checked_present_ptr = nullptr;
  bool is_host_backed_kv = false;
  void* hot_key_buf = nullptr;
  void* hot_val_buf = nullptr;
  int32_t* meta_buf = nullptr;
  DsaKvHotCacheMap map{};
  bool registered_with_stage_pool = false;

  void ReleaseBuffers() noexcept;
  ~DsaKvHotCacheState();
};

namespace {

struct DevicePrefillStageEntry {
  std::mutex mu;
  size_t active_users = 0;
  size_t allocated_kv_bytes = 0;
  int allocated_blocks = 0;
  void* stage_key = nullptr;
  void* stage_value = nullptr;
  uint32_t* block_bitmap = nullptr;
  int32_t* stage_list = nullptr;
  cudaStream_t last_stream = nullptr;
  cudaEvent_t ready_event = nullptr;

  void FreeUnlocked(int dev_id) noexcept {
    int prev = 0;
    const bool restore = (cudaGetDevice(&prev) == cudaSuccess && prev != dev_id);
    if (restore) static_cast<void>(cudaSetDevice(dev_id));
    if (ready_event != nullptr) {
      if (cudaEventDestroy(ready_event) != cudaSuccess) static_cast<void>(cudaGetLastError());
      ready_event = nullptr;
    }
    auto safe_free = [](void*& p) {
      if (p != nullptr) {
        if (cudaFree(p) != cudaSuccess) static_cast<void>(cudaGetLastError());
        p = nullptr;
      }
    };
    safe_free(stage_key);
    safe_free(stage_value);
    void* bm = block_bitmap;
    safe_free(bm);
    block_bitmap = nullptr;
    void* sl = stage_list;
    safe_free(sl);
    stage_list = nullptr;
    allocated_kv_bytes = 0;
    allocated_blocks = 0;
    last_stream = nullptr;
    if (restore) static_cast<void>(cudaSetDevice(prev));
  }
};

class SharedDsaPrefillStagePool {
 public:
  static SharedDsaPrefillStagePool& Instance() {
    static SharedDsaPrefillStagePool pool;
    return pool;
  }

  DevicePrefillStageEntry& EntryForDevice(int dev_id) {
    std::lock_guard<std::mutex> lock(map_mu_);
    auto& ptr = entries_[dev_id];
    if (!ptr) ptr = std::make_unique<DevicePrefillStageEntry>();
    return *ptr;
  }

  void Retain(int dev_id) {
    DevicePrefillStageEntry& entry = EntryForDevice(dev_id);
    std::lock_guard<std::mutex> lock(entry.mu);
    ++entry.active_users;
  }

  void Release(int dev_id) noexcept {
    DevicePrefillStageEntry* entry_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lock(map_mu_);
      auto it = entries_.find(dev_id);
      if (it == entries_.end() || !it->second) return;
      entry_ptr = it->second.get();
    }
    std::lock_guard<std::mutex> lock(entry_ptr->mu);
    if (entry_ptr->active_users > 0 && --entry_ptr->active_users == 0) {
      entry_ptr->FreeUnlocked(dev_id);
    }
  }

  Status EnsureLocked(DevicePrefillStageEntry& entry, size_t kv_bytes, int n_blocks, cudaStream_t stream,
                      DsaKvPrefillStage& out) {
    if (entry.ready_event == nullptr) {
      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&entry.ready_event, cudaEventDisableTiming));
    }
    if (entry.last_stream != nullptr && entry.last_stream != stream) {
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, entry.ready_event, 0));
    }
    if (kv_bytes > entry.allocated_kv_bytes) {
      if (entry.stage_key != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaFree(entry.stage_key));
        entry.stage_key = nullptr;
      }
      if (entry.stage_value != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaFree(entry.stage_value));
        entry.stage_value = nullptr;
      }
      CUDA_RETURN_IF_ERROR(cudaMalloc(&entry.stage_key, kv_bytes));
      const cudaError_t v_err = cudaMalloc(&entry.stage_value, kv_bytes);
      if (v_err != cudaSuccess) {
        static_cast<void>(cudaFree(entry.stage_key));
        entry.stage_key = nullptr;
        return CUDA_CALL(v_err);
      }
      entry.allocated_kv_bytes = kv_bytes;
    }
    if (n_blocks > entry.allocated_blocks) {
      if (entry.block_bitmap != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaFree(entry.block_bitmap));
        entry.block_bitmap = nullptr;
      }
      if (entry.stage_list != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaFree(entry.stage_list));
        entry.stage_list = nullptr;
      }
      const size_t bitmap_words = static_cast<size_t>((n_blocks + 31) / 32);
      CUDA_RETURN_IF_ERROR(cudaMalloc(&entry.block_bitmap, std::max<size_t>(1, bitmap_words) * sizeof(uint32_t)));
      const cudaError_t sl_err = cudaMalloc(&entry.stage_list, static_cast<size_t>(n_blocks + 1) * sizeof(int32_t));
      if (sl_err != cudaSuccess) {
        static_cast<void>(cudaFree(entry.block_bitmap));
        entry.block_bitmap = nullptr;
        return CUDA_CALL(sl_err);
      }
      entry.allocated_blocks = n_blocks;
    }
    out.stage_key = entry.stage_key;
    out.stage_value = entry.stage_value;
    out.block_bitmap = entry.block_bitmap;
    out.stage_list = entry.stage_list;
    out.n_blocks = n_blocks;
    return Status::OK();
  }

 private:
  std::mutex map_mu_;
  std::map<int, std::unique_ptr<DevicePrefillStageEntry>> entries_;
};

Status EnsureDsaHotCacheState(DsaKvHotCacheState& state, int dev_id, int n_blocks, int n_slots,
                              size_t kv_elem_bytes, int kv_num_heads, int head_size, cudaStream_t stream) {
  const int hot_capacity = n_slots * kDsaKvHotPageTokens;
  const size_t hot_kv_bytes =
      SafeInt<size_t>(kv_num_heads) * static_cast<size_t>(hot_capacity) * static_cast<size_t>(head_size) * kv_elem_bytes;
  if (state.hot_key_buf != nullptr && state.allocated_blocks == n_blocks &&
      state.allocated_slots == n_slots && state.allocated_kv_bytes == hot_kv_bytes) {
    return Status::OK();
  }

  cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &cap_status));
  ORT_RETURN_IF_NOT(cap_status == cudaStreamCaptureStatusNone,
                    "DynamicSparseAttention: cannot allocate or resize KV hot cache during CUDA graph capture.");

  if (!state.registered_with_stage_pool) {
    state.device_id = dev_id;
    SharedDsaPrefillStagePool::Instance().Retain(dev_id);
    state.registered_with_stage_pool = true;
  }
  state.ReleaseBuffers();

  const size_t meta_ints =
      static_cast<size_t>(n_blocks) + static_cast<size_t>(n_slots) * 5 + static_cast<size_t>(kDsaKvHotCtlInts);
  CUDA_RETURN_IF_ERROR(cudaMalloc(&state.hot_key_buf, hot_kv_bytes));
  cudaError_t err = cudaMalloc(&state.hot_val_buf, hot_kv_bytes);
  if (err != cudaSuccess) {
    state.ReleaseBuffers();
    return CUDA_CALL(err);
  }
  err = cudaMalloc(&state.meta_buf, meta_ints * sizeof(int32_t));
  if (err != cudaSuccess) {
    state.ReleaseBuffers();
    return CUDA_CALL(err);
  }

  state.allocated_blocks = n_blocks;
  state.allocated_slots = n_slots;
  state.allocated_kv_bytes = hot_kv_bytes;

  int32_t* cur = state.meta_buf;
  state.map.hot_key = state.hot_key_buf;
  state.map.hot_value = state.hot_val_buf;
  state.map.page_table = cur;
  cur += n_blocks;
  state.map.slot_block = cur;
  cur += n_slots;
  state.map.slot_stamp = cur;
  cur += n_slots;
  state.map.slot_ref = cur;
  cur += n_slots;
  state.map.miss_block = cur;
  cur += n_slots;
  state.map.miss_slot = cur;
  cur += n_slots;
  state.map.ctl = cur;
  state.map.n_blocks = n_blocks;
  state.map.n_slots = n_slots;
  state.map.hot_capacity = hot_capacity;

  return ResetDsaKvHotCacheMap(stream, state.map);
}

}  // namespace

void DsaKvHotCacheState::ReleaseBuffers() noexcept {
  int prev = 0;
  const bool restore = (device_id >= 0 && cudaGetDevice(&prev) == cudaSuccess && prev != device_id);
  if (restore) static_cast<void>(cudaSetDevice(device_id));
  auto safe_free = [](void*& p) {
    if (p != nullptr) {
      if (cudaFree(p) != cudaSuccess) static_cast<void>(cudaGetLastError());
      p = nullptr;
    }
  };
  safe_free(hot_key_buf);
  safe_free(hot_val_buf);
  void* m = meta_buf;
  safe_free(m);
  meta_buf = nullptr;
  map = DsaKvHotCacheMap{};
  allocated_blocks = 0;
  allocated_slots = 0;
  allocated_kv_bytes = 0;
  if (restore) static_cast<void>(cudaSetDevice(prev));
}

DsaKvHotCacheState::~DsaKvHotCacheState() {
  ReleaseBuffers();
  if (registered_with_stage_pool && device_id >= 0) {
    SharedDsaPrefillStagePool::Instance().Release(device_id);
    registered_with_stage_pool = false;
  }
}

template <typename T>
DynamicSparseAttention<T>::DynamicSparseAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() &&
                  num_heads > 0 && num_heads <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: num_heads must be a positive int.");
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() &&
                  kv_num_heads > 0 && kv_num_heads <= std::numeric_limits<int>::max() &&
                  num_heads % kv_num_heads == 0,
              "DynamicSparseAttention: kv_num_heads must be positive and divide num_heads.");
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  const int64_t is_causal = info.GetAttrOrDefault<int64_t>("is_causal", 1);
  ORT_ENFORCE(is_causal == 1, "DynamicSparseAttention only supports is_causal=1.");

  const int64_t local_window_size = info.GetAttrOrDefault<int64_t>("local_window_size", -1);
  ORT_ENFORCE(local_window_size == -1 ||
                  (local_window_size > 0 && local_window_size <= std::numeric_limits<int>::max()),
              "DynamicSparseAttention: local_window_size must be -1 or a positive int.");
  local_window_size_ = static_cast<int>(local_window_size);

  const int64_t rotary_offset = info.GetAttrOrDefault<int64_t>("rotary_offset", 0);
  ORT_ENFORCE(rotary_offset >= 0 && rotary_offset <= std::numeric_limits<int>::max(),
              "DynamicSparseAttention: rotary_offset must be a nonnegative int.");
  rotary_offset_ = static_cast<int>(rotary_offset);

  scale_ = 0.0f;
  has_scale_ = info.GetAttr("scale", &scale_).IsOK();
  qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  do_rotary_ = ParseBoolAttribute(info, "do_rotary", 0);
  rotary_interleaved_ = ParseBoolAttribute(info, "rotary_interleaved", 0);
  use_smooth_softmax_ = ParseBoolAttribute(info, "smooth_softmax", 0);
  auxiliary_kv_shared_ = ParseBoolAttribute(info, "auxiliary_kv_shared", 0);
  prefill_kernel_ = ReadCudaSessionSwitch(info, kOrtSessionOptionsCudaDsaPrefillKernel, "ORT_DSA_PREFILL_KERNEL", true);
  // Enable the tensor-core INT8-cache prefill and decode kernels by default on supported architectures (can still be
  // overridden via session options or ORT_DSA_PREFILL_INT8_TC / ORT_DSA_DECODE_INT8_TC).
  prefill_int8_tc_ =
      ReadCudaSessionSwitch(info, kOrtSessionOptionsCudaDsaPrefillInt8Tc, "ORT_DSA_PREFILL_INT8_TC", true);
  decode_int8_tc_ = ReadCudaSessionSwitch(info, kOrtSessionOptionsCudaDsaDecodeInt8Tc, "ORT_DSA_DECODE_INT8_TC", true);
  decode_int8_tc_tiles_per_split_ =
      ReadCudaSessionOption(info, kOrtSessionOptionsCudaDsaDecodeInt8TcTilesPerSplit,
                            "ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT", 0, 0, std::numeric_limits<int>::max());
  kv_hot_cache_tokens_ =
      ReadCudaSessionOption(info, kOrtSessionOptionsCudaDsaKvHotCacheTokens,
                            "ORT_DSA_KV_HOT_CACHE_TOKENS", 0, 0, std::numeric_limits<int>::max());
  kv_stream_log_ =
      ParseEnvironmentVariableWithDefault<int>("ORT_DSA_KV_STREAM_LOG", 0) != 0 ||
      ParseEnvironmentVariableWithDefault<int>("ORT_QMOE_SHARED_RESIDENT_LOG", 0) != 0;
  strict_validation_ = ParseEnvironmentVariableWithDefault<bool>(kStrictValidation, false);
  attention_mode_ = ParseAttentionMode(
      info.GetAttrOrDefault<std::string>("attention_mode", "selected_only"));
  selected_kv_source_ = ParseKvSource(
      info.GetAttrOrDefault<std::string>("selected_kv_source", "main"));
}

template <typename T>
DynamicSparseAttention<T>::~DynamicSparseAttention() {
  if (kv_stream_log_ && hot_cache_state_ && hot_cache_state_->map.ctl != nullptr) {
    const DsaKvHotCacheCounters c = ReadDsaKvHotCacheCounters(hot_cache_state_->map);
    if (c.decode_calls > 0 || c.prefill_staged_blocks > 0) {
      const double hit_pct =
          c.block_lookups > 0
              ? (100.0 * static_cast<double>(c.block_lookups > c.h2d_miss_blocks ? c.block_lookups - c.h2d_miss_blocks : 0) /
                 static_cast<double>(c.block_lookups))
              : 100.0;
      std::fprintf(stderr,
                   "DSA_KV_STREAM hot_tokens=%d hot_slots=%d decode_calls=%llu block_lookups=%llu "
                   "h2d_miss_blocks=%llu hit_pct=%.2f prefill_staged_blocks=%llu overflow=%d\n",
                   hot_cache_state_->map.hot_capacity, hot_cache_state_->map.n_slots,
                   static_cast<unsigned long long>(c.decode_calls),
                   static_cast<unsigned long long>(c.block_lookups),
                   static_cast<unsigned long long>(c.h2d_miss_blocks),
                   hit_pct,
                   static_cast<unsigned long long>(c.prefill_staged_blocks),
                   c.overflow ? 1 : 0);
    }
  }
}

template <typename T>
Status DynamicSparseAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* auxiliary_key = context->Input<Tensor>(5);
  const Tensor* auxiliary_value = context->Input<Tensor>(6);
  const Tensor* selected_indices = context->Input<Tensor>(7);
  const Tensor* selected_counts = context->Input<Tensor>(8);
  const Tensor* seqlens_k = context->Input<Tensor>(9);
  const Tensor* total_sequence_length = context->Input<Tensor>(10);
  const Tensor* cos_cache = context->Input<Tensor>(11);
  const Tensor* sin_cache = context->Input<Tensor>(12);
  const Tensor* position_ids = context->Input<Tensor>(13);
  const Tensor* q_norm_weight = context->Input<Tensor>(14);
  const Tensor* k_norm_weight = context->Input<Tensor>(15);
  const Tensor* head_sink = context->Input<Tensor>(16);
  const Tensor* committed_length =
      context->InputCount() > 17 ? context->Input<Tensor>(17) : nullptr;
  const Tensor* key_scale = context->InputCount() > 18 ? context->Input<Tensor>(18) : nullptr;
  const Tensor* value_scale = context->InputCount() > 19 ? context->Input<Tensor>(19) : nullptr;
  const bool quantized_cache = past_key != nullptr && past_key->IsDataType<int8_t>();
  // Without past inputs, the schema infers the present cache type from query.
  const bool fp16_cache = !std::is_same_v<T, MLFloat16> &&
                          (past_key != nullptr ? past_key->IsDataType<MLFloat16>() : query->IsDataType<MLFloat16>());
  const bool query_fp16 = !std::is_same_v<T, MLFloat16> &&
                          query != nullptr && query->IsDataType<MLFloat16>();
  const bool key_fp16 = !std::is_same_v<T, MLFloat16> &&
                        key != nullptr && key->IsDataType<MLFloat16>();
  const bool value_fp16 = !std::is_same_v<T, MLFloat16> &&
                          value != nullptr && value->IsDataType<MLFloat16>();
  ORT_RETURN_IF_NOT((key_scale != nullptr) == (value_scale != nullptr),
                    "DynamicSparseAttention: key_scale and value_scale must be provided together.");
  ORT_RETURN_IF_NOT(quantized_cache == (key_scale != nullptr),
                    "DynamicSparseAttention: INT8 caches require per-channel key_scale and value_scale.");
  ORT_RETURN_IF_NOT((past_key == nullptr || past_key->IsDataType<T>() || quantized_cache || fp16_cache) &&
                        (past_value == nullptr || (quantized_cache ? past_value->IsDataType<int8_t>()
                                                   : fp16_cache    ? past_value->IsDataType<MLFloat16>()
                                                                   : past_value->IsDataType<T>())),
                    "DynamicSparseAttention: cache types must match activations or both be FP16 or INT8.");
  ORT_RETURN_IF_NOT((query->IsDataType<T>() || ((fp16_cache || quantized_cache) && query_fp16)) &&
                        (key == nullptr || key->IsDataType<T>() || ((fp16_cache || quantized_cache) && key_fp16)) &&
                        (value == nullptr || value->IsDataType<T>() || ((fp16_cache || quantized_cache) && value_fp16)),
                    "DynamicSparseAttention: query, key, and value types must match T or FP16/INT8 cache.");

  DynamicSparseAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(dynamic_sparse_attention_helper::CheckInputs(
      query, key, value, past_key, past_value, auxiliary_key, auxiliary_value,
      selected_indices, selected_counts, seqlens_k, total_sequence_length,
      cos_cache, sin_cache, position_ids, q_norm_weight, k_norm_weight, head_sink,
      num_heads_, kv_num_heads_, local_window_size_, rotary_offset_, do_rotary_,
      auxiliary_kv_shared_, attention_mode_, selected_kv_source_, scale_, has_scale_,
      qk_norm_epsilon_, parameters));
  parameters.rotary_interleaved = rotary_interleaved_;
  parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr;
  if (quantized_cache) {
    ORT_RETURN_IF_NOT(past_value != nullptr &&
                          key_scale->Shape() == TensorShape({parameters.kv_num_heads * parameters.head_size}) &&
                          value_scale->Shape() == key_scale->Shape(),
                      "DynamicSparseAttention: INT8 cache scales must have shape [kv_num_heads * head_size].");
  }

  ORT_RETURN_IF_NOT(parameters.head_size <= GetDeviceProp().maxThreadsPerBlock,
                    "DynamicSparseAttention: head_size exceeds the CUDA device thread-block limit.");
  ORT_RETURN_IF(committed_length != nullptr &&
                    committed_length->Shape().Size() != 1 &&
                    committed_length->Shape().Size() != parameters.batch_size,
                "DynamicSparseAttention: committed_length must have 1 or batch_size int32 elements.");

  TensorShape output_shape({parameters.batch_size, parameters.sequence_length,
                            parameters.query_hidden_size});
  Tensor* output = context->Output(0, output_shape);
  ORT_RETURN_IF_NOT(output != nullptr, "DynamicSparseAttention: output is required.");

  TensorShape present_shape({parameters.batch_size, parameters.kv_num_heads,
                             parameters.cache_capacity, parameters.head_size});
  Tensor* present_key_output = context->OutputCount() > 1 ? context->Output(1, present_shape) : nullptr;
  Tensor* present_value_output = context->OutputCount() > 2 ? context->Output(2, present_shape) : nullptr;
  Tensor* validation_status_output =
      context->OutputCount() > 3 ? context->Output(3, TensorShape({1})) : nullptr;
  const size_t query_elements = SafeInt<size_t>(parameters.batch_size) *
                                parameters.sequence_length * parameters.query_hidden_size;
  auto prepared_query = GetScratchBuffer<T>(query_elements, GetComputeStream(context));

  const size_t cache_elements = SafeInt<size_t>(parameters.batch_size) *
                                parameters.kv_num_heads * parameters.cache_capacity *
                                parameters.head_size;
  const size_t cache_element_bytes =
      quantized_cache ? 1 : (fp16_cache ? sizeof(MLFloat16) : sizeof(T));
  const size_t cache_bytes = SafeInt<size_t>(cache_elements) * cache_element_bytes;
  IAllocatorUniquePtr<uint8_t> key_cache_scratch;
  IAllocatorUniquePtr<uint8_t> value_cache_scratch;
  void* key_cache = present_key_output == nullptr ? nullptr : present_key_output->MutableDataRaw();
  void* value_cache = present_value_output == nullptr ? nullptr : present_value_output->MutableDataRaw();
  if (key_cache == nullptr) {
    key_cache_scratch = GetScratchBuffer<uint8_t>(cache_bytes, GetComputeStream(context));
    key_cache = key_cache_scratch.get();
  }
  if (value_cache == nullptr) {
    value_cache_scratch = GetScratchBuffer<uint8_t>(cache_bytes, GetComputeStream(context));
    value_cache = value_cache_scratch.get();
  }

  using CudaT = typename OrtToCudaType<T>::type;
  DynamicSparseAttentionData<CudaT> data;
  data.prefill_kernel = prefill_kernel_;
  data.prefill_int8_tc = prefill_int8_tc_;
  data.decode_int8_tc = decode_int8_tc_;
  data.decode_int8_tc_tiles_per_split = decode_int8_tc_tiles_per_split_;
  data.query = reinterpret_cast<const CudaT*>(query->DataRaw());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->DataRaw());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->DataRaw());
  data.past_key = past_key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_key->DataRaw());
  data.past_value = past_value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(past_value->DataRaw());
  data.quantized_cache = quantized_cache;
  data.fp16_cache = fp16_cache;
  data.query_fp16 = query_fp16;
  data.key_fp16 = key_fp16;
  data.value_fp16 = value_fp16;
  data.key_scale = key_scale == nullptr ? nullptr : key_scale->Data<float>();
  data.value_scale = value_scale == nullptr ? nullptr : value_scale->Data<float>();
  data.auxiliary_key =
      auxiliary_key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(auxiliary_key->Data<T>());
  data.auxiliary_value = auxiliary_value == nullptr
                             ? data.auxiliary_key
                             : reinterpret_cast<const CudaT*>(auxiliary_value->Data<T>());
  data.selected_indices = selected_indices->Data<int32_t>();
  data.selected_counts = selected_counts->Data<int32_t>();
  data.seqlens_k = seqlens_k->Data<int32_t>();
  data.cos_cache = cos_cache == nullptr ? nullptr : reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
  data.sin_cache = sin_cache == nullptr ? nullptr : reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  data.position_ids = position_ids == nullptr ? nullptr : position_ids->Data<int64_t>();
  data.q_norm_weight =
      q_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(q_norm_weight->Data<T>());
  data.k_norm_weight =
      k_norm_weight == nullptr ? nullptr : reinterpret_cast<const CudaT*>(k_norm_weight->Data<T>());
  data.head_sink = head_sink == nullptr ? nullptr : reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  data.prepared_query = reinterpret_cast<CudaT*>(prepared_query.get());
  data.present_key = reinterpret_cast<CudaT*>(key_cache);
  data.present_value = reinterpret_cast<CudaT*>(value_cache);
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  cudaStream_t stream = Stream(context);
  if (strict_validation_) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
    ORT_RETURN_IF_NOT(capture_status == cudaStreamCaptureStatusNone,
                      "DynamicSparseAttention strict validation is not supported during CUDA graph capture.");
  }

  IAllocatorUniquePtr<int32_t> validation_error_scratch;
  int32_t* validation_error = nullptr;
  if (validation_status_output != nullptr) {
    validation_error = validation_status_output->MutableData<int32_t>();
  } else if (strict_validation_) {
    validation_error_scratch = GetScratchBuffer<int32_t>(1, GetComputeStream(context));
    validation_error = validation_error_scratch.get();
  }
  if (validation_error != nullptr) {
    const size_t validation_workspace_elements =
        GetDynamicSparseAttentionValidationWorkspaceSize(
            parameters, GetDeviceProp().sharedMemPerBlock);
    auto validation_workspace =
        GetScratchBuffer<uint32_t>(validation_workspace_elements, GetComputeStream(context));
    const int committed_length_stride =
        (committed_length != nullptr && committed_length->Shape().Size() == parameters.batch_size &&
         parameters.batch_size > 1)
            ? 1
            : 0;
    ORT_RETURN_IF_ERROR(ValidateDynamicSparseAttentionOnDevice(
        stream, data.selected_indices, data.selected_counts, data.seqlens_k,
        data.position_ids, parameters, validation_error, validation_workspace.get(),
        GetDeviceProp().sharedMemPerBlock, strict_validation_, data.key_scale, data.value_scale,
        committed_length == nullptr ? nullptr : committed_length->Data<int32_t>(),
        committed_length_stride));
  }

  const size_t attention_workspace_elements =
      GetDynamicSparseAttentionWorkspaceSize(
          parameters, sizeof(CudaT), GetDeviceProp().sharedMemPerBlock, quantized_cache, fp16_cache);
  auto attention_workspace =
      GetScratchBuffer<float>(attention_workspace_elements, GetComputeStream(context));
  data.attention_workspace = attention_workspace.get();

  const bool initialize_key_cache = data.past_key != data.present_key;
  const bool initialize_value_cache = data.past_value != data.present_value;

  // Streamed KV hot-cache / prefill-stage path when configured (`ep.cuda.dsa_kv_hot_cache_tokens` > 0) and
  // `cache_capacity` exceeds the hot-cache window for single-batch main-KV attention.
  DsaKvPrefillStage prefill_stage_desc{};
  DevicePrefillStageEntry* locked_stage_entry = nullptr;
  std::unique_lock<std::mutex> stage_lock;
  if (kv_hot_cache_tokens_ > 0 &&
      parameters.batch_size == 1 &&
      parameters.selected_kv_source == DynamicSparseAttentionKvSource::kMain &&
      !initialize_key_cache && !initialize_value_cache) {
    const int n_blocks = (parameters.cache_capacity + kDsaKvHotPageTokens - 1) / kDsaKvHotPageTokens;
    int n_slots = std::max(16, (kv_hot_cache_tokens_ + kDsaKvHotPageTokens - 1) / kDsaKvHotPageTokens);
    // Enforce >= 8 * (max_selected / 4 + 2) slots (4,112 slots = 16,448 tokens for S=8, budget=2048) on full-scale
    // QSA layers (`max_selected >= 512`) so 8-row speculative verification never overflows its hot cache.
    if (parameters.max_selected >= 512) {
      const int min_s8_slots = 8 * ((parameters.max_selected + kDsaKvHotPageTokens - 1) / kDsaKvHotPageTokens + 2);
      n_slots = std::max(n_slots, min_s8_slots);
    }
    if (n_slots < n_blocks) {
      int dev_id = 0;
      CUDA_RETURN_IF_ERROR(cudaGetDevice(&dev_id));
      bool use_streamed_layer = false;
      {
        std::lock_guard<std::mutex> hc_lock(hot_cache_mutex_);
        if (!hot_cache_state_) hot_cache_state_ = std::make_unique<DsaKvHotCacheState>();
        if (hot_cache_state_->checked_present_ptr != data.present_key) {
          cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
          CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &cap_status));
          if (cap_status == cudaStreamCaptureStatusNone) {
            cudaPointerAttributes attr{};
            const cudaError_t attr_err = cudaPointerGetAttributes(&attr, data.present_key);
            if (attr_err != cudaSuccess) {
              static_cast<void>(cudaGetLastError());
              hot_cache_state_->is_host_backed_kv = false;
            } else {
              hot_cache_state_->is_host_backed_kv = (attr.type != cudaMemoryTypeDevice);
            }
            hot_cache_state_->checked_present_ptr = data.present_key;
          }
        }
        if (hot_cache_state_->is_host_backed_kv) {
          ORT_RETURN_IF_ERROR(EnsureDsaHotCacheState(
              *hot_cache_state_, dev_id, n_blocks, n_slots, cache_element_bytes,
              parameters.kv_num_heads, parameters.head_size, stream));
          data.hot_cache = &hot_cache_state_->map;
          use_streamed_layer = true;
        }
      }

      if (use_streamed_layer) {
        const int eff_local =
            (parameters.attention_mode == DynamicSparseAttentionMode::kLocalPlusSelected && parameters.local_window_size > 0)
                ? std::min(parameters.local_window_size, parameters.cache_capacity)
                : 0;
        const int single_row_blocks =
            (std::min(parameters.max_selected, parameters.cache_capacity) + kDsaKvHotPageTokens - 1) / kDsaKvHotPageTokens +
            (eff_local + kDsaKvHotPageTokens - 1) / kDsaKvHotPageTokens + 2;
        const bool fits_decode_hot_cache =
            parameters.sequence_length <= 8 &&
            n_slots >= single_row_blocks + 2 * parameters.sequence_length;
        if (fits_decode_hot_cache) {
          data.use_hot_cache_decode = true;
        } else {
          cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
          CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &cap_status));
          ORT_RETURN_IF_NOT(cap_status == cudaStreamCaptureStatusNone,
                            "DynamicSparseAttention: prefill KV staging is only supported outside CUDA graph capture.");
          locked_stage_entry = &SharedDsaPrefillStagePool::Instance().EntryForDevice(dev_id);
          stage_lock = std::unique_lock<std::mutex>(locked_stage_entry->mu);
          ORT_RETURN_IF_ERROR(SharedDsaPrefillStagePool::Instance().EnsureLocked(
              *locked_stage_entry, cache_bytes, n_blocks, stream, prefill_stage_desc));
          data.prefill_stage = &prefill_stage_desc;
        }
      }
    }
  }

  Status launch_status = LaunchDynamicSparseAttention<CudaT>(
      stream, parameters, data, initialize_key_cache, initialize_value_cache, GetDeviceProp());
  if (locked_stage_entry != nullptr) {
    if (cudaEventRecord(locked_stage_entry->ready_event, stream) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
    }
    locked_stage_entry->last_stream = stream;
  }
  return launch_status;
}

template class DynamicSparseAttention<float>;
template class DynamicSparseAttention<MLFloat16>;
template class DynamicSparseAttention<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
