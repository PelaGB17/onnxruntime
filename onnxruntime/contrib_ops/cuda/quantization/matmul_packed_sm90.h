// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MatMulPackedParams;

cudaError_t LaunchMatMulPackedBf16Sm90WgmmaCuda(
    cudaStream_t stream,
    const __nv_bfloat16* a,
    const MatMulPackedParams& params,
    float* scratch_partials,
    __nv_bfloat16* y,
    int m);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
