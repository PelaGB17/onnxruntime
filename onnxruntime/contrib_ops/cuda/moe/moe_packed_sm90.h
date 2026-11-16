// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MoEPackedTile;
struct MoEPackedParams;

cudaError_t InitMoEPackedSm90WgmmaKernels();

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
    __nv_bfloat16* fc2_out);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
