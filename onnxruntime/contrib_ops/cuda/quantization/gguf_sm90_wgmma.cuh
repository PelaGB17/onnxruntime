// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gguf_sm90 {

// Build a canonical SM90 GMMA K-major SWIZZLE_128B (Swizzle<3,4,3>, LayoutType::B128 = 1) descriptor
// for a 1024-byte-aligned [N_rows, 64] BF16 tile stored as uint4[N_rows * 8] with swizzled index
// `row * 8 + (u4_col8 ^ (row & 7))`.
__device__ __forceinline__ uint64_t MakeGmmaDescB128KMajor(const void* smem_ptr) {
  const uint32_t addr16 = (static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) & 0x3FFFFu) >> 4;
  const uint32_t lo = addr16 | (1u << 16);
  const uint32_t hi = 64u | (1u << 30);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// Advance a K-major B128 GMMA descriptor to K-substep `k_step16` in 0..3 (offsets 0, 16, 32, 48 BF16s).
__device__ __forceinline__ uint64_t AdvanceGmmaDescK16(uint64_t base_desc, int k_step16) {
  const uint32_t lo = static_cast<uint32_t>(base_desc) + static_cast<uint32_t>(2 * k_step16);
  return (base_desc & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(lo);
}

// Store 16 bytes (8 BF16s) into a K-major B128-swizzled [N_rows, 64] shared-memory tile.
// When `aligned_16b` is true, `aligned_src` must be 16-byte aligned even if `!pred` (clamp row to 0);
// `cp.async.cg` hardware zero-fills when `src_bytes == 0` without reading global memory.
__device__ __forceinline__ void CpAsyncB128Store16B(
    uint4* smem_tile64,
    int row,
    int u4_col8,
    const void* aligned_or_raw_src,
    bool pred,
    bool aligned_16b) {
  const int dst_idx = row * 8 + (u4_col8 ^ (row & 7));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (aligned_16b) {
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_tile64 + dst_idx));
    const int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr), "l"(aligned_or_raw_src), "r"(src_bytes));
    return;
  }
#endif
  smem_tile64[dst_idx] = pred ? gguf_cuda::LoadU4Dev(aligned_or_raw_src) : make_uint4(0u, 0u, 0u, 0u);
}

__device__ __forceinline__ void CpAsyncCommitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void CpAsyncWaitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// Transpose 4x4 uint32 words across a 4-lane quad (`t = lane & 3`) to convert contiguous 8-BF16 subblocks
// (t=0: k0..7, t=1: k8..15, t=2: k16..23, t=3: k24..31) into SM90 GMMA `ALayout_64x16` register order:
//   .x -> k = {2t, 2t+1}       (wgmma #0 a0/a1)
//   .y -> k = {8+2t, 8+2t+1}   (wgmma #0 a2/a3)
//   .z -> k = {16+2t, 16+2t+1} (wgmma #1 a0/a1)
//   .w -> k = {24+2t, 24+2t+1} (wgmma #1 a2/a3)
__device__ __forceinline__ uint4 TransposeQuad4x4(uint4 v, int t) {
  const uint32_t s0 = (t & 1) ? v.x : v.y;
  const uint32_t s1 = (t & 1) ? v.z : v.w;
  const uint32_t r0 = __shfl_xor_sync(0xFFFFFFFFu, s0, 1);
  const uint32_t r1 = __shfl_xor_sync(0xFFFFFFFFu, s1, 1);
  const uint32_t a0 = (t & 1) ? r0 : v.x;
  const uint32_t a1 = (t & 1) ? v.y : r0;
  const uint32_t a2 = (t & 1) ? r1 : v.z;
  const uint32_t a3 = (t & 1) ? v.w : r1;

  const uint32_t u0 = (t & 2) ? a0 : a2;
  const uint32_t u1 = (t & 2) ? a1 : a3;
  const uint32_t q0 = __shfl_xor_sync(0xFFFFFFFFu, u0, 2);
  const uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, u1, 2);
  return make_uint4((t & 2) ? q0 : a0,
                    (t & 2) ? q1 : a1,
                    (t & 2) ? a2 : q0,
                    (t & 2) ? a3 : q1);
}

__device__ __forceinline__ void FenceAsyncSharedProxy() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void FenceWgmmaReg(uint32_t& r) {
#if defined(__CUDA_ARCH__)
  asm volatile("" : "+r"(r)::"memory");
#else
  (void)r;
#endif
}

__device__ __forceinline__ void FenceWgmmaU4(uint4& v) {
  FenceWgmmaReg(v.x);
  FenceWgmmaReg(v.y);
  FenceWgmmaReg(v.z);
  FenceWgmmaReg(v.w);
}

template <int kSubTiles>
__device__ __forceinline__ void FenceWgmmaAcc(float (&acc)[kSubTiles][2][2], int active_subtiles = kSubTiles) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int st = 0; st < kSubTiles; ++st) {
    if (st < active_subtiles) {
      asm volatile("" : "+f"(acc[st][0][0]), "+f"(acc[st][0][1]), "+f"(acc[st][1][0]), "+f"(acc[st][1][1])::"memory");
    }
  }
#else
  (void)acc;
  (void)active_subtiles;
#endif
}

__device__ __forceinline__ void WgmmaFence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void WgmmaCommitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void WgmmaWaitGroup() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ void WgmmaM64N16K16Rs(
    float (&acc)[2][2][2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
      : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
        "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
#else
  (void)acc;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

__device__ __forceinline__ void WgmmaM64N32K16Rs(
    float (&acc)[4][2][2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "{%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
      : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
        "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
        "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
        "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
#else
  (void)acc;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

__device__ __forceinline__ void WgmmaM64N64K16Rs(
    float (&acc)[8][2][2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
      : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
        "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
        "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
        "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1]),
        "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
        "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]),
        "+f"(acc[6][0][0]), "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]),
        "+f"(acc[7][0][0]), "+f"(acc[7][0][1]), "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
#else
  (void)acc;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

__device__ __forceinline__ void WgmmaM64N96K16Rs(
    float (&acc)[12][2][2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, "
      "{%48, %49, %50, %51}, %52, 1, 1, 1, 0;\n"
      : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
        "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
        "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
        "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1]),
        "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
        "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]),
        "+f"(acc[6][0][0]), "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]),
        "+f"(acc[7][0][0]), "+f"(acc[7][0][1]), "+f"(acc[7][1][0]), "+f"(acc[7][1][1]),
        "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]), "+f"(acc[8][1][1]),
        "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
        "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]),
        "+f"(acc[11][0][0]), "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
#else
  (void)acc;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

__device__ __forceinline__ void WgmmaM64N128K16Rs(
    float (&acc)[16][2][2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n"
      : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
        "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
        "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
        "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1]),
        "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
        "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]),
        "+f"(acc[6][0][0]), "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]),
        "+f"(acc[7][0][0]), "+f"(acc[7][0][1]), "+f"(acc[7][1][0]), "+f"(acc[7][1][1]),
        "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]), "+f"(acc[8][1][1]),
        "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
        "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]),
        "+f"(acc[11][0][0]), "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]),
        "+f"(acc[12][0][0]), "+f"(acc[12][0][1]), "+f"(acc[12][1][0]), "+f"(acc[12][1][1]),
        "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]), "+f"(acc[13][1][1]),
        "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
        "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
#else
  (void)acc;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

// Adaptive-N WGMMA dispatch on a 16-subtile (128-route) accumulator array using direct scalar register
// operands (never taking `&acc[0]`, so `ptxas` allocates zero local stack frame).
__device__ __forceinline__ void WgmmaM64AdaptiveN128K16Rs(
    float (&acc)[16][2][2],
    int n_inst,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint64_t desc_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  if (n_inst <= 16) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
          "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
  } else if (n_inst <= 32) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
          "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
          "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
          "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
  } else if (n_inst <= 64) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
          "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
          "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
          "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1]),
          "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]),
          "+f"(acc[6][0][0]), "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]),
          "+f"(acc[7][0][0]), "+f"(acc[7][0][1]), "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
  } else if (n_inst <= 96) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, "
        "{%48, %49, %50, %51}, %52, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]),
          "+f"(acc[1][0][0]), "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]),
          "+f"(acc[2][0][0]), "+f"(acc[2][0][1]), "+f"(acc[2][1][0]), "+f"(acc[2][1][1]),
          "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]), "+f"(acc[3][1][1]),
          "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]),
          "+f"(acc[6][0][0]), "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]),
          "+f"(acc[7][0][0]), "+f"(acc[7][0][1]), "+f"(acc[7][1][0]), "+f"(acc[7][1][1]),
          "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]), "+f"(acc[8][1][1]),
          "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]),
          "+f"(acc[11][0][0]), "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b));
  } else {
    WgmmaM64N128K16Rs(acc, a0, a1, a2, a3, desc_b);
  }
#else
  (void)acc;
  (void)n_inst;
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)desc_b;
#endif
}

}  // namespace gguf_sm90
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
