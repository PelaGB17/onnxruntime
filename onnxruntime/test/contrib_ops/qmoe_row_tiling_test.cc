// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <optional>

#include "contrib_ops/cuda/moe/qmoe_row_tiling.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace qmoe = contrib::cuda::qmoe;

TEST(QMoERowTilingTest, PartitionsRowsIncludingPartialFinalTile) {
  const auto plan = qmoe::MakeRowTilePlan(65, 16, true);

  EXPECT_TRUE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 16);
  EXPECT_EQ(plan.TileCount(), 5);
  EXPECT_EQ(plan.RowOffset(0), 0);
  EXPECT_EQ(plan.RowsInTile(0), 16);
  EXPECT_EQ(plan.RowOffset(4), 64);
  EXPECT_EQ(plan.RowsInTile(4), 1);
}

TEST(QMoERowTilingTest, CanKeepRowsUntiled) {
  const auto plan = qmoe::MakeRowTilePlan(65, qmoe::kDisabledRowTileSize, false);

  EXPECT_FALSE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 65);
  EXPECT_EQ(plan.TileCount(), 1);
  EXPECT_EQ(plan.RowsInTile(0), 65);
}

TEST(QMoERowTilingTest, DefaultsToUntiledExecution) {
  const auto plan = qmoe::MakeRowTilePlan(512, qmoe::kDisabledRowTileSize, false);

  EXPECT_FALSE(plan.IsTiled());
  EXPECT_EQ(plan.rows_per_tile, 512);
  EXPECT_EQ(plan.TileCount(), 1);
}

TEST(QMoERowTilingTest, RejectsInvalidConfiguration) {
  EXPECT_FALSE(qmoe::IsValidRowTileSize(0));
  EXPECT_FALSE(qmoe::IsValidRowTileSize(-1));
  EXPECT_FALSE(qmoe::IsValidRowTileSize(static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
  EXPECT_THROW((void)qmoe::MakeRowTilePlan(65, 0, true), OnnxRuntimeException);
}

TEST(QMoERowTilingTest, BoundsRoutingScratchByTileSize) {
  constexpr size_t runner_workspace_bytes = 1024;
  const auto tiled_plan = qmoe::MakeRowTilePlan(65, 16, true);
  const auto untiled_plan = qmoe::MakeRowTilePlan(65, 16, false);
  const auto tiled = qmoe::MakeScratchLayout(runner_workspace_bytes, tiled_plan, 2);
  const auto untiled = qmoe::MakeScratchLayout(runner_workspace_bytes, untiled_plan, 2);

  EXPECT_EQ(tiled.scales_bytes, 128);
  EXPECT_EQ(tiled.indices_bytes, 128);
  EXPECT_EQ(tiled.permutation_bytes, 128);
  EXPECT_EQ(tiled.total_bytes, 1408);
  EXPECT_LT(tiled.total_bytes, untiled.total_bytes);
}

TEST(QMoERowTilingTest, BoundsFp8ExpertsByRoutesInTile) {
  const auto decode = qmoe::MakeRowTilePlan(1, 0, false);
  const auto prefill = qmoe::MakeRowTilePlan(17, 0, false);
  const auto tiled = qmoe::MakeRowTilePlan(65, 4, true);
  EXPECT_EQ(qmoe::MaxActiveExperts(decode, 512, 10), 10);
  EXPECT_EQ(qmoe::MaxActiveExperts(prefill, 512, 10), 170);
  EXPECT_EQ(qmoe::MaxActiveExperts(tiled, 512, 10), 40);
  EXPECT_EQ(qmoe::MaxActiveExperts(prefill, 16, 10), 16);
  EXPECT_EQ(qmoe::MaxActiveExperts(decode, 1, 1), 1);

  const size_t weight_bytes = SafeInt<size_t>(qmoe::MaxActiveExperts(decode, 512, 10)) * 3 * 2560 * 640 * 2;
  EXPECT_EQ(weight_bytes, 98304000u);  // 93.75 MiB instead of 4.6875 GiB.
  EXPECT_THROW((void)qmoe::MaxActiveExperts(decode, 512, 0), OnnxRuntimeException);
  EXPECT_THROW((void)qmoe::MaxActiveExperts(decode, 512, 513), OnnxRuntimeException);
}

TEST(QMoERowTilingTest, PrefillAuxBytesCoversCompactionBiasAndMultiPassBuffers) {
  const auto plan = qmoe::MakeRowTilePlan(64, 16, true);
  qmoe::PrefillAuxLayout layout;
  layout.use_expert_compaction = true;
  layout.num_experts = 16;
  layout.runner_num_experts = 8;
  layout.hidden_size = 128;
  layout.fc1_bias_cols = 128;
  layout.fc2_bias_cols = 128;
  layout.num_compact_passes = 2;
  layout.activation_bytes = 2;
  layout.global_scale_copy_bytes = 128;

  const size_t compact_indices = 16 * 4 * sizeof(int);
  const size_t expert_to_compact = 16 * sizeof(int);
  const size_t compact_to_expert = 8 * sizeof(int);
  const size_t biases = 2 * (8 * 128 * 2);
  const size_t multi_pass = 16 * 128 * 2 + 16 * 128 * sizeof(float);
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout),
            compact_indices + expert_to_compact + compact_to_expert + biases + multi_pass + 128);

  layout.num_compact_passes = 1;
  layout.fc1_bias_cols = 0;
  layout.fc2_bias_cols = 0;
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout), compact_indices + expert_to_compact + compact_to_expert + 128);

  layout.use_expert_compaction = false;
  EXPECT_EQ(qmoe::PrefillAuxBytes(plan, 4, layout), 128u);
}

TEST(QMoERowTilingTest, PrefillWorkingSetCountsDequantizedWeightsOnce) {
  constexpr size_t live_bytes = 1000;
  constexpr size_t dma_workspace_bytes = 400;
  constexpr size_t dequant_bytes = 300;

  // The DMA workspace already carries the dequantized weights.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, dma_workspace_bytes, dequant_bytes, true), 1400u);
  // Staging-only DMA workspace: the dequantized weights sit in the arena and are added once.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, dma_workspace_bytes, dequant_bytes, false), 1700u);
  // No DMA workspace.
  EXPECT_EQ(qmoe::PrefillWorkingSetBytes(live_bytes, 0, dequant_bytes, false), 1300u);
  EXPECT_THROW((void)qmoe::PrefillWorkingSetBytes(std::numeric_limits<size_t>::max(), 1, 0, true),
               OnnxRuntimeException);
}

namespace {

// Grouped-runner scratch per call modeled as a fixed part plus a per-row cost.
struct LinearScratch {
  size_t fixed_bytes;
  size_t bytes_per_row;
  size_t operator()(int64_t rows) const { return fixed_bytes + static_cast<size_t>(rows) * bytes_per_row; }
};

// Per-row bytes of a gated (SwiGLU) FP16 MoE on SM80: per route (row x top-k) the permutation maps (8 B), the
// overlapped GEMM inputs (2 * hidden), the overlapped GEMM outputs (max(4 * inter, 2 * hidden)), the dedicated FC1
// result (2 * inter) and 12 B of routing scratch; per row the blocked row map (4 B per expert). These are the
// row-proportional terms of CutlassMoeFCRunner::getWorkspaceDeviceBufferSizes plus MakeScratchLayout.
size_t GatedFp16BytesPerRow(size_t hidden, size_t inter, size_t experts, size_t top_k) {
  const size_t per_route = 8 + 2 * hidden + std::max(4 * inter, 2 * hidden) + 2 * inter + 12;
  return top_k * per_route + 4 * experts;
}

// Closed-form oracle for a linear scratch: floor((budget - fixed) / per_row) rounded down to the granularity and
// clamped to [min, max].
int64_t ExpectedTile(const LinearScratch& s, size_t budget) {
  if (s(qmoe::kMinDefaultRowTileSize) > budget) {
    return qmoe::kMinDefaultRowTileSize;
  }
  const int64_t rows = static_cast<int64_t>((budget - s.fixed_bytes) / s.bytes_per_row);
  const int64_t rounded = rows / qmoe::kDefaultRowTileGranularity * qmoe::kDefaultRowTileGranularity;
  return std::min(rounded, qmoe::kMaxDefaultRowTileSize);
}

void ExpectLargestFittingTile(const LinearScratch& s, size_t budget, int64_t tile) {
  EXPECT_EQ(tile % qmoe::kDefaultRowTileGranularity, 0);
  EXPECT_GE(tile, qmoe::kMinDefaultRowTileSize);
  EXPECT_LE(tile, qmoe::kMaxDefaultRowTileSize);
  if (tile > qmoe::kMinDefaultRowTileSize) {
    EXPECT_LE(s(tile), budget);
  }
  if (tile < qmoe::kMaxDefaultRowTileSize) {
    EXPECT_GT(s(tile + qmoe::kDefaultRowTileGranularity), budget);
  }
}

}  // namespace

TEST(QMoERowTilingTest, ScratchBudgetPicksLargestFittingTileRoundedDown) {
  const LinearScratch s{size_t{1} << 20, GatedFp16BytesPerRow(2048, 768, 128, 8)};
  ASSERT_EQ(s.bytes_per_row, 78496u);  // Qwen3-30B-A3B: matches the measured 2,563,128,448 B at 32,640 rows.

  // A budget of exactly scratch(16384) fits 16384 rows; one byte less rounds down a full granule.
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(s(16384), s), 16384);
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(s(16384) - 1, s), 15360);
  // Budgets between two granules round down, never up.
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(s(17407), s), 16384);
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(s(17408), s), 17408);
  for (size_t budget : {s(4096), s(5000), s(16384), s(16385), s(100000)}) {
    ExpectLargestFittingTile(s, budget, qmoe::RowTileSizeForScratchBudget(budget, s));
  }
}

TEST(QMoERowTilingTest, ScratchBudgetFloorAndCap) {
  const LinearScratch s{size_t{1} << 20, GatedFp16BytesPerRow(2048, 768, 128, 8)};
  // Below the floor: never smaller than kMinDefaultRowTileSize, even if that tile exceeds the budget.
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(s(4096) - 1, s), qmoe::kMinDefaultRowTileSize);
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(0, s), qmoe::kMinDefaultRowTileSize);
  // Tiny per-row scratch: capped at kMaxDefaultRowTileSize.
  const LinearScratch tiny{0, 1};
  EXPECT_EQ(qmoe::RowTileSizeForScratchBudget(qmoe::kDefaultRowTileScratchBudgetBytes, tiny),
            qmoe::kMaxDefaultRowTileSize);
  // An invalid search range is rejected.
  EXPECT_THROW((void)qmoe::RowTileSizeForScratchBudget(1, s, 1024, 1000, 4096), OnnxRuntimeException);
  EXPECT_THROW((void)qmoe::RowTileSizeForScratchBudget(1, s, 1024, 8192, 4096), OnnxRuntimeException);
}

TEST(QMoERowTilingTest, DefaultBudgetAcrossMoEShapes) {
  struct Shape {
    const char* name;
    size_t hidden, inter, experts, top_k;
    int64_t expected_tile;
  };
  // Expected tiles at the 1.25 GiB default budget with a 1 MiB fixed part (closed form, checked below too).
  const Shape shapes[] = {
      {"Qwen3-30B-A3B", 2048, 768, 128, 8, 16384},
      {"gpt-oss-20b-like", 2880, 2880, 32, 4, 14336},
      {"Phi-3.5-MoE-like", 4096, 6400, 16, 2, 14336},
      {"Mixtral-8x7B-like", 4096, 14336, 8, 2, 6144},
  };
  for (const auto& shape : shapes) {
    SCOPED_TRACE(shape.name);
    const LinearScratch s{size_t{1} << 20, GatedFp16BytesPerRow(shape.hidden, shape.inter, shape.experts, shape.top_k)};
    const int64_t tile = qmoe::RowTileSizeForScratchBudget(qmoe::kDefaultRowTileScratchBudgetBytes, s);
    EXPECT_EQ(tile, ExpectedTile(s, qmoe::kDefaultRowTileScratchBudgetBytes));
    EXPECT_EQ(tile, shape.expected_tile);
    ExpectLargestFittingTile(s, qmoe::kDefaultRowTileScratchBudgetBytes, tile);
  }
}

TEST(QMoERowTilingTest, UnsetUsesDefaultExplicitOverridesZeroDisables) {
  constexpr int64_t default_tile = 16384;
  // Unset: the budget-derived default.
  EXPECT_EQ(qmoe::ResolveRowTileSize(std::nullopt, default_tile), default_tile);
  // Explicit value: kept exactly, larger or smaller than the default.
  EXPECT_EQ(qmoe::ResolveRowTileSize(4096, default_tile), 4096);
  EXPECT_EQ(qmoe::ResolveRowTileSize(32768, default_tile), 32768);
  // Explicit 0: tiling off, so even a very long call stays one tile.
  const int64_t off = qmoe::ResolveRowTileSize(qmoe::kDisabledRowTileSize, default_tile);
  EXPECT_EQ(off, qmoe::kDisabledRowTileSize);
  const auto plan = qmoe::MakeRowTilePlan(131072, off, off != qmoe::kDisabledRowTileSize);
  EXPECT_FALSE(plan.IsTiled());
  EXPECT_EQ(plan.TileCount(), 1);
}

TEST(QMoERowTilingTest, DefaultTileLeavesCallsUpToTheTileUntiled) {
  const LinearScratch s{size_t{1} << 20, GatedFp16BytesPerRow(2048, 768, 128, 8)};
  const int64_t tile = qmoe::RowTileSizeForScratchBudget(qmoe::kDefaultRowTileScratchBudgetBytes, s);
  ASSERT_EQ(tile, 16384);

  const auto below = qmoe::MakeRowTilePlan(tile - 1, tile, true);
  EXPECT_FALSE(below.IsTiled());
  EXPECT_EQ(below.rows_per_tile, tile - 1);
  const auto at = qmoe::MakeRowTilePlan(tile, tile, true);
  EXPECT_FALSE(at.IsTiled());
  EXPECT_EQ(at.TileCount(), 1);
  const auto above = qmoe::MakeRowTilePlan(tile + 1, tile, true);
  EXPECT_TRUE(above.IsTiled());
  EXPECT_EQ(above.TileCount(), 2);
  EXPECT_EQ(above.RowsInTile(0), tile);
  EXPECT_EQ(above.RowsInTile(1), 1);

  // A call at or below the tile has the same plan, and so the same scratch request, as the untiled path.
  for (int64_t rows : {int64_t{1}, tile - 1, tile}) {
    const auto tiled = qmoe::MakeRowTilePlan(rows, tile, true);
    const auto untiled = qmoe::MakeRowTilePlan(rows, qmoe::kDisabledRowTileSize, false);
    EXPECT_EQ(tiled.rows_per_tile, untiled.rows_per_tile);
    EXPECT_EQ(qmoe::MakeScratchLayout(1000, tiled, 8).total_bytes,
              qmoe::MakeScratchLayout(1000, untiled, 8).total_bytes);
  }
  // Above the tile the request is bounded by the tile, not by the call.
  const size_t routing_bytes_in_tile = static_cast<size_t>(tile) * 8 * 12;
  EXPECT_LE(qmoe::MakeScratchLayout(s(tile) - routing_bytes_in_tile, qmoe::MakeRowTilePlan(131072, tile, true), 8)
                .total_bytes,
            qmoe::kDefaultRowTileScratchBudgetBytes);
}

}  // namespace test
}  // namespace onnxruntime
