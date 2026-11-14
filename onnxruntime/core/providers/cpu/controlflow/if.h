// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include <mutex>

#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {
class SessionState;
class OpKernelContextInternal;

class If : public controlflow::IControlFlowKernel {
 public:
  If(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  struct Info {
    Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in);
    const GraphViewer& subgraph;

    std::vector<bool> used_implicit_inputs;
    int num_implicit_inputs;
    int num_outputs;

    std::vector<std::string> subgraph_output_names;
  };

 private:
  // A branch whose outputs are all constant initializers runs no kernels. Its outputs are staged once
  // on the kernel's device so each run is a stream-ordered device copy (CUDA-graph capturable)
  // instead of a host-to-device copy of the initializers.
  struct ConstantBranch {
    bool is_constant = false;
    std::vector<OrtValue> outputs;
  };
  Status PrepareConstantBranches(OpKernelContextInternal& ctx) const;

  mutable std::mutex constant_branch_mutex_;
  mutable bool constant_branches_prepared_ = false;
  mutable ConstantBranch then_constant_;
  mutable ConstantBranch else_constant_;

  // Info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<Info> then_info_;
  std::unique_ptr<Info> else_info_;
  std::unique_ptr<FeedsFetchesManager> then_feeds_fetches_manager_;
  std::unique_ptr<FeedsFetchesManager> else_feeds_fetches_manager_;
};
}  // namespace onnxruntime
