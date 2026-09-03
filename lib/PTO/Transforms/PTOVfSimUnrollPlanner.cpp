// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOVfSimUnrollPlanner.cpp - Invoke the VMI VfSim planner -----------===//
//===----------------------------------------------------------------------===//
//
// This pass is the PTOAS-owned boundary around VfSimulator. It deliberately
// passes generic MLIR IR to VfSim and performs no structural rewrite itself.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#ifdef PTO_ENABLE_VFSIM_IR_PLANNER
#include "native/IRPlanner.h"
#endif

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVFSIMUNROLLPLANNER
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PTOVfSimUnrollPlannerPass
    : public pto::impl::PTOVfSimUnrollPlannerBase<PTOVfSimUnrollPlannerPass> {
  using Base::Base;

  void runOnOperation() override {
#ifdef PTO_ENABLE_VFSIM_IR_PLANNER
    vfsim::PlannerOptions options;
    options.maxUnrollFactor = maxUnrollFactor;
    options.dumpCandidates = dumpCandidates;
    if (failed(vfsim::planVmiUnrollIR(getOperation(), options))) {
      signalPassFailure();
      return;
    }
    getOperation().walk([](scf::ForOp loop) {
      loop->removeAttr("pto.vmi.loop_fused");
      loop->removeAttr("pto.vmi.loop_fusion.id");
    });
#else
    getOperation().emitError()
        << "VMI VfSim unroll planning requires configuring PTOAS with "
           "-DPTO_ENABLE_VFSIM_COSTMODEL=ON";
    signalPassFailure();
#endif
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVfSimUnrollPlannerPass(
    const PTOVfSimUnrollPlannerOptions &options) {
  return std::make_unique<PTOVfSimUnrollPlannerPass>(options);
}
