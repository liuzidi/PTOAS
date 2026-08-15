// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSPLITCVMODULE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasKernelKind(ModuleOp module) {
  return module->hasAttr(FunctionKernelKindAttr::name);
}

static bool hasKernelKindChildModule(ModuleOp module) {
  return llvm::any_of(module.getOps<ModuleOp>(),
                      [](ModuleOp child) { return hasKernelKind(child); });
}

static std::optional<FunctionKernelKind>
getFunctionKernelKind(func::FuncOp funcOp) {
  auto kindAttr = funcOp->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr)
    return std::nullopt;
  return kindAttr.getKernelKind();
}

static bool hasTopLevelKernelKindFunction(ModuleOp module) {
  return llvm::any_of(module.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
    return getFunctionKernelKind(funcOp).has_value();
  });
}

static bool isSectionSplitCandidate(func::FuncOp funcOp);

static bool hasCVSections(ModuleOp module) {
  bool found = false;
  module.walk([&](func::FuncOp funcOp) {
    if (found || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
    WalkResult result = funcOp.walk([&](Operation *op) {
      if (isa<SectionCubeOp, SectionVectorOp>(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? WalkResult::interrupt()
                                   : WalkResult::advance();
  });
  return found;
}

static bool hasSectionKind(ModuleOp module, FunctionKernelKind kind) {
  bool found = false;
  module.walk([&](func::FuncOp funcOp) {
    if (found || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
    WalkResult result = funcOp.walk([&](Operation *op) {
      bool matches = kind == FunctionKernelKind::Cube
                         ? isa<SectionCubeOp>(op)
                         : isa<SectionVectorOp>(op);
      if (matches) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? WalkResult::interrupt()
                                   : WalkResult::advance();
  });
  return found;
}

static bool hasSectionKind(func::FuncOp funcOp, FunctionKernelKind kind) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    bool matches = kind == FunctionKernelKind::Cube ? isa<SectionCubeOp>(op)
                                                    : isa<SectionVectorOp>(op);
    if (matches) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasAnySection(func::FuncOp funcOp) {
  bool found = false;
  funcOp.walk([&](Operation *op) {
    if (isa<SectionCubeOp, SectionVectorOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool isSectionSplitCandidate(func::FuncOp funcOp) {
  return funcOp && !funcOp.isDeclaration() &&
         (pto::isPTOEntryFunction(funcOp) || hasAnySection(funcOp));
}

static LogicalResult verifyNoNestedSections(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status) || !isa<SectionCubeOp, SectionVectorOp>(op))
      return WalkResult::advance();
    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<SectionCubeOp, SectionVectorOp>(parent)) {
        status = op->emitError("nested pto.section.cube/vector is not allowed");
        return WalkResult::interrupt();
      }
      parent = parent->getParentOp();
    }
    return WalkResult::advance();
  });
  return status;
}

static void eraseUnusedSimtEntries(ModuleOp module) {
  SmallVector<ModuleOp> symbolTables{module};
  module.walk([&](ModuleOp nested) {
    if (nested != module)
      symbolTables.push_back(nested);
  });

  for (ModuleOp symbolTableModule : symbolTables) {
    SymbolTable symbolTable(symbolTableModule);
    SmallVector<func::FuncOp> deadEntries;
    for (func::FuncOp funcOp : symbolTableModule.getOps<func::FuncOp>()) {
      if (!funcOp->hasAttr(kPTOSimtEntryAttrName))
        continue;
      auto uses = symbolTable.getSymbolUses(funcOp, symbolTableModule);
      if (uses && uses->empty())
        deadEntries.push_back(funcOp);
    }
    for (func::FuncOp funcOp : deadEntries)
      funcOp.erase();
  }
}

static LogicalResult verifyExplicitKernelKindMatchesSections(ModuleOp module) {
  auto kindAttr = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr)
    return success();
  bool expectsCube = kindAttr.getKernelKind() == FunctionKernelKind::Cube;
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status))
      return WalkResult::interrupt();
    bool isCube = isa<SectionCubeOp>(op);
    bool isVector = isa<SectionVectorOp>(op);
    if (!isCube && !isVector)
      return WalkResult::advance();
    if (isCube != expectsCube) {
      status = op->emitError("conflicts with explicit pto.kernel_kind on its module");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

static LogicalResult verifySectionSplitCandidatesUseSections(ModuleOp module) {
  LogicalResult status = success();
  module.walk([&](func::FuncOp funcOp) {
    if (failed(status) || !isSectionSplitCandidate(funcOp))
      return WalkResult::advance();
    if (!hasAnySection(funcOp)) {
      status = funcOp.emitOpError(
          "must contain pto.section.cube or pto.section.vector in section "
          "input split by vpto-split-cv-module");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status;
}

static void eraseSectionSplitCandidatesWithoutSectionKind(ModuleOp module,
                                                          FunctionKernelKind kind) {
  SmallVector<func::FuncOp> eraseFuncs;
  module.walk([&](func::FuncOp funcOp) {
    if (isSectionSplitCandidate(funcOp) && !hasSectionKind(funcOp, kind))
      eraseFuncs.push_back(funcOp);
  });

  for (func::FuncOp funcOp : eraseFuncs)
    funcOp.erase();
}

static void replaceSectionWithBody(Operation *sectionOp) {
  Region &region = sectionOp->getRegion(0);
  Block &body = region.front();
  Block *parentBlock = sectionOp->getBlock();
  parentBlock->getOperations().splice(Block::iterator(sectionOp),
                                      body.getOperations());
  sectionOp->erase();
}

static void rewriteSectionsForKind(ModuleOp module, FunctionKernelKind kind) {
  SmallVector<Operation *> eraseSections;
  SmallVector<Operation *> inlineSections;
  module.walk([&](Operation *op) {
    if (kind == FunctionKernelKind::Cube) {
      if (isa<SectionVectorOp>(op))
        eraseSections.push_back(op);
      else if (isa<SectionCubeOp>(op))
        inlineSections.push_back(op);
    } else {
      if (isa<SectionCubeOp>(op))
        eraseSections.push_back(op);
      else if (isa<SectionVectorOp>(op))
        inlineSections.push_back(op);
    }
  });

  for (Operation *op : eraseSections)
    op->erase();
  for (Operation *op : inlineSections)
    replaceSectionWithBody(op);
  if (!eraseSections.empty() || !inlineSections.empty())
    eraseUnusedSimtEntries(module);
}

static ModuleOp cloneModuleForKind(ModuleOp source, FunctionKernelKind kind,
                                   OpBuilder &builder) {
  auto cloned = cast<ModuleOp>(source->clone());
  auto kindAttr = FunctionKernelKindAttr::get(cloned.getContext(), kind);
  cloned->setAttr(FunctionKernelKindAttr::name, kindAttr);
  eraseSectionSplitCandidatesWithoutSectionKind(cloned, kind);
  rewriteSectionsForKind(cloned, kind);
  for (func::FuncOp funcOp : cloned.getOps<func::FuncOp>())
    funcOp->setAttr(FunctionKernelKindAttr::name, kindAttr);
  builder.insert(cloned);
  return cloned;
}

static SmallVector<NamedAttribute>
getKernelChildAttrs(ModuleOp source, FunctionKernelKind kind) {
  SmallVector<NamedAttribute> attrs;
  attrs.reserve(source->getAttrs().size() + 1);
  for (NamedAttribute attr : source->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == FunctionKernelKindAttr::name)
      continue;
    attrs.push_back(attr);
  }
  attrs.push_back(NamedAttribute(
      StringAttr::get(source.getContext(), FunctionKernelKindAttr::name),
      FunctionKernelKindAttr::get(source.getContext(), kind)));
  return attrs;
}

static void cloneFuncIntoModule(func::FuncOp funcOp, ModuleOp target,
                                FunctionKernelKind kind, OpBuilder &builder) {
  builder.setInsertionPointToEnd(&target.getBodyRegion().front());
  auto cloned = cast<func::FuncOp>(funcOp->clone());
  cloned->setAttr(FunctionKernelKindAttr::name,
                  FunctionKernelKindAttr::get(cloned.getContext(), kind));
  builder.insert(cloned);
}

static void createChildModuleForFunctionKind(
    ModuleOp source, FunctionKernelKind kind, ArrayRef<func::FuncOp> helpers,
    ArrayRef<func::FuncOp> kernels, OpBuilder &builder) {
  auto child = ModuleOp::create(source.getLoc());
  child->setAttrs(DictionaryAttr::get(source.getContext(),
                                      getKernelChildAttrs(source, kind)));
  builder.insert(child);

  OpBuilder childBuilder(source.getContext());
  for (func::FuncOp helper : helpers)
    cloneFuncIntoModule(helper, child, kind, childBuilder);
  for (func::FuncOp kernel : kernels) {
    std::optional<FunctionKernelKind> kernelKind = getFunctionKernelKind(kernel);
    if (kernelKind && *kernelKind == kind)
      cloneFuncIntoModule(kernel, child, kind, childBuilder);
  }
}

static LogicalResult materializeTopLevelKernelKindFunctions(ModuleOp module) {
  SmallVector<func::FuncOp> helpers;
  SmallVector<func::FuncOp> kernels;
  bool needVector = false;
  bool needCube = false;

  for (Operation &op : module.getBodyRegion().front().getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp)
      return op.emitError()
             << "cannot split top-level kernel_kind functions with "
                "non-function sibling operation";

    std::optional<FunctionKernelKind> kind = getFunctionKernelKind(funcOp);
    if (!kind) {
      helpers.push_back(funcOp);
      continue;
    }

    kernels.push_back(funcOp);
    if (*kind == FunctionKernelKind::Vector)
      needVector = true;
    else if (*kind == FunctionKernelKind::Cube)
      needCube = true;
    else
      return funcOp.emitError("unsupported pto.kernel_kind on function");
  }

  if (kernels.empty())
    return success();

  if (hasCVSections(module))
    return module.emitError()
           << "cannot combine top-level function pto.kernel_kind splitting "
              "with pto.section.cube/vector splitting";

  SmallVector<NamedAttribute> outerAttrs;
  outerAttrs.reserve(module->getAttrs().size());
  for (NamedAttribute attr : module->getAttrs())
    if (attr.getName() != SymbolTable::getSymbolAttrName() &&
        attr.getName() != FunctionKernelKindAttr::name)
      outerAttrs.push_back(attr);

  auto outer = ModuleOp::create(module.getLoc());
  outer->setAttrs(DictionaryAttr::get(module.getContext(), outerAttrs));
  OpBuilder builder(outer.getBody(), outer.getBody()->end());
  if (needVector)
    createChildModuleForFunctionKind(module, FunctionKernelKind::Vector,
                                     helpers, kernels, builder);
  if (needCube)
    createChildModuleForFunctionKind(module, FunctionKernelKind::Cube, helpers,
                                     kernels, builder);

  module.getBodyRegion().takeBody(outer.getBodyRegion());
  module->setAttrs(outer->getAttrs());
  return success();
}

static LogicalResult materializeExplicitKernelKindSections(ModuleOp module) {
  auto kindAttr = module->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr)
    return success();
  if (failed(verifyNoNestedSections(module)) ||
      failed(verifyExplicitKernelKindMatchesSections(module)))
    return failure();
  rewriteSectionsForKind(module, kindAttr.getKernelKind());
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>())
    funcOp->setAttr(FunctionKernelKindAttr::name, kindAttr);
  return success();
}

static LogicalResult splitCVModule(ModuleOp module) {
  if (hasKernelKind(module))
    return materializeExplicitKernelKindSections(module);
  if (hasTopLevelKernelKindFunction(module))
    return materializeTopLevelKernelKindFunctions(module);
  if (hasKernelKindChildModule(module)) {
    for (ModuleOp child : module.getOps<ModuleOp>()) {
      if (!hasKernelKind(child))
        continue;
      if (failed(materializeExplicitKernelKindSections(child)))
        return failure();
    }
    return success();
  }
  if (!hasCVSections(module))
    return success();
  if (failed(verifyNoNestedSections(module)))
    return failure();
  if (failed(verifySectionSplitCandidatesUseSections(module)))
    return failure();
  bool needVector = hasSectionKind(module, FunctionKernelKind::Vector);
  bool needCube = hasSectionKind(module, FunctionKernelKind::Cube);
  if (!needVector && !needCube)
    return success();

  SmallVector<NamedAttribute> outerAttrs;
  outerAttrs.reserve(module->getAttrs().size());
  for (NamedAttribute attr : module->getAttrs())
    if (attr.getName() != SymbolTable::getSymbolAttrName())
      outerAttrs.push_back(attr);

  auto outer = ModuleOp::create(module.getLoc());
  outer->setAttrs(DictionaryAttr::get(module.getContext(), outerAttrs));
  OpBuilder builder(outer.getBody(), outer.getBody()->end());
  if (needVector)
    cloneModuleForKind(module, FunctionKernelKind::Vector, builder);
  if (needCube)
    cloneModuleForKind(module, FunctionKernelKind::Cube, builder);

  module.getBodyRegion().takeBody(outer.getBodyRegion());
  module->setAttrs(outer->getAttrs());
  return success();
}

struct VPTOSplitCVModulePass
    : public mlir::pto::impl::VPTOSplitCVModuleBase<VPTOSplitCVModulePass> {
  void runOnOperation() override {
    if (failed(splitCVModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSplitCVModulePass() {
  return std::make_unique<VPTOSplitCVModulePass>();
}
