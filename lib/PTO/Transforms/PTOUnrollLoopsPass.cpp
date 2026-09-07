// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollLoopsPass.cpp ---------------------------------------------===//
//
// Unroll explicitly annotated scf.for loops before LLVM lowering.
//
// Consumption contract for the unroll hint attributes (see PTO.h):
//
//   {pto.unroll = "full"}     - fully unrolled when the trip count is a
//                               positive constant.  Otherwise (dynamic trip)
//                               the loop cannot be unrolled: the attribute is
//                               removed with a remark and the loop is kept.
//   {pto.unroll_factor = N}   - unrolled by N when N is a signless-i32 value
//                               >= 2, the step is a positive constant, and N
//                               does not exceed max-unroll-factor (dynamic
//                               upper bounds are supported; an epilogue loop
//                               threading live-out values is generated).
//                               N == 1 is a no-op, a dynamic step or an
//                               over-cap factor cannot be unrolled: the
//                               attribute is removed with a remark in those
//                               cases.
//   {pto.vfsim.unroll_factor = N}
//                             - cost-model-selected factor with the same
//                               validation and native-unroll behavior.  It is
//                               rejected when combined with either user hint.
//
// {pto.unroll = "enable"} is never unrolled here; it is left untouched for
// pto-convert-scf-to-cf-with-loop-hints, which forwards it to the compiler's cost model as
// !llvm.loop.unroll.enable metadata.  It is recognized before every
// native-unroll guard (empty body, non-index induction variable, ...):
// those guards exist because loopUnrollByFactor cannot handle such loops,
// which is irrelevant for a hint that only becomes metadata.
//
// Anything malformed is a hard error reported here: an unknown pto.unroll
// value, both attributes on one loop, or an out-of-contract factor (wrong
// type/width, non-positive) all fail the pass.
//
// Loops without any unroll annotation are never modified.
//
// Historically this pass only handled {pto.unroll = "full"} inside SIMT
// contexts (pto-unroll-simt-for) to eliminate divergent control flow in
// SIMTVF kernels.  The SIMT restriction has been lifted because the
// annotation is always explicit user intent.
//
// Implementation note: the pass drives loopUnrollByFactor manually in a
// post-order fixpoint walk instead of going through the greedy pattern
// driver.  loopUnrollByFactor erases the rewritten loop with its own
// internal IRRewriter, bypassing the greedy driver's listener, so driver
// worklist entries pointing at erased ops would dangle.  Processing loops
// innermost-first (post-order) guarantees that by the time an outer loop is
// unrolled and erased, every other collected loop is either already
// processed or disjoint from it, so no stale pointer is ever dereferenced.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLLOOPS
#define GEN_PASS_DEF_PTOUNROLLSIMTFOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-loops"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Compute the constant trip count of *forOp*, or std::nullopt when any of
/// the bounds/step is not a compile-time constant.  Mirrors the arithmetic of
/// the historical pto-unroll-simt-for pass.
static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return std::nullopt;
  int64_t tripCount = (*ub - *lb + *step - 1) / *step;
  if (tripCount <= 0)
    return std::nullopt;
  return tripCount;
}

/// Outcome of handling one annotated loop.
enum class UnrollOutcome {
  Unchanged, //< no unroll happened (hint dropped or not applicable)
  Changed,   //< the loop was unrolled
  Error,     //< a diagnostic was emitted; the pass must fail
};

/// Shared implementation for the pto-unroll-loops pass (and its legacy
/// pto-unroll-simt-for alias).
struct PTOUnrollLoopsImpl {
  PTOUnrollLoopsImpl(int64_t maxFullUnrollTripCount, int64_t maxUnrollFactor)
      : maxFullUnrollTripCount(maxFullUnrollTripCount),
        maxUnrollFactor(maxUnrollFactor) {}

  int64_t maxFullUnrollTripCount;
  int64_t maxUnrollFactor;

  /// Try to fully unroll a loop annotated {pto.unroll = "full"}.  When the
  /// trip count is not constant the loop cannot be unrolled natively: drop
  /// the attribute with a remark and keep the loop.
  UnrollOutcome tryFullUnroll(scf::ForOp forOp) const {
    std::optional<int64_t> tripCount = getStaticTripCount(forOp);
    if (!tripCount) {
      forOp.emitRemark()
          << "'" << pto::kUnrollAttrName
          << " = \"full\"' loop has no constant trip count; cannot unroll "
             "natively, dropping the hint";
      forOp->removeAttr(pto::kUnrollAttrName);
      return UnrollOutcome::Unchanged;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "PTOUnrollLoops: fully unrolling annotated scf.for "
                  "tripCount="
               << *tripCount << " at " << forOp.getLoc() << "\n");

    // The loop is erased on success; capture the location for the guardrail
    // warning beforehand.
    Location loc = forOp.getLoc();
    if (failed(
            loopUnrollByFactor(forOp, static_cast<uint64_t>(*tripCount))))
      return UnrollOutcome::Unchanged;

    if (maxFullUnrollTripCount >= 0 && *tripCount > maxFullUnrollTripCount)
      mlir::emitWarning(loc)
          << "fully unrolled a loop with trip count " << *tripCount
          << ", which exceeds max-full-unroll-trip-count="
          << maxFullUnrollTripCount;

    return UnrollOutcome::Changed;
  }

  /// Try to unroll a loop annotated with a factor by N. Requires a statically
  /// known positive step; dynamic upper bounds are supported and produce an
  /// epilogue loop that threads live-out values.
  UnrollOutcome tryFactorUnroll(scf::ForOp forOp, int64_t factor,
                                llvm::StringRef attrName) const {
    // A huge factor makes loopUnrollByFactor clone the body an unbounded
    // number of times; refuse to unroll natively to bound compile time.
    if (maxUnrollFactor >= 0 && factor > maxUnrollFactor) {
      forOp.emitRemark()
          << "'" << attrName << "' = " << factor
          << " exceeds max-unroll-factor=" << maxUnrollFactor
          << "; cannot unroll natively, dropping the hint";
      forOp->removeAttr(attrName);
      return UnrollOutcome::Unchanged;
    }

    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!step || *step <= 0) {
      forOp.emitRemark()
          << "'" << attrName
          << "' loop has no constant positive step; cannot unroll natively, "
             "dropping the hint";
      forOp->removeAttr(attrName);
      return UnrollOutcome::Unchanged;
    }

    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollLoops: unrolling annotated scf.for "
                               "by factor="
                            << factor << " at " << forOp.getLoc() << "\n");

    // loopUnrollByFactor computes the trip count as
    // ceilDivPositive(upper - lower, step), which lowers to an *unsigned*
    // arith.divui.  A runtime `upper < lower` therefore makes the difference
    // negative, wraps it to a huge unsigned value, and turns a zero-trip loop
    // into a practically endless one (upstream flags exactly this with a
    // "TODO: Add dynamic asserts for negative lb/ub/step").  Guard both the
    // static and the dynamic case before handing the loop over.
    std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
    if (lb && ub) {
      if (*ub <= *lb) {
        forOp.emitRemark()
            << "'" << attrName
            << "' loop never iterates; dropping the hint";
        forOp->removeAttr(attrName);
        return UnrollOutcome::Unchanged;
      }
    } else {
      // Clamp the upper bound to at least the lower bound so the difference
      // can never be negative.  For `upper >= lower` the clamp is the
      // identity and the unrolled code is unchanged; for `upper < lower` it
      // makes both the unrolled main loop and the epilogue iterate zero
      // times, exactly like the original loop.
      OpBuilder builder(forOp);
      Value clampedUpperBound = builder.create<arith::MaxSIOp>(
          forOp.getLoc(), forOp.getUpperBound(), forOp.getLowerBound());
      forOp.setUpperBound(clampedUpperBound);
    }

    // loopUnrollByFactor creates the unrolled main loop plus (when the trip
    // count is not divisible or dynamic) an epilogue loop threading live-out
    // values, and promotes away single-iteration shells.  Nothing downstream
    // consumes annotations on the epilogue, so no post-hoc tagging happens
    // here: nested loops inside the body (including annotated clones picked
    // up by the next fixpoint round) are never touched.

    // Drop the factor attribute up front so neither the unrolled main loop
    // nor the epilogue clone keeps it.  The attribute is restored on failure.
    IntegerAttr factorAttr =
        IntegerAttr::get(IntegerType::get(forOp.getContext(), 32), factor);
    forOp->removeAttr(attrName);

    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(factor)))) {
      forOp->setAttr(attrName, factorAttr);
      return UnrollOutcome::Unchanged;
    }

    return UnrollOutcome::Changed;
  }

  UnrollOutcome tryUserFactorUnroll(scf::ForOp forOp,
                                    IntegerAttr factorAttr) const {
    return tryFactorUnroll(forOp, factorAttr.getInt(),
                           pto::kUnrollFactorAttrName);
  }

  UnrollOutcome tryVfSimFactorUnroll(scf::ForOp forOp,
                                     IntegerAttr factorAttr) const {
    return tryFactorUnroll(forOp, factorAttr.getInt(),
                           pto::kVfSimUnrollFactorAttrName);
  }

  /// Validate the hint attributes on one loop.  Emits a hard error for
  /// anything malformed: a wrongly typed attribute, an unknown `pto.unroll`
  /// value, both attributes on one loop, or an out-of-contract factor.
  LogicalResult validateHint(scf::ForOp forOp) const {
    Attribute unrollRaw = forOp->getAttr(pto::kUnrollAttrName);
    Attribute factorRaw = forOp->getAttr(pto::kUnrollFactorAttrName);
    Attribute vfSimFactorRaw =
        forOp->getAttr(pto::kVfSimUnrollFactorAttrName);

    // Wrong attribute *types* must not slip through as "no hint": the typed
    // getters below would return null and the loop would silently keep a
    // malformed annotation all the way down the pipeline.
    auto unrollAttr = dyn_cast_if_present<StringAttr>(unrollRaw);
    if (unrollRaw && !unrollAttr) {
      forOp.emitError() << "'" << pto::kUnrollAttrName
                        << "' must be a string attribute, got " << unrollRaw;
      return failure();
    }
    auto factorAttr = dyn_cast_if_present<IntegerAttr>(factorRaw);
    if (factorRaw && !factorAttr) {
      forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                        << "' must be a signless i32 attribute, got "
                        << factorRaw;
      return failure();
    }

    auto vfSimFactorAttr =
        dyn_cast_if_present<IntegerAttr>(vfSimFactorRaw);
    if (vfSimFactorRaw && !vfSimFactorAttr) {
      forOp.emitError() << "'" << pto::kVfSimUnrollFactorAttrName
                        << "' must be a signless i32 attribute, got "
                        << vfSimFactorRaw;
      return failure();
    }

    if (unrollAttr && factorAttr) {
      forOp.emitError()
          << "'" << pto::kUnrollAttrName << "' and '"
          << pto::kUnrollFactorAttrName
          << "' are mutually exclusive on one loop";
      return failure();
    }
    if (vfSimFactorAttr && (unrollAttr || factorAttr)) {
      forOp.emitError() << "'" << pto::kVfSimUnrollFactorAttrName
                        << "' cannot be combined with '"
                        << pto::kUnrollAttrName << "' or '"
                        << pto::kUnrollFactorAttrName << "'";
      return failure();
    }

    StringRef unrollValue = unrollAttr ? unrollAttr.getValue() : "";
    if (unrollAttr && unrollValue != pto::kUnrollFullValue &&
        unrollValue != pto::kUnrollEnableValue) {
      forOp.emitError() << "unknown '" << pto::kUnrollAttrName << "' value '"
                        << unrollAttr.getValue()
                        << "'; expected \"full\" (native full unroll) or "
                           "\"enable\" (forwarded to the compiler's cost "
                           "model by pto-convert-scf-to-cf-with-loop-hints)";
      return failure();
    }

    if (factorAttr && !pto::isValidUnrollFactorAttr(factorAttr)) {
      if (!factorAttr.getType().isSignlessInteger(32)) {
        forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                          << "' must be a signless i32 attribute, got "
                          << factorAttr.getType();
      } else {
        forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                          << "' must be a positive integer, got "
                          << factorAttr.getInt();
      }
      return failure();
    }
    if (vfSimFactorAttr && !pto::isValidUnrollFactorAttr(vfSimFactorAttr)) {
      if (!vfSimFactorAttr.getType().isSignlessInteger(32)) {
        forOp.emitError() << "'" << pto::kVfSimUnrollFactorAttrName
                          << "' must be a signless i32 attribute, got "
                          << vfSimFactorAttr.getType();
      } else {
        forOp.emitError() << "'" << pto::kVfSimUnrollFactorAttrName
                          << "' must be a positive integer, got "
                          << vfSimFactorAttr.getInt();
      }
      return failure();
    }

    return success();
  }

  /// Handle one annotated loop (hints are pre-validated by validateHint).
  /// Hints that cannot be unrolled natively are dropped with a remark (no
  /// metadata degradation path exists anymore).
  UnrollOutcome tryUnrollAnnotated(scf::ForOp forOp) const {
    auto unrollAttr = forOp->getAttrOfType<StringAttr>(pto::kUnrollAttrName);
    auto factorAttr =
        forOp->getAttrOfType<IntegerAttr>(pto::kUnrollFactorAttrName);
    auto vfSimFactorAttr = forOp->getAttrOfType<IntegerAttr>(
        pto::kVfSimUnrollFactorAttrName);

    // "enable" is the metadata hint owned by pto-convert-scf-to-cf-with-loop-hints: it never
    // reaches the native-unroll utility, so none of the guards below apply
    // to it.  This check must come first - dropping the hint on an
    // empty-body or non-index loop would break the "enable is consumed only
    // by the metadata pass" contract and silently lose the annotation.
    StringRef unrollValue = unrollAttr ? unrollAttr.getValue() : "";
    if (unrollValue == pto::kUnrollEnableValue) {
      return UnrollOutcome::Unchanged;
    }

    // Everything below only concerns the native-unroll hints ("full" and
    // pto.unroll_factor), so only those attributes are dropped.

    // loopUnrollByFactor reports success on empty-body loops without
    // changing them, which would make the fixpoint below loop forever.
    // Drop the hint on such loops instead.
    if (llvm::hasSingleElement(forOp.getBody()->getOperations())) {
      forOp.emitRemark() << "loop with a native unroll hint has an empty "
                            "body; dropping the hint";
      forOp->removeAttr(pto::kUnrollAttrName);
      forOp->removeAttr(pto::kUnrollFactorAttrName);
      forOp->removeAttr(pto::kVfSimUnrollFactorAttrName);
      return UnrollOutcome::Unchanged;
    }

    // scf.for also accepts signless integer induction variables, but
    // loopUnrollByFactor builds its bounds/step arithmetic with
    // arith::ConstantIndexOp unconditionally.  Unrolling an i16/i32 loop
    // would therefore emit mixed-type ops (e.g. arith.muli(i16, index)) and
    // an scf.for whose step no longer matches its bounds, both of which fail
    // the verifier.  Only index loops can be unrolled natively; anything
    // else keeps its loop and drops the hint.
    if (!forOp.getInductionVar().getType().isIndex()) {
      forOp.emitRemark()
          << "loop with a native unroll hint has a non-index induction "
             "variable ("
          << forOp.getInductionVar().getType()
          << "); native unrolling only supports index loops, dropping the "
             "hint";
      forOp->removeAttr(pto::kUnrollAttrName);
      forOp->removeAttr(pto::kUnrollFactorAttrName);
      return UnrollOutcome::Unchanged;
    }

    if (unrollAttr) {
      return tryFullUnroll(forOp);
    }

    if (factorAttr) {
      if (factorAttr.getInt() == 1) {
        forOp.emitRemark()
            << "'" << pto::kUnrollFactorAttrName
            << "' = 1 is a no-op; dropping the hint";
        forOp->removeAttr(pto::kUnrollFactorAttrName);
        return UnrollOutcome::Unchanged;
      }
      return tryUserFactorUnroll(forOp, factorAttr);
    }

    if (vfSimFactorAttr) {
      if (vfSimFactorAttr.getInt() == 1) {
        forOp.emitRemark()
            << "'" << pto::kVfSimUnrollFactorAttrName
            << "' = 1 is a no-op; dropping the hint";
        forOp->removeAttr(pto::kVfSimUnrollFactorAttrName);
        return UnrollOutcome::Unchanged;
      }
      return tryVfSimFactorUnroll(forOp, vfSimFactorAttr);
    }

    return UnrollOutcome::Unchanged;
  }

  LogicalResult run(func::FuncOp func) const {
    // Phase 1: validate every hint in the function, collecting all
    // diagnostics before failing (a function pass adaptor may stop
    // scheduling functions after the first failure, so per-function
    // completeness matters for deterministic test output).
    bool valid = true;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(pto::kUnrollAttrName) ||
          forOp->hasAttr(pto::kUnrollFactorAttrName) ||
          forOp->hasAttr(pto::kVfSimUnrollFactorAttrName)) {
        if (failed(validateHint(forOp))) {
          valid = false;
        }
      }
    });
    if (!valid)
      return failure();

    // Phase 2: unroll.  Only annotated loops are ever touched: unannotated
    // IR must come out byte-identical.  Unrolling an outer loop clones
    // annotated inner loops, so re-walk each round to pick the clones up
    // until a round makes no more changes.
    //
    // The walk is post-order (innermost first): loopUnrollByFactor erases
    // the unrolled loop, so processing a parent before its annotated
    // children would leave dangling pointers to the erased children in the
    // work list.  With post-order, erasing a loop only invalidates entries
    // that were already processed.
    //
    // Run to a true fixpoint rather than capping the rounds: a cap would
    // silently leave hints behind on deeply nested loops.  Every changing
    // round consumes the annotation of at least one loop, so the loop
    // terminates.
    while (true) {
      SmallVector<scf::ForOp, 8> annotated;
      func.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
        if (forOp->hasAttr(pto::kUnrollAttrName) ||
            forOp->hasAttr(pto::kUnrollFactorAttrName) ||
            forOp->hasAttr(pto::kVfSimUnrollFactorAttrName)) {
          annotated.push_back(forOp);
        }
      });
      if (annotated.empty())
        return success();

      bool changed = false;
      for (scf::ForOp forOp : annotated) {
        UnrollOutcome outcome = tryUnrollAnnotated(forOp);
        if (outcome == UnrollOutcome::Error) {
          return failure();
        }
        if (outcome == UnrollOutcome::Changed) {
          changed = true;
        }
      }
      if (!changed)
        return success();
    }
  }
};

struct PTOUnrollLoops
    : public pto::impl::PTOUnrollLoopsBase<PTOUnrollLoops> {
  using pto::impl::PTOUnrollLoopsBase<PTOUnrollLoops>::PTOUnrollLoopsBase;

  void runOnOperation() override {
    PTOUnrollLoopsImpl impl(maxFullUnrollTripCount, maxUnrollFactor);
    if (failed(impl.run(getOperation())))
      signalPassFailure();
  }
};

/// Legacy alias pass kept under the historical name "pto-unroll-simt-for".
struct PTOUnrollSIMTFor
    : public pto::impl::PTOUnrollSIMTForBase<PTOUnrollSIMTFor> {
  using pto::impl::PTOUnrollSIMTForBase<
      PTOUnrollSIMTFor>::PTOUnrollSIMTForBase;

  void runOnOperation() override {
    PTOUnrollLoopsImpl impl(maxFullUnrollTripCount, maxUnrollFactor);
    if (failed(impl.run(getOperation())))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructors
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOUnrollLoopsPass() {
  return std::make_unique<PTOUnrollLoops>();
}

std::unique_ptr<Pass> mlir::pto::createPTOUnrollSIMTForPass() {
  return std::make_unique<PTOUnrollSIMTFor>();
}
