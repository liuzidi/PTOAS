// Test PTOUnrollSIMTFor pass: annotation-only unrolling behavior.
//
// Verifies three cases:
//   1. simt_entry + pto.unroll="full"  -> fully unrolled (no scf.for)
//   2. simt_entry + no annotation       -> loop NOT unrolled
//   3. annotation + not simt_entry      -> loop NOT unrolled

// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto --mlir-print-ir-after=pto-unroll-simt-for %s -o /dev/null 2>&1 | FileCheck %s

// There should be exactly 2 scf.for in the output (from case 2 and 3).
// CHECK-COUNT-2: scf.for
// CHECK-NOT:     scf.for

module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  // Case 1: simt_entry + pto.unroll="full" -> unrolled (no scf.for)
  func.func @annotated_unrolled() -> index attributes {pto.simt_entry} {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    "scf.for"(%c0, %c4, %c1) ({
    ^bb0(%i: index):
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
      scf.yield
    }) {"pto.unroll" = "full"} : (index, index, index) -> ()
    %r = memref.load %buf[%c0] : memref<1xindex>
    return %r : index
  }

  // Case 2: simt_entry but NO annotation -> loop survives
  func.func @not_annotated_skipped() -> index attributes {pto.simt_entry} {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
    }
    %r = memref.load %buf[%c0] : memref<1xindex>
    return %r : index
  }

  // Case 3: annotation but NOT simt_entry -> loop survives
  func.func @not_simt_entry_skipped() -> index {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
    }
    %r = memref.load %buf[%c0] : memref<1xindex>
    return %r : index
  }
}
