# Rebase `main-llvm19-build` onto `hw-native-sys/main` (2026-08-31, v3)

## Scope and source state

- Source remote branch: `liuzidi/main-llvm19-build` at `5e8743190`.
- Target remote branch: `hw-native-sys/main` at `5eb87c21a`.
- Merge base: `614298ed2`.
- New branch: `rebase/hw-native-sys-main-20260831-v3`.
- Rebased history: 33 non-merge source commits, replayed with 69 sequencer steps.

The operation was a real history-preserving rebase, not a squash or tree
replacement:

```text
git rebase --rebase-merges --onto hw-native-sys/main \
  $(git merge-base liuzidi/main-llvm19-build hw-native-sys/main)
```

## Conflict resolutions

### `ba87d3578` — VMI loop fusion and load/store elision

Conflicts:

- `lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp`
- `lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp`

Resolution: retained the upstream shape-constraint architecture and restored
the local `Convert` and column-broadcast semantic families. The new VMI fusion,
memory-location, and tile-shape implementation files were registered in the
upstream `PTOTransforms` target instead of replacing its source list.

### `1b93503a0` — VMI TileLib candidates and PTODSL lowering

Conflicts:

- `include/PTO/IR/VMIOps.td`
- `lib/PTO/Transforms/VMILayoutAssignment.cpp`
- `lib/PTO/Transforms/VMILayoutSupport.cpp`
- `lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp`
- `lib/PTO/Transforms/VMIMaskGranularityAssignment.cpp`
- `ptodsl/ptodsl/_vmi_namespace.py`
- `ptodsl/ptodsl/tilelib/_selection.py`

Resolution: preserved upstream layout support and lowering structure while
retaining the local optional `updated_base` result and PTODSL VMI APIs. The
`SelectTemplateCandidate.cpp` and `VMIStubs.cpp` sources were explicitly added
to CMake. A post-rebase compile fix supplies an empty `updated_base` result type
when lowering a non-post-update stride store.

### `314baa5db` — VPTO lowering fixes

Conflicts:

- `ptodsl/ptodsl/_ops.py`
- four `test/lit/vmi_new` expectation files

Resolution: kept the upstream split Python operation architecture and the
upstream-compatible expectations rather than restoring the older monolithic
operation layer.

### `700f53089` — pass registration and CLI infrastructure

Conflicts:

- `include/PTO/Transforms/Passes.h`
- `tools/ptoas/ObjectEmission.cpp`
- `tools/ptoas/ptoas.cpp`

Resolution: preserved the upstream split driver (`ptoas.cpp`,
`ptoas_pipeline.cpp`, and `ptoas_cpp_rewrite.cpp`) and ObjectEmission/VFSIMT
contract. Local VMI, TileLib, daemon, and vecscope options were wired into that
split pipeline. Missing pass declarations and `MemPlanMode.h` inclusion were
restored after a clean-build audit.

### TileSpec and VMI API follow-up commits

Conflicts occurred in:

- `ptodsl/ptodsl/tilelib/constraints.py`
- `ptodsl/ptodsl/_vmi_namespace.py`
- `ptodsl/ptodsl/tilelib/_selection.py`

Resolution: retained the accumulated packed-dtype, compact-mode,
`s_fractal_size`, `pad_value`, and bidirectional `group_size`/`phys_vl`
semantics. Later commits were applied on top rather than choosing one side's
entire file at the end.

### `1a29a098f` and `3b31fe560` — PTO IR and compact-mode CLI fixes

Conflict: `tools/ptoas/ptoas.cpp`.

Resolution: ported the behavior into the upstream split pipeline. The memory
planner now uses the upstream typed `MemPlanMode::LOCAL_MEM_PLAN` option.

### `ec7031d99` — removed repeat-stride alignment

Conflict: `lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp`.

Resolution: kept the current unified-to-legacy implementation and applied the
new operation signature, including the optional stride-store result.

### `d8012a3dc` — simpler-compatible VPTO device ELF

Conflicts:

- `tools/ptoas/ObjectEmission.cpp`
- `tools/ptoas/ptoas.cpp`

Resolution: retained upstream ObjectEmission public APIs and VFSIMT size-fix
flow. Host-stub/wrapper source changes that applied cleanly remain, but the old
merged-device-only call shape was not allowed to replace the upstream API.
This feature therefore still needs a separate, focused port if direct merged
device ELF output is required; silently dropping VFSIMT handling was rejected.

## Post-rebase adaptations

- Added missing generated-pass prerequisites and pass factory declarations.
- Registered all new VMI/fusion sources in `lib/PTO/Transforms/CMakeLists.txt`.
- Restored parameterized PTODSL/TileLang expansion, template metadata,
  candidate selection, VMI fusion, and vecscope barrier passes.
- Restored automatic TileLib daemon startup and cleanup in the split compiler
  pipeline. Focused lifecycle tests cover repeated startup, graceful SIGTERM,
  socket cleanup, timeout cleanup, and unrelated-child handling.
- Preserved upstream ObjectEmission and VFSIMT size-fix interfaces.

## Build and test results

Environment:

- Python: `/home/liuzidi/conda-envs/ptoas/bin/python3`
- LLVM/MLIR: `/home/liuzidi/llvm-workspace/llvm-project/build-shared`
- PTOAS version: `0.61`

Build:

```text
cmake --build build --target clean
cmake --build build --parallel 64 --target all
```

Result: successful after the adaptations above.

PythonDSL:

```text
PATH=/home/liuzidi/PTOAS/build/tools/ptoas:$PATH \
  /home/liuzidi/conda-envs/ptoas/bin/python3 \
  -m unittest discover -s ptodsl/tests
```

Result: **252 tests passed, 0 failed** in 101.933 seconds.

Full lit:

```text
/home/liuzidi/llvm-workspace/llvm-project/build-shared/bin/llvm-lit \
  build/test/lit
```

Result: **1931 discovered; 1873 passed, 57 failed, 1 unsupported** in
140.75 seconds.

The remaining failures are no longer a common daemon-startup failure. They are
concentrated in:

- VMI layout/lowering output expectations;
- fusion-region and fusion-plan IR expectations;
- TileLib candidate/expansion expectations;
- vecscope barrier expectations;
- Bisheng command-line argument tests.

An origin check against `hw-native-sys/main` shows that 45 of the 57 failing
test paths already exist on the target branch, while 12 were introduced by the
rebased VMI/vecscope changes. This proves that the failure set is mixed, not
that all 57 are new regressions. The 12 newly introduced paths are:

```text
pto/materialize_tile_handles_fusion_region_subview.pto
vmi_new/vmi_to_vpto_vector_scalar_native.pto
vpto/bisheng_vf_object_argv.pto
vpto/ptodsl_vmi_local_elementwise_candidates.pto
vpto/ptodsl_vmi_sinkhorn_grouped_candidates.pto
vpto/ptodsl_vmi_sqrt_ops.pto
vpto/vecscope_membar_min_rope_kv_cache.pto
vpto/vecscope_membar_multi_kind.pto
vpto/vecscope_membar_same_iteration_raw.pto
vpto/vecscope_membar_same_iteration_waw.pto
vpto/vlds_vsts_addptr_fallback.pto
vpto/vmi_fusion_region_loop_elide.pto
```

The 45 existing paths still require a same-toolchain baseline run to decide
whether they are inherited incompatibilities or regressions caused by changed
semantics. The current evidence therefore supports “mixed and unresolved,”
not “all caused by rebase incompatibility” or “all newly introduced.”

The complete output is available locally at `/tmp/lit-final-v3.log` for this
run. The branch is buildable and PythonDSL-clean, but the 57 lit failures are
an explicit remaining rebase risk and must not be represented as a green
integration.

## Daemon and compliance audit

- No daemon owned by the current user remained after the tests.
- `git diff --check` passed.
- The changed-code prefilter reported 1009 errors and 12 warnings across the
  full rebased delta relative to `hw-native-sys/main`. Most findings are
  pre-existing style debt in replayed source commits (especially brace and
  line-length findings); they were not suppressed or misreported as clean.

## Conclusion

The branch is a genuine per-commit rebase and preserves the target branch's
split compiler and ObjectEmission architecture. It compiles fully and passes
all PythonDSL tests. It is not yet suitable to merge without addressing or
accepting the 57 lit regressions and deciding how to port the direct merged
device ELF mode onto the upstream VFSIMT-aware API.
