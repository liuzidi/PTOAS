# Repro: VPTO vecscope mask escape on rmsnorm bf16 [8,4096]

## Symptom

`ptoas` fails with:

```
loc("-":12:30): error: 'pto.plt_b32' op cannot infer resultless pto.vecscope
because VPTO vector-scope data cannot have external users;
escaping value type is '!pto.mask<b32>'
Error: VPTO emission pipeline failed.
```

The error is raised during VPTO emission (after vecscope inference), regardless
of the VMI/fusion route — it reproduces with the plain non-VMI level3 path as
well (see "Repro" below).

## Repro

Environment (host `liuzidi`, aarch64):
- CANN: `~/Ascend/cann-9.0.1/cann-9.0.1` (bisheng 15.0.5, target aarch64)
- ptoas: this repo, commit `95801edd`, build `build/tools/ptoas/ptoas` (version 0.59)
- env: `source $ASCEND_HOME_PATH/set_env.sh; PYTHONPATH=$PWD/ptodsl`

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --enable-op-fusion=false --disable-bisheng-vf-fusion \
  test/vpto/cases/repro/rmsnorm-bf16-8x4096-vecscope-escape/kernel.pto \
  -o /tmp/rmsnorm.fatobj.o
```

Plain level3 (no fusion flags) fails identically:

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  test/vpto/cases/repro/rmsnorm-bf16-8x4096-vecscope-escape/kernel.pto \
  -o /tmp/rmsnorm.fatobj.o
```

## Source file

`kernel.pto` — RMSNorm on bf16 `[8,4096]`, generated from a pypto higher-level
DSL (manual level3 `addr=` UB allocation, the `pto.alloc_tile addr=%c..._i64`
form). Input source is a two-module `.pto` with `make_tensor_view` +
`partition_view` + `tload/tcvt/tmul/trowsum/trowexpanddiv/tcolexpandmul/tstore`,
two `scf.for` loops (reduce phase + apply phase) over 32 double-steps.

## Notes

- The `loc("-":12:30)` is an **intermediate-IR** location, not a source `.pto`
  line — line 12 of the source is `%c4096_index = arith.constant 4096 : index`,
  which has no `pto.plt_b32`. The `:12:30` points into the IR after vecscope
  inference.
- A smaller sibling case `test/vpto/cases/*/fa-softmax-dn-init-rowplusone`
  compiles fine on the same ptoas; the difference is this kernel uses the
  manual-`addr` level3 form with a `bf16` element type and a mask that appears
  to escape the inferred vecscope boundary.
- The dsv4-rmsnorm-sixway experiment ran this same `.pto` successfully using a
  **different (private) ptoas build** at
  `/data/lishengtao/private-ptoas-lab/...` (SHA256 of that ptoas binary:
  `6606e2dadfc35ede3964d790559e4ccb3eca9eccaf0b4e230c9a556d7ff6e78e`), so the
  regression appears to be specific to this build/commit, not the `.pto` itself.
