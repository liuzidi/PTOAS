#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    Tile,
    TileSpec,
    f16,
    f32,
    for_,
    i32,
    make_mask,
    scalar_const,
    tile_template,
    vadd,
    vecscope,
    vlds,
    vsts,
)
from ptodsl.tilelib.registry import TileTemplateRegistry
from ptodsl.vmi_tilelib import (
    VMI_TILELIB_REGISTRY,
    vmi_tadd_block64,
    vmi_texp_block64,
)
from ptodsl.vmi_tilelib_helper import instantiate_candidate


TILE_SHAPE = (32, 256)


@tile_template(op="tadd", name="legacy_vpto_tadd")
def legacy_vpto_tadd(src0: Tile, src1: Tile, dst: Tile):
    with vecscope():
        rows, cols = dst.valid_shape
        with for_(0, rows, step=1) as row:
            remained = scalar_const(256, i32)
            with for_(0, cols, step=64) as col:
                mask, _ = make_mask(dst.element_type, remained)
                lhs = vlds(src0[row, col:])
                rhs = vlds(src1[row, col:])
                vsts(vadd(lhs, rhs, mask), dst[row, col:], mask)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def specialize_tadd(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_tadd_block64.specialize(src0=spec, src1=spec, dst=spec)


def specialize_texp(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_texp_block64.specialize(src=spec, dst=spec)


def check_canonical_block_map() -> None:
    block_map = CanonicalBlockMap(TILE_SHAPE, logical_lanes=64)
    expect(block_map.blocks_per_row == 4, "[32,256]xf32 should contain four blocks per row")
    expect(block_map.logical_block_count == 128, "[32,256]xf32 should contain 128 blocks")

    coordinate = block_map.coordinate(67)
    expect(coordinate.row == 16, "logical block 67 should map to row 16")
    expect(coordinate.block_in_row == 3, "logical block 67 should be block 3 in its row")
    expect(coordinate.col_start == 192, "logical block 67 should start at column 192")
    expect(coordinate.linear_offset == 4288, "logical block 67 should start at linear offset 4288")
    expect(coordinate.active_lanes == 64, "the initial f32 contract should activate 64 lanes")

    expect_raises(
        lambda: CanonicalBlockMap((32, 250), logical_lanes=64),
        ValueError,
        "full logical blocks",
    )


def check_candidate_ir() -> tuple[str, str]:
    tadd = specialize_tadd()
    tadd.verify()
    tadd_text = tadd.mlir_text()
    expect("pto.vecscope" not in tadd_text, "VMI templates must remain scope-free")
    expect(tadd_text.count("scf.for") == 1, "tadd candidate should contain one flat loop")
    expect("arith.constant 128 : index" in tadd_text, "tadd should iterate over 128 blocks")
    expect(tadd_text.count("pto.vmi.vload") == 2, "tadd should issue two VMI loads")
    expect(tadd_text.count("pto.vmi.vadd") == 1, "tadd should issue one VMI add")
    expect(tadd_text.count("pto.vmi.vstore") == 1, "tadd should issue one VMI store")
    expect(tadd_text.count("pto.tile_buf_addr") == 3, "tadd should materialize three tile pointers")
    expect(
        tadd_text.rfind("pto.tile_buf_addr") < tadd_text.index("scf.for"),
        "tadd tile pointers should be materialized before the logical-block loop",
    )
    expect("!pto.vmi.vreg<64xf32>" in tadd_text, "tadd should use 64 logical f32 lanes")
    expect("pto.vlds" not in tadd_text, "VMI candidate should not emit physical vlds")
    expect("pto.vsts" not in tadd_text, "VMI candidate should not emit physical vsts")

    texp = specialize_texp()
    texp.verify()
    texp_text = texp.mlir_text()
    expect("pto.vecscope" not in texp_text, "VMI templates must remain scope-free")
    expect(texp_text.count("scf.for") == 1, "texp candidate should contain one flat loop")
    expect(texp_text.count("pto.vmi.vload") == 1, "texp should issue one VMI load")
    expect(texp_text.count("pto.vmi.vexp") == 1, "texp should issue one VMI exp")
    expect(texp_text.count("pto.vmi.vstore") == 1, "texp should issue one VMI store")

    expect_raises(
        lambda: specialize_tadd(dtype=f16).mlir_text(),
        ValueError,
        "require an f32 logical block",
    )
    expect_raises(
        lambda: specialize_texp(shape=(32, 250)).mlir_text(),
        ValueError,
        "full logical blocks",
    )
    return tadd_text, texp_text


def check_provider_helper() -> None:
    registered_tadd = VMI_TILELIB_REGISTRY.lookup("tadd", "a5")
    expect(
        len(registered_tadd) == 1,
        "tadd must have one registered canonical VMI template",
    )
    expect(
        registered_tadd[0] is vmi_tadd_block64,
        "the registered tadd template must be the exported canonical implementation",
    )

    raw_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 256],
        "valid_shape": [32, 256],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    text = artifact.mlir_text()
    expect("pto.vmi.vadd" in text, "provider helper should instantiate the tadd VMI candidate")
    expect(text.count("scf.for") == 1, "provider helper should preserve one logical-block loop")

    exp_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.texp",
        operand_specs=[raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    )
    expect(
        "pto.vmi.vexp" in exp_artifact.mlir_text(),
        "provider helper should accept the default texp precision contract",
    )

    tmul_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tmul",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    expect("pto.vmi.vmul" in tmul_artifact.mlir_text(), "tmul should lower to VMI")

    compact_tile_spec = {
        **raw_tile_spec,
        "shape": [1, 32],
        "valid_shape": [1, 32],
    }
    compact_add = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[compact_tile_spec, compact_tile_spec, compact_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("!pto.vmi.vreg<32xf32>" in compact_add, "compact state should use 32 lanes")

    scalar_spec = {"kind": "scalar", "dtype": "f32"}
    tmuls = instantiate_candidate(
        target="a5",
        op_name="pto.tmuls",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("%arg1: f32" in tmuls, "tmuls should preserve its runtime scalar parameter")
    expect("pto.vmi.vmuls" in tmuls, "tmuls should lower to VMI scalar multiply")

    scalar_expectations = {
        "tadds": "pto.vmi.vadds",
        "tmaxs": "pto.vmi.vmaxs",
        "tmins": "pto.vmi.vmins",
    }
    for op_name, expected_op in scalar_expectations.items():
        text = instantiate_candidate(
            target="a5",
            op_name=f"pto.{op_name}",
            operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")

    tdivs = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vbrc" in tdivs, "tdivs should broadcast its scalar operand")
    expect("pto.vmi.vdiv" in tdivs, "tdivs should lower to VMI vector divide")
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tdivs",
            operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high_precision"},
        ),
        ValueError,
        "does not support context attrs",
    )

    reduced_tile_spec = {
        **raw_tile_spec,
        "shape": [32, 1],
        "valid_shape": [32, 1],
        "config": {**raw_tile_spec["config"], "b_layout": "col_major"},
    }
    rowmax = instantiate_candidate(
        target="a5",
        op_name="pto.trowmax",
        operand_specs=[raw_tile_spec, raw_tile_spec, reduced_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(rowmax.count("scf.for") == 1, "rowmax should emit only one runtime loop")
    expect(rowmax.count("pto.vmi.vcmax") == 4, "rowmax should statically unroll four VL blocks")
    expect("!pto.vmi.vreg<1xf32>" in rowmax, "rowmax should produce 1-lane reductions")

    row_expand = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpandsub",
        operand_specs=[raw_tile_spec, reduced_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vbrc" in row_expand, "row expand should broadcast one value per row")

    f16_tile_spec = {**raw_tile_spec, "dtype": "f16"}
    tcvt = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in tcvt, "tcvt should lower to VMI conversion")

    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tdiv",
            operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "no PTODSL VMI candidate",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.texp",
            operand_specs=[raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high"},
        ),
        ValueError,
        "does not support context attrs",
    )

    duplicate_module = ModuleType("ptodsl_test_duplicate_vmi_candidates")
    duplicate_module.VMI_TILELIB_REGISTRY = TileTemplateRegistry()

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_a", ir_level="vmi")
    def duplicate_tadd_a(src0: Tile, src1: Tile, dst: Tile):
        pass

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_b", ir_level="vmi")
    def duplicate_tadd_b(src0: Tile, src1: Tile, dst: Tile):
        pass

    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_a)
    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_b)
    sys.modules[duplicate_module.__name__] = duplicate_module
    try:
        expect_raises(
            lambda: instantiate_candidate(
                target="a5",
                op_name="pto.tadd",
                operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
                provider_module=duplicate_module.__name__,
                context_attrs={},
            ),
            LookupError,
            "requires exactly one canonical candidate",
            "found 2",
        )
    finally:
        del sys.modules[duplicate_module.__name__]


def check_legacy_vpto_compatibility() -> None:
    spec = TileSpec(TILE_SHAPE, f32)
    artifact = legacy_vpto_tadd.specialize(src0=spec, src1=spec, dst=spec)
    artifact.verify()
    text = artifact.mlir_text()
    expect(text.count("scf.for") == 2, "legacy VPTO template should retain its two loops")
    expect("pto.vlds" in text, "legacy VPTO template should still emit vlds")
    expect("pto.vadd" in text, "legacy VPTO template should still emit vadd")
    expect("pto.vsts" in text, "legacy VPTO template should still emit vsts")


def check_vmi_to_vpto_lowering(name: str, mlir_text: str, expected_op: str) -> None:
    ptoas = shutil.which("ptoas")
    expect(ptoas is not None, "ptoas must be available for VMI-to-VPTO regression coverage")
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"{name}.pto"
        input_path.write_text(mlir_text, encoding="utf-8")
        completed = subprocess.run(
            [
                ptoas,
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--enable-vmi",
                "--emit-vpto",
                str(input_path),
                "-o",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    expect(
        completed.returncode == 0,
        f"VMI-to-VPTO lowering failed for {name}:\n{completed.stderr}",
    )
    expect("pto.vmi." not in completed.stdout, f"{name} should contain no VMI ops after lowering")
    expect(expected_op in completed.stdout, f"{name} should lower to {expected_op}")
    expect(completed.stdout.count("scf.for") == 1, f"{name} should preserve one flat loop")


def main() -> None:
    check_canonical_block_map()
    check_legacy_vpto_compatibility()
    check_provider_helper()
    tadd_text, texp_text = check_candidate_ir()
    check_vmi_to_vpto_lowering("vmi_tadd_block64", tadd_text, "pto.vadd")
    check_vmi_to_vpto_lowering("vmi_texp_block64", texp_text, "pto.vexp")
    print("ptodsl_vmi_tile_template: PASS")


if __name__ == "__main__":
    main()
