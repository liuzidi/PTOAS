# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trsqrt — default precision only."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


template_trsqrt = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt",
    vector_op=pto.vrsqrt,
    dtypes=_DTYPES,
)


template_trsqrt_1d = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt_1d",
    vector_op=pto.vrsqrt,
    dtypes=_DTYPES,
    traversal="1d",
)


# PyPTO's high-precision rsqrt emits the 3-operand form (src, tmp) -> dst
# where tmp is the iterative-refine workspace consumed by the pto-isa path.
# The default-precision PTODSL body does not read the workspace.
_TMP_DTYPES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
]

template_trsqrt_tmp = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt_tmp",
    vector_op=pto.vrsqrt,
    dtypes=_TMP_DTYPES,
    has_tmp=True,
    candidate_id=2,
)

template_trsqrt_tmp_1d = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt_tmp_1d",
    vector_op=pto.vrsqrt,
    dtypes=_TMP_DTYPES,
    has_tmp=True,
    traversal="1d",
    candidate_id=3,
)


from ._vmi_common import (  # noqa: E402
    _context_attr,
    canonical_vmi_template,
    emit_rsqrt_vmi,
    narrow_full_row_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt",
    dtypes=(("f16", "f16"), ("f32", "f32")),
    context_constraints={"precisionType": ("default",)},
)
def vmi_trsqrt(src: pto.Tile, dst: pto.Tile):
    emit_rsqrt_vmi(src, dst, high_precision=False)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt_with_tmp",
    dtypes=(
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ),
    context_constraints={"precisionType": ("default", "high_precision")},
)
def vmi_trsqrt_with_tmp(src: pto.Tile, dst: pto.Tile, tmp: pto.Tile):
    _ = tmp
    emit_rsqrt_vmi(
        src,
        dst,
        high_precision=_context_attr(src, "precisionType", "default")
        == "high_precision",
    )


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt_with_tmp_narrow",
    dtypes=(
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ),
    context_constraints={"precisionType": ("default", "high_precision")},
    constraints=(narrow_full_row_vmi_constraint,),
    requires_full_physical_row=False,
)
def vmi_trsqrt_with_tmp_narrow(src: pto.Tile, dst: pto.Tile, tmp: pto.Tile):
    # PyPTO high-precision rsqrt materializes a 3-operand form on compact
    # rows (e.g. 1x8 f32 row-reduce tails) whose byte width is below the
    # 128-byte minimum of the standard elementwise candidates. Memory
    # planning pads every UB slot to 256 bytes, so a full-row VMI access
    # stays inside the slot. The tmp buffer only feeds the iterative-refine
    # pto-isa path; the VMI emit computes rsqrt from src alone.
    _ = tmp
    emit_rsqrt_vmi(
        src,
        dst,
        high_precision=_context_attr(src, "precisionType", "default")
        == "high_precision",
    )