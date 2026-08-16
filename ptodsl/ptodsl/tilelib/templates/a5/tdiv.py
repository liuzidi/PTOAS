# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tdiv."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from .div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


@tilelib.tile_template(
    op="pto.tdiv",
    target="a5",
    name="template_tdiv",
    dtypes=[("f16", "f16", "f16"), ("f32", "f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    layouts=["row_major"],
    memory_spaces=["ub"],
    priority=0,
    id=0,
    loop_depth=2,
    is_post_update=False,
)
def template_tdiv(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    precision_type = pto.get_op_attr("precisionType", "default")

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            if precision_type == "high_precision":
                if str(dtype) == "f32":
                    divided = _div_ieee754_f32_impl(lhs, rhs, mask)
                else:
                    divided = _div_ieee754_f16_impl(lhs, rhs, mask)
            else:
                divided = pto.vdiv(lhs, rhs, mask)
            pto.vsts(divided, dst[row, col:], mask)


from ._vmi_common import (  # noqa: E402
    FLOAT_DTYPES,
    _context_attr,
    _div_high_precision,
    _vdiv as _vmi_vdiv,
    canonical_vmi_template,
    emit_elementwise_vmi,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="tdiv",
    name="vmi_tdiv",
    dtypes=(("f16", "f16", "f16"), ("f32", "f32", "f32")),
    context_constraints={"precisionType": ("default", "high_precision")},
    min_row_bytes=128,
)
def vmi_tdiv(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    if _context_attr(src0, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src0, src1),
            lambda values, mask: _div_high_precision(values[0], values[1], mask),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src0, src1),
        lambda values, mask: _vmi_vdiv(values[0], values[1], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdiv",
    name="vmi_tdiv_sinkhorn_compact",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default",)},
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tdiv_sinkhorn_compact(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src0, src1),
        lambda values, mask: _vmi_vdiv(values[0], values[1], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )
