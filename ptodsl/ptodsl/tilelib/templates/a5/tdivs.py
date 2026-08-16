# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tdivs."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import _common_constraints
from .div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl


_DTYPES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
]


def _tile_scalar_tile(operand_kinds=(), **_):
    return operand_kinds == ("tile", "scalar", "tile")


def _scalar_tile_tile(operand_kinds=(), **_):
    return operand_kinds == ("scalar", "tile", "tile")


def _div(lhs, rhs, dtype, mask, precision_type):
    if precision_type == "high_precision":
        if str(dtype) == "f32":
            return _div_ieee754_f32_impl(lhs, rhs, mask)
        return _div_ieee754_f16_impl(lhs, rhs, mask)
    return pto.vdiv(lhs, rhs, mask)


def _emit_tdivs_body(src, scalar, dst, *, scalar_lhs=False):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    src_cols = src.shape[1]
    dst_cols = dst.shape[1]
    lanes = pto.elements_per_vreg(dtype)
    precision_type = pto.get_op_attr("precisionType", "default")
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            src_addr = pto.addptr(src_ptr, row * src_cols + col)
            value = pto.vlds(src_addr, 0)
            scalar_value = pto.vbr(scalar)
            lhs, rhs = (scalar_value, value) if scalar_lhs else (value, scalar_value)
            result = _div(lhs, rhs, dtype, mask, precision_type)
            dst_addr = pto.addptr(dst_ptr, row * dst_cols + col)
            pto.vsts(result, dst_addr, 0, mask)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tdivs",
    target="a5",
    name="template_tdivs_tile_scalar",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst") + [_tile_scalar_tile],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "scalar"),
)
def template_tdivs_tile_scalar(src: pto.Tile, scalar, dst: pto.Tile):
    _emit_tdivs_body(src, scalar, dst)


@tilelib.tile_template(
    op="pto.tdivs",
    target="a5",
    name="template_tdivs_scalar_tile",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst") + [_scalar_tile_tile],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "scalar"),
)
def template_tdivs_scalar_tile(scalar, src: pto.Tile, dst: pto.Tile):
    _emit_tdivs_body(src, scalar, dst, scalar_lhs=True)


from ._vmi_common import (  # noqa: E402
    FLOAT_DTYPES,
    _context_attr,
    _divide_by_scalar,
    _divide_by_scalar_high_precision,
    _divide_scalar_by_vector,
    _divide_scalar_by_vector_high_precision,
    _operand_kinds_are,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f32,
)


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default", "high_precision")},
    constraints=(_operand_kinds_are(("tile", "scalar", "tile")),),
)
def vmi_tdivs(src: pto.Tile, scalar: f32, dst: pto.Tile):
    if _context_attr(src, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src,),
            lambda values, mask: _divide_by_scalar_high_precision(
                values[0], scalar, mask
            ),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_by_scalar(values[0], scalar, mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs_scalar_tile",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default", "high_precision")},
    constraints=(_operand_kinds_are(("scalar", "tile", "tile")),),
)
def vmi_tdivs_scalar_tile(scalar: f32, src: pto.Tile, dst: pto.Tile):
    if _context_attr(src, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src,),
            lambda values, mask: _divide_scalar_by_vector_high_precision(
                scalar, values[0], mask
            ),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_scalar_by_vector(scalar, values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )
