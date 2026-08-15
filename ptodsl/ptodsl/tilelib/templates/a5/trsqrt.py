# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trsqrt."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import _common_constraints, register_unary
from .div_hp import _div_ieee754_f16_impl, _div_ieee754_f32_impl
from .tsqrt import sqrt_high_precision


def _is_default_precision(precisionType="default", **_):
    return precisionType != "high_precision"


def _has_tmp(operand_kinds=(), **_):
    return operand_kinds == ("tile", "tile", "tile")


def _is_high_precision_with_tmp(precisionType="default", **context):
    return precisionType == "high_precision" and _has_tmp(**context)


def _emit_trsqrt_body(src, dst, *, high_precision=False):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    src_cols = src.shape[1]
    dst_cols = dst.shape[1]
    lanes = pto.elements_per_vreg(dtype)
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            src_addr = pto.addptr(src_ptr, row * src_cols + col)
            value = pto.vlds(src_addr, 0)
            if high_precision:
                root = sqrt_high_precision(value, mask, dtype)
                one = pto.vbr(pto.f32(1.0) if str(dtype) == "f32" else pto.f16(1.0))
                if str(dtype) == "f32":
                    result = _div_ieee754_f32_impl(one, root, mask)
                else:
                    result = _div_ieee754_f16_impl(one, root, mask)
            else:
                result = pto.vrsqrt(value, mask)
            dst_addr = pto.addptr(dst_ptr, row * dst_cols + col)
            pto.vsts(result, dst_addr, 0, mask)
            col_loop.update(remained=remained)


template_trsqrt = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt",
    vector_op=pto.vrsqrt,
    dtypes=[
        ("f16", "f16"),
        ("f32", "f32"),
    ],
    constraints=[_is_default_precision],
)


@tilelib.tile_template(
    op="pto.trsqrt",
    target="a5",
    name="template_trsqrt_with_tmp",
    dtypes=[
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst") + [
        _has_tmp,
        _is_default_precision,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "unary"),
)
def template_trsqrt_with_tmp(src: pto.Tile, dst: pto.Tile, tmp: pto.Tile):
    _ = tmp
    _emit_trsqrt_body(src, dst)


@tilelib.tile_template(
    op="pto.trsqrt",
    target="a5",
    name="template_trsqrt_high_precision",
    dtypes=[
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst") + [
        _is_high_precision_with_tmp,
    ],
    id=2,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "unary"),
)
def template_trsqrt_high_precision(src: pto.Tile, dst: pto.Tile, tmp: pto.Tile):
    _ = tmp
    _emit_trsqrt_body(src, dst, high_precision=True)


from ._vmi_common import (  # noqa: E402
    _context_attr,
    canonical_vmi_template,
    emit_rsqrt_vmi,
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
    dtypes=(("f16", "f16", "f16"), ("f32", "f32", "f32")),
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
