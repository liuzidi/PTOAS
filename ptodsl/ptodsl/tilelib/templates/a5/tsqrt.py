# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsqrt."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import _common_constraints, register_unary


def _is_default_precision(precisionType="default", **_):
    return precisionType != "high_precision"


def _is_high_precision(precisionType="default", **_):
    return precisionType == "high_precision"


def sqrt_high_precision(src, mask, dtype):
    """High-precision sqrt ported from lib/TileOps/sqrt_hp.py."""
    if str(dtype) == "f16":
        subnormal_mask = pto.vcmps(src, pto.f16("0x03ff"), mask, pto.CmpMode.LT)
        scaled_src = pto.vmuls(src, pto.f16("0x6c00"), subnormal_mask)
        src_adjusted = pto.vsel(scaled_src, src, subnormal_mask)

        root = pto.vsqrt(src_adjusted, mask)
        scaled_root = pto.vmuls(root, pto.f16("0x2400"), subnormal_mask)
        return pto.vsel(scaled_root, root, subnormal_mask)

    subnormal_mask = pto.vcmps(src, pto.f32(1.0), mask, pto.CmpMode.LT)
    scaled_src = pto.vmuls(src, pto.f32(16777216.0), subnormal_mask)
    src_adjusted = pto.vsel(scaled_src, src, subnormal_mask)

    one = pto.vbr(pto.f32(1.0))
    neg_one = pto.f32(-1.0)
    half = pto.f32(0.5)

    root = pto.vsqrt(src_adjusted, mask)
    reciprocal = pto.vdiv(one, root, mask)

    neg_reciprocal = pto.vmuls(reciprocal, neg_one, mask)
    err = pto.vmul(reciprocal, src_adjusted, mask)
    one_adjusted = pto.vmula(one, err, neg_reciprocal, mask)
    half_reciprocal = pto.vmuls(reciprocal, half, mask)
    refined = pto.vmula(reciprocal, one_adjusted, half_reciprocal, mask)

    result = pto.vmul(refined, src_adjusted, mask)
    neg_result = pto.vmuls(result, neg_one, mask)
    err = pto.vmula(src_adjusted, result, neg_result, mask)
    half_refined = pto.vmuls(refined, half, mask)
    correction = pto.vmul(err, half_refined, mask)
    corrected = pto.vadd(correction, result, mask)

    scaled_corrected = pto.vmuls(corrected, pto.f32(0.000244140625), mask)
    result = pto.vsel(scaled_corrected, corrected, subnormal_mask)

    src_bits = pto.vbitcast(src_adjusted, pto.ui32)
    is_inf = pto.vcmps(src_bits, pto.ui32(0x7f800000), mask, pto.CmpMode.EQ)
    src_with_sign = pto.vor(src_bits, pto.vbr(pto.ui32(0x80000000)), mask)
    is_zero = pto.vcmps(src_with_sign, pto.ui32(0x80000000), mask, pto.CmpMode.EQ)
    special_mask = pto.por(is_zero, is_inf, mask)
    return pto.vsel(src_adjusted, result, special_mask)


template_tsqrt = register_unary(
    op="pto.tsqrt",
    name="template_tsqrt",
    vector_op=pto.vsqrt,
    dtypes=[
        ("f16", "f16"),
        ("f32", "f32"),
    ],
    constraints=[_is_default_precision],
)


@tilelib.tile_template(
    op="pto.tsqrt",
    target="a5",
    name="template_tsqrt_high_precision",
    dtypes=[
        ("f16", "f16"),
        ("f32", "f32"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst") + [_is_high_precision],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "unary"),
)
def template_tsqrt_high_precision(src: pto.Tile, dst: pto.Tile):
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
            result = sqrt_high_precision(value, mask, dtype)
            dst_addr = pto.addptr(dst_ptr, row * dst_cols + col)
            pto.vsts(result, dst_addr, 0, mask)
            col_loop.update(remained=remained)


from ._vmi_common import (  # noqa: E402
    _context_attr,
    canonical_vmi_template,
    emit_sqrt_high_precision_vmi,
    emit_sqrt_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tsqrt",
    name="vmi_tsqrt",
    dtypes=(("f16", "f16"), ("f32", "f32")),
    context_constraints={"precisionType": ("default", "high_precision")},
)
def vmi_tsqrt(src: pto.Tile, dst: pto.Tile):
    if _context_attr(src, "precisionType", "default") == "high_precision":
        emit_sqrt_high_precision_vmi(src, dst)
        return
    emit_sqrt_vmi(src, dst)
