# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tlog."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import _common_constraints, register_unary


def _is_default_precision(precisionType="default", **_):
    return precisionType != "high_precision"


def _is_high_precision(precisionType="default", **_):
    return precisionType == "high_precision"


template_tlog = register_unary(
    op="pto.tlog",
    name="template_tlog",
    vector_op=pto.vln,
    dtypes=[
        ("f16", "f16"),
        ("f32", "f32"),
    ],
    constraints=[_is_default_precision],
)


@tilelib.tile_template(
    op="pto.tlog",
    target="a5",
    name="template_tlog_high_precision",
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
def template_tlog_high_precision(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    if str(dtype) == "f16":
        subnormal_threshold = pto.f16("0x03FF")
        mul_factor = pto.f16("0x6400")
        compensation = pto.f16(-6.931471805599453094172)
    else:
        subnormal_threshold = pto.f32("0x007FFFFF")
        mul_factor = pto.f32("0x4B000000")
        compensation = pto.f32(-15.9423851528787421)

    lanes = pto.elements_per_vreg(dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            vinput = pto.vlds(src[row, col:])
            cmp_mask = pto.vcmps(vinput, subnormal_threshold, mask, pto.CmpMode.LT)
            scaled = pto.vmuls(vinput, mul_factor, mask)
            selected_input = pto.vsel(scaled, vinput, cmp_mask)
            log_result = pto.vln(selected_input, mask)
            compensated = pto.vadds(log_result, compensation, mask)
            result = pto.vsel(compensated, log_result, cmp_mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)
