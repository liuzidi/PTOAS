# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trecip — default precision only."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.trecip",
    target="a5",
    name="template_trecip",
    dtypes=[
        ("f16", "f16"),
        ("f32", "f32"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "reciprocal"),
)
def template_trecip(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    one_scalar = pto.f16(1.0) if str(dtype) == "f16" else pto.f32(1.0)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[row, col:])
            one = pto.vbr(one_scalar)
            result = pto.vdiv(one, value, mask)
            pto.vsts(result, dst[row, col:], mask)


from ._vmi_common import (  # noqa: E402
    _context_attr,
    canonical_vmi_template,
    emit_recip_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="trecip",
    name="vmi_trecip",
    dtypes=(("f16", "f16"), ("f32", "f32")),
    context_constraints={"precisionType": ("default", "high_precision")},
)
def vmi_trecip(src: pto.Tile, dst: pto.Tile):
    emit_recip_vmi(
        src,
        dst,
        high_precision=_context_attr(src, "precisionType", "default")
        == "high_precision",
    )
