# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsubs."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import same_dtype_signatures
from ._elementwise import _common_constraints


@tilelib.tile_template(
    op="pto.tsubs",
    target="a5",
    name="template_tsubs",
    dtypes=same_dtype_signatures(3),
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_common_constraints("src", "dst"),
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "scalar"),
)
def template_tsubs(src: pto.Tile, scalar, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[row, col:])
            scalar_value = pto.vbr(scalar)
            result = pto.vsub(value, scalar_value, mask)
            pto.vsts(result, dst[row, col:], mask)


from ._vmi_common import (  # noqa: E402
    _negate_scalar,
    _vadds as _vmi_vadds,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f32,
)


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs",
    dtypes=(("f32", "f32", "f32"),),
)
def vmi_tsubs(src: pto.Tile, scalar: f32, dst: pto.Tile):
    negated = _negate_scalar(scalar, f32)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
    )
