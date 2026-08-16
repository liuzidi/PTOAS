# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmaxs."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_scalar_binary


template_tmaxs = register_scalar_binary(
    op="pto.tmaxs",
    name="template_tmaxs",
    vector_op=pto.vmaxs,
    dtypes=same_dtype_signatures(3),
)


from ._vmi_common import (  # noqa: E402
    _vmaxs as _vmi_vmaxs,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f32,
)


@canonical_vmi_template(
    target="a5",
    op="tmaxs",
    name="vmi_tmaxs",
    dtypes=(("f32", "f32", "f32"),),
)
def vmi_tmaxs(src: pto.Tile, scalar: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmaxs(values[0], scalar, mask),
    )
