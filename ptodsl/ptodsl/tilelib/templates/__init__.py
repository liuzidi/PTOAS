# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compatibility loader for the canonical source or packaged TileOps."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from importlib import import_module

from .._template_package import load_template as _load_template, tileops_package


_VMI_TEMPLATE_OPS = {
    "pto.tadd",
    "pto.tadds",
    "pto.tcvt",
    "pto.tcolmax",
    "pto.tcolsum",
    "pto.tcolexpandadd",
    "pto.tcolexpanddiv",
    "pto.tcolexpandmul",
    "pto.tcolexpandsub",
    "pto.tdivs",
    "pto.tdiv",
    "pto.texp",
    "pto.trecip",
    "pto.trsqrt",
    "pto.tsqrt",
    "pto.tmax",
    "pto.tmaxs",
    "pto.tmins",
    "pto.tmov",
    "pto.tmul",
    "pto.tmuls",
    "pto.trowexpandsub",
    "pto.trowmax",
    "pto.trowsum",
    "pto.tsub",
}


@lru_cache(maxsize=None)
def load_template(op: str, target: str) -> bool:
    """Load the canonical TileOps template and any registered VMI provider."""

    loaded = _load_template(op, target)
    if target == "a5" and op in _VMI_TEMPLATE_OPS:
        import_module("lib.TileOps.a5.vmi")
        loaded = True
    return loaded


load_template_with_vmi = load_template

_tileops = tileops_package()
__path__ = [str(Path(_tileops.__file__).resolve().parent)]

__all__ = ["load_template", "load_template_with_vmi"]
