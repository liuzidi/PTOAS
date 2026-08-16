"""Build-tree ptoas CLI entry point."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import sys
import os
import ctypes

def launch(argv: Sequence[str], *, wrapper: Path) -> int:
    build_root = Path("/data/liuzidi/PTOAS/build")
    source_root = Path("/data/liuzidi/PTOAS")
    python_root = build_root / "python"
    mlir_core = Path("/data/liuzidi/llvm-workspace/llvm-project/build-shared/tools/mlir/python_packages/mlir_core")
    # ptoas_pkg = build/python/ptoas (only the ptoas package, NOT mlir)
    
    env = os.environ
    env["PTOAS_HOME"] = str(build_root)
    env["PTOAS_BIN"] = str(wrapper)
    env["PTOAS_TILEOPS_DIR"] = str(source_root / "lib/TileOps")
    env["PTOAS_PYTHON_EXE"] = sys.executable
    env["PTO_PYTHON_ROOT"] = str(mlir_core)
    
    # Order: mlir_core (provides 'mlir') > build/python/ptoas (provides 'ptoas') > source ptodsl
    # NOTE: build/python has 'mlir' dir that shadows mlir_core - avoid it!
    # Instead use mlir_core for 'mlir' and only ptoas package from build
    ptoas_only = str(python_root)  # has ptoas/ and ptoas_wheel_bootstrap.py
    paths = [str(mlir_core), str(source_root / "ptodsl")]
    if 'PYTHONPATH' in env:
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = ':'.join(paths)
    
    # Also add ptoas package to sys.path for the main process
    # The ptoas.mlir redirect files handle the rest
    sys.path.insert(0, str(python_root))
    
    full_argv = [str(wrapper)]
    if not any('--tilelang-path' in a for a in argv):
        full_argv.extend(["--tilelang-path", str(source_root / "lib/TileOps")])
    if not any('--tilelang-pkg-path' in a for a in argv):
        full_argv.extend(["--tilelang-pkg-path", str(python_root)])
    if not any('--ptodsl-pkg-path' in a for a in argv):
        full_argv.extend(["--ptodsl-pkg-path", str(source_root / "ptodsl")])
    full_argv.extend(argv)
    
    so_path = python_root / "ptoas" / "mlir" / "_mlir_libs" / "libPTOASCompiler.so"
    lib = ctypes.CDLL(str(so_path), mode=0)
    entry = lib.ptoas_entrypoint
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    entry.restype = ctypes.c_int
    argv_bytes = [os.fsencode(a) for a in full_argv]
    c_argv = (ctypes.c_char_p * len(argv_bytes))(*argv_bytes)
    return int(entry(len(argv_bytes), c_argv))
