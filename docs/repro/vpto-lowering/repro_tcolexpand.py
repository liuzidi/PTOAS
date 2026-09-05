import sys
sys.path.insert(0, "/home/liuzidi/pypto-lib")
import torch
import pypto.language as pl
from golden import run_jit, TensorSpec

T, D = 8, 64

@pl.jit
def colexpand_kernel(x: pl.Tensor[[T, D], pl.FP32], base: pl.Tensor[[D], pl.FP32],
                     y: pl.Tensor[[T, D], pl.FP32]):
    for blk in pl.spmd(1, name_hint="colexpand"):
        b = pl.reshape(base[0:D], [1, D])
        y[0:T, 0:D] = pl.add(x[0:T, 0:D], pl.col_expand(x[0:T, 0:D], b))[0:T, 0:D]

def golden_fn(tensors):
    tensors["y"][:] = tensors["x"] + tensors["x"] + tensors["base"]

result = run_jit(
    fn=colexpand_kernel,
    specs=[
        TensorSpec("x", [T, D], torch.float32, init_value=torch.randn),
        TensorSpec("base", [D], torch.float32, init_value=torch.randn),
        TensorSpec("y", [T, D], torch.float32, is_output=True),
    ],
    golden_fn=golden_fn,
    compile_only=True,
    runtime_cfg=dict(platform="a5", device_id=0),
)
print("COMPILE-PASS" if result.passed else "COMPILE-FAIL")
