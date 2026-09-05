import sys
sys.path.insert(0, "/home/liuzidi/pypto-lib")
import torch
import pypto.language as pl
from golden import run_jit, TensorSpec

T, D = 8, 64

@pl.jit
def gather_cmp(x: pl.Tensor[[T, D], pl.FP32], idx: pl.Tensor[[T, 8], pl.INT32],
               y: pl.Tensor[[T, 8], pl.FP32]):
    for blk in pl.spmd(1, name_hint="gather_cmp"):
        y[0:T, 0:8] = pl.gather(x[0:T, 0:D], dim=-1, index=idx[0:T, 0:8])[0:T, 0:8]

def golden_fn(tensors):
    tensors["y"][:] = torch.gather(tensors["x"], -1, tensors["idx"])

result = run_jit(
    fn=gather_cmp,
    specs=[
        TensorSpec("x", [T, D], torch.float32, init_value=torch.randn),
        TensorSpec("idx", [T, 8], torch.int32, init_value=lambda: torch.randint(0, D, [T, 8])),
        TensorSpec("y", [T, 8], torch.float32, is_output=True),
    ],
    golden_fn=golden_fn,
    compile_only=True,
    runtime_cfg=dict(platform="a5", device_id=0),
)
print("COMPILE-PASS" if result.passed else "COMPILE-FAIL")
