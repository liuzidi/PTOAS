import sys
sys.path.insert(0, "/home/liuzidi/pypto-lib")
import torch
import pypto.language as pl
from golden import run_jit, TensorSpec

T, D = 8, 1024  # expert_shared: [SH_ROW_PAD=8, ACT_INTER_TILE=1024]

@pl.jit
def rowmax_kernel(x: pl.Tensor[[T, D], pl.FP32], y: pl.Tensor[[1, T], pl.FP32]):
    for blk in pl.spmd(1, name_hint="rowmax"):
        amax = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // 1024, stage=1):
            chunk = x[0:T, 0:1024]
            chunk_amax = pl.reshape(pl.row_max(pl.abs(chunk)), [1, T])
            amax = pl.maximum(amax, chunk_amax)
        y[0:1, 0:T] = amax[0:1, 0:T]

def golden_fn(tensors):
    tensors["y"][:] = tensors["x"].abs().max(-1, keepdim=True).values

result = run_jit(
    fn=rowmax_kernel,
    specs=[
        TensorSpec("x", [T, D], torch.float32, init_value=torch.randn),
        TensorSpec("y", [1, T], torch.float32, is_output=True),
    ],
    golden_fn=golden_fn,
    compile_only=True,
    runtime_cfg=dict(platform="a5", device_id=0),
)
print("COMPILE-PASS" if result.passed else "COMPILE-FAIL")
