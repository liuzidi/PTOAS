import sys
sys.path.insert(0, "/home/liuzidi/pypto-lib")
import torch
import pypto.language as pl
from golden import run_jit, TensorSpec

T, D = 8, 64

@pl.jit
def rsqrt_hp(x: pl.Tensor[[T, D], pl.FP32], y: pl.Tensor[[T, D], pl.FP32]):
    # 对齐 qkv_proj_rope.py:304 的模式：row_sum 求方差 -> high_precision rsqrt
    for blk in pl.spmd(1, name_hint="rsqrt_hp"):
        sq = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(D // 32, stage=2):
            chunk = x[0:T, kb*32:(kb+1)*32]
            sq = pl.add(sq, pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, T]))
        inv = pl.rsqrt(pl.add(pl.mul(sq, 1.0 / D), 1e-6), high_precision=True)
        y[0:T, 0:D] = pl.mul(x[0:T, 0:D], pl.reshape(inv, [T, 1]))[0:T, 0:D]

def golden_fn(tensors):
    xx = tensors["x"]
    tensors["y"][:] = xx / torch.sqrt(xx.square().mean(-1, keepdim=True) + 1e-6)

result = run_jit(
    fn=rsqrt_hp,
    specs=[
        TensorSpec("x", [T, D], torch.float32, init_value=torch.randn),
        TensorSpec("y", [T, D], torch.float32, is_output=True),
    ],
    golden_fn=golden_fn,
    compile_only=True,
    runtime_cfg=dict(platform="a5", device_id=0),
)
print("COMPILE-PASS" if result.passed else "COMPILE-FAIL")
