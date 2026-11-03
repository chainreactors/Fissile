"""Row-wise FP8 quantization; extracted without arithmetic changes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_rows(X, Q, S, M, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + col, (row < M) & (col < K), 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / 448.0, 1.0e-12)
    rounded = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    tl.store(Q + row * K + col, rounded, col < K)
    tl.store(S + row, scale)


def quantize_fp8(x):
    if (
        x.dtype != torch.bfloat16
        or not x.is_cuda
        or x.ndim != 2
        or not x.is_contiguous()
    ):
        raise ValueError("FP8 expects contiguous 2D CUDA BF16 input")
    m, k = x.shape
    pad = triton.cdiv(m, 16) * 16
    q = torch.empty((pad, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((pad, 1), device=x.device, dtype=torch.float32)
    _fp8_rows[(pad,)](
        x, q, scale, m, K=k, BLOCK=triton.next_power_of_2(k), enable_fp_fusion=False
    )
    return q, scale
