"""Experimental Qwen3.5 pointwise fusion with the original rounding boundaries.

No parameter, quantization, candidate, vocabulary, or cache changes. The
torch-reduce mode retains Torch's FP32 mean reduction; the triton-reduce mode
also fuses that reduction and must be validated separately.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _squares(X, S, Count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < Count, 0).to(tl.float32)
    tl.store(S + offset, x * x, offset < Count)


@triton.jit
def _silu(x):
    return tl.div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def _rms(
    X,
    W,
    G,
    V,
    Y,
    M,
    N: tl.constexpr,
    EPS: tl.constexpr,
    GATED: tl.constexpr,
    WEIGHT_BF16: tl.constexpr,
    REDUCE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    offset = row[:, None] * N + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < N)
    x = tl.load(X + offset, mask, 0).to(tl.float32)
    if REDUCE:
        variance = tl.sum(x * x, 1) / N
    else:
        variance = tl.load(V + row, row < M, 0)
    inv = libdevice.rsqrt(variance + EPS)
    weight = tl.load(W + col, col < N, 0).to(tl.float32)
    normalized = x * inv[:, None]
    if GATED:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
        weighted = normalized * weight[None, :]
        if WEIGHT_BF16:
            weighted = weighted.to(tl.bfloat16).to(tl.float32)
        gate = tl.load(G + offset, mask, 0).to(tl.float32)
        result = weighted * _silu(gate)
    else:
        result = normalized * (1.0 + weight[None, :])
    tl.store(Y + offset, result, mask)


@triton.jit
def _silu_mul(G, U, Y, Count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offset, offset < Count, 0).to(tl.float32)
    up = tl.load(U + offset, offset < Count, 0).to(tl.float32)
    activated = _silu(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offset, activated * up, offset < Count)


def rms_norm(x, weight, eps, gate=None, *, reduce="torch-reduce"):
    if reduce == "native-order":
        from fissile.kernels.native_rms import rms_norm as native_rms

        return native_rms(x, weight, eps, gate)
    if reduce not in ("torch-reduce", "triton-reduce"):
        raise ValueError("Unknown reduction")
    if (
        x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.numel()
        or x.ndim < 2
        or weight.ndim != 1
        or weight.numel() != x.shape[-1]
        or weight.device != x.device
        or weight.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Expected CUDA BF16 activations and BF16/FP32 RMSNorm weights")
    if gate is not None and (
        gate.shape != x.shape
        or gate.device != x.device
        or gate.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Invalid gate")
    matrix = x.contiguous().view(-1, x.shape[-1])
    m, n = matrix.shape
    output = torch.empty_like(matrix)
    variance = output
    if reduce == "torch-reduce":
        squares = torch.empty_like(matrix, dtype=torch.float32)
        _squares[(triton.cdiv(matrix.numel(), 1024),)](
            matrix, squares, matrix.numel(), BLOCK=1024
        )
        variance = squares.mean(-1)
    gates = gate.contiguous().view_as(matrix) if gate is not None else matrix
    rows = 4 if n <= 256 else 1
    _rms[(triton.cdiv(m, rows),)](
        matrix,
        weight,
        gates,
        variance,
        output,
        m,
        N=n,
        EPS=eps,
        GATED=gate is not None,
        WEIGHT_BF16=weight.dtype == torch.bfloat16,
        REDUCE=reduce == "triton-reduce",
        ROWS=rows,
        BLOCK=triton.next_power_of_2(n),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output.view_as(x)


def silu_mul(gate, up):
    if (
        gate.dtype != torch.bfloat16
        or up.dtype != torch.bfloat16
        or gate.shape != up.shape
        or not gate.is_cuda
        or gate.device != up.device
    ):
        raise ValueError("Expected matching CUDA BF16 tensors")
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    _silu_mul[(triton.cdiv(gate.numel(), 1024),)](
        gate, up, out, gate.numel(), BLOCK=1024, enable_fp_fusion=False
    )
    return out
