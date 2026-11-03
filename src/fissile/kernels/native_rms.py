"""Experimental fused RMSNorm retaining the installed Torch reduction order.

The FP32 contiguous mean uses four sequential accumulators per CUDA lane,
then a descending binary reduction. See the installed ATen Reduce.cuh
input_vectorized_thread_reduce_impl and block_x_reduce. This specialization
only covers the selected model's power-of-two widths (128, 256, 4096).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from fissile.kernels.pointwise import _silu


@triton.jit
def _native_rms(
    X,
    W,
    G,
    Y,
    M,
    N: tl.constexpr,
    EPS: tl.constexpr,
    GATED: tl.constexpr,
    WEIGHT_BF16: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, WIDTH)
    a = tl.full((ROWS, WIDTH), 0, tl.float32)
    b = tl.full((ROWS, WIDTH), 0, tl.float32)
    c = tl.full((ROWS, WIDTH), 0, tl.float32)
    d = tl.full((ROWS, WIDTH), 0, tl.float32)
    for step in range(N // (4 * WIDTH)):
        off = row[:, None] * N + step * 4 * WIDTH + lane[None, :] * 4
        x0 = tl.load(X + off + 0, row[:, None] < M, 0).to(tl.float32)
        x1 = tl.load(X + off + 1, row[:, None] < M, 0).to(tl.float32)
        x2 = tl.load(X + off + 2, row[:, None] < M, 0).to(tl.float32)
        x3 = tl.load(X + off + 3, row[:, None] < M, 0).to(tl.float32)
        a = a + x0 * x0
        b = b + x1 * x1
        c = c + x2 * x2
        d = d + x3 * x3
    variance = tl.sum(((a + b) + c) + d, 1) * (1.0 / N)
    inv = libdevice.rsqrt(variance + EPS)
    col = tl.arange(0, N)
    off = row[:, None] * N + col[None, :]
    x = tl.load(X + off, row[:, None] < M, 0).to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    normalized = x * inv[:, None]
    if GATED:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
        weighted = normalized * weight[None, :]
        if WEIGHT_BF16:
            weighted = weighted.to(tl.bfloat16).to(tl.float32)
        gate = tl.load(G + off, row[:, None] < M, 0).to(tl.float32)
        result = weighted * _silu(gate)
    else:
        result = normalized * (1.0 + weight[None, :])
    tl.store(Y + off, result, row[:, None] < M)


def rms_norm(x, weight, eps, gate=None, *, rows=None):
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
    if n not in (128, 256, 4096):
        raise ValueError("Unvalidated native reduction width")
    # ATen contiguous FP32 mean: vector length four, maximum 512 threads.
    dim0 = min(n // 4, 512)
    dim1 = min(1 << (m.bit_length() - 1), 512)
    block_x = min(dim0, 32)
    block_y = min(dim1, 512 // block_x)
    block_x = min(dim0, 512 // block_y)
    if n // block_x >= min(block_y * 16, 256):
        raise ValueError("Unsupported cross-warp row reduction")
    gates = gate.contiguous().view_as(matrix) if gate is not None else matrix
    out = torch.empty_like(matrix)
    rows = rows or (4 if n <= 256 else 1)
    _native_rms[(triton.cdiv(m, rows),)](
        matrix,
        weight,
        gates,
        out,
        m,
        N=n,
        EPS=eps,
        GATED=gate is not None,
        WEIGHT_BF16=weight.dtype == torch.bfloat16,
        WIDTH=block_x,
        ROWS=rows,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out.view_as(x)
