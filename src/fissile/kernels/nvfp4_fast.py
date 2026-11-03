"""Experimental NVFP4 quantization with fused absolute maximum reduction.

Preserves the original block-16/global-max recipe. Runtime row counts avoid
compiling a separate quantization kernel for every input length.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _amax_parts(X, Parts, Count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offset, offset < Count, 0).to(tl.float32)
    tl.store(Parts + tl.program_id(0), tl.max(tl.abs(values), 0))


@triton.jit
def _amax_finish(Parts, Global, Count, BLOCK: tl.constexpr):
    offset = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(Parts + offset, offset < Count, 0), 0)
    tl.store(Global, tl.maximum(maximum, 1.0e-12) / 2688.0)


@triton.jit
def _pack(X, Q, S, Global, M, PAD_M, SCALE_M, K: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    row = block // (K // 16)
    col = block % (K // 16)
    lane = tl.arange(0, 16)
    values = tl.load(
        X + row[:, None] * K + col[:, None] * 16 + lane[None, :], row[:, None] < M, 0
    ).to(tl.float32)
    global_scale = tl.load(Global)
    maximum = tl.max(tl.abs(values), 1)
    scale = tl.minimum(tl.maximum((maximum / 6.0) / global_scale, 0.015625), 448.0).to(
        tl.float8e4nv
    )
    normalized = values * ((1.0 / global_scale) / scale.to(tl.float32)[:, None])
    magnitude = tl.abs(normalized)
    code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude < 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude < 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(magnitude < 3.5, 5, tl.where(magnitude <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    sign = (normalized.to(tl.int32, bitcast=True) >> 28) & 8
    low, high = tl.split(tl.reshape((code | sign).to(tl.uint8), (BLOCKS, 8, 2)))
    packed = low | (high << 4)
    tl.store(
        Q + row[:, None] * (K // 2) + col[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        row[:, None] < PAD_M,
    )
    scale_index = (
        ((row // 128) * (K // 64) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    tl.store(S + scale_index, scale, row < SCALE_M)


def global_scale(x):
    blocks = triton.cdiv(x.numel(), 16384)
    parts = torch.empty(blocks, device=x.device, dtype=torch.float32)
    result = torch.empty((), device=x.device, dtype=torch.float32)
    _amax_parts[(blocks,)](x, parts, x.numel(), BLOCK=16384, num_warps=8)
    _amax_finish[(1,)](
        parts,
        result,
        blocks,
        BLOCK=triton.next_power_of_2(blocks),
        enable_fp_fusion=False,
    )
    return result


def quantize(x):
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] % 64
        or not x.shape[0]
    ):
        raise ValueError(
            "NVFP4 requires nonempty contiguous CUDA BF16 matrix with K divisible by 64"
        )
    m, k = x.shape
    padded = triton.cdiv(m, 16) * 16
    scale_m = triton.cdiv(m, 128) * 128
    q = torch.empty((padded, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty(
        scale_m * (k // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    g = global_scale(x)
    _pack[(triton.cdiv(scale_m * (k // 16), 64),)](
        x, q, scales, g, m, padded, scale_m, K=k, BLOCKS=64, enable_fp_fusion=False
    )
    return q.view(torch.float4_e2m1fn_x2), scales, g
