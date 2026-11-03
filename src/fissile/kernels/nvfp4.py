"""Experimental dense NVFP4 W4A4 using the installed PyTorch cuBLAS interface.

Format: E2M1 values, E4M3 scales per 16 values, FP32 global scale, and
SWIZZLE_32_4_4 scales as documented by NVIDIA cuBLAS. No weight file is changed.
This is approximate inference, not a replacement admitted for the product.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(
    X,
    Q,
    S,
    Global,
    M: tl.constexpr,
    K: tl.constexpr,
    PAD_M: tl.constexpr,
    SCALE_M: tl.constexpr,
    BLOCKS: tl.constexpr,
):
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
    scale_float = scale.to(tl.float32)
    normalized = values * ((1.0 / global_scale) / scale_float[:, None])
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
    code = (code | sign).to(tl.uint8)
    low, high = tl.split(tl.reshape(code, (BLOCKS, 8, 2)))
    packed = low | (high << 4)
    tl.store(
        Q + row[:, None] * (K // 2) + col[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        row[:, None] < PAD_M,
    )
    # Each 128x4 scale tile is interleaved as [32,4,4].
    scale_index = (
        ((row // 128) * (K // 64) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    tl.store(S + scale_index, scale, row < SCALE_M)


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
            "NVFP4 expects nonempty contiguous CUDA BF16 matrix with K divisible by 64"
        )
    m, k = x.shape
    padded = triton.cdiv(m, 16) * 16
    scale_m = triton.cdiv(m, 128) * 128
    q = torch.empty((padded, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty(
        scale_m * (k // 16), device=x.device, dtype=torch.float8_e4m3fn
    )
    global_scale = x.abs().amax().float().clamp_min(1.0e-12) / 2688.0
    _quantize[(triton.cdiv(scale_m * (k // 16), 64),)](
        x,
        q,
        scales,
        global_scale,
        m,
        k,
        padded,
        scale_m,
        BLOCKS=64,
        enable_fp_fusion=False,
    )
    return q.view(torch.float4_e2m1fn_x2), scales, global_scale


def matmul(qa, sa, ga, qb, sb, gb, *, out=None):
    """BF16 W4A4 product. ``out`` may be a column slice of a wider row-major
    buffer; cuBLASLt then writes with that leading dimension, without a copy."""
    args = (
        qa,
        qb.t(),
        [sa, ga],
        [2, 0],
        [1, 0],
        [sb, gb],
        [2, 0],
        [1, 0],
        None,
        torch.bfloat16,
        [],
        False,
    )
    if out is None:
        return torch.ops.aten._scaled_mm_v2(*args)
    return torch.ops.aten._scaled_mm_v2.out(*args, out=out)
