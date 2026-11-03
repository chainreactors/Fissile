"""Experimental NVFP4 activation packing for the selected SM121 runtime.

Uses Blackwell's native E2M1 conversion. The optional two-candidate scale rule
is inspired by Four Over Six (https://arxiv.org/abs/2512.02010v5): select the
lower-error FP8-rounded scale from block-amax/6 and block-amax/4. This variant
retains the project's original tensor-wide dynamic scale and format; it does
not claim to reproduce the paper's complete quantization/training recipe.
"""

import torch
import triton
import triton.language as tl

from fissile.kernels.nvfp4_fast import global_scale
from fissile.kernels.fp4_codes import _magnitudes


@triton.jit
def _convert_pairs(values, BLOCKS: tl.constexpr):
    lo, hi = tl.split(tl.reshape(values, (BLOCKS, 8, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 bits; cvt.rn.satfinite.e2m1x2.f32 bits, $2, $1; mov.b32 $0, {bits, 0, 0, 0}; }",
        constraints="=r,f,f",
        args=[lo, hi],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8)


@triton.jit
def _restore_magnitudes(packed, BLOCKS: tl.constexpr):
    lo = _magnitudes((packed & 7).to(tl.int32))
    hi = _magnitudes(((packed >> 4) & 7).to(tl.int32))
    return tl.reshape(tl.join(lo, hi), (BLOCKS, 16))


@triton.jit
def _pack(
    X,
    Q,
    S,
    G,
    M,
    PAD_M,
    SCALE_M,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    row = block // (K // 16)
    col = block % (K // 16)
    lane = tl.arange(0, 16)
    values = tl.load(
        X + row[:, None] * K + col[:, None] * 16 + lane[None, :], row[:, None] < M, 0
    ).to(tl.float32)
    g = tl.load(G)
    amax = tl.max(tl.abs(values), 1)
    s6 = tl.minimum(tl.maximum((amax / 6.0) / g, 0.015625), 448.0).to(tl.float8e4nv)
    n6 = values * ((1.0 / g) / s6.to(tl.float32)[:, None])
    q6 = _convert_pairs(n6, BLOCKS)
    if ADAPTIVE:
        s4 = tl.minimum(tl.maximum((amax / 4.0) / g, 0.015625), 448.0).to(tl.float8e4nv)
        n4 = values * ((1.0 / g) / s4.to(tl.float32)[:, None])
        q4 = _convert_pairs(n4, BLOCKS)
        restored6 = (_restore_magnitudes(q6, BLOCKS) * s6.to(tl.float32)[:, None]) * g
        restored4 = (_restore_magnitudes(q4, BLOCKS) * s4.to(tl.float32)[:, None]) * g
        e6 = tl.abs(values) - restored6
        e4 = tl.abs(values) - restored4
        prefer4 = tl.sum(e4 * e4, 1) < tl.sum(e6 * e6, 1)
        packed = tl.where(prefer4[:, None], q4, q6)
        scale = tl.where(prefer4, s4.to(tl.float32), s6.to(tl.float32)).to(
            tl.float8e4nv
        )
    else:
        packed = q6
        scale = s6
    tl.store(
        Q + row[:, None] * (K // 2) + col[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        row[:, None] < PAD_M,
    )
    index = (
        ((row // 128) * (K // 64) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    tl.store(S + index, scale, row < SCALE_M)


def quantize(x, *, adaptive=True):
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] % 64
        or not x.shape[0]
    ):
        raise ValueError("Expected nonempty CUDA BF16 matrix, K divisible by 64")
    m, k = x.shape
    padded = triton.cdiv(m, 16) * 16
    scale_m = triton.cdiv(m, 128) * 128
    q = torch.empty((padded, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty(scale_m * (k // 16), device=x.device, dtype=torch.float8_e4m3fn)
    g = global_scale(x)
    _pack[(triton.cdiv(scale_m * (k // 16), 64),)](
        x,
        q,
        s,
        g,
        m,
        padded,
        scale_m,
        K=k,
        BLOCKS=64,
        ADAPTIVE=adaptive,
        enable_fp_fusion=False,
    )
    return q.view(torch.float4_e2m1fn_x2), s, g
