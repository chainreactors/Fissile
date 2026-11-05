"""NVFP4 activation packing with a supplied (static) global scale.

``quantize_static(x, g)`` is ``nvfp4_adaptive.quantize(x, adaptive=False)`` with the
FP32 global scale supplied instead of derived from the tensor amax: one launch, same
per-block arithmetic and layouts.
"""

import torch
import triton

from fissile.kernels.nvfp4_adaptive import _pack


def _layouts(m, k, device):
    padded = triton.cdiv(m, 16) * 16
    scale_m = triton.cdiv(m, 128) * 128
    q = torch.empty((padded, k // 2), device=device, dtype=torch.uint8)
    s = torch.empty(scale_m * (k // 16), device=device, dtype=torch.float8_e4m3fn)
    return padded, scale_m, q, s


def _check_global(g, device):
    if (
        not isinstance(g, torch.Tensor)
        or g.dtype != torch.float32
        or g.numel() != 1
        or g.device != device
    ):
        raise ValueError("Global scale must be one FP32 value on the activation device")


def quantize_static(x, g):
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] % 64
        or not x.shape[0]
    ):
        raise ValueError("Expected nonempty CUDA BF16 matrix, K divisible by 64")
    _check_global(g, x.device)
    m, k = x.shape
    padded, scale_m, q, s = _layouts(m, k, x.device)
    _pack[(triton.cdiv(scale_m * (k // 16), 64),)](
        x, q, s, g, m, padded, scale_m, K=k, BLOCKS=64, ADAPTIVE=False,
        enable_fp_fusion=False,
    )
    return q.view(torch.float4_e2m1fn_x2), s, g
