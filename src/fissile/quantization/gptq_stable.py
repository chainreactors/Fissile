"""Equilibrated FP64 offline inverse for full-Hessian NVFP4 GPTQ.

The 1% damping and the quantization/inference kernels are unchanged. Diagonal
equilibration and both Cholesky factorizations use FP64; the resulting upper
factor is converted to FP32 for offline compensation. No inference FP64 work.
"""

import torch

from fissile.quantization.gptq import quantize_weight


def prepare_inverse(hessian, *, damping=0.01, act_order=False):
    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape[0] % 128
        or not hessian.is_cuda
        or hessian.dtype not in (torch.float32, torch.float64)
        or not 0 < damping <= 1
        or not bool(torch.isfinite(hessian).all())
    ):
        raise ValueError("Expected a finite full CUDA input moment matrix")
    h = hessian.double()
    h = (h + h.t()) * 0.5
    order = (
        torch.argsort(h.diagonal(), descending=True, stable=True)
        if act_order
        else torch.arange(len(h), device=h.device)
    )
    damp = (h.diagonal().mean() * damping).clamp_min(1.0e-6)
    h = h.index_select(0, order).index_select(1, order)
    h.diagonal().add_(damp)
    scale = h.diagonal().sqrt()
    correlation = h / scale[:, None] / scale[None, :]
    lower = torch.linalg.cholesky(correlation)
    inverse = torch.cholesky_inverse(lower)
    upper = (
        (torch.linalg.cholesky(inverse, upper=True) / scale[None, :])
        .float()
        .contiguous()
    )
    if not bool(torch.isfinite(upper).all()) or not bool((upper.diagonal() > 0).all()):
        raise ValueError("FP64 inverse is not representable for FP32 compensation")
    return upper, order.contiguous()
