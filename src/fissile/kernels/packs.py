"""Safe serialization boundary for the existing E2M1/E4M3/FP32 packed format."""

import hashlib
from pathlib import Path
import torch
from fissile.config import sha256


def weight_hash(weight):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Expected effective BF16 projection weights")
    return hashlib.sha256(
        memoryview(weight.detach().contiguous().cpu().view(torch.uint8).numpy())
    ).hexdigest()


def load_pack(path, metadata, *, device):
    from safetensors.torch import load_file

    if sha256(path) != metadata["sha256"]:
        raise ValueError("Prepared projection changed")
    saved = load_file(str(path), device="cpu")
    rows, width = metadata["shape"]
    if (
        rows <= 0
        or width <= 0
        or rows % 128
        or width % 128
        or set(saved) != {"q", "s", "g"}
    ):
        raise ValueError("Invalid packed projection topology")
    q, s, g = (saved[k] for k in ("q", "s", "g"))
    if (
        q.dtype != torch.uint8
        or q.shape != (rows, width // 2)
        or s.dtype != torch.uint8
        or s.shape != (rows * (width // 16),)
        or g.dtype != torch.float32
        or g.shape != torch.Size([])
        or not bool(torch.isfinite(g))
        or not bool(g > 0)
        or not bool(((s >= 1) & (s <= 126)).all())
    ):
        raise ValueError("Invalid packed values or microscaling layout")
    return (
        q.to(device).view(torch.float4_e2m1fn_x2),
        s.to(device).view(torch.float8_e4m3fn),
        g.to(device),
    )
