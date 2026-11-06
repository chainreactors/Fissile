"""Unmodified TensorFold W4A16 prompt kernels.

Offline packing converts this project's NVFP4 codes/scales without requantizing.
Inference retains BF16 activations. This is NOT arithmetic-equivalent to W4A4.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _words(Q, OUT, COUNT, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = i & 1
    lane = (i >> 1) & 31
    j = (i >> 6) & 7
    tile = i >> 9
    group = tile % (K // 64)
    row = (tile // (K // 64)) * 64 + j * 8 + (lane >> 2)
    word = tl.full((BLOCK,), 0, tl.uint32)
    for p in tl.static_range(8):
        offset = (p % 4) * 8 + p // 4
        col = group * 64 + 32 * v + 2 * (lane & 3) + offset
        byte = tl.load(Q + row * (K // 2) + col // 2, i < COUNT, 0).to(tl.uint32)
        code = (byte >> (4 * (col % 2))) & 15
        word = word | (code << (4 * p))
    tl.store(OUT + i, word, i < COUNT)


@triton.jit
def _scales(S, OUT, COUNT, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = (i // (256 * (K // 64))) * 64 + (i // 4) % 64
    col = ((i // 256) % (K // 64)) * 4 + i % 4
    old = (
        ((row // 128) * (K // 64) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    tl.store(OUT + i, tl.load(S + old, i < COUNT, 0), i < COUNT)


@dataclass(frozen=True)
class Packed:
    words: torch.Tensor
    scales: torch.Tensor
    global_scale: float
    n: int
    k: int


def repack(packed):
    q, s, g = packed
    if (
        q.ndim != 2
        or q.dtype != torch.float4_e2m1fn_x2
        or s.dtype != torch.float8_e4m3fn
        or g.dtype != torch.float32
        or g.numel() != 1
        or any(
            not t.is_cuda or not t.is_contiguous() or t.device != q.device
            for t in packed
        )
    ):
        raise ValueError("Expected this project CUDA NVFP4 pack")
    n, k = q.shape[0], q.shape[1] * 2
    if n % 128 or k % 64 or s.numel() != n * (k // 16):
        raise ValueError("Wrong aligned pack topology")
    value = float(g.item())
    if not 0 < value < float("inf"):
        raise ValueError("Invalid global scale")
    words = torch.empty(n * k // 8, device=q.device, dtype=torch.uint32)
    scales = torch.empty(n * k // 16, device=q.device, dtype=torch.uint8)
    _words[(triton.cdiv(words.numel(), 256),)](
        q.view(torch.uint8), words, words.numel(), K=k, BLOCK=256
    )
    _scales[(triton.cdiv(scales.numel(), 256),)](
        s.view(torch.uint8), scales, scales.numel(), K=k, BLOCK=256
    )
    return Packed(words, scales, value, n, k)


class TensorFoldW4A16:
    def __init__(self, library):
        self.library = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.library))
        ptr = ctypes.c_void_p
        self.lib.fissile_tf_error.restype = ctypes.c_char_p
        self.lib.fissile_tf_initialize.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        self.lib.fissile_tf_initialize.restype = ctypes.c_int
        self.lib.fissile_tf_run.argtypes = [
            ctypes.c_int,
            ptr,
            ptr,
            ptr,
            ctypes.c_float,
            ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        self.lib.fissile_tf_run.restype = ctypes.c_int
        sms, classic, lean = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._check(
            self.lib.fissile_tf_initialize(
                ctypes.byref(sms), ctypes.byref(classic), ctypes.byref(lean)
            )
        )
        self.sms = sms.value
        self.shared_memory = {"classic": classic.value, "lean": lean.value}

    def _check(self, status):
        if status:
            raise RuntimeError(
                "TensorFold CUDA: " + self.lib.fissile_tf_error().decode()
            )

    def linear(self, x, w, *, tactic="lean"):
        if (
            tactic not in ("classic", "lean")
            or x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
            or x.shape[1] != w.k
            or not len(x)
            or x.device != w.words.device
            or x.device != w.scales.device
        ):
            raise ValueError("Expected matching contiguous CUDA BF16 activation matrix")
        out = torch.empty((len(x), w.n), device=x.device, dtype=torch.bfloat16)
        self._check(
            self.lib.fissile_tf_run(
                int(tactic == "lean"),
                x.data_ptr(),
                w.words.data_ptr(),
                w.scales.data_ptr(),
                w.global_scale,
                out.data_ptr(),
                len(x),
                w.n,
                w.k,
                self.sms,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        )
        return out
