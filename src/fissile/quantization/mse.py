# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Weight-only fitting of NVFP4 scales; uses no evaluation examples or labels.

Adapted from NVIDIA Model Optimizer's exhaustive 126-scale MSE sweep, commit
1c40150d459e4817e43c9fe3db75f3b6e3a24417. Changes: emit packed E2M1 and hardware
scale layout directly, retain this project's FP32 global scale arithmetic, use
a non-unrolled loop, and require complete block-aligned weight matrices.
See licenses/Apache-2.0.txt and THIRD_PARTY_NOTICES.md for upstream attribution.
"""
import torch
import triton
import triton.language as tl

from fissile.kernels.nvfp4_fast import global_scale
from fissile.kernels.fp4_codes import _codes, _magnitudes


@triton.jit
def _mse_pack(X, Q, S, Global, Candidates, M, K: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    row = block // (K // 16)
    col = block % (K // 16)
    lane = tl.arange(0, 16)
    values = tl.load(X + block[:, None] * 16 + lane[None, :], row[:, None] < M, 0).to(
        tl.float32
    )
    magnitude = tl.abs(values)
    g = tl.load(Global)
    best_loss = tl.full((BLOCKS,), float("inf"), tl.float32)
    best_scale = tl.full((BLOCKS,), 1.0, tl.float32)
    best_codes = tl.full((BLOCKS, 16), 0, tl.int32)
    for index in tl.range(126):
        scale = tl.load(Candidates + index)
        codes = _codes(magnitude * ((1.0 / g) / scale))
        restored = (_magnitudes(codes) * scale) * g
        delta = magnitude - restored
        loss = tl.sum(delta * delta, 1)
        better = loss < best_loss
        best_loss = tl.where(better, loss, best_loss)
        best_scale = tl.where(better, scale, best_scale)
        best_codes = tl.where(better[:, None], codes, best_codes)
    sign = (values.to(tl.int32, bitcast=True) >> 28) & 8
    low, high = tl.split(tl.reshape((best_codes | sign).to(tl.uint8), (BLOCKS, 8, 2)))
    tl.store(
        Q + block[:, None] * 8 + tl.arange(0, 8)[None, :],
        low | (high << 4),
        row[:, None] < M,
    )
    scale_index = (
        ((row // 128) * (K // 64) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    tl.store(S + scale_index, best_scale.to(tl.float8e4nv), row < M)


def quantize_weight(x):
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] % 128
        or x.shape[1] % 64
        or not x.numel()
    ):
        raise ValueError(
            "Weight MSE requires CUDA BF16 matrix, N divisible by 128 and K by 64"
        )
    m, k = x.shape
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty(m * (k // 16), device=x.device, dtype=torch.float8_e4m3fn)
    g = global_scale(x)
    candidates = (
        torch.arange(1, 127, device=x.device, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
    )
    _mse_pack[(triton.cdiv(m * (k // 16), 64),)](
        x, q, s, g, candidates, m, K=k, BLOCKS=64, num_warps=4, enable_fp_fusion=False
    )
    return q.view(torch.float4_e2m1fn_x2), s, g


def dequantize(q, s, g):
    """Slow independent Torch unpacking for validation only, outside inference."""
    m, k = q.shape[0], q.shape[1] * 2
    raw = q.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(m, k)
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device, dtype=torch.float32
    )
    values = lookup[(codes & 7).long()] * torch.where((codes & 8) > 0, -1.0, 1.0)
    rows = torch.arange(m, device=q.device)[:, None]
    cols = torch.arange(k // 16, device=q.device)[None, :]
    indices = (
        ((rows // 128) * (k // 64) + cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + cols % 4
    )
    scales = s.view(torch.uint8)[indices].view(torch.float8_e4m3fn).float()
    return (values.reshape(m, k // 16, 16) * scales[:, :, None] * g).reshape(m, k)
