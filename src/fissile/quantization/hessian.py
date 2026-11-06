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
"""Input-moment-weighted NVFP4 block scales for GPTQ preparation.

Adapted from NVIDIA Model Optimizer's local-Hessian scale sweep, commit
1c40150d459e4817e43c9fe3db75f3b6e3a24417. For every 16-value block, each of the
126 positive E4M3 scales is scored by the quadratic error under the block's
16x16 input second moment. Changes: evaluate this project's packed-value and
FP32 global-scale arithmetic and emit packed E2M1 and swizzled E4M3 directly.
See licenses/Apache-2.0.txt and THIRD_PARTY_NOTICES.md for upstream attribution.
"""

import torch
import triton
import triton.language as tl

from fissile.kernels.fp4_codes import _codes, _magnitudes
from fissile.kernels.nvfp4_fast import global_scale


@triton.jit
def _hessian_pack(X, H, Q, S, Global, Candidates, M, K: tl.constexpr, ROWS: tl.constexpr):
    col = tl.program_id(0) % (K // 16)
    row = (tl.program_id(0) // (K // 16)) * ROWS + tl.arange(0, ROWS)
    lane = tl.arange(0, 16)
    off = row[:, None] * K + col * 16 + lane[None, :]
    w = tl.load(X + off, row[:, None] < M, 0).to(tl.float32)
    magnitude = tl.abs(w)
    g = tl.load(Global)
    sign = tl.where(w < 0, -1.0, 1.0)
    h = tl.load(H + col * 256 + lane[:, None] * 16 + lane[None, :]).to(tl.float32)
    best_loss = tl.full((ROWS,), float("inf"), tl.float32)
    best_scale = tl.full((ROWS,), 1.0, tl.float32)
    best_codes = tl.full((ROWS, 16), 0, tl.int32)
    for index in tl.range(126):
        scale = tl.load(Candidates + index)
        codes = _codes(magnitude * ((1.0 / g) / scale))
        restored = (_magnitudes(codes) * scale) * g
        delta = sign * (magnitude - restored)
        hd = tl.dot(delta, h, input_precision="ieee")
        loss = tl.sum(hd * delta, 1)
        better = loss < best_loss
        best_loss = tl.where(better, loss, best_loss)
        best_scale = tl.where(better, scale, best_scale)
        best_codes = tl.where(better[:, None], codes, best_codes)
    sign_bits = (w.to(tl.int32, bitcast=True) >> 28) & 8
    low, high = tl.split(tl.reshape((best_codes | sign_bits).to(tl.uint8), (ROWS, 8, 2)))
    tl.store(Q + row[:, None] * (K // 2) + col * 8 + tl.arange(0, 8)[None, :], low | (high << 4), row[:, None] < M)
    scale_index = ((row // 128) * (K // 64) + col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
    tl.store(S + scale_index, best_scale.to(tl.float8e4nv), row < M)


def local_moments(hessian):
    """The 16x16 diagonal blocks of a full input moment, as FP32."""
    width = hessian.shape[0]
    return torch.stack([hessian[i : i + 16, i : i + 16] for i in range(0, width, 16)]).float().contiguous()


def quantize_weight(x, local):
    """Packed codes, swizzled block scales and global scale of ``x``.

    ``local`` holds the FP32 16x16 input moments of every 16-column block.
    """
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] % 128
        or x.shape[1] % 64
        or not x.numel()
    ):
        raise ValueError("Requires CUDA BF16 weights, N divisible by 128 and K by 64")
    m, k = x.shape
    if local.shape != (k // 16, 16, 16) or local.device != x.device or local.dtype != torch.float32:
        raise ValueError("Expected matching CUDA FP32 block input moments")
    # Offline preparation, so synchronizing these validity checks is intentional.
    if not bool(torch.isfinite(local).all()) or not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite input moments or weights")
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty(m * (k // 16), device=x.device, dtype=torch.float8_e4m3fn)
    g = global_scale(x)
    candidates = torch.arange(1, 127, device=x.device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    _hessian_pack[(triton.cdiv(m, 32) * (k // 16),)](
        x, local.contiguous(), q, s, g, candidates, m, K=k, ROWS=32, num_warps=4, enable_fp_fusion=False
    )
    return q.view(torch.float4_e2m1fn_x2), s, g
