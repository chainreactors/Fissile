"""Full input-Hessian GPTQ with static NVFP4 grids and optional static act-order.

Column compensation spans the entire input dimension. Processing blocks of 128
only batches the updates; it does not discard cross-block curvature. Scales are
fixed in original 16-column groups before reordering, and packed outputs return
to original order, so inference needs no permutation. Original weights survive.
Algorithm references: GPTQ (arXiv:2210.17323), MR-GPTQ (arXiv:2509.23202).
"""

import torch
import triton
import triton.language as tl

from fissile.kernels.fp4_codes import _codes, _magnitudes


@triton.jit
def _compensate_block(
    X, U, S, P, Codes, Errors, G, M, Start, K: tl.constexpr, ROWS: tl.constexpr
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, 128)
    work = tl.load(X + rows[:, None] * K + Start + cols[None, :], rows[:, None] < M, 0)
    global_scale = tl.load(G)
    for offset in tl.range(128):
        column = Start + offset
        original_column = tl.load(P + column)
        scale = tl.load(S + rows * (K // 16) + original_column // 16, rows < M, 1.0)
        value = tl.sum(tl.gather(work, tl.full((ROWS, 1), offset, tl.int32), 1), 1)
        code = _codes(tl.abs(value) * ((1.0 / global_scale) / scale))
        reconstructed = (_magnitudes(code) * scale) * global_scale
        reconstructed = tl.where(value < 0, -reconstructed, reconstructed)
        sign = (value.to(tl.int32, bitcast=True) >> 28) & 8
        tl.store(Codes + rows * K + column, (code | sign).to(tl.uint8), rows < M)
        diagonal = tl.load(U + column * K + column)
        error = tl.div_rn(value - reconstructed, diagonal)
        tl.store(Errors + rows * 128 + offset, error, rows < M)
        upper = tl.load(U + column * K + Start + cols)
        updated = work - error[:, None] * upper[None, :]
        work = tl.where(cols[None, :] >= offset, updated, work)


@triton.jit
def _restore_pack(
    Codes, InverseOrder, Packed, Count, K: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (K // 2)
    pair = offset % (K // 2)
    first = tl.load(InverseOrder + pair * 2)
    second = tl.load(InverseOrder + pair * 2 + 1)
    lo = tl.load(Codes + row * K + first, offset < Count, 0)
    hi = tl.load(Codes + row * K + second, offset < Count, 0)
    tl.store(Packed + offset, lo | (hi << 4), offset < Count)


def prepare_inverse(hessian, *, damping=0.01, act_order=False):
    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape[0] % 128
        or not hessian.is_cuda
        or hessian.dtype != torch.float32
        or not 0 < damping <= 1
        or not bool(torch.isfinite(hessian).all())
    ):
        raise ValueError("Expected a finite CUDA FP32 full input moment matrix")
    h = (hessian + hessian.t()) * 0.5
    order = (
        torch.argsort(h.diagonal(), descending=True, stable=True)
        if act_order
        else torch.arange(len(h), device=h.device)
    )
    damp = (h.diagonal().mean() * damping).clamp_min(1.0e-6)
    h = h.index_select(0, order).index_select(1, order)
    h.diagonal().add_(damp)
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True).contiguous()
    return upper, order.contiguous()


def unpack_scales(scales, rows, width):
    r = torch.arange(rows, device=scales.device)[:, None]
    c = torch.arange(width // 16, device=scales.device)[None, :]
    index = ((r // 128) * (width // 64) + c // 4) * 512
    index = index + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    return (
        scales.view(torch.uint8)[index].view(torch.float8_e4m3fn).float().contiguous()
    )


def quantize_weight(weight, prepared, grid):
    """Compensate using grid=(packed codes, swizzled scales, global scale).

    The initial packed codes are deliberately ignored: only their frozen scale
    grid is retained. Both full-Hessian and static-order preparation are offline.
    """
    if (
        weight.ndim != 2
        or not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or not weight.numel()
        or weight.shape[0] % 128
        or weight.shape[1] % 128
    ):
        raise ValueError("Expected aligned, contiguous CUDA BF16 weight matrix")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Full GPTQ preparation requires IEEE FP32 matrix products")
    rows, width = weight.shape
    upper, order = prepared
    _, scales, global_scale = grid
    if (
        upper.shape != (width, width)
        or upper.dtype != torch.float32
        or upper.device != weight.device
        or not upper.is_contiguous()
        or order.shape != (width,)
        or order.dtype != torch.int64
        or order.device != weight.device
        or not order.is_contiguous()
        or scales.numel() != rows * (width // 16)
        or scales.dtype != torch.float8_e4m3fn
        or scales.device != weight.device
        or not scales.is_contiguous()
        or global_scale.numel() != 1
        or global_scale.dtype != torch.float32
        or global_scale.device != weight.device
    ):
        raise ValueError("Mismatched inverse or static scale grid")
    if (
        not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(upper).all())
        or not bool((upper.diagonal() > 0).all())
        or not torch.equal(
            order.sort().values, torch.arange(width, device=weight.device)
        )
        or not bool(torch.isfinite(global_scale).all())
        or not bool((global_scale > 0).all())
    ):
        raise ValueError("Invalid weight, inverse, permutation or global scale")
    grid_values = unpack_scales(scales, rows, width)
    if not bool(torch.isfinite(grid_values).all()) or not bool((grid_values > 0).all()):
        raise ValueError("Invalid microscaling grid")
    work = weight.float().index_select(1, order).contiguous()
    codes = torch.empty((rows, width), dtype=torch.uint8, device=weight.device)
    errors = torch.empty((rows, 128), dtype=torch.float32, device=weight.device)
    for start in range(0, width, 128):
        _compensate_block[(triton.cdiv(rows, 16),)](
            work,
            upper,
            grid_values,
            order,
            codes,
            errors,
            global_scale,
            rows,
            start,
            K=width,
            ROWS=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
        end = start + 128
        if end < width:
            # The upper-right Hessian block carries errors into all later columns.
            work[:, end:].sub_(errors @ upper[start:end, end:])
    packed = torch.empty((rows, width // 2), dtype=torch.uint8, device=weight.device)
    inverse_order = torch.argsort(order).contiguous()
    _restore_pack[(triton.cdiv(packed.numel(), 512),)](
        codes, inverse_order, packed, packed.numel(), K=width, BLOCK=512
    )
    return packed.view(torch.float4_e2m1fn_x2), scales, global_scale
