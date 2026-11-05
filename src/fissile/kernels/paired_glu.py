"""Exact paired GLU using a fixed full-BF16 SiLU function table (128 KiB).

Offline weight layout changes only: gate64/up64 rows and matching scale bytes.
No requantization, FP32 global intermediate, reduced model or output vocabulary.
"""

from dataclasses import dataclass
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import torch

from fissile._vendor.sm12x.register_paired import RegisterPairedGemmKernel, lookup_silu


@cute.jit
def rounded_silu(value):
    x = value.to(cutlass.Float32)
    denominator = 1.0 + cute.math.exp(-x, fastmath=False)
    return cute.math.div(
        x, denominator, rounding=cute.math.RoundingMode.NEAREST_EVEN
    ).to(cutlass.BFloat16)


@cute.kernel
def _silu_kernel(x: cute.Tensor, y: cute.Tensor):
    index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
    y[index] = rounded_silu(x[index])


@cute.jit
def _silu_launch(xp: cute.Pointer, yp: cute.Pointer, stream: cuda.CUstream):
    x = cute.make_tensor(xp, cute.make_layout((65536,), stride=(1,)))
    y = cute.make_tensor(yp, cute.make_layout((65536,), stride=(1,)))
    _silu_kernel(x, y).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)


def _target_arch(capability):
    targets = {(12, 0): "sm_120a", (12, 1): "sm_121a"}
    try:
        return targets[tuple(capability)]
    except KeyError:
        raise ValueError("Paired GLU requires an explicit SM120 or SM121 target") from None


def _build_silu_table(*, gpu_arch="sm_121a"):
    x = (
        torch.arange(65536, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    out = torch.empty_like(x)
    args = (
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, 16),
        make_ptr(cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, 16),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    fn = cute.compile(_silu_launch, *args, options="--gpu-arch=" + gpu_arch)
    fn(*args)
    return x, out


@cute.kernel
def _lookup_test_kernel(x: cute.Tensor, y: cute.Tensor, table: cute.Tensor):
    index = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
    y[index] = lookup_silu(x[index], table)


@cute.jit
def _lookup_test_launch(
    xp: cute.Pointer, yp: cute.Pointer, tp: cute.Pointer, stream: cuda.CUstream
):
    layout = cute.make_layout((65536,), stride=(1,))
    _lookup_test_kernel(
        cute.make_tensor(xp, layout),
        cute.make_tensor(yp, layout),
        cute.make_tensor(tp, layout),
    ).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)


def silu_all_codes(*, capability=(12, 1)):
    gpu_arch = _target_arch(capability)
    if torch.cuda.get_device_capability() != tuple(capability):
        raise RuntimeError("SiLU test device does not match the selected target")
    x, table = _build_silu_table(gpu_arch=gpu_arch)
    out = torch.empty_like(x)
    args = tuple(
        make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, 16)
        for t in (x, out, table)
    )
    args += (cuda.CUstream(torch.cuda.current_stream().cuda_stream),)
    fn = cute.compile(_lookup_test_launch, *args, options="--gpu-arch=" + gpu_arch)
    fn(*args)
    torch.cuda.synchronize()
    return x, out


@dataclass
class PairedWeights:
    q: torch.Tensor
    s: torch.Tensor
    globals: torch.Tensor
    n: int
    k: int


def pair_weights(gate, up):
    qg, sg, gg = gate
    qu, su, gu = up
    if qg.shape != qu.shape or qg.ndim != 2:
        raise ValueError("Gate/up shapes must match")
    n, k = qg.shape[0], qg.shape[1] * 2
    if not n or not k or n % 128 or k % 128:
        raise ValueError("Require original N/K128 aligned packs")
    if any(
        not t.is_cuda or t.device != qg.device or not t.is_contiguous()
        for t in (*gate, *up)
    ):
        raise ValueError("All packed tensors must be contiguous on one CUDA device")
    if qg.dtype != torch.float4_e2m1fn_x2 or qu.dtype != qg.dtype:
        raise ValueError("FP4 code type mismatch")
    if (
        sg.dtype != torch.float8_e4m3fn
        or su.dtype != sg.dtype
        or sg.numel() != n * k // 16
        or su.shape != sg.shape
    ):
        raise ValueError("E4M3 physical scale storage mismatch")
    if (
        gg.dtype != torch.float32
        or gu.dtype != torch.float32
        or gg.numel() != 1
        or gu.numel() != 1
    ):
        raise ValueError("Require scalar FP32 global scales")
    q = torch.stack(
        (
            qg.view(torch.uint8).reshape(n // 64, 64, k // 2),
            qu.view(torch.uint8).reshape(n // 64, 64, k // 2),
        ),
        1,
    ).reshape(2 * n, k // 2)
    gs = sg.view(torch.uint8).reshape(n // 128, k // 64, 32, 4, 4)
    us = su.view(torch.uint8).reshape(n // 128, k // 64, 32, 4, 4)
    low = torch.cat((gs[:, :, :, :2, :], us[:, :, :, :2, :]), 3)
    high = torch.cat((gs[:, :, :, 2:, :], us[:, :, :, 2:, :]), 3)
    s = torch.stack((low, high), 1).reshape(-1)
    globals_ = torch.stack((gg.reshape(()), gu.reshape(())))
    if not bool(torch.isfinite(globals_).all()) or not bool((globals_ > 0).all()):
        raise ValueError("Invalid global scales")
    # Full inverse checks, not merely decoded sample agreement.
    qi = q.reshape(n // 64, 2, 64, k // 2)
    si = s.reshape(n // 128, 2, k // 64, 32, 4, 4)
    for arm, original in enumerate((gate, up)):
        recovered_q = qi[:, arm].reshape(n, k // 2)
        recovered_s = torch.cat(
            (
                si[:, 0, :, :, arm * 2 : arm * 2 + 2, :],
                si[:, 1, :, :, arm * 2 : arm * 2 + 2, :],
            ),
            3,
        ).reshape(-1)
        if not torch.equal(
            recovered_q, original[0].view(torch.uint8)
        ) or not torch.equal(recovered_s, original[1].view(torch.uint8)):
            raise ValueError("Offline byte permutation is not exactly reversible")
        if not torch.equal(globals_[arm], original[2].reshape(())):
            raise ValueError("Global scale changed")
    return PairedWeights(
        q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn), globals_, n, k
    )


class _Entry:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.op = RegisterPairedGemmKernel(
            16,
            (128, 128),
            (1, 1),
            tile_k=128,
            single_work_tile_per_cta=True,
            enable_pdl=False,
        )

    @cute.jit
    def launch(
        self,
        ap: cute.Pointer,
        bp: cute.Pointer,
        sap: cute.Pointer,
        sbp: cute.Pointer,
        cp: cute.Pointer,
        alphap: cute.Pointer,
        lutp: cute.Pointer,
        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(
            ap, cute.make_layout((m, self.k, 1), stride=(self.k, 1, m * self.k))
        )
        b = cute.make_tensor(
            bp,
            cute.make_layout(
                (self.n * 2, self.k, 1), stride=(self.k, 1, self.n * 2 * self.k)
            ),
        )
        c = cute.make_tensor(
            cp, cute.make_layout((m, self.n, 1), stride=(self.n, 1, m * self.n))
        )
        sa = cute.make_tensor(sap, cute.make_layout((1,), stride=(1,)))
        sb = cute.make_tensor(sbp, cute.make_layout((1,), stride=(1,)))
        alpha = cute.make_tensor(alphap, cute.make_layout((2,), stride=(1,)))
        table = cute.make_tensor(lutp, cute.make_layout((65536,), stride=(1,)))
        # Full grid, not the SM-count cap used by persistent multi-tile CTAs.
        self.op(a, b, sa, sb, c, alpha, 2147483647, stream, silu_lut=table)


class _FP4Entry(_Entry):
    """Same kernel; the epilogue writes the NVFP4 pack of the GLU output."""

    @cute.jit
    def launch(
        self,
        ap: cute.Pointer,
        bp: cute.Pointer,
        sap: cute.Pointer,
        sbp: cute.Pointer,
        cp: cute.Pointer,
        alphap: cute.Pointer,
        lutp: cute.Pointer,
        qp: cute.Pointer,
        sp: cute.Pointer,
        gp: cute.Pointer,
        m: cutlass.Int32,
        scale_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(
            ap, cute.make_layout((m, self.k, 1), stride=(self.k, 1, m * self.k))
        )
        b = cute.make_tensor(
            bp,
            cute.make_layout(
                (self.n * 2, self.k, 1), stride=(self.k, 1, self.n * 2 * self.k)
            ),
        )
        c = cute.make_tensor(
            cp, cute.make_layout((m, self.n, 1), stride=(self.n, 1, m * self.n))
        )
        sa = cute.make_tensor(sap, cute.make_layout((1,), stride=(1,)))
        sb = cute.make_tensor(sbp, cute.make_layout((1,), stride=(1,)))
        alpha = cute.make_tensor(alphap, cute.make_layout((2,), stride=(1,)))
        table = cute.make_tensor(lutp, cute.make_layout((65536,), stride=(1,)))
        codes = cute.make_tensor(qp, cute.make_layout((m, self.n // 2), stride=(self.n // 2, 1)))
        scales = cute.make_tensor(sp, cute.make_layout((scale_rows * (self.n // 16),), stride=(1,)))
        global_scale = cute.make_tensor(gp, cute.make_layout((1,), stride=(1,)))
        self.op(a, b, sa, sb, c, alpha, 2147483647, stream, silu_lut=table,
                fp4_codes=codes, fp4_scales=scales, fp4_global=global_scale)


class RegisterPairedGLU:
    def __init__(self, *, capability=(12, 1)):
        self.gpu_arch = _target_arch(capability)
        if torch.cuda.get_device_capability() != tuple(capability):
            raise RuntimeError("Paired GLU device does not match the selected target")
        # Fixed mathematical function table, not a request/output/KV cache.
        _, self.silu_table = _build_silu_table(gpu_arch=self.gpu_arch)
        self.compiled = {}
        self.compile_records = []

    def prepare(self, activation, paired, *, guarded=False, fp4_global=None):
        """``fp4_global`` (one FP32 value): the call returns the NVFP4 pack of the
        output (codes, block scales, global scale) written by the epilogue,
        equal to ``nvfp4_static.quantize_static`` of the BF16 output."""
        qa, sa, ga = activation
        m = qa.shape[0]
        n, k = paired.n, paired.k
        if qa.dtype != torch.float4_e2m1fn_x2 or qa.shape != (m, k // 2) or m % 16:
            raise ValueError("Activation codes/padding mismatch")
        if (
            sa.dtype != torch.float8_e4m3fn
            or sa.numel() != ((m + 127) // 128) * 128 * k // 16
        ):
            raise ValueError("Activation scale layout mismatch")
        if ga.dtype != torch.float32 or ga.numel() != 1:
            raise ValueError("Activation global scale mismatch")
        if (
            paired.q.shape != (2 * n, k // 2)
            or paired.s.numel() != 2 * n * k // 16
            or paired.globals.shape != (2,)
        ):
            raise ValueError("Paired weight topology mismatch")
        if any(
            not t.is_cuda or t.device != qa.device or not t.is_contiguous()
            for t in (qa, sa, ga, paired.q, paired.s, paired.globals, self.silu_table)
        ):
            raise ValueError("Operands must be contiguous on one CUDA device")
        storage = None
        if guarded:
            storage = torch.full(
                (m * n + 512,), float("nan"), dtype=torch.bfloat16, device=qa.device
            )
            out = storage[256:-256].reshape(m, n)
        else:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=qa.device)
        alpha = ga * paired.globals
        if fp4_global is not None:
            if guarded or fp4_global.dtype != torch.float32 or fp4_global.numel() != 1:
                raise ValueError("NVFP4 output needs one FP32 global scale and no guards")
            scale_rows = (m + 127) // 128 * 128
            codes = torch.empty((m, n // 2), dtype=torch.uint8, device=qa.device)
            scales = torch.empty(scale_rows * n // 16, dtype=torch.float8_e4m3fn, device=qa.device)
        spec = (
            (qa, cutlass.Float4E2M1FN, 32),
            (paired.q, cutlass.Float4E2M1FN, 32),
            (sa, cutlass.Float8E4M3FN, 16),
            (paired.s, cutlass.Float8E4M3FN, 16),
            (out, cutlass.BFloat16, 16),
            (alpha, cutlass.Float32, 4),
            (self.silu_table, cutlass.BFloat16, 16),
        )
        if fp4_global is not None:
            spec += (
                (codes, cutlass.Uint8, 16),
                (scales, cutlass.Uint8, 16),
                (fp4_global, cutlass.Float32, 4),
            )
        if any(t.data_ptr() % alignment for t, dtype, alignment in spec):
            raise ValueError("Pointer alignment mismatch")
        args = tuple(
            make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, alignment)
            for t, dtype, alignment in spec
        )
        args += (cutlass.Int32(m),)
        if fp4_global is not None:
            args += (cutlass.Int32(scale_rows),)
        args += (cuda.CUstream(torch.cuda.current_stream().cuda_stream),)
        key = (n, k, fp4_global is not None)
        if key not in self.compiled:
            entry = (_FP4Entry if fp4_global is not None else _Entry)(n, k)
            start = time.perf_counter()
            compiled = cute.compile(entry.launch, *args, options="--gpu-arch=" + self.gpu_arch)
            self.compiled[key] = (entry, compiled)
            self.compile_records.append(
                {
                    "n": n,
                    "k": k,
                    "seconds": time.perf_counter() - start,
                    "main_gemm_kernels": 1,
                    "full_grid": True,
                    "output_tile": [128, 64],
                }
            )
        fn = self.compiled[key][1]
        keepalive = (activation, paired, out, alpha, storage, self.silu_table)
        if fp4_global is not None:
            keepalive += (codes, scales, fp4_global)

        def run():
            _ = keepalive
            fn(*args)
            if fp4_global is not None:
                return codes.view(torch.float4_e2m1fn_x2), scales, fp4_global
            return out

        run.guards = (storage[:256], storage[-256:]) if guarded else None
        return run, {
            "m": m,
            "n": n,
            "k": k,
            "joined_weight_rows": 2 * n,
            "main_gemm_kernels": 1,
            "global_intermediate": False,
            "output_elements": m * n,
        }
