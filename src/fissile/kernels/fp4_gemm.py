"""Pinned CUTLASS SM121 NVFP4 W4A4 GEMM (128x128x256 tile, two stages, cooperative).

Same operands, scale layout and FP32 ``alpha = global_a * global_b`` as
``fissile.kernels.nvfp4.matmul``; every output accumulates its full K range in
one CTA. Bitwise equality with the cuBLAS path must be qualified per deployment.
"""

import ctypes
from pathlib import Path

import torch


class CutlassFP4Gemm:
    def __init__(self, library):
        self.library = Path(library).resolve(strict=True)
        lib = ctypes.CDLL(str(self.library))
        ptr = ctypes.c_void_p
        lib.fissile_fp4_gemm_error.restype = ctypes.c_char_p
        lib.fissile_fp4_gemm_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.fissile_fp4_gemm_version.restype = ctypes.c_int
        lib.fissile_fp4_gemm_initialize.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
        lib.fissile_fp4_gemm_initialize.restype = ctypes.c_int
        lib.fissile_fp4_gemm_run.argtypes = [
            ptr,
            ctypes.c_int64,
            ptr,
            ptr,
            ptr,
            ptr,
            ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        lib.fissile_fp4_gemm_run.restype = ctypes.c_int
        self.lib = lib
        version = ctypes.c_int()
        if lib.fissile_fp4_gemm_version(ctypes.byref(version)) != 1:
            raise RuntimeError("Unsupported fp4_gemm ABI")
        self.cutlass_version = version.value
        sms, shared = ctypes.c_int(), ctypes.c_int()
        self._check(
            lib.fissile_fp4_gemm_initialize(ctypes.byref(sms), ctypes.byref(shared))
        )
        self.sms, self.shared_memory = sms.value, shared.value

    def _check(self, status):
        if status:
            raise RuntimeError(
                f"CUTLASS FP4 GEMM status {status}: {self.lib.fissile_fp4_gemm_error().decode()}"
            )

    def __call__(self, qa, sa, ga, qb, sb, gb, *, out=None):
        if qa.ndim != 2 or qb.ndim != 2 or qa.shape[1] != qb.shape[1]:
            raise ValueError("Packed operands must be matrices with matching K")
        m, n, k = qa.shape[0], qb.shape[0], qa.shape[1] * 2
        device = qa.device
        if any(
            not t.is_cuda or t.device != device or not t.is_contiguous()
            for t in (qa, sa, ga, qb, sb, gb)
        ):
            raise ValueError("Operands must be contiguous on one CUDA device")
        if qa.dtype != torch.float4_e2m1fn_x2 or qb.dtype != qa.dtype:
            raise ValueError("Expected E2M1 packed operands")
        if sa.dtype != torch.float8_e4m3fn or sb.dtype != sa.dtype:
            raise ValueError("Expected E4M3 block scales")
        if any(g.dtype != torch.float32 or g.numel() != 1 for g in (ga, gb)):
            raise ValueError("Expected scalar FP32 global scales")
        if sa.numel() != (m + 127) // 128 * 128 * (k // 16) or sb.numel() != (
            n + 127
        ) // 128 * 128 * (k // 16):
            raise ValueError("Scale storage does not match SWIZZLE_32_4_4 layout")
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        elif (
            out.dtype != torch.bfloat16
            or out.device != device
            or tuple(out.shape) != (m, n)
            or out.stride(1) != 1
            or out.stride(0) < n
            or out.stride(0) % 8
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "Output must be a row-major BF16 [m, n] view, 16-byte aligned"
            )
        alpha = ga * gb
        self._check(
            self.lib.fissile_fp4_gemm_run(
                out.data_ptr(),
                out.stride(0),
                qa.data_ptr(),
                qb.data_ptr(),
                sa.data_ptr(),
                sb.data_ptr(),
                alpha.data_ptr(),
                m,
                n,
                k,
                torch.cuda.current_stream(device).cuda_stream,
            )
        )
        return out
