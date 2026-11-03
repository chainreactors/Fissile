"""gb10-mixed-v1 arithmetic with a pinned CUTLASS NVFP4 GEMM instead of cuBLAS heuristics.

Routes, packs, activation quantization, paired GLU, FP8 and W4A16 paths are
inherited unchanged. Only the W4A4 GEMM kernel differs; it accumulates each
output's full K range in one CTA, like cuBLAS's data-parallel kernels.
"""

from fissile.providers.gb10 import GB10Provider


class GB10CutlassProvider(GB10Provider):
    def __init__(self, config):
        super().__init__(config)
        from fissile.kernels.fp4_gemm import CutlassFP4Gemm

        self.fp4_gemm = CutlassFP4Gemm(config.resources["fp4_gemm_library"].path)

    def _fp4_matmul(self, qa, sa, ga, qb, sb, gb, *, out=None):
        return self.fp4_gemm(qa, sa, ga, qb, sb, gb, out=out)
