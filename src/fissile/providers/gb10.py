"""Qualified GB10 numerical implementation, independent of model and engine layout."""

import torch
from safetensors.torch import load_file
from fissile.config import require_gpu
from fissile.kernels.nvfp4_adaptive import quantize
from fissile.kernels.nvfp4 import matmul
from fissile.kernels.fp8 import quantize_fp8
from fissile.kernels.tensorfold import TensorFoldW4A16, repack
from fissile.kernels.paired_glu import RegisterPairedGLU, pair_weights
from fissile.kernels.pointwise import silu_mul
from fissile.runtime.graph import VALID_ROWS


class GB10Provider:
    def __init__(self, config):
        require_gpu(config.provider.capability)
        self.kernel = TensorFoldW4A16(config.resources["tensorfold_library"].path)
        self.paired_kernel = RegisterPairedGLU(capability=config.provider.capability)
        self.paired_max_rows = config.provider.paired_mlp_max_rows
        expected = load_file(str(config.resources["silu_lut"].path), device="cpu")[
            "bits"
        ]
        finite = (torch.arange(65536) & 0x7F80) != 0x7F80
        if expected.shape != (65536,) or expected.dtype != torch.int16:
            raise ValueError("Invalid fixed BF16 SiLU lookup table")
        if not torch.equal(
            self.paired_kernel.silu_table.view(torch.int16).cpu()[finite],
            expected[finite],
        ):
            raise ValueError(
                "Actual finite SiLU table differs from the configured table"
            )

    @staticmethod
    def _fp4_matmul(qa, sa, ga, qb, sb, gb, *, out=None):
        """W4A4 GEMM used by every FP4 projection; subclasses may pin a kernel."""
        return matmul(qa, sa, ga, qb, sb, gb, out=out)

    @staticmethod
    def matrix(x):
        if (
            not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or x.dtype != torch.bfloat16
        ):
            raise ValueError("Expected CUDA BF16 activations")
        return VALID_ROWS.mask(x.reshape(-1, x.shape[-1]).contiguous())

    def _prepared_projection(self, route, weight, packed):
        """Return compute(activation, rows, out=None).

        Without ``out`` the result is the first ``rows`` output rows. With
        ``out`` (all padded activation rows, possibly a column slice of a wider
        buffer) the same GEMM writes there instead; nothing is returned.
        """
        if route == "fp8":
            qw, sw = quantize_fp8(weight.contiguous())

            def compute(activation, rows, out=None):
                qa, sa = activation
                options = dict(
                    scale_a=sa,
                    scale_b=sw.t(),
                    out_dtype=torch.bfloat16,
                    use_fast_accum=False,
                )
                if out is not None:
                    torch._scaled_mm(qa, qw.t(), **options, out=out)
                    return None
                return torch._scaled_mm(qa, qw.t(), **options)[:rows]

        elif route == "w4a16":
            packed = repack(packed)

            def compute(activation, rows, out=None):
                if rows != 1 or out is not None:
                    raise ValueError(
                        "Only final-token, full-vocabulary scoring is qualified"
                    )
                return self.kernel.linear(activation, packed, tactic="lean")

        elif route == "fp4":

            def compute(activation, rows, out=None):
                if out is not None:
                    self._fp4_matmul(*activation, *packed, out=out)
                    return None
                return self._fp4_matmul(*activation, *packed)[:rows]

        else:
            raise ValueError("Unsupported GB10 projection route: " + route)
        return compute

    @staticmethod
    def _activation(route, x):
        if route == "fp4":
            return quantize(x, adaptive=False)
        if route == "fp8":
            return quantize_fp8(x)
        if route == "w4a16":
            return x
        raise ValueError("Unsupported GB10 projection route: " + route)

    def projection(self, route, weight, packed):
        compute = self._prepared_projection(route, weight, packed)

        def apply(x):
            return compute(self._activation(route, x), len(x))

        return apply

    def candidate_readout(self, route, weight, packed, *, block=128):
        """Final-token logits for requested vocabulary rows only; others are -inf.

        Runs the same TensorFold W4A16 kernel on the packed 128-row blocks that
        contain the candidates (packed rows are stored in contiguous 64-row
        groups), then masks every non-candidate row. Log-softmax over the result
        is normalized over the candidates, not the full vocabulary.
        """
        if route != "w4a16":
            raise ValueError("Candidate readout is defined for the W4A16 head only")
        head = repack(packed)
        words_per_block = block * head.k // 8
        scales_per_block = block * head.k // 16

        def apply(x, candidates):
            ids = sorted(set(int(i) for i in candidates))
            if len(x) != 1 or not ids or ids[0] < 0 or ids[-1] >= head.n:
                raise ValueError("Candidate readout needs one row and valid token IDs")
            blocks = sorted({i // block for i in ids})
            sub = type(head)(
                torch.cat(
                    [
                        head.words[b * words_per_block : (b + 1) * words_per_block]
                        for b in blocks
                    ]
                ),
                torch.cat(
                    [
                        head.scales[b * scales_per_block : (b + 1) * scales_per_block]
                        for b in blocks
                    ]
                ),
                head.global_scale,
                block * len(blocks),
                head.k,
            )
            part = self.kernel.linear(x, sub, tactic="lean")
            index = torch.tensor(
                [blocks.index(i // block) * block + i % block for i in ids],
                device=x.device,
            )
            logits = torch.full(
                (1, head.n), float("-inf"), dtype=part.dtype, device=x.device
            )
            logits[0, torch.tensor(ids, device=x.device)] = part[0, index]
            return logits

        return apply

    def projection_group(self, projections):
        """Reuse activation packs only within one same-input projection call.

        Individual weights, scales, GEMM shapes and output order are unchanged.
        The local dictionary never survives this call or keys by tensor address.
        """
        prepared = tuple(
            (route, self._prepared_projection(route, weight, packed))
            for route, weight, packed in projections
        )
        if not prepared:
            raise ValueError("A projection group cannot be empty")

        def apply(x):
            activations = {}
            outputs = []
            for route, compute in prepared:
                if route not in activations:
                    activations[route] = self._activation(route, x)
                outputs.append(compute(activations[route], len(x)))
            return tuple(outputs)

        return apply

    def joined_projection_group(self, projections):
        """Write same-input sibling projections into column slices of one output.

        Same activation reuse, weight packs, scales, GEMM shapes, accumulation
        and column order as ``projection_group`` followed by concatenation;
        only the output location changes, removing the concatenation copy.
        """
        routes = {"fp4", "fp8"}
        prepared = []
        start = 0
        for route, weight, packed in projections:
            if route not in routes:
                raise ValueError("Joined projection output requires FP4/FP8 routes")
            width = weight.shape[0]
            prepared.append(
                (route, self._prepared_projection(route, weight, packed), start, width)
            )
            start += width
        if not prepared:
            raise ValueError("A projection group cannot be empty")
        total = start

        def apply(x):
            rows = len(x)
            activations = {}
            out = None
            for route, compute, start, width in prepared:
                if route not in activations:
                    activations[route] = self._activation(route, x)
                padded = activations[route][0].shape[0]
                if out is None:
                    out = torch.empty(
                        (padded, total), dtype=torch.bfloat16, device=x.device
                    )
                elif out.shape[0] != padded:
                    raise RuntimeError("Grouped activation padding differs by route")
                compute(activations[route], rows, out=out[:, start : start + width])
            return out[:rows]

        return apply

    def mlp(self, packs, *, long_silu):
        if long_silu not in ("torch", "rounded"):
            raise ValueError(
                "Choose an explicit qualified long-input SiLU implementation"
            )
        g, u, d = packs
        paired = pair_weights(g, u)

        def compute(matrix):
            m = len(matrix)
            a = quantize(matrix, adaptive=False)
            if m <= self.paired_max_rows:
                call, _ = self.paired_kernel.prepare(a, paired)
                hidden = call()[:m].contiguous()
            else:
                gate = self._fp4_matmul(*a, *g)[:m]
                up = self._fp4_matmul(*a, *u)[:m]
                hidden = (
                    (torch.nn.functional.silu(gate) * up).contiguous()
                    if long_silu == "torch"
                    else silu_mul(gate, up)
                )
            return self._fp4_matmul(*quantize(hidden, adaptive=False), *d)[:m]

        return compute
