"""Static NVFP4 activation scales on the gb10-mixed-v2 kernels.

An activation *site* is one W4A4 activation: the input of a projection or of a
projection group, the MLP input and the MLP hidden activation. A site is named by
the first projection key that consumes it (``...mlp.gate_proj`` for the MLP input,
``...mlp.down_proj`` for the hidden activation).

``gb10-mixed-v2-static`` quantizes every site with a fixed FP32 global scale from a
calibrated scale file instead of the per-forward tensor amax: one pack launch per
activation, and each GEMM's alpha (activation scale x weight scale) computed once. Scores differ from gb10-mixed-v2 and need
their own accuracy qualification.

``gb10-mixed-v2-calibration`` keeps the gb10-mixed-v2 arithmetic exactly and records
the largest per-forward global scale of every site (see
``scripts/calibrate_activation_scales.py``).
"""

import json
import os
import torch
from fissile.kernels.nvfp4_adaptive import quantize
from fissile.kernels.nvfp4_static import quantize_static
from fissile.kernels.paired_glu import pair_weights
from fissile.kernels.pointwise import silu_mul
from fissile.providers.gb10_cutlass import GB10CutlassProvider
from fissile.providers.static_scales import (
    CALIBRATION_FORMAT,
    CALIBRATION_OUTPUT,
    load_activation_scales,
)


class _SiteProvider(GB10CutlassProvider):
    """gb10-mixed-v2 routes with every FP4 activation quantized per named site."""

    activation_sites = True

    def __init__(self, config):
        super().__init__(config)
        self.sites = {}

    def _site(self, site, width):
        if not isinstance(site, str):
            raise ValueError("FP4 projections need an activation site")
        if site in self.sites:
            raise ValueError("Activation site bound twice: " + site)
        self.sites[site] = int(width)
        return site

    def _quantize(self, site, x):
        raise NotImplementedError

    def _gemm(self, site, packed):
        """compute(activation, rows, out=None), as ``_prepared_projection('fp4')``."""
        return self._prepared_projection("fp4", None, packed)

    def projection(self, route, weight, packed, *, site=None):
        if route != "fp4":
            return super().projection(route, weight, packed)
        site = self._site(site, weight.shape[1])
        compute = self._gemm(site, packed)

        def apply(x):
            return compute(self._quantize(site, x), len(x))

        return apply

    def _prepare_group(self, projections, site):
        prepared, bound = [], False
        for route, weight, packed in projections:
            if route == "fp4":
                if not bound:
                    site, bound = self._site(site, weight.shape[1]), True
                compute = self._gemm(site, packed)
            else:
                compute = self._prepared_projection(route, weight, packed)
            prepared.append((route, compute, weight.shape[0]))
        if not prepared:
            raise ValueError("A projection group cannot be empty")

        def activations(x):
            cache = {}

            def get(route):
                if route not in cache:
                    cache[route] = (
                        self._quantize(site, x) if route == "fp4" else self._activation(route, x)
                    )
                return cache[route]

            return get

        return prepared, activations

    def projection_group(self, projections, *, site=None):
        prepared, activations = self._prepare_group(projections, site)

        def apply(x):
            get = activations(x)
            return tuple(compute(get(route), len(x)) for route, compute, _ in prepared)

        return apply

    def joined_projection_group(self, projections, *, site=None):
        if any(route not in ("fp4", "fp8") for route, _, _ in projections):
            raise ValueError("Joined projection output requires FP4/FP8 routes")
        prepared, activations = self._prepare_group(projections, site)
        total = sum(width for _, _, width in prepared)

        def apply(x):
            rows, get, out, start = len(x), activations(x), None, 0
            for route, compute, width in prepared:
                a = get(route)
                if out is None:
                    out = torch.empty((a[0].shape[0], total), dtype=torch.bfloat16, device=x.device)
                elif out.shape[0] != a[0].shape[0]:
                    raise RuntimeError("Grouped activation padding differs by route")
                compute(a, rows, out=out[:, start : start + width])
                start += width
            return out[:rows]

        return apply

    def _paired_hidden(self, activation, paired, rows, site):
        """Packed MLP hidden activation from the paired gate/up GLU kernel."""
        call, _ = self.paired_kernel.prepare(activation, paired)
        return self._quantize(site, call()[:rows].contiguous())

    def mlp(self, packs, *, long_silu, sites=None):
        if long_silu not in ("torch", "rounded"):
            raise ValueError("Choose an explicit qualified long-input SiLU implementation")
        if not isinstance(sites, tuple) or len(sites) != 2:
            raise ValueError("The MLP needs (input, hidden) activation sites")
        g, u, d = packs
        paired = pair_weights(g, u)
        s_in = self._site(sites[0], g[0].shape[1] * 2)
        s_hidden = self._site(sites[1], d[0].shape[1] * 2)
        gate_c, up_c, down_c = self._gemm(s_in, g), self._gemm(s_in, u), self._gemm(s_hidden, d)

        def compute(matrix):
            m = len(matrix)
            a = self._quantize(s_in, matrix)
            if m <= self.paired_max_rows:
                return down_c(self._paired_hidden(a, paired, m, s_hidden), m)
            gate, up = gate_c(a, m), up_c(a, m)
            hidden = (
                (torch.nn.functional.silu(gate) * up).contiguous()
                if long_silu == "torch"
                else silu_mul(gate, up)
            )
            return down_c(self._quantize(s_hidden, hidden), m)

        return compute


class GB10CalibrationProvider(_SiteProvider):
    """gb10-mixed-v2 arithmetic; records each site's largest per-forward global scale.

    After every forward (all sites seen once) the running maxima are written to the
    file named by ``FISSILE_ACTIVATION_CALIBRATION_OUTPUT``.
    """

    def __init__(self, config):
        super().__init__(config)
        self.output = os.environ.get(CALIBRATION_OUTPUT)
        if not self.output:
            raise ValueError(CALIBRATION_OUTPUT + " must name the calibration record")
        self.maxima, self.seen, self.forwards = {}, set(), 0

    def _quantize(self, site, x):
        a = quantize(x, adaptive=False)
        if site in self.seen:
            raise RuntimeError("Activation site ran twice in one forward: " + site)
        self.seen.add(site)
        if site in self.maxima:
            torch.maximum(self.maxima[site], a[2].reshape(1), out=self.maxima[site])
        else:
            self.maxima[site] = a[2].reshape(1).clone()
        if len(self.seen) == len(self.sites):
            self.seen.clear()
            self.forwards += 1
            self._write()
        return a

    def _write(self):
        keys = sorted(self.maxima)
        values = torch.cat([self.maxima[k] for k in keys]).tolist()
        record = {
            "format": CALIBRATION_FORMAT,
            "forwards": self.forwards,
            "sites": dict(self.sites),
            "max_global_scale": dict(zip(keys, values)),
        }
        tmp = self.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f)
        os.replace(tmp, self.output)


class GB10StaticProvider(_SiteProvider):
    """Calibrated static per-site global scales (resource ``activation_scales``)."""

    def __init__(self, config):
        super().__init__(config)
        self.scales = load_activation_scales(
            config.resources["activation_scales"].path,
            profile=config.profile,
            recipe=config.recipe,
        )
        self.globals = {
            site: torch.tensor([value], dtype=torch.float32, device="cuda")
            for site, value in self.scales.items()
        }

    def _scale(self, site):
        try:
            return self.globals[site]
        except KeyError:
            raise ValueError("No static activation scale for site: " + site) from None

    def _site(self, site, width):
        site = super()._site(site, width)
        self._scale(site)
        return site

    def _quantize(self, site, x):
        return quantize_static(x, self._scale(site))

    def _gemm(self, site, packed):
        qb, sb, gb = packed
        alpha = (self._scale(site) * gb).reshape(1).contiguous()

        def compute(activation, rows, out=None):
            qa, sa, ga = activation
            if out is not None:
                self.fp4_gemm(qa, sa, ga, qb, sb, gb, out=out, alpha=alpha)
                return None
            return self.fp4_gemm(qa, sa, ga, qb, sb, gb, alpha=alpha)[:rows]

        return compute

    def validate_sites(self):
        if set(self.sites) != set(self.scales):
            raise ValueError("Activation scale file does not match the bound sites")
        return len(self.sites)
