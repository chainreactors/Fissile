"""SGLang layout adapter. Shared numerical execution lives in runtime.mixed."""

import json
import os
from types import MethodType
import torch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration as Original
from fissile.runtime.mixed import MixedRuntime


class PackedMethod:
    """SGLang linear dispatch interface, intentionally not UnquantizedLinearMethod."""

    def __init__(self, function):
        self.function = function

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None, **kwargs):
        if bias is not None:
            raise ValueError("Only bias-free selected dense projections are supported")
        return self.function(x)


def _concatenate(outputs):
    return torch.cat(outputs, dim=-1)


class Qwen3_5ForConditionalGeneration(Original):
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if hasattr(self, "_fissile_runtime"):
            raise RuntimeError("Online weight reload is not qualified")
        runtime = MixedRuntime(os.environ["FISSILE_CONFIG"])
        from fissile.backends.registry import get_backend

        get_backend("sglang").validate(runtime.config)
        out = runtime.config.report_dir
        out.mkdir(parents=True, exist_ok=False)
        if len(self.model.layers) != len(runtime.profile.mlps):
            raise ValueError("Expected all model layers")
        for i, layer in enumerate(self.model.layers):
            prefix = f"layers.{i}"
            gate_width = runtime.metadata[prefix + ".mlp.gate_proj"]["shape"][0]
            gate, up = layer.mlp.gate_up_proj.weight.split(gate_width, dim=0)
            mlp = runtime.mlp(
                prefix + ".mlp", gate, up, layer.mlp.down_proj.weight, long_silu="torch"
            )

            def forward(this, x, forward_batch=None, _fn=mlp):
                return _fn(x)

            layer.mlp.forward = MethodType(forward, layer.mlp)
            if hasattr(layer, "linear_attn"):
                attn = layer.linear_attn
                widths = [
                    runtime.metadata[prefix + ".linear_attn." + k]["shape"][0]
                    for k in ("in_proj_qkv", "in_proj_z")
                ]
                qkv, z = attn.in_proj_qkvz.weight.split(widths, dim=0)
                attn.in_proj_qkvz.quant_method = PackedMethod(
                    runtime.joined_projection_group(
                        [
                            prefix + ".linear_attn.in_proj_qkv",
                            prefix + ".linear_attn.in_proj_z",
                        ],
                        [qkv, z],
                        _concatenate,
                    )
                )
                attn.out_proj.quant_method = PackedMethod(
                    runtime.projection(
                        prefix + ".linear_attn.out_proj", attn.out_proj.weight
                    )
                )
                # Do not let a derived BF16 cache bypass mixed projections.
                attn._fused_in_proj_weight = None
                attn.finalize_fused_in_proj = MethodType(lambda this: None, attn)
            else:
                keys = [
                    prefix + ".self_attn." + n for n in ("q_proj", "k_proj", "v_proj")
                ]
                widths = [runtime.metadata[k]["shape"][0] for k in keys]
                parts = layer.qkv_proj.weight.split(widths, dim=0)
                layer.qkv_proj.quant_method = PackedMethod(
                    runtime.joined_projection_group(keys, parts, _concatenate)
                )
                layer.o_proj.quant_method = PackedMethod(
                    runtime.projection(
                        prefix + ".self_attn.o_proj", layer.o_proj.weight
                    )
                )
        readout = os.environ.get("FISSILE_READOUT", "full")
        if readout == "full":
            head = runtime.projection("lm_head", self.lm_head.weight)
        elif readout == "candidates":
            candidate_head = runtime.candidate_readout("lm_head", self.lm_head.weight)

            def head(x, _model=self):
                if _model._fissile_candidates is None:
                    raise ValueError("Candidate readout requires label token IDs")
                return candidate_head(x, _model._fissile_candidates)

        else:
            raise ValueError("Unknown FISSILE_READOUT: " + readout)
        self.lm_head.quant_method = PackedMethod(head)
        self._fissile_readout = readout
        self._fissile_candidates = None
        self._fissile_runtime = runtime
        self._fissile_forwards = 0
        report = runtime.validate_coverage()
        report["engine"] = "sglang"
        report["readout"] = readout
        with (out / "quant-install.json").open("x") as f:
            json.dump(report, f, indent=2)
        return loaded

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        if (
            input_ids.ndim != 1
            or not 1 < len(input_ids) <= self._fissile_runtime.profile.max_context
            or forward_batch.batch_size != 1
        ):
            raise ValueError(
                "Only singleton complete prefill scoring up to 16384 tokens is qualified; "
                f"shape={tuple(input_ids.shape)}, batch_size={forward_batch.batch_size}, "
                f"extend_lengths={getattr(forward_batch, 'extend_seq_lens_cpu', None)}"
            )
        requested = getattr(forward_batch, "token_ids_logprobs", None)
        self._fissile_candidates = requested[0] if requested else None
        try:
            result = super().forward(
                input_ids, positions, forward_batch, *args, **kwargs
            )
        finally:
            self._fissile_candidates = None
        self._fissile_forwards += 1
        self._fissile_runtime.validate_forward(self._fissile_forwards)
        if self._fissile_forwards == 1:
            out = self._fissile_runtime.config.report_dir
            with (out / "first-forward-coverage.json").open("x") as f:
                json.dump(
                    {
                        "complete": True,
                        "all_projections_executed": True,
                        "expected_projections": len(
                            self._fissile_runtime.profile.projections
                        ),
                        "calls": self._fissile_runtime.calls,
                    },
                    f,
                    indent=2,
                )
        return result


EntryClass = Qwen3_5ForConditionalGeneration
