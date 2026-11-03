"""vLLM layout adapter. Shared numerical execution lives in runtime.mixed."""

import json
import os
from types import MethodType
import torch
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as Original,
)
from fissile.runtime.mixed import MixedRuntime


class PackedMethod:
    """vLLM linear dispatch interface, intentionally not UnquantizedLinearMethod."""

    def __init__(self, function):
        self.function = function

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None, **kwargs):
        if bias is not None:
            raise ValueError("Only bias-free selected dense projections are supported")
        return self.function(x)


class Qwen3_5ForConditionalGeneration(Original):
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if hasattr(self, "_fissile_runtime"):
            raise RuntimeError("Online weight reload is not qualified")
        runtime = MixedRuntime(os.environ["FISSILE_CONFIG"])
        from fissile.backends.registry import get_backend

        get_backend("vllm").validate(runtime.config)
        out = runtime.config.report_dir
        out.mkdir(parents=True, exist_ok=False)
        language = self.language_model
        layers = language.model.layers
        if len(layers) != len(runtime.profile.mlps):
            raise ValueError("Expected all model layers")
        for i, layer in enumerate(layers):
            prefix = f"layers.{i}"
            gate_width = runtime.metadata[prefix + ".mlp.gate_proj"]["shape"][0]
            gate, up = layer.mlp.gate_up_proj.weight.split(gate_width, dim=0)
            mlp = runtime.mlp(
                prefix + ".mlp", gate, up, layer.mlp.down_proj.weight, long_silu="torch"
            )
            layer.mlp.forward = MethodType(lambda this, x, _fn=mlp: _fn(x), layer.mlp)
            if hasattr(layer, "linear_attn"):
                attn = layer.linear_attn
                keys = [prefix + ".linear_attn." + k for k in ("in_proj_qkv", "in_proj_z")]
                widths = [runtime.metadata[k]["shape"][0] for k in keys]
                attn.in_proj_qkvz.quant_method = PackedMethod(
                    runtime.joined_projection_group(
                        keys, attn.in_proj_qkvz.weight.split(widths, dim=0), _concatenate
                    )
                )
                attn.out_proj.quant_method = PackedMethod(
                    runtime.projection(prefix + ".linear_attn.out_proj", attn.out_proj.weight)
                )
            else:
                attn = layer.self_attn
                keys = [prefix + ".self_attn." + n for n in ("q_proj", "k_proj", "v_proj")]
                widths = [runtime.metadata[k]["shape"][0] for k in keys]
                attn.qkv_proj.quant_method = PackedMethod(
                    runtime.joined_projection_group(
                        keys, attn.qkv_proj.weight.split(widths, dim=0), _concatenate
                    )
                )
                attn.o_proj.quant_method = PackedMethod(
                    runtime.projection(prefix + ".self_attn.o_proj", attn.o_proj.weight)
                )
        language.lm_head.quant_method = PackedMethod(
            runtime.projection("lm_head", language.lm_head.weight)
        )
        self._fissile_runtime = runtime
        report = runtime.validate_coverage()
        report["engine"] = "vllm"
        with (out / "quant-install.json").open("x") as f:
            json.dump(report, f, indent=2)
        return loaded

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        # Scoring requests (at least two tokens) are checked at the session
        # boundary; vLLM's startup warmup also runs a one-token forward.
        tokens = positions.shape[-1]
        if not 1 <= tokens <= self._fissile_runtime.profile.max_context:
            raise ValueError(f"Only complete prefill up to 16384 tokens is qualified; got {tokens}")
        before = dict(self._fissile_runtime.calls)
        result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        self._fissile_runtime.validate_counts(before, head=0, body=1)
        return result

    def compute_logits(self, hidden_states):
        if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
            raise ValueError("Only final-token, single-request readout is qualified")
        before = dict(self._fissile_runtime.calls)
        logits = super().compute_logits(hidden_states)
        self._fissile_runtime.validate_counts(before, head=1, body=0)
        return logits


def _concatenate(outputs):
    return torch.cat(outputs, dim=-1)
