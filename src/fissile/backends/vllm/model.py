"""vLLM adapter. Module layout lives in backends.layouts; math in runtime.mixed."""

import json
import os
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _Qwen3_5,
)
from fissile.backends.layouts import PackedMethod, get_layout, install_fused, module_at
from fissile.runtime.mixed import MixedRuntime


class FissileModel:
    """Mixin for a vLLM conditional-generation class with a ``language_model``."""

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
        install_fused(runtime, language, engine="vllm")
        head = module_at(language, get_layout(runtime.profile.name).head)
        head.quant_method = PackedMethod(runtime.projection(runtime.profile.head, head.weight))
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
            raise ValueError(
                f"Only complete prefill up to {self._fissile_runtime.profile.max_context} "
                f"tokens is qualified; got {tokens}"
            )
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


class Qwen3_5ForConditionalGeneration(FissileModel, _Qwen3_5):
    pass


# Engine architecture name -> Fissile model class, registered with vLLM at start.
MODELS = {cls.__name__: cls for cls in (Qwen3_5ForConditionalGeneration,)}
