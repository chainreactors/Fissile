"""SGLang adapter. Module layout lives in backends.layouts; math in runtime.mixed."""

import json
import os
from types import MethodType
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration as _Qwen3_5
from fissile.backends.layouts import PackedMethod, get_layout, install_fused, module_at
from fissile.runtime.mixed import MixedRuntime
from fissile.runtime.graph import VALID_ROWS


class FissileModel:
    """Mixin for an SGLang causal-LM class; combine it with the engine's model class."""

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if hasattr(self, "_fissile_runtime"):
            raise RuntimeError("Online weight reload is not qualified")
        runtime = MixedRuntime(os.environ["FISSILE_CONFIG"])
        from fissile.backends.registry import get_backend

        get_backend("sglang").validate(runtime.config)
        out = runtime.config.report_dir
        out.mkdir(parents=True, exist_ok=False)
        layers = install_fused(runtime, self, engine="sglang")
        # Do not let SGLang's derived BF16 input-projection caches bypass the
        # installed projections.
        for layer in layers:
            for module in layer.modules():
                if hasattr(module, "finalize_fused_in_proj"):
                    module._fused_in_proj_weight = None
                    module.finalize_fused_in_proj = MethodType(lambda this: None, module)
        head_key = runtime.profile.head
        head_module = module_at(self, get_layout(runtime.profile.name).head)
        readout = os.environ.get("FISSILE_READOUT", "full")
        if readout == "full":
            head = runtime.projection(head_key, head_module.weight)
        elif readout == "candidates":
            candidate_head = runtime.candidate_readout(head_key, head_module.weight)

            def head(x, _model=self):
                if _model._fissile_candidates is None:
                    raise ValueError("Candidate readout requires label token IDs")
                return candidate_head(x, _model._fissile_candidates)

        else:
            raise ValueError("Unknown FISSILE_READOUT: " + readout)
        head_module.quant_method = PackedMethod(head)
        self._fissile_readout = readout
        self._fissile_candidates = None
        self._fissile_runtime = runtime
        self._fissile_forwards = 0
        VALID_ROWS.allocate(head_module.weight.device)
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
                "Only singleton complete prefill scoring up to "
                f"{self._fissile_runtime.profile.max_context} tokens is qualified; "
                f"shape={tuple(input_ids.shape)}, batch_size={forward_batch.batch_size}, "
                f"extend_lengths={getattr(forward_batch, 'extend_seq_lens_cpu', None)}"
            )
        requested = getattr(forward_batch, "token_ids_logprobs", None)
        self._fissile_candidates = requested[0] if requested else None
        # Under prefill graph replay input_ids are padded to a bucket; the live
        # length comes from the batch metadata.
        live = getattr(forward_batch, "extend_seq_lens_cpu", None)
        live = int(live[0]) if live is not None and len(live) else len(input_ids)
        if not 1 < live <= len(input_ids):
            raise ValueError(f"Live token count {live} outside input of {len(input_ids)}")
        VALID_ROWS.publish(live)
        before = dict(self._fissile_runtime.calls)
        try:
            result = super().forward(
                input_ids, positions, forward_batch, *args, **kwargs
            )
        finally:
            self._fissile_candidates = None
        self._fissile_forwards += 1
        mode = self._fissile_runtime.validate_pass(before)
        if self._fissile_forwards == 1:
            out = self._fissile_runtime.config.report_dir
            with (out / "first-forward-coverage.json").open("x") as f:
                json.dump(
                    {
                        "complete": True,
                        "all_projections_executed": True,
                        "mode": mode,
                        "expected_projections": len(
                            self._fissile_runtime.profile.projections
                        ),
                        "calls": self._fissile_runtime.calls,
                    },
                    f,
                    indent=2,
                )
        return result


class Qwen3_5ForConditionalGeneration(FissileModel, _Qwen3_5):
    pass


EntryClass = [Qwen3_5ForConditionalGeneration]
