"""Model-specific norm/MLP overlay with reversible installation."""

from contextlib import contextmanager
from types import MethodType
import torch
from fissile.kernels.pointwise import rms_norm, silu_mul


@contextmanager
def pointwise_fusion(model, *, reduce="torch-reduce"):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Requires frozen evaluation model")
    saved = []
    stats = {
        "enabled": True,
        "reduce": reduce,
        "norms": 0,
        "gated_norms": 0,
        "mlps": 0,
        "norm_calls": 0,
        "gated_norm_calls": 0,
        "mlp_calls": 0,
        "fallback_calls": 0,
    }

    def eligible(module, x):
        return (
            not module.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype == torch.bfloat16
        )

    try:
        for name, module in list(model.named_modules()):
            if ".language_model." not in name:
                continue
            kind = type(module).__name__
            original = module.forward
            if kind == "Qwen3_5RMSNorm":
                stats["norms"] += 1

                def forward(self, x, _original=original):
                    if not stats["enabled"]:
                        return _original(x)
                    if not eligible(self, x):
                        stats["fallback_calls"] += 1
                        return _original(x)
                    stats["norm_calls"] += 1
                    return rms_norm(x, self.weight, self.eps, reduce=reduce)

            elif kind == "Qwen3_5RMSNormGated":
                stats["gated_norms"] += 1

                def forward(self, hidden_states, gate, _original=original):
                    if not stats["enabled"]:
                        return _original(hidden_states, gate)
                    if not eligible(self, hidden_states):
                        stats["fallback_calls"] += 1
                        return _original(hidden_states, gate)
                    stats["gated_norm_calls"] += 1
                    return rms_norm(
                        hidden_states,
                        self.weight,
                        self.variance_epsilon,
                        gate,
                        reduce=reduce,
                    )

            elif kind == "Qwen3_5MLP":
                stats["mlps"] += 1

                def forward(self, x, _original=original):
                    if not stats["enabled"]:
                        return _original(x)
                    if not eligible(self, x):
                        stats["fallback_calls"] += 1
                        return _original(x)
                    stats["mlp_calls"] += 1
                    return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))

            else:
                continue
            saved.append(
                (module, "forward" in vars(module), vars(module).get("forward"))
            )
            module.forward = MethodType(forward, module)
        if (stats["norms"], stats["gated_norms"], stats["mlps"]) != (81, 24, 32):
            raise ValueError(f"Wrong selected-model topology: {stats}")
        yield stats
    finally:
        for module, own, original in reversed(saved):
            if own:
                module.forward = original
            else:
                del module.forward
