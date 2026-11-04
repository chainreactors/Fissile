"""Reversible HF module-layout adapter for an already merged frozen model."""

from contextlib import contextmanager, ExitStack
from types import MethodType


def modules_for(model, profile):
    from .layouts import get_layout

    layout = get_layout(profile.name)
    expected = profile.shapes()
    modules = {}
    for name, leaf in model.named_modules():
        if name == layout.hf_head:
            key = profile.head
        elif name.startswith(layout.hf_prefix):
            key = name[len(layout.hf_prefix) :]
        else:
            continue
        if key in expected:
            modules[key] = leaf
    if set(modules) != set(expected):
        raise ValueError(f"Model modules do not match the {profile.name} layout")
    return modules


@contextmanager
def optimized(model, config):
    import torch
    from fissile.runtime.mixed import MixedRuntime
    from .torch_pointwise import pointwise_fusion
    from .layouts import get_layout

    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Requires model.eval().requires_grad_(False)")
    runtime = MixedRuntime(config)
    from .registry import get_backend

    get_backend("torch").validate(runtime.config)
    modules = modules_for(model, runtime.profile)
    layout = get_layout(runtime.profile.name)
    if any(
        not isinstance(m, torch.nn.Linear) or m.bias is not None
        for m in modules.values()
    ):
        raise ValueError("Use a merged model with no active PEFT layers")
    with ExitStack() as stack:
        norm_stats = stack.enter_context(pointwise_fusion(model, reduce="native-order"))

        def replace(module, fn):
            own, previous = "forward" in vars(module), vars(module).get("forward")

            def restore():
                if own:
                    module.forward = previous
                else:
                    del module.forward

            stack.callback(restore)
            module.forward = MethodType(fn, module)

        fused_keys = {k for mlp in runtime.plan.fused_mlps for k in mlp.keys}
        for key, leaf in modules.items():
            if key in fused_keys:
                continue
            apply = runtime.projection(key, leaf.weight)

            def forward(this, x, _fn=apply):
                if torch.is_grad_enabled() or this.training:
                    raise ValueError("Frozen inference_mode required")
                return _fn(x)

            replace(leaf, forward)
        for group in runtime.plan.fused_mlps:
            prefix = group.prefix
            parts = [modules[k].weight for k in group.keys]
            apply = runtime.mlp(prefix, *parts, long_silu="rounded")
            mlp = model.get_submodule(layout.hf_prefix + prefix)

            def forward(this, x, _fn=apply):
                if torch.is_grad_enabled() or this.training:
                    raise ValueError("Frozen inference_mode required")
                return _fn(x)

            replace(mlp, forward)
        report = runtime.validate_coverage()
        report.update(
            projections=runtime.calls, pointwise=norm_stats, parameters_modified=False
        )
        original = model.forward
        forwards = 0

        def forward(this, *args, **kwargs):
            nonlocal forwards
            ids = kwargs.get("input_ids", args[0] if args else None)
            if (
                ids is None
                or ids.ndim != 2
                or ids.shape[0] != 1
                or not 1 < ids.shape[1] <= runtime.profile.max_context
            ):
                raise ValueError("Single complete prefill input required")
            if (
                kwargs.get("use_cache") is not False
                or kwargs.get("past_key_values") is not None
            ):
                raise ValueError("Scoring requires use_cache=False and no past state")
            if kwargs.get("logits_to_keep") != 1:
                raise ValueError("Scoring requires logits_to_keep=1")
            result = original(*args, **kwargs)
            forwards += 1
            runtime.validate_forward(forwards)
            return result

        replace(model, forward)
        yield report


@contextmanager
def open_session(config, model):
    """Common backend contract; caller owns the loaded, frozen model."""
    import torch
    from .contracts import Score

    class Session:
        def score(self, input_ids, candidate_token_ids):
            logits = model(
                input_ids=torch.tensor([input_ids], device=model.device),
                use_cache=False,
                logits_to_keep=1,
            ).logits
            values = (
                logits[0, -1]
                .float()
                .log_softmax(-1)[list(candidate_token_ids)]
                .tolist()
            )
            return Score(tuple(values), len(input_ids))

    with torch.inference_mode(), optimized(model, config):
        yield Session()
