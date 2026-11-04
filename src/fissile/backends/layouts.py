"""Engine module layouts: one description per model profile, shared by every adapter.

A layout maps canonical projection keys to engine modules. Fused-projection
engines (SGLang, vLLM) store several projections in one linear module whose
output rows concatenate them. Transformers keeps one module per projection
under a language-model prefix.
"""

from dataclasses import dataclass, field
from types import MethodType


@dataclass(frozen=True)
class FusedLinear:
    """One engine linear module; its output rows concatenate ``keys`` in order.

    ``path`` is relative to a decoder layer and ``keys`` to the layer's
    canonical prefix (``layers.{i}``).
    """

    path: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    profile: str
    # Fused-projection engines, relative to the engine's language model.
    layers: str
    head: str
    # Gated MLP module relative to a decoder layer; replaced as a whole when the
    # recipe fuses the MLP, otherwise its linears are bound individually.
    mlp: str
    linears: tuple[FusedLinear, ...]
    # Transformers: module-name prefix of decoder projections and head module.
    hf_prefix: str
    hf_head: str
    # Per-engine module path overrides, e.g. {"sglang": {"self_attn.qkv_proj": "qkv_proj"}}.
    paths: dict = field(default_factory=dict)
    layer_key: str = "layers.{}"


_LAYOUTS = {}


def register_layout(layout):
    if layout.profile in _LAYOUTS:
        raise ValueError("Layout already registered for profile: " + layout.profile)
    _LAYOUTS[layout.profile] = layout


def get_layout(profile):
    try:
        return _LAYOUTS[profile]
    except KeyError:
        raise ValueError("No engine layout for model profile: " + str(profile)) from None


def module_at(owner, path):
    for part in path.split("."):
        owner = getattr(owner, part)
    return owner


class PackedMethod:
    """Linear dispatch interface for fused engines, not an unquantized method."""

    def __init__(self, function):
        self.function = function

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None, **kwargs):
        if bias is not None:
            raise ValueError("Only bias-free selected dense projections are supported")
        return self.function(x)


def _concatenate(outputs):
    import torch

    return torch.cat(outputs, dim=-1)


def install_fused(runtime, model, *, engine, long_silu="torch"):
    """Bind every body projection of ``runtime.profile`` in a fused engine model.

    ``model`` is the engine's language model. Gated MLPs fused by the recipe
    replace the MLP module's forward; other projections become the quant
    method of their linear module. The output head is left to the adapter.
    Returns the engine's decoder layers.
    """
    layout = get_layout(runtime.profile.name)
    paths = layout.paths.get(engine, {})
    shapes = runtime.profile.shapes()
    fused = {m.prefix: m for m in runtime.plan.fused_mlps}
    layers = module_at(model, layout.layers)
    for i, layer in enumerate(layers):
        key = layout.layer_key.format(i)
        done = set()
        group = fused.get(f"{key}.{layout.mlp}")
        if group is not None:
            relative = {linear.keys: linear.path for linear in layout.linears}

            def linear_for(*canonical):
                path = relative[tuple(k[len(key) + 1 :] for k in canonical)]
                return module_at(layer, paths.get(path, path))

            mlp = module_at(layer, paths.get(layout.mlp, layout.mlp))
            gate_up = linear_for(group.gate, group.up)
            down = linear_for(group.down)
            gate, up = gate_up.weight.split(runtime.metadata[group.gate]["shape"][0], dim=0)
            fn = runtime.mlp(group.prefix, gate, up, down.weight, long_silu=long_silu)
            mlp.forward = MethodType(lambda this, x, *args, _fn=fn, **kwargs: _fn(x), mlp)
            done.update(group.keys)
        for linear in layout.linears:
            keys = tuple(f"{key}.{k}" for k in linear.keys)
            present = [k in shapes for k in keys]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"Profile covers only part of {key}.{linear.path}")
            if set(keys) <= done:
                continue
            if done.intersection(keys):
                raise ValueError(f"{key}.{linear.path} mixes fused and unfused projections")
            module = module_at(layer, paths.get(linear.path, linear.path))
            if len(keys) == 1:
                fn = runtime.projection(keys[0], module.weight)
            else:
                widths = [runtime.metadata[k]["shape"][0] for k in keys]
                fn = runtime.joined_projection_group(
                    keys, module.weight.split(widths, dim=0), _concatenate
                )
            module.quant_method = PackedMethod(fn)
            done.update(keys)
        if not done:
            raise ValueError(f"Engine layer {i} has no projections in the profile")
    return layers


QWEN35_DENSE = Layout(
    profile="qwen3.5-dense-9b",
    layers="model.layers",
    head="lm_head",
    mlp="mlp",
    linears=(
        FusedLinear("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
        FusedLinear("mlp.down_proj", ("mlp.down_proj",)),
        FusedLinear(
            "linear_attn.in_proj_qkvz", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")
        ),
        FusedLinear("linear_attn.out_proj", ("linear_attn.out_proj",)),
        FusedLinear(
            "self_attn.qkv_proj",
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ),
        FusedLinear("self_attn.o_proj", ("self_attn.o_proj",)),
    ),
    hf_prefix="model.language_model.",
    hf_head="lm_head",
    # SGLang's full-attention decoder layer owns its projections directly.
    paths={"sglang": {"self_attn.qkv_proj": "qkv_proj", "self_attn.o_proj": "o_proj"}},
)
register_layout(QWEN35_DENSE)
