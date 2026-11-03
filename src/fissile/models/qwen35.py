"""Single explicit model capability profile, not automatic model generalization."""

from .spec import MLP, ModelProfile, Projection

PROFILE = "qwen3.5-dense-9b"
LAYERS = 32
HIDDEN = 4096
INTERMEDIATE = 12288
VOCAB = 248320


def projection_shapes():
    shapes = {"lm_head": [VOCAB, HIDDEN]}
    for i in range(LAYERS):
        base = f"layers.{i}."
        for name, shape in {
            "gate_proj": [INTERMEDIATE, HIDDEN],
            "up_proj": [INTERMEDIATE, HIDDEN],
            "down_proj": [HIDDEN, INTERMEDIATE],
        }.items():
            shapes[base + "mlp." + name] = shape
        if i % 4 == 3:
            for name, shape in {
                "q_proj": [8192, HIDDEN],
                "k_proj": [1024, HIDDEN],
                "v_proj": [1024, HIDDEN],
                "o_proj": [HIDDEN, HIDDEN],
            }.items():
                shapes[base + "self_attn." + name] = shape
        else:
            for name, shape in {
                "in_proj_qkv": [8192, HIDDEN],
                "in_proj_z": [4096, HIDDEN],
                "out_proj": [HIDDEN, HIDDEN],
            }.items():
                shapes[base + "linear_attn." + name] = shape
    return shapes


def canonical_key(key):
    if key == "base_model.model.lm_head":
        return "lm_head"
    prefix = "base_model.model.model.language_model."
    return key[len(prefix) :] if key.startswith(prefix) else key


def route(key):
    """Compatibility helper; precision selection lives in recipes."""
    from fissile.recipes import get_recipe, DEFAULT_RECIPE

    return get_recipe(DEFAULT_RECIPE).route(
        next(p for p in SPEC.projections if p.key == key)
    )


SPEC = ModelProfile(
    name=PROFILE,
    projections=tuple(
        Projection(
            key,
            tuple(shape),
            (
                "head"
                if key == "lm_head"
                else (
                    "attention_kv"
                    if key.endswith((".self_attn.k_proj", ".self_attn.v_proj"))
                    else "body"
                )
            ),
        )
        for key, shape in projection_shapes().items()
    ),
    mlps=tuple(
        MLP(
            f"layers.{i}.mlp",
            *(f"layers.{i}.mlp.{n}_proj" for n in ("gate", "up", "down")),
        )
        for i in range(LAYERS)
    ),
    vocab_size=VOCAB,
    max_context=16384,
)
