# Extending Fissile

## Scoring API

```python
from fissile.backends.registry import open_backend

# SGLang: model is a local directory; the adapter owns the engine.
# Torch: model is an already loaded, merged, frozen model.
with open_backend("sglang", "config.json", model) as session:
    result = session.score(input_ids, candidate_token_ids)
    print(result.token_logprobs, result.prompt_tokens)
```

Sessions validate token IDs, prompt length, distinct candidates and the result
contract. Closing the context shuts down an engine the adapter created, or
restores the methods of a caller-owned model.

## Add an engine

An engine adapter is a context manager receiving `(config_path, model)` and
yielding an object with:

```python
def score(input_ids, candidate_token_ids) -> Score:
    ...  # Score(tuple_of_candidate_logprobs, prompt_tokens)
```

Register it with the combinations it supports:

```python
from fissile.backends.registry import BackendSpec, register_backend

register_backend(BackendSpec(
    name="my-engine",
    factory="my_package.adapter:open_session",
    combinations=frozenset({(
        "qwen3.5-dense-9b",
        "mixed-fp4-fp8-w4a16-v1",
        "gb10-mixed-v1",
    )}),
))
```

The adapter loads the config, maps the engine's merged or split weights to
canonical projection keys and binds the shared runtime:

| Runtime call | Use |
|---|---|
| `MixedRuntime.projection(key, weight)` | One projection |
| `MixedRuntime.projection_group(keys, weights)` | Same-input projections; one output per key, shared activation pack |
| `MixedRuntime.joined_projection_group(keys, weights, concatenate)` | Same-input projections consumed concatenated; one `[..., sum(widths)]` output |
| `MixedRuntime.mlp(prefix, gate, up, down, long_silu=...)` | Paired gate/up MLP |
| `MixedRuntime.candidate_readout(key, weight)` | Output head for requested candidate tokens only |
| `validate_coverage()` / `validate_forward(count)` | Check that every projection is installed and executed once per forward |

Every binding verifies the effective weight against the manifest hash. Import
the registration module before calling `open_backend`, including in spawned
worker processes.

Engines that store several projections in one linear module (SGLang, vLLM) call
`fissile.backends.layouts.install_fused(runtime, language_model, engine="name")`,
which binds every body projection of the profile from its layout; the adapter
binds the output head (`runtime.profile.head`).

## Add a model

1. **Profile** (`models/`): canonical projection keys, shapes, roles (`body`,
   `attention_kv`, `head`) and gated-MLP groups, registered with
   `register_profile`.
2. **Engine layout** (`backends/layouts.py`): where engines keep those
   projections. One layout serves SGLang, vLLM and PyTorch.

   ```python
   from fissile.backends.layouts import FusedLinear, Layout, register_layout

   register_layout(Layout(
       profile="my-model",
       layers="model.layers",   # decoder layers in the engine's language model
       head="lm_head",          # output head in the engine's language model
       mlp="mlp",               # gated MLP module in a decoder layer
       linears=(
           FusedLinear("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
           FusedLinear("mlp.down_proj", ("mlp.down_proj",)),
           FusedLinear("self_attn.qkv_proj",
                       ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
           FusedLinear("self_attn.o_proj", ("self_attn.o_proj",)),
       ),
       hf_prefix="model.",      # Transformers module-name prefix
       hf_head="lm_head",
       paths={"sglang": {}},    # per-engine module path overrides
   ))
   ```

3. **Engine model class**: combine the adapter mixin with the engine's model
   class and list it in `EntryClass` (`backends/sglang/model.py`) or `MODELS`
   (`backends/vllm/model.py`).

   ```python
   class MyModelForCausalLM(FissileModel, EngineMyModelForCausalLM):
       pass
   ```

4. **Support and weights**: register a recipe and provider whose `profiles`
   include the model, add the backend combinations, then prepare packs with
   `prepare_projection` or import existing ones with
   `fissile import-packs --profile my-model --recipe my-recipe`.

## Other extension points

| Extension | Interface |
|---|---|
| Model | `ModelProfile` + `register_profile` |
| Engine module layout | `Layout` + `register_layout` |
| Precision recipe | `Recipe` + `register_recipe` |
| Kernel provider | `ProviderSpec` + `register_provider`, implementing `KernelProvider` |

A provider declares routes, packed routes, fusions, alignment, activation dtype,
CUDA capability and resources. Its factory implements `matrix`, `projection`
and `mlp`, and optionally `projection_group`, `joined_projection_group` and
`candidate_readout`. A config v2 file supplies exactly the resources its
provider declares.

`tests/test_extensions.py` shows an external adapter, model profile, recipe and
provider registered without modifying Fissile.
