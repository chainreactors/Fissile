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

## Other extension points

| Extension | Interface |
|---|---|
| Model | `ModelProfile` + `register_profile` |
| Precision recipe | `Recipe` + `register_recipe` |
| Kernel provider | `ProviderSpec` + `register_provider`, implementing `KernelProvider` |

A provider declares routes, packed routes, fusions, alignment, activation dtype,
CUDA capability and resources. Its factory implements `matrix`, `projection`
and `mlp`, and optionally `projection_group`, `joined_projection_group` and
`candidate_readout`. A config v2 file supplies exactly the resources its
provider declares.

`tests/test_extensions.py` shows an external adapter, model profile, recipe and
provider registered without modifying Fissile.
