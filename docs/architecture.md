# Architecture

## Components

```text
examples -> backend API -> engine adapter -> runtime -> provider -> kernels
                                          -> models / recipes / config
quantization -> kernels
```

| Component | Responsibility |
|---|---|
| `models/` | Model profiles: canonical projections, shapes, roles, MLP groups, vocabulary and context |
| `recipes/` | Role-to-precision routing, fusion policy and preparation contract |
| `providers/` | Kernel providers: capabilities, resources and numerical implementations |
| `runtime/` | Weight identity checks, canonical binding and execution coverage |
| `backends/` | Engine adapters: engine layout, resource lifecycle and scoring sessions |
| `kernels/` | Tensor math: NVFP4/FP8 packing, GEMMs, paired GLU, W4A16 head |
| `quantization/` | Offline GPTQ projection preparation |

A model profile and a recipe form an `ExecutionPlan`. The selected provider
validates the plan's routes, fusions, dtype and shape alignment against the
device capability. Each backend declares the exact `(profile, recipe,
provider)` combinations it supports; `open_backend` checks the combination
before importing engine code. Core imports and backend discovery are CPU-only.

## Kernel providers

| Provider | W4A4 GEMM | Resources |
|---|---|---|
| `gb10-mixed-v1` | cuBLAS (`torch._scaled_mm_v2`) | TensorFold library, SiLU table |
| `gb10-mixed-v2` | Pinned CUTLASS SM121 kernel | TensorFold library, SiLU table, FP4 GEMM library |
| `sm120-mixed-v1` | cuBLAS | TensorFold library, SiLU table |

All providers implement the same recipe: NVFP4 W4A4 body projections, FP8
rowwise full-attention K/V, a TensorFold NVFP4 W4A16 output head and paired
gate/up GLU with a BF16 SiLU lookup epilogue.

## Engine adapters

**SGLang.** An external model package loads the merged checkpoint, verifies
every projection against the manifest, then installs Fissile projection methods
and paired MLPs. Same-input projections share one activation pack per call and
write into column slices of one output buffer. SGLang keeps its attention,
normalization and recurrent-state kernels. The scoring engine runs
single-request complete prefill with batching, chunked prefill, radix caching
and, by default, CUDA graphs disabled. The paired GLU kernel serves inputs up to
1,536 rows; longer inputs use separate gate/up GEMMs and Torch SiLU. The
`sglang-candidates` backend computes the output head only for the requested
candidate tokens.

Prefill CUDA graphs are off by default. `open_backend(..., prefill_cuda_graph=True)`
or `--prefill-cuda-graph` in `examples/score_sglang.py` replays the transformer
body from SGLang's breakable prefill graph for inputs up to 4,096 tokens; longer
inputs run eagerly. [Measurements](../bench/README.md#sglang-prefill-cuda-graph).

**vLLM.** The adapter registers a Qwen3.5 model class that installs the same
Fissile projections and paired MLPs after vLLM loads the checkpoint. The engine
core runs in-process with the V1 model runner, one sequence at a time, eager
execution and no prefix caching; a full prompt always fits one scheduling step,
so prefill is never chunked. vLLM samples at least one token, so scoring
requests one transport token and returns the full-vocabulary raw log
probabilities of the requested candidates (at most 128) at that position; the
transport token is discarded. Each forward must run every body projection once
and each logits call must run the output head once on one row. Attention,
linear-attention and normalization kernels are vLLM's, so raw scores differ
from the SGLang backend.

**PyTorch.** A reversible overlay replaces methods on an already loaded,
merged, frozen model and restores them in reverse order on exit, including on
exceptions. It uses the rounded pointwise path and a final-token output head.

## Registration

Models, recipes, providers and backends are registered explicitly in Python;
duplicate names are rejected. See the [extension guide](extensions.md).
