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
| `backends/layouts.py` | Engine module layout of each model profile, shared by all adapters |
| `backends/` | Engine adapters: resource lifecycle and scoring sessions |
| `kernels/` | Tensor math: NVFP4/FP8 packing, GEMMs, paired GLU, W4A16 head |
| `quantization/` | Calibration input moments and GPTQ pack preparation (`fissile quantize`) |

A model profile and a recipe form an `ExecutionPlan`. The selected provider
validates the plan's routes, fusions, dtype and shape alignment against the
device capability. Each backend declares the exact `(profile, recipe,
provider)` combinations it supports; `open_backend` checks the combination
before importing engine code. Core imports and backend discovery are CPU-only.

## Kernel providers

| Provider | W4A4 GEMM | Paired MLP rows | Resources |
|---|---|---:|---|
| `gb10-mixed-v1` | cuBLAS (`torch._scaled_mm_v2`) | 1,536 | TensorFold library, SiLU table |
| `gb10-mixed-v2` | Pinned CUTLASS SM121 kernel | 1,536 | TensorFold library, SiLU table, FP4 GEMM library |
| `sm120-mixed-v1` | cuBLAS | 1,536 | TensorFold library, SiLU table |
| `gb10-mixed-v2-calibration` | As `gb10-mixed-v2` | 1,536 | As `gb10-mixed-v2` |
| `gb10-mixed-v2-static` | As `gb10-mixed-v2` | 1,536 | As `gb10-mixed-v2`, activation scales |

Each provider declares `paired_mlp_max_rows`: inputs up to that many rows use
the paired gate/up kernel, longer inputs separate gate/up GEMMs.

All providers implement the same recipe: NVFP4 W4A4 body projections, FP8
rowwise full-attention K/V, a TensorFold NVFP4 W4A16 output head and paired
gate/up GLU with a BF16 SiLU lookup epilogue.

### Static activation scales

Off by default. `open_backend(..., activation_scales="scales.json")` or
`--activation-scales scales.json` in `examples/score_sglang.py` runs a
`gb10-mixed-v2` configuration as `gb10-mixed-v2-static` on the SGLang backends:
every W4A4 activation uses a fixed global scale per activation site (a
projection's or projection group's input, the MLP input and the MLP hidden
activation, named by the first consuming projection key) instead of the
per-forward tensor-wide amax. Each activation is packed in one launch, GEMM
alphas are computed once, and the paired GLU writes the packed hidden activation
from its epilogue, bitwise equal to packing its BF16 output. The install report
records the provider and the scale file's sha256. Not combined with prefill CUDA
graphs.

Scales come from the same configuration's calibration variant, which keeps the
`gb10-mixed-v2` arithmetic and records each site's largest per-forward scale:

```bash
python scripts/calibrate_activation_scales.py --config config.json \
    --model MODEL --inputs suite.jsonl --output scales.json   # every 29th row
```

Activations larger than the calibrated range saturate, so scores depend on how
representative the calibration rows are.

GB10, full suite (29,182 questions), scales from every 29th question:

| Provider | Accuracy | Mean latency | Queued req/s¹ |
|---|---:|---:|---:|
| `gb10-mixed-v2` | 75.18% | 56.41 ms | 17.18 |
| `gb10-mixed-v2-static` | 75.01% | 52.62 ms | 18.62 |

On the 28,175 questions outside the calibration rows: 74.98% vs 75.15%
(McNemar z = 1.19). ¹ 300 inputs of at most 1,536 tokens submitted at once,
candidate readout; short inputs gain less because they are bound by kernel
launch overhead. [Measurements](../bench/README.md#sglang-static-activation-scales).

## Engine adapters

**SGLang.** An external model package loads the merged checkpoint, verifies
every projection against the manifest, then installs JevInfer projection methods
and paired MLPs. Same-input projections share one activation pack per call and
write into column slices of one output buffer. SGLang keeps its attention,
normalization and recurrent-state kernels. By default the scoring engine runs
single-request complete prefill with batching, chunked prefill, radix caching
and CUDA graphs disabled. Inputs longer than the provider's
paired MLP row limit use separate gate/up GEMMs and Torch SiLU. The
`sglang-candidates` backend computes the output head only for the requested
candidate tokens.

Prefill CUDA graphs are off by default. `open_backend(..., prefill_cuda_graph=True)`
or `--prefill-cuda-graph` in `examples/score_sglang.py` replays the transformer
body from SGLang's breakable prefill graph for inputs up to 4,096 tokens; longer
inputs run eagerly. [Measurements](../bench/README.md#sglang-prefill-cuda-graph).

Batched scoring is off by default. `open_backend(..., batch=True,
activation_scales="scales.json")` adds `score_batch(items)`: up to 16
sequences per forward, a shared prefix of at least 192 tokens (cut to a
multiple of 64) prefilled once, and SGLang's radix cache kept across calls.
Scores are bitwise equal to scoring each sequence alone on an empty cache:

- static activation scales make every quantized row independent of the
  other rows in a forward (per-forward amax scales couple them, so `batch=True`
  requires a `row_independent` provider);
- SGLang deterministic inference makes attention and reductions
  batch-invariant;
- the radix cache keeps recurrent state only at the end of each cached prompt
  (`no_buffer`; intermediate checkpoints do not reproduce a full prefill), and
  a sequence that resumes a cached prompt ending on a multiple of 64 tokens
  (the GDN chunk size) reproduces a full prefill. The session tracks cached
  prompts and clears the cache before scoring a prompt that extends one ending
  off that grid.

Deterministic inference changes attention reduction order, so raw scores
differ from the default (unbatched) engine.

**vLLM.** The adapter registers a Qwen3.5 model class that installs the same
JevInfer projections and paired MLPs after vLLM loads the checkpoint. The engine
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
