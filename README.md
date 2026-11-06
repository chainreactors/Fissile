# Fissile

**Mixed-precision inference for SGLang, vLLM and PyTorch on NVIDIA GB10 (DGX Spark).**

Fissile runs an NVFP4/FP8 mixed-precision recipe inside existing inference
engines. Models, precision recipes, kernel providers and engine adapters are
separate components; the engines keep their own scheduling, attention and
state kernels.

## Performance

One NVIDIA GB10, dense Qwen3.5-9B-derived model, final-token candidate scoring,
batch size 1. Full evaluation suite: 27,438 requests, 29,182 questions.

| Engine | Execution | Accuracy | Mean latency | Speedup |
|---|---|---:|---:|---:|
| SGLang | BF16 | 76.38% | 137.11 ms | 1.00× |
| SGLang | Fissile `gb10-mixed-v2`, candidate readout | 75.18% | 56.43 ms | **2.43×** |
| SGLang | Fissile `gb10-mixed-v2`, candidate readout, static activation scales¹ | 75.01% | 52.62 ms | **2.56×** |
| vLLM | BF16 | 76.42% | 155.34 ms | 1.00× |
| vLLM | Fissile `gb10-mixed-v2` | 74.69% | 79.73 ms | **1.95×** |
| PyTorch | BF16 | 76.31% | 235.00 ms | 1.00× |
| PyTorch | Fissile | 74.72% | 66.07 ms | **3.56×** |

¹ Opt-in. Speedup against BF16 timed in the same session on one GB10, in which
BF16 took 134.73 ms and `gb10-mixed-v2` 56.41 ms
([details](bench/README.md#sglang-static-activation-scales)).

[Benchmarks, method and data](bench/README.md)

## Features

- **Mixed-precision recipe**: NVFP4 W4A4 projections, FP8 full-attention K/V and
  an NVFP4 W4A16 output head.
- **Fused GLU**: one paired gate/up NVFP4 GEMM with a BF16 SiLU lookup epilogue.
- **Grouped projections**: same-input projections share one activation pack and
  write directly into one output buffer.
- **Pinned CUTLASS NVFP4 GEMM** (`gb10-mixed-v2`): one SM121 kernel for every
  W4A4 projection, bitwise equal to the cuBLAS path.
- **Candidate-only readout** (`sglang-candidates`): the output head computes
  only the requested candidate tokens.
- **Prefill CUDA graph** (SGLang, opt-in): replays the transformer body for
  inputs up to 4,096 tokens.
- **Static activation scales** (SGLang, opt-in): calibrated
  per-site NVFP4 activation scales replace per-forward amax reductions; the
  paired GLU writes the packed hidden activation from its epilogue.
- **Engine adapters**: SGLang and vLLM model adapters and a reversible PyTorch
  overlay share one runtime and scoring API.
- **Artifact integrity**: manifests with effective-weight hashes, packed-file
  hashes and shape checks; build receipts for native libraries.
- **Weight preparation**: `fissile quantize` calibrates a BF16 model and prepares
  every projection pack (input-moment-weighted block scales, GPTQ); existing
  NVFP4 packs can be imported.

## Supported configurations

| Component | Supported |
|---|---|
| Hardware | NVIDIA GB10 (SM121); RTX 5090 D (SM120, SGLang) |
| Platform | Linux aarch64 or x86_64, Python 3.12, CUDA 13 |
| Model | Dense Qwen3.5-9B |
| Engines | SGLang 0.5.21, vLLM 0.29.0, PyTorch 2.13 |
| Workload | Complete-prompt, final-token candidate scoring |

| Backend | `model` argument | `token_logprobs` |
|---|---|---|
| `sglang` | Model directory | Full-vocabulary log probabilities |
| `sglang-candidates` | Model directory | Log probabilities normalized over the candidates |
| `vllm` | Model directory | Full-vocabulary log probabilities |
| `torch` | Loaded, merged, frozen model | Full-vocabulary log probabilities |

| Kernel provider | Description |
|---|---|
| `gb10-mixed-v1` | GB10 default; cuBLAS NVFP4 GEMM |
| `gb10-mixed-v2` | GB10; pinned CUTLASS NVFP4 GEMM |
| `gb10-mixed-v2-static` | `gb10-mixed-v2` with static activation scales (`activation_scales`) |
| `gb10-mixed-v2-calibration` | `gb10-mixed-v2` recording activation scales for calibration |
| `sm120-mixed-v1` | SM120 |

## Install

```bash
python -m pip install .
fissile --version
fissile backends
```

The base package has no GPU dependencies. Engine environment: Torch
2.13.0+cu130, SGLang 0.5.21, Transformers 5.12.1, Triton 3.7.1 and CuTe DSL
4.8.0 ([setup](docs/artifacts.md#sglang-environment)); vLLM 0.29.0 uses its own
environment ([setup](docs/artifacts.md#vllm-environment)).

## Quick start

Build the native resources (CPU compilation) and the SiLU table:

```bash
fissile build-tensorfold --output artifacts/libfissile_tensorfold.so
fissile build-fp4-gemm --output artifacts/libfissile_fp4_gemm.so --cutlass-dir /path/to/cutlass
python scripts/build_silu_table.py --output artifacts/silu.safetensors
```

Prepare the projection packs from a merged BF16 model and calibration inputs
(GPU):

```bash
fissile quantize --model /path/to/merged-model --calibration calibration.jsonl \
  --packs-dir artifacts/packs --output artifacts/manifest.json
```

Each calibration line contains the `input_ids` of one representative, unlabeled
input ([details](docs/artifacts.md#prepare-packs)).

Copy [`examples/config-v2-gb10-cutlass.json`](examples/config-v2-gb10-cutlass.json),
set the paths and the SiLU table hash, then validate and score:

```bash
fissile validate --config config.json
python examples/score_sglang.py --config config.json \
  --model /path/to/merged-model --inputs inputs.jsonl --output results.jsonl
```

Each input line contains `id`, `input_ids` and `candidate_token_ids`.

Optional static activation scales: calibrate once, then pass the
scale file at startup:

```bash
python scripts/calibrate_activation_scales.py --config config.json \
  --model /path/to/merged-model --inputs inputs.jsonl --output scales.json
python examples/score_sglang.py --config config.json --activation-scales scales.json \
  --model /path/to/merged-model --inputs inputs.jsonl --output results.jsonl
```

## Python API

```python
from fissile.backends.registry import open_backend

with open_backend("sglang", "config.json", model_dir) as session:
    result = session.score(input_ids, candidate_token_ids)
    print(result.token_logprobs, result.prompt_tokens)
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration and artifacts](docs/artifacts.md)
- [Extending Fissile](docs/extensions.md)
- [Benchmarks](bench/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Acknowledgements

Fissile builds on [SGLang](https://github.com/sgl-project/sglang),
[PyTorch](https://github.com/pytorch/pytorch),
[TensorFold](https://github.com/ashhart/TensorFold) (W4A16 head kernels),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) and
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) (SM12x GEMM and scale layouts),
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (FP4 scale search),
[Triton](https://github.com/triton-lang/triton) and
[Transformers](https://github.com/huggingface/transformers). Weight preparation
follows [GPTQ](https://arxiv.org/abs/2210.17323) and
[MR-GPTQ](https://arxiv.org/abs/2509.23202). Licenses and pinned sources:
[third-party notices](THIRD_PARTY_NOTICES.md).
