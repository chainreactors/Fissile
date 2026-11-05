# Benchmarks

Unless noted, measurements use one NVIDIA GB10 (SM121, 128 GB unified memory,
driver 580.178.04, CUDA 13.0), Python 3.12, Torch 2.13.0+cu130, SGLang 0.5.21,
Transformers 5.12.1, Triton 3.7.1 and CuTe DSL 4.8.0.

Workload: dense Qwen3.5-9B-derived merged checkpoint, all 32 layers, complete
prompts, final-token scoring over the full 248,320-token vocabulary, zero
generated tokens, batch size 1, radix cache and CUDA graphs disabled. The mixed
recipe has 184 NVFP4 projections, 16 FP8 K/V projections, one W4A16 head and 32
paired MLPs.

## SGLang: full evaluation suite

27,438 requests, 29,182 questions, 11.14M input tokens.

| Execution | Accuracy | Mean latency | Speedup |
|---|---:|---:|---:|
| SGLang BF16 | 76.38% | 137.11 ms | 1.00× |
| Fissile `gb10-mixed-v1` | 75.19% | 59.72 ms | 2.30× |
| Fissile `gb10-mixed-v2`, `sglang-candidates` | 75.18% | 56.43 ms | **2.43×** |

| Question type | BF16 | `gb10-mixed-v2` |
|---|---:|---:|
| choice (24,639) | 76.35% | 75.13% |
| noul (3,629) | 80.85% | 79.97% |
| score (914) | 59.19% | 57.44% |

`gb10-mixed-v1` and `gb10-mixed-v2` differ on 2 of 29,182 decisions: one
question present in two panels whose top two candidates have bitwise-identical
logits. The largest candidate-probability difference between them is 6×10⁻⁸.

Method: one engine per device; 48-request warm-up, then one timed pass over
every question. Decision: four-decimal candidate probabilities, lexical tie
break. The same configuration produces bitwise-identical raw scores on three
GB10 devices.

[Results](results/gb10-sglang-full.json)

## PyTorch: full evaluation suite

| Execution | Accuracy | Mean latency | Requests/s | Speedup |
|---|---:|---:|---:|---:|
| PyTorch BF16 | 76.31% (22,270) | 235.00 ms | 4.26 | 1.00× |
| PyTorch + Fissile | 74.72% (21,804) | 66.07 ms | 15.13 | **3.56×** |

Times sum recorded request scoring, including encoding and execution checks,
excluding loading and preparation.

[Results and per-panel accuracy](results/gb10-pytorch-full.json)

## vLLM: full evaluation suite

vLLM 0.29.0 with Transformers 5.17.0; other software as above. Same protocol as
SGLang: BF16 is plain vLLM with the engine settings of the `vllm` backend. vLLM
samples one transport token per question in both arms.

| Execution | Accuracy | Mean latency | Speedup |
|---|---:|---:|---:|
| vLLM BF16 | 76.42% (22,300) | 155.34 ms | 1.00× |
| Fissile `gb10-mixed-v2` | 74.69% (21,797) | 79.73 ms | **1.95×** |

| Question type | BF16 | `gb10-mixed-v2` |
|---|---:|---:|
| choice (24,639) | 76.38% | 74.54% |
| noul (3,629) | 80.96% | 80.35% |
| score (914) | 59.41% | 56.24% |

[Results](results/gb10-vllm-full.json)

## SGLang: ScopeJudge

1,043 questions, 346–2,742 tokens (mean 534). Both engines resident; one
warm-up pass each; three interleaved rounds with fixed-seed shuffling; 3,129
timed calls per arm.

| Execution | Mean | P50 | P95 | Requests/s | Correct | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| SGLang BF16 | 138.94 ms | 129.89 ms | 202.66 ms | 7.20 | 1002 | 83.32% |
| Fissile `gb10-mixed-v1` | 66.09 ms | 57.32 ms | 96.89 ms | 15.13 | 999 | 79.92% |

### Fissile optimizations

Each row is a paired comparison against the preceding implementation; raw
candidate scores (or, for candidate readout, all decisions) are identical.

| Optimization | Mean latency | Speedup | 95% CI |
|---|---:|---:|---:|
| Shared activation packing | 66.32 → 65.08 ms | 1.019× | 1.018–1.020 |
| Grouped projection output | — | 1.037× | 1.035–1.038 |
| Pinned CUTLASS NVFP4 GEMM (`gb10-mixed-v2`) | — | 1.030× | 1.026–1.034 |
| Candidate-only readout (`sglang-candidates`) | — | 1.071×¹ | 1.069–1.074 |
| **v1 → v2 + `sglang-candidates`** | **64.86 → 59.30 ms** | **1.094×** | 1.090–1.098 |

¹ Measured on an 8,480-question sample of the full suite.

CI: paired question-level bootstrap, 2,000 resamples.

## vLLM: ScopeJudge

1,043 questions (mean 534 tokens) on one GB10 per provider. vLLM receives all
questions at once; SGLang receives 64 concurrent requests through an HTTP
gateway. Both engines run one sequence at a time.

| Engine | Provider | Requests/s | Correct | Balanced accuracy |
|---|---|---:|---:|---:|
| SGLang | `gb10-mixed-v1` | 16.29 | 999 | 79.92% |
| vLLM | `gb10-mixed-v1` | 11.34 | 994 | 81.29% |
| SGLang | `gb10-mixed-v2` | 16.90 | 999 | 79.92% |
| vLLM | `gb10-mixed-v2` | 11.53 | 994 | 81.29% |

vLLM and SGLang agree on 1,022 of 1,043 decisions; the mean candidate
probability difference is 0.040 (maximum 0.331). Both providers give identical
vLLM results, and repeated vLLM runs are bitwise identical.

[Results](results/gb10-vllm-scope.json)

## SGLang: prefill CUDA graph

`prefill_cuda_graph=True` compared with eager prefill on 1,043 ScopeJudge
questions (mean 534 tokens), 64 concurrent requests through an HTTP gateway.

| Provider | Device | GPU busy, eager | Throughput, graph vs eager | Questions with different scores |
|---|---|---:|---:|---:|
| `sm120-mixed-v1` | RTX 5090 D | about 22% | about 3× | 0 of 24 fixed probes¹ |
| `gb10-mixed-v1` | GB10 | about 89% | 0.99× | 260 of 1,043 (5 decisions) |
| `gb10-mixed-v2` | GB10 | about 90% | 0.96× | 260 of 1,043 (5 decisions) |

¹ The full question set was not scored on SM120.

## SGLang: static activation scales

`activation_scales` on the full evaluation suite with
`sglang-candidates`, same protocol as above. Scales: largest per-forward global
scale of each of the 128 activation sites over every 29th question (1,007
questions, 390,491 tokens), margin 1. BF16, `gb10-mixed-v2` and static scales
ran one after another on one GB10.

| Execution | Accuracy | Outside calibration rows | Mean latency | Speedup |
|---|---:|---:|---:|---:|
| SGLang BF16 | 76.38% | 76.36% | 134.73 ms | 1.00× |
| `gb10-mixed-v2` | 75.18% | 75.15% | 56.41 ms | 2.39× |
| `gb10-mixed-v2-static` | 75.01% | 74.98% | 52.62 ms | **2.56×** |

The providers agree on 26,821 decisions; 901 questions are correct only with
`gb10-mixed-v2`, 851 only with static scales (McNemar z = 1.19). Scales at twice
the calibrated maximum scored 74.83% (z = 2.44).

| Input tokens | Questions | BF16 | `gb10-mixed-v2` | `gb10-mixed-v2-static` |
|---|---:|---:|---:|---:|
| ≤ 256 | 13,381 | 104.99 ms | 37.69 ms | 34.42 ms |
| 257–512 | 13,530 | 118.77 ms | 43.56 ms | 40.92 ms |
| 513–1,536 | 1,725 | 160.69 ms | 69.39 ms | 64.71 ms |
| > 1,536 | 546 | 1,177.37 ms | 792.41 ms | 750.36 ms |

Queued throughput (first 300 suite inputs of at most 1,536 tokens submitted at
once, one sequence at a time, means of 3–6 engine runs): `gb10-mixed-v2` 17.18
requests/s; static scales without the GLU pack epilogue 17.96; static scales
18.62 (1.08×). The GLU pack epilogue is bitwise equal to packing the BF16 GLU
output, so it does not change scores.

[Results](results/gb10-static-scales.json)

## Reproduce

Inputs are JSONL rows with `id`, `input_ids`, `candidate_token_ids`, `labels`
and `expected`.

```bash
python scripts/benchmark_sglang.py --model /path/to/merged-model \
  --config /path/to/fissile.json --inputs /path/to/inputs.jsonl \
  --output /path/to/new-run --host-policy /path/to/host-policy.json \
  --gpu-lock /path/to/gpu.lock --rounds 3
```

`--baseline-config` with `--baseline-source`/`--candidate-source` compares two
Fissile builds; `--require-exact` fails on any raw-score difference;
`--candidate-backend sglang-candidates` selects candidate readout;
`--activation-scales` runs the candidate arm with static activation scales. The host
policy lists allowed hostnames in `mainline_compute_hosts`. Paired runs need
80 GiB of free memory for two resident engines.

## Data

| File | Contents |
|---|---|
| [gb10-sglang-full.json](results/gb10-sglang-full.json) | SGLang full suite |
| [gb10-pytorch-full.json](results/gb10-pytorch-full.json) | PyTorch full suite, per panel |
| [gb10-vllm-full.json](results/gb10-vllm-full.json) | vLLM full suite |
| [gb10-scope.json](results/gb10-scope.json) · [csv](results/gb10-scope.csv) | ScopeJudge BF16 vs Fissile |
| [gb10-activation-reuse.json](results/gb10-activation-reuse.json) · [csv](results/gb10-activation-reuse.csv) | Shared activation packing |
| [gb10-spark-optimization.json](results/gb10-spark-optimization.json) · [csv](results/gb10-spark-optimization.csv) | Grouped output, pinned GEMM |
| [gb10-candidate-readout.json](results/gb10-candidate-readout.json) · [csv](results/gb10-candidate-readout.csv) | Candidate-only readout |
| [gb10-vllm-scope.json](results/gb10-vllm-scope.json) | ScopeJudge vLLM vs SGLang |
| [gb10-static-scales.json](results/gb10-static-scales.json) | Static activation scales, full suite |

CSV files contain anonymous request indices, timings, labels and raw candidate
scores; no prompts or weights.
