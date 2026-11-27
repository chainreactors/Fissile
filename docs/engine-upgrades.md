# Engine upgrades

JevInfer runs inside SGLang and vLLM. Its kernels replace the projections and
the output head; the engine keeps attention, normalization and recurrent-state
kernels, scheduling and caching. Those engine kernels are compiled by the
installed Triton, CuTe DSL and FlashInfer, and FP4 activations turn last-bit
differences in them into changed decisions on near-tie questions. Every new
engine release or compiler therefore changes raw scores, even when no engine
code on the scoring path changed. Upgrades are validated as complete stacks and
checked for accuracy against the measured size of this numerical noise.

## Validated stacks

`fissile.backends.engines.VALIDATED` lists, per engine release, the versions
of the packages in `STACK` (torch, Triton, CuTe DSL, FlashInfer):

| Engine | Release | torch | Triton | CuTe DSL | FlashInfer |
|---|---|---|---|---|---|
| `sglang` (`sglang`, `sglang-candidates`) | 0.5.23 | 2.14.1 | 3.8.0 | 4.8.0 | 0.7.2 |
| `sglang` (`sglang`, `sglang-candidates`) | 0.5.21 | 2.13.0 | 3.7.1 | 4.8.0 | 0.6.18 |
| `vllm` | 0.29.0 | 2.13.0 | 3.7.1 | 4.8.0 | 0.6.18 |

`open_backend` refuses any other installed combination with
`UnvalidatedEngineError` before importing engine code;
`open_backend(..., allow_unvalidated_engine=True)` or
`FISSILE_ALLOW_UNVALIDATED_ENGINE=1` admits it. The install report
(`quant-install.json`) records the engine release, the stack and whether it is
validated; `fissile backends` lists the validated stacks.

## Numerical noise

Measured on the full evaluation suite (29,182 questions), `gb10-mixed-v2`,
`sglang-candidates`, with numerically different but otherwise equivalent
stacks:

| Stack | Accuracy | Raw scores equal to the first row | Decisions changed |
|---|---:|---:|---:|
| SGLang 0.5.21, Triton 3.7.1 | 75.18% | — | — |
| SGLang 0.5.21, Triton 3.7.1, deterministic inference | 74.79% | 980 | 2,266 |
| SGLang 0.5.21, Triton 3.8.0 | 74.82% | 923 | 1,712 |
| SGLang 0.5.23 (torch 2.14.1, Triton 3.8.0) | 74.82% | 923 | 1,712 |

SGLang 0.5.23 is bitwise equal to SGLang 0.5.21 with Triton 3.8.0 on every
question: the differences come from Triton's code generation for SGLang's
linear-attention and gated-norm kernels. JevInfer's own kernels give bitwise
equal outputs under both Triton releases. BF16 accuracy moves by less than
0.02 points between these stacks (76.38%, 76.39%).

The first row is the stack on which the configuration was evaluated; any other
stack scores about 0.35 points lower. Upgrade checks therefore compare with a
perturbed run of the validated stack (deterministic inference), not with its
original run.

## Validating a stack

1. Create an environment with the candidate engine release:

   ```bash
   python scripts/engine_env.py --engine sglang==0.5.23 --output env-sglang-0.5.23
   ```

   Package indexes, mirrors and proxies come from your pip configuration.
   `env-sglang-0.5.23/stack.json` records the installed stack.
2. Run the CPU tests in it: `python -m unittest discover -s tests`.
3. Once per validated stack, create its reference on the full evaluation
   inputs (`--perturbed` adds the deterministic-inference run):

   ```bash
   python scripts/check_engine.py --backend sglang-candidates --config config.json \
     --model MODEL --inputs suite.jsonl --bf16 --perturbed --output reference-sglang-0.5.21
   ```

4. Check the candidate in its environment against that reference:

   ```bash
   python scripts/check_engine.py --backend sglang-candidates --config config.json \
     --model MODEL --inputs suite.jsonl --bf16 --activation-scales scales.json \
     --reference reference-sglang-0.5.21 --output check-sglang-0.5.23
   ```

   `check-sglang-0.5.23/report.json` holds the stack, startup and mean latency
   per arm, accuracy, the comparisons with the reference and the verdict:

   | Check | Passes when |
   |---|---|
   | `installed` | The install report covers every projection |
   | `repeatable` | Rescored inputs are bitwise equal |
   | `batch_exact` | `score_batch` is bitwise equal to `score` (with `--activation-scales`) |
   | `bf16_not_worse` | BF16 accuracy is not significantly lower than the reference's (McNemar z ≥ −1.96) |
   | `fissile_not_worse` | JevInfer accuracy is not significantly lower than the reference's perturbed run (McNemar z ≥ −1.96) |

   Use the full evaluation inputs: on 1,043 ScopeJudge questions alone, a
   0.36-point change on the full suite is not visible.
5. Add the release and its stack to `VALIDATED` and the matching extra in
   `pyproject.toml`, and record the check's accuracy and latency in the
   benchmarks.

`--bf16` and `--perturbed` are available for the SGLang backends; vLLM checks
compare JevInfer runs only.

## What the adapters depend on

Review upstream changes to these interfaces when a check fails to install.

| SGLang | Used for |
|---|---|
| `SGLANG_EXTERNAL_MODEL_PACKAGE`, `Qwen3_5ForConditionalGeneration` | Model class replacement, weight loading |
| Module layout (`backends/layouts.py`) | Projection, MLP and head installation |
| `Engine.score(..., label_token_ids, apply_softmax, return_token_logprobs)` | Prefill-only scoring, one candidate list per item |
| `ForwardBatch`: `forward_mode`, `batch_size`, `extend_seq_lens(_cpu)`, `seq_lens(_cpu)`, `token_ids_logprobs` | Forward checks, candidate readout |
| Input-length reservation in `get_worker_info` | `context_length` = profile limit + 7 |
| Server arguments in `backends/sglang/engine.py` | Engine settings, including `enable_deterministic_inference`, `mamba_radix_cache_strategy` and `disable_overlap_schedule` for batched scoring |
| `Engine.flush_cache` | Batched scoring cache resets |

| vLLM | Used for |
|---|---|
| `ModelRegistry`, `Qwen3_5ForConditionalGeneration` | Model class replacement, weight loading |
| `LLM` arguments in `backends/vllm/engine.py`, `VLLM_ENABLE_V1_MULTIPROCESSING`, `VLLM_USE_V2_MODEL_RUNNER` | In-process V1 engine core, one sequence |
| `SamplingParams(logprobs=..., logprobs_mode="raw_logprobs")` | Candidate log probabilities at the transport token |

## Deployments

- Install the new stack in its own environment next to the current one;
  switch, verify, and keep the previous environment for rollback.
- Raw scores change with the stack: regenerate every frozen expectation
  (startup answers, exactness references) on the new stack.
- Record the stack with the deployment; the install report of each start
  contains it.
