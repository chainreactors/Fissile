"""Run tokenized candidate-scoring examples. No model/dataset is bundled."""

import argparse, json, time
from pathlib import Path
from fissile.backends.registry import open_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", default="sglang", choices=("sglang", "sglang-candidates", "vllm"))
    p.add_argument(
        "--prefill-cuda-graph",
        action="store_true",
        help="Replay prefill from SGLang's breakable CUDA graph (see docs/architecture.md)",
    )
    p.add_argument(
        "--activation-scales",
        help="Calibrated activation scale file: run a gb10-mixed-v2 config with static "
        "NVFP4 activation scales (see docs/architecture.md)",
    )
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in Path(a.inputs).open()]
    options = {"prefill_cuda_graph": True} if a.prefill_cuda_graph else {}
    if a.activation_scales:
        options["activation_scales"] = a.activation_scales
    if options and a.backend == "vllm":
        p.error("--prefill-cuda-graph and --activation-scales apply to the SGLang backends")
    with open_backend(a.backend, a.config, a.model, **options) as session, output.open("x") as f:
        for r in rows:
            start = time.perf_counter()
            result = session.score(r["input_ids"], r["candidate_token_ids"])
            record = {
                "id": r["id"],
                "token_logprobs": result.token_logprobs,
                "prompt_tokens": result.prompt_tokens,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "generated_tokens": 0,
            }
            if result.prompt_tokens != len(r["input_ids"]):
                raise RuntimeError("Input was changed")
            f.write(json.dumps(record) + chr(10))
            f.flush()


if __name__ == "__main__":
    main()
