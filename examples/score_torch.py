"""One-request-at-a-time final-token scoring with an externally supplied merged model."""

import argparse, json, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText
    from fissile.backends.registry import open_backend

    if Path(a.output).exists():
        raise FileExistsError(a.output)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
            local_files_only=True,
            trust_remote_code=False,
        )
        .eval()
        .requires_grad_(False)
    )
    with (
        open_backend("torch", a.config, model) as session,
        Path(a.output).open("x") as f,
    ):
        for line in Path(a.inputs).open():
            r = json.loads(line)
            ids = r["input_ids"]
            candidates = r["candidate_token_ids"]
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = session.score(ids, candidates)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            f.write(
                json.dumps(
                    dict(
                        id=r["id"],
                        token_logprobs=result.token_logprobs,
                        elapsed_ms=elapsed,
                        prompt_tokens=result.prompt_tokens,
                        generated_tokens=0,
                    )
                )
                + chr(10)
            )
            f.flush()


if __name__ == "__main__":
    main()
