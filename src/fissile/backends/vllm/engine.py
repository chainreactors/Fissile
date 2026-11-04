"""Supported vLLM launch configuration; full-input single-request scoring.

vLLM always samples at least one token, so scoring requests one transport token
and reads full-vocabulary raw log probabilities of the requested candidates at
that position. The transport token is discarded.
"""

from contextlib import contextmanager
import os
from pathlib import Path
from fissile.config import Config

# vLLM returns log probabilities for at most this many requested token IDs.
MAX_CANDIDATES = 128


def resource_settings(environ=None):
    """Deployment-only memory limit; never override numerical engine settings."""
    env = os.environ if environ is None else environ
    size = int(env.get("FISSILE_VLLM_KV_CACHE_BYTES", str(2 << 30)))
    if not 1 << 28 <= size <= 1 << 40:
        raise ValueError("FISSILE_VLLM_KV_CACHE_BYTES must be between 256 MiB and 1 TiB")
    return {"kv_cache_memory_bytes": size}


def engine_settings(model_path, context_length, environ=None):
    return dict(
        model=str(Path(model_path).resolve()),
        dtype="bfloat16",
        # The prompt plus the one transport token must fit; inputs are never
        # truncated. One step holds a full prompt, so prefill is never chunked.
        max_model_len=context_length + 1,
        max_num_batched_tokens=context_length,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        mamba_ssm_cache_dtype="float32",
        logprobs_mode="raw_logprobs",
        generation_config="vllm",
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
        **resource_settings(environ),
    )


@contextmanager
def scoring_engine(config, model_path):
    config = Path(config).resolve()
    cfg = Config.load(config)
    from fissile.backends.registry import get_backend

    get_backend("vllm").validate(cfg)
    environment = {
        "FISSILE_CONFIG": str(config),
        # The engine core runs in this process so the registered model class and
        # FISSILE_CONFIG reach the worker that loads the weights.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        # Model runner V2 warms kernels with multi-request decode and prompt
        # log probability steps; V1 runs one complete prefill and a one-row
        # sampler warmup, which the single-request adapter accepts.
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }
    saved = {k: os.environ.get(k) for k in environment}
    os.environ.update(environment)
    engine = None
    try:
        from vllm import LLM, ModelRegistry
        from fissile.backends.vllm.model import MODELS

        for architecture, cls in MODELS.items():
            ModelRegistry.register_model(architecture, cls)
        engine = LLM(**engine_settings(model_path, cfg.plan.profile.max_context))
        yield engine
    finally:
        try:
            if engine is not None:
                del engine
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@contextmanager
def open_session(config, model):
    """Full-vocabulary log probabilities of the candidate tokens."""
    from fissile.backends.contracts import Score

    class Session:
        def __init__(self, engine):
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            self.engine, self.params, self.prompt = engine, SamplingParams, TokensPrompt

        def score(self, input_ids, candidate_token_ids):
            candidates = list(candidate_token_ids)
            if not 1 <= len(candidates) <= MAX_CANDIDATES:
                raise ValueError("vLLM scores 1-128 candidate tokens per request")
            params = self.params(max_tokens=1, temperature=0.0, logprob_token_ids=candidates)
            (output,) = self.engine.generate(
                [self.prompt(prompt_token_ids=list(input_ids))], params, use_tqdm=False
            )
            first = output.outputs[0].logprobs[0]
            if len(output.prompt_token_ids) != len(input_ids):
                raise RuntimeError("Input was changed")
            return Score(tuple(first[t].logprob for t in candidates), len(output.prompt_token_ids))

    with scoring_engine(config, model) as engine:
        yield Session(engine)
