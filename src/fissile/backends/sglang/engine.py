"""Supported SGLang launch configuration; full-input single-request zero-generation scoring."""

from contextlib import contextmanager
import os
from pathlib import Path
from fissile.config import Config


def resource_settings(environ=None):
    """Deployment-only resource limits; never override numerical engine settings."""
    env = os.environ if environ is None else environ
    fraction = float(env.get("FISSILE_SGLANG_MEM_FRACTION_STATIC", "0.35"))
    if not 0 < fraction < 1:
        raise ValueError("FISSILE_SGLANG_MEM_FRACTION_STATIC must be between 0 and 1")
    settings = {"mem_fraction_static": fraction}
    if "FISSILE_SGLANG_NCCL_PORT" in env:
        port = int(env["FISSILE_SGLANG_NCCL_PORT"])
        if not 1024 <= port <= 65535:
            raise ValueError("FISSILE_SGLANG_NCCL_PORT must be in 1024..65535")
        settings["nccl_port"] = port
    return settings


READOUTS = ("full", "candidates")

# Token-count buckets for the optional breakable prefill graph. 1,536 is a bucket
# boundary so padding never moves an input from the paired GLU kernel to the
# long-input path. Longer inputs run eagerly.
PREFILL_GRAPH_BUCKETS = (
    (16, 32, 48, 64, 96, 128, 160, 192, 224, 256)
    + tuple(range(320, 1537, 64))
    + tuple(range(1664, 4097, 128))
)


def prefill_graph_settings(enabled):
    if not enabled:
        return {"cuda_graph_backend_prefill": "disabled"}
    return {
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_bs_prefill": list(PREFILL_GRAPH_BUCKETS),
    }


@contextmanager
def scoring_engine(config, model_path, *, readout="full", prefill_cuda_graph=False):
    if readout not in READOUTS:
        raise ValueError("Unknown readout: " + str(readout))
    config = Path(config).resolve()
    cfg = Config.load(config)
    from fissile.backends.registry import get_backend

    get_backend("sglang").validate(cfg)
    resources = resource_settings()
    context_length = cfg.plan.profile.max_context
    old = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    old_config = os.environ.get("FISSILE_CONFIG")
    old_readout = os.environ.get("FISSILE_READOUT")
    if old and old != "fissile.backends.sglang":
        raise RuntimeError("Refuse to replace another external model plugin")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "fissile.backends.sglang"
    os.environ["FISSILE_CONFIG"] = str(config)
    os.environ["FISSILE_READOUT"] = readout
    engine = None
    try:
        import sglang

        engine = sglang.Engine(
            model_path=str(Path(model_path).resolve()),
            dtype="bfloat16",
            # SGLang 0.5.21 reserves six positions in get_worker_info(), then
            # rejects input length >= that limit. Reserve seven slots so the
            # full public max_context is accepted; never truncate the input.
            context_length=context_length + 7,
            disable_radix_cache=True,
            cuda_graph_backend_decode="disabled",
            **prefill_graph_settings(prefill_cuda_graph),
            chunked_prefill_size=-1,
            **resources,
            max_running_requests=1,
            max_total_tokens=2 * context_length,
            max_prefill_tokens=context_length,
            attention_backend="triton",
            linear_attn_backend="triton",
            linear_attn_prefill_backend="triton",
            mamba_backend="triton",
            mamba_ssm_dtype="float32",
            log_level="warning",
        )
        yield engine
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            if old is None:
                os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            else:
                os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = old
            if old_config is None:
                os.environ.pop("FISSILE_CONFIG", None)
            else:
                os.environ["FISSILE_CONFIG"] = old_config
            if old_readout is None:
                os.environ.pop("FISSILE_READOUT", None)
            else:
                os.environ["FISSILE_READOUT"] = old_readout


@contextmanager
def open_session(config, model, *, readout="full", prefill_cuda_graph=False):
    """Common backend contract; this adapter owns the SGLang engine.

    ``readout="full"`` returns full-vocabulary log probabilities. ``"candidates"``
    computes logits for the requested candidate tokens only, so the returned
    log probabilities are normalized over the candidates; their softmax (the
    decision) is mathematically unchanged. ``prefill_cuda_graph=True`` replays
    the transformer body from SGLang's breakable prefill graph for inputs up to
    4,096 tokens; see docs/architecture.md before enabling it.
    """
    from fissile.backends.contracts import Score

    class Session:
        def __init__(self, engine):
            self.engine = engine

        def score(self, input_ids, candidate_token_ids):
            result = self.engine.score(
                query=[],
                items=[list(input_ids)],
                label_token_ids=list(candidate_token_ids),
                apply_softmax=True,
                return_token_logprobs=True,
                temperature=1.0,
            )
            return Score(tuple(result.token_logprobs[0]), result.prompt_tokens)

    with scoring_engine(
        config, model, readout=readout, prefill_cuda_graph=prefill_cuda_graph
    ) as engine:
        yield Session(engine)


@contextmanager
def open_candidate_session(config, model, *, prefill_cuda_graph=False):
    """Candidate-normalized log probabilities; decisions match ``open_session``."""
    with open_session(
        config, model, readout="candidates", prefill_cuda_graph=prefill_cuda_graph
    ) as session:
        yield session
