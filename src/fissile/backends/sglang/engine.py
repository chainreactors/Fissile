"""Supported SGLang launch configuration; full-input zero-generation scoring."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
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

# Token-count buckets for the optional breakable prefill graph; longer inputs run
# eagerly. A provider's paired_mlp_max_rows must be a bucket boundary so padding
# never moves an input from the paired GLU kernel to the long-input path.
PREFILL_GRAPH_BUCKETS = (
    (16, 32, 48, 64, 96, 128, 160, 192, 224, 256)
    + tuple(range(320, 1537, 64))
    + tuple(range(1664, 4097, 128))
)


def prefill_graph_settings(enabled, paired_mlp_max_rows=0):
    if not enabled:
        return {"cuda_graph_backend_prefill": "disabled"}
    if (
        0 < paired_mlp_max_rows < PREFILL_GRAPH_BUCKETS[-1]
        and paired_mlp_max_rows not in PREFILL_GRAPH_BUCKETS
    ):
        raise ValueError(
            f"paired_mlp_max_rows={paired_mlp_max_rows} is not a prefill graph bucket"
        )
    return {
        "cuda_graph_backend_prefill": "breakable",
        "cuda_graph_bs_prefill": list(PREFILL_GRAPH_BUCKETS),
    }


MAX_BATCH = 16  # sequences per batched scoring call
PREFIX_MIN = 192  # shared prefix length worth prefilling once before a batch
# GDN chunk size (page size 1). Under deterministic inference, a sequence that
# resumes a cached prompt ending on this grid reproduces a full prefill bit for bit.
CACHE_GRID = 64
CACHE_PROMPTS = 4096  # tracked cached prompts before the cache is cleared


def batch_settings(batch, context_length):
    """Engine settings for one-at-a-time (default) or batched scoring.

    Batched scoring runs several sequences per forward under deterministic
    inference (batch-invariant attention and reductions) and keeps SGLang's
    radix cache with recurrent state only at the end of each cached prompt
    (``no_buffer``; the intermediate checkpoints of the other strategies do not
    reproduce a full prefill). ``no_buffer`` needs the overlap scheduler off."""
    if not batch:
        return dict(disable_radix_cache=True, max_running_requests=1, max_total_tokens=2 * context_length)
    return dict(
        disable_radix_cache=False,
        max_running_requests=MAX_BATCH,
        max_total_tokens=4 * context_length,
        page_size=1,
        enable_deterministic_inference=True,
        mamba_radix_cache_strategy="no_buffer",
        disable_overlap_schedule=True,
    )


class PrefixCache:
    """Prompts in SGLang's radix cache since it was last cleared.

    The cache resumes a prompt only where an earlier prompt ended. A prompt
    that extends a cached prompt ending off the CACHE_GRID would continue
    mid-chunk and lose bitwise equality, so it needs an empty cache first.
    Prompts are tracked by (length, polynomial hash); a collision only costs a
    clear.
    """

    MODULUS = (1 << 61) - 1
    BASE = 1_000_003

    def __init__(self, grid=CACHE_GRID, limit=CACHE_PROMPTS):
        self.grid, self.limit = grid, limit
        self.clear()

    def clear(self):
        self.ends = set()  # (length, hash) of cached prompts ending off the grid
        self.lengths = set()
        self.prompts = 0

    def _scan(self, ids, lengths=frozenset()):
        """Prefix hashes at ``lengths`` shorter than ``ids``, and the hash of ``ids``."""
        found, h = [], 0
        for n, token in enumerate(ids, 1):
            h = (h * self.BASE + token + 1) % self.MODULUS
            if n in lengths and n < len(ids):
                found.append((n, h))
        return found, (len(ids), h)

    def plan(self, prompts):
        """``"split"`` if a prompt extends another prompt of the same call that
        ends off the grid; ``"clear"`` if one extends such a cached prompt or the
        cache is full; else ``"reuse"``."""
        lengths = self.lengths | {len(ids) for ids in prompts if len(ids) % self.grid}
        scans = [self._scan(ids, lengths) for ids in prompts]
        ends = {end for _, end in scans if end[0] % self.grid}
        if any(item in ends for found, _ in scans for item in found):
            return "split"
        if self.prompts + len(prompts) > self.limit or any(
            item in self.ends for found, _ in scans for item in found
        ):
            return "clear"
        return "reuse"

    def add(self, prompts):
        for ids in prompts:
            self.prompts += 1
            if len(ids) % self.grid:
                self.lengths.add(len(ids))
                self.ends.add(self._scan(ids)[1])


@contextmanager
def scoring_engine(
    config, model_path, *, readout="full", prefill_cuda_graph=False, activation_scales=None, batch=False
):
    if activation_scales is not None:
        # Static activation scales: run the gb10-mixed-v2 configuration as its
        # gb10-mixed-v2-static variant for the lifetime of this engine.
        if prefill_cuda_graph:
            raise ValueError("Static activation scales are not supported with prefill CUDA graphs")
        from fissile.providers.static_scales import derived_config

        with tempfile.TemporaryDirectory(prefix="fissile-static-") as directory:
            with scoring_engine(
                derived_config(config, directory, scales=activation_scales),
                model_path,
                readout=readout,
                batch=batch,
            ) as engine:
                yield engine
        return
    if readout not in READOUTS:
        raise ValueError("Unknown readout: " + str(readout))
    if batch and prefill_cuda_graph:
        raise ValueError("Batched scoring is not supported with prefill CUDA graphs")
    config = Path(config).resolve()
    if batch and not Config.load(config, verify_files=False).provider.row_independent:
        raise ValueError(
            "Batched scoring needs a row-independent kernel provider; "
            "use static activation scales (activation_scales=...)"
        )
    cfg = Config.load(config)
    from fissile.backends.registry import get_backend

    get_backend("sglang").validate(cfg)
    resources = resource_settings()
    context_length = cfg.plan.profile.max_context
    old = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    old_config = os.environ.get("FISSILE_CONFIG")
    old_readout = os.environ.get("FISSILE_READOUT")
    old_batch = os.environ.get("FISSILE_MAX_BATCH")
    if old and old != "fissile.backends.sglang":
        raise RuntimeError("Refuse to replace another external model plugin")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "fissile.backends.sglang"
    os.environ["FISSILE_CONFIG"] = str(config)
    os.environ["FISSILE_READOUT"] = readout
    os.environ["FISSILE_MAX_BATCH"] = str(MAX_BATCH if batch else 1)
    engine = None
    try:
        import sglang

        engine = sglang.Engine(
            model_path=str(Path(model_path).resolve()),
            dtype="bfloat16",
            # SGLang 0.5.21 to 0.5.23 reserve six positions in get_worker_info(),
            # then reject input length >= that limit. Reserve seven slots so the
            # full public max_context is accepted; never truncate the input.
            context_length=context_length + 7,
            cuda_graph_backend_decode="disabled",
            **prefill_graph_settings(
                prefill_cuda_graph,
                cfg.provider.paired_mlp_max_rows if prefill_cuda_graph else 0,
            ),
            chunked_prefill_size=-1,
            **resources,
            **batch_settings(batch, context_length),
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
            if old_batch is None:
                os.environ.pop("FISSILE_MAX_BATCH", None)
            else:
                os.environ["FISSILE_MAX_BATCH"] = old_batch


@contextmanager
def open_session(
    config, model, *, readout="full", prefill_cuda_graph=False, activation_scales=None, batch=False
):
    """Common backend contract; this adapter owns the SGLang engine.

    ``readout="full"`` returns full-vocabulary log probabilities. ``"candidates"``
    computes logits for the requested candidate tokens only, so the returned
    log probabilities are normalized over the candidates; their softmax (the
    decision) is mathematically unchanged. ``prefill_cuda_graph=True`` replays
    the transformer body from SGLang's breakable prefill graph for inputs up to
    4,096 tokens; see docs/architecture.md before enabling it.
    ``activation_scales`` (a calibrated scale file) runs a gb10-mixed-v2
    configuration with static NVFP4 activation scales (gb10-mixed-v2-static).
    ``batch=True`` (requires ``activation_scales``) adds ``score_batch``:
    several sequences per forward, a shared prefix of at least PREFIX_MIN tokens
    prefilled once, and prefixes cached across calls. Scores are bitwise equal
    to scoring each sequence alone on an empty cache.
    """
    from fissile.backends.contracts import Score

    class Session:
        def __init__(self, engine):
            self.engine = engine
            self.cache = PrefixCache() if batch else None

        def _score(self, items, labels):
            return self.engine.score(
                query=[],
                items=items,
                label_token_ids=labels,
                apply_softmax=True,
                return_token_logprobs=True,
                temperature=1.0,
            )

        def _admit(self, prompts, plan):
            if plan == "clear":
                result = self.engine.flush_cache()
                if getattr(result, "success", True) is False:
                    raise RuntimeError("SGLang did not clear its cache")
                self.cache.clear()
            # Recorded before scoring: a failed call may still have cached them.
            self.cache.add(prompts)

        def score(self, input_ids, candidate_token_ids):
            ids = list(input_ids)
            if batch:
                self._admit([ids], self.cache.plan([ids]))
            result = self._score([ids], list(candidate_token_ids))
            return Score(tuple(result.token_logprobs[0]), result.prompt_tokens)

        def score_batch(self, items):
            items = [(list(ids), list(candidates)) for ids, candidates in items]
            if not batch or len(items) == 1:
                return [self.score(*item) for item in items]
            if len(items) > MAX_BATCH:
                raise ValueError(f"At most {MAX_BATCH} items per batch")
            prompts = [ids for ids, _ in items]
            plan = self.cache.plan(prompts)
            if plan == "split":
                return [self.score(*item) for item in items]
            self._admit(prompts, plan)
            shared = 0
            for column in zip(*prompts):
                if len(set(column)) != 1:
                    break
                shared += 1
            shared = shared // CACHE_GRID * CACHE_GRID
            if shared >= PREFIX_MIN:
                # Ends on the grid, so every item resumes it exactly.
                self._score([prompts[0][:shared]], [items[0][1][:1]])
            result = self._score(prompts, [candidates for _, candidates in items])
            return [
                Score(tuple(row), len(ids))
                for row, ids in zip(result.token_logprobs, prompts, strict=True)
            ]

    with scoring_engine(
        config,
        model,
        readout=readout,
        prefill_cuda_graph=prefill_cuda_graph,
        activation_scales=activation_scales,
        batch=batch,
    ) as engine:
        yield Session(engine)


@contextmanager
def open_candidate_session(config, model, *, prefill_cuda_graph=False, activation_scales=None, batch=False):
    """Candidate-normalized log probabilities; decisions match ``open_session``."""
    with open_session(
        config,
        model,
        readout="candidates",
        prefill_cuda_graph=prefill_cuda_graph,
        activation_scales=activation_scales,
        batch=batch,
    ) as session:
        yield session
