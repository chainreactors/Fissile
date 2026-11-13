"""Explicit external adapters; discovery never imports an engine or starts CUDA."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from .contracts import CheckedSession


@dataclass(frozen=True)
class BackendSpec:
    name: str
    factory: str  # module:context_manager(config, model)
    combinations: frozenset[tuple[str, str, str]]  # profile, recipe, provider
    workload: str = "single-request candidate-token scoring"
    engine: str | None = None  # distribution checked against engines.VALIDATED

    def validate(self, config):
        if (
            config.profile,
            config.recipe,
            config.kernel_provider,
        ) not in self.combinations:
            raise ValueError(
                "Backend does not support this model/recipe/provider combination"
            )


_BACKENDS = {}
BACKENDS = MappingProxyType(_BACKENDS)


def register_backend(spec):
    if spec.name in _BACKENDS:
        raise ValueError("Backend already registered: " + spec.name)
    if not spec.combinations or ":" not in spec.factory:
        raise ValueError("Backend requires explicit capabilities and a factory")
    _BACKENDS[spec.name] = spec


def get_backend(name):
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError("Unsupported backend: " + str(name)) from None


def backend(name):
    """Engine module lookup. Prefer open_backend for a checked scoring session."""
    return import_module(get_backend(name).factory.split(":", 1)[0])


def describe_backends():
    from .engines import VALIDATED, installed_version

    return {
        name: dict(
            factory=s.factory,
            combinations=sorted(s.combinations),
            workload=s.workload,
            engine=s.engine,
            validated_engines=VALIDATED.get(s.engine, {}),
            installed_engine_version=installed_version(s.engine) if s.engine else None,
        )
        for name, s in BACKENDS.items()
    }


@contextmanager
def open_backend(name, config, model, *, allow_unvalidated_engine=False, **options):
    """Checked scoring session. The installed engine release must be validated
    for the backend (engines.VALIDATED) unless ``allow_unvalidated_engine``."""
    from fissile.config import Config
    from pathlib import Path
    from .engines import check_engine

    # Validate capabilities and the engine release before importing external
    # engine code. The adapter validates file integrity before installation
    # through Config/MixedRuntime.
    config = Path(config).resolve()
    cfg = Config.load(config, verify_files=False)
    spec = get_backend(name)
    spec.validate(cfg)
    if spec.engine:
        check_engine(spec.engine, allow_unvalidated=allow_unvalidated_engine)
    module, symbol = spec.factory.split(":", 1)
    with getattr(import_module(module), symbol)(config, model, **options) as session:
        checked = CheckedSession(session, cfg.plan.profile)
        try:
            yield checked
        finally:
            checked.closed = True


_INITIAL = frozenset({("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v1")})
register_backend(BackendSpec("torch", "fissile.backends.torch:open_session", _INITIAL))
# Same engine, kernels and logits for the requested tokens; only final-token rows
# for the candidates are computed, so log probabilities are candidate-normalized.
register_backend(
    BackendSpec(
        "sglang-candidates",
        "fissile.backends.sglang.engine:open_candidate_session",
        _INITIAL
        | {
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2"),
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2-calibration"),
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2-static"),
        },
        workload="single-request candidate-normalized scoring",
        engine="sglang",
    )
)
register_backend(
    BackendSpec(
        "vllm",
        "fissile.backends.vllm.engine:open_session",
        _INITIAL | {("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2")},
        engine="vllm",
    )
)
register_backend(
    BackendSpec(
        "sglang",
        "fissile.backends.sglang.engine:open_session",
        _INITIAL
        | {
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "sm120-mixed-v1"),
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2"),
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2-calibration"),
            ("qwen3.5-dense-9b", "mixed-fp4-fp8-w4a16-v1", "gb10-mixed-v2-static"),
        },
        engine="sglang",
    )
)
