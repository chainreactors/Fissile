"""Inference-engine releases each backend has been validated on, with the
packages that determine their numerics.

Fissile installs its kernels into an engine's model classes and keeps the
engine's attention, normalization and recurrent-state kernels. Those kernels
are compiled by the installed Triton, CuTe DSL and FlashInfer, so a new engine
release or a new compiler can change raw scores even when no engine code on the
scoring path changed: FP4 activations turn last-bit differences into changed
near-tie decisions. A validated release is therefore an exact engine version
plus the exact versions of the packages in STACK. ``open_backend`` refuses any
other installed combination unless the caller allows it, which is how a new
combination is validated (docs/engine-upgrades.md). Lookups read package
metadata only and never import an engine.
"""

from importlib import metadata
import os

# Packages whose versions change the numerics of the engine kernels or of
# Fissile's JIT-compiled kernels.
STACK = ("torch", "triton", "nvidia-cutlass-dsl", "flashinfer-python")

# engine -> release -> STACK versions. Added after scripts/check_engine.py
# passes on the full evaluation inputs and the published benchmarks were rerun.
VALIDATED = {
    "sglang": {
        "0.5.21": {"torch": "2.13.0", "triton": "3.7.1", "nvidia-cutlass-dsl": "4.8.0",
                   "flashinfer-python": "0.6.18"},
        "0.5.23": {"torch": "2.14.1", "triton": "3.8.0", "nvidia-cutlass-dsl": "4.8.0",
                   "flashinfer-python": "0.7.2"},
    },
    "vllm": {
        "0.29.0": {"torch": "2.13.0", "triton": "3.7.1", "nvidia-cutlass-dsl": "4.8.0",
                   "flashinfer-python": "0.6.18"},
    },
}
ALLOW_ENV = "FISSILE_ALLOW_UNVALIDATED_ENGINE"


class UnvalidatedEngineError(RuntimeError):
    pass


def installed_version(package):
    """Installed release without a local build tag, or None."""
    try:
        return metadata.version(package).split("+", 1)[0]
    except metadata.PackageNotFoundError:
        return None


def engine_status(engine):
    """Installed engine release and STACK versions, and whether they match a
    validated combination (``stack_mismatch`` lists the packages that differ)."""
    version = installed_version(engine)
    stack = {p: installed_version(p) for p in STACK}
    expected = VALIDATED.get(engine, {}).get(version)
    mismatch = sorted(p for p in STACK if expected is not None and stack[p] != expected.get(p))
    return {
        "engine": engine,
        "engine_version": version,
        "stack": stack,
        "stack_mismatch": mismatch,
        "engine_validated": expected is not None and not mismatch,
    }


def check_engine(engine, *, allow_unvalidated=False):
    """Status of ``engine``; raises unless the installed combination is
    validated or allowed. ``allow_unvalidated`` or
    ``FISSILE_ALLOW_UNVALIDATED_ENGINE=1`` admits another combination, for
    validating it."""
    status = engine_status(engine)
    if status["engine_version"] is None:
        raise RuntimeError(f"{engine} is not installed")
    allowed = allow_unvalidated or os.environ.get(ALLOW_ENV) == "1"
    if not status["engine_validated"] and not allowed:
        releases = VALIDATED.get(engine, {})
        if status["engine_version"] in releases:
            expected = releases[status["engine_version"]]
            detail = "; ".join(f"{p} {status['stack'][p]} (validated: {expected[p]})"
                               for p in status["stack_mismatch"])
            reason = f"{engine} {status['engine_version']} was validated with other packages: {detail}"
        else:
            reason = (f"{engine} {status['engine_version']} is not validated with Fissile "
                      f"(validated: {', '.join(releases) or 'none'})")
        raise UnvalidatedEngineError(
            reason + ". Validate it with scripts/check_engine.py (docs/engine-upgrades.md) "
            f"or set {ALLOW_ENV}=1 to run it unvalidated."
        )
    return status
