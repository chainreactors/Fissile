"""CPU-only kernel capability declarations and explicit, lazy provider selection."""

from dataclasses import dataclass, replace
from importlib import import_module
from typing import Protocol, Callable, Any


class KernelProvider(Protocol):
    """Factories return matrix functions; runtime owns reshaping and coverage."""

    def matrix(self, x: Any) -> Any: ...
    def projection(self, route: str, weight: Any, pack: Any) -> Callable: ...
    def mlp(self, packs: tuple, *, long_silu: str) -> Callable: ...


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    integrity: str  # sha256 or build_receipt


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    factory: str
    profiles: frozenset[str]
    recipes: frozenset[str]
    capability: tuple[int, int]
    activation_dtype: str
    routes: frozenset[str]
    packed_routes: frozenset[str]
    fusions: frozenset[str]
    alignment: int
    resources: tuple[ResourceSpec, ...] = ()
    # Largest row count served by the paired gate/up MLP kernel; longer inputs
    # use separate gate/up GEMMs. Required with the ``paired_mlp`` fusion.
    paired_mlp_max_rows: int = 0

    def __post_init__(self):
        if self.alignment < 1 or not self.packed_routes <= self.routes:
            raise ValueError("Invalid provider capabilities")
        if len({r.name for r in self.resources}) != len(self.resources):
            raise ValueError("Duplicate provider resources")
        if any(r.integrity not in {"sha256", "build_receipt"} for r in self.resources):
            raise ValueError("Unknown resource integrity contract")
        if self.paired_mlp_max_rows < 0 or (
            "paired_mlp" in self.fusions and not self.paired_mlp_max_rows
        ):
            raise ValueError("paired_mlp fusion requires a positive paired_mlp_max_rows")

    def validate(self, plan, *, capability=None, activation_dtype="bfloat16"):
        if (
            plan.profile.name not in self.profiles
            or plan.recipe.name not in self.recipes
        ):
            raise ValueError("Provider does not support selected model/recipe")
        if capability is not None and tuple(capability) != self.capability:
            raise ValueError("Provider does not support device capability")
        if activation_dtype != self.activation_dtype:
            raise ValueError("Provider does not support activation dtype")
        if (
            not set(plan.routes.values()) <= self.routes
            or not plan.recipe.fusions <= self.fusions
        ):
            raise ValueError("Provider does not implement required routes/fusions")
        if any(n % self.alignment for p in plan.profile.projections for n in p.shape):
            raise ValueError("Projection shape violates provider alignment")

    def create(self, config):
        self.validate(config.plan)
        module, symbol = self.factory.split(":", 1)
        return getattr(import_module(module), symbol)(config)


_PROVIDERS = {}


def register_provider(spec):
    if spec.name in _PROVIDERS:
        raise ValueError("Kernel provider already registered: " + spec.name)
    _PROVIDERS[spec.name] = spec


def get_provider(name):
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError("Unsupported kernel provider: " + str(name)) from None


DEFAULT_PROVIDER = "gb10-mixed-v1"
register_provider(
    ProviderSpec(
        name=DEFAULT_PROVIDER,
        factory="fissile.providers.gb10:GB10Provider",
        profiles=frozenset({"qwen3.5-dense-9b"}),
        recipes=frozenset({"mixed-fp4-fp8-w4a16-v1"}),
        capability=(12, 1),
        activation_dtype="bfloat16",
        routes=frozenset({"fp4", "fp8", "w4a16"}),
        packed_routes=frozenset({"fp4", "w4a16"}),
        fusions=frozenset({"paired_mlp"}),
        alignment=128,
        resources=(
            ResourceSpec("tensorfold_library", "build_receipt"),
            ResourceSpec("silu_lut", "sha256"),
        ),
        paired_mlp_max_rows=1536,
    )
)
register_provider(
    replace(
        get_provider(DEFAULT_PROVIDER),
        name="sm120-mixed-v1",
        factory="fissile.providers.sm120:SM120Provider",
        capability=(12, 0),
    )
)
# Same recipe arithmetic as gb10-mixed-v1; W4A4 GEMMs use one pinned CUTLASS
# SM121 kernel (bitwise-checked against cuBLAS) instead of cuBLAS heuristics.
register_provider(
    replace(
        get_provider(DEFAULT_PROVIDER),
        name="gb10-mixed-v2",
        factory="fissile.providers.gb10_cutlass:GB10CutlassProvider",
        resources=get_provider(DEFAULT_PROVIDER).resources
        + (ResourceSpec("fp4_gemm_library", "build_receipt"),),
    )
)

# Static NVFP4 activation scales (see providers/gb10_static.py):
# calibration keeps the gb10-mixed-v2 arithmetic and records per-site scales;
# static uses a calibrated scale file instead of per-forward tensor amax.
register_provider(
    replace(
        get_provider("gb10-mixed-v2"),
        name="gb10-mixed-v2-calibration",
        factory="fissile.providers.gb10_static:GB10CalibrationProvider",
    )
)
register_provider(
    replace(
        get_provider("gb10-mixed-v2"),
        name="gb10-mixed-v2-static",
        factory="fissile.providers.gb10_static:GB10StaticProvider",
        resources=get_provider("gb10-mixed-v2").resources
        + (ResourceSpec("activation_scales", "sha256"),),
    )
)
