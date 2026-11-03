"""Explicit numerical recipes, independent of model layout and engine imports."""

from dataclasses import dataclass
from types import MappingProxyType
from fissile.models.spec import ModelProfile, Projection


@dataclass(frozen=True)
class Recipe:
    name: str
    profiles: frozenset[str]
    role_routes: tuple[tuple[str, str], ...]
    preparation_method: str
    fusions: frozenset[str] = frozenset()

    def __post_init__(self):
        if len(dict(self.role_routes)) != len(self.role_routes):
            raise ValueError("Duplicate role routing rules")
        if not self.profiles or not self.preparation_method:
            raise ValueError("Explicit model support and preparation method required")

    def route(self, projection: Projection):
        try:
            return dict(self.role_routes)[projection.role]
        except KeyError:
            raise ValueError(
                "Recipe has no route for role: " + projection.role
            ) from None


@dataclass(frozen=True)
class ExecutionPlan:
    profile: ModelProfile
    recipe: Recipe

    def __post_init__(self):
        if self.profile.name not in self.recipe.profiles:
            raise ValueError("Recipe does not support model profile")
        routes = self.routes
        if self.recipe.fusions - {"paired_mlp"}:
            raise ValueError("Unsupported runtime fusion contract")
        if "paired_mlp" in self.recipe.fusions:
            for mlp in self.profile.mlps:
                if any(routes[k] != "fp4" for k in mlp.keys):
                    raise ValueError("Paired MLP requires three FP4 projections")

    @property
    def routes(self):
        return MappingProxyType(
            {p.key: self.recipe.route(p) for p in self.profile.projections}
        )

    @property
    def fused_mlps(self):
        return self.profile.mlps if "paired_mlp" in self.recipe.fusions else ()


_RECIPES = {}


def register_recipe(recipe):
    if recipe.name in _RECIPES:
        raise ValueError("Recipe already registered: " + recipe.name)
    _RECIPES[recipe.name] = recipe


def get_recipe(name):
    try:
        return _RECIPES[name]
    except KeyError:
        raise ValueError("Unsupported recipe: " + str(name)) from None


DEFAULT_RECIPE = "mixed-fp4-fp8-w4a16-v1"
register_recipe(
    Recipe(
        name=DEFAULT_RECIPE,
        profiles=frozenset({"qwen3.5-dense-9b"}),
        role_routes=(("body", "fp4"), ("attention_kv", "fp8"), ("head", "w4a16")),
        preparation_method="full-h16-order",
        fusions=frozenset({"paired_mlp"}),
    )
)
