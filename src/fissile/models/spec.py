"""Engine-independent model topology. No numerical policy belongs here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    key: str
    shape: tuple[int, int]
    role: str = "body"


@dataclass(frozen=True)
class MLP:
    prefix: str
    gate: str
    up: str
    down: str

    @property
    def keys(self):
        return (self.gate, self.up, self.down)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    projections: tuple[Projection, ...]
    mlps: tuple[MLP, ...]
    vocab_size: int
    max_context: int

    def __post_init__(self):
        shapes = self.shapes()
        if not self.projections or len(shapes) != len(self.projections):
            raise ValueError("Empty or duplicate projection keys")
        if self.vocab_size <= 0 or self.max_context < 2:
            raise ValueError("Invalid vocabulary or context limit")
        if any(len(p.shape) != 2 or min(p.shape) <= 0 for p in self.projections):
            raise ValueError("Invalid projection shape")
        used = set()
        prefixes = set()
        for mlp in self.mlps:
            if (
                mlp.prefix in prefixes
                or len(set(mlp.keys)) != 3
                or used.intersection(mlp.keys)
            ):
                raise ValueError("Overlapping MLP groups")
            prefixes.add(mlp.prefix)
            used.update(mlp.keys)
            if not set(mlp.keys) <= shapes.keys():
                raise ValueError("MLP references unknown projections")
            gate, up, down = (shapes[k] for k in mlp.keys)
            if gate != up or down != gate[::-1]:
                raise ValueError("Incompatible gated MLP shapes")

    def shapes(self):
        return {p.key: list(p.shape) for p in self.projections}

    @property
    def head(self):
        """Canonical key of the single output-head projection (role ``head``)."""
        heads = [p.key for p in self.projections if p.role == "head"]
        if len(heads) != 1:
            raise ValueError("Profile must have exactly one head projection")
        return heads[0]


_PROFILES = {}


def register_profile(profile):
    if profile.name in _PROFILES:
        raise ValueError("Model profile already registered: " + profile.name)
    _PROFILES[profile.name] = profile


def get_profile(name):
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError("Unsupported model profile: " + str(name)) from None
