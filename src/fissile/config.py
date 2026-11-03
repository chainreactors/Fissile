"""CPU-only configuration and manifest validation; fail closed for unsupported models."""

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from types import MappingProxyType
from .models import DEFAULT_PROFILE, get_profile
from .recipes import DEFAULT_RECIPE, ExecutionPlan, get_recipe
from .providers import DEFAULT_PROVIDER, get_provider


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA256")
    return value


def contained(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact entry must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact entry escapes artifact root")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_manifest(m, root, *, verify_files=True, recipe=None, profile=None):
    if m.get("format") != "fissile-packs-v1" or m.get("complete") is not True:
        raise ValueError("Incomplete or unsupported pack manifest")
    selected = get_profile(m.get("profile"))
    if profile is not None and selected.name != profile:
        raise ValueError("Manifest and configuration profiles differ")
    recipe = get_recipe(recipe or m.get("recipe", DEFAULT_RECIPE))
    ExecutionPlan(selected, recipe)
    if m.get("recipe", recipe.name) != recipe.name:
        raise ValueError("Manifest and configuration recipes differ")
    if (
        m.get("method") != recipe.preparation_method
        or m.get("labels_consumed") is not False
    ):
        raise ValueError("Pack preparation does not match the unlabeled recipe")
    projections = m.get("projections", {})
    shapes = selected.shapes()
    if set(projections) != set(shapes):
        raise ValueError(
            f"Exactly {len(shapes)} expected model projections are required"
        )
    paths = set()
    for key, entry in projections.items():
        if entry.get("shape") != shapes[key]:
            raise ValueError("Wrong projection shape: " + key)
        digest(entry["sha256"])
        digest(entry["effective_weight_sha256"])
        path = contained(root, entry["path"])
        if path in paths:
            raise ValueError("Duplicate packed projection path")
        paths.add(path)
        if verify_files and sha256(path) != entry["sha256"]:
            raise ValueError("Packed bytes changed: " + key)
    return m


@dataclass(frozen=True)
class Resource:
    path: Path
    sha256: str | None = None


@dataclass(frozen=True)
class Config:
    manifest: Path
    artifacts_dir: Path
    report_dir: Path
    resources: object
    profile: str = DEFAULT_PROFILE
    recipe: str = DEFAULT_RECIPE
    kernel_provider: str = DEFAULT_PROVIDER

    def __post_init__(self):
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))

    @property
    def plan(self):
        return ExecutionPlan(get_profile(self.profile), get_recipe(self.recipe))

    @property
    def provider(self):
        return get_provider(self.kernel_provider)

    # Read-only compatibility accessors for v1 callers.
    @property
    def tensorfold_library(self):
        return self.resources["tensorfold_library"].path

    @property
    def silu_lut(self):
        return self.resources["silu_lut"].path

    @property
    def silu_lut_sha256(self):
        return self.resources["silu_lut"].sha256

    def validate(self, *, verify_files=True):
        self.provider.validate(self.plan)
        required = {r.name: r for r in self.provider.resources}
        if set(self.resources) != set(required):
            raise ValueError("Resources must exactly match selected kernel provider")
        for name, resource in self.resources.items():
            spec = required[name]
            if spec.integrity == "sha256":
                digest(resource.sha256)
            elif resource.sha256 is not None:
                raise ValueError(
                    "Build-receipt resource must not supply a separate hash"
                )
            if verify_files:
                actual = sha256(resource.path)
                expected = (
                    resource.sha256
                    if spec.integrity == "sha256"
                    else read_json(resource.path.with_suffix(".build.json")).get(
                        "library_sha256"
                    )
                )
                if actual != expected:
                    raise ValueError(
                        "Resource differs from its integrity record: " + name
                    )
        if verify_files:
            validate_manifest(
                read_json(self.manifest),
                self.artifacts_dir,
                recipe=self.recipe,
                profile=self.profile,
            )
        return self

    @classmethod
    def load(cls, path, *, verify_files=True):
        path = Path(path).resolve()
        data = read_json(path)
        common = {"format", "profile", "manifest", "artifacts_dir", "report_dir"}
        if data.get("format") == "fissile-config-v1":
            if set(data) != common | {
                "tensorfold_library",
                "silu_lut",
                "silu_lut_sha256",
            }:
                raise ValueError("Unknown or missing legacy configuration fields")
            resources = {
                "tensorfold_library": {"path": data["tensorfold_library"]},
                "silu_lut": {
                    "path": data["silu_lut"],
                    "sha256": data["silu_lut_sha256"],
                },
            }
            recipe, provider = DEFAULT_RECIPE, DEFAULT_PROVIDER
        elif data.get("format") == "fissile-config-v2":
            if set(data) != common | {"recipe", "kernel_provider", "resources"}:
                raise ValueError("Unknown or missing configuration fields")
            resources = data["resources"]
            recipe, provider = data["recipe"], data["kernel_provider"]
        else:
            raise ValueError("Unsupported configuration format")
        resolved = {}
        for name, entry in resources.items():
            if (
                not isinstance(entry, dict)
                or "path" not in entry
                or set(entry) - {"path", "sha256"}
            ):
                raise ValueError("Invalid resource descriptor: " + name)
            resolved[name] = Resource(
                (path.parent / entry["path"]).resolve(), entry.get("sha256")
            )
        cfg = cls(
            **{
                k: (path.parent / data[k]).resolve()
                for k in ("manifest", "artifacts_dir", "report_dir")
            },
            resources=resolved,
            profile=data["profile"],
            recipe=recipe,
            kernel_provider=provider,
        )
        return cfg.validate(verify_files=verify_files)


def require_gpu(capability=(12, 1)):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != tuple(
        capability
    ):
        raise RuntimeError(
            f"Kernel provider requires CUDA capability {tuple(capability)}"
        )
