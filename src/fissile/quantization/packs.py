"""Prepare every projection pack from input moments and write the manifest."""

import json
from pathlib import Path

from fissile.config import validate_manifest
from fissile.models import get_profile
from fissile.recipes import DEFAULT_RECIPE, get_recipe
from .prepare import METHOD


def check_outputs(profile, recipe, packs_dir, manifest):
    """Selected profile and recipe; refuses existing outputs and other methods."""
    profile, recipe = get_profile(profile), get_recipe(recipe)
    if profile.name not in recipe.profiles:
        raise ValueError(f"Recipe {recipe.name} does not support {profile.name}")
    if recipe.preparation_method != METHOD:
        raise ValueError(f"Packs are prepared with {METHOD}; {recipe.name} requires {recipe.preparation_method}")
    for path in (Path(packs_dir), Path(manifest)):
        if path.exists():
            raise FileExistsError(path)
    return profile, recipe


def prepare_packs(weights, moments, groups, packs_dir, manifest, *, profile, recipe=DEFAULT_RECIPE,
                  calibration=None, progress=None):
    """Write ``packs_dir/<key>.safetensors`` for every projection and the manifest.

    ``weights`` maps canonical keys to effective CUDA BF16 weights; ``moments``
    maps each group of ``groups`` to its full FP64 input moment and is consumed.
    """
    from .prepare import prepare_moment, prepare_projection

    profile, recipe = check_outputs(profile, recipe, packs_dir, manifest)
    shapes = profile.shapes()
    if set(weights) != set(shapes) or sorted(k for g in groups for k in g) != sorted(shapes):
        raise ValueError(f"Weights and groups must cover the {len(shapes)} {profile.name} projections")
    packs_dir = Path(packs_dir)
    packs_dir.mkdir(parents=True)
    projections = {}
    for group in groups:
        hessian = moments.pop(group)
        prepared = prepare_moment(hessian)
        for key in group:
            projections[key] = prepare_projection(
                weights[key], hessian, packs_dir / f"{key}.safetensors", prepared=prepared
            )
            if progress:
                progress("prepare", len(projections), len(shapes))
        del hessian, prepared
    m = dict(format="fissile-packs-v1", profile=profile.name)
    if recipe.name != DEFAULT_RECIPE:
        m["recipe"] = recipe.name
    m.update(complete=True, method=METHOD, labels_consumed=False,
             projections={key: projections[key] for key in shapes})
    if calibration is not None:
        m["calibration"] = calibration
    validate_manifest(m, packs_dir)
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    return m
