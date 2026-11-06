"""Portable copies of the projection packs listed in a manifest."""

from pathlib import Path
import shutil
from .config import read_json, validate_manifest, contained


def export_bundle(manifest, artifacts_dir, output):
    """Copy only approved packed files to a relocatable external bundle."""
    m = validate_manifest(read_json(manifest), artifacts_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for e in m["projections"].values():
        target = contained(output, e["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained(artifacts_dir, e["path"]), target)
    validate_manifest(m, output)
    with (output / "manifest.json").open("x") as f:
        json.dump(m, f, indent=2)
    return output
