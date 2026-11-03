"""Portable import of previously prepared weights. No calibration, training or labels."""

from pathlib import Path
import shutil
from .config import read_json, sha256, validate_manifest, contained
from .models.qwen35 import PROFILE, canonical_key
import json


def import_legacy(prepared, output):
    prepared = Path(prepared).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    summary = read_json(prepared / "summary.json")
    protocol = read_json(prepared / "protocol.json")
    if not summary.get("complete") or (prepared / "failed.json").exists():
        raise ValueError("Prepared run is incomplete")
    if (
        protocol.get("labels_consumed") is not False
        or protocol.get("holdout_consumed") is not False
    ):
        raise ValueError("Calibration labels/holdout must not have been consumed")
    if (
        summary.get("method") != "full-h16-order"
        or protocol.get("method") != "full-h16-order"
    ):
        raise ValueError("Unsupported preparation method")
    if summary.get("adapter_sha256_after") != protocol.get("adapter_sha256"):
        raise ValueError("Prepared adapter identity does not reconcile")
    projections = {}
    for old, e in summary["projections"].items():
        key = canonical_key(old)
        if key in projections:
            raise ValueError("Duplicate canonical projection")
        projections[key] = {
            k: e[k] for k in ("path", "shape", "sha256", "effective_weight_sha256")
        }
    m = dict(
        format="fissile-packs-v1",
        profile=PROFILE,
        complete=True,
        method="full-h16-order",
        labels_consumed=False,
        projections=projections,
        source_summary_sha256=sha256(prepared / "summary.json"),
    )
    validate_manifest(m, prepared)
    # Manifest-only import: weights stay external, unchanged and untracked.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(m, f, indent=2)
        f.write(chr(10))
    return m


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
