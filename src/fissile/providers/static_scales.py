"""CPU-only parts of static activation scales: the scale-file contract and the
calibration/static variants of a gb10-mixed-v2 configuration. The kernels live
in providers/gb10_static.py."""

import json
import math
from pathlib import Path
from fissile.config import read_json

BASE_PROVIDER = "gb10-mixed-v2"
SCALES_FORMAT = "fissile-activation-scales-v1"
CALIBRATION_FORMAT = "fissile-activation-calibration-v1"
CALIBRATION_OUTPUT = "FISSILE_ACTIVATION_CALIBRATION_OUTPUT"


def derived_config(config, directory, *, scales=None):
    """Write a gb10-mixed-v2 configuration as its calibration variant
    (``scales=None``) or its static-scale variant into ``directory``.

    Paths become absolute; the scale file is recorded with its sha256.
    Returns the path of the written configuration.
    """
    from fissile.config import Config, sha256

    cfg = Config.load(config, verify_files=False)
    if cfg.kernel_provider != BASE_PROVIDER:
        raise ValueError("Static activation scales extend a gb10-mixed-v2 configuration")
    data = cfg.as_dict()
    if scales is None:
        data["kernel_provider"] = "gb10-mixed-v2-calibration"
    else:
        scales = Path(scales).resolve(strict=True)
        data["kernel_provider"] = "gb10-mixed-v2-static"
        data["resources"]["activation_scales"] = {"path": str(scales), "sha256": sha256(scales)}
    path = Path(directory) / "fissile.json"
    with path.open("x") as f:
        json.dump(data, f, indent=2)
    return path


def load_activation_scales(path, *, profile, recipe):
    data = read_json(path)
    if data.get("format") != SCALES_FORMAT:
        raise ValueError("Not a " + SCALES_FORMAT + " file")
    if data.get("profile") != profile or data.get("recipe") != recipe:
        raise ValueError("Activation scales were calibrated for another profile/recipe")
    scales = data.get("scales")
    if not isinstance(scales, dict) or not scales:
        raise ValueError("Activation scale file has no scales")
    for site, value in scales.items():
        if (
            not isinstance(site, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid activation scale for site " + str(site))
    return {site: float(value) for site, value in scales.items()}
