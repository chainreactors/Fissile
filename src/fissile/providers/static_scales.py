"""CPU-only scale-file contract of static activation scales. The kernels live in
providers/gb10_static.py."""

import math
from fissile.config import read_json

SCALES_FORMAT = "fissile-activation-scales-v1"
CALIBRATION_FORMAT = "fissile-activation-calibration-v1"
CALIBRATION_OUTPUT = "FISSILE_ACTIVATION_CALIBRATION_OUTPUT"


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
