"""CPU-only model discovery. A registered profile runs only on backends that declare it."""

from .spec import MLP, ModelProfile, Projection, get_profile, register_profile
from .qwen35 import SPEC

register_profile(SPEC)
DEFAULT_PROFILE = SPEC.name
