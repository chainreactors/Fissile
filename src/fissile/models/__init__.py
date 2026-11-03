"""CPU-only model discovery. Registration does not imply engine qualification."""

from .spec import MLP, ModelProfile, Projection, get_profile, register_profile
from .qwen35 import SPEC

register_profile(SPEC)
DEFAULT_PROFILE = SPEC.name
