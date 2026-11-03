"""Explicit SM120 target; shares the existing SM12x mixed-precision arithmetic."""

from fissile.providers.gb10 import GB10Provider


class SM120Provider(GB10Provider):
    def __init__(self, config):
        if config.provider.capability != (12, 0):
            raise ValueError("SM120 provider requires capability (12, 0)")
        super().__init__(config)
