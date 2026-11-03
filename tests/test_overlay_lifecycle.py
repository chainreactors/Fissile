"""Engine-free lifecycle tests: restore overrides after success and partial failure."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from fissile.backends.torch import optimized
from fissile.models.qwen35 import projection_shapes
from fissile.models import get_profile, DEFAULT_PROFILE
from fissile.recipes import ExecutionPlan, get_recipe, DEFAULT_RECIPE
from fissile.providers import DEFAULT_PROVIDER


class Linear:
    bias = None
    training = False

    def __init__(self):
        self.weight = object()

    def forward(self, x):
        return x


class Module:
    training = False

    def forward(self, x):
        return x


class Model(Module):
    def __init__(self):
        self.modules = {
            ("lm_head" if k == "lm_head" else "model.language_model." + k): Linear()
            for k in projection_shapes()
        }
        for i in range(32):
            self.modules[f"model.language_model.layers.{i}.mlp"] = Module()
        # Existing instance override is deliberately distinct from class forward.
        self.modules["lm_head"].forward = lambda x: ("prior", x)

    def parameters(self):
        return []

    def named_modules(self):
        return self.modules.items()

    def get_submodule(self, name):
        return self.modules[name]


class Runtime:
    fail = False

    def __init__(self, config):
        self.calls = {}
        self.profile = get_profile(DEFAULT_PROFILE)
        self.plan = ExecutionPlan(self.profile, get_recipe(DEFAULT_RECIPE))
        self.config = SimpleNamespace(
            profile=DEFAULT_PROFILE,
            recipe=DEFAULT_RECIPE,
            kernel_provider=DEFAULT_PROVIDER,
        )

    def projection(self, key, weight):
        if self.fail:
            raise ValueError("simulated failed identity check")
        return lambda x: x

    def mlp(self, prefix, *parts, **kwargs):
        return lambda x: x

    def validate_coverage(self):
        return {}


@contextmanager
def pointwise(model, **kwargs):
    mlp = model.modules["model.language_model.layers.0.mlp"]
    assert "forward" not in vars(mlp)
    mlp.forward = lambda x: ("pointwise", x)
    try:
        yield {}
    finally:
        del mlp.forward


class Lifecycle(unittest.TestCase):
    def run_case(self, fail_install=False, fail_body=False):
        model = Model()
        before = {name: dict(vars(m)) for name, m in model.named_modules()}
        runtime_type = type("ConfiguredRuntime", (Runtime,), {"fail": fail_install})
        fake = {
            "torch": SimpleNamespace(
                nn=SimpleNamespace(Linear=Linear), is_grad_enabled=lambda: False
            ),
            "fissile.runtime.mixed": SimpleNamespace(MixedRuntime=runtime_type),
            "fissile.backends.torch_pointwise": SimpleNamespace(
                pointwise_fusion=pointwise
            ),
        }
        with patch.dict("sys.modules", fake):
            try:
                with optimized(model, "external-config"):
                    if fail_body:
                        raise RuntimeError("caller failed")
            except (ValueError, RuntimeError):
                if not (fail_install or fail_body):
                    raise
        for name, module in model.named_modules():
            self.assertEqual(vars(module), before[name], name)
        self.assertNotIn("forward", vars(model))

    def test_normal_restore(self):
        self.run_case()

    def test_partial_install_restore(self):
        self.run_case(fail_install=True)

    def test_caller_exception_restore(self):
        self.run_case(fail_body=True)


if __name__ == "__main__":
    unittest.main()
