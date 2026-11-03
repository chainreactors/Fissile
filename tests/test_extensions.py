"""Exercise an external backend/provider on a different topology without GPU imports."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import json
import os
import subprocess
import sys
import tempfile
import unittest

from fissile.backends.contracts import Score
from fissile.backends.registry import BackendSpec, register_backend, open_backend
from fissile.config import Config, sha256
from fissile.models import (
    ModelProfile,
    Projection,
    get_profile,
    register_profile,
    DEFAULT_PROFILE,
)
from fissile.providers import (
    ProviderSpec,
    get_provider,
    register_provider,
    DEFAULT_PROVIDER,
)
from fissile.recipes import (
    Recipe,
    ExecutionPlan,
    get_recipe,
    register_recipe,
    DEFAULT_RECIPE,
)


PROFILE = ModelProfile(
    "test-two-projections",
    (Projection("body", (128, 128)), Projection("readout", (256, 128), "head")),
    (),
    256,
    16,
)
RECIPE = Recipe(
    "test-unfused",
    frozenset({PROFILE.name}),
    (("body", "test-linear"), ("head", "test-linear")),
    "test-preparation",
)
PROVIDER = ProviderSpec(
    "test-cpu",
    "external_test_plugin:Provider",
    frozenset({PROFILE.name}),
    frozenset({RECIPE.name}),
    (0, 0),
    "bfloat16",
    frozenset({"test-linear"}),
    frozenset(),
    frozenset(),
    1,
)
BACKEND = BackendSpec(
    "test-external",
    "external_test_plugin:session",
    frozenset({(PROFILE.name, RECIPE.name, PROVIDER.name)}),
)
register_profile(PROFILE)
register_recipe(RECIPE)
register_provider(PROVIDER)
register_backend(BACKEND)


class Matrix:
    def __init__(self, shape, value=1):
        self.shape, self.value = shape, value

    def reshape(self, *shape):
        return Matrix(shape, self.value)


class Extensions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        projections = {}
        for p in PROFILE.projections:
            artifact = self.root / (p.key + ".pack")
            artifact.write_bytes(p.key.encode())
            projections[p.key] = dict(
                path=artifact.name,
                shape=list(p.shape),
                sha256=sha256(artifact),
                effective_weight_sha256="a" * 64,
            )
        (self.root / "manifest.json").write_text(
            json.dumps(
                dict(
                    format="fissile-packs-v1",
                    profile=PROFILE.name,
                    recipe=RECIPE.name,
                    complete=True,
                    method=RECIPE.preparation_method,
                    labels_consumed=False,
                    projections=projections,
                )
            )
        )
        self.data = dict(
            format="fissile-config-v2",
            profile=PROFILE.name,
            recipe=RECIPE.name,
            kernel_provider=PROVIDER.name,
            manifest="manifest.json",
            artifacts_dir=".",
            report_dir="reports",
            resources={},
        )
        self.path = self.root / "config.json"
        self.write_config()

    def write_config(self):
        self.path.write_text(json.dumps(self.data))

    def plugin(self, *, broken_result=False):
        from fissile.runtime.mixed import MixedRuntime

        module = ModuleType("external_test_plugin")
        events = []

        class Provider:
            def __init__(self, config):
                events.append("provider")

            def matrix(self, x):
                return x

            def projection(self, route, weight, pack):
                assert route == "test-linear" and pack is None
                return lambda x: Matrix((x.shape[0], weight.shape[0]), x.value * 2)

        @contextmanager
        def session(config, model):
            runtime = MixedRuntime(config)
            functions = [
                runtime.projection(p.key, Matrix(p.shape)) for p in PROFILE.projections
            ]
            report = runtime.validate_coverage()
            assert report["routes"] == {"test-linear": 2, "paired_mlp": 0}

            class Session:
                count = 0

                def score(self, ids, candidates):
                    x = Matrix((len(ids), 128))
                    for f in functions:
                        x = f(x)
                    self.count += 1
                    runtime.validate_forward(self.count)
                    return Score(
                        tuple(-float(x.value + i) for i in candidates),
                        0 if broken_result else len(ids),
                    )

            events.append("opened")
            try:
                yield Session()
            finally:
                events.append("closed")

        module.Provider, module.session = Provider, session
        return module, events

    @contextmanager
    def external(self, **kwargs):
        module, events = self.plugin(**kwargs)
        with patch.dict(
            sys.modules,
            {
                "external_test_plugin": module,
                "fissile.kernels.packs": SimpleNamespace(
                    weight_hash=lambda w: "a" * 64
                ),
            },
        ):
            yield events

    def test_external_backend_reuses_runtime_with_new_profile_recipe_and_provider(self):
        with self.external() as events:
            with open_backend(BACKEND.name, self.path, None) as session:
                self.assertEqual(
                    session.score([1, 2, 3], [4, 5]), Score((-8.0, -9.0), 3)
                )
                self.assertEqual(session.score([2, 3], [1]).prompt_tokens, 2)
            self.assertEqual(events, ["provider", "opened", "closed"])
            with self.assertRaises(RuntimeError):
                session.score([1, 2], [3])

    def test_session_closes_on_caller_exception(self):
        with self.external() as events:
            with self.assertRaisesRegex(RuntimeError, "caller"):
                with open_backend(BACKEND.name, self.path, None):
                    raise RuntimeError("caller")
            self.assertEqual(events[-1], "closed")

    def test_runtime_rejects_missing_duplicate_or_bypassed_projections(self):
        from fissile.runtime.mixed import MixedRuntime

        with self.external():
            runtime = MixedRuntime(self.path)
            first, second = PROFILE.projections
            f = runtime.projection(first.key, Matrix(first.shape))
            with self.assertRaisesRegex(ValueError, "already installed"):
                runtime.projection(first.key, Matrix(first.shape))
            with self.assertRaisesRegex(ValueError, "coverage"):
                runtime.validate_coverage()
            g = runtime.projection(second.key, Matrix(second.shape))
            runtime.validate_coverage()
            f(Matrix((2, 128)))
            with self.assertRaisesRegex(RuntimeError, "bypass"):
                runtime.validate_forward(1)
            g(Matrix((2, 128)))
            runtime.validate_forward(1)
            f(Matrix((2, 128)))
            with self.assertRaisesRegex(RuntimeError, "extra execution"):
                runtime.validate_forward(1)

    def test_backend_result_contract_and_cleanup(self):
        with self.external(broken_result=True) as events:
            with self.assertRaisesRegex(RuntimeError, "result contract"):
                with open_backend(BACKEND.name, self.path, None) as session:
                    session.score([1, 2], [3])
            self.assertEqual(events[-1], "closed")

    def test_invalid_inputs_do_not_execute_backend(self):
        with self.external():
            with open_backend(BACKEND.name, self.path, None) as session:
                for ids, candidates in [
                    ([1], [3]),
                    ([1, 2], [3, 3]),
                    ([1, True], [3]),
                    ([1, 2], [256]),
                    ([1, 2], []),
                ]:
                    with self.assertRaises(ValueError):
                        session.score(ids, candidates)
                self.assertEqual(session.session.count, 0)

    def test_resources_follow_provider_not_gb10_defaults(self):
        cfg = Config.load(self.path)
        self.assertEqual(dict(cfg.resources), {})
        self.data["resources"] = {"unrequested": {"path": "missing"}}
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Resources"):
            Config.load(self.path, verify_files=False)

    def test_unsupported_combination_rejected_before_engine_import(self):
        with self.assertRaisesRegex(ValueError, "Backend does not support"):
            with open_backend("sglang", self.path, None):
                self.fail("must not open")
        self.assertNotIn("sglang", sys.modules)

    def test_capability_dtype_shape_and_fusion_checks(self):
        plan = ExecutionPlan(get_profile(DEFAULT_PROFILE), get_recipe(DEFAULT_RECIPE))
        provider = get_provider(DEFAULT_PROVIDER)
        provider.validate(plan, capability=(12, 1))
        for kwargs in [dict(capability=(9, 0)), dict(activation_dtype="float16")]:
            with self.assertRaises(ValueError):
                provider.validate(plan, **kwargs)
        for changed in [
            replace(provider, alignment=8192),
            replace(provider, routes=frozenset({"fp4", "w4a16"})),
            replace(provider, fusions=frozenset()),
        ]:
            with self.assertRaises(ValueError):
                changed.validate(plan)

    def test_recipe_fusion_incompatibility(self):
        recipe = replace(
            get_recipe(DEFAULT_RECIPE),
            role_routes=(("body", "fp8"), ("attention_kv", "fp8"), ("head", "w4a16")),
        )
        with self.assertRaisesRegex(ValueError, "Paired MLP"):
            ExecutionPlan(get_profile(DEFAULT_PROFILE), recipe)

    def test_duplicate_registration_is_rejected(self):
        for fn, obj in [
            (register_backend, BACKEND),
            (register_provider, PROVIDER),
            (register_recipe, RECIPE),
            (register_profile, PROFILE),
        ]:
            with self.assertRaisesRegex(ValueError, "already registered"):
                fn(obj)

    def test_manifest_profile_mismatch_is_rejected(self):
        p = self.root / "manifest.json"
        manifest = json.loads(p.read_text())
        manifest["profile"] = DEFAULT_PROFILE
        p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "profiles differ"):
            Config.load(self.path)

    def test_discovery_and_runtime_import_remain_cpu_only(self):
        import fissile

        code = "import sys; from fissile.backends.registry import describe_backends; from fissile.runtime.mixed import MixedRuntime; from fissile.config import Config; assert len(describe_backends()) == 4; assert not {'torch','sglang','vllm','triton','cutlass'} & sys.modules.keys()"
        env = dict(
            os.environ, PYTHONPATH=str(Path(fissile.__file__).resolve().parent.parent)
        )
        subprocess.run([sys.executable, "-S", "-c", code], env=env, check=True)

    def test_sglang_shutdown_error_still_restores_environment(self):
        from fissile.backends.sglang.engine import scoring_engine

        cfg = SimpleNamespace(
            profile=DEFAULT_PROFILE,
            recipe=DEFAULT_RECIPE,
            kernel_provider=DEFAULT_PROVIDER,
            plan=ExecutionPlan(
                get_profile(DEFAULT_PROFILE), get_recipe(DEFAULT_RECIPE)
            ),
        )

        class Engine:
            def __init__(self, **kwargs):
                pass

            def shutdown(self):
                raise RuntimeError("shutdown failed")

        with (
            patch.dict(sys.modules, {"sglang": SimpleNamespace(Engine=Engine)}),
            patch("fissile.backends.sglang.engine.Config.load", return_value=cfg),
            patch.dict(os.environ, {"FISSILE_CONFIG": "previous-config"}),
        ):
            old_plugin = os.environ.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            try:
                with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                    with scoring_engine(self.path, self.root):
                        pass
                self.assertEqual(os.environ["FISSILE_CONFIG"], "previous-config")
                self.assertNotIn("SGLANG_EXTERNAL_MODEL_PACKAGE", os.environ)
            finally:
                if old_plugin is not None:
                    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = old_plugin
