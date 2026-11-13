"""Validated engine stacks and the engine upgrade check (CPU only)."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from fissile.backends import engines
from fissile.backends.engines import STACK, VALIDATED, UnvalidatedEngineError, check_engine, engine_status
from fissile.backends.registry import describe_backends, get_backend, open_backend

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("check_engine", ROOT / "scripts/check_engine.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
spec = importlib.util.spec_from_file_location("engine_env", ROOT / "scripts/engine_env.py")
engine_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_env)

RELEASE, STACK_0521 = next(iter(VALIDATED["sglang"].items()))


def installed(versions):
    """Fake package metadata: {package: version}; missing packages are not installed."""
    def fake(name):
        if name not in versions:
            raise engines.metadata.PackageNotFoundError(name)
        return versions[name]

    return mock.patch.object(engines.metadata, "version", fake)


def environment(**changes):
    return {"sglang": RELEASE + "+local", **{p: v + "+cu130" for p, v in STACK_0521.items()}, **changes}


def config_file(directory):
    path = Path(directory) / "c.json"
    path.write_text(json.dumps({
        "format": "fissile-config-v2", "profile": "qwen3.5-dense-9b", "recipe": "mixed-fp4-fp8-w4a16-v1",
        "kernel_provider": "gb10-mixed-v2", "manifest": "m.json", "artifacts_dir": "packs",
        "report_dir": "reports",
        "resources": {"tensorfold_library": {"path": "tf.so"},
                      "silu_lut": {"path": "silu.safetensors", "sha256": "0" * 64},
                      "fp4_gemm_library": {"path": "gemm.so"}}}))
    return path


class Stacks(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(engines.ALLOW_ENV, None)

    def test_backends_name_their_engine_and_stacks_are_complete(self):
        self.assertEqual({n: get_backend(n).engine for n in ("torch", "sglang", "sglang-candidates", "vllm")},
                         {"torch": None, "sglang": "sglang", "sglang-candidates": "sglang", "vllm": "vllm"})
        for engine, releases in VALIDATED.items():
            for release, stack in releases.items():
                self.assertEqual(set(stack), set(STACK), (engine, release))
        with installed(environment()):
            described = describe_backends()["sglang"]
        self.assertEqual(described["validated_engines"], VALIDATED["sglang"])
        self.assertEqual(described["installed_engine_version"], RELEASE)

    def test_validated_release_needs_its_stack(self):
        with installed(environment()):
            status = check_engine("sglang")
        self.assertTrue(status["engine_validated"])
        self.assertEqual(status["stack"], STACK_0521)
        with installed(environment(triton="99.0.0")):
            self.assertEqual(engine_status("sglang")["stack_mismatch"], ["triton"])
            with self.assertRaisesRegex(UnvalidatedEngineError, "triton 99.0.0 .validated: "):
                check_engine("sglang")
            self.assertFalse(check_engine("sglang", allow_unvalidated=True)["engine_validated"])
        missing = environment()
        del missing["flashinfer-python"]
        with installed(missing):
            self.assertEqual(engine_status("sglang")["stack_mismatch"], ["flashinfer-python"])

    def test_unknown_release_or_engine_missing(self):
        with installed(environment(sglang="99.0.0")):
            with self.assertRaisesRegex(UnvalidatedEngineError, "is not validated"):
                check_engine("sglang")
            with mock.patch.dict(os.environ, {engines.ALLOW_ENV: "1"}):
                self.assertEqual(check_engine("sglang")["engine_version"], "99.0.0")
        with installed({}):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                check_engine("vllm", allow_unvalidated=True)

    def test_open_backend_refuses_before_importing_the_engine(self):
        with tempfile.TemporaryDirectory() as d:
            loaded = set(sys.modules)
            with installed(environment(sglang="99.0.0")):
                with self.assertRaises(UnvalidatedEngineError):
                    with open_backend("sglang-candidates", config_file(d), "model"):
                        pass
            self.assertFalse({"sglang", "fissile.backends.sglang.engine"} & (set(sys.modules) - loaded))


ROWS = {
    "a": {"id": "a", "labels": ["x", "y"], "expected": "x"},
    "b": {"id": "b", "labels": ["x", "y"], "expected": "y"},
    "c": {"id": "c", "labels": ["x", "y"], "expected": "x"},
}
RIGHT, WRONG = [-0.1, -2.0], [-2.0, -0.1]  # favour "x" / "y"


def records(*logprobs, ms=10.0):
    return [dict(id=i, logprobs=lp, elapsed_ms=ms) for i, lp in zip("abc", logprobs)]


class UpgradeCheck(unittest.TestCase):
    def test_compare_records_and_accuracy(self):
        reference = records(RIGHT, RIGHT, WRONG)
        candidate = records(RIGHT, WRONG, WRONG, ms=8.0)
        out = check.compare_records(candidate, reference, ROWS)
        self.assertEqual((out["compared"], out["bitwise_equal"], out["decisions_changed"]), (3, 2, 1))
        self.assertAlmostEqual(out["max_logprob_diff"], 1.9)
        self.assertEqual((out["only_reference_correct"], out["only_candidate_correct"]), (0, 1))
        self.assertAlmostEqual(out["mcnemar_z"], 1.0)
        self.assertEqual((out["mean_ms"], out["reference_mean_ms"]), (8.0, 10.0))
        self.assertEqual(check.accuracy(candidate, ROWS), dict(labelled=3, correct=2, accuracy=2 / 3))
        with self.assertRaises(ValueError):
            check.compare_records(candidate, [dict(id="z", logprobs=[0.0], elapsed_ms=1.0)], ROWS)

    def test_gate_uses_the_reference_perturbed_run(self):
        arms = dict(fissile=records(RIGHT, RIGHT, WRONG), bf16=records(RIGHT, WRONG, RIGHT))
        reference = dict(fissile=records(RIGHT, WRONG, RIGHT), perturbed=records(RIGHT, RIGHT, WRONG),
                         bf16=records(RIGHT, WRONG, RIGHT))
        comparisons, gate = check.compare_with_reference(arms, reference, ROWS)
        self.assertEqual(gate, "fissile_vs_perturbed")
        self.assertEqual(sorted(comparisons), ["bf16_vs_bf16", "fissile_vs_fissile", "fissile_vs_perturbed"])
        self.assertEqual(comparisons["fissile_vs_perturbed"]["bitwise_equal"], 3)
        _, gate = check.compare_with_reference(arms, dict(fissile=reference["fissile"]), ROWS)
        self.assertEqual(gate, "fissile_vs_fissile")

    def test_verdict(self):
        good = dict(mcnemar_z=-0.5, accuracy=0.9)
        report = dict(install=dict(complete=True), repeat=dict(scored=4, equal=4), batch=dict(scored=8, equal=8),
                      reference=dict(gate="fissile_vs_perturbed",
                                     comparisons=dict(fissile_vs_perturbed=good, bf16_vs_bf16=good,
                                                      fissile_vs_fissile=dict(mcnemar_z=-3.0, accuracy=0.8))))
        result = check.verdict(report)
        self.assertTrue(result["passed"], result)  # the reference's own fissile run is not the gate
        self.assertEqual(set(result["checks"]),
                         {"installed", "repeatable", "batch_exact", "bf16_not_worse", "fissile_not_worse"})
        bad = dict(mcnemar_z=-2.5, accuracy=0.8)
        for change in (dict(repeat=dict(scored=4, equal=3)), dict(batch=dict(scored=8, equal=7)),
                       dict(install=dict(complete=False))):
            with self.subTest(change=change):
                self.assertFalse(check.verdict({**report, **change})["passed"])
        for key in ("fissile_vs_perturbed", "bf16_vs_bf16"):
            with self.subTest(key=key):
                worse = json.loads(json.dumps(report))
                worse["reference"]["comparisons"][key] = bad
                self.assertFalse(check.verdict(worse)["passed"])
        minimal = dict(install=dict(complete=True), repeat=dict(scored=1, equal=1))
        self.assertEqual(check.verdict(minimal)["checks"], dict(installed=True, repeatable=True))

    def test_phase_config_paths_do_not_depend_on_the_working_directory(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            config_file(d)
            (d / "out").mkdir()
            cwd = os.getcwd()
            os.chdir(d)
            try:
                path, reports = check.phase_config(Path("c.json"), Path("out"), "fissile")
            finally:
                os.chdir(cwd)
            data = json.loads((d / path).read_text())
            self.assertEqual(reports, (d / "out/reports-fissile").resolve())
            self.assertEqual(data["report_dir"], str(reports))
            self.assertEqual(data["manifest"], str((d / "m.json").resolve()))


class EngineEnvironment(unittest.TestCase):
    def test_plan(self):
        commands = engine_env.plan("sglang==0.5.23", "/env", torch_index="https://index", fissile="/src",
                                   python="/usr/bin/python3")
        self.assertEqual(commands[0], ["/usr/bin/python3", "-m", "venv", "/env"])
        self.assertEqual(commands[2], ["/env/bin/python", "-m", "pip", "install", "--extra-index-url",
                                       "https://index", "sglang==0.5.23"])
        self.assertEqual(commands[3][-2:], ["--no-deps", "/src"])
        for bad in ("sglang", "sglang>=0.5", "torch==2.14.1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                engine_env.plan(bad, "/env")


if __name__ == "__main__":
    unittest.main()
