"""CPU-only checks for fissile quantize: input groups, calibration files and output checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import contextlib
import io

from fissile.backends.layouts import FusedLinear, Layout, input_groups, register_layout
from fissile.models import ModelProfile, Projection, get_profile
from fissile.quantization.calibration import read_calibration
from fissile.quantization.packs import check_outputs, prepare_packs
from fissile.recipes import Recipe, register_recipe

PROFILE = get_profile("qwen3.5-dense-9b")


def toy(name, keys):
    profile = ModelProfile(name, tuple(Projection(k, (128, 128)) for k in keys)
                           + (Projection("readout", (128, 128), "head"),), (), 128, 8)
    register_layout(Layout(profile=name, layers="model.layers", head="lm_head", mlp="mlp",
                           linears=(FusedLinear("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
                                    FusedLinear("mlp.down_proj", ("mlp.down_proj",))),
                           hf_prefix="model.", hf_head="lm_head"))
    return profile


class InputGroups(unittest.TestCase):
    def test_every_projection_once_by_shared_input(self):
        groups = input_groups(PROFILE)
        keys = [k for g in groups for k in g]
        self.assertEqual(len(groups), 129)
        self.assertEqual(sorted(keys), sorted(PROFILE.shapes()))
        self.assertEqual(groups[-1], ("lm_head",))
        self.assertIn(("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"), groups)
        self.assertIn(("layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z"), groups)
        self.assertIn(("layers.3.self_attn.q_proj", "layers.3.self_attn.k_proj", "layers.3.self_attn.v_proj"), groups)
        self.assertIn(("layers.3.self_attn.o_proj",), groups)

    def test_layout_groups_and_partial_coverage(self):
        full = toy("test-quantize-groups", ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj",
                                            "layers.0.mlp.down_proj"))
        self.assertEqual(input_groups(full), [("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"),
                                              ("layers.0.mlp.down_proj",), ("readout",)])
        partial = toy("test-quantize-partial", ("layers.0.mlp.gate_proj", "layers.0.mlp.down_proj"))
        with self.assertRaises(ValueError):
            input_groups(partial)
        uncovered = toy("test-quantize-uncovered", ("layers.0.mlp.down_proj", "layers.0.extra"))
        with self.assertRaises(ValueError):
            input_groups(uncovered)


class Calibration(unittest.TestCase):
    def read(self, lines, profile=PROFILE):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.jsonl"
            path.write_text("\n".join(lines) + "\n")
            return read_calibration(path, profile)

    def test_reads_token_ids_and_ignores_other_fields(self):
        rows = self.read([json.dumps({"id": "a", "input_ids": [1, 2, 3], "labels": ["x"]}), "",
                          json.dumps({"input_ids": [PROFILE.vocab_size - 1]})])
        self.assertEqual(rows, [[1, 2, 3], [PROFILE.vocab_size - 1]])

    def test_rejects_invalid_inputs(self):
        bad = [{"ids": [1]}, {"input_ids": []}, {"input_ids": [PROFILE.vocab_size]}, {"input_ids": [-1]},
               {"input_ids": [True]}, {"input_ids": [1.0]}, {"input_ids": "12"}, [1, 2]]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.read([json.dumps(row)])
        short = ModelProfile("test-quantize-context", (Projection("readout", (128, 128), "head"),), (), 128, 4)
        with self.assertRaises(ValueError):
            self.read([json.dumps({"input_ids": [1] * 5})], short)
        with self.assertRaises(ValueError):
            self.read([""])


class Outputs(unittest.TestCase):
    def test_refuses_existing_outputs_and_other_methods(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            profile, recipe = check_outputs(PROFILE.name, "mixed-fp4-fp8-w4a16-v1", root / "packs", root / "m.json")
            self.assertEqual((profile.name, recipe.preparation_method), (PROFILE.name, "full-h16-order"))
            (root / "m.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                check_outputs(PROFILE.name, recipe.name, root / "packs", root / "m.json")
            register_recipe(Recipe("test-quantize-recipe", frozenset({PROFILE.name}),
                                    (("body", "fp4"), ("attention_kv", "fp8"), ("head", "w4a16")), "other-method"))
            with self.assertRaises(ValueError):
                check_outputs(PROFILE.name, "test-quantize-recipe", root / "packs", root / "n.json")

    def test_prepare_packs_requires_full_coverage_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            groups = input_groups(PROFILE)
            weights = {k: None for k in PROFILE.shapes() if k != "lm_head"}
            with self.assertRaises(ValueError):
                prepare_packs(weights, {}, groups, root / "packs", root / "m.json", profile=PROFILE.name)
            self.assertFalse((root / "packs").exists())


class Command(unittest.TestCase):
    def test_quantize_command_options(self):
        from fissile import cli

        out = io.StringIO()
        with mock.patch("sys.argv", ["fissile", "quantize", "--help"]), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as exit:
                cli.main()
        self.assertEqual(exit.exception.code, 0)
        for option in ("--model", "--calibration", "--packs-dir", "--output", "--max-rows-per-forward"):
            self.assertIn(option, out.getvalue())


if __name__ == "__main__":
    unittest.main()
