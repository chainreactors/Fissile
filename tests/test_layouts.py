"""CPU-only checks for shared engine layouts, profile heads and legacy import."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fissile.artifacts import canonical_key, import_legacy
from fissile.backends.layouts import QWEN35_DENSE, get_layout, install_fused, register_layout
from fissile.models import MLP, ModelProfile, Projection, get_profile, register_profile
from fissile.providers import DEFAULT_PROVIDER, get_provider
from fissile.recipes import Recipe, register_recipe


class Weight:
    def __init__(self, name):
        self.name = name

    def split(self, widths, dim):
        assert dim == 0
        count = len(widths) if isinstance(widths, list) else 2
        return tuple(f"{self.name}[{i}]" for i in range(count))


def linear(name):
    return SimpleNamespace(weight=Weight(name))


class Runtime:
    """Records bindings; same interface as MixedRuntime for install_fused."""

    def __init__(self, fused=True):
        keys = {
            "lm_head": [16, 8],
            "layers.0.linear_attn.in_proj_qkv": [32, 8],
            "layers.0.linear_attn.in_proj_z": [16, 8],
            "layers.0.linear_attn.out_proj": [8, 8],
            "layers.1.self_attn.q_proj": [32, 8],
            "layers.1.self_attn.k_proj": [8, 8],
            "layers.1.self_attn.v_proj": [8, 8],
            "layers.1.self_attn.o_proj": [8, 8],
        }
        for i in (0, 1):
            keys.update({f"layers.{i}.mlp.{n}_proj": s for n, s in
                         (("gate", [24, 8]), ("up", [24, 8]), ("down", [8, 24]))})
        self.profile = SimpleNamespace(name=QWEN35_DENSE.profile, shapes=lambda: keys, head="lm_head")
        self.metadata = {k: {"shape": v} for k, v in keys.items()}
        mlps = tuple(MLP(f"layers.{i}.mlp", *(f"layers.{i}.mlp.{n}_proj" for n in ("gate", "up", "down")))
                     for i in (0, 1))
        self.plan = SimpleNamespace(fused_mlps=mlps if fused else ())
        self.bound = []

    def mlp(self, prefix, gate, up, down, *, long_silu):
        self.bound.append(("mlp", prefix, gate, up, down.name, long_silu))
        return lambda x: ("mlp", prefix, x)

    def projection(self, key, weight):
        self.bound.append(("projection", key, weight.name))
        return lambda x: ("projection", key, x)

    def joined_projection_group(self, keys, weights, concatenate):
        self.bound.append(("group", tuple(keys), tuple(weights)))
        return lambda x: ("group", tuple(keys), x)


def engine_model(engine):
    mlp = lambda i: SimpleNamespace(gate_up_proj=linear(f"gu{i}"), down_proj=linear(f"d{i}"))
    layer0 = SimpleNamespace(mlp=mlp(0), linear_attn=SimpleNamespace(
        in_proj_qkvz=linear("qkvz0"), out_proj=linear("out0")))
    attention = SimpleNamespace(qkv_proj=linear("qkv1"), o_proj=linear("o1"))
    layer1 = (SimpleNamespace(mlp=mlp(1), **vars(attention)) if engine == "sglang"
              else SimpleNamespace(mlp=mlp(1), self_attn=attention))
    return SimpleNamespace(model=SimpleNamespace(layers=[layer0, layer1]))


class Layouts(unittest.TestCase):
    def test_fused_engines_share_one_layout(self):
        for engine in ("sglang", "vllm"):
            with self.subTest(engine=engine):
                runtime, model = Runtime(), engine_model(engine)
                layers = install_fused(runtime, model, engine=engine)
                self.assertIs(layers, model.model.layers)
                self.assertEqual(runtime.bound, [
                    ("mlp", "layers.0.mlp", "gu0[0]", "gu0[1]", "d0", "torch"),
                    ("group", ("layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z"),
                     ("qkvz0[0]", "qkvz0[1]")),
                    ("projection", "layers.0.linear_attn.out_proj", "out0"),
                    ("mlp", "layers.1.mlp", "gu1[0]", "gu1[1]", "d1", "torch"),
                    ("group", ("layers.1.self_attn.q_proj", "layers.1.self_attn.k_proj",
                               "layers.1.self_attn.v_proj"), ("qkv1[0]", "qkv1[1]", "qkv1[2]")),
                    ("projection", "layers.1.self_attn.o_proj", "o1"),
                ])
                self.assertEqual(layers[0].mlp.forward("x", "batch", flag=True), ("mlp", "layers.0.mlp", "x"))
                qkv = layers[1].qkv_proj if engine == "sglang" else layers[1].self_attn.qkv_proj
                self.assertEqual(qkv.quant_method.apply(None, "x")[0], "group")
                with self.assertRaises(ValueError):
                    qkv.quant_method.apply(None, "x", bias=object())

    def test_unfused_mlp_binds_its_linears(self):
        runtime, model = Runtime(fused=False), engine_model("vllm")
        install_fused(runtime, model, engine="vllm")
        self.assertIn(("group", ("layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"), ("gu0[0]", "gu0[1]")),
                      runtime.bound)
        self.assertIn(("projection", "layers.0.mlp.down_proj", "d0"), runtime.bound)
        self.assertFalse(any(b[0] == "mlp" for b in runtime.bound))
        self.assertFalse(hasattr(model.model.layers[0].mlp, "forward"))

    def test_partial_or_missing_layers_rejected(self):
        runtime = Runtime()
        del runtime.metadata["layers.0.linear_attn.in_proj_z"]
        shapes = dict(runtime.profile.shapes()); del shapes["layers.0.linear_attn.in_proj_z"]
        runtime.profile.shapes = lambda: shapes
        with self.assertRaises(ValueError):
            install_fused(runtime, engine_model("vllm"), engine="vllm")
        model = engine_model("vllm")
        model.model.layers.append(model.model.layers[0])
        with self.assertRaises(ValueError):
            install_fused(Runtime(), model, engine="vllm")

    def test_registry(self):
        self.assertIs(get_layout(QWEN35_DENSE.profile), QWEN35_DENSE)
        with self.assertRaises(ValueError):
            get_layout("unknown-profile")
        with self.assertRaises(ValueError):
            register_layout(QWEN35_DENSE)


class ProfileHead(unittest.TestCase):
    def test_head_comes_from_role(self):
        self.assertEqual(get_profile("qwen3.5-dense-9b").head, "lm_head")
        renamed = ModelProfile("t", (Projection("body", (8, 8)), Projection("readout", (16, 8), "head")), (), 16, 4)
        self.assertEqual(renamed.head, "readout")
        for projections in [(Projection("body", (8, 8)),),
                            (Projection("a", (8, 8), "head"), Projection("b", (8, 8), "head"))]:
            with self.assertRaises(ValueError):
                ModelProfile("t", projections, (), 16, 4).head


class LegacyImport(unittest.TestCase):
    def test_canonical_key_ignores_module_prefixes(self):
        keys = {"layers.1.mlp.up_proj": 0, "layers.11.mlp.up_proj": 0, "lm_head": 0}
        self.assertEqual(canonical_key("base_model.model.lm_head", keys), "lm_head")
        self.assertEqual(canonical_key("decoder.layers.11.mlp.up_proj", keys), "layers.11.mlp.up_proj")
        self.assertEqual(canonical_key("layers.1.mlp.up_proj", keys), "layers.1.mlp.up_proj")
        for name in ("decoder.layers.2.mlp.up_proj", "lm_head_extra"):
            with self.assertRaises(ValueError):
                canonical_key(name, keys)

    def test_other_profile_and_prefix(self):
        profile = ModelProfile("test-legacy-import", (Projection("block.proj", (128, 128)),
                                                     Projection("readout", (128, 128), "head")), (), 128, 4)
        register_profile(profile)
        register_recipe(Recipe("test-legacy-recipe", frozenset({profile.name}),
                               (("body", "fp4"), ("head", "w4a16")), "test-preparation"))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entries = {}
            for key in ("block.proj", "readout"):
                (root / key).write_bytes(b"pack")
                from fissile.config import sha256
                entries["wrapper.decoder." + key] = dict(path=key, shape=[128, 128], sha256=sha256(root / key),
                                                         effective_weight_sha256="e" * 64)
            (root / "summary.json").write_text(json.dumps(dict(
                complete=True, method="test-preparation", adapter_sha256_after="b" * 64, projections=entries)))
            (root / "protocol.json").write_text(json.dumps(dict(
                method="test-preparation", labels_consumed=False, holdout_consumed=False, adapter_sha256="b" * 64)))
            with self.assertRaises(ValueError):
                import_legacy(root, root / "default-recipe.json", profile.name)
            manifest = import_legacy(root, root / "manifest.json", profile.name, "test-legacy-recipe")
            self.assertEqual(manifest["recipe"], "test-legacy-recipe")
            self.assertEqual(manifest["method"], "test-preparation")
            self.assertEqual(manifest["profile"], profile.name)
            self.assertEqual(sorted(manifest["projections"]), ["block.proj", "readout"])


class PairedMlpLimit(unittest.TestCase):
    def test_declared_by_provider(self):
        for name in (DEFAULT_PROVIDER, "gb10-mixed-v2", "sm120-mixed-v1"):
            self.assertEqual(get_provider(name).paired_mlp_max_rows, 1536)
        with self.assertRaises(ValueError):
            replace(get_provider(DEFAULT_PROVIDER), paired_mlp_max_rows=0)
        with self.assertRaises(ValueError):
            replace(get_provider(DEFAULT_PROVIDER), paired_mlp_max_rows=-1)


if __name__ == "__main__":
    unittest.main()
