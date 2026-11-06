import json, subprocess, sys, tempfile, unittest, ast, os
from pathlib import Path
from fissile.config import contained, validate_manifest, Config, sha256
from fissile.models.qwen35 import projection_shapes, route


def fixture(root):
    projections = {}
    for i, (key, shape) in enumerate(projection_shapes().items()):
        name = f"weights/{i}.pack"
        p = root / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(str(i).encode())
        projections[key] = {
            "path": name,
            "shape": shape,
            "sha256": sha256(p),
            "effective_weight_sha256": "a" * 64,
        }
    return dict(
        format="fissile-packs-v1",
        profile="qwen3.5-dense-9b",
        complete=True,
        method="full-h16-order",
        labels_consumed=False,
        projections=projections,
    )


class Portability(unittest.TestCase):
    def test_cpu_import_is_lazy(self):
        import fissile

        p = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys,fissile; from fissile.backends.registry import BACKENDS; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert 'vllm' not in sys.modules; print(len(BACKENDS))",
            ],
            capture_output=True,
            text=True,
            check=True,
            env=dict(
                os.environ,
                PYTHONPATH=str(Path(fissile.__file__).resolve().parent.parent),
            ),
        )
        self.assertEqual(p.stdout.strip(), "4")

    def test_missing_wrong_shape_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = fixture(root)
            validate_manifest(m, root)
            key = next(iter(m["projections"]))
            e = m["projections"][key]
            e["shape"] = [128, 128]
            with self.assertRaises(ValueError):
                validate_manifest(m, root)
            e["shape"] = projection_shapes()[key]
            (root / e["path"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate_manifest(m, root)
            del m["projections"][key]
            with self.assertRaises(ValueError):
                validate_manifest(m, root, verify_files=False)

    def test_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "root"
            root.mkdir()
            for p in ["../outside", "/tmp/outside"]:
                with self.assertRaises(ValueError):
                    contained(root, p)
            (root / "escape").symlink_to(root.parent)
            with self.assertRaises(ValueError):
                contained(root, "escape/a")

    def test_duplicate_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = fixture(root)
            a, b = list(m["projections"])[:2]
            m["projections"][b]["path"] = m["projections"][a]["path"]
            with self.assertRaises(ValueError):
                validate_manifest(m, root, verify_files=False)

    def test_config_relative_paths_and_binary_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = fixture(root)
            (root / "manifest.json").write_text(json.dumps(m))
            (root / "table").write_bytes(b"lut")
            (root / "lib.so").write_bytes(b"library")
            (root / "lib.build.json").write_text(
                json.dumps(dict(library_sha256=sha256(root / "lib.so")))
            )
            data = dict(
                format="fissile-config-v1",
                profile="qwen3.5-dense-9b",
                manifest="manifest.json",
                artifacts_dir=".",
                tensorfold_library="lib.so",
                silu_lut="table",
                silu_lut_sha256=sha256(root / "table"),
                report_dir="reports",
            )
            (root / "config.json").write_text(json.dumps(data))
            old = Config.load(root / "config.json")
            self.assertEqual(old.artifacts_dir, root)
            modern = {
                k: data[k]
                for k in ("profile", "manifest", "artifacts_dir", "report_dir")
            }
            modern.update(
                format="fissile-config-v2",
                recipe=old.recipe,
                kernel_provider=old.kernel_provider,
                resources={
                    "tensorfold_library": {"path": "lib.so"},
                    "silu_lut": {"path": "table", "sha256": sha256(root / "table")},
                },
            )
            (root / "v2.json").write_text(json.dumps(modern))
            self.assertEqual(Config.load(root / "v2.json"), old)
            (root / "lib.so").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                Config.load(root / "config.json")

    def test_routes(self):
        from collections import Counter

        self.assertEqual(
            Counter(route(k) for k in projection_shapes()),
            {"fp4": 184, "fp8": 16, "w4a16": 1},
        )

    def test_kernel_dependency_direction(self):
        import fissile

        root = Path(fissile.__file__).parent
        for p in (root / "kernels").glob("*.py"):
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.ImportFrom):
                    self.assertFalse(
                        (n.module or "").startswith(
                            (
                                "fissile.backends",
                                "fissile.quantization",
                                "sglang",
                                "transformers",
                            )
                        ),
                        str(p),
                    )


if __name__ == "__main__":
    unittest.main()
