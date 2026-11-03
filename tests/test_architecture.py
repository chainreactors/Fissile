"""Keep the published package independent of engine installation and workspace paths."""

import ast
from pathlib import Path
import unittest
import fissile


class Architecture(unittest.TestCase):
    def test_core_has_only_standard_library_and_core_imports(self):
        root = Path(fissile.__file__).parent
        forbidden = (
            "torch",
            "triton",
            "sglang",
            "transformers",
            "cutlass",
            "cuda",
            "fissile.runtime",
            "fissile.backends",
        )
        for relative in [
            "__init__.py",
            "config.py",
            "artifacts.py",
            "models/qwen35.py",
            "models/spec.py",
            "recipes/__init__.py",
            "providers/__init__.py",
        ]:
            for node in ast.walk(ast.parse((root / relative).read_text())):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                # require_gpu is the sole explicit opt-in GPU boundary in config.
                if relative == "config.py" and names == ["torch"]:
                    continue
                for name in names:
                    self.assertFalse(name.startswith(forbidden), (relative, name))

    def test_runtime_and_quantization_do_not_import_engines(self):
        root = Path(fissile.__file__).parent
        for directory in ["runtime", "quantization", "providers"]:
            for file in (root / directory).glob("*.py"):
                for node in ast.walk(ast.parse(file.read_text())):
                    names = []
                    if isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    for name in names:
                        self.assertFalse(
                            name.startswith(
                                ("sglang", "transformers", "fissile.backends")
                            ),
                            (file.name, name),
                        )
                        if directory == "runtime":
                            self.assertFalse(
                                name.startswith(
                                    ("fissile.models.qwen35", "fissile.providers.gb10")
                                ),
                                (file.name, name),
                            )


if __name__ == "__main__":
    unittest.main()
