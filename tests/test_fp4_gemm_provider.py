"""gb10-mixed-v2: same recipe arithmetic as v1, pinned CUTLASS W4A4 GEMM resource."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import fissile
from fissile.build import build_fp4_gemm
from fissile.providers import DEFAULT_PROVIDER, get_provider
from fissile.backends.registry import get_backend
from fissile.models import get_profile
from fissile.recipes import ExecutionPlan, get_recipe

PACKAGE = Path(fissile.__file__).parent


class PinnedFP4Gemm(unittest.TestCase):
    def test_v2_differs_from_v1_only_by_gemm_resource(self):
        v1, v2 = get_provider(DEFAULT_PROVIDER), get_provider('gb10-mixed-v2')
        self.assertEqual(DEFAULT_PROVIDER, 'gb10-mixed-v1')
        for field in ('profiles', 'recipes', 'capability', 'activation_dtype', 'routes',
                      'packed_routes', 'fusions', 'alignment'):
            self.assertEqual(getattr(v1, field), getattr(v2, field), field)
        self.assertEqual(v2.resources[:len(v1.resources)], v1.resources)
        extra = v2.resources[len(v1.resources):]
        self.assertEqual([(r.name, r.integrity) for r in extra], [('fp4_gemm_library', 'build_receipt')])
        plan = ExecutionPlan(get_profile('qwen3.5-dense-9b'), get_recipe('mixed-fp4-fp8-w4a16-v1'))
        v2.validate(plan, capability=(12, 1))
        with self.assertRaises(ValueError):
            v2.validate(plan, capability=(12, 0))

    def test_only_qualified_sglang_combination(self):
        cfg = SimpleNamespace(profile='qwen3.5-dense-9b', recipe='mixed-fp4-fp8-w4a16-v1',
                              kernel_provider='gb10-mixed-v2')
        get_backend('sglang').validate(cfg)
        with self.assertRaises(ValueError):
            get_backend('torch').validate(cfg)

    def test_every_fp4_gemm_uses_the_overridable_hook(self):
        tree = ast.parse((PACKAGE / 'providers/gb10.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GB10Provider')
        direct = []
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name == '_fp4_matmul':
                continue
            direct += [method.name for n in ast.walk(method)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'matmul']
        self.assertEqual(direct, [])
        sub = ast.parse((PACKAGE / 'providers/gb10_cutlass.py').read_text())
        cls = next(n for n in sub.body if isinstance(n, ast.ClassDef))
        self.assertEqual(sorted(n.name for n in cls.body if isinstance(n, ast.FunctionDef)),
                         ['__init__', '_fp4_matmul'])

    def test_build_is_cpu_only_and_records_cutlass(self):
        with tempfile.TemporaryDirectory() as d:
            cutlass = Path(d) / 'cutlass'
            (cutlass / 'include/cutlass').mkdir(parents=True)
            (cutlass / 'tools/util/include').mkdir(parents=True)
            (cutlass / 'include/cutlass/version.h').write_text(
                '#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 5\n#define CUTLASS_PATCH 0\n')
            output = Path(d) / 'libfp4.so'
            def compile_stub(cmd, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertIn('-gencode=arch=compute_121a,code=sm_121a', cmd)
                self.assertIn('-I' + str(cutlass.resolve() / 'include'), cmd)
                output.write_bytes(b'test-library')
            with patch('fissile.build.subprocess.run', side_effect=compile_stub), patch(
                    'fissile.build.subprocess.check_output', return_value='test compiler'):
                receipt = build_fp4_gemm(output, cutlass, '/test/nvcc')
            self.assertEqual(receipt['cutlass_version'], '4.5.0')
            self.assertFalse(receipt['gpu_kernels_launched'])
            self.assertIn('fp4_gemm.cu', receipt['sources'])
            self.assertTrue(output.with_suffix('.build.json').exists())
            with self.assertRaises(FileExistsError):
                build_fp4_gemm(output, cutlass, '/test/nvcc')

    def test_build_rejects_incomplete_cutlass_tree(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_fp4_gemm(Path(d) / 'out.so', d, '/test/nvcc')

    def test_source_is_packaged_and_pins_one_tactic(self):
        source = (PACKAGE / 'kernels/fp4_gemm.cu').read_text()
        self.assertIn('Shape<_128, _128, _256>', source)
        self.assertIn('StageCount<2>', source)
        self.assertIn('KernelTmaWarpSpecializedCooperative', source)
        self.assertNotIn('StreamK', source)

    def test_candidate_readout_backend_is_explicit(self):
        spec = get_backend('sglang-candidates')
        self.assertEqual(spec.workload, 'single-request candidate-normalized scoring')
        for provider in ('gb10-mixed-v1', 'gb10-mixed-v2'):
            get_backend('sglang-candidates').validate(SimpleNamespace(
                profile='qwen3.5-dense-9b', recipe='mixed-fp4-fp8-w4a16-v1', kernel_provider=provider))
        with self.assertRaises(ValueError):
            spec.validate(SimpleNamespace(profile='qwen3.5-dense-9b', recipe='mixed-fp4-fp8-w4a16-v1',
                                          kernel_provider='sm120-mixed-v1'))


if __name__ == '__main__':
    unittest.main()
