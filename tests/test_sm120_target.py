"""Explicit device dispatch must not change the existing GB10 default."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import fissile
from fissile.build import build_tensorfold
from fissile.providers import DEFAULT_PROVIDER, get_provider
from fissile.backends.registry import get_backend
from fissile.models import get_profile
from fissile.recipes import ExecutionPlan, get_recipe


class SM120Target(unittest.TestCase):
    def test_provider_requires_explicit_target(self):
        self.assertEqual(DEFAULT_PROVIDER, 'gb10-mixed-v1')
        gb10 = get_provider(DEFAULT_PROVIDER)
        sm120 = get_provider('sm120-mixed-v1')
        plan = ExecutionPlan(get_profile('qwen3.5-dense-9b'),
                             get_recipe('mixed-fp4-fp8-w4a16-v1'))
        gb10.validate(plan, capability=(12, 1))
        sm120.validate(plan, capability=(12, 0))
        for provider, wrong in ((gb10, (12, 0)), (sm120, (12, 1))):
            with self.assertRaises(ValueError):
                provider.validate(plan, capability=wrong)
        self.assertEqual(sm120.routes, gb10.routes)
        self.assertEqual(sm120.resources, gb10.resources)

    def test_only_explicit_sglang_combination(self):
        cfg = SimpleNamespace(profile='qwen3.5-dense-9b',
                              recipe='mixed-fp4-fp8-w4a16-v1',
                              kernel_provider='sm120-mixed-v1')
        get_backend('sglang').validate(cfg)
        with self.assertRaises(ValueError):
            get_backend('torch').validate(cfg)

    def test_cute_target_map_without_importing_cuda(self):
        source = Path(fissile.__file__).parent / 'kernels/paired_glu.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_target_arch')
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        target = scope['_target_arch']
        self.assertEqual(target((12, 0)), 'sm_120a')
        self.assertEqual(target((12, 1)), 'sm_121a')
        with self.assertRaises(ValueError):
            target((9, 0))

    def test_build_pins_matching_host_and_device_targets(self):
        for arch in ('sm_120', 'sm_121'):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as d:
                output = Path(d) / 'kernel.so'
                def compile_stub(cmd, **kwargs):
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    self.assertIn('-DFISSILE_CUDA_MINOR=' + arch[-1], cmd)
                    self.assertIn('-gencode=arch=compute_' + arch[3:] + ',code=' + arch, cmd)
                    output.write_bytes(b'test-library')
                with patch('fissile.build.subprocess.run', side_effect=compile_stub), patch(
                        'fissile.build.subprocess.check_output', return_value='test compiler'):
                    receipt = build_tensorfold(output, '/test/nvcc', cuda_arch=arch)
                self.assertEqual(receipt['cuda_arch'], arch)
                self.assertFalse(receipt['gpu_kernels_launched'])
                self.assertTrue(output.with_suffix('.build.json').exists())

    def test_unknown_build_target_rejected_before_compilation(self):
        with self.assertRaises(ValueError):
            build_tensorfold('/unused.so', cuda_arch='sm_90')
