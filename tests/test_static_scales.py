"""Static activation scales: registration, scale files and derived configs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from fissile.backends.registry import get_backend
from fissile.backends.sglang.engine import scoring_engine
from fissile.config import Config
from fissile.providers import get_provider
from fissile.providers.static_scales import SCALES_FORMAT, derived_config, load_activation_scales

PROFILE, RECIPE = 'qwen3.5-dense-9b', 'mixed-fp4-fp8-w4a16-v1'


class StaticScaleProviders(unittest.TestCase):
    def test_differ_from_v2_only_by_factory_and_scale_resource(self):
        v2 = get_provider('gb10-mixed-v2')
        for name, extra in (('gb10-mixed-v2-calibration', []), ('gb10-mixed-v2-static', [('activation_scales', 'sha256')])):
            spec = get_provider(name)
            for field in ('profiles', 'recipes', 'capability', 'routes', 'packed_routes', 'fusions',
                          'alignment', 'paired_mlp_max_rows'):
                self.assertEqual(getattr(spec, field), getattr(v2, field), field)
            self.assertEqual(spec.resources[:len(v2.resources)], v2.resources)
            self.assertEqual([(r.name, r.integrity) for r in spec.resources[len(v2.resources):]], extra)
            for backend in ('sglang', 'sglang-candidates'):
                get_backend(backend).validate(type('C', (), dict(profile=PROFILE, recipe=RECIPE, kernel_provider=name)))

    def test_scale_file_contract(self):
        good = {'format': SCALES_FORMAT, 'profile': PROFILE, 'recipe': RECIPE, 'scales': {'layers.0.mlp.gate_proj': 0.01}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 's.json'
            path.write_text(json.dumps(good))
            self.assertEqual(load_activation_scales(path, profile=PROFILE, recipe=RECIPE), {'layers.0.mlp.gate_proj': 0.01})
            for bad in ({'format': 'other'}, {'profile': 'x'}, {'scales': {}}, {'scales': {'a': 0.0}},
                        {'scales': {'a': float('inf')}}, {'scales': {'a': True}}, {'scales': {'a': -1.0}}):
                path.write_text(json.dumps(good | bad))
                with self.assertRaises(ValueError):
                    load_activation_scales(path, profile=PROFILE, recipe=RECIPE)


class DerivedConfigs(unittest.TestCase):
    def write_base(self, d, provider='gb10-mixed-v2'):
        base = {'format': 'fissile-config-v2', 'profile': PROFILE, 'recipe': RECIPE, 'kernel_provider': provider,
                'manifest': 'm.json', 'artifacts_dir': 'packs', 'report_dir': 'reports',
                'resources': {'tensorfold_library': {'path': 'tf.so'},
                              'silu_lut': {'path': 'silu.safetensors', 'sha256': '0' * 64},
                              'fp4_gemm_library': {'path': 'gemm.so'}}}
        (d / 'c.json').write_text(json.dumps(base))
        return d / 'c.json'

    def test_static_and_calibration_variants_of_a_v2_config(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            config = self.write_base(d)
            (d / 'scales.json').write_text('{}')
            (d / 'static').mkdir()
            (d / 'calibration').mkdir()
            static = Config.load(derived_config(config, d / 'static', scales=d / 'scales.json'), verify_files=False)
            self.assertEqual(static.kernel_provider, 'gb10-mixed-v2-static')
            self.assertEqual(static.resources['activation_scales'].sha256, hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(static.manifest, (d / 'm.json').resolve())
            self.assertEqual(static.resources['silu_lut'].sha256, '0' * 64)
            self.assertIsNone(static.resources['fp4_gemm_library'].sha256)
            calibration = Config.load(derived_config(config, d / 'calibration'), verify_files=False)
            self.assertEqual(calibration.kernel_provider, 'gb10-mixed-v2-calibration')
            self.assertEqual(set(calibration.resources), {'tensorfold_library', 'silu_lut', 'fp4_gemm_library'})

    def test_only_gb10_mixed_v2_configs_and_no_prefill_graph(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'scales.json').write_text('{}')
            config = self.write_base(d, provider='gb10-mixed-v1')
            with self.assertRaises(ValueError):
                derived_config(config, d, scales=d / 'scales.json')
            with self.assertRaises(ValueError):
                with scoring_engine(config, d, prefill_cuda_graph=True, activation_scales=d / 'scales.json'):
                    pass


if __name__ == '__main__':
    unittest.main()
