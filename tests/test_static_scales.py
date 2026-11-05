"""Static activation scales: registration and scale-file contract."""
import json
from pathlib import Path
import tempfile
import unittest
from fissile.backends.registry import get_backend
from fissile.providers import get_provider
from fissile.providers.static_scales import SCALES_FORMAT, load_activation_scales

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


if __name__ == '__main__':
    unittest.main()
