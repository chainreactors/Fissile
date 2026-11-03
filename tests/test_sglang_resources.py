import unittest
from types import SimpleNamespace
from unittest.mock import patch
from fissile.backends.sglang.engine import resource_settings


class ResourceSettings(unittest.TestCase):
    def test_default_is_unchanged(self):
        self.assertEqual(resource_settings({}), {"mem_fraction_static": 0.35})

    def test_shared_host_resources(self):
        self.assertEqual(resource_settings({
            "FISSILE_SGLANG_MEM_FRACTION_STATIC": "0.65",
            "FISSILE_SGLANG_NCCL_PORT": "29516",
        }), {"mem_fraction_static": 0.65, "nccl_port": 29516})

    def test_bad_fraction_rejected(self):
        for value in ("0", "1", "-0.1", "NaN", "inf", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_settings({"FISSILE_SGLANG_MEM_FRACTION_STATIC": value})

    def test_bad_port_rejected(self):
        for value in ("1", "65536", "0", "bad", "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_settings({"FISSILE_SGLANG_NCCL_PORT": value})

    def test_other_environment_cannot_override_arithmetic(self):
        self.assertEqual(resource_settings({"dtype": "float16", "max_running_requests": "32"}),
                         {"mem_fraction_static": 0.35})

    def test_full_prompt_capacity_reserves_engine_slot(self):
        from fissile.backends.sglang.engine import scoring_engine
        kwargs = {}
        session = SimpleNamespace(shutdown=lambda: None)
        def create(**settings):
            kwargs.update(settings)
            return session
        config = SimpleNamespace(plan=SimpleNamespace(profile=SimpleNamespace(max_context=16384)))
        with patch('fissile.backends.sglang.engine.Config.load', return_value=config), patch(
                'fissile.backends.registry.get_backend', return_value=SimpleNamespace(validate=lambda _:None)), patch.dict(
                'sys.modules', {'sglang':SimpleNamespace(Engine=create)}), patch.dict('os.environ', {}, clear=True):
            with scoring_engine('/test/config.json', '/test/model') as engine:
                self.assertIs(engine, session)
        self.assertEqual(kwargs['context_length'], 16391)
        self.assertGreater(kwargs['context_length'] - 6, 16384)
        self.assertEqual(kwargs['max_prefill_tokens'], 16384)
        self.assertEqual(kwargs['max_total_tokens'], 32768)
        self.assertEqual(kwargs['max_running_requests'], 1)


class CandidateReadoutPlumbing(unittest.TestCase):
    def _run(self, readout, environ):
        from fissile.backends.sglang.engine import scoring_engine
        seen = {}
        def create(**settings):
            seen['readout'] = __import__('os').environ.get('FISSILE_READOUT')
            return SimpleNamespace(shutdown=lambda: None)
        config = SimpleNamespace(plan=SimpleNamespace(profile=SimpleNamespace(max_context=16384)))
        with patch('fissile.backends.sglang.engine.Config.load', return_value=config), patch(
                'fissile.backends.registry.get_backend', return_value=SimpleNamespace(validate=lambda _: None)), patch.dict(
                'sys.modules', {'sglang': SimpleNamespace(Engine=create)}), patch.dict('os.environ', environ, clear=True):
            import os
            with scoring_engine('/test/config.json', '/test/model', readout=readout):
                pass
            after = os.environ.get('FISSILE_READOUT')
        return seen['readout'], after

    def test_readout_is_set_for_engine_and_restored(self):
        self.assertEqual(self._run('full', {}), ('full', None))
        self.assertEqual(self._run('candidates', {'FISSILE_READOUT': 'full'}), ('candidates', 'full'))

    def test_unknown_readout_rejected_before_engine(self):
        from fissile.backends.sglang.engine import scoring_engine
        with self.assertRaises(ValueError):
            with scoring_engine('/test/config.json', '/test/model', readout='top-k'):
                pass
