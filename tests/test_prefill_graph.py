import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fissile.backends.sglang.engine import PREFILL_GRAPH_BUCKETS, prefill_graph_settings
from fissile.runtime.mixed import MixedRuntime


class PrefillGraphSettings(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertEqual(prefill_graph_settings(False), {"cuda_graph_backend_prefill": "disabled"})

    def test_buckets(self):
        settings = prefill_graph_settings(True)
        self.assertEqual(settings["cuda_graph_backend_prefill"], "breakable")
        buckets = settings["cuda_graph_bs_prefill"]
        self.assertEqual(buckets, sorted(set(buckets)))
        self.assertIn(1536, buckets)
        self.assertEqual(buckets[-1], 4096)
        # SGLang refuses to pad more than 2x; adjacent buckets must stay within it.
        self.assertLessEqual(max(b / a for a, b in zip(buckets, buckets[1:])), 2)
        self.assertEqual(tuple(buckets), PREFILL_GRAPH_BUCKETS)

    def test_paired_mlp_limit_must_be_a_bucket_boundary(self):
        for limit in (0, 1536, 8192):
            self.assertEqual(prefill_graph_settings(True, limit)["cuda_graph_backend_prefill"], "breakable")
        with self.assertRaises(ValueError):
            prefill_graph_settings(True, 1500)
        self.assertEqual(prefill_graph_settings(False, 1500), {"cuda_graph_backend_prefill": "disabled"})

    def _engine_kwargs(self, **options):
        from fissile.backends.sglang.engine import scoring_engine
        kwargs = {}
        def create(**settings):
            kwargs.update(settings)
            return SimpleNamespace(shutdown=lambda: None)
        config = SimpleNamespace(plan=SimpleNamespace(profile=SimpleNamespace(max_context=16384)),
                                 provider=SimpleNamespace(paired_mlp_max_rows=1536))
        with patch('fissile.backends.sglang.engine.Config.load', return_value=config), patch(
                'fissile.backends.registry.get_backend', return_value=SimpleNamespace(validate=lambda _: None)), patch.dict(
                'sys.modules', {'sglang': SimpleNamespace(Engine=create)}), patch.dict('os.environ', {}, clear=True):
            with scoring_engine('/test/config.json', '/test/model', **options):
                pass
        return kwargs

    def test_engine_default_keeps_eager_prefill(self):
        kwargs = self._engine_kwargs()
        self.assertEqual(kwargs['cuda_graph_backend_prefill'], 'disabled')
        self.assertNotIn('cuda_graph_bs_prefill', kwargs)
        self.assertEqual(kwargs['max_running_requests'], 1)

    def test_engine_flag_enables_breakable_graph(self):
        kwargs = self._engine_kwargs(prefill_cuda_graph=True)
        self.assertEqual(kwargs['cuda_graph_backend_prefill'], 'breakable')
        self.assertEqual(kwargs['cuda_graph_bs_prefill'], list(PREFILL_GRAPH_BUCKETS))
        self.assertEqual(kwargs['cuda_graph_backend_decode'], 'disabled')
        self.assertEqual(kwargs['max_running_requests'], 1)


class ValidatePass(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace()
        keys = ["lm_head"] + ["layers.%d.p" % i for i in range(8)]
        self.runtime.expected_routes = {k: "fp4" for k in keys}
        self.runtime.calls = {k: 3 for k in keys}
        self.runtime.profile = SimpleNamespace(head="lm_head")

    def run_pass(self, step):
        before = dict(self.runtime.calls)
        step(self.runtime.calls)
        return MixedRuntime.validate_pass(self.runtime, before)

    def test_eager_pass(self):
        def eager(calls):
            for k in calls:
                calls[k] += 1
        self.assertEqual(self.run_pass(eager), "eager")

    def test_graph_replay_pass(self):
        def replay(calls):
            calls["lm_head"] += 1
        self.assertEqual(self.run_pass(replay), "graph_replay")

    def test_bypassed_projection_rejected(self):
        def bypass(calls):
            for k in calls:
                if k != "layers.5.p":
                    calls[k] += 1
        with self.assertRaises(RuntimeError):
            self.run_pass(bypass)

    def test_output_head_must_run_once(self):
        for extra in (0, 2):
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                self.run_pass(lambda calls: calls.__setitem__("lm_head", calls["lm_head"] + extra))

    def test_replay_after_incomplete_capture_rejected(self):
        self.runtime.calls["layers.2.p"] += 1  # a captured pass that missed other projections
        with self.assertRaises(RuntimeError):
            self.run_pass(lambda calls: calls.__setitem__("lm_head", calls["lm_head"] + 1))


if __name__ == "__main__":
    unittest.main()
