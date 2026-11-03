import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_sglang.py'
spec = importlib.util.spec_from_file_location('fissile_benchmark_sglang', path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkStatistics(unittest.TestCase):
    def test_expected_settings_match_fissile_engine(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from fissile.backends.sglang.engine import scoring_engine
        kwargs = {}
        def create(**settings):
            kwargs.update(settings)
            return SimpleNamespace(shutdown=lambda: None)
        config = SimpleNamespace(plan=SimpleNamespace(profile=SimpleNamespace(max_context=16384)))
        with patch('fissile.backends.sglang.engine.Config.load', return_value=config), patch(
                'fissile.backends.registry.get_backend', return_value=SimpleNamespace(validate=lambda _: None)), patch.dict(
                'sys.modules', {'sglang': SimpleNamespace(Engine=create)}), patch.dict('os.environ', {}, clear=True):
            with scoring_engine('/test/config.json', '/test/model'):
                pass
        expected = benchmark.baseline_settings('/test/model')
        self.assertEqual({k: kwargs[k] for k in expected}, expected)

    def test_readout_ties_follow_scope_contract(self):
        self.assertEqual(benchmark.decode_choice(['in_scope','out_of_scope'], [-1.,-1.]), 'in_scope')
        self.assertEqual(benchmark.decode_choice(['out_of_scope','in_scope'], [-1.,-1.]), 'in_scope')
        self.assertEqual(benchmark.decode_choice(['in_scope','out_of_scope'], [-2.,-.1]), 'out_of_scope')
        self.assertEqual(benchmark.decode_choice(['in_scope','out_of_scope'], [-1.,-.999999]), 'in_scope')

    def test_interpolated_percentile(self):
        self.assertEqual(benchmark.percentile([4, 1, 3, 2], .5), 2.5)
        self.assertEqual(benchmark.percentile([7], .95), 7)

    def test_quality_uses_unique_inputs_and_class_balance(self):
        rows = [dict(id='a', expected='in'), dict(id='b', expected='out')]
        records = []
        for round_id in range(2):
            for arm, milliseconds in [('bf16', 20), ('mixed', 10)]:
                for row in rows:
                    records.append(dict(row, arm=arm, round=round_id,
                        elapsed_ms=milliseconds, prompt_tokens=100,
                        predicted=row['expected'] if arm=='bf16' else 'in'))
        result = benchmark.summarize(records, rows, 2)
        self.assertEqual(result['bf16']['requests'], 4)
        self.assertEqual(result['bf16']['unique_requests'], 2)
        self.assertEqual(result['bf16']['requests_per_second'], 50)
        self.assertEqual(result['mixed']['input_tokens_per_second'], 10000)
        self.assertEqual(result['mixed']['per_round'][0]['balanced_accuracy'], .5)
        self.assertEqual(result['mixed']['inconsistent_labels_across_rounds'], 0)
        records[-1]['predicted'] = 'out'
        result = benchmark.summarize(records, rows, 2)
        self.assertEqual(result['mixed']['inconsistent_labels_across_rounds'], 1)

    def test_missing_measurements_rejected(self):
        with self.assertRaises(AssertionError):
            benchmark.summarize([], [dict(id='a', expected='in')], 1)

    def test_exact_pairs_and_rejected_drift(self):
        records = [dict(round=0, id='a', arm=arm, logprobs=[-1., -2.])
                   for arm in ('bf16', 'mixed')]
        self.assertTrue(benchmark.validate_pairs(records, 1, require_exact=True))
        records[-1]['logprobs'] = [-1., -2.000001]
        self.assertFalse(benchmark.validate_pairs(records, 1))
        with self.assertRaises(ValueError):
            benchmark.validate_pairs(records, 1, require_exact=True)

    def test_missing_or_duplicate_pairs_rejected(self):
        r = dict(round=0, id='a', arm='bf16', logprobs=[-1., -2.])
        for records in ([], [r], [r, r]):
            with self.assertRaises(ValueError):
                benchmark.validate_pairs(records, 1)
