import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).parents[1]/'scripts/analyze_prefix_reuse.py'
spec=importlib.util.spec_from_file_location('fissile_prefix_reuse',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PrefixOpportunity(unittest.TestCase):
    def test_reuse_counts_only_after_first_computation(self):
        r=module.analyze([[1,2,3],[1,2,4],[1,7]],lengths=(1,2,3))
        self.assertEqual(r['input_tokens'],8);self.assertEqual(r['common_prefix_tokens'],1)
        self.assertEqual([x['hypothetical_avoided_tokens'] for x in r['prefixes']],[2,2,0])
        self.assertEqual(r['prefixes'][2]['eligible_requests'],2)

    def test_shortest_input_bounds_common_prefix(self):
        r=module.analyze([[1],[1,2]],lengths=(4,))
        self.assertEqual(r['common_prefix_tokens'],1)
        self.assertEqual(r['prefixes'][0]['eligible_requests'],0)
        self.assertEqual(r['prefixes'][0]['largest_group'],0)

    def test_bad_inputs_rejected(self):
        for seq in ([],[[]],[[True]],[[1.0]],[[-1]]):
            with self.assertRaises(ValueError):module.analyze(seq)
        for lengths in ([],[0],[-1],[True]):
            with self.assertRaises(ValueError):module.analyze([[1]],lengths)
