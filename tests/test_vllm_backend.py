import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fissile.backends.registry import get_backend
from fissile.backends.vllm.engine import MAX_CANDIDATES, engine_settings, resource_settings
from fissile.runtime.mixed import MixedRuntime


class VllmSettings(unittest.TestCase):
    def test_full_prompt_in_one_step_and_one_request(self):
        s = engine_settings('/m', 16384, {})
        self.assertEqual(s['max_model_len'], 16385)  # prompt + transport token
        self.assertEqual(s['max_num_batched_tokens'], 16384)
        self.assertEqual(s['max_num_seqs'], 1)
        self.assertTrue(s['enforce_eager'])
        self.assertFalse(s['enable_prefix_caching'])
        self.assertEqual(s['logprobs_mode'], 'raw_logprobs')
        self.assertEqual(s['mamba_ssm_cache_dtype'], 'float32')
        self.assertEqual(s['dtype'], 'bfloat16')

    def test_memory_limit(self):
        self.assertEqual(resource_settings({}), {'kv_cache_memory_bytes': 2 << 30})
        self.assertEqual(resource_settings({'FISSILE_VLLM_KV_CACHE_BYTES': str(4 << 30)}),
                         {'kv_cache_memory_bytes': 4 << 30})
        for value in ('0', '1', 'bad', str(1 << 41)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_settings({'FISSILE_VLLM_KV_CACHE_BYTES': value})

    def test_registered_combinations(self):
        spec = get_backend('vllm')
        for provider in ('gb10-mixed-v1', 'gb10-mixed-v2'):
            spec.validate(SimpleNamespace(profile='qwen3.5-dense-9b', recipe='mixed-fp4-fp8-w4a16-v1',
                                          kernel_provider=provider))
        with self.assertRaises(ValueError):
            spec.validate(SimpleNamespace(profile='qwen3.5-dense-9b', recipe='mixed-fp4-fp8-w4a16-v1',
                                          kernel_provider='sm120-mixed-v1'))

    def test_session_reads_candidate_logprobs_and_restores_environment(self):
        from fissile.backends.vllm import engine
        calls = {}

        class LLM:
            def __init__(self, **kwargs):
                calls['kwargs'] = kwargs
                env = __import__('os').environ
                calls['env'] = tuple(env.get(k) for k in ('FISSILE_CONFIG', 'VLLM_ENABLE_V1_MULTIPROCESSING',
                                                          'VLLM_USE_V2_MODEL_RUNNER'))

            def generate(self, prompts, params, use_tqdm):
                calls['params'] = params
                logprobs = {t: SimpleNamespace(logprob=-float(t)) for t in params['logprob_token_ids']}
                return [SimpleNamespace(prompt_token_ids=prompts[0]['prompt_token_ids'],
                                        outputs=[SimpleNamespace(logprobs=[logprobs])])]

        registry = SimpleNamespace(register_model=lambda name, cls: calls.setdefault('registered', name))
        vllm = SimpleNamespace(LLM=LLM, ModelRegistry=registry, SamplingParams=lambda **k: k)
        inputs = SimpleNamespace(TokensPrompt=lambda **k: k)
        model = SimpleNamespace(MODELS={'Qwen3_5ForConditionalGeneration': object})
        config = SimpleNamespace(plan=SimpleNamespace(profile=SimpleNamespace(max_context=16384)))
        with patch('fissile.backends.vllm.engine.Config.load', return_value=config), patch(
                'fissile.backends.registry.get_backend', return_value=SimpleNamespace(validate=lambda _: None)), patch.dict(
                'sys.modules', {'vllm': vllm, 'vllm.inputs': inputs, 'fissile.backends.vllm.model': model}), patch.dict(
                'os.environ', {}, clear=True):
            import os
            with engine.open_session('/test/config.json', '/test/model') as session:
                score = session.score([5, 6, 7], [11, 3])
                with self.assertRaises(ValueError):
                    session.score([5, 6], list(range(MAX_CANDIDATES + 1)))
            self.assertNotIn('FISSILE_CONFIG', os.environ)
            self.assertNotIn('VLLM_ENABLE_V1_MULTIPROCESSING', os.environ)
            self.assertNotIn('VLLM_USE_V2_MODEL_RUNNER', os.environ)
        self.assertEqual(score.token_logprobs, (-11.0, -3.0))
        self.assertEqual(score.prompt_tokens, 3)
        self.assertEqual(calls['registered'], 'Qwen3_5ForConditionalGeneration')
        self.assertEqual(calls['env'], ('/test/config.json', '0', '0'))
        self.assertEqual(calls['params'], {'max_tokens': 1, 'temperature': 0.0, 'logprob_token_ids': [11, 3]})


class ValidateCounts(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace()
        keys = ['lm_head'] + ['layers.%d.p' % i for i in range(6)]
        self.runtime.expected_routes = {k: 'fp4' for k in keys}
        self.runtime.calls = {k: 2 for k in keys}
        self.runtime.profile = SimpleNamespace(head='lm_head')

    def check(self, step, **expected):
        before = dict(self.runtime.calls)
        step(self.runtime.calls)
        MixedRuntime.validate_counts(self.runtime, before, **expected)

    def test_body_then_head(self):
        def body(calls):
            for k in calls:
                if k != 'lm_head':
                    calls[k] += 1
        self.check(body, head=0, body=1)
        self.check(lambda calls: calls.__setitem__('lm_head', calls['lm_head'] + 1), head=1, body=0)

    def test_bypass_and_extra_head_rejected(self):
        def partial(calls):
            for k in list(calls)[2:]:
                calls[k] += 1
        with self.assertRaises(RuntimeError):
            self.check(partial, head=0, body=1)
        with self.assertRaises(RuntimeError):
            self.check(lambda calls: calls.__setitem__('lm_head', calls['lm_head'] + 2), head=1, body=0)


if __name__ == '__main__':
    unittest.main()
