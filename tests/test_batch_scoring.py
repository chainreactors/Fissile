"""CPU-only checks for batched multi-question scoring."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fissile.backends.contracts import CheckedSession, Score
from fissile.backends.sglang import engine as sglang_engine
from fissile.backends.sglang.engine import (
    CACHE_GRID,
    MAX_BATCH,
    PREFIX_MIN,
    PrefixCache,
    batch_settings,
    open_session,
    scoring_engine,
)
from fissile.models import get_profile
from fissile.providers import get_provider

PROFILE = get_profile("qwen3.5-dense-9b")


class Serial:
    def __init__(self):
        self.calls = []

    def score(self, ids, candidates):
        self.calls.append(("score", ids))
        return Score(tuple(-1.0 for _ in candidates), len(ids))


class Batched(Serial):
    def score_batch(self, items):
        self.calls.append(("batch", len(items)))
        return [Score(tuple(-1.0 for _ in c), len(ids)) for ids, c in items]


class FakeEngine:
    """Records score/flush calls; one row of -1.0 per item and label."""

    def __init__(self):
        self.calls = []

    def score(self, *, query, items, label_token_ids, **kw):
        self.calls.append(("score", [len(i) for i in items], label_token_ids))
        labels = label_token_ids if isinstance(label_token_ids[0], list) else [label_token_ids] * len(items)
        rows = [[-1.0] * len(row) for row in labels]
        return type("R", (), dict(token_logprobs=rows, prompt_tokens=sum(len(i) for i in items)))

    def flush_cache(self):
        self.calls.append(("flush",))
        return type("F", (), dict(success=True))


@contextmanager
def fake_session(batch=True):
    engine = FakeEngine()

    @contextmanager
    def fake_engine(*args, **kwargs):
        yield engine

    with mock.patch.object(sglang_engine, "scoring_engine", fake_engine):
        with open_session("config.json", "model", batch=batch) as session:
            yield session, engine


def ids(n, start=1):
    return list(range(start, start + n))


class BatchScoring(unittest.TestCase):
    def test_engine_settings(self):
        self.assertEqual(batch_settings(False, 16384),
                         dict(disable_radix_cache=True, max_running_requests=1, max_total_tokens=32768))
        self.assertEqual(batch_settings(True, 16384),
                         dict(disable_radix_cache=False, max_running_requests=MAX_BATCH, max_total_tokens=65536,
                              page_size=1, enable_deterministic_inference=True,
                              mamba_radix_cache_strategy="no_buffer", disable_overlap_schedule=True))
        self.assertEqual(PREFIX_MIN % CACHE_GRID, 0)
        with self.assertRaises(ValueError):
            with scoring_engine("config.json", "model", batch=True, prefill_cuda_graph=True):
                pass

    def test_batch_needs_row_independent_provider(self):
        for name in ("gb10-mixed-v1", "sm120-mixed-v1", "gb10-mixed-v2", "gb10-mixed-v2-calibration"):
            self.assertFalse(get_provider(name).row_independent, name)
        self.assertTrue(get_provider("gb10-mixed-v2-static").row_independent)
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "c.json"
            config.write_text(json.dumps({
                "format": "fissile-config-v2", "profile": PROFILE.name, "recipe": "mixed-fp4-fp8-w4a16-v1",
                "kernel_provider": "gb10-mixed-v2", "manifest": "m.json", "artifacts_dir": "packs",
                "report_dir": "reports",
                "resources": {"tensorfold_library": {"path": "tf.so"},
                              "silu_lut": {"path": "silu.safetensors", "sha256": "0" * 64},
                              "fp4_gemm_library": {"path": "gemm.so"}}}))
            with self.assertRaisesRegex(ValueError, "activation_scales"):
                with scoring_engine(config, "model", batch=True):
                    pass

    def test_score_batch_uses_engine_batching_or_falls_back(self):
        items = [([1, 2, 3], [7, 8]), ([1, 2, 4, 5], [9])]
        batched = CheckedSession(Batched(), PROFILE)
        self.assertEqual([r.prompt_tokens for r in batched.score_batch(items)], [3, 4])
        self.assertEqual(batched.session.calls, [("batch", 2)])
        serial = CheckedSession(Serial(), PROFILE)
        self.assertEqual(len(serial.score_batch(items)), 2)
        self.assertEqual([c[0] for c in serial.session.calls], ["score", "score"])

    def test_score_batch_validates_items_and_results(self):
        session = CheckedSession(Batched(), PROFILE)
        for items in ([], [([1], [7])], [([1, 2], [7, 7])], [([1, 2], [PROFILE.vocab_size])]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                session.score_batch(items)

        class Short(Batched):
            def score_batch(self, items):
                return super().score_batch(items)[:-1]

        with self.assertRaises(RuntimeError):
            CheckedSession(Short(), PROFILE).score_batch([([1, 2], [7]), ([1, 3], [8])])


class Cache(unittest.TestCase):
    def test_only_extensions_of_off_grid_prompts_need_a_clear(self):
        cache = PrefixCache(grid=4)
        cache.add([ids(6)])  # ends off the grid
        self.assertEqual(cache.plan([ids(9)]), "clear")  # continues it at 6
        self.assertEqual(cache.plan([ids(6)]), "reuse")  # the same prompt again
        self.assertEqual(cache.plan([ids(5) + [99, 100]]), "reuse")  # branches inside it
        self.assertEqual(cache.plan([ids(3)]), "reuse")  # a prefix of it
        cache.add([ids(8, 50)])  # ends on the grid: never tracked
        self.assertEqual(cache.plan([ids(12, 50)]), "reuse")
        cache.clear()
        self.assertEqual(cache.plan([ids(9)]), "reuse")

    def test_prompts_of_one_call_and_limit(self):
        cache = PrefixCache(grid=4, limit=3)
        self.assertEqual(cache.plan([ids(6), ids(9)]), "split")
        self.assertEqual(cache.plan([ids(8), ids(9)]), "reuse")
        self.assertEqual(cache.plan([ids(6), ids(6)]), "reuse")
        cache.add([ids(2, 100), ids(2, 200)])
        self.assertEqual(cache.plan([ids(2, 300)]), "reuse")
        self.assertEqual(cache.plan([ids(2, 300), ids(2, 400)]), "clear")


class Session(unittest.TestCase):
    def test_batch_prefills_aligned_prefix_and_passes_item_labels(self):
        shared = ids(PREFIX_MIN + 40)
        items = [(shared + [900, 901], [7, 8]), (shared + [902], [9])]
        with fake_session() as (session, engine):
            scores = session.score_batch(items)
        self.assertEqual([s.prompt_tokens for s in scores], [len(shared) + 2, len(shared) + 1])
        self.assertEqual([len(s.token_logprobs) for s in scores], [2, 1])
        self.assertEqual(engine.calls, [("score", [PREFIX_MIN], [[7]]),
                                        ("score", [len(shared) + 2, len(shared) + 1], [[7, 8], [9]])])

    def test_short_prefix_is_not_prefilled_and_cache_is_kept(self):
        with fake_session() as (session, engine):
            session.score_batch([(ids(100) + [900], [7]), (ids(100) + [901], [8])])
            session.score(ids(100) + [902], [7])
        self.assertEqual([c[0] for c in engine.calls], ["score", "score"])

    def test_cache_cleared_before_extending_an_off_grid_prompt(self):
        with fake_session() as (session, engine):
            session.score(ids(70), [7])
            session.score(ids(80), [7])  # continues the prompt ending at 70
            session.score(ids(90, 5), [7])
        self.assertEqual([c[0] for c in engine.calls], ["score", "flush", "score", "score"])

    def test_items_extending_each_other_are_scored_one_by_one(self):
        with fake_session() as (session, engine):
            session.score_batch([(ids(70), [7]), (ids(80), [8])])
        self.assertEqual([c[0] for c in engine.calls], ["score", "flush", "score"])

    def test_without_batch_every_item_is_scored_alone(self):
        with fake_session(batch=False) as (session, engine):
            session.score_batch([(ids(70), [7]), (ids(80), [8])])
        self.assertEqual([c[0] for c in engine.calls], ["score", "score"])


if __name__ == "__main__":
    unittest.main()
