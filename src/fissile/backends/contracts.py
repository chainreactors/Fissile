"""Stable, engine-free candidate-scoring contract."""

from dataclasses import dataclass
from typing import Protocol, Sequence
import math


@dataclass(frozen=True)
class Score:
    token_logprobs: tuple[float, ...]
    prompt_tokens: int


class ScoringSession(Protocol):
    def score(
        self, input_ids: Sequence[int], candidate_token_ids: Sequence[int]
    ) -> Score: ...


class CheckedSession:
    """Validate at the API boundary; engines own execution and resource cleanup."""

    def __init__(self, session, profile):
        self.session = session
        self.profile = profile
        self.closed = False

    def score(self, input_ids, candidate_token_ids):
        if self.closed:
            raise RuntimeError("Scoring session is closed")
        ids, candidates = tuple(input_ids), tuple(candidate_token_ids)
        if not (
            1 < len(ids) <= self.profile.max_context
            and candidates
            and all(
                type(i) is int and 0 <= i < self.profile.vocab_size
                for i in ids + candidates
            )
            and len(set(candidates)) == len(candidates)
        ):
            raise ValueError(
                "Expected complete token IDs and distinct single-token candidates"
            )
        result = self.session.score(ids, candidates)
        if (
            not isinstance(result, Score)
            or result.prompt_tokens != len(ids)
            or len(result.token_logprobs) != len(candidates)
        ):
            raise RuntimeError("Backend violated the scoring result contract")
        if any(not math.isfinite(x) for x in result.token_logprobs):
            raise RuntimeError("Backend returned nonfinite candidate scores")
        return result
