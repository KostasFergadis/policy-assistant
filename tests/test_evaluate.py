from types import SimpleNamespace

import pytest

import evaluate
from app.providers.base import ProviderUnavailable, RetrievedChunk
from evaluate import EvalResult, score


def test_score_counts_each_metric():
    card = score([
        EvalResult(expected=("Leave Policy", "3.2"), retrieved=[("Leave Policy", "3.2")], cited=[("Leave Policy", "3.2")], grounded=True, declined=False, seconds=2.0, quotes_total=1, quotes_verified=1),
        EvalResult(expected=("Leave Policy", "3.3"), retrieved=[], cited=[], grounded=False, declined=False, seconds=4.0, quotes_total=2, quotes_verified=0),
        EvalResult(expected=None, retrieved=[], cited=[], grounded=False, declined=True, seconds=1.0, quotes_total=0, quotes_verified=0),
    ])
    assert card.retrieval_rate == 0.5
    assert card.citation_rate == 0.5
    assert card.refusal_rate == 1.0
    assert card.quote_pass_rate == pytest.approx(1 / 3)
    assert card.average_seconds == pytest.approx(7 / 3)


def test_answer_with_failed_quotes_is_not_counted_as_declined():
    card = score([
        EvalResult(expected=None, retrieved=[], cited=[], grounded=False, declined=False, seconds=1.0, quotes_total=1, quotes_verified=0),
    ])
    assert card.refusal_rate == 0.0


def test_provider_failure_counts_as_a_miss_and_run_continues(monkeypatch):
    class FailingProvider:
        def answer(self, question, chunks):
            raise ProviderUnavailable("slow")

    chunk = RetrievedChunk(1, "Leave Policy", "2.1", "Annual Leave", "Body", 0.9)
    monkeypatch.setattr(evaluate, "real_retrieve", lambda question: [chunk])
    settings = SimpleNamespace(min_similarity=0.5)
    item = {"question": "q", "document": None, "section_number": None}
    result = evaluate.run_question(item, settings, FailingProvider())
    assert result.declined is False
    assert result.cited == []
