import pytest

from evaluate import EvalResult, score


def test_score_counts_each_metric():
    card = score([
        EvalResult(expected=("Leave Policy", "3.2"), retrieved=[("Leave Policy", "3.2")], cited=[("Leave Policy", "3.2")], grounded=True, seconds=2.0, quotes_total=1, quotes_verified=1),
        EvalResult(expected=("Leave Policy", "3.3"), retrieved=[], cited=[], grounded=False, seconds=4.0, quotes_total=2, quotes_verified=0),
        EvalResult(expected=None, retrieved=[], cited=[], grounded=False, seconds=1.0, quotes_total=0, quotes_verified=0),
    ])
    assert card.retrieval_rate == 0.5
    assert card.citation_rate == 0.5
    assert card.refusal_rate == 1.0
    assert card.quote_pass_rate == pytest.approx(1 / 3)
    assert card.average_seconds == pytest.approx(7 / 3)


def test_grounded_unanswerable_question_counts_as_not_declined():
    card = score([
        EvalResult(expected=None, retrieved=[], cited=[("Leave Policy", "3.2")], grounded=True, seconds=1.0, quotes_total=1, quotes_verified=1),
    ])
    assert card.refusal_rate == 0.0
