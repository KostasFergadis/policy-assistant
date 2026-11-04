"""Score retrieval, citations and refusals against tests/eval_questions.json."""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.main import build_provider, real_retrieve
from app.verification import verify_citations

QUESTIONS = Path(__file__).parent / "tests" / "eval_questions.json"
Ref = tuple[str, str]


@dataclass(frozen=True)
class EvalResult:
    expected: Ref | None
    retrieved: list[Ref]
    cited: list[Ref]
    grounded: bool
    seconds: float
    quotes_total: int
    quotes_verified: int


@dataclass(frozen=True)
class Scorecard:
    retrieval_rate: float
    citation_rate: float
    refusal_rate: float
    quote_pass_rate: float
    average_seconds: float


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score(results: list[EvalResult]) -> Scorecard:
    answerable = [r for r in results if r.expected]
    unanswerable = [r for r in results if not r.expected]
    return Scorecard(
        retrieval_rate=_rate(sum(r.expected in r.retrieved for r in answerable), len(answerable)),
        citation_rate=_rate(sum(r.expected in r.cited for r in answerable), len(answerable)),
        refusal_rate=_rate(sum(not r.grounded for r in unanswerable), len(unanswerable)),
        quote_pass_rate=_rate(
            sum(r.quotes_verified for r in results), sum(r.quotes_total for r in results)
        ),
        average_seconds=sum(r.seconds for r in results) / len(results),
    )


def run_question(item: dict, settings, provider) -> EvalResult:
    expected = (item["document"], item["section_number"]) if item["document"] else None
    started = time.perf_counter()
    chunks = real_retrieve(item["question"])
    retrieved = [(c.document, c.section_number) for c in chunks]
    best = chunks[0].similarity if chunks else 0.0
    answer, citations, total = "(below similarity threshold)", [], 0
    if provider and best >= settings.min_similarity:
        result = provider.answer(item["question"], chunks)
        answer, total = result.answer, len(result.citations)
        citations = verify_citations(result.citations, chunks)
    seconds = time.perf_counter() - started

    cited = [(c.document, c.section_number) for c in citations]
    outcome = "cited" if expected in cited else ("no citation" if not cited else "other section")
    if not expected:
        outcome = "declined" if not cited else "NOT DECLINED"
    found = "yes" if expected in retrieved else "no"
    print(f"best={best:.3f} top5={found:3} {outcome:13} {seconds:5.1f}s  {item['question']}")
    if provider:
        print(f"    {answer[:150]}")
    return EvalResult(expected, retrieved, cited, bool(cited), seconds, total, len(citations))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--think", action="store_true", help="turn on model thinking")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the answer model")
    args = parser.parse_args()

    settings = get_settings()
    if args.think:
        settings = dataclasses.replace(settings, ollama_think=True)
    provider = None if args.retrieval_only else build_provider(settings)

    items = json.loads(QUESTIONS.read_text())
    card = score([run_question(item, settings, provider) for item in items])

    print(f"\nRetrieval: {card.retrieval_rate:.0%} (target 90%)")
    if provider:
        print(f"Cited expected section: {card.citation_rate:.0%} (target 80%)")
        print(f"Declined unanswerable: {card.refusal_rate:.0%} (target 100%)")
        print(f"Quotes verified: {card.quote_pass_rate:.0%}")
    print(f"Average time: {card.average_seconds:.1f}s")


if __name__ == "__main__":
    main()
