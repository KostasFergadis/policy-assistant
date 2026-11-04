import pytest

from app.providers.base import RawCitation, RetrievedChunk
from app.verification import verify_citations


def make_chunk(chunk_id=1, content="Staff get 20 days\nof annual leave each year."):
    return RetrievedChunk(chunk_id, "Leave Policy", "2.1", "Annual Leave", content, 0.8)


CHUNK = make_chunk()


def test_exact_quote_passes():
    result = verify_citations([RawCitation(0, "20 days")], [CHUNK])
    assert result[0].quoted_text == "20 days"
    assert (result[0].document, result[0].section_number, result[0].section_title) == (
        "Leave Policy",
        "2.1",
        "Annual Leave",
    )


def test_whitespace_differences_pass():
    assert len(verify_citations([RawCitation(0, "20 days of  annual leave")], [CHUNK])) == 1


def test_invented_quote_removed():
    assert verify_citations([RawCitation(0, "30 days")], [CHUNK]) == []


def test_case_difference_removed():
    assert verify_citations([RawCitation(0, "STAFF GET")], [CHUNK]) == []


@pytest.mark.parametrize("index", [1, -1])
def test_unknown_reference_removed(index):
    assert verify_citations([RawCitation(index, "20 days")], [CHUNK]) == []


@pytest.mark.parametrize("quote", ["", "   ", "\n"])
def test_blank_quote_removed(quote):
    assert verify_citations([RawCitation(0, quote)], [CHUNK]) == []


def test_duplicate_quote_returned_once():
    citations = [RawCitation(0, "20 days"), RawCitation(0, "20 days")]
    assert len(verify_citations(citations, [CHUNK])) == 1


def test_same_quote_in_two_parts_of_section_returned_once():
    other = make_chunk(chunk_id=2, content="Reminder: 20 days.")
    citations = [RawCitation(0, "20 days"), RawCitation(1, "20 days")]
    assert len(verify_citations(citations, [CHUNK, other])) == 1
