import logging
import re
from dataclasses import dataclass

from app.providers.base import RawCitation, RetrievedChunk

logger = logging.getLogger(__name__)

MIN_QUOTE_WORDS = 2


@dataclass(frozen=True)
class VerifiedCitation:
    document: str
    section_number: str
    section_title: str
    quoted_text: str


def _normalize(text: str) -> str:
    return " ".join(text.split())


def quote_in_text(quote: str, text: str) -> bool:
    """Match whole words only, so '5 days' does not match inside '25 days'."""
    return re.search(rf"(?<!\w){re.escape(_normalize(quote))}(?!\w)", _normalize(text)) is not None


def _is_substantial(quote: str) -> bool:
    return len(quote.split()) >= MIN_QUOTE_WORDS and any(c.isalnum() for c in quote)


def verify_citations(
    citations: list[RawCitation], chunks: list[RetrievedChunk]
) -> list[VerifiedCitation]:
    verified: list[VerifiedCitation] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        quote = _normalize(citation.quote)
        if not _is_substantial(quote):
            logger.warning("Removed quote too short to verify: %r", citation)
            continue
        if not 0 <= citation.chunk_index < len(chunks):
            logger.warning("Removed citation with unknown reference: %r", citation)
            continue
        chunk = chunks[citation.chunk_index]
        if not quote_in_text(quote, chunk.content):
            logger.warning("Removed quote not found in source: %r", citation)
            continue
        key = (chunk.document, chunk.section_number, quote)
        if key in seen:
            continue
        seen.add(key)
        verified.append(
            VerifiedCitation(chunk.document, chunk.section_number, chunk.section_title, quote)
        )
    return verified
