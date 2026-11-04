import logging
from dataclasses import dataclass

from app.providers.base import RawCitation, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedCitation:
    document: str
    section_number: str
    section_title: str
    quoted_text: str


def _normalize(text: str) -> str:
    return " ".join(text.split())


def verify_citations(
    citations: list[RawCitation], chunks: list[RetrievedChunk]
) -> list[VerifiedCitation]:
    verified: list[VerifiedCitation] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        quote = _normalize(citation.quote)
        if not quote:
            logger.warning("Removed blank quote")
            continue
        if not 0 <= citation.chunk_index < len(chunks):
            logger.warning("Removed citation with unknown reference: %r", citation)
            continue
        chunk = chunks[citation.chunk_index]
        if quote not in _normalize(chunk.content):
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
