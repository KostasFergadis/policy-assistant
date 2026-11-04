from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document: str
    section_number: str
    section_title: str
    content: str
    similarity: float
