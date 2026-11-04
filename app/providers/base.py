from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document: str
    section_number: str
    section_title: str
    content: str
    similarity: float


@dataclass(frozen=True)
class RawCitation:
    chunk_index: int
    quote: str


@dataclass(frozen=True)
class ProviderAnswer:
    answer: str
    citations: list[RawCitation]
    covered: bool = True


class ProviderUnavailable(Exception):
    pass


class ProviderOutputError(Exception):
    pass


class AnswerProvider(Protocol):
    model_name: str

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> ProviderAnswer: ...

    def check(self) -> None: ...
