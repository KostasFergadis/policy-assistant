import json
import logging
import re

import httpx

from app.providers.base import (
    ProviderAnswer,
    ProviderOutputError,
    ProviderUnavailable,
    RawCitation,
    RetrievedChunk,
)
from app.providers.prompts import OLLAMA_FORMAT_RULES, SYSTEM_PROMPT
from app.verification import quote_in_text

logger = logging.getLogger(__name__)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"chunk_ref": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["chunk_ref", "quote"],
            },
        },
    },
    "required": ["covered", "answer", "citations"],
}


def _user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    sections = "\n\n".join(
        f"[S{i}] {c.document}, section {c.section_number} {c.section_title}\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )
    return (
        f"Policy sections:\n\n{sections}\n\nQuestion: {question}\n\n"
        "Do not mention labels such as S1 in the answer text. Name the policy instead."
    )


def _chunk_index(chunk_ref: str, quote: str, chunks: list[RetrievedChunk]) -> int:
    """Find the cited chunk from its label or section, or else from the quote itself.

    Small models fill the reference field inconsistently, so a reference that cannot
    be resolved falls back to the retrieved section that contains the quote.
    """
    label = chunk_ref.strip().strip("[]")
    if label.startswith("S") and label[1:].isdigit():
        index = int(label[1:]) - 1
        return index if 0 <= index < len(chunks) else -1
    numbers = set(re.findall(r"\d+(?:\.\d+)*", label))
    for index, chunk in enumerate(chunks):
        if chunk.document.lower() in label.lower() and chunk.section_number in numbers:
            return index
    for index, chunk in enumerate(chunks):
        if quote_in_text(quote, chunk.content):
            return index
    return -1


def _parse(content: str, chunks: list[RetrievedChunk]) -> ProviderAnswer:
    data = json.loads(content)
    if not isinstance(data.get("answer"), str) or not isinstance(data.get("citations"), list):
        raise ValueError("Unexpected answer shape")
    citations = [
        RawCitation(_chunk_index(item["chunk_ref"], item["quote"], chunks), item["quote"])
        for item in data["citations"]
    ]
    return ProviderAnswer(data["answer"], citations, data.get("covered") is not False)


class OllamaProvider:
    def __init__(self, settings, client: httpx.Client | None = None):
        self.settings = settings
        self.model_name = settings.ollama_model
        self.client = client or httpx.Client(timeout=httpx.Timeout(10, read=180))

    def _unavailable(self) -> ProviderUnavailable:
        return ProviderUnavailable("Could not reach Ollama. Start it with: ollama serve")

    def _missing_model(self) -> ProviderUnavailable:
        return ProviderUnavailable(f"Answer model is missing. Run: ollama pull {self.model_name}")

    def _chat(self, question: str, chunks: list[RetrievedChunk]) -> str:
        try:
            response = self.client.post(
                f"{self.settings.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "stream": False,
                    "think": self.settings.ollama_think,
                    "format": ANSWER_SCHEMA,
                    "options": {"temperature": 0, "num_ctx": 8192},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT + OLLAMA_FORMAT_RULES},
                        {"role": "user", "content": _user_message(question, chunks)},
                    ],
                },
            )
        except httpx.TimeoutException as error:
            raise ProviderUnavailable("The model took too long to answer. Try again.") from error
        except httpx.HTTPError as error:
            raise self._unavailable() from error
        if response.status_code == 404:
            raise self._missing_model()
        if response.status_code != 200:
            raise ProviderUnavailable(f"Ollama returned an error ({response.status_code}).")
        return response.json()["message"]["content"]

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> ProviderAnswer:
        for attempt in (1, 2):
            content = self._chat(question, chunks)
            try:
                return _parse(content, chunks)
            except (ValueError, KeyError, TypeError, AttributeError):
                logger.warning("Unusable model output on attempt %d", attempt)
        raise ProviderOutputError("The model returned an answer that could not be read.")

    def check(self) -> None:
        try:
            response = self.client.get(f"{self.settings.ollama_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise self._unavailable() from error
        names = {model["name"] for model in response.json()["models"]}
        if self.model_name not in names:
            raise self._missing_model()
