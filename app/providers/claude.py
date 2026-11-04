import os

import anthropic

from app.providers.base import (
    ProviderAnswer,
    ProviderOutputError,
    ProviderUnavailable,
    RawCitation,
    RetrievedChunk,
)
from app.providers.prompts import SYSTEM_PROMPT

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _document(chunk: RetrievedChunk) -> dict:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": chunk.content},
        "title": f"{chunk.document}, section {chunk.section_number} {chunk.section_title}",
        "citations": {"enabled": True},
    }


class ClaudeProvider:
    def __init__(self, settings, client=None):
        self.model_name = settings.claude_model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def check(self) -> None:
        if self._client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderUnavailable("Claude needs a key. Add ANTHROPIC_API_KEY to the .env file.")

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> ProviderAnswer:
        self.check()
        try:
            response = self.client.beta.messages.create(
                model=self.model_name,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": "low"},
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *[_document(chunk) for chunk in chunks],
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
        except anthropic.AuthenticationError as error:
            raise ProviderUnavailable("Claude rejected the API key. Check the .env file.") from error
        except anthropic.RateLimitError as error:
            raise ProviderUnavailable("Claude is rate limited. Try again shortly.") from error
        except anthropic.APIStatusError as error:
            raise ProviderUnavailable(f"Claude returned an error ({error.status_code}).") from error
        except anthropic.APIConnectionError as error:
            raise ProviderUnavailable("Could not reach the Claude API.") from error

        if response.stop_reason == "refusal":
            raise ProviderOutputError("Claude declined to answer this question.")

        text_blocks = [block for block in response.content if block.type == "text"]
        citations = [
            RawCitation(citation.document_index, citation.cited_text)
            for block in text_blocks
            for citation in block.citations or []
        ]
        return ProviderAnswer("".join(block.text for block in text_blocks), citations)
