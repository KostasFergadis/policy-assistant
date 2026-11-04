from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.providers.base import (
    ProviderOutputError,
    ProviderUnavailable,
    RawCitation,
    RetrievedChunk,
)
from app.providers.claude import ClaudeProvider

SETTINGS = SimpleNamespace(claude_model="claude-opus-5-5")
CHUNKS = [
    RetrievedChunk(1, "Leave Policy", "2.1", "Annual Leave", "25 days.", 0.8),
    RetrievedChunk(2, "Expense Policy", "3.2", "Mileage", "45 pence.", 0.7),
]


def text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def citation(index, cited):
    return SimpleNamespace(document_index=index, cited_text=cited)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response or SimpleNamespace(stop_reason="end_turn", content=[text_block("A")])
        self.error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_sends_each_chunk_as_cited_document():
    client = FakeClient()
    ClaudeProvider(SETTINGS, client=client).answer("how much?", CHUNKS)
    content = client.calls[0]["messages"][0]["content"]
    documents = [block for block in content if block["type"] == "document"]
    assert len(documents) == 2
    assert documents[0]["citations"] == {"enabled": True}
    assert documents[0]["source"] == {"type": "text", "media_type": "text/plain", "data": "25 days."}
    assert "Leave Policy" in documents[0]["title"] and "2.1" in documents[0]["title"]
    assert content[-1] == {"type": "text", "text": "how much?"}


def test_request_settings():
    client = FakeClient()
    ClaudeProvider(SETTINGS, client=client).answer("q", CHUNKS)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["max_tokens"] == 16000
    assert "thinking" not in call


def test_maps_citations():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[text_block("You get "), text_block("45 pence.", [citation(1, "x")])],
    )
    result = ClaudeProvider(SETTINGS, client=FakeClient(response)).answer("q", CHUNKS)
    assert result.answer == "You get 45 pence."
    assert result.citations == [RawCitation(1, "x")]


def test_ignores_non_text_blocks():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking=""), text_block("A")],
    )
    assert ClaudeProvider(SETTINGS, client=FakeClient(response)).answer("q", CHUNKS).answer == "A"


def test_refusal_raises():
    response = SimpleNamespace(stop_reason="refusal", content=[])
    with pytest.raises(ProviderOutputError):
        ClaudeProvider(SETTINGS, client=FakeClient(response)).answer("q", CHUNKS)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(SETTINGS).check()


def test_check_passes_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    ClaudeProvider(SETTINGS).check()


def api_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return [
        anthropic.RateLimitError("slow", response=httpx2.Response(429, request=request), body=None),
        anthropic.AuthenticationError("no", response=httpx2.Response(401, request=request), body=None),
        anthropic.APIConnectionError(request=request),
    ]


@pytest.mark.parametrize("error", api_errors())
def test_api_errors_become_unavailable(error):
    with pytest.raises(ProviderUnavailable):
        ClaudeProvider(SETTINGS, client=FakeClient(error=error)).answer("q", CHUNKS)
