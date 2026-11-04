import json

import httpx
import pytest

from app import embeddings
from app.chunking import Section
from app.embeddings import EmbeddingError, embed_documents, embed_query, embedding_text


def use_transport(monkeypatch, handler):
    monkeypatch.setattr(embeddings, "_transport", httpx.MockTransport(handler))


def echo_handler(seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in body["input"]]})

    return handler


def test_embed_query_uses_query_prefix(monkeypatch):
    seen = []
    use_transport(monkeypatch, echo_handler(seen))
    assert len(embed_query("hello")) == 768
    assert seen[0][0] == "/api/embed"
    assert seen[0][1]["input"] == ["search_query: hello"]


def test_embed_documents_uses_document_prefix(monkeypatch):
    seen = []
    use_transport(monkeypatch, echo_handler(seen))
    assert len(embed_documents(["a", "b"])) == 2
    assert seen[0][1]["input"] == ["search_document: a", "search_document: b"]


def test_embedding_text_includes_context():
    text = embedding_text("Leave Policy", Section("2.1", "Annual Leave", 1, "Staff get 25 days."))
    assert "Leave Policy" in text and "2.1 Annual Leave" in text and "Staff get 25 days." in text


def test_connection_error_names_fix(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    use_transport(monkeypatch, handler)
    with pytest.raises(EmbeddingError, match="ollama serve"):
        embed_query("hello")


def test_missing_model_names_fix(monkeypatch):
    use_transport(monkeypatch, lambda request: httpx.Response(404, json={"error": "not found"}))
    with pytest.raises(EmbeddingError, match="ollama pull nomic-embed-text"):
        embed_query("hello")
