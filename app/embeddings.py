import httpx

from app.chunking import Section
from app.config import get_settings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_transport: httpx.BaseTransport | None = None


class EmbeddingError(Exception):
    pass


def embedding_text(doc_title: str, section: Section) -> str:
    return f"{doc_title}\n{section.section_number} {section.section_title}\n{section.content}"


def _embed(inputs: list[str]) -> list[list[float]]:
    settings = get_settings()
    try:
        with httpx.Client(transport=_transport, timeout=60) as client:
            response = client.post(
                f"{settings.ollama_url}/api/embed",
                json={"model": settings.embedding_model, "input": inputs},
            )
    except httpx.HTTPError as error:
        raise EmbeddingError("Could not reach Ollama. Start it with: ollama serve") from error
    if response.status_code == 404:
        raise EmbeddingError(
            f"Embedding model is missing. Run: ollama pull {settings.embedding_model}"
        )
    if response.status_code != 200:
        raise EmbeddingError(f"Ollama returned an error ({response.status_code}) while embedding.")
    return response.json()["embeddings"]


def embed_documents(items: list[str]) -> list[list[float]]:
    return _embed([DOCUMENT_PREFIX + item for item in items])


def embed_query(text: str) -> list[float]:
    return _embed([QUERY_PREFIX + text])[0]
