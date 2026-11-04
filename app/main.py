import logging
from pathlib import Path
from typing import Annotated, Callable

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from app import db
from app.config import Settings, get_settings
from app.embeddings import EmbeddingError, embed_query
from app.providers.base import (
    AnswerProvider,
    ProviderOutputError,
    ProviderUnavailable,
    RetrievedChunk,
)
from app.providers.claude import ClaudeProvider
from app.providers.ollama import OllamaProvider
from app.verification import verify_citations

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this in the policies."
STATIC_DIR = Path(__file__).parent.parent / "static"
DB_DOWN_MESSAGE = "Could not reach the database. Run: docker compose up -d --wait db"
EMPTY_INDEX_MESSAGE = "No policies are loaded. Run: python ingest.py"


class ServiceUnavailable(Exception):
    pass


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


def real_retrieve(question: str) -> list[RetrievedChunk]:
    query_embedding = embed_query(question)
    try:
        with db.connect() as conn:
            if db.chunk_count(conn) == 0:
                raise ServiceUnavailable(EMPTY_INDEX_MESSAGE)
            return db.search(conn, query_embedding)
    except psycopg.errors.UndefinedTable as error:
        raise ServiceUnavailable(EMPTY_INDEX_MESSAGE) from error
    except psycopg.OperationalError as error:
        raise ServiceUnavailable(DB_DOWN_MESSAGE) from error


def build_provider(settings: Settings) -> AnswerProvider:
    if settings.answer_provider == "ollama":
        return OllamaProvider(settings)
    if settings.answer_provider == "claude":
        return ClaudeProvider(settings)
    raise ValueError(f"Unknown ANSWER_PROVIDER: {settings.answer_provider}")


def _database_status() -> dict:
    try:
        with db.connect() as conn:
            if db.chunk_count(conn) == 0:
                return {"ok": False, "detail": EMPTY_INDEX_MESSAGE}
    except psycopg.errors.UndefinedTable:
        return {"ok": False, "detail": EMPTY_INDEX_MESSAGE}
    except psycopg.OperationalError:
        return {"ok": False, "detail": DB_DOWN_MESSAGE}
    return {"ok": True}


def _ollama_status() -> dict:
    try:
        embed_query("health check")
    except EmbeddingError as error:
        return {"ok": False, "detail": str(error)}
    return {"ok": True}


def _provider_status(provider: AnswerProvider) -> dict:
    try:
        provider.check()
    except ProviderUnavailable as error:
        return {"ok": False, "detail": str(error)}
    return {"ok": True}


def create_app(
    retrieve: Callable[[str], list[RetrievedChunk]] | None = None,
    provider: AnswerProvider | None = None,
) -> FastAPI:
    settings = get_settings()
    retrieve = retrieve or real_retrieve
    provider = provider or build_provider(settings)
    app = FastAPI(title="Policy Assistant")

    def detail_response(status: int, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @app.exception_handler(ServiceUnavailable)
    @app.exception_handler(EmbeddingError)
    @app.exception_handler(ProviderUnavailable)
    def unavailable(request: Request, error: Exception) -> JSONResponse:
        return detail_response(503, error)

    @app.exception_handler(ProviderOutputError)
    def bad_output(request: Request, error: Exception) -> JSONResponse:
        return detail_response(502, error)

    @app.exception_handler(Exception)
    def unexpected(request: Request, error: Exception) -> JSONResponse:
        logger.exception("Unexpected error", exc_info=error)
        return JSONResponse(status_code=500, content={"detail": "Something went wrong."})

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        chunks = retrieve(request.question)
        retrieved = [
            {
                "document": c.document,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "similarity": c.similarity,
            }
            for c in chunks
        ]
        if not chunks or chunks[0].similarity < settings.min_similarity:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "grounded": False,
                "citations": [],
                "retrieved": retrieved,
                "model": provider.model_name,
            }
        result = provider.answer(request.question, chunks)
        citations = verify_citations(result.citations, chunks)
        return {
            "answer": result.answer,
            "grounded": bool(citations),
            "citations": [vars(citation) for citation in citations],
            "retrieved": retrieved,
            "model": provider.model_name,
        }

    @app.get("/api/health")
    def health() -> dict:
        return {
            "database": _database_status(),
            "ollama": _ollama_status(),
            "provider": _provider_status(provider),
            "model": provider.model_name,
        }

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
