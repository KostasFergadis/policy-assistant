import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.chunking import ParsedDocument
from app.config import get_settings
from app.providers.base import RetrievedChunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  title     TEXT NOT NULL,
  filename  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS chunks (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  document_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section_number  TEXT NOT NULL,
  section_title   TEXT NOT NULL,
  part            INTEGER NOT NULL DEFAULT 1,
  content         TEXT NOT NULL,
  embedding       VECTOR(768) NOT NULL,
  UNIQUE (document_id, section_number, part)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
"""


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or get_settings().database_url, connect_timeout=3)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def chunk_count(conn: psycopg.Connection) -> int:
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    conn.commit()
    return count


def replace_all(
    conn: psycopg.Connection, docs: list[tuple[ParsedDocument, list[list[float]]]]
) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM documents")
        for document, embeddings in docs:
            document_id = conn.execute(
                "INSERT INTO documents (title, filename) VALUES (%s, %s) RETURNING id",
                (document.title, document.filename),
            ).fetchone()[0]
            for section, embedding in zip(document.sections, embeddings, strict=True):
                conn.execute(
                    """
                    INSERT INTO chunks
                      (document_id, section_number, section_title, part, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        section.section_number,
                        section.section_title,
                        section.part,
                        section.content,
                        Vector(embedding),
                    ),
                )


def search(
    conn: psycopg.Connection, query_embedding: list[float], limit: int = 5
) -> list[RetrievedChunk]:
    query = Vector(query_embedding)
    rows = conn.execute(
        """
        SELECT c.id, d.title, c.section_number, c.section_title, c.content,
               1 - (c.embedding <=> %s) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """,
        (query, query, limit),
    ).fetchall()
    conn.commit()
    return [RetrievedChunk(*row) for row in rows]
