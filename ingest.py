"""Load every policy in policies/ into Postgres. Safe to run repeatedly."""

import sys
from pathlib import Path

import psycopg

from app import db
from app.chunking import PolicyFormatError, parse_policy
from app.embeddings import EmbeddingError, embed_documents, embedding_text

POLICY_DIR = Path(__file__).parent / "policies"


def main() -> int:
    try:
        files = sorted(POLICY_DIR.glob("*.md"))
        if not files:
            raise PolicyFormatError(f"No .md files found in {POLICY_DIR}")
        parsed = [parse_policy(f.read_text(encoding="utf-8"), f.name) for f in files]
        loads = [
            (doc, embed_documents([embedding_text(doc.title, s) for s in doc.sections]))
            for doc in parsed
        ]
        with db.connect() as conn:
            db.init_schema(conn)
            db.replace_all(conn, loads)
            count = db.chunk_count(conn)
    except (PolicyFormatError, EmbeddingError) as error:
        print(f"Ingest failed: {error}", file=sys.stderr)
        return 1
    except psycopg.OperationalError:
        print("Ingest failed: could not reach Postgres. Run: docker compose up -d db", file=sys.stderr)
        return 1
    print(f"Loaded {len(parsed)} documents and {count} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
