import psycopg
import pytest

from app import db
from app.chunking import ParsedDocument, Section

ADMIN_URL = "postgresql://policy:policy@127.0.0.1:5433/postgres"
TEST_URL = "postgresql://policy:policy@127.0.0.1:5433/policy_test"


def vector(position: int) -> list[float]:
    values = [0.0] * 768
    values[position] = 1.0
    return values


def doc(*numbers: str) -> ParsedDocument:
    return ParsedDocument("Doc", "doc.md", [Section(n, f"Title {n}", 1, f"Body {n}") for n in numbers])


@pytest.fixture
def conn():
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'policy_test'").fetchone():
            admin.execute("CREATE DATABASE policy_test")
    connection = db.connect(TEST_URL)
    connection.execute("DROP TABLE IF EXISTS chunks, documents")
    connection.commit()
    db.init_schema(connection)
    yield connection
    connection.close()


def test_search_returns_closest_first(conn):
    db.replace_all(conn, [(doc("1.1", "1.2"), [vector(0), vector(1)])])
    results = db.search(conn, vector(1))
    assert results[0].section_number == "1.2"
    assert results[0].document == "Doc"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].similarity == pytest.approx(0.0)


def test_search_respects_limit(conn):
    db.replace_all(conn, [(doc("1.1", "1.2", "1.3"), [vector(0), vector(1), vector(2)])])
    assert len(db.search(conn, vector(0), limit=2)) == 2


def test_replace_all_twice_leaves_no_duplicates(conn):
    load = [(doc("1.1", "1.2"), [vector(0), vector(1)])]
    db.replace_all(conn, load)
    db.replace_all(conn, load)
    assert db.chunk_count(conn) == 2


def test_replace_all_removes_deleted_sections(conn):
    db.replace_all(conn, [(doc("1.1", "1.2"), [vector(0), vector(1)])])
    db.replace_all(conn, [(doc("1.1"), [vector(0)])])
    assert db.chunk_count(conn) == 1


def test_failed_replace_keeps_existing_data(conn):
    db.replace_all(conn, [(doc("1.1", "1.2"), [vector(0), vector(1)])])
    with pytest.raises(psycopg.Error):
        db.replace_all(conn, [(doc("1.1"), [[1.0, 2.0]])])
    assert db.chunk_count(conn) == 2
