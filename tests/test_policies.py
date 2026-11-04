from pathlib import Path

from app.chunking import parse_policy


def test_all_policies_parse_and_are_ascii():
    files = sorted(Path("policies").glob("*.md"))
    assert len(files) == 7
    for f in files:
        text = f.read_text()
        assert text.isascii(), f.name
        assert len(parse_policy(text, f.name).sections) >= 6
