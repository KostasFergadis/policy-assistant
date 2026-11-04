import pytest

from app.chunking import PolicyFormatError, parse_policy

DOC = "# Leave Policy\n\n## 3.2 Annual Leave\n\nStaff get 20 days.\n\n## 3.3 Sick Leave\n\nStaff get 10 days.\n"


def test_parses_title_and_numbered_sections():
    d = parse_policy(DOC, "leave.md")
    assert d.title == "Leave Policy"
    assert d.filename == "leave.md"
    assert [(s.section_number, s.section_title) for s in d.sections] == [
        ("3.2", "Annual Leave"),
        ("3.3", "Sick Leave"),
    ]
    assert d.sections[0].content == "Staff get 20 days."


def test_splits_long_section_at_paragraph_breaks():
    body = "\n\n".join(["x" * 900, "y" * 900, "z" * 900])
    d = parse_policy(f"# T\n\n## 1.1 Long\n\n{body}\n", "t.md")
    assert [s.part for s in d.sections] == [1, 2, 3]
    assert {s.section_number for s in d.sections} == {"1.1"}
    assert all(len(s.content) <= 1500 for s in d.sections)


def test_short_paragraphs_are_packed_together():
    body = "\n\n".join(["a" * 600, "b" * 600, "c" * 600])
    d = parse_policy(f"# T\n\n## 1.1 Long\n\n{body}\n", "t.md")
    assert [s.part for s in d.sections] == [1, 2]
    assert d.sections[0].content == "a" * 600 + "\n\n" + "b" * 600


def test_single_oversized_paragraph_stays_whole():
    d = parse_policy(f"# T\n\n## 1.1 Long\n\n{'x' * 2000}\n", "t.md")
    assert len(d.sections) == 1


@pytest.mark.parametrize(
    "text",
    [
        "no headings",
        "# Title only\n",
        "# T\n\n## Unnumbered\n\nBody\n",
        "## 1.1 No title\n\nBody\n",
        "# T\n\n## 1.1 Empty\n\n",
    ],
)
def test_bad_files_raise_error_naming_file(text):
    with pytest.raises(PolicyFormatError, match="bad.md"):
        parse_policy(text, "bad.md")


def test_text_before_first_section_is_an_error_not_silently_dropped():
    with pytest.raises(PolicyFormatError, match="bad.md"):
        parse_policy("# T\n\nIntro text.\n\n## 1.1 A\n\nBody\n", "bad.md")
