import re
from dataclasses import dataclass

MAX_CHARS = 1500
SECTION_HEADING = re.compile(r"^## (\d+(?:\.\d+)*)\s+(.+)$")


class PolicyFormatError(Exception):
    pass


@dataclass(frozen=True)
class Section:
    section_number: str
    section_title: str
    part: int
    content: str


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    filename: str
    sections: list[Section]


def _pack_paragraphs(body: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    for paragraph in paragraphs:
        if parts and len(parts[-1]) + 2 + len(paragraph) <= MAX_CHARS:
            parts[-1] = f"{parts[-1]}\n\n{paragraph}"
        else:
            parts.append(paragraph)
    return parts


def parse_policy(text: str, filename: str) -> ParsedDocument:
    def fail(reason: str) -> PolicyFormatError:
        return PolicyFormatError(f"{filename}: {reason}")

    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or not lines[0].startswith("# "):
        raise fail("the first line must be a '# Title' heading")
    title = lines[0][2:].strip()

    headings: list[tuple[str, str]] = []
    bodies: list[list[str]] = []
    for line in lines[1:]:
        if line.startswith("## "):
            match = SECTION_HEADING.match(line)
            if not match:
                raise fail(f"section heading has no number: {line!r}")
            headings.append((match.group(1), match.group(2).strip()))
            bodies.append([])
        elif bodies:
            bodies[-1].append(line)
        elif line.strip():
            raise fail(f"text before the first section: {line!r}")

    if not headings:
        raise fail("no numbered sections found")

    sections: list[Section] = []
    for (number, section_title), body in zip(headings, bodies):
        parts = _pack_paragraphs("\n".join(body))
        if not parts:
            raise fail(f"section {number} is empty")
        sections.extend(
            Section(number, section_title, index, content)
            for index, content in enumerate(parts, start=1)
        )
    return ParsedDocument(title, filename, sections)
