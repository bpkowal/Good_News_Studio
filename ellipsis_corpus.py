"""Load the English Hoosier Ellipsis Corpus for the presence detector.

The corpus is the Indiana University NLP Lab release in resources/thec_eng
(Apache-2.0). A paired example contributes two rows that share a group id:
the gapped sentence is present, and the spelled-out sentence is absent.
Context lines marked B: and A: are attached to both rows. Group ids are
assigned before any split, so a holdout example cannot leak its restoration
into training.

This loader does not score sentences and does not propose missing words.
"""

import csv
import hashlib
from pathlib import Path


SPLIT_SALT = "thec-eng-detector-v1"
DEFAULT_ROOT = Path(__file__).resolve().parent / "resources" / "thec_eng" / "data"


def surface(sentence):
    """Drop the corpus gap marks. Real text does not contain them."""
    text = sentence.replace("___", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = " ".join(text.split())
    for mark in (" .", " ,", " ?", " !", " ;", " :"):
        text = text.replace(mark, mark.strip())
    return text.strip()


def split_of(group_id, salt=SPLIT_SALT):
    """Frozen 70/15/15 split. The text of the example is not an input."""
    digest = hashlib.sha256(f"{salt}\n{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "dev"
    return "test"


def _context_text(prefix, line):
    body = line[2:].strip() if line.startswith(prefix) else line.strip()
    return surface(body)


def parse_paired_text(text, source):
    """Parse THEC blocks of gapped sentence, ----, spelled-out sentence."""
    lines = text.splitlines()
    index = 0
    cursor = 0
    found = []

    def blank_or_comment(line):
        stripped = line.strip()
        return not stripped or stripped.startswith("#")

    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if blank_or_comment(stripped):
            cursor += 1
            continue
        before, after = [], []
        while stripped.startswith(("B:", "A:")):
            target = before if stripped.startswith("B:") else after
            target.append(_context_text(stripped[:2], stripped))
            cursor += 1
            while cursor < len(lines) and blank_or_comment(lines[cursor]):
                cursor += 1
            if cursor >= len(lines):
                return found
            stripped = lines[cursor].strip()
        elided_parts = [stripped]
        cursor += 1
        while cursor < len(lines) and lines[cursor].strip() != "----":
            piece = lines[cursor].strip()
            cursor += 1
            if not piece or piece.startswith("#"):
                continue
            if piece.startswith("B:"):
                before.append(_context_text("B:", piece))
            elif piece.startswith("A:"):
                after.append(_context_text("A:", piece))
            else:
                elided_parts.append(piece)
        if cursor >= len(lines) or lines[cursor].strip() != "----":
            continue
        cursor += 1
        full_parts = []
        while cursor < len(lines):
            piece = lines[cursor].strip()
            if not piece:
                cursor += 1
                if full_parts:
                    break
                continue
            if piece.startswith("#"):
                cursor += 1
                if full_parts:
                    break
                continue
            if piece == "----":
                break
            if piece.startswith(("B:", "A:")) and full_parts:
                target = before if piece.startswith("B:") else after
                target.append(_context_text(piece[:2], piece))
                cursor += 1
                continue
            if piece.startswith(("B:", "A:")) and not full_parts:
                target = before if piece.startswith("B:") else after
                target.append(_context_text(piece[:2], piece))
                cursor += 1
                continue
            if full_parts:
                break
            full_parts.append(piece)
            cursor += 1
        elided = surface(" ".join(elided_parts))
        full = surface(" ".join(full_parts))
        if not elided:
            continue
        group_id = f"{source}:{index}"
        index += 1
        context = " ".join(part for part in before + after if part)
        found.append(dict(group_id=group_id, text=_join(context, elided), label=1, source=source))
        if full and full != elided:
            found.append(dict(group_id=group_id, text=_join(context, full), label=0, source=source))
    return found


def _join(context, sentence):
    return f"{context} {sentence}".strip() if context else sentence


def _load_ellie(path):
    found = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sentence = surface(row.get("Sentence") or "")
            if not sentence:
                continue
            group = row.get("ID") or str(len(found))
            found.append(dict(
                group_id=f"ELLie.csv:{group}", text=sentence, label=1, source="ELLie.csv"))
    return found


def _load_distractors(path, source):
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        found.append(dict(
            group_id=f"{source}:{number}", text=surface(stripped), label=0, source=source))
    return found


def load_examples(root=None):
    """Return rows with group_id, text, label, source. label 1 means a gap is present."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".csv", ".tct"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        relative = str(path.relative_to(root))
        if path.name == "ELLie.csv":
            rows.extend(_load_ellie(path))
            continue
        if "distractors" in path.parts:
            rows.extend(_load_distractors(path, relative))
            continue
        rows.extend(parse_paired_text(path.read_text(encoding="utf-8"), relative))
    return rows


def rows_for_split(rows, split, salt=SPLIT_SALT):
    return [row for row in rows if split_of(row["group_id"], salt) == split]
