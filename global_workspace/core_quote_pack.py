"""Authored CORE dialect excerpts for compact specialists. No MiniLM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# One standing identity excerpt per framework. These are specification, not search hits.
CORE_PACK_SOURCES: dict[str, tuple[str, str]] = {
    "utilitarian": ("utilitarian_corpus", "bentham_01.md"),
    "deontological": ("deontological_corpus", "kant_001.md"),
    "virtue": ("virtue_ethics_corpus", "aristotle_001.md"),
    "care": ("care_ethics_corpus", "tronto_003.md"),
    "rawlsian": ("rawlsian_ethics_corpus", "rawls_001.md"),
}


def _first_excerpt(body: str) -> str:
    block = re.search(r"^>\s*(.+)$", body, flags=re.MULTILINE)
    if block:
        return " ".join(block.group(1).strip().strip('"“”').split())
    quoted = re.search(r"[“\"]([^”\"]{40,400})[”\"]", body)
    if quoted:
        return " ".join(quoted.group(1).split())
    return " ".join(body.split())[:280]


def load_core_quote(specialist: str, *, root: Path = ROOT) -> dict[str, str] | None:
    spec = CORE_PACK_SOURCES.get(specialist)
    if spec is None:
        return None
    directory, filename = spec
    path = root / directory / filename
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            raw = yaml.safe_load(parts[1]) or {}
            if isinstance(raw, dict):
                metadata = raw
            body = parts[2]
    excerpt = _first_excerpt(body)
    if not excerpt:
        return None
    return {
        "passage_id": f"{specialist}:{filename}",
        "source_file": filename,
        "author": str(metadata.get("author") or "Unknown"),
        "source": str(metadata.get("source") or metadata.get("title") or filename),
        "text": excerpt[:400],
    }


def format_core_dialect_contract(quote: dict[str, str] | None) -> str:
    if not quote:
        return "CORE DIALECT CONTRACT: NONE."
    return (
        "CORE DIALECT CONTRACT (not scenario evidence): The following approved "
        "CORE excerpt specifies this Parliament's standing interpretation of your "
        "framework. It may inform your ranking rule. It must not add parties, "
        "quantities, outcomes, or probabilities, and it must not set cd=true. "
        "Closed-world scores remain WORLD_ESTABLISHED effects only. "
        f"[{quote['passage_id']}] {quote['author']}, {quote['source']}: "
        f"{quote['text']}"
    )


def core_pack_for_specialists(names: list[str], *, root: Path = ROOT) -> dict[str, dict[str, str]]:
    packed = {}
    for name in names:
        quote = load_core_quote(name, root=root)
        if quote:
            packed[name] = quote
    return packed
