#!/usr/bin/env python3
"""
Condense Markdown quote files into one file per philosophical school.

This version matches your repo layout where each school is its own top-level
directory (e.g., utilitarian_corpus, virtue_ethics_corpus, Nozick_corpus, etc.)
rather than having a single "quotes/" root.

Examples:
  # Auto-detect *all* *_corpus folders under the repo root and write outputs
  # into quotes_combined/ (created if it doesn't exist):
  python condense_quotes.py

  # Provide specific source folders explicitly:
  python condense_quotes.py --sources utilitarian_corpus virtue_ethics_corpus

  # Change output directory and include only approved files:
  python condense_quotes.py --dest quotes_combined --approved-only

  # Require a specific tag in YAML frontmatter:
  python condense_quotes.py --require-tag canonical

Features:
- Deterministic order (sorted file names).
- Optional filtering to only include `status: approved` (YAML frontmatter).
- Optional tag filtering (require presence of a tag in YAML `tags:` list).
- Preserves original Markdown content verbatim.
- Inserts lightweight section headers per source file.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import sys
import re

try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # We’ll gracefully handle lack of PyYAML


# -------------------------- Frontmatter helpers -------------------------- #
def parse_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (metadata, body). If no/invalid frontmatter, metadata is None."""
    if not text.startswith("---\n"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    raw_meta = parts[1]
    body = parts[2].lstrip("\n")
    if yaml is None:
        # If PyYAML not installed, just return raw meta as a string
        return {"_raw_frontmatter": raw_meta.strip()}, body
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except Exception:
        # If malformed, treat as no metadata and keep original text as body
        return None, text
    return meta, body


def should_include(
    meta: Optional[Dict[str, Any]],
    approved_only: bool,
    required_tag: Optional[str],
) -> bool:
    if approved_only:
        if not meta or meta.get("status") != "approved":
            return False
    if required_tag:
        tags = []
        if meta:
            tags = meta.get("tags", [])
        if not isinstance(tags, list) or required_tag not in tags:
            return False
    return True


# --------------------------- Core condensation --------------------------- #
def condense_school(
    school_dir: Path,
    out_file: Path,
    approved_only: bool,
    required_tag: Optional[str],
    include_file_headers: bool,
) -> int:
    md_files: List[Path] = sorted(school_dir.glob("*.md"))
    count_included = 0

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as out:
        # File header
        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        out.write(f"# {school_dir.name}\n\n")
        out.write(f"_Combined on {ts}_\n\n")

        for md in md_files:
            text = md.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)

            if not should_include(meta, approved_only, required_tag):
                continue

            if include_file_headers:
                title = None
                if isinstance(meta, dict):
                    title = meta.get("title")
                header = (title or md.stem).strip()
                out.write(f"\n---\n\n## {header}\n\n")

            out.write(body.rstrip() + "\n\n")
            count_included += 1

    return count_included


# ----------------------------- Auto-detection ---------------------------- #
CORPUS_DIR_PATTERN = re.compile(r".*?_corpus$", re.IGNORECASE)

def autodetect_corpus_dirs(root: Path) -> List[Path]:
    """
    Find top-level subdirectories under `root` whose names end with `_corpus`
    (case-insensitive), e.g., 'utilitarian_corpus', 'virtue_ethics_corpus',
    'Nozick_corpus', etc. Only include dirs that contain at least one .md file.
    """
    dirs = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        if CORPUS_DIR_PATTERN.match(d.name):
            has_md = any(d.glob("*.md"))
            if has_md:
                dirs.append(d)
    return dirs


# --------------------------------- CLI ---------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Condense Markdown quotes per school (auto-detects *_corpus folders).")
    p.add_argument(
        "--sources",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit list of corpus folders to condense. If omitted, auto-detects *_corpus dirs under --root.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root to search for *_corpus directories when --sources is omitted (default: current dir).",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("quotes_combined"),
        help="Output folder for combined files (default: quotes_combined/).",
    )
    p.add_argument(
        "--approved-only",
        action="store_true",
        help="Include only files with YAML frontmatter `status: approved`.",
    )
    p.add_argument(
        "--require-tag",
        type=str,
        default=None,
        help="Only include files whose YAML `tags:` contains this tag.",
    )
    p.add_argument(
        "--no-file-headers",
        action="store_true",
        help="Do not insert per-file section headers inside combined output.",
    )
    args = p.parse_args()

    # Resolve sources
    if args.sources:
        sources = [s.resolve() for s in args.sources]
    else:
        sources = autodetect_corpus_dirs(args.root.resolve())

    if not sources:
        print("⚠️ No corpus folders found. Provide --sources or ensure *_corpus folders exist under --root.", file=sys.stderr)
        return 1

    total_files = 0
    for school_dir in sources:
        if not school_dir.exists() or not school_dir.is_dir():
            print(f"⚠️ Skipping non-directory: {school_dir}", file=sys.stderr)
            continue
        out_file = args.dest / f"{school_dir.name}.md"
        included = condense_school(
            school_dir=school_dir,
            out_file=out_file,
            approved_only=args.approved_only,
            required_tag=args.require_tag,
            include_file_headers=not args.no_file_headers,
        )
        print(f"✅ {school_dir.name}: wrote {included} file(s) → {out_file}")
        total_files += included

    if total_files == 0:
        print("⚠️ No files were included. Check filters or folder contents.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())