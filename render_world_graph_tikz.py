#!/usr/bin/env python3
"""Render a saved world-model JSON artifact as standalone TikZ diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from global_workspace.world_graph_tikz import (
    compile_world_graph_tikz_bundle,
    write_world_graph_tikz_bundle,
)


def _world_and_clauses(payload: Mapping[str, Any]) -> tuple[dict, list, str]:
    grounding = payload.get("grounding")
    if isinstance(grounding, Mapping):
        rejected = grounding.get("rejected_candidate")
        if isinstance(rejected, Mapping) and isinstance(rejected.get("world_model"), dict):
            return (
                rejected["world_model"],
                list(grounding.get("clauses") or []),
                "RAW_REJECTED_CANDIDATE_PRECOMPILATION",
            )
        if isinstance(grounding.get("world_model"), dict):
            return (
                grounding["world_model"],
                list(grounding.get("clauses") or []),
                "SAVED_GROUNDING_WORLD",
            )
    rejected = payload.get("rejected_candidate")
    if isinstance(rejected, Mapping) and isinstance(rejected.get("world_model"), dict):
        return (
            rejected["world_model"],
            list(payload.get("clauses") or []),
            "RAW_REJECTED_CANDIDATE_PRECOMPILATION",
        )
    if isinstance(payload.get("world_model"), dict):
        return (
            payload["world_model"],
            list(payload.get("clauses") or []),
            "SAVED_GROUNDING_WORLD",
        )
    if isinstance(payload.get("actions"), list) and isinstance(payload.get("effects"), list):
        return dict(payload), [], "RAW_WORLD_MODEL"
    raise ValueError("input does not contain a recognizable world_model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="World model or grounding diagnostic JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile generated .tex files when pdflatex is installed",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    world, clauses, stage = _world_and_clauses(payload)
    destination = args.output_dir or args.input.with_suffix("").with_name(
        f"{args.input.stem}_tikz"
    )
    manifest = write_world_graph_tikz_bundle(
        destination,
        world_model=world,
        clauses=clauses,
        representation_stage=stage,
    )
    result: dict[str, Any] = {"directory": str(destination), **manifest}
    if args.compile:
        result["compilation"] = compile_world_graph_tikz_bundle(destination)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
