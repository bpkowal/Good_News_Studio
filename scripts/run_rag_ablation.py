#!/usr/bin/env python3
"""Replay MiniLM retrieval on the latest committed worlds; optionally launch three-way runs.

Default is an offline overlap report against stored testimony. Pass --execute to
run rag / no-rag / skip-original pipeline arms that reuse each admitted world.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from global_workspace.rag_ablation import (
    ABLATION_ARMS,
    collect_live_summaries,
    compare_ablation_arms,
    find_arm_trace,
    make_embedder,
    pipeline_command,
    replay_retrieval_for_world,
    select_committed_worlds,
    summarize_workspace_trace,
    write_replay_inputs,
)


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "workspace_outputs",
        help="Directory containing workspace_workspace_*.json traces",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where to write the ablation report (default eval_outputs/rag-ablation-<stamp>)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run live pipeline arms after the offline retrieval replay",
    )
    parser.add_argument(
        "--rebuild-live",
        action="store_true",
        help="Rewrite live_ablation.json from traces already in --run-dir",
    )
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--openai-model", default="o3")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=1200.0)
    return parser.parse_args()


def write_live_report(run_dir: Path, live_summaries: dict) -> None:
    comparisons = {
        label: compare_ablation_arms(arms)
        for label, arms in live_summaries.items()
    }
    (run_dir / "live_ablation.json").write_text(
        json.dumps({"summaries": live_summaries, "comparisons": comparisons}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.rebuild_live:
        if args.run_dir is None:
            print("--rebuild-live requires --run-dir", file=sys.stderr)
            return 2
        summaries = collect_live_summaries(args.run_dir)
        write_live_report(args.run_dir, summaries)
        print(f"Rebuilt {args.run_dir / 'live_ablation.json'}")
        return 0
    worlds = select_committed_worlds(args.output_dir, limit=max(1, args.limit))
    if not worlds:
        print(f"No COMMITTED workspace traces in {args.output_dir}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "eval_outputs" / f"rag-ablation-{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Worlds:", ", ".join(f"{world.label} ({world.path.name})" for world in worlds))
    embedder = make_embedder()
    offline = []
    for world in worlds:
        print(f"Replaying retrieval for {world.label}...", flush=True)
        report = replay_retrieval_for_world(world, embedder=embedder)
        offline.append(report)
        for agent, record in report["agents"].items():
            print(
                f"  {agent}: mode={record.get('mode')} core="
                f"{sum(1 for item in record.get('evidence') or [] if item.get('tier') == 'CORE')} "
                f"cited_in_testimony={record.get('testimony_cited_count')} "
                f"cited_in_cycles={record.get('cycle_cited_count')} "
                f"mean_coverage={record.get('mean_testimony_coverage')}",
                flush=True,
            )
    (run_dir / "offline_retrieval.json").write_text(
        json.dumps(offline, indent=2), encoding="utf-8"
    )
    commands = []
    live_summaries: dict[str, dict] = {}
    if args.execute:
        for world in worlds:
            replay_root = run_dir / world.label
            scenario_path = write_replay_inputs(world, replay_root)
            live_summaries[world.label] = {}
            for arm in ABLATION_ARMS:
                arm_dir = replay_root / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                cache_src = replay_root / "last_problem_framing.json"
                if cache_src.exists():
                    (arm_dir / "last_problem_framing.json").write_text(
                        cache_src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                command = pipeline_command(
                    python=sys.executable,
                    pipeline=ROOT / "global_workspace_pipeline.py",
                    scenario_path=scenario_path,
                    output_dir=arm_dir,
                    arm=arm,
                    agents=world.agents,
                    presentation_actions=world.presentation_actions,
                    backend=args.backend,
                    openai_model=args.openai_model,
                    max_cycles=args.max_cycles,
                    time_budget=args.time_budget,
                )
                commands.append({"world": world.label, "arm": arm, "command": command})
                print(f"Executing {world.label}/{arm}...", flush=True)
                completed = subprocess.run(command, cwd=ROOT, check=False)
                trace_path = find_arm_trace(arm_dir)
                summary = {
                    "returncode": completed.returncode,
                    "trace": str(trace_path) if trace_path else "",
                }
                if trace_path:
                    summary.update(
                        summarize_workspace_trace(
                            json.loads(trace_path.read_text(encoding="utf-8")),
                            arm=arm,
                        )
                    )
                live_summaries[world.label][arm] = summary
        write_live_report(run_dir, live_summaries)
    (run_dir / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(f"Wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
