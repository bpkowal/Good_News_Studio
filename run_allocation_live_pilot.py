"""Run held-out allocation cases through production world grounding only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("evals/heldout_allocation_mutations_live.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("eval_outputs/allocation-live-pilot"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--escalate-world-model", action="store_true")
    parser.add_argument("--pairwise-relation-audit", action="store_true")
    parser.add_argument("--pairwise-audit-max-pairs", type=int, default=12)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = list(manifest.get("cases") or [])
    selected = cases[max(0, args.start):max(0, args.start) + max(0, args.count)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for index, case in enumerate(selected, start=max(0, args.start)):
        case_id = str(case["id"])
        actions = [str(item) for item in case.get("actions") or []]
        command = [
            sys.executable, "parliament.py",
            "--mode", "workspace",
            "--question", str(case["prompt"]),
            "--backend", "openai",
            "--actions", *actions,
            "--accept-actions",
            "--stop-after-world",
            "--skip-original-agents",
            "--no-rag",
            "--no-framing-cache",
            "--max-cycles", "1",
            "--performance-output",
            str(args.output_dir / f"{case_id}.performance.json"),
        ]
        command.append(
            "--escalate-world-model"
            if args.escalate_world_model else "--no-world-escalation"
        )
        if args.pairwise_relation_audit:
            command.extend([
                "--pairwise-relation-audit",
                "--pairwise-audit-max-pairs",
                str(max(0, args.pairwise_audit_max_pairs)),
            ])
        started = time.monotonic()
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = round(time.monotonic() - started, 3)
        output = completed.stdout or ""
        (args.output_dir / f"{case_id}.log").write_text(output)
        committed = "status: COMMITTED" in output
        rejected = "status: REJECTED" in output or "World grounding rejected" in output
        result = {
            "index": index,
            "id": case_id,
            "oracle": case.get("oracle"),
            "exit_code": completed.returncode,
            "elapsed_seconds": elapsed,
            "committed": committed,
            "rejected": rejected,
            "log": f"{case_id}.log",
            "performance": f"{case_id}.performance.json",
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps(
            {
                "manifest": str(args.manifest),
                "escalated": bool(args.escalate_world_model),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        ))
    return 0 if all(item["committed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
