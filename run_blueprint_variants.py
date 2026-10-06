"""Run cloze plus Parliament admission on a battery of ethical variants.

The battery is meant to show where the current blueprint plan constructs an
admitted world, where it correctly withholds, and where it still fails.
It does not run moral deliberation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from blueprint_cloze_chooser import choose_by_cloze, openai_complete
from run_blueprint_parliament import (
    ADMITTED_WORLD_STATUSES,
    DEFAULT_OPENAI_ENV,
    DEFAULT_PARLIAMENT_PYTHON,
    DEFAULT_PARLIAMENT_ROOT,
    _parliament_admission_script,
)


PROJECT_ROOT = Path(__file__).resolve().parent

VARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "alloc_grouped",
        "expect": "admit exclusive_allocation",
        "text": (
            "A clinic has one dose of antiviral. Dr. Rivera must give the antiviral "
            "to either one child or three adult patients, but not both. If Dr. Rivera "
            "gives the antiviral to the child, the child has an 80% chance of survival. "
            "If Dr. Rivera gives the antiviral to the three adult patients, each of "
            "the three adult patients has a 60% chance of survival. The patients who "
            "do not receive the antiviral will die."
        ),
    },
    {
        "id": "alloc_antivenom",
        "expect": "admit exclusive_allocation",
        "text": (
            "Dr. Chen has one vial of antivenom. Imani needs the antivenom. Pavel "
            "needs the antivenom. Dr. Chen can administer the antivenom to either "
            "Imani or Pavel, but not both. If Dr. Chen administers the antivenom to "
            "Imani, Imani will recover. If Dr. Chen administers the antivenom to "
            "Pavel, Pavel will recover."
        ),
    },
    {
        "id": "alloc_two_groups",
        "expect": "admit exclusive_allocation",
        "text": (
            "A ward has one ventilator. Dr. Okonkwo must give the ventilator to "
            "either two children or five adult patients, but not both. If Dr. Okonkwo "
            "gives the ventilator to the two children, each of the two children has "
            "a 70% chance of survival. If Dr. Okonkwo gives the ventilator to the "
            "five adult patients, each of the five adult patients has a 40% chance "
            "of survival. The patients who do not receive the ventilator will die."
        ),
    },
    {
        "id": "alloc_or_only",
        "expect": "withhold exclusivity",
        "text": (
            "A clinic has one dose of medicine. Ada can give the medicine to Ben "
            "or Cara. If Ada gives the medicine to Ben, Ben has a 95% chance of "
            "survival. If Ada gives the medicine to Cara, Cara has a 5% chance of "
            "survival. The patient who does not get the medicine will die."
        ),
    },
    {
        "id": "alloc_three_way",
        "expect": "fail or withhold; two-action only",
        "text": (
            "A clinic has one dose of medicine. Ada must give the medicine to Ana, "
            "Ben, or Cara, but not more than one of them. If Ada gives the medicine "
            "to Ana, Ana will recover. If Ada gives the medicine to Ben, Ben will "
            "recover. If Ada gives the medicine to Cara, Cara will recover."
        ),
    },
    {
        "id": "water_or",
        "expect": "withhold exclusivity",
        "text": (
            "An AI bot must decide whether to devote water to a farm that can "
            "sustain a city's dietary needs or a small town that will lose 100 "
            "people to drowning."
        ),
    },
    {
        "id": "rescue_both",
        "expect": "admit rescue_contrast",
        "text": (
            "Maria can save either the child or the dog, but not both. "
            "If Maria saves the child, the child will live and the dog will drown. "
            "If Maria saves the dog, the dog will live and the child will drown."
        ),
    },
    {
        "id": "rescue_truncated",
        "expect": "withhold or incomplete rescue",
        "text": (
            "A child and a dog are in the water. Maria can save the child, but not "
            "the dog. If Maria saves the child, the child will live."
        ),
    },
    {
        "id": "trolley_omission",
        "expect": "admit omission_harm",
        "text": (
            "Five workers are on the track. One worker is on the side track. "
            "Maria can pull the lever, but not the brake. "
            "If Maria pulls the lever, one worker will die. "
            "If Maria does not pull the lever, five workers will die."
        ),
    },
    {
        "id": "ability_only",
        "expect": "withhold ability_permission",
        "text": "Ada can give medicine to Ben.",
    },
    {
        "id": "promise",
        "expect": "withhold promise_reliance",
        "text": "Ada promised Ben that Ada would deliver the medicine.",
    },
    {
        "id": "disputed_report",
        "expect": "withhold disputed_report",
        "text": "Ada claims the medicine is safe. Ben says the medicine is deadly.",
    },
    {
        "id": "diversion",
        "expect": "admit diversion_redirection",
        "text": (
            "Maria can divert the trolley toward one worker, and one worker will die."
        ),
    },
    {
        "id": "uncertain_risk",
        "expect": "admit uncertain_risk",
        "text": "Ada may administer medicine to Ben, and Ben could die.",
    },
)


def _summarize_cloze(result: dict[str, Any]) -> dict[str, Any]:
    winner = next(
        (row for row in result.get("considered") or []
         if row.get("blueprint_id") == result.get("chosen_blueprint_id")),
        None,
    )
    proposal = (result.get("proposals") or [None])[0]
    world = ((proposal or {}).get("candidate") or {}).get("world_model") or {}
    rejected = [
        {"id": item["id"], "reason": item.get("reason"),
         "completion": item.get("completion")}
        for item in (winner or {}).get("items") or []
        if item.get("verdict") == "rejected"
    ]
    unfilled = [
        item["id"] for item in (winner or {}).get("items") or []
        if item.get("core") and item.get("verdict") != "accepted"
    ]
    return {
        "chosen_blueprint_id": result.get("chosen_blueprint_id"),
        "status": result.get("status"),
        "ranking": result.get("ranking"),
        "exclusivity": (result.get("question") or {}).get("exclusivity"),
        "ethical_question": (result.get("question") or {}).get("ethical_question"),
        "world_withheld": result.get("world_withheld") or [],
        "admission_authorized": bool((proposal or {}).get("admission_authorized")),
        "core_filled": (winner or {}).get("core_filled"),
        "optional_filled": (winner or {}).get("optional_filled"),
        "rejected": rejected,
        "unfilled_core": unfilled,
        "actions": [row.get("intervention") for row in world.get("actions") or []],
        "effect_count": len(world.get("effects") or []),
        "causal_link_count": len(world.get("causal_links") or []),
        "construction_problems": (proposal or {}).get("construction_problems") or [],
        "notes": (proposal or {}).get("notes") or [],
    }


def _admit(proposal: dict[str, Any], scenario: str, actions: list[str],
           parliament_root: Path, parliament_python: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "blueprint_candidate.json"
        trace = Path(directory) / "frozen_world_trace.json"
        helper = Path(directory) / "admit.py"
        candidate.write_text(json.dumps({
            "scenario": scenario,
            "actions": actions,
            "blueprint_result": {"proposals": [proposal]},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        helper.write_text(_parliament_admission_script(), encoding="utf-8")
        completed = subprocess.run(
            [str(parliament_python), str(helper), str(parliament_root),
             str(candidate), str(trace), str(PROJECT_ROOT)],
            text=True, capture_output=True, timeout=120,
        )
    if completed.returncode:
        return {
            "status": "REJECTED",
            "error": (completed.stdout + completed.stderr).strip()[-2000:],
        }
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--openai-env", type=Path, default=DEFAULT_OPENAI_ENV)
    parser.add_argument("--parliament-root", type=Path, default=DEFAULT_PARLIAMENT_ROOT)
    parser.add_argument("--parliament-python", type=Path, default=DEFAULT_PARLIAMENT_PYTHON)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "diagnostics" / "blueprint_variant_battery")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)

    inner = openai_complete(args.model, args.openai_env)

    def complete(messages):
        print(f"  openai call ({len(messages[-1]['content'])} chars)", flush=True)
        return inner(messages)

    print("Warming spaCy / Z10 before the battery...", flush=True)
    from blueprint_cloze_chooser import assess_question
    assess_question("Ada waits.")
    print("Warm.", flush=True)

    selected = [
        row for row in VARIANTS
        if not args.only or row["id"] in args.only
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in selected:
        print(f"\n=== {variant['id']} ===", flush=True)
        cloze = choose_by_cloze(variant["text"], complete)
        cloze_path = args.output_dir / f"{variant['id']}_cloze.json"
        cloze_path.write_text(
            json.dumps(cloze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = _summarize_cloze(cloze)
        summary["id"] = variant["id"]
        summary["expect"] = variant["expect"]
        proposal = (cloze.get("proposals") or [None])[0]
        if proposal and proposal.get("admission_authorized") and proposal.get("candidate"):
            actions = summary["actions"] or [
                row["intervention"]
                for row in proposal["candidate"]["world_model"]["actions"]
            ]
            summary["parliament"] = _admit(
                proposal, variant["text"], actions,
                args.parliament_root.expanduser().resolve(),
                args.parliament_python.expanduser().absolute(),
            )
        else:
            summary["parliament"] = {"status": "NOT_ATTEMPTED"}
        rows.append(summary)
        print(json.dumps({
            "chosen": summary["chosen_blueprint_id"],
            "status": summary["status"],
            "authorized": summary["admission_authorized"],
            "parliament": summary["parliament"].get("status"),
            "actions": summary["actions"],
            "withheld": summary["world_withheld"][:1],
        }, indent=2), flush=True)

    report = {
        "model": args.model,
        "variant_count": len(rows),
        "admitted": [row["id"] for row in rows
                     if row["parliament"].get("status") in ADMITTED_WORLD_STATUSES
                     and row["parliament"].get("world_model_status") in ADMITTED_WORLD_STATUSES],
        "withheld": [row["id"] for row in rows if row["status"] == "WITHHELD"],
        "rejected": [row["id"] for row in rows
                     if row["parliament"].get("status") == "REJECTED"],
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps({
        "admitted": report["admitted"],
        "withheld": report["withheld"],
        "rejected": report["rejected"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
