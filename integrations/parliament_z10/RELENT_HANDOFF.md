# Full RelEnt handoff verification

Verified on 2026-10-03. The remote `relent-framework` head is
`614af0c1601dd7169842f130f1c26a5dd0b30564` (September 22). The integration
checkout was already at that revision; its four locally patched files are
recorded in `diagnostics/trolley_relent_verified_replay/handoff.json`.
This identifies the latest RelEnt branch revision, not a claim that every other
branch has older code.

The earlier runner configuration skipped original agents and used two compact
frameworks, one cycle, and disabled synthesis/planning/audits. Preparation-only
runs did not deliberate at all. Full deliberation is now the default:
five frameworks, original testimony, three cycles, native synthesis/planning/audits.
Corpus RAG remains a selectable option, off by default.

## Live result

Replayed the previously admitted world in
`diagnostics/trolley_omission_up_to_snuff/frozen_world_trace.json` through the
native `global_workspace_pipeline.py` using OpenAI o3.

- Exit status: 0; three cycles completed.
- Original testimony: all five frameworks, no source errors.
- Frozen-world validation: passed; scenario and action identities matched.
- World-generation calls: 0.
- Judgment: `CONTESTED_RECOMMENDATION`, `pulls the lever`.
- Stopping reason: `cycle_budget`; this is not unanimous ethical agreement.

Artifacts: `diagnostics/trolley_relent_verified_replay/handoff.json`,
`deliberation_report.md`, `workspace_scenario_20261003_231711.json`,
`workspace_scenario_20261003_231711_answer.txt`, and
`semantic_preservation_trace.json` in the same directory. The report exposes
original testimony, per-cycle dissent, framework ledgers, and audit findings.
Remaining grounding issues include duplicate worker descriptions and challenges
that treat the lever as a morally affected party. A proposed warning action stayed
a candidate; it did not replace the admitted world.

A separate fresh chooser run in `diagnostics/trolley_relent_full_pipeline`
selected `omission_harm` but failed admission for AV3/AV6 derivation assumptions
and source-polarity changes. Its graph is in `candidate_graphs.md`. The successful
replay does not establish that fresh generation is reliable.

## Run a new scenario

```bash
./run_blueprint_parliament.sh --scenario-file my_scenario.txt
```

The terminal prompts for frameworks, cycles, and corpus RAG. To make those choices
explicit and avoid prompts:

```bash
./run_blueprint_parliament.sh --scenario-file my_scenario.txt \
  --agents utilitarian deontological virtue care rawlsian \
  --max-cycles 3 --no-rag --time-budget 900 \
  --output-dir diagnostics/my_full_relent_run
```

Every attempt produces candidate graphs; admitted worlds additionally produce
world topology and the frozen trace. Completed deliberation produces a native
answer, complete trace, and `parliament/deliberation_report.md`. The manifest
records the actual Parliament revision and distinguishes withholding, admission
failure, and deliberation failure. `--prototype-deliberation` remains an explicit
reduced smoke mode.

Runner verification: 12 unit tests passed; `git diff --check` passed.
