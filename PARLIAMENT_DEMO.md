# Inspectable Parliament vertical slice

Live source-to-world-to-framework command (uses the existing configured API):

```bash
.venv/bin/python run_parliament_demo.py --scenario demo_scenarios/medicine_promise.txt --output demo_output/medicine_promise
```

The runner calls the existing blueprint command for Z10, inventory, cloze,
composition, candidate graphs and native admission/collapse. If a unique world
freezes, it calls existing independent native ledger generation for all five
frameworks, which also projects Logic_Puzzles conflicts. This is the framework
ledger path; it does not also run the old sequential collective RelEnt debate or
claim a governing verdict. Add `--review` for existing bounded targeted native
review, `--prepare-only` to stop at admission, or `--baseline` for an additional
same-model direct-analysis call. Baseline comparisons are descriptive questions,
not automatically scored claims about the model.

No-API inspection of the saved promise run:

```bash
.venv/bin/python run_parliament_demo.py --scenario demo_scenarios/medicine_promise.txt --replay diagnostics/argument_quality_pilot_fixes_20261006/competing_promise/standard --output demo_output/promise_replay
```

Use a new output directory for every run. `DEMO_REPORT.md` and JSON distinguish
FROZEN, INTERPRETIVE_BEAM, CONSTRUCTION_WITHHELD, ADMISSION_FAILED,
FRAMEWORK_PARTIAL, COMPLETED and SOFTWARE_FAILED. A partial framework path is
inspectable evidence, not an automatic failed moral conclusion. Recorded preference
and operative participation are separate: rejected ledgers/native abstention cannot
appear FULL. Hinges are copied from recorded reversal fields with record paths and
unverified model attribution; missing hinges stay missing. No report may introduce
a new moral dependency to fill a blank.

Reports embed Mermaid diagrams from the recorded world, using the existing world
renderer. A frozen run shows its world supplied to framework assessment; an
interpretive beam shows each competing admitted graph labeled as not sent to
agents. Candidate trace source must match the report source. Rendering does not
change admission, selection or world records.

Dependencies retain the existing native checkout/interpreter defaults; override
`--parliament-root` and `--parliament-python` when moving to another installation.
Credentials are loaded by existing backends, never copied into demo reports.
Replay reads existing records and checks source identity; it does not rerun native
admission or make a new live reliability claim. Provider failures are logged and
produce a report. Missing interpreter/checkouts remain software failures rather
than epistemic abstentions.

Run the offline report and source-integrity checks:

```bash
.venv/bin/python -m unittest test_parliament_demo
```

Audit all eight frozen sources through Z10 and primitive inventory without calls:

```bash
.venv/bin/python run_parliament_demo_suite.py --output demo_output/source_audit
```

For an explicit live suite subset, add `--execute --cases medicine probability_swap`.
`SUITE_RESULTS.json` compares recorded freeze status with the frozen target. Offline
audits explicitly say `NOT_RUN` for freezing; they cannot count as admission successes.

`SUITE_RESULTS.md` provides a readable status table. A software failure after
freezing fails the suite even when the freeze expectation matched. Topology,
unknowns and argument/hinge quality still require content review.

Failed-construction reports include the saved candidate graphs, per-blueprint
missing slots and admission rejection details. Those graphs are explicitly
unaccepted and were not sent to agents. `--replay` accepts either a pipeline
directory or a demo output directory containing `pipeline/`; it makes no API calls.

See `DEMO_SCENARIO_ENVELOPE.md` for the evidence behind family selection. All eight
fixtures are small exploratory regression targets, not a benchmark success-rate
estimate. Full live suite execution and independent content review are needed
before promising this envelope to an outside researcher.
