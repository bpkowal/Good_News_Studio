# Matched comparisons

The first pilot compares three continuations of the same five saved opening responses:

- Independent: stop after the opening assessments.
- Legacy: the native Parliament engine runs one further cycle with peer arguments available to all five frameworks.
- Targeted: independent assessments receive at most two localized objections, routed to their originating frameworks.

All arms use the same admitted two-action world, exact source text, construction advisory, framework commitment excerpts and opening model responses. The follow-ups use o3 with a 3,072-token output limit. Each framework may use at most one follow-up call; no separate repair call is allowed in this comparison. Each arm has a maximum allowance of ten logical calls including the five openings. Actual usage differs: targeted review does not spend unused calls. Provider retries are not included in logical call counts.

The deterministic deontological display-map closure is applied uniformly. It renders `fm` from unchanged typed `dp` fields and retains the raw response and transformation provenance. No semantic classification is repaired by this transformation.

The legacy arm uses native ledger validation and deliberation, with synthesis, planning, auxiliary audits and cycle extensions disabled. This is a bounded loop comparison, not a benchmark of the entire production pipeline. The new arms make no collective judgment.

Run from the workspace using a fresh output directory:

```bash
.venv/bin/python run_logic_puzzles.py --matched-comparison \
  --trace diagnostics/logic_puzzles_independent_live/framework_generation.json \
  --output-dir diagnostics/logic_puzzles_matched_next
```

The runner uses the existing native Parliament checkout and interpreter by default. Override `--parliament-root` and `--parliament-python` if their locations change. The native worker loads the existing API environment without recording credentials.

Every arm emits its graph, native records and prompts/responses. `matched_comparison.json` records input fingerprints, admission, claim counts, missing framework ledgers, native calibration errors, framework warnings, internal conflicts, recommendations and calls. `matched_comparison.md` links all three graphs. Rejected targeted revisions leave the original claims operative and preserve the attempted revision separately.

Interpret admission and semantic quality separately. A drop in errors caused by losing a ledger is not success. A reported conflict can represent appropriate uncertainty. Agreement is descriptive, not an objective. Native diagnostics do not independently establish semantic correctness. A single scenario with one sampled continuation cannot establish a statistical winner.

The ordinary loop uses cumulative continuity validation, whereas targeted review revalidates in a fresh owner workspace and then reconciles its revision. The current experiment measures these whole paths; it does not isolate prompting from acceptance policy. The [first live findings](diagnostics/logic_puzzles_matched_live/RESULT.md) show why this distinction matters: both paths proposed the same means-relation correction, but only one made it operative, and the accepted revision lost a promise reference from its competing-duty field.

The [identical-response acceptance replay](diagnostics/logic_puzzles_acceptance_replay/RESULT.md) isolates the procedural difference without new model calls. Replay it with:

```bash
/tmp/parliament-smoke-env/bin/python logic_puzzles_acceptance.py \
  --output-dir diagnostics/logic_puzzles_acceptance_next
```

The replay never creates a live backend. It compares fresh targeted acceptance with recurrent native acceptance under an ordinary agenda, an assigned open-mode challenge, and an assigned proposal-review challenge. Full field changes, failed repair attempts, audit inputs and every graph are outputs. It does not change production validation rules.

Targeted review now uses the recurrent native route by default: replay the saved prior response into a native ledger, then request the assigned revision in `PROPOSAL_REVIEW`. Prior replay is recorded separately from new adapter calls. Native rejected updates retain the original assessment and are reported as `RETAINED_AFTER_NATIVE_REJECTION`. Reports expose all changed record and candidate fields. [Saved-response regression results](diagnostics/logic_puzzles_native_review_tests/RESULT.md) cover one accepted correction and one rejected revision.

The acceptance comparison retains the earlier fresh-workspace output as a historical control; it does not invoke that old route as the production default.

Rejected deontological bundles can now receive one deterministic native revalidation of an addressed means-relation correction. The rejected bundle stays visible as `DISPUTED_NONOPERATIVE`; the patch retains prior duty fields and requires native acceptance before becoming operative. Harm classification and duty derivation remain outside this bounded isolation step. [Pilot results and graph](diagnostics/logic_puzzles_partial_revision/RESULT.md) show the correction accepted while prior verdicts and promise references survive. No extra API call is used for isolation.

The next bounded step also isolates an explicitly proposed harm downgrade to `UNRESOLVED`, provided the proposal explicitly supplies `CONTESTED` or `UNKNOWN` resolution. A resolution change is recorded as a dependent companion to the harm patch; it does not create another review question. Affirmative harm replacements and resolved adjudications cannot use this path. Native review still decides acceptance.

Uncertainty replay renders verdict, relation, governing norm, priority basis and resolution from the **committed native ledger**, rather than the stronger original raw model response. Otherwise removing a calibration warning can accidentally resurrect a prohibition that native calibration previously downgraded. Missing committed fields prevent isolation instead of triggering a guessed reconstruction. Full duty revisions remain disputed and nonoperative. [Uncertainty replay results](diagnostics/logic_puzzles_uncertainty_revision/RESULT.md) preserve the conflicted verdict, promise reference and unresolved priorities while accepting the classification corrections. Validator-warning removal is reported separately from unresolved semantics; no new API calls were made.
