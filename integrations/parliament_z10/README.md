# Z10 advisory bridge: first Parliament integration

The branch-choice increment adds `branch_choice_context.patch` after the existing
locked patches. It transports an optional, world-hash-bound interpretation of
intervention, omitted intervention and physical continuation through frozen replay
to the native specialist's primary, repair, recurrent and contingency prompts.
The independent-framework and RelEnt entrypoints share this specialist transport.
It does not change the world schema or make omission morally permissible.
Reconstruct through `pipeline.lock.json`; do not apply the patch twice to an
already patched checkout. See the [bounded replay and live results](../../diagnostics/trolley_choice_pipeline_20261014/RESULT.md).

`negation_boundary_validation.patch` follows that patch in the runtime lock.
It recognizes explicit negation and refraining/abstaining descriptions of paired
choices, and distinguishes hypothetical reversal prose from current-world numeric
assertions. Unsupported operative boundary parameters remain uncommitted.
See the [saved-response validation comparison](../../diagnostics/negation_boundary_validation_20261015/RESULT.md).

Z10 remains frozen. The bridge gives Parliament its complete candidate package
alongside the original scenario, plus readable reconstruction role frames.
It does not replace the input with a guessed corrected sentence.

For `Maria can save the child, but not the dog.`, the additional reading identifies
the omitted `save` predicate, subject `Maria`, and object `the dog`. Its polarity
remains unresolved, and the original question about `NOT MODAL(P)` versus
`MODAL(NOT P)` stays in the packet. Parliament can consider inability to save the
dog versus ability to refrain, without this adapter choosing one. Named-role
alternatives remain separate propositions. Unimplemented verb-phrase ellipsis,
gapping, and sluicing reconstructions remain missing-content questions.

## Files and boundary

Current runtime setup is pinned by [`pipeline.lock.json`](../../pipeline.lock.json)
and [`PARLIAMENT_RUNTIME.md`](../../PARLIAMENT_RUNTIME.md). The demo/blueprint
runners now default to `.runtime/parliament` and `.runtime/venv/bin/python`.
The `/tmp` paths below describe the historical experiments. Use the lock's ordered
patch list (including the canonical discourse patch) for new reconstruction.

- `parliament_z10_bridge.py` in the project root validates schema 0.4 packages,
  matches exact source text, preserves every package field, and renders advisory
  instructions and reconstruction notes. No candidates are selected by default.
- `grounding_hook.patch` is the full diff of the live Parliament checkout against
  `614af0c1601dd7169842f130f1c26a5dd0b30564`. Apply it once to that commit.
  It adds the optional `parser_evidence_packet` argument to
  Parliament's `ground_actions_in_scenario`. It requires `EVIDENCE_ONLY`, validates
  before backend calls, and includes the packet in primary and repair grounding
  prompts. Existing world admission checks still run. Auxiliary calls do not
  receive this packet. The result records advisory model-call attempts;
  deterministic/early-return paths can bypass the model and have no such record.
  The patch also begins a constrained draft compiler for exclusive allocations:
  when a nonreceipt row has exactly one source-grounded same-action transfer,
  Parliament deterministically supplies its quantity, parent edge, and derivation
  type. Ambiguous transfer parents remain untouched. A transfer to the affected
  person may directly parent a stipulated welfare outcome; mentioning the
  transferred resource alone no longer licenses an invented process-state node.
  Complement recompilation reads magnitude from the indivisible-resource
  proposition, preventing a rival recipient headcount in retained constraint
  provenance from leaking into the nonrecipient effect during frozen replay.
- `run_parliament_z10.py` runs that grounding API using either a response fixture
  or an explicitly requested live OpenAI backend. This first integration stops at
  world grounding: it does not wire the interactive launcher, its framing cache,
  or downstream ethical deliberation to Z10.

The separate packet version is `z10-advisory/1`; this is not Parliament's native
Stage-1 packet schema. IDs retain their Z10 package scope. Parser evidence IDs
must not be used as Parliament clause citations. Source hash checks detect
mismatches; they are not authenticity or semantic-support checks. Template markers
inside data strings are escaped without changing JSON values.

## Use

The hook was applied and tested in `/tmp/parliament-smoke-614af0c`, based on
`bpkowal/Good_News_Studio`, branch `relent-framework`, commit
`614af0c1601dd7169842f130f1c26a5dd0b30564`. The patch is saved here for persistence.
On another checkout at that commit, apply it with `git apply` using its absolute
path. Do not apply it again to the already patched temporary checkout.

From this parser project's root, prepare evidence using its frozen environment:

```sh
.venv/bin/python parliament_z10_bridge.py \
  --text-file integrations/parliament_z10/example.txt \
  --output diagnostics/z10_parliament_example_packet.json \
  --notes-output diagnostics/z10_parliament_example_notes.txt
```

Run the offline grounding probe in Parliament's isolated Python 3.11 environment:

```sh
/tmp/parliament-smoke-env/bin/python run_parliament_z10.py \
  --parliament-root /tmp/parliament-smoke-614af0c \
  --packet diagnostics/z10_parliament_example_packet.json \
  --actions 'save the child' 'save the dog' \
  --response-file integrations/parliament_z10/rejected_response.json \
  --output diagnostics/z10_parliament_grounding_smoke.json
```

Expected outcome: `REJECTED`, with two advisory model-call attempts. The fixture
deliberately returns `{}` so the test proves prompt delivery and continued
rejection of invalid output; it does not measure the model's understanding.
The runner exits successfully when the probe completes; inspect `grounding.status`
for admission. It saves original evidence, actions, results, and fixture prompts.
For a live experiment, replace `--response-file ...` with `--openai-model MODEL`
and use an environment with Parliament's provider dependencies and credentials.
No live calls were made during this implementation.

## Validation and remaining work

### Framework presentation and duty roles (October 9)

`framework_presentation.patch` is additive to the integrated native checkout used
for the October 8 trolley runs. It fixes the deontological formatter to name the
`duty_bearer` on both sides of a duty conflict, with `protected_party` and
`competing_protected_party` remaining beneficiaries of those obligations. It also
removes character slicing from the main rationale, application, action comparison,
decision rule, and reversal fields, and short schema limits on explanation fields.
The overall model response budget and native admission criteria are unchanged.
Apply with `git apply --check` first on a matching checkout; do not apply twice.
The patch was round-trip checked against its four saved pre-change files.

`parliament_framework_presentation.py` retains raw model explanations alongside
native decisions. Saved-report export can refresh legacy prose only from that
report's matching source, world, preference, and ledger status. This prevents an
older export from restoring the misleading “five workers' duty to rescue” wording.
It does not turn a contested agent interpretation into an accepted moral judgment.

The October 9 replay and validation record are in
`diagnostics/trolley_framework_views_20261009/RESULT.md`. No API calls were made
for that presentation replay. The latest run passed 52 checks; two native
integration checks could not import the temporary checkout because
`global_workspace/care_ledger.py` was missing. The earlier 53-check run, before
that checkout became incomplete, passed. These are separate verification results.

### Recognition projection increment

The October 8 trolley scene increment adds `scene_recognition.patch` on top of
`recognition_gap.patch`. It is applied to `/tmp/parliament-smoke-614af0c`.
The canonical module remains `parliament_recognition.py`; copy it to
`global_workspace/recognition_bridge.py` when updating that integrated checkout,
or apply the additive patch after `git apply --check` on a matching older copy.
ASSERTED world propositions become ledger records only when their entire statement
is copied from the source. Modal wording survives verbatim: “cannot stop” never
licenses a stopping event. These records share the admitted world with all agents.

`recognition_gap.patch` is additive to the current grounding/discourse integration.
It is already applied to `/tmp/parliament-smoke-614af0c`; do not apply it twice.
It adds `global_workspace/recognition_bridge.py` (the workspace source is
`parliament_recognition.py`) and updates four native consumers. Apply it to a
checkout with the existing integration using `git apply --check` followed by
`git apply`. A round-trip check against the pre-increment native files reproduces
all five tested files exactly.

The adapter preserves explicit local probabilities as decimal masses, including
sub-one-percent values. It never changes outcome modality or invents magnitude,
duration, reversibility or utilities. Unsupported bounds, conflicting values and
overall/local probability combinations stay unknown.

Restored admitted worlds now project separate source quantity, exclusivity and
conditional-rule recognition records. These are exact-bind ledger records, not
new occurrence nodes. Quantity cards require an admitted resource party and the
native allocation-complement proof; exclusivity needs its own explicit source
clause. Rule cards require admitted same-action endpoints and a supported
source-stipulated link. Their original conditional/modal wording survives.

Supported components of strengthened agent claims are recorded without accepting
the entire compound. The quantity fragment in “Only one dose of the life-saving
medicine is available (so only one patient can be aided)” can be identified while
“only”, “life-saving” and the stronger conclusion remain unestablished. This
increment does not provide unrestricted paraphrase or inference matching.

Run the no-API saved-input recognition replay with:

```bash
/tmp/parliament-smoke-env/bin/python replay_parliament_recognition.py
```

See `diagnostics/recognition_gap_saved_replay/RESULT.md` for results and limits.

The next bounded increment recognizes stock directly from an admitted RESOURCE
party's own provenance. It no longer requires an exclusive-allocation complement
or transfer effect. Evidence must still match the original text and the resource
identity; stock quantity does not imply availability, benefit or exclusivity.

The medicine duty-ledger failure was a generated classification contradiction:
`dt=IMPERFECT` with `pb=PERFECT_DUTY` for both actions. The deontological prompt now
explicitly explains the consistency requirement and the unresolved-priority path.
Validation still rejects the contradiction; no norm or verdict is automatically
rewritten. A native synthetic probe verifies that a complete `CONFLICTED` ledger
with unresolved priority remains representable. This is not a live-model success
claim. See `diagnostics/recognition_gap_resource_duty/RESULT.md`.

The runner now writes `admitted_graph_comparison.json` and `.md` after admission.
Semantic equivalence includes participant identity/kind, quantities, outcomes,
modality, conditions, and causal/temporal links; identical action labels alone
cannot collapse different worlds. Blueprint names, record IDs and provenance
wording do not determine equivalence.

When relative support and equivalence leave a beam unresolved, the runner makes
one targeted review call over at most three distinct admitted interpretations.
`selection_review.json` and `.md` retain the request, response, source citations,
remaining questions and missing-content reports. A successful review selects an
existing immutable admitted proposal. It cannot edit world atoms or promote
hypothesis overlays. Other candidate graphs remain visible. This is attributed
model interpretation, not proof of semantic correctness or a normative verdict.
An abstention, service failure or incomplete/false citation keeps the beam open.

The trolley admission regression now checks both paths: its supported core must
pass, while its AV3/AV6 overlay must remain visible and fail when submitted as
established effects. See `diagnostics/bounded_graph_selection_20261006/RESULT.md`
for regression results and the live review's remaining limitation.

Nine bridge tests passed, including the actual patched Parliament function with
a fake backend: initial and repair delivery, baseline isolation, invalid-input
rejection before model calls, role alternatives, conditions, modal scope, lossless
transport, source immutability, and JSON/template escaping.
The saved medicine first draft and a cleaner explicit either/or variant both now
commit without another model call. Their nonrecipient rows inherit `one`, point to
the selected recipient transfer, and are typed
`EXCLUSIVE_ALLOCATION_COMPLEMENT`. A deliberately ambiguous two-parent draft is
left unchanged. Existing process-intermediate tests continue to reject shortcuts
when a separate process bearer is actually required.
The held-out antivenom variant changes the actor, recipients, transfer verb,
resource, container word, and outcome. Its first replay exposed a lexical
`dose`/`vial` dependency. The compiler now uses typed exclusive-allocation evidence
and source-bound singular resource structure; it commits the antivenom replay while
preserving `one vial`.
Two later live `o3` runs are recorded in
`diagnostics/PARLIAMENT_Z10_LIVE_ALLOCATION_TESTS.md`. Antivenom committed after a
repair pass; an epinephrine-injector variant remained rejected after its repair
added allocation branches without their complete factual provenance. These runs
show that topology construction generalizes, while evidence and quantity closure
still need to become deterministic compiler responsibilities.
Z10's 13 focused tests passed; dataset integrity and all frozen baseline hashes
passed. Parliament's same 71-test regression sample remained at 64 passes,
six existing compiler errors, and one existing presentation assertion failure;
the failing test identities are unchanged. See
`diagnostics/parliament_medicine_controller_regression.log` and the prior smoke
report.

```sh
PARLIAMENT_SMOKE_ROOT=/tmp/parliament-smoke-614af0c \
PARLIAMENT_SMOKE_PYTHON=/tmp/parliament-smoke-env/bin/python \
.venv/bin/python -m unittest test_parliament_z10_bridge -v
```

The advisory bridge establishes transport. The constrained compiler separately
provides the first deterministic semantic construction step. Next, compare raw-source
grounding with advisory grounding on reviewed development gold, keeping model,
actions, and generation settings fixed. Measure ellipsis roles, operator scope,
unsupported additions, abstention, and admission separately. Preserve the original
scenario in both arms. Full packages increase prompt size; no silent truncation or
lossy compacting has been added. The held-out set has not been used for inference.

A fluent rewritten version is a later, explicitly labelled interpretation view:
generate it only from a validated selected bundle, retain blockers and unresolved
readings, and map every inserted phrase to its reconstruction evidence. A rewrite
must never become replacement source evidence for Parliament's admission checks.
# Argument-quality fixes (bounded increment)

`argument_quality_fixes.patch` is an incremental native patch against the tested
smoke checkout after recognition integration. Its apply-and-roundtrip check
reproduces all four modified native files exactly. It leaves the Parliament
world schema and Care/Rawlsian validators unchanged.

- Explicit single-person survival statements carry a separate count projection.
  Arithmetic validation can use it for survivor endpoints; welfare magnitude and
  duration remain unknown. Repeated patient identities cannot be counted twice.
- Standalone affirmative past-tense promises between admitted named people become
  exact source evidence for commitment occurrence. Conditional, negated, quoted,
  and reported promises are not promoted by this bounded recognizer. Promise
  performance and normative priority remain unestablished.
- A schema-valid duty classification/priority mismatch can use one existing
  repair allowance. It must retain the original duty type. Exhausted budgets and
  still-inconsistent repairs remain visible failures.

Workspace review now retains all open conflicts, permits one residual round for
an unreviewed origin (at most two issues), and reports attempted/unattempted
coverage and actual calls. The matched debate comparison keeps its original
single-round budget explicitly; it must not silently acquire extra review calls.
Native warning retirement requires an accepted relevant change, surviving record
identities and fresh revalidation. A resolved model answer or removed records
cannot retire an objection. Care isolation retains native prior classifications
and proposed explanations; status-only resolutions are ineligible.

Ranking records now distinguish sole non-fallback support from competitive
separation. Fallback graphs remain inspectable as noncompeting members, and
same-family semantic alternatives still undergo separation checks.

The first-pass argument-quality pilot retains one call per cell for comparability
with the original run. The two-call duty repair and residual review paths have
separate regression coverage. `argument_work_record.json` preserves model-authored
arguments alongside native acceptance status, including rejected arguments.

### Full OpenAI RelEnt/RAG profile

See [the runtime guide](../../PARLIAMENT_RUNTIME.md) for the separate
`full_relent` environment, exact dependencies, pinned offline MiniLM cache, and
five-framework retrieval smoke check. It shares the locked native source but
never installs retrieval dependencies into the compact environment.

### Candidate maxims

The locked `candidate_maxims.patch` adds attributed practical-reasoning records
to deontological assessments and their native projection. See
[Candidate maxims](../../CANDIDATE_MAXIMS.md) for the fields, targeted challenges,
report behavior, compatibility, and regression commands.

The subsequent `causal_means_reasoning.patch` records proposed instrumental belief
and purposive use separately and checks same-action physical means/end references.
The physical-scene producer supplies source-supported `CAUSES` links for the
intervention while preserving omission dependencies. Historical frozen worlds
are not migrated in place.
