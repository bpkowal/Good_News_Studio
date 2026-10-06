# Proposed structured-conflict branch

Status: `Logic_Puzzles` branch started. Read-only claim/conflict projection,
isolated native generation and one bounded targeted review round implemented.
Collective judgment changes and a matched comparison remain planned.

## Current increment

`run_logic_puzzles.py` projects the final cycle's committed native ledger records
from a saved Parliament trace. It retains full records, attributed commitments,
effect references, proposition dependencies, calibration objections, reported
internal conflicts and reversal conditions. Rejected ledger proposals remain in
diagnostics. Its proposed review queue contains at most two issues; no review or
model call is executed. This increment does not change the shared world.

Run from the repository root:

```bash
.venv/bin/python run_logic_puzzles.py \
  --trace diagnostics/promise_specialist_handoff/live/workspace_scenario_20261004_234129.json
.venv/bin/python -m unittest test_logic_puzzles
```

Outputs are `diagnostics/logic_puzzles_projection/conflict_graph.json` and
`conflict_graph.md`. The saved two-action run yields 31 nodes and 23 edges.
Three regression tests check lossless, deterministic projection without source
mutation, rejected records and dangling references, and avoidance of false
conflicts from different preferences alone. This establishes inspectable data;
it does not yet demonstrate improved deliberation.

## Independent framework generation

The runner now launches one fresh, one-cycle native Parliament workspace per
framework against exactly two admitted actions. Each starts from the same frozen
world and seeded propositions, with no peer positions, previous testimony vote,
episodic memory or generated peer hypotheses. It reuses native framework schemas,
CORE dialect excerpts where available, and transactional ledger validation.
Frameworks still retain different normative commitments. No collective judgment
is inferred from the separate native runs.

```bash
.venv/bin/python run_logic_puzzles.py --generate-frameworks \
  --trace diagnostics/promise_specialist_handoff/live/workspace_scenario_20261004_234129.json \
  --output-dir diagnostics/logic_puzzles_independent_live
```

Defaults use the existing pinned patched Parliament checkout and Python runtime;
`--parliament-root`, `--parliament-python`, `--model`, `--core-root`, and
`--frameworks` allow explicit alternatives. The native worker loads the existing
RAGAIMODEL `.env` without exporting its contents. The OpenAI backend remains
unchanged and uses its existing schema-constrained API calls. See the
[official structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

Each framework gets at most two adapter calls (initial plus native repair).
Existing provider retry behavior is unchanged. Generation writes exact shared
input, prompts/schemas/responses, full native traces, per-framework propositions,
a readable generation summary and the combined conflict graph. New proposition
IDs remain scoped to their originating framework, preventing different generated
premises with the same local ID from merging. Invalid submissions are displayed
as diagnostics rather than becoming committed claim nodes. This is an experiment
in independent assessment, not an added admission gate for the legacy pipeline.

Five projection/isolation tests plus the existing five source-advisory tests pass.
Capture tests exercise all five native framework paths, identical seeded inputs,
empty peer state, rejected responses and the two-call bound without making API
calls. Live semantic quality is assessed separately.

The first live `o3` pass completed all five frameworks with one adapter call each.
All five submissions were native `VALID`: utilitarian and virtue ledgers committed;
deontology, care and Rawls committed with uncertainty. The combined graph has 14
native claim records, 54 nodes and 33 edges. It retains three calibration objections
and three reported internal conflicts from deontology. The source world remained
unchanged. These are usable independent outputs, not a demonstration of improved
ethical conclusions; a matched legacy comparison remains next.

## Targeted review increment

```bash
.venv/bin/python run_logic_puzzles.py --targeted-review \
  --trace diagnostics/logic_puzzles_independent_live/framework_generation.json \
  --output-dir diagnostics/logic_puzzles_targeted_review_live
```

Select at most two existing graph objections. Bundle issues owned by the same
framework into one coherent native assessment; this avoids two competing rewrites
of a single claim. The packet includes full targeted records, their attributed
support and dependencies, the original source advisory and the immutable world.
It excludes unrelated peer assessments, votes and confidence. Native `qa` answers
the bundle; the report separately checks each original calibration objection.

Revisions run through the existing native ledger transaction path in a fresh
one-cycle workspace, with at most two adapter calls per framework. Reconciliation
replaces only that framework's admitted proposal; failed or misaddressed attempts
remain diagnostic and leave original claims operative. Every original assessment
is retained. New local hypotheses remain attributed and never modify the world.
`NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR` describes an absent calibration warning,
not independently verified semantic correctness. A model's `RESOLVED` answer
does not erase an objection still reported by the validator. Native uncertainty
is preserved and collective judgment remains `NOT_ADJUDICATED`.

Outputs include original and revised graphs, exact review packets/model calls,
full native traces and `targeted_review.md` with original/proposed records and
residual issues. Five review regressions cover packet locality, the two-issue
limit, failed/misaddressed revisions, untouched frameworks and false resolution
claims. The native challenge ID prefix is `CHALLENGE:`; tests verify the agenda
survives native broadcast filtering and that the `qa` schema is actually required.

The live answer revised `INTENDED_AS_MEANS` to `FORESEEN_SIDE_EFFECT` and left
`DOING_HARM` as `UNRESOLVED`. Its duplicate prose action map contradicted one typed
duty verdict, so the native parser initially discarded the ledger. A deterministic
review-only rendering now derives that display map from the typed duty fields,
retaining the raw answer and transformation provenance. It does not change duties,
participants, scope, premises or the world. Unsupported typed content still receives
ordinary native validation. This closes a redundant representation, rather than
loosening semantic admission.

Replaying that same saved live response through the corrected path (no new API
call) reconciles the native revision with `COMMITTED_WITH_UNCERTAINTY`. Both original
calibration warnings disappear; doing harm remains unresolved. The recommendation
stays not to pull the lever. A remaining required-duty/consistent-relation warning
and the other frameworks' native warnings remain visible in the graph. No majority
decision or governing verdict is inferred. The failed handoff and original live
answer remain separate artifacts. Results:
`diagnostics/logic_puzzles_targeted_review_live/closed_replay_final/targeted_review.md`.

All 17 review, projection/isolation and source-advisory tests pass, including a
saved-response native regression for the exact revision and retained uncertainty.

Preserve the admitted-world schema, blueprints, composition engine, source
advisory, original framework ledgers, and the legacy runner as a comparison arm.
Start from the existing committed_native_reasoning adapters, proposition ledger,
framework internal conflicts and DeliberativeProblemState. The current system
already emits structured assessments; this branch changes their scheduling and
shared representation rather than recreating them from essays.

## First bounded experiment

1. Independent framework assessments against the same frozen world and scoped
   source advisory. Suppress peer votes, confidence and salient-policy exposure
   during this pass. Preserve framework-specific commitments.
2. Project committed records losslessly into attributed CLAIM nodes; retain
   native ledger IDs and full records, not truncated display summaries. Show
   rejected proposals separately so projection does not erase diagnostic evidence.
3. Attach SUPPORT and DEPENDENCY edges to canonical evidence/proposition IDs.
   FRAMEWORK_COMMITMENT is normative authority local to its origin; it is not an
   empirical world fact. Hypotheses stay attributed and hypothetical.
4. Create typed conflict records: factual contradiction, unsupported inference,
   normative priority disagreement, or internal framework conflict. Different
   action preferences alone are a disagreement, not a logical contradiction.
   Model-proposed collisions remain proposed until their endpoints and grounds
   are reviewable; unknown relations remain explicit.
5. On at most two actions, select at most two localized issues for one targeted
   challenge round. Send only the relevant native records, source/evidence and
   scoped premises. Answers identify an objection target and proposed revision,
   retained commitment, or unresolved conflict. Keep a readable explanation.
6. Reconcile through existing transactional ledgers. Produce a conflict graph,
   revisions and residual disagreement. Consensus or majority does not promote
   a premise or settle a normative conflict.

Use nodes for CLAIM, FRAMEWORK_COMMITMENT, REVERSAL_CONDITION and
UNRESOLVED_CONFLICT; SUPPORT, OBJECTION and DEPENDENCY are directed relations.
Objections must distinguish challenges to facts, inference and normative priority.
Claim identity includes origin, action, scope, target and native-ledger reference.
A display paraphrase is not a semantic identity key.

## Evaluation before replacing legacy deliberation

Compare the legacy loop, independent assessments alone, and independent
assessments plus targeted conflict review on identical frozen worlds and matched
models, token/call budgets and seeds where available. Include lever choice,
allocation and a positive/negative promise pair. No more than two actions in the
first pilot. Measure grounding/citation errors, retained source scope, hypothesis
promotion/contamination, material conflict recall and false conflicts, unsupported
inferences corrected, native framework retention, revisions justified by evidence,
latency and cost. Admission and governing status remain separate metrics.

Agreement is not the primary success criterion. A useful unresolved conflict
with accurate support is preferable to invented agreement. Add a graph-only arm
if needed to identify whether targeted exchange adds benefit beyond projection.

The cited controlled debate study uses verifiable logical puzzles; it motivates
this experiment but does not establish an advantage for ethical deliberation.
Source: https://arxiv.org/abs/2511.07784

## Limits

This branch cannot cure weak reasoners, detect every hidden premise, guarantee
independence from shared model priors, or make every normative dispute decidable.
A common graph must preserve the different geometries of welfare, duties,
relationships, virtues and fairness; do not flatten them into one confidence vote.
Use general claim/dependency primitives and native adapters, not a new dilemma
family or schema per scenario. The first deliverable is inspection and a paired
experiment, not another admission gate.
