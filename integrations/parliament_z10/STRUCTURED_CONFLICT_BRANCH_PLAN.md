# Proposed structured-conflict branch

Status: `Logic_Puzzles` branch started. Read-only claim/conflict projection and
isolated native framework generation implemented. Targeted review and collective
judgment changes remain planned.

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
ethical conclusions; targeted review and a matched legacy comparison remain next.

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
