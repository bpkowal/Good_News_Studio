# Burden-preserving comparisons: bounded increment

Instruction version `argument-framework-instructions/0.3`; pilot envelope
`neutral-framework-pilot/0.2`; the live cohort used comparison review
`complaint-comparison/0.1`. A zero-call post-hoc projection uses
`complaint-comparison/0.2`.

## Diagnosis from the first cooling reply

The agent rejected A using Mira's displacement, left B versus C unresolved, and
then declared B uniquely nonrejectable. It also described B as avoiding both
Jo's unmet need and Sal's income loss, although B does not cool Jo's home.
Its comparison records named reasons but had no explicit alternative pair or
record of the same person's position under each alternative. Its final preference
was not distinguished from a claim that all alternatives had been ruled out.

This is a diagnosis of a visible reasoning path, not a claim about hidden model
cognition. The proposed mechanism targets those missing intermediate steps.

## Constructive change

Before ranking, the model records relevant burdens with stable person/burden IDs
and positions under every supplied option. Each cell preserves the position's
quantities, probabilities, scope, and unknowns, along with factual ground handles.
PRESENT in two cells does not mean equal burden: mortality risk can remain while
changing from certainty to 20%. Consent can alter the justification for imposing
a cost without making that physical or financial cost disappear.

Each comparison specifies two actions, the relevant reasons, introduced/relieved/
retained burdens and a comparison status. Reasons against a third option cannot
silently resolve this pair. Retained burdens remain visible even if a defended
normative argument limits their force. The final record distinguishes a unique
defensibility claim, a provisional preference, multiple defensible choices and
an unresolved conclusion. An open comparison does not prohibit a provisional
recommendation or trigger a compiler-selected substitute.

This scaffolding is common to all four specialists. It supplies no severity
weights, lexical priority among complaints, or correct recommendation. Each
framework still gives its own ethical warrants. The new matrix and comparison
scope extend the person/value reason structure credited in
[CONTRACTUALIST_RESEARCH.md](CONTRACTUALIST_RESEARCH.md); they are not claimed as
the paper's weighted ASPIC+ evaluation.

## What deterministic review can and cannot establish

`complaint_comparison.py` checks reference coverage, annotation transitions,
retained burdens, pair-local reason references and whether the stated comparison
statuses support the stated claim of uniqueness. Findings are a separate
investigative record. They do not remove the argument, edit its recommendation,
decide moral priority or modify the world. A well-formed matrix can itself be
factually wrong, and a structurally consistent comparison can be morally weak.
Prose contradictions and misrepresented severity still need content review.

Offline contrasts exercise retained burdens, unknown risk, changes in probability,
consent with continuing cost, irrelevant third options, provisional recommendations,
unsupported uniqueness, incomplete matrices, and action/person renaming. These
check general bookkeeping invariants, not predicted moral verdicts.

## Next live cohort, fixed before calls

Run the same four frozen cases and four frameworks: 16 independent o3 calls,
low reasoning, 8192 completion tokens, no retries, repairs, peer input or collective
judgment. Preserve every technical failure and raw argument graph. No input text,
action set or expected moral answer is changed. Source/implementation/request
hashes freeze before the calls in a fresh results directory.

Review each framework/case using the existing 0–2 content rubric, with specific
attention to source-faithful positions, retained burdens, defended comparisons,
uniqueness versus provisionality, and the exact consent/assurance contrasts.
Report technical findings separately from those substantive judgments. Compare
the first cooling reply only as an exploratory before/after observation: both
format guidance and comparison scaffolding changed, and one sample is not a
treatment-effect estimate. A changed action is neither required nor evidence of
improvement by itself. Any assistant-only review is explicitly labeled.

```bash
.runtime/venv/bin/python run_contractualist_pilot.py \
  --output diagnostics/contractualist_comparison_v03_live_20261025 --execute
```

The active Rawlsian roster and both production pipelines remain unchanged. This
study must earn an integration decision by producing defensible additional reasons.

## Recorded outcome

The 16-call cohort completed; see
[content review](diagnostics/contractualist_comparison_v03_live_20261025/CONTENT_REVIEW.md).
The comparison scaffold improved retention of burdens and made some preferences
explicitly provisional, but it did not reliably produce defended priority judgments
or eliminate contradiction between uniqueness claims and residual disagreement.
Fifteen graphs retained binding issues; all raw responses and partial graphs survive.

After the cohort, a deterministic position-change projection was applied to the
same 16 replies with zero API calls. It preserves original annotations and proposed
rankings; missing/duplicate cells remain unknown. This improves the reliability of
comparison presentation, not the normative reasoning of the original replies.
