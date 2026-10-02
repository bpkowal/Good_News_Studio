# Dilemma interpretation evaluation set — v1 draft

This is a separate, parser-independent evaluation set. Z10 is the frozen baseline;
no parser, proposer, resource, or existing regression fixture was changed to build
it. The set has **32 cases: 16 development and 16 held-out**, with the same broad
construction coverage but disjoint scenario and template families.

The annotations are assistant-authored **proposed gold**, not independently
adjudicated ground truth. They were written without running any parser or model
on these inputs. Integrity checks do not establish linguistic correctness.

## Contents

- `dev.jsonl`: development cases with proposed annotations.
- `heldout.jsonl`: author-visible local holdout; use explicitly only for evaluation.
- `manifest.json`: split digests, provenance, evaluation policy, and checksums of
  20 baseline code/resource files plus installed NLP package versions.
- `dataset.py`: integrity validation and gold-free model-input export. It uses no
  parser, model, network service, or model-based semantic judge.
- `test_dataset.py`: integrity, leakage, drift and input-export checks.

| Primary construction | Development | Held-out |
| --- | ---: | ---: |
| Stripping | 2 | 2 |
| Verb-phrase ellipsis | 2 | 2 |
| Gapping | 1 | 1 |
| Sluicing | 1 | 1 |
| Noun-phrase ellipsis | 1 | 1 |
| Comparative ellipsis | 1 | 1 |
| Fragment answers | 1 | 1 |
| Coreference | 1 | 1 |
| Truncation | 2 | 2 |
| Non-ellipsis controls | 4 | 4 |

Secondary phenomena include negation, modality, attribution, conditions,
counterfactuals, belief, collective action, quantities, attempts, decisions, and
strict/sloppy possessive ambiguity. Cases concern allocation, disclosure, rescue,
consent, access and procedural fairness. The target is interpreting their premises,
not agreeing with a prescribed ethical verdict.

This is a small pilot: one or two examples per construction per split cannot
establish general accuracy or reliable population estimates.

## Annotation contract

Each JSONL record contains an ID, split, scenario/template family, primary
construction, secondary phenomena, original text, exact evidence spans, review
status, and a `gold` object. Evidence offsets are half-open Unicode code points;
`text[start:end]` must equal the recorded evidence. `source` preserves the full
context; `focus` marks the targeted expression rather than invented missing words.

`gold` contains:

- `readings`: scoped natural-language interpretations with stable local IDs and
  source references. These describe what the passage says or supports as a reading;
  they do not authorize world-state commitments.
- `reading_policy`: one target reading, nonexhaustive competing readings, or
  abstention when missing text cannot be recovered. An explicit scope question is
  safe abstention but does not count as generating both operator readings.
- `checks`: independently scorable requirements for roles, scope, references,
  quantities, occurrence, alternatives, and missing content.
- `forbidden_inferences`: examples of unsupported promotions or interpretations.
  These examples are not an exhaustive blacklist of possible hallucinations.
- `open_questions`: specific unresolved content, especially for truncation.
- `downstream_probes`: a question and a semantic answer criterion, including which
  readings must be considered. Exact answer wording is not required. A probe's
  `required_reading_ids` are alternatives to consider, not jointly asserted facts.

All annotations have `annotation_coverage: targeted_not_exhaustive`. Additional
valid propositions elsewhere in a passage must not be counted as false proposals
merely because they are outside this targeted gold. Identity is occurrence-based;
repeated names or compatible gender do not automatically establish coreference.

## Split and review policy

Scenario and template family labels are disjoint across the two splits. Broad
linguistic constructions intentionally recur. Do not move a paraphrase, name swap,
or whitespace variant into the opposite split. Semantic equivalents of an existing
case belong to its original family and split, even if their wording differs.

The integrity tool checks family labels and exact text overlap; it cannot prove
that two differently labeled examples use different semantic templates. Before a
formal comparison, a reviewer should audit those boundaries and overlap with the
older development fixtures, review the alternatives and forbidden inferences, and
adjudicate uncertain cases. In particular, acceptability of contrast readings may
depend on discourse focus. If a case remains disputed, mark it for analysis and
exclude it from headline accuracy instead of forcing a single answer.

Held-out examples are visible to the dataset author and present locally. This is
not a secret or contamination-free external benchmark. They have not been used
for inference or tuning during dataset creation. Do not inspect held-out outputs
while choosing prompts or modifying the system. If they are used for development,
retire that holdout and build a new one. Record annotation revisions under a new
version and reviewed digests before comparative evaluation.

## Comparison protocol

1. Adjudicate and freeze the annotations before reporting accuracy.
2. Compare frozen Z10, whole-passage model proposals, and Z10 plus targeted model
   proposals. Give all systems the same original text. A proposition model may be
   a fourth baseline. Do not expose construction labels, focus spans, questions or
   gold answers to the model in the interpretation task.
3. Tune only on development. Record model/version, prompt, decoding settings,
   package versions, input/output digests, latency, token use and any cost. Preserve
   raw output, adapter output, validation errors and failed/empty responses.
4. Evaluate held-out once the compared configurations are fixed. Report every
   case; invalid outputs or unavailable candidates must not disappear from recall
   denominators. Keep model-output failures separate from adapter/contract errors.
5. For downstream probes, compare raw-text reasoning with reasoning supplied the
   interpretation package. Provide the original text in both; hide reference
   answers. Require evidence/reading citations and analyze answers per compatible
   branch. A citation is not proof of support, and an uncertain reading must not
   become a factual premise merely because it was selected.

No model calls, Z10 runs on this set, scoring claims or automatic semantic judge
are included in this increment.

## Scoring rubric

Use blinded human review for semantic alignment, with a second review for disputed
judgments. NLI or LLM judgments may assist but are not verification. Match whole
scoped readings one-to-one; do not use string equality as semantic equivalence.

Report raw counts and denominators **per construction and per split**:

| Measure | Definition |
| --- | --- |
| Target reading recall | Matched expected readings / expected readings |
| False proposal rate | Semantically unsupported targeted readings / targeted proposed readings |
| Scope preservation | Passed applicable scope checks / applicable scope checks, with missing representations counted separately and in end-to-end scope coverage |
| Role/reference accuracy | Correct targeted assignments / proposed targeted assignments; also report missing required assignments |
| Abstention | No-proposal cases, explicit unresolved questions, and unresolved proposed readings separately |
| Ambiguity preservation | Cases retaining all specified competing readings / annotated ambiguous cases; report explicit abstention alongside this |
| Unsupported promotion | Occurrence, certainty, causal or normative claims exceeding source scope |
| Downstream fidelity | Probes answered consistently with evidence and all relevant alternatives / probes |
| Contract validity | Valid adapted packages / all attempted packages, separately from semantic metrics |

An unmatched prediction is not automatically false: gold alternatives are
nonexhaustive. Review new supported readings, record adjudication, and apply any
annotation correction consistently to every system. Duplicate readings cannot earn
extra recall. Abstention on truncation can be correct; abstention on a recoverable
case still loses recall. Do not combine all measures into an unexplained scalar.
Use `null` when a denominator is zero. Report control false positives and incomplete
inputs separately so a system proposing nothing cannot appear successful.

## Commands and baseline freeze

From the repository root:

```sh
.venv/bin/python evals/dilemma_interpretation_v1/dataset.py
.venv/bin/python -m unittest discover -s evals/dilemma_interpretation_v1 -p 'test_*.py' -v
.venv/bin/python evals/dilemma_interpretation_v1/dataset.py --export-inputs /tmp/dilemma-dev-inputs.jsonl
```

The export defaults to development and includes only `id` and `text`. Held-out
export requires `--split heldout`. Integrity checks inspect both splits structurally
but do not send either split to a parser/model or print the gold.

Baseline digests detect edits to Z10 and its local dependencies/resources. They do
not provide a separate checkout or make files read-only. External model weights
are not vendored; recorded package versions are a reproducibility requirement,
not a checksum of every installed dependency. Do not refresh baseline hashes to
silence a drift error. Future systems should have separate entry points or isolated
checkouts. Preserve this baseline when shared dependencies change.
