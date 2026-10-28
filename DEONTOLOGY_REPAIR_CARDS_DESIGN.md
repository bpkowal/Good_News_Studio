# Local derivation repair: bounded design

Status: first increment implemented October 20. The existing review exports card-discovery sidecars; a separate local-repair API stages native-revalidated, nonvoting ledger revisions. Existing full-response review and operative ledgers are unchanged.

## First increment and replay

`deontology_repair_cards.py` implements the two families below, trusted field permissions, immutable input hashes, atomic person-role repairs, dependent-closure invalidation and whole-ledger native revalidation. `logic_puzzles_review.py` writes `derivation_repair_cards.json` beside review outputs without adding it to the existing full-review prompt. `request_repair` exposes a compact one-card backend request, but has not been exercised live.

Run the saved-response experiment with:

```bash
.runtime/venv/bin/python replay_deontology_repairs.py
```

Inspect [replay results](diagnostics/deontology_repair_cards_20261020/RESULT.md). Both saved local corrections survive; the changed norm, restored prohibition, priority claims and A1 edits are excluded. This is an explicit projection of a saved full reply, not evidence that a fresh local prompt will elicit those corrections. The original run is not rewritten and the staged ledger cannot vote. Lack of remaining classification cards does not establish adequate moral reasoning. Authority, duty priority, exceptions and risk significance still need argument.

## Diagnosis supported by the code

The native calibrator already records typed calibration issues and lowers unsupported adjudication to CONFLICTED/CONTESTED. The review path still requests a full delegate response; its continuity fingerprint includes verdict, governing norm, priority and harm/application classifications. A revised bundle can therefore contain a useful correction and a fresh unsupported categorical conclusion, then be rejected together.

logic_puzzles_partial_review.py is an existing partial-transaction starting point. It isolates hr/mr changes, but the hr branch only permits UNRESOLVED with contested/unknown resolution. It does not carry all of the companion maxim fields required to correct instrumental belief/use consistently. Merely bypassing continuity for the existing full review would therefore be unsafe.

The previously restored ledger is already qualified, not an unqualified approved prohibition. Unsupported classifications remain inspectable inside it; source world facts were not contaminated by the rejected revision. The target is retaining valid local corrections while preserving that qualification.

## Repair-card contract

A versioned review-side RepairCard is not a world fact or an ethical verdict. Required records:

- Deterministic card ID, native issue kind, action ID, target record ID, and hashes of the frozen world and committed source record.
- Current values, exact evidence references and source paths, validator contract/version, and violated invariant.
- Allowed mutation paths and required atomic companion fields. The model cannot expand these permissions.
- Proposed repair question, explanation and evidence requirements. No desired moral verdict.
- Explicit dependent paths to invalidate and paths/commitments that must remain unchanged.
- Trusted provenance distinguishing validator-produced evidence from model-authored interpretations.

The response contains card ID, proposed local changes, evidence IDs, explanation, and unresolved questions. It contains no rewritten world, other action, scores or final ranking. Stale source hashes or out-of-scope edits reject the local transaction while retaining its proposed explanation for inspection.

## First two families

1. HARM_CLASSIFICATION: use the existing HARM_RELATION_GRAPH_MISALIGN issue to show the affected action's direct intervention, relevant protected party, outcome modality and graph paths. Quote DOING_HARM requirements as this implementation's classification contract, not a universal moral theorem. A withdrawn doing-harm classification does not establish permissibility; risky conduct can still be criticized through a separately defended norm.
2. PERSON_ROLE_CONSISTENCY: jointly review mr and mx.person_role, instrumental_belief, instrumental_use, practical_reasoning, and relevant means/end references. A repaired enum alone is insufficient if its companion fields still assert the opposite role. External causal necessity does not establish an actor's instrumental belief or intended use. Missing intention stays unknown.

Permit proposed alternatives or abstention; accept them only through the relevant structural checks and retain unassessed semantic content as attributed. Do not hard-code MIXED or FORESEEN_SIDE_EFFECT as the desired answer.

## Transaction and dependency handling

Start from the actual committed record, not the stronger raw model response. Preserve the original raw response, effective ledger and every patch in separate immutable records.

After an upstream correction, invalidate any adjudication explicitly relying on that premise. For the first increment, conservatively reopen the affected action's verdict/priority closure using the existing CONFLICTED, UNRESOLVED and CONTESTED states. Preserve old pr/rs text as superseded argument history; do not present it as a current supported justification. Candidate maxim assessment/permissibility and action-level preference also need explicit provisional/stale status where their support was removed. The sibling action, its evidence and the framework's commitments remain untouched.

Run native revalidation on the locally assembled candidate. Classify the actual diff as EVIDENCE_REPAIR, APPLICATION_REVISION, PRINCIPLE_REVISION or MIXED/UNRESOLVED. A model-supplied category is not authorization. Only trusted card scope and observed edits can permit a narrow evidence/application repair through continuity. Principle edits and mixed changes keep the stronger guard; local repair cannot smuggle a changed priority principle under a classification label.

A failed attempt must not discard an independently valid repair to another card family. The resulting ledger must nevertheless validate as a whole, because local fields can interact. Never promote a local patch straight into an operative moral vote.

## Effective resolution

Retain the model's proposed resolution separately from the effective committed status. An active decisive defect forces CONTESTED. Absence of known cards is not proof of semantic adequacy, completeness, grounded moral authority, or a settled priority. The first two-family increment deliberately cannot restore a categorical verdict; a separate adjudication pass must defend any new closure.

## Authority and party-grounding caveats

moral_authority=NOT_ASSESSED currently belongs to the source-rule prompt view, not to a newly established frozen-world moral fact. It marks missing assessment, not proof that a rule has no authority. A source rule's wording cannot by itself certify moral bindingness. An agent can offer an attributed argument for its authority; it cannot silently promote institutional provenance into a shared moral fact. A later authority card should require the explicit bridge and retain unresolved applicability/exception questions. Do not detect this solely by keyword or citation matching.

A protected-party claim need not always have an action-local welfare effect. A duty can protect someone from imposing risk without producing a positive benefit node in the alternative branch. Use a supported normative relation or an explicitly qualified interpretation where appropriate. Missing grounding is a question, not automatic proof of row contamination and not permission to invent an avoided-harm effect.

## Evaluation before integration

First use saved replies only, with zero API calls:

- Original contradiction: both relevant card families are generated from existing typed issues with exact evidence paths.
- Existing corrected review: consistent companion-field corrections survive locally; attempted restoration of PROHIBITED/PRIMARY/PERFECT_DUTY is withheld pending fresh adjudication.
- Invalid repair: changing a framework commitment, A1, evidence or source world is rejected and cannot leak through a self-declared EVIDENCE_REPAIR label.
- Stale card, incompatible companion fields, absent references and malformed patches leave the original record/history recoverable.
- A1 and frozen-world hashes remain unchanged; accepted control records without those issues remain identical.

Then run one bounded live local-repair probe on the same saved A0, one family at a time, without regenerating the scenario or all framework judgments. Measure surviving local corrections, genuinely remaining dependencies, prompt size, principle preservation and native acceptance separately. A preferred action change or increased count of RESOLVED labels is not a success criterion.

Defer generic philosophical prompt expansion, a complete normative ontology, automatic authority adjudication, and effect-only party rejection. Norm/authority and adjudication cards can follow once the first two repair families work reliably.
