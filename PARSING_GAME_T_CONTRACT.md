# T candidate package contract — draft 0.2

Z8 adds schema 0.4 reconstruction identities and antecedent provenance, specified
in [PARSING_GAME_Z8.md](PARSING_GAME_Z8.md), on top of the Z7 condition-content extension.

Z7 adds schema 0.3 condition-content records and selection rules, specified in
[PARSING_GAME_Z7.md](PARSING_GAME_Z7.md). The 0.1/0.2 contract below remains the
legacy base; schema 0.3 requires those additional records.

Status: incrementally implemented by T2. S remains unchanged. T2 exports schema
0.2; the validator also accepts legacy 0.1 packages, but rejects 0.2 scope fields
in them. Package and selection versions must match. The base examples below use
0.1 for compatibility; use 0.2 for new exports containing the scope extensions.

The package preserves text evidence and competing interpretations for downstream
reasoning. A consumer selects supported candidates or leaves questions unresolved.
Neither exporting nor selecting a candidate commits a fact to Parliament's world
state. Contract validity is not semantic correctness or external truth verification.

## 1. Package envelope

```json
{
  "schema_version": "0.1",
  "package_id": "pkg_demo",
  "document": {"id": "doc_demo", "text": "Maria can leave."},
  "producer": {"name": "parsing_game_T", "version": "T0", "resources": []},
  "evidence": [],
  "nodes": [],
  "candidates": [],
  "choice_sets": [],
  "open_questions": [],
  "coverage": {"status": "partial", "unrepresented_evidence_ids": [], "limitations": []}
}
```

All six arrays/objects after `producer` are required, including when empty.
`coverage` must never imply complete understanding merely because all tokens occur
in evidence spans. `unrepresented_evidence_ids` records known gaps, not exhaustive
coverage. No normative verdict, recommended action, or world-state mutation belongs
in this package.

IDs are unique within a package, immutable once exported, and scoped to its
`package_id`. Sentence boundaries do not reset IDs. Re-parsing or revising an
export creates a new package ID; this version promises no cross-revision identity.
Unknown schema versions are rejected, not interpreted as the nearest known version.

## 2. Evidence and nodes

Evidence records have `id`, `start`, `end`, and `text`. Offsets are half-open Unicode
code-point offsets into the exact document text; no whitespace normalization occurs.
The invariant is `document.text[start:end] == evidence.text`, with
`0 <= start < end <= len(text)`. Discontinuous support uses multiple evidence IDs.
Tokens, dependency analyses, and model outputs are provenance, not source evidence.

Node records have:

- `id` and `kind`: `mention`, `proposition`, or `choice_point`.
- `label`: a display string, never an identifier or a semantic guarantee.
- `evidence_ids`: a nonempty list of source evidence IDs.

A proposition additionally has `predicate` (normalized lemma/display predicate)
and `predicate_evidence_ids` (a nonempty subset of its evidence). Propositions
include events and states; existence of a proposition node does not assert occurrence.
A mention represents one occurrence in text, not a resolved world entity.
A choice point anchors a textual decision such as “decide whether”; its node
alone does not establish an exhaustive set of available actions.

Node records are anchors for hypotheses. A node becomes part of a selected
interpretation only through selected candidates or explicit selection in the
consumer result. Unselected nodes remain available in the package.

No merged entity clusters are needed in 0.1. Candidate `SAME_REFERENT` edges link
mentions. Selecting such an edge does not destructively merge their evidence.
Group cardinality and membership must not be inferred from mention identity.

Validator increment amendment: the envelope may additionally contain
`identity_constraints` (absent means empty). Each record has `id`, `mention_a`,
`mention_b`, nonempty `evidence_ids`, and `provenance` in the candidate provenance
format. It declares that these two distinct mention endpoints must refer to
different referents. IDs share the package namespace. Constraints are immutable
package input, not consumer additions. Selected identity edges must respect them
both directly and through transitive chains. Different names alone do not create
such constraints. The current exporter does not automatically generate them;
their semantic support needs separate review.

## 3. Candidate records

Every candidate has these required fields:

```json
{
  "id": "c_ability",
  "type": "MODALITY",
  "arguments": {"proposition": "p_leave"},
  "value": "ability",
  "evidence_ids": ["e_can"],
  "scope": {"polarity": "positive", "contexts": []},
  "provenance": [{"producer": "modal_classifier", "version": "T0", "method": "rule", "resource_ids": []}],
  "assessment": {"status": "proposed", "score": null},
  "requires": [],
  "exclusive_with": []
}
```

`assessment.status` is `proposed`, `preferred`, or `unresolved`; none means accepted
as true. `score` is null or `{value, kind, source, calibration_id}`. Kinds are
`uncalibrated_score`, `uncalibrated_probability`, and `calibrated_probability`.
Only probability kinds require values in [0,1]; all values must be finite.
A calibrated probability requires a non-null calibration artifact ID in
`producer.resources`. Scores from different sources are not implicitly comparable.
Resource entries contain `id`, `version`, and `description`.

`requires` lists candidate IDs that must also be selected. Dependencies are acyclic.
`exclusive_with` is symmetric and lists interpretations that cannot both be selected.
Contradictory claims by different speakers need not be exclusive: scope matters.
All references must resolve inside the package. Evidence establishes traceability,
not automatic semantic support.

### Minimal typed vocabulary

| Type | Required arguments | Value |
| --- | --- | --- |
| `PREDICATION` | `proposition` | null |
| `PARTICIPANT` | `proposition`, `mention` | `subject`, `object`, `agent`, `patient`, `destination`, `location`, `controller` |
| `SAME_REFERENT` | `mention_a`, `mention_b` | null |
| `EVENT_LINK` | `parent`, `child` propositions | `complement`, `purpose`, `attempt`, `unresolved` |
| `MODALITY` | `proposition` | `prediction`, `possibility`, `ability`, `permission`, `obligation`, `unresolved` |
| `CONDITIONAL_ON` | `consequence`, `condition` propositions | null |
| `OPTION_OF` | `proposition`, `choice_point` | null |
| `QUANTITY` | `mention` | `{operator: "exact", amount: nonnegative integer, unit: string or null}` |

`PREDICATION` licenses a proposition reading, including a state reading, without
asserting that it occurred. A quantity records explicit cardinality only; “all”
requires an open question until a quantification extension is defined. Controller
candidates remain distinct from syntactic objects.

Each candidate involving a proposition, except its own `PREDICATION`, must require
the applicable predication candidate(s). This permits competing readings of the
same span without treating all proposed events as present in the selected graph.
Subject/object are syntactic roles; agent/patient need separate semantic evidence.
An event link supplies no default child-entailment rule.

This vocabulary deliberately starts with structure and scope. Causal, harm,
belief, intention, and promise relations need explicit extensions and fixtures.
Unsupported content is recorded in coverage/open questions rather than forced into
one of these types. Negated propositions use scope; “not act” is not an invented
positive action node.

## 4. Scope is compositional

`scope.polarity` is `positive`, `negative`, or `unresolved`.
`scope.contexts` is an ordered outermost-to-innermost list. Each context has a
`kind`, nonempty `evidence_ids`, and the following additional fields:

- `conditional`: `condition_proposition_id`.
- `hypothetical`: no additional required reference.
- `attributed`: `source_mention_id`, nullable when the speaker is unknown. In 0.2,
  optional `report_proposition_id` links the reporting event, whose predication
  must be in the candidate's dependency closure. T2 always emits this reference
  for supported reporting complements, preserving the report's own qualifications.
- `questioned`: no additional required reference.
- `modal`: `modality_candidate_id`.
- `modal_choice` (0.2): `modality_choice_set_id`, referencing a nonempty
  `interpretation`/`at_most_one` choice set of MODALITY candidates for an argument
  proposition. A modality open question must cover all members. This qualification
  survives abstention; no member is implicitly selected. While that question is
  unresolved, candidates carrying the context and their dependents are provisional.
  A predication anchor may carry this context without requiring a member, avoiding
  a dependency cycle. Modality candidates carry outer scope, not their own choice.

These may be nested or combined. An empty context list is not an occurrence flag;
it merely records that this candidate has no identified outer context.
An unresolved scope interpretation must generate an open question. There is no
default promotion from local affirmative wording to actual occurrence.

For T2 participant and event-link records, polarity qualifies the associated
proposition reading (the parent for an event link). Negative polarity does not
deny the existence of a mention or deny its grammatical role. Modality polarity
qualifies the operator separately: in the supported `must not` construction the
proposition is negative and modal readings remain positive. Other locally negated
MD constructions have unresolved polarity and a blocking scope question; selecting
an unresolved-polarity candidate cannot resolve that question. Lexical modality
such as `not required to` remains an explicit blocking gap rather than an inferred
obligation. Attribution does not entail that either the report or its content occurred.

The condition proposition remains recoverable even if its participants have
unresolved references. Conditional candidates require its predication candidate.
`CONDITIONAL_ON` scopes the consequence, not both propositions by default.
Modal contexts require the referenced modality candidate. To avoid a dependency
cycle, the modality candidate itself carries the outer scope, not its own modal
context; its predication anchor need not require it. Assertions about participants
or event occurrence must not be inferred from the anchor alone.

## 5. Alternatives and open questions

Choice sets have `id`, `kind` (`interpretation` or `scenario_option`),
`candidate_ids`, `selection_rule` (`at_most_one` or `any_subset`), and
`exhaustive` (Boolean). An interpretation choice set is a parser ambiguity.
A scenario-option set describes alternatives in the situation. These are distinct.
Listing two scenario options does not claim that either occurred or that they
cannot both occur; an exclusivity restriction needs textual evidence, recorded as
`evidence_ids` on the choice set. No “exactly one” rule is imposed in 0.1.

Open questions have `id`, `kind`, `evidence_ids`, `candidate_ids`, `question`, and
`blocking_for` (candidate IDs). Kinds include `attachment`, `reference`, `scope`,
`modality`, `missing_representation`, and `unsupported_semantics`.
Empty candidate lists are legal: the correct interpretation may not be generated.
Every unresolved candidate must appear in an open question. A consumer may resolve
an ambiguity by selecting a compatible candidate, or explicitly leave it open.
Missing candidates and low confidence never justify silently deleting evidence.

## 6. Consumer selection contract

```json
{
  "schema_version": "0.1",
  "package_id": "pkg_demo",
  "selected_node_ids": [],
  "selected_candidate_ids": [],
  "question_resolutions": [],
  "extensions": []
}
```

Selections refer to immutable package records; consumers cannot rewrite their
polarity, scope, arguments, or evidence. Selected nodes include all arguments and
scope references of selected candidates. Each open question receives a resolution:
`{question_id, status, selected_candidate_ids, rationale}`, where status is
`resolved_by_selection` or `unresolved`. Selection resolutions cite at least one
selected candidate from that question. Unresolved blockers keep affected candidates
provisional even if selected.

Missing interpretations may be proposed in `extensions`, each with a unique ID,
`origin: "llm_inferred"`, supporting evidence IDs, a description, and
`verification_status: "unverified"`. They are outside the supported selection graph
and cannot satisfy candidate dependencies. New nodes require a later typed extension
protocol; merely declaring one “new” does not authorize it.

## 7. Validator result and limits

`validate_candidate_selection(package, selection)` returns:

```json
{
  "contract_valid": true,
  "errors": [],
  "unresolved_question_ids": [],
  "provisional_candidate_ids": [],
  "unverified_extension_ids": [],
  "identity_components": [],
  "semantic_support": "not_assessed",
  "world_state_commitment": "not_authorized"
}
```

Validate schema, ID uniqueness, exact evidence spans, endpoint types, dependencies,
exclusive selections, scope references, and question-resolution bookkeeping.
Reject extra fields that attempt to override immutable candidates or claim actual
occurrence. Reject unknown candidate types rather than guessing their semantics.
Partial selections are valid; unresolved questions are visible incompleteness,
not automatically validation errors. A validator cannot prove a span supports a
claim, discover missing scope, or prevent every mistaken coreference chain.
Those are separate semantic evaluation responsibilities.

`identity_components` lists mention-ID groups implied by selected identity edges,
including singleton selected mentions. It is a diagnostic equivalence closure,
not a destructive merge or verified world identity. Provisional status propagates
from unresolved question candidates/blockers through selected dependencies.
Errors contain `code` and `path`. All records reject undeclared fields. The current
implementation accepts only `coverage.status: "partial"`.

The caller must retain and pass the original package. Matching a package ID is not
a cryptographic integrity check; the validator cannot detect a caller rewriting
the package itself. Consumers supply selection records, not replacement packages.

## 8. First acceptance fixtures

Before implementing the exporter, freeze expected packages/selections for:

1. “Maria can leave.” Predication plus ability interpretation; no actual leaving.
2. “Maria did not leave.” Negative scope survives selection unchanged.
3. “If Maria pulls the lever, the trolley will stop.” Condition link plus prediction.
4. “Maria saw Anna. She left.” Competing reference candidates, no forced merge.
5. “They were too late to work.” Alternative attachment readings survive abstention.
6. “The worker tried to leave.” Attempt link; child occurrence remains unestablished.
7. “Maria must decide whether to act.” A choice anchor; no inferred duty to act and
   no fabricated exhaustive option set.

Include deliberately invalid selections: dangling IDs, altered scope, unsupported
types, incompatible candidates, omitted dependencies, and extensions used as facts.
Measure candidate recall, candidate precision, span accuracy, ambiguity-set size,
and scope preservation separately from contract-validation success.

This document defines the interface only. It adds no parser behavior, classifier,
exporter, validator implementation, or test results.
