# Parsing game R: error attribution and explicit repair evidence

R leaves Q unchanged. It adds an explicit passive-repair evidence contract and
audit schema 2. Parser output schema is 8; CEM remains the same four-action model
with 11 features and policy schema 2. There are no new training examples or
new semantic classes.

```sh
python parsing_game_R.py
python -m unittest test_parsing_game_R -v
python audit_parsing_game_R.py --output diagnostics/R_audit.json
python audit_parsing_game_R.py --no-repair --output diagnostics/R_without_repair.json
```

The audit defaults to seeds 7, 42, and 91; `--seeds 42` selects one. Normal parsing
does not run the audit. R uses its own diagnostic prefix and rolling probe file
`diagnostics/parsing_game_R_user_probes.json`, keeping the latest five entries.
User probes are not copied into dated audit archives.

## Concern 1: distinguish recovery, interpretation, and gating

Audit records distinguish two recovered arguments from a numeric observation,
annotated pair correctness, relation scope, raw decisions, and gated outcomes.
Missing/wrong pairs are extraction failures. An annotated out-of-action-space
relation is recorded separately when a candidate exists. Unlabelled cases remain
unreviewed, not automatically unsupported or incorrect.

Both CEM and baseline now report conditional accuracy, disagreements, false
commitments, false rejections, gating failures, and abstention. Every rate includes
its numerator and denominator. The paired comparison reports both-correct,
CEM-only-correct, baseline-only-correct, and both-wrong.
Raw disagreement is observable without gold labels and includes all candidates
with two raw decisions; it is not itself an accuracy or error metric.

Conditional prediction scoring requires a correct annotated in-scope pair.
Gating failure attribution additionally requires a correct raw interpretation
and a Boolean `expected_eligible` annotation. A false commitment or rejection
can be counted as a pipeline error without being blamed on the gate when the
upstream pair/decision was wrong. Baseline and CEM use the same validator.

`pair_and_raw_action_accuracy` deliberately names what it measures. It is not
final graph accuracy. The separate gate outcomes and existing assertion-aware
suite metrics expose downstream behavior.

Potential representation collisions still require identical vectors with
conflicting labels on correct, representable pairs, and are flagged for review.
A wrong raw prediction is a policy-error candidate, not proof that the features
were sufficient. A near-collision alone does not establish representation loss.

The six cases in `resources/R_commitment_audit.json` contain explicit eligibility
expectations for positive, negative, modal, temporal-adjunct, provisional-repair,
and quantified claims. They are local policy checks, not real-world truth labels,
and are excluded from CEM fitting. Their summary is separate from the original
39-case benchmark. Those older cases have no eligibility labels: their false
commitment and gating-failure rates remain null, not zero.

The audit API also accepts `argument_token_indices` (two gold head indices) to
disambiguate repeated mentions. Without that annotation, repeated textual gold
mentions leave pair correctness unscored rather than guessing an occurrence.

## Concern 2: a repair must expose its evidential limits

`assess_passive_fragment()` returns a versioned record containing:

- affected nominal, agent/cause candidate, predicate, and proposed direction;
- required construction checks and supporting evidence;
- original parser tag/dependency and lexical-policy provenance;
- competing interpretations, failed checks, and eligibility;
- a provisional or rejected interpretation status.

Required checks cover the licensed form, affected nominal position, lack of a
direct object, source availability, local dependency configuration, auxiliaries,
event complements, and source coordination. Supporting checks separately record
adjacency, attachment, lexical compatibility, and the parser's participle tag.
The insomnia example explicitly preserves its conflicting finite VBD tag.

Competing evidence distinguishes `detected`, `not_detected`, `unresolved`, and
`unassessed`. Deadline and event-valued by constructions have bounded checks;
path/proximity and instrument/means are explicitly unassessed by this adapter.
The active/elliptical alternative remains unresolved. Sentence/dependency checks
are not a general clause-boundary disambiguator.

Rejected attempts remain in candidate evidence as `passive_fragment_assessment`,
including failures such as a temporal source or unsupported predicate. Only
provisional accepted candidates become `structural_hypothesis` records used by
the existing repair path. Rejection of a repair is not rejection of an ordinary
fully parsed passive or active claim.

R does not emit `supported` for these repaired fragments: it has no evidence
that closes all competing readings. `stress -> insomnia` remains provisional,
its contextual occurrence unknown, and its graph commitment blocked. Preserving
an expected arrow is only one part of repair success.

## Validation

Before the controller-preservation patch, the suite contained 39 tests: 37 passed and the two existing structural expected
failures remain. It reuses Q/P/N regressions and adds explicit rejected-repair
checks, injected false-commitment/false-rejection gates, extraction-versus-gating
attribution, missing-label handling, unsupported-label handling, and repeated
mention annotations.

On the original 39-case benchmark, the seed-42 audit retains 39/39 for both raw
CEM and baseline with zero disagreements. On the separate six labelled policy
checks, both pipelines have zero false commitments, false rejections, and gating
failures. These small fixture results do not establish general reliability or
incremental predictive value from CEM. CEM development remains unchanged while
the audit collects evidence of whether learned weighting contributes.

## Controller preservation (added; tests not yet run)

Controller candidates belong to event links, independently of a parent's entity
object slot. Each parent proposition has an `event_links` list, one record per
discovered child. The corresponding complement relation exposes `controller_link`,
and world-state event packets also retain the parent's links. This metadata is
serialized with rolling probes and audit proposition records.

For `The manager allowed the workers to leave`, the intended record is:

```text
parent.object: absent
child.subject: the workers
event_link.controller_candidate: the workers (with original mention offsets)
event_link.controller_source: child_subject
event_link.configuration: distinct_child_subject
```

`force` may retain an explicit parent object while exposing the same child-subject
candidate on its link. `try` uses `subject_control` only when the existing adapter
provides subject-transfer provenance; simple subject identity otherwise means
`shared_subject_candidate`. An object matching the child's subject is recorded
as `parent_object_matches_child_subject`, not as proof of a general object-control
semantic rule.

Support distinguishes explicit child subjects, adapter-provided subject control,
and unvalidated recovered child subjects. Missing child subjects remain missing;
no parent noun is borrowed to fill them. Each link preserves its own candidate,
including in nested/multiple-complement structures. Finite `ccomp` links retain
their child subjects on the propositions but are not classified as controllers.

Controller records are candidates, always ineligible for direct world commitment.
They do not alter parent objects, CEM features, relation classes, assertion,
entailment, or occurrence. The smoke display now prints `syntactic object` and
`controller_candidate` separately. Normal probe output displays controller source,
support, and configuration as well.

Six new regression methods cover cause/allow/force, nested subject control,
adjunct/passive-agent separation, finite reporting clauses, missing/multiple
controllers, and display/serialization. These tests and the full suite have
**not been run after this patch**, as requested. Earlier validation numbers above
refer to the preceding R revision.
