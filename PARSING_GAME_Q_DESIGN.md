# Proposed Q: causal adjudication audit and passive-fragment hypotheses

Status: design only. P remains unchanged.

## Findings from P

Using the default seed-42 trained policy, the six non-training single-pair suites
contain 39 sentences. P recovers numeric observations for 38. Raw CEM and a
deterministic semantic-class/direction baseline both predict the gold action on
38/38 available observations. The missing observation is `Insomnia caused by
stress yesterday.` These are small development fixtures, not evidence of broad
generalization or independent confirmation of CEM's value. The baseline consumes
the same upstream lexical/structural knowledge as the model.

The installed spaCy model tags the fragment's `caused` as VBD/finite, `Insomnia`
as active nsubj, and `by` as prep. Thus a VBN-only repair misses the regression.
It also tags `by` as agent in `Damage caused by noon.` An agent dependency alone
does not reliably distinguish a causal source from a deadline.

## Increment 1: measure the adjudicator's contribution

Keep the four actions and 11 numeric features. Do not train on ArgumentContext,
expand the label space, or tune on the diagnostic fixtures.

Add a separate audit runner that records, per candidate:

- recovered arguments, their span/role provenance, and ArgumentContext;
- whether a valid numeric observation reached CEM;
- raw CEM action, scores, and uncalibrated margin, before validation gates;
- deterministic baseline action using semantic class and structural direction;
- final gated interpretation and each rejection reason;
- annotated pair correctness, gold causal action, and relation scope where supplied;
- semantic-adapter results and unresolved relations alongside the causal audit.

Report candidate coverage, pair accuracy, conditional raw-action accuracy,
end-to-end accuracy, abstention, and false commitments with explicit denominators.
For baseline versus CEM report both-correct, CEM-only-correct,
baseline-only-correct, and both-wrong. Use the same candidates and commitment
gates for the two pipelines; otherwise extraction/gating differences confound
the comparison. Repeat fixed seeds on unchanged splits to expose instability.

Diagnostic taxonomy is multi-label and annotation-aware:

- A: argument extraction mismatch, confirmed against argument annotations;
- B: annotated relation outside the four-action causal task (not a CEM error);
- C: verified correct arguments, equal numeric observations, conflicting gold
  actions, after reviewing labels and scope;
- D: raw policy error on a valid, in-scope pair; the cause remains unproven until
  feature/label checks distinguish learning failure from representation failure;
- unreviewed: missing annotations; machine flags nominate cases for inspection.

Near-equal vectors and deterministic disagreements are investigation signals,
not proof of C or D. Gate failures do not automatically count as policy failures.
Preserve separate assertion/entailment diagnostics. Rolling unlabelled probes
remain unscored and excluded from training and permanent evaluation archives.

## Increment 2: bounded reduced-passive construction adapter

Keep the original Doc and attachment record. Introduce an alternative structural
hypothesis rather than overwriting spaCy tags or promoting every by-object.

An explicitly documented local lexical-construction resource licenses passive
forms for supported transitive causal predicates. Membership in the causal
lexicon alone is insufficient. Record active/passive valency and surface forms,
including past/participle ambiguity; do not pretend these entries are supplied by
VerbNet unless separately verified there.

Candidate requirements:

1. A licensed past/participle-compatible surface form in a bounded clause.
2. A recoverable affected nominal on the left or governing a reduced relative.
3. A directly attached by-phrase with a nominal complement.
4. No competing direct object, event complement, or unresolved clause attachment.
5. No identified temporal/deadline reading or unresolved competing by-role.

The finite VBD tag in the insomnia example is recorded as conflicting parser
evidence, not silently discarded. Preserve an active/elliptical alternative when
it remains viable. Absence of a temporal marker alone is not positive evidence
of an agent. A construction-supported candidate may still be provisional.

Temporal-role validation must apply to ordinary dependency-derived by-agents as
well as repaired candidates, since `Damage caused by noon` is already mislabelled
upstream. Use explicit temporal evidence with provenance and retain unresolved
cases. Event-valued `by cutting cables` remains an event attachment; cables must
not become the causal source.

For the target fragment, expose the proposed relation `stress -> insomnia`, with
`reduced_passive_hypothesis` provenance and a separate structural-resolution
status. `Yesterday` remains temporal context. Do not put stress in the event's
direct-object field. Reuse the existing CEM vector on the proposed pair to audit
its decision; this is hypothesis generation upstream, not a new learned feature.

Keep three outcomes distinct: proposed causal direction, textual assertion, and
eligibility for world commitment. A provisional repair must not become eligible
merely because CEM agrees. Preserve negation, modality, attribution and questions.
Do not call the regression fully fixed just because the expected arrow appears:
report separately whether the original assertion/commitment criteria are met.

## Acceptance fixtures

Freeze development cases before implementation, then reserve unseen noun/verb
combinations as held-out cases. Suggested contrast families:

- `Insomnia caused by stress yesterday.` and explicit `was caused` counterpart;
- `Flooding caused by rainfall overnight.` / `Damage caused by corrosion.`;
- `Rain caused flooding by noon.` / `Damage caused by noon.`;
- `The bat flew by the blind man.`;
- `Damage caused by cutting cables.`;
- `Damage not caused by corrosion.`;
- `Damage may be caused by corrosion.`;
- `Officials said damage was caused by corrosion.`;
- `Was damage caused by corrosion?`.

Require correct candidate roles and preservation of adjuncts, no broad by-based
passive flip, no new false commitments in temporal/path/negated/modal/attributed
cases, and no loss of P's intransitive or quantification fixes. Retain all P
regressions, including its two outstanding structural expected failures.

Deliver Q as a new script with a versioned audit schema, local construction
resource, tests, and documentation. Keep audit instrumentation and construction
repair separately switchable so their effects can be measured independently.
