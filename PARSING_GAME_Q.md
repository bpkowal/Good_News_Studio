# Parsing game Q: audit and provisional passive repair

Q leaves P unchanged. Its output schema is 6; the causal policy schema remains 2
with the same 11 numeric features and four actions.

```sh
python parsing_game_Q.py
python parsing_game_Q.py --sentence "Insomnia caused by stress yesterday."
python parsing_game_Q.py --no-repair --sentence "Insomnia caused by stress yesterday."
python -m unittest test_parsing_game_Q -v
python audit_parsing_game_Q.py --output diagnostics/Q_audit.json
python audit_parsing_game_Q.py --no-repair --output diagnostics/Q_audit_without_repair.json
```

## Passive construction hypothesis

The versioned local resource `resources/reduced_passive_Q.json` licenses forms
for cause/trigger/induce/produce. It is an application policy, not VerbNet data.
An unmarked passive proposal requires a supported surface form, a preceding
affected nominal, a directly attached nominal by-source, no direct object or
event complement, and no auxiliary or coordinated source. Explicit temporal
heads and spaCy DATE/TIME evidence exclude deadline objects. Negation and other
assertion cues remain separate from direction.

For the regression Q now proposes `stress -> Insomnia`. It records spaCy's original
VBD tag and dependency, its resource version, a provisional structural status,
and an unresolved active/elliptical alternative. It does not rewrite the Doc.
`Yesterday` is preserved as temporal attachment evidence. Stress does not become
the event's direct object.

The causal record retains affirmative local assertion wording, but its gate
includes `provisional_structure_requires_validation`. The enriched proposition's
occurrence is `unknown`. Thus direction/pair/assertion benchmark scores recover,
while world commitment remains blocked. This is a hypothesis-level repair, not
proof that the fragment has only one interpretation.

Temporal validation also applies to ordinary passive by-agents: `Damage caused
by noon` does not create a noon-causes-damage claim, even if spaCy labels by as
agent. `Rain caused flooding by noon` retains its ordinary causal pair.
Temporal detection is bounded and incomplete; unrecognized temporal language
and general word-sense ambiguity remain limitations.

## CEM contribution audit

The audit is a separate optional runner. Normal parsing does not run it.
`--no-repair` independently disables Q's passive and temporal-by policies. The
audit uses fixed seeds 7, 42, and 91 (overridable with `--seeds`) and trains only
on the existing training split. It reads no rolling probe file.

Each audit record includes arguments, original candidate provenance, exact
ArgumentContext, features, raw CEM scores/action/margin, deterministic baseline,
and both gated claims. The baseline passes through the same claim validator using
a fixed-action policy, so gate differences do not confound the comparison.
Margins are uncalibrated. Primary benchmark candidates have gold annotations;
additional discovered candidates are recorded separately as unreviewed.

Summary counts expose denominators for observation coverage, annotated pair
accuracy, conditional raw classification, final pair/action accuracy, abstention,
and false commitment. Existing suite end-to-end scores are included separately;
they do not measure contextual occurrence or verified world truth. False
commitment rates are null unless explicit eligibility annotations are supplied.

Diagnostic flags distinguish annotated argument mismatches (A), annotated
out-of-action-space relations (B), potential exact-vector collisions requiring
label/scope review (C), and raw policy errors needing feature review (D).
An unlabelled unresolved result is not automatically an error. Repeated gold
mentions without disambiguated offsets leave pair scoring unknown.

## Validation

The suite has 34 tests: 32 pass and two existing structural expected failures
remain (passive controller recovery and coordinated complement attachment).
It reuses P/N tests, tests the regression's proposed direction and blocked
commitment, and contrasts deadline, path, event-valued, negative, modal,
attributed, and questioned constructions. Added noun/verb combinations remain
outside training. Audit tests exercise deliberate policy errors, annotation
absence, argument mismatch, out-of-scope labels, and collision reporting.

On the six existing non-training single-pair suites, the three tested seeds each
produce 39/39 recovered pairs and 39/39 raw correct classifications for both CEM
and the deterministic baseline. Neither wins any comparison on these fixtures.
This does not establish generalization or incremental value from CEM. The
ordinary holdout's earlier 7/8 score returns to 8/8 for its original scoring
criteria; the provisional fragment still does not pass world commitment.

Q-prefixed dated diagnostics and `diagnostics/parsing_game_Q_user_probes.json`
preserve version separation. The rolling file retains only the latest five user
inputs; their content is not copied into audit/evaluation archives.
