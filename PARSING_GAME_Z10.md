# Z10: complete stripping alternatives and construction coverage

Use `parsing_game_Z10.export_candidate_graph(text, package_id=None)`.
Z10 prepares Z5 role hypotheses, separates identities with Z8, applies Z9's modal
scope questions, then runs Z6/Z7 condition completeness checks. Prior exporters
remain available. Schema 0.4 and consumer selection validation are unchanged.

## Reconstructed readings

Z8 already retained unchanged participants for simple subject, object and
prepositional recipient contrasts. Z10 fixes incomplete ambiguous bundles and
contrast remnants incorrectly included among the spoken clause's participants.

For “Lila gives Omar the medicine, but not Nora”, three provisional readings are
available, each with a separate proposition and its own participant bundle:

| Contrast | Subject | Object | Destination/recipient |
| --- | --- | --- | --- |
| Recipient | Lila | the medicine | Nora |
| Object | Lila | Nora | Omar |
| Subject | Nora | the medicine | Omar |

The spoken clause retains Lila, the medicine and Omar. Its participant selection
can coexist with any single reconstruction. Reconstruction anchors are mutually
exclusive, so mixing roles from different branches fails validation even through
their dependency closures. Antecedent provenance remains explicit. Partial
selections remain legal; exporting a bundle does not require selecting all roles.

The added subject alternative is bounded to bare proper-name remnants contrasting
with a proper-name subject. Prepositional remnants keep their destination reading.
These are unranked hypotheses, not guarantees of semantic plausibility or exhaustive
alternatives. General focus, pronoun contrast, ambiguous antecedents and omitted
adjuncts still need work. “Complete” here means retaining compatible represented
participants for each supported reading, not complete sentence understanding.

Z9's explicit `NOT MODAL(P)` versus `MODAL(NOT P)` question survives in each modal
reconstruction. The question blocks the reconstructed modal, predication and roles;
their unresolved polarity cannot be resolved merely by selecting them. Neither
operator order is silently chosen. Condition and attribution scope are retained.

## Construction-level evaluation

[Frozen fixtures](fixtures/ellipsis_semantic_coverage.json) contain 21 hand-authored
development probes, including negative controls and multiple expected readings.
They are not a held-out benchmark. `eval_ellipsis_coverage.py` evaluates the typed
reconstruction export, not the independent proposer's text suggestions or the
entire spoken-clause graph. Each reading must match predicate and all participant
roles one-to-one; duplicate or incomplete predictions count as false proposals.

| Metric | Z8 stripping | Z10 stripping |
| --- | --- | --- |
| Candidate recall | 7/12 | 11/12 |
| False proposals | 1/8 | 0/11 |
| Correct proposed roles | 17/18 | 26/26 |
| Scope preserved among structurally matched readings | 6/7 | 11/11 |
| Positive cases with no proposal | 1/9 | 1/9 |
| Proposals remaining unresolved | 8/8 | 11/11 |

The remaining stripping miss is additive “Susan works at night, and Bill too.”
VP ellipsis (3 expected readings), gapping (2), and sluicing (2) still have no typed
reconstruction export: recall is zero and no-proposal abstention is 100%.
Their scope/role accuracy is `null`, not a vacuous success. The report retains raw
denominators and separates unresolved proposals from absence of proposals.
All 21 packages pass contract validation, illustrating why that metric alone
cannot measure semantic coverage.

`ellipsis_coverage.py` replaces a generic sentence-gap notice with a specific
missing-content question when a bounded stripping, bare auxiliary, gapping or
wh-remnant cue is present. It names omitted predicate/roles, antecedent or question
content without fabricating a reconstruction. Untargeted generic notices and source
evidence remain; coverage stays partial. These cues can miss constructions or raise
false alarms outside the fixtures. All 16 positive fixture cases receive specific
questions, with zero such notices on the five controls.

Saved reports: [Z8 baseline](diagnostics/Z8_ellipsis_semantic_coverage.json) and
[Z10 results](diagnostics/Z10_ellipsis_semantic_coverage.json).

```sh
.venv/bin/python eval_ellipsis_coverage.py --exporter Z10 --output diagnostics/Z10_ellipsis_semantic_coverage.json
.venv/bin/python -m unittest test_parsing_game_Z9 test_parsing_game_Z10 test_ellipsis_coverage -v
```

Validation: 182 tests passed across the T–Z10, validator, reference, scope,
ellipsis and original-text alignment regression suites. New coverage includes
joint spoken/reconstructed selections, all three contrast branches, missing
dependencies, incompatible branch combinations, direct conditional consequences,
modal ambiguity, exact source offsets and the 21 whitespace/punctuation variants.
Metric tests deliberately inject wrong scope, duplicate readings and false
predicates to verify that the evaluator counts them as errors.
