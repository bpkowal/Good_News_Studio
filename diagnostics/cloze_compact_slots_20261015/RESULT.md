# Compact cloze slots

Offline increment; zero API calls. Historical frozen worlds remain unchanged.

The saved-answer replay preserves all accepted slots across 14 cases and 42
template attempts, with zero core-fill regressions. Four controlled overlong
answer probes demonstrate smaller actor/action/outcome and process/patient spans.
These probes do not estimate live generation quality or native admission rates.

Validation: 118 tests run, 114 passed and 4 skipped because their native test
environment variables were not configured. Exit status 0.

```bash
.venv/bin/python -m unittest test_blueprint_compact_slots test_blueprint_cloze_chooser test_blueprint_discourse test_blueprint_typed_slots test_trolley_cloze_recovery test_blueprint_kind_license test_blueprint_evidence_graph test_blueprint_world_beam test_blueprint_proposal_contract -q
```

The rescue regression now expects compact intervention labels and additionally
checks that both actions retain Maria's actor-party reference. Topology counts,
effect predicates and the visible missing-process construction problem remain
checked. No admission rule was relaxed.

See [the review](../../CLOZE_BLUEPRINT_REVIEW.md),
[saved replay](saved_completion_comparison.json), and
[controlled probes](controlled_span_comparison.json).
