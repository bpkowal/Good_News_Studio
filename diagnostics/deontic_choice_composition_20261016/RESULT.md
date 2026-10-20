# Rule + choice/outcome composition

Final live result: **ADMITTED**, selected variant **amended** of `deontic_rule`.
Native status COMMITTED; frozen occurrence replay valid. Two actions, five
effects, three links and two norms. Exposure and death remain POSSIBLE and
ADVERSE. Ten-minute duration is retained. No framework calls were made.

The scenario text is unchanged from the original laboratory test.

- [Final generated Mermaid](live_final/world_state_topology.md)
- [All final candidate attempts](live_final/candidate_graphs.md)
- [Native admission and selection manifest](live_final/preparation_manifest.json)
- [Final attributed source review](live_final/selection_review.json)
- [Earlier inaccurate safety rationale](live_prepare/selection_review.json)
- [First rejected construction](probe/candidate_graphs.md)
- [Intermediate process/duration construction](process_duration/candidate_graphs.md)

Nine focused tests pass, including native integration:

```bash
.venv/bin/python -m unittest test_deontic_choice_composition -q
```

The broader run records 120 tests: 115 passed, four skipped and one failed.
An additional known failing trolley selection test was excluded from that batch
and reproduced separately. Both failures reproduce with the unchanged checkpoint
cloze module and the same native runtime. This is not a clean full-suite pass.
See [regression log](regression.log), [counts](regression.json),
[checkpoint reproduction](checkpoint_regression.json), and
[checkpoint log](checkpoint_regression.log).

The final focused run includes the whitespace and adverse-exposure corrections
made after that broader batch. No world schema, native validator, runtime patch,
or selection scoring rule changed. Admission does not establish a moral verdict
or complete semantic coverage.
