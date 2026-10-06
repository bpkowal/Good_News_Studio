# Explicit uncertainty revision

Native recurrent review accepted two isolated changes from the saved rejected duty proposal:

| A0 field | Prior committed value | New committed value |
| --- | --- | --- |
| Harm relation | DOING_HARM | UNRESOLVED |
| Means relation | INTENDED_AS_MEANS | FORESEEN_SIDE_EFFECT |
| Verdict | CONFLICTED | CONFLICTED |
| Governing norm | UNRESOLVED | UNRESOLVED |
| Priority basis | UNRESOLVED | UNRESOLVED |
| Resolution | CONTESTED | CONTESTED |

Both addressed calibration objections are no longer reported. This is acceptance of a bounded classification revision, not independent semantic verification or a settled moral judgment. The full duty proposal remains `DISPUTED_NONOPERATIVE`; A1 stays PERMISSIBLE. The source/world grounding, other four frameworks and competing-duty promise reference remain unchanged. Anna and the promise have not been added as typed world nodes or relations.

The replay exposed why the committed baseline matters. The original raw response said PROHIBITED, PRIMARY and RESOLVED; native calibration had committed CONFLICTED, UNRESOLVED and CONTESTED. Reusing the raw values after removing the calibration warnings could restore the stronger verdict. The isolation step now explicitly renders the committed fields, records those differences, and submits the classification patches to native review. It changes no native validation rule.

The harm path only accepts an explicit proposed `UNRESOLVED` classification with explicit `CONTESTED` or `UNKNOWN` resolution. It records any resolution change as a companion dependent on the harm change. Tests reject affirmative harm substitutions and absent/incompatible resolution. Means-only acceptance remains covered separately.

Validation: **30 tests passed**, including native acceptance of the saved revision, means-only regression, preservation of committed duty fields and all other framework candidates, companion-resolution bookkeeping and unchanged source input. All execution used saved responses or deterministic native replay; **zero new API calls**.

Inspect the [graph](conflict_graph.md), [full field changes and remaining uncertainty](targeted_review.md), or [machine-readable review](targeted_review.json).

```bash
.venv/bin/python -m unittest \
  test_logic_puzzles_partial_review test_logic_puzzles_review \
  test_logic_puzzles_acceptance test_logic_puzzles_comparison \
  test_logic_puzzles test_parliament_source_constructions
```
