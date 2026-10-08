# Bounded graph comparison and selection

Implemented the three proposed changes without changing Parliament's world-state
schema or weakening native admission.

1. The trolley regression uses the runner's `supported_core` projection. Six
   supported effects must pass native admission; AV3/AV6 remain in the overlay
   and must still fail admission when submitted as established effects.
2. Semantic comparison considers participants, roles, quantities, effects,
   modalities, conditions and links. Blueprint names and record IDs do not
   determine equivalence. Identical actions no longer hide different outcomes.
   Comparisons show only changed records alongside complete candidate views.
3. An unresolved beam triggers one source-based review of at most three distinct
   admitted interpretations. The reviewer can select an existing proposal or
   abstain, recording missing content and unresolved questions. Every difference
   involving a selected candidate requires rationale and exact original-clause
   evidence. No response can rewrite a candidate or promote its overlay. The
   selection remains attributed model interpretation, not semantic proof.

The runner records all admitted comparisons, review input/response and source
citations. All candidate graphs and the separate admission/selection outcomes
remain visible. Source-backed amendments can still pass and be selected: the
native composition test now exercises the targeted selection path for the
explicitly added warning-signal state rather than treating same actions as
semantic invariance.

**Final regression: 129 distinct tests passed** (134 executions, including five
accidentally repeated Logic_Puzzles tests), no failures or skips. See
[final log](regression_final.log). The initial suite exposed the composition
test's old same-actions selection assumption; the updated test retains all its
native admission assertions and explicitly tests supported amendment selection.

## Live trolley result and limit

Ran the exact trolley fixture with the current OpenAI chooser and targeted review,
requesting one Parliament cycle if selection succeeded. Six supported candidates
passed admission. The reviewer abstained, so **Parliament deliberation did not
run**. One additional review of the saved candidates, with clarified instructions
distinguishing missing content from contradiction, also abstained. This acceptance
target remains unmet; no candidate was forced through selection.

The differences are concrete: the omission candidate records the one-worker
count but lacks an ability annotation; the conditional candidate omits that count;
an ability amendment additionally types the lever as a person. These differences
are retained in the [comparison](live/focused_comparison_final.json), rather than
hidden by identical action labels. See the [live review](live/selection_review.json)
and [refined saved-candidate review](live/selection_review_refined.json).

[All candidate graphs](live/candidate_graphs.md) include the supported cores and
AV3/AV6 hypothesis overlays. The
[manifest](live/preparation_manifest.json) records admission and NOT_RUN
deliberation separately. No moral judgment was produced by this live test.

Reproduce with the existing one-command runner:

```bash
./run_blueprint_parliament.sh \
  --scenario-file diagnostics/trolley_exact_fixture_live_20261006/scenario.txt \
  --output-dir diagnostics/bounded_graph_selection_20261006/repeat \
  --no-rag --max-cycles 1
```
