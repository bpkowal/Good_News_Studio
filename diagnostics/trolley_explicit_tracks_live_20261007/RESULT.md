# Corrected trolley scenario: live pipeline run

Input: [explicit trolley and tracks](../../demo_scenarios/trolley_explicit_tracks.txt).
The previous demo was preserved. This run used the existing live blueprint chooser
(`gpt-4o-mini`) and native Parliament admission. Framework generation was not
invoked because no candidate passed admission.

Result: **ADMISSION_FAILED**.

- `rescue_contrast` and `omission_harm` were withheld for missing required slots.
- `conditional_outcome` produced a graph, but native admission rejected E3/E6
  marked POSSIBLE without conditions and their conditional links without condition
  references. The source snippets attached to those effects omitted the if clauses.
- The action labels include whole condition-and-outcome clauses. The graph types
  the trolley as PERSON and does not represent either track as a location node.
  These are semantic construction defects beyond the reported admission errors.
- Primitive composition was withheld. No frozen world or moral assessment exists.

Inspect the [failed candidate graph](pipeline/candidate_graphs.md#conditional_outcome-candidate-graph),
[admission manifest](pipeline/preparation_manifest.json), and
[generated report](DEMO_REPORT.md). Nothing was patched or selected manually to
force admission, and the original viewer demo was not replaced.
