# Choice plus conditional outcome: live pipeline test

Run on 2026-10-04 using the OpenAI chooser and full RelEnt deliberation.
Input: `fixtures/primitive_composition_signal_scenario.txt`.

```bash
./run_blueprint_parliament.sh \
  --scenario-file fixtures/primitive_composition_signal_scenario.txt \
  --output-dir diagnostics/primitive_composition_live_signal
```

The live chooser ranked omission_harm, conditional_outcome, and rescue_contrast.
Both original omission/conditional proposals admitted six effects. Their amended
proposals, the flexible candidate, and primitive composition admitted seven
effects including the neutral warning-signal stopping outcome. Rescue_contrast
was withheld for missing slots. Every attempt is in `candidate_graphs.md`.

The runner selected the amended omission proposal by its unchanged ranking.
It committed two actions, seven effects and five links with no repairs. Full
Parliament ran all five original agents for three cycles, with no RAG retrieval
and zero world-generation calls. The pipeline status is COMPLETED; the final
judgment and selected action are UNRESOLVED. Pulling the lever is the plurality
preference, not a governing recommendation.

The composition matches the amended conditional candidate's topology and effect
readings. It matches the amended omission candidate's effect readings but differs
in party quantity metadata: omission records `one` on the worker node, whereas
composition records it on the death effect only. The comparison retains this
difference rather than forcing agreement.

The deliberation raises a separate limitation: agents introduce details about a
trolley, track switching, guardian roles, and sole ability that the four-sentence
input does not establish. Audit/ledger uncertainty remains, and some answer
verification labels are more confident than the supplied evidence warrants.
No new mechanism or event was added to the frozen input graph to settle those
arguments. This run establishes live construction, admission and handoff; it does
not establish trustworthy final ethical adjudication or general composition.

Artifacts:

- `preparation_manifest.json`: admission and final deliberation status.
- `primitive_inventory.json`: extracted construction evidence and scope.
- `primitive_comparison.json`: candidate differences.
- `world_state_topology.md`: selected admitted graph.
- `parliament/deliberation_report.md`: complete deliberation report.
- `parliament/workspace_scenario_20261004_085108_answer.txt`: final answer.
