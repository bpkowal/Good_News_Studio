# Exact trolley fixture: live run

Ran the scenario copied directly from `TROLLEY` in
`test_blueprint_cloze_chooser.py`, using the live OpenAI chooser and the current
native Parliament admission pipeline. No code or validation rules were changed.

**The supported omission graph passed admission. The unfiltered graph still fails
the same AV3/AV6 checks. Full deliberation did not run.**

| Submission | Result |
| --- | --- |
| Live omission proposal including AV3/AV6 as world effects | REJECTED: unestablished assumptions and POLARITY_INVERTED transformations |
| Runner's supported omission core, six effects/four links | COMMITTED |
| AV3/AV6 | Retained in a separate hypothesis overlay |
| Entire candidate search | Six admitted attempts, one rejected conditional-outcome baseline |
| Selection and deliberation | UNRESOLVED_BEAM; Parliament deliberation NOT_RUN |

The decisive distinction is the submission path, not simply a better model
response. The live proposal also generated the averted-harm hypotheses. The
runner submits the supported core, whereas the failing unit test sends the
unfiltered candidate directly to `parse_world_model`. Replaying the unfiltered
live proposal reproduces the exact AV3/AV6 rejection:
[raw admission check](raw_live_candidate_admission.json).

View [all candidate graphs](run/candidate_graphs.md), including the original
omission graph, hypothesis overlay and supported core. The
[preparation manifest](run/preparation_manifest.json) records admission results;
the [world beam](run/world_beam.json) preserves competing interpretations.

This run shows that the current runner can admit a supported graph for this exact
scenario. It does not make the unfiltered admission test pass or establish a
completed moral judgment. The remaining pipeline obstacle is selecting among
non-equivalent admitted interpretations.

Reproduce:

```bash
./run_blueprint_parliament.sh \
  --scenario-file diagnostics/trolley_exact_fixture_live_20261006/scenario.txt \
  --output-dir diagnostics/trolley_exact_fixture_live_20261006/repeat \
  --no-rag
```
