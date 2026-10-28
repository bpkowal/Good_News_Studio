# A five-perspective demo with Contractualist questions

Status: the live blanket run is available in the viewer. See the
[run result and reasoning limitations](diagnostics/contractualist_frontend_20261026/RESULT.md).

`run_contractualist_demo.py` runs fresh independent responses from Utilitarian,
Deontological, Care and Virtue specialists plus the question-capable Contractualist
argument adapter, using one already admitted frozen world. Rawls is excluded from
this run; saved Rawlsian runs keep their original identities and remain available.

The first four use the existing native generation and admission path, capped at
two adapter calls each. The Contractualist adapter uses one o3 call with an
8192-token ceiling, no SDK retries or peer responses. It gets the same exact
scenario, admitted world and canonical action records, preserving world/action
IDs and trace hashes. It generates an attributed argument and critical questions,
with no fabricated native ledger, Rawls metric or operative vote. A conditional
recommendation remains possible; questions alone are a legitimate contribution.

The report and viewer expose the fifth perspective in the framework cards and
its critical questions among the decision hinges. `CLARIFICATION` means no action
ranking is claimed. It does not mean a technical failure or automatic abstention
by the other agents. The report first records the native Care, welfare and
obligation coordinates, retaining missing values. The saved live demo now also
has the existing Decisions probability pass and ten-fold packet-ablation
sensitivity ranges. Its three criteria are Care responsiveness, maximizing
expected survivors, and deontological permissibility. Native records remain
nested for inspection. Contractualist arguments, assumptions and questions enter
the assessment packet as attributed reasoning, with no Contractualist numeric
axis or fabricated native ledger.

Open the [interpolated 3D view](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=topography).
The existing Shepard interpolation (p=2) draws a display surface through the two
assessed action loci, alongside sensitivity boxes and the recorded 2D
projections. It adds no new actions, evidence or moral verdict. See the
[topography result](diagnostics/contractualist_frontend_20261026/topography_update/RESULT.md)
for the preserved inputs and checks.

This is the compact independent framework path, not full sequential RelEnt/RAG
deliberation or collective adjudication. It reuses the admitted world instead of
repeating construction. Existing world-representation defects remain inspectable.
Neither semantic completeness nor moral correctness follows from admission.

From the repository root, use the parser/report environment for the runner; it
launches native framework work in the separately pinned runtime:

```bash
.venv/bin/python run_contractualist_demo.py \
  --trace diagnostics/thermal_blanket_end_to_end_20261023/report_input/frozen_world_trace.json \
  --output diagnostics/contractualist_frontend_20261026 \
  --checkpoint-tag contractualist-front-end
```

Use a new output directory for each live run. Requests, provider responses,
technical findings, source/world hashes and run manifests remain available in
the output. The frozen trace and previous reports are not rewritten. Register
the new `report/DEMO_REPORT.json` in `frontend/saved_runs.json`, export with
`.venv/bin/python export_parliament_viewer.py`, and serve with
`.venv/bin/python serve_parliament_viewer.py`.

The viewer presents these as saved live outputs; viewing, exporting and browsing
make no new model calls. Agent identities and open questions survive export, and
the original generated argument remains available through its Markdown link.

To assess the figures for a new saved run, make twenty additional API requests
(two actions, ten ablations each), then export the viewer:

```bash
.venv/bin/python run_parliament_decisions_topography.py \
  --report diagnostics/contractualist_frontend_20261026/report/DEMO_REPORT.json \
  --jackknife --write --checkpoint-tag contractualist-interpolated-topography
.venv/bin/python export_parliament_viewer.py
```

The probability pass changes presentation topography only. It does not rerun
framework reasoning or grant missing native ledger commitments.
