# Parliament saved-run viewer

From the repository root:

```bash
.venv/bin/python serve_parliament_viewer.py
```

Open **http://127.0.0.1:7860**. No Node, npm, API key or network connection is needed.
The server exposes only the static viewer directory on loopback. Ctrl+C stops it.

Medicine, promise and trolley are bundled saved runs. Select a run, inspect its
world/evidence, compare frameworks, examine recorded hinges (including
abstention boundaries when no ranking was admitted), or open topography
to see Care, welfare and obligation together. Click graph nodes and causal
relations for provenance. Topography plots recorded loci: hollow markers
are omitted axes, not zeros. After an offline Decisions pass, axes are
probabilities that each action satisfies the Care, welfare, or obligation
criterion given the Parliament packet; a Care ranking that copies the welfare
ranking is marked degenerate rather than an attractor. The 3D composite
interpolates a display-only surface when at least two actions have all three
coordinates; jackknife boxes stay recorded intervals. The record list supplies a keyboard-friendly
alternative. Source excerpts highlight only exact matches. Candidate trolley
graphs are never labeled as having been sent to agents.

The interface is read-only. No live execution or world selection is implemented.
Download a presentation JSON, or open one with the import control. Unknown versions
and invalid shapes show an error. No moral verdict is inferred from graph admission.

The **Thermal blanket contractualist** run uses four fresh native framework passes
and a fresh Contractualist argument in place of Rawls, all over the same accepted
blanket world. Open
`http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=frameworks`
to inspect its questions, conditional preference and original argument. The old
Rawlsian run remains separate. `CLARIFICATION` denotes a question-focused
contribution with no claimed ranking; Contractualist conditional preferences are
marked `ATTENUATED`, with no fabricated native ledger or coordinate. See
[the runner design](../CONTRACTUALIST_DEMO.md).

The catalog also includes **Thermal blanket promise**, a new live rescue allocation
plus promise case, and its stopped first attempt. Open
`http://127.0.0.1:7860/?run=thermal_blanket_promise` to inspect the frozen world,
five framework records and bounded review. Its FRAMEWORK_PARTIAL status is
intentional: unresolved arguments remain unresolved. See
[the run review](../diagnostics/thermal_blanket_end_to_end_20261023/RESULT.md)
for the saved-response continuation and remaining representation defects.
Additional safe catalog IDs now export through the same presentation contract.

Refresh the bundled three reports after generating updated saved artifacts.
`--refresh` and a bare export read `frontend/saved_runs.json`
(`parliament-saved-runs/1`); that catalog currently pins
`diagnostics/decision_topography_suite_20261009`.

```bash
.venv/bin/python serve_parliament_viewer.py --refresh
```

Or override the catalog with a report directory containing `medicine/`,
`promise/`, `trolley/`:

```bash
.venv/bin/python export_parliament_viewer.py --reports path/to/reports
```

See `FRONTEND_DESIGN.md` for the presentation contract and the next live-job increment.
Mermaid 10.9.3 is vendored under `dist/vendor`; its upstream license is MIT.

Framework cards preserve full readable explanations and link to complete Markdown
records. Assumptions distinguish model-reported premises, recorded interpretive
commitments, and the run's use of scenario givens. Count provenance is separate
from the stated count itself. Reversal descriptions name the current and alternative
actions and remain labeled as agent reports. Deontological summaries name the
decision-maker's duties toward protected people; they do not assign those duties
to the people at risk. Older reports are refreshed from their own matching native
records during export, without changing decisions or source files.

Restart the local server after updating `serve_parliament_viewer.py` so Markdown
links open as readable text in the browser rather than downloads.

The graph now uses the full content width and initially fits both viewport dimensions.
Compact links are the default: action–effect membership lines are hidden when a
recorded path within the same action already connects the effect from a direct
root. All nodes and causal relations remain visible; **Show full links** restores
all membership lines. This display reduction does not change the frozen world.

Purple **SOURCE LEDGER** nodes display saved claims introduced through source
text, outside the frozen topology. Dashed links labeled **ledger support reference**
use only support IDs already recorded in that ledger and present as world nodes.
Click a purple node to inspect the untouched ledger record and exact source text.
No promise-to-duty or promise-to-outcome relation is inferred by the display.
`--trolley-report` can still override only the trolley pin without rewriting the
catalog:

```bash
.venv/bin/python export_parliament_viewer.py --trolley-report diagnostics/trolley_scene_complete_live_20261008/DEMO_REPORT.json
```

Refresh plotted topography from the Decisions API (viewer remains a saved replay):

```bash
.venv/bin/python run_parliament_decisions_topography.py --report diagnostics/parliament_demo_20261007/reports/promise/DEMO_REPORT.json --write
```

When a saved topography includes jackknife 95% intervals, the scatter uses those
intervals (ten contiguous packet-text ablations) instead of a single point
estimate, and keeps every omitted tenth so the most hinging slice can be
inspected. Re-export from the catalog after writing the report:

```bash
.venv/bin/python run_parliament_decisions_topography.py --report diagnostics/decision_topography_suite_20261009/trolley/DEMO_REPORT.json --jackknife --write
.venv/bin/python export_parliament_viewer.py
```

## October 9 combined suite

The pinned catalog includes medicine, competing promise, and admitted trolley.
All three contain ten-fold packet-omission estimates and inspectable omitted slices.
Medicine and promise were evaluated live on October 9; trolley retains its previous
saved analysis. The 3D surface is display-only; 2D projections retain recorded points.

Run `.venv/bin/python serve_parliament_viewer.py --refresh`, open
http://127.0.0.1:7860, and choose **04 Topography**. Switching scenarios retains
that tab. Viewing or refreshing saved results makes no API calls.

Intervals describe sensitivity to omitting contiguous tenths of a saved packet.
They are not calibrated uncertainty about moral truth. The welfare predicate asks
about expected survivors rather than total welfare. The existing Care/welfare
“copy” flag identifies matching rankings, not proof that one assessment copied another.

See `diagnostics/decision_topography_suite_20261009/RESULT.md` for live results.

## Reviewed explanation prose

The current medicine and promise cards use reviewed plain-language summaries of
saved framework arguments, with readable uncertainty and reconsideration text.
Each is labeled as an editorial summary, not a new framework response. Original
agent fields, objections, and prior presentation text remain in the linked Markdown.
`frontend/reviewed_explanations.json` binds each summary to a hash of its source
scenario, world, argument, participation, preference, and ledger status. Changed
runs fall back to their own recorded explanations rather than reusing stale prose.
This presentation pass makes no API calls and does not alter topography inputs.

Compact diagrams now bundle matching action/direct-effect records into one box.
Select that box to inspect both original IDs; **Show full graph** restores the
separate boxes. Independent mechanisms, victims, probabilities, and ledger claims
are not merged. Ungoverned scene assertions appear as clickable context statements;
propositions referenced by modal/report/commitment records remain in the topology.

The **Version and run provenance** disclosure separates original generation, report replay/revisions, and viewer export versions. See [Run versioning](../RUN_VERSIONING.md). Historical generation versions remain explicitly unknown.

Receipt logic is isolated in `dist/provenance-receipts.mjs`. Run its offline tests with `node frontend/test_provenance_receipts.mjs` when Node is installed. The viewer loads finalized export checkpoints, labels served renderer mismatches separately, and keeps raw records collapsible.
