# Parliament saved-run viewer

From the repository root:

```bash
.venv/bin/python serve_parliament_viewer.py
```

Open **http://127.0.0.1:7860**. No Node, npm, API key or network connection is needed.
The server exposes only the static viewer directory on loopback. Ctrl+C stops it.

Medicine, promise and trolley are bundled saved runs. Select a run, inspect its
world/evidence, compare frameworks, or examine recorded hinges. Click graph nodes
and causal relations for provenance. The record list supplies a keyboard-friendly
alternative. Source excerpts highlight only exact matches. Candidate trolley
graphs are never labeled as having been sent to agents.

The interface is read-only. No live execution or world selection is implemented.
Download a presentation JSON, or open one with the import control. Unknown versions
and invalid shapes show an error. No moral verdict is inferred from graph admission.

Refresh the bundled three reports after generating updated saved artifacts:

```bash
.venv/bin/python serve_parliament_viewer.py --refresh
```

Or export from a report directory containing `medicine/`, `promise/`, `trolley/`:

```bash
.venv/bin/python export_parliament_viewer.py --reports path/to/reports
```

See `FRONTEND_DESIGN.md` for the presentation contract and the next live-job increment.
Mermaid 10.9.3 is vendored under `dist/vendor`; its upstream license is MIT.

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
# Latest trolley saved run

Refresh the viewer with the October 8 admitted trolley while preserving the
historical reports:

```bash
.venv/bin/python export_parliament_viewer.py --trolley-report diagnostics/trolley_scene_complete_live_20261008/DEMO_REPORT.json
```
