# Combined decision topography — October 9, 2026

Medicine and competing promise now have the same ten-fold packet-omission analysis
as trolley. Both live runs completed: 20 API requests each, 40 new requests total.
Every action/axis cell has 10/10 probability responses. Trolley retains its saved
analysis with no new API requests.

The default catalog bundles all three reports. The viewer shows their 3D display
surfaces, recorded loci and interval boxes, three 2D projections, framework arrows,
and the exact packet slices whose omission changed estimates most.

## New results

Values are jackknife means with 95% intervals. “Welfare” here asks whether the action
maximizes expected survivors; it does not measure total welfare.

| Scenario / recipient | Care | Expected-survivor criterion | Permissibility |
| --- | --- | --- | --- |
| Medicine / Ben | .701 [.613, .789] | .964 [.888, 1.000] | .662 [.308, 1.000] |
| Medicine / Cara | .071 [.000, .227] | .000 [.000, .000] | .158 [.000, .463] |
| Promise / Ben | .541 [.265, .817] | 1.000 [1.000, 1.000] | .411 [.054, .768] |
| Promise / Cara | .268 [.033, .503] | .000 [.000, .000] | .526 [.390, .662] |

Medicine separates on Care and the survivor criterion, but permissibility intervals
overlap. Promise separates only on the survivor criterion: Care and permissibility
intervals overlap. These estimates measure sensitivity to packet omissions; they
do not establish moral certainty or resolve native objections. A zero-width interval
means these ten omissions produced no variation. The existing Care/welfare
“copy” flag identifies matching rankings, not proof of copying.

## Input consistency and validation

Medicine's preparation manifest now resolves to its original five-framework native
trace instead of a three-framework fallback. Promise retains its three recorded
frameworks. Source scenarios and frozen worlds match the original reports; new
analyses were written to separate suite copies. Trolley's historical topography
packet remains intact.

52 relevant topography, viewer, and demo regression tests pass; see
[regression.log](regression.log). Browser inspection confirmed all three scenarios
load, medicine and promise expose their new intervals and omitted text, and each
scenario renders its 3D surface and recorded 2D projections.

## Open all three

```bash
cd "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms"
.venv/bin/python serve_parliament_viewer.py --refresh
```

Open http://127.0.0.1:7860, select a scenario, and choose **04 Topography**.
This command refreshes saved exports and serves the viewer without API calls.
If a viewer is already running in another terminal, stop it with Ctrl+C before
starting another on the same port.

Reports: [medicine](medicine/DEMO_REPORT.md), [promise](promise/DEMO_REPORT.md),
[trolley](trolley/DEMO_REPORT.md). Source paths and hashes are in
[source_manifest.json](source_manifest.json).
