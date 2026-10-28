# Interpolated 3D figures for the Contractualist demo

Open [Topography in the viewer](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=topography).

The existing Decisions pass supplied the previously missing plot coordinates.
Twenty live requests assessed ten packet-text ablations for each action; all
ten folds returned usable values on every axis. The existing renderer now draws
its 16×16 Shepard surface (p=2), retaining action markers, sensitivity boxes,
native-framework arrows and three 2D projections. The projections remain recorded
points rather than additional interpolated observations.

| Action | Care responsiveness | Maximizes expected survivors | Deontologically permissible |
| --- | --- | --- | --- |
| A0: Give blanket to Rowan | 0.609 [0.426, 0.792] | 0.982 [0.822, 1.000] | 0.384 [0.036, 0.732] |
| A1: Give blanket to Sasha | 0.343 [0.000, 0.725] | 0.000 [0.000, 0.000] | 0.531 [0.333, 0.729] |

Values are means and 95% packet-ablation sensitivity intervals for model
probabilities of satisfying the named criterion. They are not survival chances,
total welfare or confidence in moral truth. Source survival chances remain 80%
and 60%. Care and permissibility intervals overlap. No attractor is established;
the coincident Care/welfare rankings retain their existing degeneracy warning.
The surface is a display interpolation from two action assessments, not a new
set of feasible actions or a resolved collective judgment.

The assessment packet now preserves argument-adapter critical questions,
assumptions, dependencies, original argument and technical findings. This avoids
reducing the Contractualist response to its provisional preference. It adds no
Contractualist numeric axis, native ledger or native-framework arrow. Earlier
Rawlsian runs remain separately available.

Verification confirms that only `topography` and `presentation_revisions` changed
in the report. The source, frozen world, all five agent records, open questions,
and judgment status are unchanged; original native coordinates remain nested
under each action. The viewer exports the updated figures without model calls.

Thirty-nine Decisions, demo-adapter and viewer regression tests passed. The
headless browser rendered one visible surface containing 256 finite polygons;
the localhost payload and all five original argument links also passed checks.

Inspect the [browser screenshot](browser.png), [verification](verification.json),
[offline verifier](verify.py), [before report](report_before.json),
[input hashes](input_hashes.json), [exact assessment packet](assessment_packet.json)
and [live pass log](live_api.log). Run manifests and presentation revisions retain
execution provenance. Recheck without API calls:

```bash
.venv/bin/python diagnostics/contractualist_frontend_20261026/topography_update/verify.py
```
