# Interpolated decision space

Open [the Topography tab](http://127.0.0.1:7860/?run=thermal_blanket_promise&view=topography).

The initial native plot lacked Care and welfare coordinates, so the existing renderer correctly omitted the surface. The established Decisions pass now supplies complete Care, expected-survivor and permissibility coordinates for both actions. Twenty live requests evaluated ten packet-text ablations per action; all ten folds returned usable values on all axes.

| Action | Care | Maximizes expected survivors | At least deontologically permissible |
| --- | --- | --- | --- |
| Give blanket to Rowan | 0.578 [0.314, 0.842] | 0.990 [0.904, 1.000] | 0.329 [0.000, 0.665] |
| Give blanket to Sasha | 0.418 [0.086, 0.750] | 0.000 [0.000, 0.000] | 0.663 [0.436, 0.890] |

Values are jackknife means and 95% sensitivity intervals for model probabilities of satisfying the named criterion given the Parliament packet. They are not survival chances or calibrated confidence in moral truth. The source survival chances remain 80% and 60%. Overlapping intervals on Care and obligation remain visible.

The existing renderer applies Shepard inverse-distance weighting (p=2) to the complete action loci and draws its 16×16 surface mesh. It retains the action markers, uncertainty boxes, axis projections and framework arrows. This is a display surface estimated from two action assessments; it does not create additional alternatives, facts, framework votes or an operative recommendation. The run remains FRAMEWORK_PARTIAL and its four review issues remain open.

The [before snapshot](native_topography_before.json), [input hashes](input_hashes.json) and [verification](verification.json) establish that the world and framework records did not change. In-place presentation provenance retains the original report snapshot. Thirty Decisions/viewer tests pass in [the regression log](regression.log).

The viewer now accepts `&view=topography` to open this tab directly. Reproduce the probability pass, making twenty new API requests:

```bash
.venv/bin/python run_parliament_decisions_topography.py \
  --report diagnostics/thermal_blanket_end_to_end_20261023/report/DEMO_REPORT.json \
  --jackknife --write
.venv/bin/python export_parliament_viewer.py
```
