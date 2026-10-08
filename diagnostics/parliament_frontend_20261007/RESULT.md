# Saved-run frontend verification

Implemented on `Parliament_Frontend`: a local read-only viewer and [design document](../../FRONTEND_DESIGN.md).

Start from the repository root:

```bash
.venv/bin/python serve_parliament_viewer.py
```

Open http://127.0.0.1:7860. The bundled runs are medicine, promise, and trolley. Use `--refresh` to export the current saved demo reports again.

## Checks

- 24 tests pass: `test_parliament_viewer` and `test_parliament_demo`. See [test log](tests.log).
- `git diff --check` passes.
- Chrome renders the locally bundled Mermaid diagrams. The medicine frozen world and the trolley candidate graphs were inspected in the browser.
- Graph record inspection highlights exact source excerpts. Candidate worlds retain their "not sent to agents" labels.
- Framework participation and native judgment are displayed separately. The decision-hinge panel retains originating record paths and uncertainty.
- Browser console inspection returned no errors or warnings during the checked session.

The viewer/exporter does not modify saved reports, invoke models, load credentials, or select a world. Tests check source/world preservation, abstention preservation, beam handling, version rejection, local assets, and omission of local execution paths from the presentation bundle.

## Limits

This is a saved-run inspection interface, not a live pipeline controller. Import accepts the `parliament-view/1` presentation format, rather than arbitrary native trace files. Browser file import/download and mobile layouts have not received end-to-end verification in this increment. The existing graph renderer is reused; this frontend does not correct semantic mistakes in saved worlds.

## Graph readability follow-up

The frozen medicine graph repeats action wording on direct effect records and
adds membership lines alongside downstream relation paths. Action/effect IDs
remain distinct. The compact display hides six redundant membership lines in
medicine, retaining all nodes and recorded relation edges. Full links can be
restored. Effects with no same-action path from a direct effect remain connected.

The topology now gets the full content width. Initial Fit considers both width
and height, and can enlarge a small graph instead of capping scale at 1. Browser
measurement confirmed a 480 × 150 diagram within a 524 × 428 viewport at initial
scale. This verifies fit, not a readability guarantee on every screen.

26 tests pass after this change, including preservation of original worlds and
same-action isolation for compact display. No accepted world was rewritten.

## Source-ledger overlay

The two frozen allocation worlds are identical, but the promise report includes
a separate SOURCE_TEXT ledger entry. The viewer now displays that entry as a
purple SOURCE LEDGER addition with support-reference links to Ada and Cara.
No world record or semantic relation was added. Browser inspection confirmed the
purple fill and exact promise-sentence highlighting on selection. Medicine has no
extra source-text node; trolley candidate diagrams do not receive ledger overlays.
28 tests pass, including separate-layer preservation and candidate isolation.
