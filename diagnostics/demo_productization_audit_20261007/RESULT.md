# Vertical-slice productization audit

The existing eight-source suite, one-command runner, optional direct-model baseline,
human report, and saved-run viewer remain the demo path. This pass adds no world
schema or philosophical machinery.

- All eight frozen sources passed the offline Z10/inventory and exact-span audit.
  See [suite table](SUITE_RESULTS.md). Admission and frameworks were NOT_RUN.
- The construction/admission regression ran 89 tests successfully, with four skips.
- The final report/viewer/trolley regression ran 35 tests successfully. The trolley
  check includes native admission of revised candidates from saved live answers.
- Admission failure reports now embed saved unaccepted graphs, record per-macro
  construction attempts, distinguish required gaps from optional unfilled slots,
  and retain native rejection reasons. No failed candidate becomes an agent world.
- A software failure after a successful freeze now fails the suite. Freeze success
  alone no longer masks a broken framework handoff.
- Replay accepts either a pipeline folder or its parent demo output folder.

Inspect the [original failed trolley proposals](../demo_productization_trolley_inspection_20261007/DEMO_REPORT.md).
That report replays the original failed run; it does not apply the revised compiler.
The regression separately verifies the revised construction. No fresh API calls
were made in this productization audit, and trolley remains outside the reliable
full-demo envelope. Track geometry, full semantic coverage, and argument quality
are not certified by native admission.

The next evidence needed is a small fresh full-path run on the GREEN allocation
fixtures, followed by content review of grounding, live objections and recorded
hinges. This audit does not establish an end-to-end success rate.
