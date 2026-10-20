# October 20 backup checkpoint

Branch: Parliament_Frontend. This checkpoint preserves the October 15–19 working changes and their saved experiments.

Included work:

- Compact cloze fills and source-bound deontic rule bundles, including the laboratory choice/risk/duration composition.
- Deontology source-rule attention, retention and review of unadmitted arguments, and the matched rule-attention pilot.
- Governing-duty consistency calibration, audited classification revisions, and exact-response offline replays.
- The all-five-agent laboratory rerun, bounded review, frozen-world diagrams, user-facing reports and original model-call records.
- Tests, fixtures, design notes, runtime integration patch and updated pipeline lock.

Results remain mixed. Better retention/admission does not establish better moral reasoning; unresolved objections and failed proposals are preserved. The all-agent laboratory run used seven live adapter calls and did not produce a collective verdict. See diagnostics/lab_all_agents_20261019/RESULT.md.

The duty-consistency verification ran 12 native tests successfully and 71 application/runtime/review tests with 69 passing and two skipped. Before/after replays used zero API calls; two accepted native control ledgers remained exactly identical. See diagnostics/duty_consistency_20261019/RESULT.md. Earlier failed test invocations remain saved with their environment explanations.

## Reproducible native runtime

The generated .runtime/parliament checkout is not separately committed: its modifications are represented by the ordered integration patches and exact tree checksum in pipeline.lock.json. Source inspection at backup time confirms that it matches the locked patched tree. Dependency environments, credentials and local model caches are not included.

Native base commit: 614af0c1601dd7169842f130f1c26a5dd0b30564

Patched native tree SHA-256: 4f8e024e4b5d6316b78e65cc61ee3d722f0fc8cce7d25b56f33d72a978456842

The backup includes the saved provenance archives and failed attempts as well as successful outputs. This document records checkpoint contents; it is not a claim that every experiment succeeded.
