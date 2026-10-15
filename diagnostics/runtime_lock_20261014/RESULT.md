# Runtime lock and preflight: first increment

Implemented `pipeline.lock.json`, `parliament_runtime.py`, a persistent ignored
`.runtime/parliament` checkout, and `.runtime/venv` (Python 3.11). The source is
the exact `614af0c1601dd7169842f130f1c26a5dd0b30564` base plus six ordered,
hash-checked integration patches. Installed direct/transitive package versions
are pinned in `runtime-requirements.lock` and checked before native imports.

The historical discourse patch did not apply cleanly as saved. A canonical patch
with the same additions was produced from unique exact original contexts; the
historical file remains unchanged. A second clean base checkout with all six
patches reproduces the final source hash. No native validator was weakened.

## Verification

- [Current runtime check](runtime_check.json): ready, `LOCKED_PATCH_SET`.
- [Independent reconstruction](reconstruction_check.json): same patched source hash.
- [Idempotent setup](idempotent_setup_check.json): matching runtime preserved.
- [Targeted suite](targeted_tests_final.log): 105 tests ran; 101 passed, four skipped.
  Includes real trolley native admission using the persistent runtime.
- [Final runtime/report checks](runtime_report_tests_final.log): 34 passed after
  explicit runtime-failure classification/report linkage was added.
- [Missing-runtime smoke report](missing_runtime_smoke/DEMO_REPORT.md): software
  failure at `runtime_setup`, zero child commands, inspectable runtime JSON.
- `git diff --check` passes. No model/provider calls were made.

The earlier [broader suite](regression.log) ran 109 tests: 104 passed, four skipped,
one failed. Its existing seven-effect composition-selection test still selects
six effects. That semantic selection failure is not marked resolved by restoring
the runtime, and the entire broader suite is not claimed green.

## Boundary

The locked capability is compact native admission plus independent framework
generation, used by `run_parliament_demo.py`. The alternate full RelEnt/original
agent/RAG path is explicitly unsupported until its optional dependency/resource
profile is pinned; it is reported before chooser calls. Saved-report replay stays
offline and does not require the runtime. Run manifests and artifact lineage are
the next increment; this work does not yet capture every parser code change,
prompt, corpus revision, or package wheel hash.

See [operating instructions](../../PARLIAMENT_RUNTIME.md). No existing experiment
worlds or Git branch pointers were rewritten; runtime copies are ignored by Git.
