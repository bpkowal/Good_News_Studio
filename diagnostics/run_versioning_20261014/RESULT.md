# Application versioning and lineage

Both pipeline paths now create immutable execution records and an application
source archive. Commit/branch, dirty working tree, exact source hashes, parent/child
run IDs, model identifiers, resolved configuration, launcher environment, pipeline
lock/runtime profile, input hashes, and output hashes are recorded. Startup records
survive an interrupted run; final records preserve success and failure status.

Report generation, historical replay, later Decisions topography revisions, and
viewer export are separate identities. Postprocessing retains its input report.
The viewer exposes a version/provenance disclosure. Old generation provenance stays
`UNKNOWN_HISTORICAL`; current code is not attributed retroactively to old agents.

The multi-profile lock now explicitly uses `parliament-runtime/2`, retaining
compact-only v1 compatibility. The compact profile needed one correction exposed
by the native regression: PyYAML 6.0.3 is required for authored CORE files. It is
now pinned and imported by preflight (18 compact distributions; 116 full RelEnt/RAG
distributions). RAG's larger dependencies remain isolated.

## Validation

- [Broad suite](regression_current.log): **164 tests ran; 160 passed, four skipped**.
- [Final presentation-summary checks](summary_metadata_regression.log): 49 passed.
- [Runtime-lock compatibility checks](lock_v2_regression.log): 12 passed.
- [Compact readiness](compact_runtime_final.json) and [full readiness](full_runtime_final.json):
  both ready against the same final lock hash.
- [Fresh saved medicine replay](replay_final/DEMO_REPORT.md),
  [its execution manifest](replay_final/run_manifest.json), and
  [integrity verification](replay_final_verification.json): valid; source did not
  change during the replay. Zero model calls.
- [Viewer export integrity](viewer_verification_final.json): valid, including
  exported JSON and argument files. Browser inspection confirmed the scenario,
  Mermaid graph, and “Version and run provenance” disclosure render successfully.
- The deliberately missing full runtime also retains a
  [failure manifest](full_preflight_failure/run_manifest.json), independently
  verified. It makes no model calls.
- `git diff --check` passed.

Initial broader tests failed on six obsolete `/tmp` imports. Those test references
now use the pinned runtime constants. Five immediately passed; the remaining one
exposed the compact PyYAML omission above. Earlier failing logs are retained rather
than rewritten. No semantic validation rules were relaxed.

No live LLM calls were made, and the work was not committed or pushed.

## Operating instructions

See [Run versioning](../../RUN_VERSIONING.md) for entrypoints, storage, scope,
limitations, and commands:

```bash
.venv/bin/python parliament_run_provenance.py verify path/to/run_manifest.json
.venv/bin/python parliament_run_provenance.py compare first/run_manifest.json second/run_manifest.json
```

Manifests verify saved bytes, not ethical correctness. Remote model identifiers can
remain aliases; archived responses document the actual run. Keep source ZIPs and
revision/input snapshots with their manifests. Historical missing provenance cannot
be reconstructed by labeling it with today's revision.
