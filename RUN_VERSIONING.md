# Run versions and lineage

The multi-profile runtime lock is `parliament-runtime/2`; the reader retains compact-only v1 compatibility and rejects a multi-profile lock mislabeled as v1. Older readers reject v2 instead of silently checking the wrong profile.

Both pipeline paths now record `parliament-run/1` execution manifests. Git commits
identify reviewed history; source fingerprints and archives identify the actual
working code, including uncommitted and untracked inputs. A branch name alone is
never a run version.

## Records produced automatically

| Entry | Execution recorded |
| --- | --- |
| `run_parliament_demo.py` | Compact demo or saved replay |
| `run_blueprint_parliament.py` | Preparation or full RelEnt |
| `logic_puzzles_frameworks.py` | Independent framework generation |
| `logic_puzzles_review.py` | Targeted review |
| `run_parliament_decisions_topography.py` | Separate report revision, retaining its input |
| `export_parliament_viewer.py` | Separate viewer export, retaining its input reports |

A normal run directory contains:

- `run_started.json`: initial identity/configuration, written before model work.
- `application_source.zip`: the source/configuration bytes captured at startup.
- `run_manifest.json`: finalized status, resolved configuration, runtime check,
  launcher Python/distributions, input and output hashes, and source-change flag.

Parent and child runs have distinct IDs and explicit `parent_run_id` links.
The full native deliberation's files and resolved agent/model/RAG/cycle settings
belong to its blueprint-run manifest. Its internal native stages are not separate
versioned Python applications.

Postprocessing revisions use unique files under the report's `provenance/` directory.
They retain the original input report and a source archive. Viewer exports similarly
keep unique build records under the exported data's `provenance/` directory. Repeated
exports do not overwrite those records. Keep these folders with their artifacts;
Git metadata alone cannot recover uncommitted source snapshots.

## What the version identifies

The application record contains commit, branch, working-tree status, individual
source-file hashes, a combined fingerprint, and the archive hash. Source capture
covers tracked and untracked code/configuration/docs/fixtures and viewer assets;
it excludes runtime environments, diagnostics, generated viewer data, symlinks,
`.env` and explicitly named credential/secret files. Deleted source remains absent
from the snapshot and visible in working-tree status. This archive is an application
snapshot, not a full Git repository or Python-environment backup.

The manifest also records the pipeline-lock hash, separate runtime profile/check,
resolved model identifiers and run settings. Compact framework CLI runs preserve
copies and hashes of the external authored CORE files they read. Full RAG resources
remain governed by the runtime lock. Credentials/environment contents and exception
messages are not recorded by this metadata layer.

`source_changed_during_run` flags application edits while execution was underway.
The initial archive remains intact; that flag means a single initial snapshot cannot
certify every later import. A completed process does not mean admission, ethical
resolution, or semantic correctness: consult the report's pipeline status separately.

## Generation versus presentation

Reports distinguish `generation_provenance`, their own `run_provenance`, and later
`presentation_revisions`. A saved replay gets a new presentation identity, never a
newly invented generation identity. When a historical run lacks a manifest, its
original provenance is explicitly `UNKNOWN_HISTORICAL`.

The viewer's “Version and run provenance” disclosure preserves those distinctions
and adds the exporter/renderer asset hashes. Rendering an old result with new code
is recorded as a new export, not a new agent analysis. Historical reports are not
backfilled with today's commit.

## Inspect, verify, and compare

```bash
.venv/bin/python parliament_run_provenance.py verify path/to/run_manifest.json
.venv/bin/python parliament_run_provenance.py compare path/to/first/run_manifest.json path/to/second/run_manifest.json
```

Verification checks saved source archive bytes/inventory, input resource snapshots,
and recorded artifacts. Later authorized report edits will change an older record's
artifact hash; inspect the later revision and its retained input report. Comparison
separates application revision, runtime lock/check, launcher environment, inputs,
configuration, execution path, and replay/live mode. Unknown manifest versions are
rejected.

Hashes detect differences; they are not digital signatures. Model identifiers may
be service aliases, and recording their names/settings does not freeze remote model
weights or guarantee identical future generations. Saved responses and artifacts
are the evidence for what happened in a particular run.


## Checkpoint labels and stage receipts

Append a repeatable `--checkpoint-tag` label to either runner or a viewer export:

```bash
.venv/bin/python run_parliament_demo.py --scenario scenario.txt --output diagnostics/new_demo --checkpoint-tag demo-baseline-v1
.venv/bin/python export_parliament_viewer.py --checkpoint-tag viewer-baseline-v1
```

Labels allow 1–64 letters, digits, dots, underscores or hyphens, beginning with a
letter or digit. Children inherit parent labels and may add their own. Labels apply
to runs, not Git commits; they do not assert that an ethical conclusion is correct.
No Git tag or commit is created by these commands.

Each observed stage appends a `parliament-checkpoint/1` receipt to `checkpoints.jsonl`
(or a unique per-revision journal). Receipts include run ID, stage, status, timestamp,
labels, and hashes of artifacts captured at that moment. Snapshot bytes are retained
under `inputs/`; a later rewrite does not erase an earlier checkpoint's evidence.
Final manifests contain the checkpoint list and journal hash, and `verify` checks
both. Replay uses `saved_world_loaded`, never inventing a new `world_frozen` event.
A missing historical checkpoint stays missing. These are receipts, not resumable
execution snapshots or automatic approval gates.

The viewer separates original generation, report/replay, later report revisions,
and viewer export. It checks the saved JSON bytes against the finalized export
manifest, and checks served HTML/JS/CSS/Mermaid/helper files against recorded hashes.
It displays `MATCHED`, `MISMATCH`, or `UNAVAILABLE` for the JSON receipt, and separately
reports renderer differences. This checks consistency of files, not their authenticity,
source ZIPs, or the original agents' reasoning. Imported JSON cannot supply its own
trusted verification result. Missing receipt support does not prevent graph inspection.
