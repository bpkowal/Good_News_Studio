# Viewer provenance and checkpoint tags

Implemented and verified without LLM API calls or changes to admitted worlds.

The viewer now separates original generation, report/replay, later report revisions,
and viewer export. It shows run IDs, source/commit fingerprints, uncommitted-work
status, inherited labels, and observed checkpoints. Raw records remain collapsible.
Historical provenance is explicitly unknown where it was never recorded.

Final browser QA showed **MATCHED** for the served medicine JSON and its finalized
export manifest, and **MATCHED** for served renderer assets, including the receipt
helper module. The panel listed the `viewer-checkpoints-v1` tag and four actual
export stages. Manifest downloads are restricted to the matching local run ID.
Missing verification support is nonblocking. Imported JSON cannot claim its own
trusted receipt result.

Checkpoint receipts retain per-stage artifact bytes and hashes, including before
later rewrites. Journals are append-only during execution; their hashes and contents
are verified against final manifests. Stage receipts and user labels are tracking
metadata, not ethical approval, Git tags, or resumable checkpoints.

## Verification

- [Broad regression](regression_final.log): **166 tests ran; 162 passed, four skipped**.
- [Receipt module tests](../../frontend/test_provenance_receipts.mjs): seven passed
  through the Node REPL with fake fetch responses and real SHA-256, zero network calls.
  Covered matching JSON, altered JSON, missing receipts, disallowed manifest paths,
  changed renderer assets, missing renderer provenance, and imported-receipt handling.
- [Viewer integrity and tag record](verification.json): valid.
- [Tagged saved replay](tagged_replay/DEMO_REPORT.md) and
  [manifest](tagged_replay/run_manifest.json): valid, with five stage receipts;
  [verification](replay_verification.json).
- Final browser inspection confirmed scenario, graph, provenance rows, matching
  receipt, matching renderer, download link, tag, and four export checkpoints.
- `git diff --check` passed.

Append `--checkpoint-tag demo-baseline-v1` to either runner or the viewer exporter.
Children inherit labels and can add their own. See
[checkpoint operating instructions](../../RUN_VERSIONING.md#checkpoint-labels-and-stage-receipts).

The browser verifies JSON and served renderer consistency only. It does not verify
source ZIPs or the original reasoning; the CLI verifier remains available for full
saved-artifact checks. Source archives, checkpoints, and worlds are not silently
rewritten to make a receipt pass.
