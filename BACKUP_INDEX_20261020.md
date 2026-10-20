# Where current work is backed up

Checked October 20, 2026. One GitHub repository contains the application and native Parliament histories: [bpkowal/Good_News_Studio](https://github.com/bpkowal/Good_News_Studio).

| Working component | GitHub location | Backup contents |
| --- | --- | --- |
| Parser, blueprints, frontend, Logic_Puzzles adapter/review code, tests and experiments | [Parliament_Frontend](https://github.com/bpkowal/Good_News_Studio/tree/Parliament_Frontend) | October 20 application checkpoint 8bac010 plus this backup index |
| Native Parliament source actually used by compact and RelEnt runtime profiles | [backup/relent-live-20261020](https://github.com/bpkowal/Good_News_Studio/tree/backup/relent-live-20261020) | Exact working-source snapshot, commit 96cf5c94ff1a379bbb4040ec446ab3998b0d78f6 |
| Historical performance/tooling branch | [perf/workspace-traces](https://github.com/bpkowal/Good_News_Studio/tree/perf/workspace-traces) | Existing local commit 35f74007eeef9cb4e0fcc5c0520fad1b7fc52a34, dated September 9, published by fast-forward from 76f2eeb |
| Older generated runtime reconstruction copy | [Saved patch](diagnostics/backup_audit_20261020/older_reconstruction_copy.patch) | Preserves its tracked changes and two new modules against native base 614af0c; labeled older verification copy, not the current live runtime |

## Why the branch names can look stale

Logic_Puzzles and question-before-world are earlier branch snapshots. Changes to files that originated there have been made on Parliament_Frontend, so those older branch references do not advance automatically. Their local heads match the GitHub heads inspected during this audit: Logic_Puzzles=5bf924c and question-before-world=f8d5a4c.

The live .runtime/parliament working tree remains on relent-framework at the pinned base 614af0c. Its 13 modified tracked files and two new modules are intentionally retained as local applied patches. They are backed up both as reproducible ordered patches in the application branch and as directly browsable native source on the dated backup branch. The isolated backup did not alter the live checkout's HEAD or the runtime lock.

The separate RAGAIMODEL checkout also points to relent-framework at 614af0c and has no modified tracked project source. Its perf/workspace-traces branch contained one older unpublished commit, 35f7400; that exact original commit was transferred through the isolated backup clone and fast-forwarded to the matching GitHub branch. Other named local development branches matched their remote tips during the audit. Its local-llama2 submodule reports only an untracked .DS_Store; model caches and operating-system metadata are excluded from application backups. Other discovered repositories are vendored MLC/TVM/llama.cpp dependencies, not active Parliament development branches.

This backup does not merge work into main, Logic_Puzzles, question-before-world or relent-framework. The perf/workspace-traces publication preserves its existing history; no force-push was used. Selecting a branch for future integration is a separate operation from preserving the working source.

## Verification

Every live native source-inventory entry was checked against the isolated backup before commit. The source matches pipeline.lock.json's patched tree checksum. The [audit](diagnostics/backup_audit_20261020/audit.json) records paths, hashes and commits. Native source SHA-256: 4f8e024e4b5d6316b78e65cc61ee3d722f0fc8cce7d25b56f33d72a978456842.

[Application checkpoint contents](CHECKPOINT_20261020.md) includes test results and experiment limitations. GitHub backup is not a claim that all experiments succeeded.
