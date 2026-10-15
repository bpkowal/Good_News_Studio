# Parliament source located on GitHub

Checked the seven repositories returned for `bpkowal` and all nine branches of
`bpkowal/Good_News_Studio` on 2026-10-14. The other six repositories were last
pushed in 2016–2017. No newer Parliament repository appeared in that inventory.

The newest branch containing all eleven missing source files is
[`relent-framework`](https://github.com/bpkowal/Good_News_Studio/tree/relent-framework),
at `614af0c1601dd7169842f130f1c26a5dd0b30564` (2026-09-22).
Repository-tree queries were not truncated.

| Branch | Tip date | Missing-package coverage |
| --- | --- | --- |
| Logic_Puzzles | 2026-10-06 | None of the eleven native files |
| question-before-world | 2026-10-05 | None of the eleven native files |
| relent-framework | 2026-09-22 | All eleven |
| correctness-tooling | 2026-09-12 | All eleven, older branch |
| perf/workspace-traces | 2026-09-04 | Nine; lacks frozen replay and quote pack |
| global-workspace/parliament-V2 | 2026-08-27 | Seven; older implementation |
| deploy/render | 2025-10-23 | None of the eleven |
| alpha-version | 2025-09-16 | None of the eleven |
| main | 2025-07-20 | None of the eleven |

Latest file-change commits within the selected native branch:

| File | Last change |
| --- | --- |
| global_workspace/action_identity.py | 614af0c, 2026-09-22 |
| global_workspace/frozen_world_replay.py | 614af0c, 2026-09-22 |
| global_workspace/local_specialists.py | 614af0c, 2026-09-22 |
| global_workspace/world_state.py | 614af0c, 2026-09-22 |
| global_workspace/engine.py | 614af0c, 2026-09-22 |
| global_workspace/epistemic_ledger.py | 614af0c, 2026-09-22 |
| global_workspace/scenario_semantics.py | 614af0c, 2026-09-22 |
| global_workspace_pipeline.py | 614af0c, 2026-09-22 |
| global_workspace/openai_backend.py | c4f4f615d2ee5d0c207498fa3767e6a9a0c25912, 2026-09-10 |
| global_workspace/structured_io.py | ea6ed808a1ebb547c133a093dbe503db75c1197a, 2026-09-09 |
| global_workspace/core_quote_pack.py | 35f74007eeef9cb4e0fcc5c0520fad1b7fc52a34, 2026-09-09 |

The runners' deleted temporary checkout used this same base commit. Its later
integration work is retained as local patches in `integrations/parliament_z10/`:
grounding_hook, discourse_workspace, recognition_gap, argument_quality_fixes,
scene_recognition, and framework_presentation. A fresh GitHub checkout alone is
not equivalent to the previously patched runtime. Restore the pinned base as a
complete package, inspect/apply the required patches in their appropriate order,
and check imports/native regressions before changing the runtime path. Patch
compatibility has not been retested during this read-only GitHub search.

No GitHub repositories, local branch pointers, or runner defaults were changed.
