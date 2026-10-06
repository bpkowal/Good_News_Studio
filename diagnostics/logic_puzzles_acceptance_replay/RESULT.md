# What differs between the acceptance paths

The identical saved targeted response was replayed four ways, with no API calls and no changes to native ledger or continuity validators. The decisive comparison is between the two recurrent paths with the same prior state and the same assigned challenge.

| Path | Result for the proposed means correction |
| --- | --- |
| Fresh targeted workspace, then reconciliation | Accepted with uncertainty |
| Recurrent workspace, assigned challenge, `OPEN_DELIBERATION` | Rejected; prior ledger preserved |
| Recurrent workspace, assigned challenge, `PROPOSAL_REVIEW` | Accepted with uncertainty |

The native specialist's `_audit_framework_state_change` returns immediately when `previous_framework_state` is empty. That is what happens in fresh targeted review. The bridge's `reconcile` then checks ownership, schema validity, native commitment and matching challenge ID; it does not run continuity against the original framework state.

When prior state exists, the native audit considers typed state changes, a framework-relevant explanation and review context. Its `review_supplied_reason` requires `broadcast.constraint != "OPEN_DELIBERATION"`, along with the reasoning-effect/challenge and explanation checks. In this saved response those other checks pass. Changing only the procedural constraint from `OPEN_DELIBERATION` to `PROPOSAL_REVIEW` makes the revision pass. [Open-mode audit](recurrent_assigned_open/continuity_audit.json) and [review-mode audit](recurrent_assigned_review/continuity_audit.json) preserve the exact inputs and decisions.

The fourth path used the naturally generated ordinary agenda. That agenda addressed `CHALLENGE:5654544c698b66dd`, while the unchanged saved targeted response answered `CHALLENGE:3067e134dbd3e7c9`. It entered native repair and reached the replay call limit before completing cycle two. It therefore retained the opening ledger without reaching the second continuity audit. This is an address mismatch, not another continuity rejection. The [cycle inputs](recurrent_ordinary/cycle_inputs.json) and [denied repair](recurrent_ordinary/denied_repair.json) make it visible. No repair response was generated.

All worlds and original source artifacts remained unchanged. This is an isolated one-owner experiment: recurrent quorum is set to one so the owner can reach cycle two, rather than the five-framework pilot's minimum of two valid specialists. Ledger validation remains native. Every path received exactly the same raw revision; its fingerprint is recorded in [results](acceptance_paths.json).

Acceptance still does not mean the replacement is fully supported. Both accepting paths retain the doing-harm calibration objection. Their replacements also omit Anna and the promise from the competing-duty field. The complete before/after field differences are recorded, not only the corrected means relation. Marking a review does not protect untouched content from a whole-record rewrite.

The next implementation should route targeted revisions through an explicit native review against the prior committed state, rather than treating them as fresh opening assessments. It should present the proposed field changes and retained commitments as an inspectable update. The existing native review route can admit this correction without weakening validation; the promise loss remains a separate preservation problem to resolve. No production acceptance policy was changed in this investigation.

All four graphs are linked from [acceptance paths](acceptance_paths.md).
