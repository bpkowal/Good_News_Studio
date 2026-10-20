# Bounded duty-consistency correction

Offline replay, 2026-10-19. **Zero API calls.** All source-world and saved-response identity checks pass in [REPLAY_AUDIT.json](REPLAY_AUDIT.json). The admitted world remains unchanged.

| Saved submission | Before | After |
| --- | --- | --- |
| October 16 original | No duty ledger | COMMITTED_WITH_UNCERTAINTY; both action verdicts CONFLICTED |
| Rule-context reply, unchanged | Semantic validation rejection | Structurally VALID but no ledger: separate invalid display-map fields remain |
| Same rule-context reply with existing display-only closure | Semantic validation rejection | COMMITTED_WITH_UNCERTAINTY; changed duty classification retained as UNVERIFIED and CONFLICTED |
| Previously accepted control | COMMITTED_WITH_UNCERTAINTY | Identical native ledger |
| Previously accepted saved-argument review | COMMITTED_WITH_UNCERTAINTY | Identical native ledger |

The newly retained original and display-closed submissions have framework_vote_status=ABSTAIN and selection_status=PROVISIONAL. Recording a contested ledger does not authorize a categorical vote or establish its moral argument. The remaining raw display failure is visible, not silently fixed in the main comparison.

## Correction

The duty type classifies the primary norm. Perfect-duty priority is now calibrated against the identified governing norm. If the competing norm governs, the primary norm's classification cannot establish the competing norm's type. Without a separately represented competing classification, priority remains unresolved; the validator does not guess from prose.

A primary imperfect-duty / perfect-priority mismatch still triggers one bounded repair when the existing call allowance permits. Repeating the mismatch now reaches the native calibration layer as a contested interpretation rather than causing the entire duty ledger to be discarded. Invalid response shapes and missing required semantic fields still fail validation.

A model may revise a duty classification during repair. Python records the prior/revised types, both norm labels, its actual justification and priority explanation, and hashes of both native parser inputs. An optional internal ledger provenance field carries this audit. The revision is marked NOT_INDEPENDENTLY_ASSESSED / UNVERIFIED and calibrated to CONFLICTED pending independent review. Merely supplying an explanation does not certify it. No world-state or model-response fields were added.

In the repaired rule-context case, A1 changed from IMPERFECT to PERFECT_NEGATIVE. Both interpretations survive in the audit, while its priority becomes UNRESOLVED and resolution CONTESTED. The weak justification has not become an approved perfect duty. Other issues—foreseen risk versus settled harm, rule authority and emergency qualification—remain questions for reasoning review.

## Regression and reproducibility

- Native tests: 12 passed, including governing-norm binding, contested retention, revision provenance, candidate maxims and negation boundaries. [Log](native_regression_final.log).
- Application/runtime/review tests: 71 ran, 69 passed, 2 skipped. [Log](application_regression_final.log).
- Both previously accepted control ledgers match exactly before and after.
- [Runtime check](runtime_check.json): READY. The new appended patch reconstructs the exact locked native tree.
- Early combined test attempts used the wrong interpreter for two application tests (missing NumPy), then Python 3.9 for a native module requiring newer dataclasses. Their failure logs remain preserved; the final split runs above use the project's established native and application environments. No dependencies were installed to mask these environment errors.

The change is distributed as [duty_consistency.patch](../../integrations/parliament_z10/duty_consistency.patch) and pinned in [pipeline.lock.json](../../pipeline.lock.json). Existing patches remain in their original order; this one is appended. The shared native code is used by both compact and full RelEnt profiles, but no live RelEnt/API run was performed.

To replay these saved inputs again, choose a new output directory:

```bash
.runtime/venv/bin/python replay_duty_consistency.py \
  --output-dir diagnostics/duty_consistency_replay_new
```

[Before records](before_complete/comparison.json) · [After records](after_final/comparison.json). A read-only replay driver has no live backend. No verdict was selected to make the experiment look successful.
