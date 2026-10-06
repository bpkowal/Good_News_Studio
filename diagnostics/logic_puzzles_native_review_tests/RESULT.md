# Revised native-review test

Two saved responses were tested against the prior native ledger. No new model calls were made. A requested live API run was not approved and did not execute.

| Saved revision | Native result | Operative means relation |
| --- | --- | --- |
| Means correction | Accepted with uncertainty | `FORESEEN_SIDE_EFFECT` |
| Conflicting duty/relation revision | Rejected; original retained | `INTENDED_AS_MEANS` |

Both runs replayed the **exact original committed ledger**, then reviewed in `PROPOSAL_REVIEW` with the assigned challenge and prior framework state present. Both preserved the shared world and all four untouched frameworks. Prior replay uses a saved response and does not spend a new API call. Native record and candidate differences are displayed in full.

In the accepted case, the intended-as-means objection is no longer reported. The doing-harm objection remains open, and the native calibration still reports unsupported decisive premises. The prior promise/Anna references disappear from the competing-duty fields; this is now visible in the record changes. Routing is working, but semantic preservation is not solved by admission.

The promise is explicitly in the exact source sentence: “Maria promised Anna to pull the lever.” In this frozen fixture, the typed world contains no promise relation and no Anna party node; its parties are Maria, one worker, the lever and five workers. The framework's promise-duty interpretation is an attributed normative assessment, not an admitted world edge. Earlier explanations blurred those layers. Removing a framework reference is not the same as removing a typed promise edge from the world. Source evidence, typed world content and framework interpretation must be evaluated separately.

Inspect the accepted case: [graph](means_correction/conflict_graph.md), [full review and field changes](means_correction/targeted_review.md).

Inspect the rejected case: [graph](conflicting_duty/conflict_graph.md), [full review](conflicting_duty/targeted_review.md).

Machine-readable checks and operative records: [summary](summary.json).
