# Two-round inference review: live result

**Partial improvement, with diminishing returns.** The remaining live follow-up preserved the earlier correction and made its interpretation more explicitly provisional. It did not adequately repair the disputed bridge. The history now reports STOP_ROUND_LIMIT with the gap still visible.

| Item | First inference follow-up | Second inference follow-up |
| --- | --- | --- |
| Instrumental belief/use | UNRESOLVED | UNRESOLVED |
| Side-effect account | Preferred reading, insufficiently defended | Explicit provisional interpretive default, still insufficiently defended |
| New problem | Absence used to favor a reading | Causal facilitation suggested as sufficient for intended use |
| Native validation | Passed, nonvoting | Passed, nonvoting |
| Assistant content assessment | PARTIAL | PARTIAL |

The second reply explicitly says its classification is not a proven fact about Mira's psychology. That is useful qualification. But it still treats the absence of a depicted instrumental role as support for the preferred account. It also asks whether exposure expedites rescue, “which would reclassify it as an intended means.” Helping cause an outcome does not by itself establish the actor's belief and purposive use. The word “solely” in its explanation also overreads what the purpose clause establishes.

A coherent candidate rescue maxim can still be evaluated conditionally. The system need not establish Mira's actual mental state before doing any ethical reasoning. This reply did not fully articulate that conditional argument, nor settle permissibility or duty priority.

## Engineering behavior

One fresh o3 request, 21.541 seconds, 3,097 prompt tokens and 1,622 completion tokens (including 960 reasoning tokens). It received the stored first-round corrections, remaining gap and argument dependencies. No scenario regeneration, peer debate or additional repair request occurred.

Native revalidation retained the local fields. The world, original ledger and A1 remain unchanged. The staged ledger is STAGED_NONVOTING, verdict CONFLICTED, resolution CONTESTED, governing norm and priority UNRESOLVED. The separate HARM_CLASSIFICATION card and A1 party-grounding warning remain open.

The response and its content review are separate hash-linked events. A new unsupported causal-to-intention claim has its own identity; it was not added to the world. The remaining gap survives budget exhaustion. An offline check confirmed that requesting a third round fails before any provider call. Twenty-six history/protocol/repair tests pass.

Inspect the [raw live request, response and native audit](../deontology_inference_round2_live_20261021/live_probe.json), [reviewed history](inference_history.json), [dependency summary](summary.json), and [invariance/budget checks](verification.json).

## What this supports

The history mechanism worked, and one follow-up corrected an overconfident claim. A second follow-up mainly improved qualification while retaining a closely related inference error. This single sequence, assessed non-blind by the assistant, does not establish a general benefit from more rounds.

The next useful experiment is a small cross-case comparison with a fixed rubric and mistaken-reviewer controls. Additional rounds on this case should not be the default response. No further live test was run here.
