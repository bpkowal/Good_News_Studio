# One live local repair: person-role consistency

**Partial success.** One fresh o3 response repaired the contradictory person-role bundle and survived native validation. This does not settle the laboratory decision or establish improved moral reasoning generally.

One successful provider request, 7.205 seconds, 1,960 prompt tokens and 657 completion tokens (including 384 reasoning tokens). The earlier sandbox attempt failed to connect; its separate failure record is retained. No second model response or automatic retry was used.

| Field | Before | Live proposal |
| --- | --- | --- |
| Means relation | FORESEEN_SIDE_EFFECT | FORESEEN_SIDE_EFFECT |
| Person role | FORESEEN_AFFECTED | FORESEEN_AFFECTED |
| Instrumental belief | RELIES_ON_PERSON | DOES_NOT_RELY |
| Instrumental use | USES_PERSON | DOES_NOT_USE |
| Means effects | E1 | E1 |

The explanation says opening the door aims at reaching the technician and treats exposure as a side effect. However, it also claims exposure is neither necessary nor contributory to rescue. The supplied graph does not establish that negative causal claim, and external causal necessity alone would not establish the actor's practical reasoning. Treat this as an attributed interpretation, not a verified intention or a proven absence of instrumental contribution. The empty `unresolved_questions` list overstates that remaining uncertainty.

The response stayed within the permitted fields. Python reopened priority, universalization and permissibility; the staged verdict is CONFLICTED, resolution CONTESTED, governing norm and priority UNRESOLVED. The ledger is STAGED_NONVOTING. Norms, the source world, original ledger and A1 remain unchanged. A1's pre-existing protected-party grounding warning remains.

## Bookkeeping defect exposed and corrected offline

Native recalibration omitted the old harm warning after the row became contested, although DOING_HARM was unchanged. The initial live artifact therefore incorrectly had no remaining cards. The app now carries that previously issued harm issue separately while the explicit native harm check still reports the conflict. It does not rewrite the native calibration result or infer a new moral verdict.

The exact saved response was revalidated after this correction with **zero additional API calls**. Its person-role correction still survives; the HARM_CLASSIFICATION card remains open. Twenty-two repair/native regression tests pass, including person-first and harm-first repair ordering.

Inspect [original request and provider response](live_probe.json) and [corrected offline revalidation and outstanding card](corrected_revalidation.json). The first artifact is preserved as executed, including the original bookkeeping defect.

The next substantive question is whether a focused follow-up can qualify the unsupported causal/intention inference. This run did not attempt that follow-up, repair harm classification, or choose an action.
