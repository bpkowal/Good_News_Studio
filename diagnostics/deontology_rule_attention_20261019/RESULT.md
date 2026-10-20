# Bounded deontology rule-attention pilot

2026-10-19. Five live o3 adapter calls: control 1, rule context 2, saved-argument challenge 2. Both fresh arms used the same frozen laboratory world, core excerpt, 3072-token requested ceiling and two-call allowance. All [matched controls](live/controls.json) and the independently computed [call audit](call_audit.json) pass. The source world remained unchanged. No collective verdict or full RelEnt run was attempted.

## Result

The engineering changes worked: rule references can reach the prompt intact and the previously unadmitted deontology argument can receive native review. This sample does **not** establish that the extra initial rule instruction improves reasoning or admission.

| Run | Rule consideration and argument | Native outcome |
| --- | --- | --- |
| [Control](live/control/report/arguments/deontological.md) | Mentions safety compliance, but relies on the familiar perfect non-harm / imperfect rescue hierarchy. Authority and emergency exception are not examined. | COMMITTED_WITH_UNCERTAINTY; prefers waiting. Unsupported doing-harm classification and other warnings remain. |
| [Rule-context arm](live/rule_context/deontological/model_calls.json) | Makes containment the named duty and cites the actual N0/N1 contents. It still does not examine institutional legitimacy or the emergency exception. | SEMANTIC_VALIDATION_ERROR; no operative preference or ledger. The repair changed dt from IMPERFECT to PERFECT_NEGATIVE rather than defending another priority basis. |
| [Challenge to the October 16 saved reply](live/saved_argument_review/report/arguments/deontological.md) | Withdraws instrumental use of staff; records DOES_NOT_RELY and DOES_NOT_USE, and asks whether the laboratory rule is morally authoritative and whether uncertain risk meets the duty threshold. | COMMITTED_WITH_UNCERTAINTY; still prefers waiting. Adjudication remains CONTESTED, A1 priority UNRESOLVED, and a protected-party grounding warning remains. |

The third row is a separate recovery probe, not another matched treatment arm. It starts from the exact saved October 16 reply. A ledger with uncertainty is not an independently established moral conclusion.

## What improved, and what did not

The saved-reply challenge produced a substantive local correction: exposed staff are now foreseen affected parties rather than people the actor believes she must employ to rescue the technician. The answer also makes institutional authority and the risk threshold explicit open questions instead of silently treating them as settled.

However, the preferred action did not change, and its justification remains incomplete. The revised record still calls possible exposure DOING_HARM and describes the risk as grave without source support establishing severity. Its universalization argument assumes that emergency opening undermines predictable containment without examining a narrowly specified emergency maxim. Its means_effect_ids still include E2, the exposure outcome, even while it denies instrumental use. These are remaining argument issues; the two reconciled native warnings do not establish semantic resolution. The unresolved rule-authority and risk questions are the important result, not the model's RESOLVED label.

The first-prompt treatment improved source-rule identification but failed the harder applicability/authority/emergency analysis. One sample cannot distinguish stochastic variation from an instruction effect; it provides no evidence for dependable superiority. The control itself admitted better than the earlier October 16 reply, illustrating why admission alone is not a measure of improved reasoning.

## Engineering delivered

- Existing norm records are projected into a read-only prompt view with exact governed content, force, bearer, authority and source-clause references. Unknown bearer/reference identity remains unknown. Institutional moral authority is NOT_ASSESSED, not silently granted.
- Deontology receives a compact instruction to distinguish rule compliance, moral permissibility, emergency conflict and independent duties using existing response fields. No world or native response schema was expanded.
- Rejected submissions appear as PROPOSED_ARGUMENT nodes with world_state_authority=NONE. Their admission errors and authored maxims can be inspected and scheduled for review. They gain no world-support edges or operative status merely by being visible.
- An unadmitted prior reply is retained as context, not seeded as an operative ledger. Its new revision must pass the existing native admission path. Previously admitted arguments retain their recurrent review path.
- Generated candidates carry an informational source-rule-attention record marked NOT_INDEPENDENTLY_ASSESSED. It is not a semantic validator or an admission gate.

The original live review counted its retained-context file as one prior_replay_call. Bookkeeping was corrected; an [exact-response offline native replay](native_replay/AUDIT.json) reproduced the same committed ledger with zero new API calls and zero operative-prior replay calls. The original live records remain preserved.

## Verification

50 related tests ran: 48 passed, 2 skipped. See [regression.log](regression.log). Native recovery, retained uncertainty and frozen-world integrity were checked; the native validator and pinned runtime were unchanged. The successful saved-reply native result also reproduced exactly offline. Reports were rendered from saved data without API calls and include the world diagram.

## Suggested next bounded step

The failed initial arm suggests replacing some overloaded ranking work with a short source-rule assessment before ranking, within the same call budget, rather than adding more instructions to the existing large prompt. First preserve one inspectable answer for applicability, authority and emergency qualification; then ask the agent to derive a conditional ranking from it. Compare that with this retained baseline before wider adoption. No additional live experiment or transfer cases were run here.
