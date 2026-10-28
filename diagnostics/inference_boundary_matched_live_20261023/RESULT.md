# Rigorous boundary-card probe

**The card improves inspectability; this pilot does not establish a consistent reasoning advantage.** The strongest failures were in representation and support checking, not simply unwillingness to answer a follow-up.

## Live comparison

Sixteen fresh o3 requests: eight cases, ordinary and typed review, one sample per arm. No retries. Actual provider requests were checked for identical user prompts, model, reasoning effort and completion budget within every pair. Both arms received the same source, argument annotations and potentially mistaken reviewer objection. Only the output schema differed. Thus this is an equal-information format ablation, not a test of the full card against an unassisted model. Synthetic inputs are short and generous compared with native Parliament prompts.

Both formats generally corrected the targeted simple errors. Both also resisted four misleading demands: erase intended use when its mechanism fails, discard conditional maxim analysis, replace explicit intentional employment with UNKNOWN, and infer unconditional permission from consent. There were local differences in attribution and wording but no clear consistent substantive benefit from typed output. See the [non-blind content review](CONTENT_REVIEW.md); no total score or treatment-effect claim is made.

Typed output revealed new failures. In the consent and rule-priority cases, the reply withdrew the conclusion in prose but retained its old proposition under ATTRIBUTED_INFERENCE, with only the bridge explanation calling it unsupported. There is no explicit withdrawal lifecycle in the typed map. In the partial-graph case, the model used CONDITIONAL to express indeterminacy but supplied no antecedent, so its good prose accompanied an invalid typed graph.

| Check | Ordinary | Typed |
| --- | --- | --- |
| Provider responses returned | 8/8 | 8/8 |
| Raw base response contract valid | 3/8 | 6/8 |
| Raw complete format valid | 3/8 | 5/8 |
| Base contract valid after offline citation-alias replay | 8/8 | 8/8 |
| Complete format valid after citation-alias replay | 8/8 | 7/8 |

The failed base contracts used references such as `E1:/provenance/0/excerpt` where a bare evidence ID was expected. Those paths actually resolve. An offline diagnostic resolved them to the existing source ID while separately preserving their original path and value; it did not alter the saved replies or assess semantic support. No model call was repeated. These transport failures must not be counted as moral reasoning failures. The typed conditional/unknown error remains after citation resolution.

Identical user prompts do not mean identical provider prompt-token counts: the larger response schema contributes overhead. Ordinary used 7,570 prompt and 4,594 completion tokens; typed used 9,834 and 7,237. Mean request latencies were 4.974 and 6.355 seconds respectively. This sample does not demonstrate enough quality improvement to justify that overhead as a default.

## Adversarial diagnostics

Ten author-labeled stress/control cases exposed six gaps against the desired invariants:

1. Relabeling causal text as PURPOSE/BELIEF/PURPOSIVE_USE can silence the finding.
2. Negated practical-reasoning propositions still satisfy a check that counts claim kinds.
3. Conditional ancestry can be hidden through intermediate claims because scope checking is local.
4. A matching source quote does not establish qualifiers such as “sole.” This is a known semantic limitation, not a surprising string-check bug.
5. An UNKNOWN conclusion can receive an overbroad conditional-promotion question.
6. A malformed null claim text produces AttributeError instead of a controlled contract error.

The four controls—bad reference, fabricated quote, valid conditional analysis and valid mistaken-belief interpretation—behaved as expected. This deliberately adversarial collection is not an estimated failure rate in ordinary use. The diagnostic unit test passes by confirming the gaps remain exposed, not by treating the semantically bad cases as successful reasoning.

## Verification and limits

Fifty evaluation/boundary/protocol/history/repair/native tests and 24 existing follow-up/review/partial-review tests passed. No native source, runtime lock, admitted world or operative ledger was modified. These live tasks assessed argument revisions only; they were not submitted to native admission. The prior native characterization test still documents its physical-path restriction for intended-means classifications.

The review was performed non-blind by the assistant. There is one sample per condition, no independent annotator and no statistical estimate of superiority. Exact prompts, schemas, provider outputs and usage are retained. Existing eight fixtures were used, so this is not a held-out scenario benchmark; the content criteria were withheld from the model, but both arms had helpful author-written annotations.

Inspect [frozen design and controls](controls.json), [actual request matching](actual_matching_checks.json), [citation-resolution replay](citation_resolution_audit.json), [adversarial cases](adversarial_audit.json), and [provider summary](summary.json).

The proposed broader response is [argument construction and support tracking](../../ARGUMENT_SUPPORT_ARCHITECTURE.md), rather than another laboratory-specific card or more rounds on the same exhausted issue.
