# Invariants for inference repair

Design review, October 21, 2026. Based on saved laboratory responses and current code. This document proposes the next implementation; it does not change native validation, world schemas or operative ledgers. No new API calls were made for this review.

Implementation update, October 23: the first bounded increment now implements typed argument annotations, exact support locations and the causal/practical-reasoning boundary card, with eight contrasting fixtures. See [the offline cards](diagnostics/inference_boundaries_20261023/RESULT.md). Native path validation is deliberately unchanged; its behavior is captured in a characterization test. The remaining implementation sequence below still proposes a later separation of physical-path validation from candidate-maxim support.

## Diagnosis

The evidence does not identify a temperature problem or establish that the model is generally too generative or deterministic. It shows a more specific failure: the model changes fields when challenged, but repeatedly supplies an insufficient bridge to preserve its preferred interpretation. The first inference review withdrew definite non-reliance/use. The second retained those unknowns, yet again favored a side-effect account from absent evidence and suggested causal facilitation could establish intended use. This is responsiveness plus interpretive anchoring, not literal determinism.

There is an engineering contribution as well. We ask a model for mutually consistent enum values, then primarily validate consistency, IDs and topology. The explanatory bridge remains mostly prose. A reply can therefore pass the same checks while reproducing the inference error in new wording. Requiring more open questions does not by itself create a better argument.

The main native prompt already explains the practical-reasoning distinction: an end, belief that employing a person can further it, and employing that person for that purpose. It explicitly says causal paths do not establish instrumental belief. Missing instruction is not the whole explanation. See `global_workspace/local_specialists.py`, the candidate-maxim instructions around lines 5558–5580.

The native validator nevertheless requires `_means_path_support` for every `INTENDED_AS_MEANS` classification, even though maxims are documented as candidate interpretations. Its helper describes an intermediate causal burden as instrumental, and the generated question asks whether a path makes the burden instrumental to the end. See `global_workspace/deontology_ledger.py`, `_means_path_support`, `calibrate_deontological_adjudication`, and the unresolved-question generation around line 1000. These checks are useful for physical support; their wording and scope risk conflating actual causal efficacy with attributed practical reasoning. A mistaken-belief fixture must distinguish those cases before changing the rule.

The local PERSON_ROLE_CONSISTENCY card catches contradictory values. It does not establish support for a *consistent* causal-to-intention inference. The generic inference protocol can challenge that inference, but has no dedicated representation of the bridge being crossed. The missing capability is a reusable semantic boundary check, not another scenario template or a larger philosophical prompt.

## Correction to an earlier content review

The first review said E1 could not support the rescue purpose because its outcome is only door-opening. That was too categorical. E1's provenance excerpt contains “Mira can open the door now to reach the technician.” A citation can refer to that source wording if its supporting path is identified. It does not establish that rescue is the *sole* purpose, a complete account of actual intention, or the causal irrelevance of exposure. Citation review must inspect both represented content and provenance text, and distinguish what each supports. Reviewers need these invariants too.

## General invariants

| Invariant | What it permits | Unsupported transition to challenge |
| --- | --- | --- |
| Preserve the claim's kind | Separate source text, actual causal structure, attributed belief, candidate maxim and normative judgment | A physical path silently becomes a psychological or moral fact |
| Bind support to the precise claim | Cite the evidence/provenance field supporting the particular predicate and qualifiers | A valid evidence ID certifies the entire sentence, including “only,” certainty or purpose |
| Preserve epistemic strength | A missing representation leaves a question open; an explicit closed-world scope may justify a bounded negative | Missing edge becomes no contribution, no intention, or no competing duty |
| Preserve interpretation scope | Evaluate a proposed maxim conditionally without claiming it describes the actor's actual psychology | A useful candidate reading becomes the established or uniquely correct reading |
| Expose the bridge | State the premise or assumption licensing each cross-kind inference | “Therefore” hides a new belief, exclusivity claim or duty priority |
| Distinguish instrumental use from efficacy | An actor may believe a means works even if it fails; a causal contribution may occur without purposive use | Causal effectiveness proves intended use, or lack of effectiveness proves no intended use |
| Separate intended use from moral prohibition | Analyze consent, treatment as an end and the applicable normative argument separately | Using someone as a means automatically becomes using them merely as a means or a prohibited act |
| Reopen dependent conclusions only | Withdrawal or qualification changes the support status of dependent claims; independent claims survive | Repaired fields silently retain a conclusion whose premise was withdrawn, or invalidate unrelated reasoning |
| Distinguish correction from agreement | A reviewer may be mistaken; a defended original claim can survive | Compliance, native admission or the model's own disposition counts as semantic success |

These are representation and support obligations, not a complete ethical theory. Physical efficacy, intention, consent and permissibility can legitimately interact; the inference must state the additional premises rather than collapse them.

## First specialized card: causal role versus intended use

Use the existing inference-review/history mechanism with a reusable rule ID such as `CAUSAL_ROLE_TO_INTENDED_USE`. It remains an investigative objection until its bridge is assessed. Do not auto-detect a philosophical error from a word such as “means.”

The card should expose five separate items:

1. The physical claim: what process, person or burden contributes to what end, with modality and exact support.
2. The practical-reasoning claim: what end the actor seeks, what employing the person is believed to achieve, and whether the person is employed for that purpose.
3. The mode of interpretation: source-stated practical reasoning, attributed inference, conditional candidate maxim, or unknown. Source-stated reasoning is still a scenario stipulation, not external psychological verification.
4. The bridge and its support: which facts or declared assumptions connect the physical account to the practical-reasoning account.
5. The dependent conclusion and available repair: defend the bridge, retain a conditional interpretation, withdraw it, or leave the relevant claim unresolved.

Do not assume an effect on a person is equivalent to employing that person. Identify the person, the operation, the person's contribution and the agent's proposed reason separately.

Supported conditional output could read: “Under a maxim of rescue by opening the door, without reliance on staff exposure, exposure is a side effect within that proposed maxim. The scenario does not establish that this is Mira's complete actual maxim.” This is useful reasoning. It is neither a recovered mental fact nor a permissibility verdict.

The card must also accept a supported contrary answer. If the scenario explicitly stipulates that the actor employs a person for an end, the system should not force UNKNOWN simply because causal efficacy is absent or the reviewer doubts the maxim.

## Small implementation sequence

1. Freeze contrasting saved/synthetic fixtures before modifying the native checker. Include positive support and mistaken-reviewer controls, not just invalid arguments.
2. Extend the argument-review packet with claim kind, interpretation mode and explicit bridge dependencies. Keep those in the argument layer; no new world-state schema is needed. Preserve provenance paths such as `/effects/0/provenance/0/excerpt` alongside effect predicates.
3. Add the causal-role/intended-use card as an instance of the generic cross-kind inference review. Reuse the same history, edit scope and two-round budget. The card should request a revised argument, not merely another disclaimer.
4. Separate the native check for an asserted physical path from consistency of the proposed maxim. An unsupported physical assertion stays unsupported; a hypothetical or mistaken belief must not manufacture that path in the world. Do not simply remove all path checks.
5. Compare supported corrections, surviving unsupported bridges and conditional argument usefulness separately from native admission. Retain unresolved limitations without forcing every interpretation into abstention.

### Minimum contrasting fixtures

| Construction | Required distinction |
| --- | --- |
| A burden causally helps, while the source explicitly says the actor does not know of it | Physical contribution does not establish a belief or purposive use |
| The actor believes employing a person will help and acts for that purpose, but the mechanism fails | Attributed intended use is distinguishable from actual efficacy |
| A rescue maxim is proposed but not stipulated as the actor's actual policy | Conditional maxim analysis remains available without psychological promotion |
| No causal role is represented in a partial graph | Missing support does not establish a negative physical or mental claim |
| The source explicitly states purposeful employment of a person | A mistaken reviewer cannot force unsupported uncertainty |
| A person is employed as a means with stated consent | Intended use does not automatically establish mere-means treatment or prohibition |
| A source sentence supplies a rescue purpose but no exclusivity | Provenance supports purpose, not “solely” or a complete maxim |
| A posted rule and an emergency duty conflict | Institutional wording does not by itself establish moral priority; the same bridge-review mechanics apply |

The first implementation should concentrate on the causal/practical-reasoning distinction and provenance precision. The rule-priority fixture tests generality; it does not require a new automated normative authority checker in the same increment.

## Evaluation limits

Current content assessments are non-blind assistant reviews of a small number of outputs. They show concrete recurring failures and an earlier reviewer overstatement, not a measured diagnosis of model internals. The proposed invariants can make reasoning inspectable and catch some contradictions deterministically. Semantic support and normative justification still require assessment; matching types and filling bridge fields do not prove the bridge valid.
