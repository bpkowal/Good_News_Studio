# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: NONE.

## Readable explanation

The deontological agent asks what Ada owes each patient, rather than treating the larger survival chance as sufficient on its own. It argues that Ada should carry out her duty to rescue effectively: choosing Ben gives the treatment a much better chance of saving someone, without using Cara as a tool to save him. Its proposed argument calls treating Ben required and treating Cara prohibited. The review does not accept that duty classification, so those proposed verdicts never become an operative deontological judgment. The recorded preference is Ben, but the framework abstains.

Presentation basis: REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT.

## Assumptions and interpretive commitments

- Model reported assumption: Probability difference grounds rescue duty
- Recorded interpretive commitment: Prefer “Ada gives the medicine to Ben” when it secures far higher survival chance for one patient without using the other merely as a means.
- Recorded interpretive commitment: Duty to rescue requires choosing patient with better survival chance, not tallying aggregate welfare.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- The argument has not established why the duty to rescue makes one option obligatory and the other forbidden. The review also retains a challenge to the one-dose premise, although that restriction is stated in the source.

## Original presentation and objections

```json
{
  "reason": "Expected DEONTOLOGICAL_DUTY_LEDGER, but no matching committed ledger is operative. Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can be aided)? duty to rescue effectively. Prefer “Ada gives the medicine to Ben” when it secures far higher survival chance for one patient without using the other merely as a means. Recorded comparison — Ada gives the medicine to Ben: REQUIRED: fulfills duty to save when only one can be aided, non-arbitrary rule.; Ada gives the medicine to Cara: PROHIBITED: disregards effective rescue duty, arbitrarily sacrifices better chance. Premise rebound from PROP:WORLD:E7 to canonical PROP:HYPOTHESIS:67d6bcaf133579a0.",
  "reason_authority": "RECORDED_ARGUMENT_FIELDS",
  "live_objections": [
    "perfect-duty priority for A0 lacks a perfect-duty classification",
    "perfect-duty priority for A1 lacks a perfect-duty classification",
    "rejected update: perfect-duty priority for A0 lacks a perfect-duty classification",
    "rejected update: perfect-duty priority for A1 lacks a perfect-duty classification",
    "Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can be aided)?"
  ],
  "reversal_explanations": [
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: rescue duty no longer decisive. This is an agent-reported condition, not an independently established threshold.",
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Cara survival > Ben survival. This is an agent-reported condition, not an independently established threshold."
  ]
}
```

## Explanation provenance

```json
{
  "basis_sha256": "6bd9b309417f8eac8ad006421cbb5817eabb392c297a4f4cc2aea63f6fb8ffd4",
  "support_fields": [
    "recorded_argument",
    "position",
    "preference",
    "ledger_status",
    "live_objections",
    "decision_boundaries",
    "scenario",
    "world"
  ],
  "reviewer": "Codex editorial review of saved arguments; no new framework model call"
}
```

## Conditions for reconsidering

- The agent says it would reconsider treating Ben if Cara had the greater chance of survival. It also says its rescue-duty priority could change, but does not specify what would justify that change.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "stronger rescue duty",
  "framework_application": "NONE",
  "decision_rule": "Prefer A0 when it secures far higher survival chance for one patient without using the other merely as a means.",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "REQUIRED: fulfills duty to save when only one can be aided, non-arbitrary rule.",
    "Ada gives the medicine to Cara": "PROHIBITED: disregards effective rescue duty, arbitrarily sacrifices better chance."
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.76,
      "Ada gives the medicine to Cara": 0.24
    },
    "recommended_action": "Ada gives the medicine to Ben",
    "rationale": "duty to rescue effectively",
    "decision_rule": "prefer A0 when it offers far higher survival probability",
    "framework_action_map": {
      "Ada gives the medicine to Ben": "REQUIRED: universalizable maxim rescuing most likely life",
      "Ada gives the medicine to Cara": "PROHIBITED: violates imperfect duty of rescue by choosing far weaker chance"
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "greater survival probability fixes rescue duty",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "REQUIRED",
      "Ada gives the medicine to Cara": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "Ada gives the medicine to Ben",
    "factual_reversal_threshold": "Cara survival probability exceeds Ben’s",
    "normative_reversal_threshold": "rescued-life priority shifts to Cara",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "expected DEONTOLOGICAL_DUTY_LEDGER, but no matching committed ledger is operative",
  "investigative_claim": "Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can be aided)?",
  "committed_native_ledger": {},
  "unsupported_assumption": "Probability difference grounds rescue duty",
  "normative_reversal_threshold": "rescue duty no longer decisive",
  "factual_reversal_threshold": "Cara survival > Ben survival",
  "model_argument_fields": null
}
```
