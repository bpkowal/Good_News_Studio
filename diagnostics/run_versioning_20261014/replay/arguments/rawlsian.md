# Rawlsian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

Decisive Rawlsian numerical comparison rests on mixed or unresolved positions. Would verifying this unestablished consequence reverse the admitted ranking: There is only one dose of medicine, so only one patient can benefit? Higher survival chance for all behind veil. prefer action with higher expected survival when worst-off equal. Recorded comparison — Ada gives the medicine to Ben: IMPROVES: gives one patient 95 % survival; least-advantaged no worse; Ada gives the medicine to Cara: WORSENS: uses scarce dose on 5 % case; least-advantaged no better

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Model reported assumption: utility-uncertainty audit not Rawls-decisive
- Recorded interpretive commitment: prefer action with higher expected survival when worst-off equal
- Recorded interpretive commitment: 95 % vs 5 % survival chance
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- decisive numbers are not tied to the affected subject or Rawlsian position
- Rawlsian preference lacks a stated difference in position or principle priority
- Rawlsian position for A0 conflicts with its framework map
- Rawlsian position for A1 conflicts with its framework map
- A0 IMPROVES lacked directional support; committed as UNCERTAIN
- A1 WORSENS lacked directional support; committed as UNCERTAIN
- rejected update: Rawlsian position for A0 conflicts with its framework map
- rejected update: Rawlsian position for A1 conflicts with its framework map
- rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN
- rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN
- rejected update: decisive numbers are not tied to the affected subject or Rawlsian position
- rejected update: Rawlsian preference lacks a stated difference in position or principle priority
- Would verifying this unestablished consequence reverse the admitted ranking: There is only one dose of medicine, so only one patient can benefit?

## Conditions for reconsidering

- The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: lexical liberty priority introduced. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Cara survival probability rises above Ben’s. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "better chance saves life",
  "framework_application": "NONE」「fr\":true,",
  "decision_rule": "prefer action with higher expected survival when worst-off equal",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "IMPROVES: gives one patient 95 % survival; least-advantaged no worse",
    "Ada gives the medicine to Cara": "WORSENS: uses scarce dose on 5 % case; least-advantaged no better"
  },
  "proposed_framework_state": {
    "constraint": "FAIRNESS",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.6950000000000001,
      "Ada gives the medicine to Cara": 0.305
    },
    "recommended_action": "Ada gives the medicine to Ben",
    "rationale": "Higher survival chance for all behind veil",
    "decision_rule": "prefer action whose expected survival is higher when worst-off equal",
    "framework_action_map": {
      "Ada gives the medicine to Ben": "MIXED: Improves Ben’s BASIC_INTEREST_SECURITY dramatically; Cara’s identical position worsens – least-advantaged unchanged but expectation higher",
      "Ada gives the medicine to Cara": "MIXED: Slightly improves Cara’s BASIC_INTEREST_SECURITY; Ben’s identical position worsens – least-advantaged unchanged and expectation lower"
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "95 % vs 5 % survival for the prospective recipient",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "PERMISSIBLE",
      "Ada gives the medicine to Cara": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "Ada gives the medicine to Ben",
    "factual_reversal_threshold": "Cara survival probability rises above Ben’s",
    "normative_reversal_threshold": "lexical liberty conflict arises",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {
      "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
      "ranking_classification_justification": "No liberty difference; compare life prospects as primary goods",
      "lexical_priority_justification": "Liberty stage satisfied, proceed to maximin over survival",
      "liberty_status": {
        "A0": "NOT_APPLICABLE",
        "A1": "NOT_APPLICABLE"
      },
      "positions": [
        {
          "action_id": "A0",
          "subject": "each patient behind veil",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "IMPROVES",
          "compared_to_action_id": "A1",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "higher expected chance to live",
          "principle_basis": "MAXIMIN_PRIMARY_GOODS"
        },
        {
          "action_id": "A1",
          "subject": "each patient behind veil",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "WORSENS",
          "compared_to_action_id": "A0",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "lower expected chance to live",
          "principle_basis": "MAXIMIN_PRIMARY_GOODS"
        }
      ]
    },
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "decisive Rawlsian numerical comparison rests on mixed or unresolved positions",
  "investigative_claim": "Would verifying this unestablished consequence reverse the admitted ranking: There is only one dose of medicine, so only one patient can benefit?",
  "committed_native_ledger": {
    "ledger_kind": "RAWLSIAN_POSITION_LEDGER",
    "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
    "records": [
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:ccaaa46814578e5c",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "compared_to_action_id": "A1",
        "effect": "UNCERTAIN",
        "proposed_effect": "IMPROVES",
        "subject": "possible patient",
        "proposed_subject": "possible patient",
        "dimension_subject_repaired": false,
        "affected_subject": "possible patient",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "UNCERTAIN",
        "group_node_id": "RAWLS_SUBJECT:a263a3a77528386e",
        "subject_node_id": "RAWLS_SUBJECT:a263a3a77528386e",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "ACTION_GRAPH",
        "group_binding_mode": "ACTION_GRAPH",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "UNSUPPORTED_DIRECTION",
        "reason": "higher expected survival",
        "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
        "ranking_classification_justification": "worst-off equal; move to expectation",
        "lexical_priority_justification": "no basic-liberty difference so secondary rule applies",
        "liberty_status": "NOT_APPLICABLE",
        "subject_selection_status": "BOUND_TO_ACTION",
        "group_selection_status": "BOUND_TO_ACTION",
        "provenance": [
          "delegate:rawlsian",
          "cycle:1"
        ]
      },
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:7e3be85dd2a3c918",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "compared_to_action_id": "A0",
        "effect": "UNCERTAIN",
        "proposed_effect": "WORSENS",
        "subject": "possible patient",
        "proposed_subject": "possible patient",
        "dimension_subject_repaired": false,
        "affected_subject": "possible patient",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "UNCERTAIN",
        "group_node_id": "RAWLS_SUBJECT:a263a3a77528386e",
        "subject_node_id": "RAWLS_SUBJECT:a263a3a77528386e",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "ACTION_GRAPH",
        "group_binding_mode": "ACTION_GRAPH",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "UNSUPPORTED_DIRECTION",
        "reason": "lower expected survival",
        "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
        "ranking_classification_justification": "worst-off equal; move to expectation",
        "lexical_priority_justification": "no basic-liberty difference so secondary rule applies",
        "liberty_status": "NOT_APPLICABLE",
        "subject_selection_status": "BOUND_TO_ACTION",
        "group_selection_status": "BOUND_TO_ACTION",
        "provenance": [
          "delegate:rawlsian",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "utility-uncertainty audit not Rawls-decisive",
  "normative_reversal_threshold": "lexical liberty priority introduced",
  "factual_reversal_threshold": "Cara survival probability rises above Ben’s",
  "model_argument_fields": null
}
```
