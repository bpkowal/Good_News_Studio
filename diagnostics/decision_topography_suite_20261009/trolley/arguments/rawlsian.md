# Rawlsian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED.

## Readable explanation

Decisive aggregation is not licensed by the committed Rawlsian ranking basis. Veil favors lower risk. prefer “pulls the lever” when it lowers the worst individual’s chance of death. Recorded comparison — “pulls the lever”: MIXED: lever saves five and distributes only a 1⁄6 death risk to each worker; “does not”: WORSENS: inaction leaves a 5⁄6 death risk on each worker

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer “pulls the lever” when it lowers the worst individual’s chance of death
- Recorded interpretive commitment: chance of death 1⁄6 vs 5⁄6
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- framework map for pulls the lever lacks framework-specific grounds
- framework map for does not lacks framework-specific grounds
- decisive numbers are not tied to the affected subject or Rawlsian position
- Rawlsian position for A0 conflicts with its framework map
- numeric reversal threshold lacks a scenario-grounded quantity

## Conditions for reconsidering

- The agent reports a switch from “pulls the lever” to “does not”. Condition: agency-based duty never to kill overrides risk gains. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Veil favors lower risk",
  "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"dcp\":[],\"ep\":[{\"c\":\"the trolley will kill that worker",
  "decision_rule": "prefer A0 when it lowers the worst individual’s chance of death",
  "landscape_cases": {
    "pulls the lever": "MIXED: lever saves five and distributes only a 1⁄6 death risk to each worker",
    "does not": "WORSENS: inaction leaves a 5⁄6 death risk on each worker"
  },
  "proposed_framework_state": {
    "constraint": "PUBLIC_RULE",
    "action_scores": {
      "pulls the lever": 0.6950000000000001,
      "does not": 0.305
    },
    "recommended_action": "pulls the lever",
    "rationale": "Veil favors lower risk",
    "decision_rule": "prefer A0 when it lowers the worst individual’s chance of death",
    "framework_action_map": {
      "pulls the lever": "MIXED: lowers death risk for workers but fatal for one",
      "does not": "WORSENS: maintains higher death risk for each worker"
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "chance of death 1⁄6 vs 5⁄6",
    "action_admissibility": {
      "pulls the lever": "PERMISSIBLE",
      "does not": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "pulls the lever",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "agency-based duty never to kill overrides risk gains",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "rule chosen under veil to minimise maximum death risk",
      "lexical_priority_justification": "no basic liberty conflict established",
      "liberty_status": {
        "A0": "NOT_APPLICABLE",
        "A1": "NOT_APPLICABLE"
      },
      "positions": [
        {
          "action_id": "A0",
          "subject": "each worker behind veil",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "IMPROVES",
          "compared_to_action_id": "A1",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "A0 lowers individual death risk",
          "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
        },
        {
          "action_id": "A1",
          "subject": "each worker behind veil",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "WORSENS",
          "compared_to_action_id": "A0",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "A1 leaves higher individual death risk",
          "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
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
  "framework_vote_reason": "decisive aggregation is not licensed by the committed Rawlsian ranking basis",
  "investigative_claim": "",
  "committed_native_ledger": {
    "ledger_kind": "RAWLSIAN_POSITION_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:97f0c4c20221b8d7",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "compared_to_action_id": "A1",
        "effect": "MIXED",
        "proposed_effect": "IMPROVES",
        "subject": "each worker behind veil",
        "proposed_subject": "each worker behind veil",
        "dimension_subject_repaired": false,
        "affected_subject": "each worker behind veil",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "MIXED",
        "group_node_id": "RAWLS_SUBJECT:057f6cc4da7a58f3",
        "subject_node_id": "RAWLS_SUBJECT:057f6cc4da7a58f3",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "LEXICAL",
        "group_binding_mode": "LEXICAL",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "MIXED_COMPARISON",
        "reason": "A0 lowers individual death risk",
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "rule chosen under veil to minimise maximum death risk",
        "lexical_priority_justification": "no basic liberty conflict established",
        "liberty_status": "NOT_APPLICABLE",
        "subject_selection_status": "BOUND_TO_ACTION",
        "group_selection_status": "BOUND_TO_ACTION",
        "provenance": [
          "delegate:rawlsian",
          "cycle:1"
        ]
      },
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:1c14ec0a550cdf37",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "compared_to_action_id": "A0",
        "effect": "MIXED",
        "proposed_effect": "WORSENS",
        "subject": "each worker behind veil",
        "proposed_subject": "each worker behind veil",
        "dimension_subject_repaired": false,
        "affected_subject": "each worker behind veil",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "MIXED",
        "group_node_id": "RAWLS_SUBJECT:057f6cc4da7a58f3",
        "subject_node_id": "RAWLS_SUBJECT:057f6cc4da7a58f3",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "LEXICAL",
        "group_binding_mode": "LEXICAL",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "MIXED_COMPARISON",
        "reason": "A1 leaves higher individual death risk",
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "rule chosen under veil to minimise maximum death risk",
        "lexical_priority_justification": "no basic liberty conflict established",
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
  "unsupported_assumption": "",
  "normative_reversal_threshold": "agency-based duty never to kill overrides risk gains",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Veil favors lower risk",
    "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"dcp\":[],\"ep\":[{\"c\":\"the trolley will kill that worker",
    "decision_rule": "prefer A0 when it lowers the worst individual’s chance of death",
    "landscape_cases": {
      "A0": "MIXED: lever saves five and distributes only a 1⁄6 death risk to each worker",
      "A1": "WORSENS: inaction leaves a 5⁄6 death risk on each worker"
    },
    "factual_reversal_threshold": "risk distribution equalises (e.g., 3 vs 3)",
    "normative_reversal_threshold": "agency-based duty never to kill overrides risk gains"
  }
}
```
