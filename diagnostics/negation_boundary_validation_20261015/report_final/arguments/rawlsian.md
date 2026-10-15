# Rawlsian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED.

## Readable explanation

Decisive Rawlsian numerical comparison rests on mixed or unresolved positions. fewer workers die. prefer the action that leaves the least-advantaged group (those at risk of death) best off. Recorded comparison — “pulls the lever”: IMPROVES: by acting, only one worker dies instead of ten, raising the primary good of basic security for the worst-off group.; “does not pull the lever”: WORSENS: inaction lets ten workers die, leaving the worst-off far worse in basic security.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer the action that leaves the least-advantaged group (those at risk of death) best off
- Recorded interpretive commitment: death count among workers (1 vs 10)
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; ten workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- decisive numbers are not tied to the affected subject or Rawlsian position

## Conditions for reconsidering

- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: intentional-killing ban given lexical liberty priority. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: side track contains ten or more workers. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "fewer workers die",
  "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"dcp\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"ep\":[{\"c\":\"the tram will kill one worker\",\"p\":\"PROP:WORLD:E3\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E3\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"it will kill ten workers\",\"p\":\"PROP:WORLD:E7\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E7\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: worst-off face one death instead of ten, so primary good of life is better distributed\",\"A1\":\"WORSENS: worst-off face ten deaths, markedly worse distribution of the primary good of life\"},\"nr\":\"DECISIVE\",\"np\":\"death count among workers (1 vs 10)\",\"rp\":{\"A0\":{\"g\":\"track workers\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"only one rather than ten are killed\"},\"A1\":{\"g\":\"track workers\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"ten rather than one are killed\"}},\"rb\":\"MAXIMIN_PRIMARY_GOODS\",\"rbc\":\"life is a fundamental primary good; liberty conflicts are absent\",\"lpj\":\"lexical liberty priority not triggered—no stated liberty restriction\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
  "decision_rule": "prefer the action that leaves the least-advantaged group (those at risk of death) best off",
  "landscape_cases": {
    "pulls the lever": "IMPROVES: by acting, only one worker dies instead of ten, raising the primary good of basic security for the worst-off group.",
    "does not pull the lever": "WORSENS: inaction lets ten workers die, leaving the worst-off far worse in basic security."
  },
  "proposed_framework_state": {
    "constraint": "FAIRNESS",
    "action_scores": {
      "pulls the lever": 0.76,
      "does not pull the lever": 0.24
    },
    "recommended_action": "pulls the lever",
    "rationale": "fewer workers die",
    "decision_rule": "prefer the action that leaves the least-advantaged group (those at risk of death) best off",
    "framework_action_map": {
      "pulls the lever": "IMPROVES: worst-off face one death instead of ten, so primary good of life is better distributed",
      "does not pull the lever": "WORSENS: worst-off face ten deaths, markedly worse distribution of the primary good of life"
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "death count among workers (1 vs 10)",
    "action_admissibility": {
      "pulls the lever": "PERMISSIBLE",
      "does not pull the lever": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "pulls the lever",
    "factual_reversal_threshold": "side track contains ten or more workers",
    "normative_reversal_threshold": "intentional-killing ban given lexical liberty priority",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {
      "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
      "ranking_classification_justification": "life is a fundamental primary good; liberty conflicts are absent",
      "lexical_priority_justification": "lexical liberty priority not triggered—no stated liberty restriction",
      "liberty_status": {
        "A0": "NOT_APPLICABLE",
        "A1": "NOT_APPLICABLE"
      },
      "positions": [
        {
          "action_id": "A0",
          "subject": "track workers",
          "subject_kind": "GROUP",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "IMPROVES",
          "compared_to_action_id": "A1",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "only one rather than ten are killed",
          "principle_basis": "MAXIMIN_PRIMARY_GOODS"
        },
        {
          "action_id": "A1",
          "subject": "track workers",
          "subject_kind": "GROUP",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "WORSENS",
          "compared_to_action_id": "A0",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "ten rather than one are killed",
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
  "investigative_claim": "",
  "committed_native_ledger": {
    "ledger_kind": "RAWLSIAN_POSITION_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:ecff0bd33b6236a0",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "compared_to_action_id": "A1",
        "effect": "MIXED",
        "proposed_effect": "IMPROVES",
        "subject": "track workers",
        "proposed_subject": "track workers",
        "dimension_subject_repaired": false,
        "affected_subject": "track workers",
        "subject_kind": "GROUP",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "MIXED",
        "group_node_id": "RAWLS_SUBJECT:aa90fcbabf8956cf",
        "subject_node_id": "RAWLS_SUBJECT:aa90fcbabf8956cf",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "LEXICAL",
        "group_binding_mode": "LEXICAL",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "MIXED_COMPARISON",
        "reason": "only one rather than ten are killed",
        "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
        "ranking_classification_justification": "life is a fundamental primary good; liberty conflicts are absent",
        "lexical_priority_justification": "lexical liberty priority not triggered—no stated liberty restriction",
        "liberty_status": "NOT_APPLICABLE",
        "subject_selection_status": "BOUND_TO_ACTION",
        "group_selection_status": "BOUND_TO_ACTION",
        "provenance": [
          "delegate:rawlsian",
          "cycle:1"
        ]
      },
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e9e1f43bcccb4f61",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "compared_to_action_id": "A0",
        "effect": "MIXED",
        "proposed_effect": "WORSENS",
        "subject": "track workers",
        "proposed_subject": "track workers",
        "dimension_subject_repaired": false,
        "affected_subject": "track workers",
        "subject_kind": "GROUP",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "MIXED",
        "group_node_id": "RAWLS_SUBJECT:aa90fcbabf8956cf",
        "subject_node_id": "RAWLS_SUBJECT:aa90fcbabf8956cf",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "LEXICAL",
        "group_binding_mode": "LEXICAL",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "MIXED_COMPARISON",
        "reason": "ten rather than one are killed",
        "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
        "ranking_classification_justification": "life is a fundamental primary good; liberty conflicts are absent",
        "lexical_priority_justification": "lexical liberty priority not triggered—no stated liberty restriction",
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
  "normative_reversal_threshold": "intentional-killing ban given lexical liberty priority",
  "factual_reversal_threshold": "side track contains ten or more workers",
  "model_argument_fields": {
    "rationale": "fewer workers die",
    "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"dcp\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"ep\":[{\"c\":\"the tram will kill one worker\",\"p\":\"PROP:WORLD:E3\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E3\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"it will kill ten workers\",\"p\":\"PROP:WORLD:E7\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E7\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: worst-off face one death instead of ten, so primary good of life is better distributed\",\"A1\":\"WORSENS: worst-off face ten deaths, markedly worse distribution of the primary good of life\"},\"nr\":\"DECISIVE\",\"np\":\"death count among workers (1 vs 10)\",\"rp\":{\"A0\":{\"g\":\"track workers\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"only one rather than ten are killed\"},\"A1\":{\"g\":\"track workers\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"ten rather than one are killed\"}},\"rb\":\"MAXIMIN_PRIMARY_GOODS\",\"rbc\":\"life is a fundamental primary good; liberty conflicts are absent\",\"lpj\":\"lexical liberty priority not triggered—no stated liberty restriction\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
    "decision_rule": "prefer the action that leaves the least-advantaged group (those at risk of death) best off",
    "landscape_cases": {
      "A0": "IMPROVES: by acting, only one worker dies instead of ten, raising the primary good of basic security for the worst-off group.",
      "A1": "WORSENS: inaction lets ten workers die, leaving the worst-off far worse in basic security."
    },
    "factual_reversal_threshold": "side track contains ten or more workers",
    "normative_reversal_threshold": "intentional-killing ban given lexical liberty priority"
  }
}
```
