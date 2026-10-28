# Rawlsian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

Decisive aggregation is not licensed by the committed Rawlsian ranking basis. higher survival chance. prefer the action that offers the higher survival probability to whichever hiker one might be. Recorded comparison — “gives the blanket to Rowan”: IMPROVES: Giving Rowan the blanket yields an 80 % survival chance, maximizing the worst-case prospect one would accept behind the veil.; “gives the blanket to Sasha”: WORSENS: Giving Sasha the blanket leaves only a 60 % survival chance for the possible self and thus a lower maximin position, though it honours a prior promise.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer the action that offers the higher survival probability to whichever hiker one might be
- Recorded interpretive commitment: Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- framework map for gives the blanket to Rowan lacks framework-specific grounds
- decisive numbers are not tied to the affected subject or Rawlsian position
- A0 IMPROVES lacked directional support; committed as UNCERTAIN
- A1 WORSENS lacked directional support; committed as UNCERTAIN

## Conditions for reconsidering

- The agent reports a switch from “gives the blanket to Rowan” to “gives the blanket to Sasha”. Condition: Promise-keeping outweighs maximin survival. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “gives the blanket to Rowan” to “gives the blanket to Sasha”. Condition: Rowan’s survival chance drops below Sasha’s. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "higher survival chance",
  "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"fm\":{\"A0\":\"IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.\",\"A1\":\"WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains.\"},\"nr\":\"DECISIVE\",\"np\":\"Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket\",\"rp\":{\"A0\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"higher survival probability\"},\"A1\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"lower survival probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.\",\"lpj\":\"No basic-liberty conflict is present; thus lexical priority is not triggered.\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"sps\":[\"PROP:COMMITMENT:K0\"],\"dcp\":[],\"ep\":[{\"c\":\"promised would get the blanket\",\"p\":\"PROP:COMMITMENT:K0\",\"dc\":false,\"a\":\"GLOBAL\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"da\":\"decisive ethical axis\",\"t\":\"attempted comparison rule\",\"tf\":\"NONE\",\"dr\":\"prefer A0 when its reason outweighs A1\",\"ft\":\"NONE\",\"nt\":\"prefer A1 if its value is overriding\",\"z\":0.8,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]}}",
  "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
  "landscape_cases": {
    "gives the blanket to Rowan": "IMPROVES: Giving Rowan the blanket yields an 80 % survival chance, maximizing the worst-case prospect one would accept behind the veil.",
    "gives the blanket to Sasha": "WORSENS: Giving Sasha the blanket leaves only a 60 % survival chance for the possible self and thus a lower maximin position, though it honours a prior promise."
  },
  "proposed_framework_state": {
    "constraint": "FAIRNESS",
    "action_scores": {
      "gives the blanket to Rowan": 0.6950000000000001,
      "gives the blanket to Sasha": 0.305
    },
    "recommended_action": "gives the blanket to Rowan",
    "rationale": "higher survival chance",
    "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
    "framework_action_map": {
      "gives the blanket to Rowan": "IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.",
      "gives the blanket to Sasha": "WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains."
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket",
    "action_admissibility": {
      "gives the blanket to Rowan": "PERMISSIBLE",
      "gives the blanket to Sasha": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "gives the blanket to Rowan",
    "factual_reversal_threshold": "Rowan’s survival chance drops below Sasha’s",
    "normative_reversal_threshold": "Promise-keeping outweighs maximin survival",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
      "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
      "liberty_status": {
        "A0": "NOT_APPLICABLE",
        "A1": "NOT_APPLICABLE"
      },
      "positions": [
        {
          "action_id": "A0",
          "subject": "either stranded hiker",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "IMPROVES",
          "compared_to_action_id": "A1",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "higher survival probability",
          "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
        },
        {
          "action_id": "A1",
          "subject": "either stranded hiker",
          "subject_kind": "INDIVIDUAL",
          "dimension": "BASIC_INTEREST_SECURITY",
          "basic_liberty_kind": "NOT_APPLICABLE",
          "institutional_relation": "NATURAL_CONTINGENCY",
          "additional_dimensions": [],
          "effect": "WORSENS",
          "compared_to_action_id": "A0",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "lower survival probability",
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
    "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
    "records": [
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:c02471f5a91ce289",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "compared_to_action_id": "A1",
        "effect": "UNCERTAIN",
        "proposed_effect": "IMPROVES",
        "subject": "either stranded hiker",
        "proposed_subject": "either stranded hiker",
        "dimension_subject_repaired": false,
        "affected_subject": "either stranded hiker",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "UNCERTAIN",
        "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
        "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "ACTION_GRAPH",
        "group_binding_mode": "ACTION_GRAPH",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "UNSUPPORTED_DIRECTION",
        "reason": "higher survival probability",
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
        "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
        "liberty_status": "NOT_APPLICABLE",
        "subject_selection_status": "BOUND_TO_ACTION",
        "group_selection_status": "BOUND_TO_ACTION",
        "provenance": [
          "delegate:rawlsian",
          "cycle:1"
        ]
      },
      {
        "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
        "framework": "RAWLSIAN",
        "specialist": "rawlsian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "compared_to_action_id": "A0",
        "effect": "UNCERTAIN",
        "proposed_effect": "WORSENS",
        "subject": "either stranded hiker",
        "proposed_subject": "either stranded hiker",
        "dimension_subject_repaired": false,
        "affected_subject": "either stranded hiker",
        "subject_kind": "INDIVIDUAL",
        "dimension": "BASIC_INTEREST_SECURITY",
        "basic_liberty_kind": "NOT_APPLICABLE",
        "institutional_relation": "NATURAL_CONTINGENCY",
        "additional_dimensions": [],
        "dimension_bundle": [
          "BASIC_INTEREST_SECURITY"
        ],
        "comparative_effect": "UNCERTAIN",
        "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
        "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
        "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
        "subject_binding_mode": "ACTION_GRAPH",
        "group_binding_mode": "ACTION_GRAPH",
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "UNSUPPORTED_DIRECTION",
        "reason": "lower survival probability",
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
        "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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
  "normative_reversal_threshold": "Promise-keeping outweighs maximin survival",
  "factual_reversal_threshold": "Rowan’s survival chance drops below Sasha’s",
  "model_argument_fields": {
    "rationale": "higher survival chance",
    "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"fm\":{\"A0\":\"IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.\",\"A1\":\"WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains.\"},\"nr\":\"DECISIVE\",\"np\":\"Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket\",\"rp\":{\"A0\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"higher survival probability\"},\"A1\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"lower survival probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.\",\"lpj\":\"No basic-liberty conflict is present; thus lexical priority is not triggered.\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"sps\":[\"PROP:COMMITMENT:K0\"],\"dcp\":[],\"ep\":[{\"c\":\"promised would get the blanket\",\"p\":\"PROP:COMMITMENT:K0\",\"dc\":false,\"a\":\"GLOBAL\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"da\":\"decisive ethical axis\",\"t\":\"attempted comparison rule\",\"tf\":\"NONE\",\"dr\":\"prefer A0 when its reason outweighs A1\",\"ft\":\"NONE\",\"nt\":\"prefer A1 if its value is overriding\",\"z\":0.8,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]}}",
    "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
    "landscape_cases": {
      "A0": "IMPROVES: Giving Rowan the blanket yields an 80 % survival chance, maximizing the worst-case prospect one would accept behind the veil.",
      "A1": "WORSENS: Giving Sasha the blanket leaves only a 60 % survival chance for the possible self and thus a lower maximin position, though it honours a prior promise."
    },
    "factual_reversal_threshold": "Rowan’s survival chance drops below Sasha’s",
    "normative_reversal_threshold": "Promise-keeping outweighs maximin survival"
  }
}
```
