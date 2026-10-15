# Virtue — recorded argument

Participation: ATTENUATED. Native ledger: COMMITTED.

## Readable explanation

The virtue agent asks what a prudent and compassionate person would do with the only available dose. It favors Ben because the same scarce treatment has a far better chance of saving his life. In its account, compassion for Cara still matters, but choosing the much less effective treatment risks wasting the opportunity to save someone. This reasoning relies on there being no other morally relevant difference between the patients. That equality is part of the agent’s argument, not something established by the survival figures. Its preference for Ben remains provisional.

Presentation basis: REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT.

## Assumptions and interpretive commitments

- Model reported assumption: Higher efficacy is decisive for prudent stewardship
- Recorded interpretive commitment: Prefer “Ada gives the medicine to Ben” when its survival likelihood far exceeds “Ada gives the medicine to Cara” with no morally relevant difference in patients.
- Recorded interpretive commitment: Stakes highlight prudence but numbers inform rather than replace virtue judgment.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- The agent has not settled whether the patients differ in morally relevant ways beyond their survival chances. The saved review also keeps its scarcity-related premise open.

## Original presentation and objections

```json
{
  "reason": "Prudence saves life. Prefer “Ada gives the medicine to Ben” when its survival likelihood far exceeds “Ada gives the medicine to Cara” with no morally relevant difference in patients. Recorded comparison — Ada gives the medicine to Ben: Allocates scarce medicine to give 95 % survival; exemplifies prudence, beneficence, compassion.; Ada gives the medicine to Cara: Allocates to 5 % survival; shows compassion but risks imprudence and waste of life-saving chance. Dependency continuity preserved unresolved decision-critical premises from the prior operative framework state because the recommendation did not change.",
  "reason_authority": "RECORDED_ARGUMENT_FIELDS",
  "live_objections": [
    "dependency continuity preserved unresolved proposition IDs: PROP:HYPOTHESIS:3481f0238231ad3a",
    "Would verifying this unestablished consequence reverse the admitted ranking: The medicine is a scarce resource that can be given to only one patient?"
  ],
  "reversal_explanations": [
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Higher efficacy is decisive for prudent stewardship. This is an agent-reported condition, not an independently established threshold.",
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Impartial equality overrides outcome maximisation. This is an agent-reported condition, not an independently established threshold.",
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Cara survival probability exceeds Ben’s. This is an agent-reported condition, not an independently established threshold."
  ]
}
```

## Explanation provenance

```json
{
  "basis_sha256": "432c580a91cab98c85292fd4dad92aacdc853bbf054f582355cb26506d9398e2",
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

- The agent says it would reconsider treating Ben if Cara’s survival chance became greater, or if it gave equal treatment priority over choosing the more effective rescue. The record does not settle how that equality principle would choose between the patients.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Prudence saves life",
  "framework_application": "NONE",
  "decision_rule": "Prefer A0 when its survival likelihood far exceeds A1 with no morally relevant difference in patients.",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "Allocates scarce medicine to give 95 % survival; exemplifies prudence, beneficence, compassion.",
    "Ada gives the medicine to Cara": "Allocates to 5 % survival; shows compassion but risks imprudence and waste of life-saving chance."
  },
  "proposed_framework_state": {
    "constraint": "CHARACTER",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.9,
      "Ada gives the medicine to Cara": 0.1
    },
    "recommended_action": "Ada gives the medicine to Ben",
    "rationale": "Prudence saves life",
    "decision_rule": "Prefer A0 when its survival likelihood far exceeds A1 with no morally relevant difference in patients.",
    "framework_action_map": {
      "Ada gives the medicine to Ben": "EXEMPLIFIES: healer prudently stewards scarce resource, maximizing chance to save life.",
      "Ada gives the medicine to Cara": "MIXED: compassion shown, yet imprudent stewardship choosing far less effective outcome."
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "Stakes highlight prudence but numbers inform rather than replace virtue judgment.",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "PERMISSIBLE",
      "Ada gives the medicine to Cara": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "Ada gives the medicine to Ben",
    "factual_reversal_threshold": "Cara survival probability exceeds Ben’s",
    "normative_reversal_threshold": "Impartial equality overrides outcome maximisation",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {
      "ranking_basis": "PRACTICAL_WISDOM",
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "EXEMPLIFIES",
          "actor_role": "healer",
          "virtues": "prudence, beneficence, compassion",
          "vice_risk": "NONE",
          "circumstance": "95 % vs 5 % survival under one-dose scarcity",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "Uses skill wisely to save life"
        },
        {
          "action_id": "A1",
          "verdict": "MIXED",
          "actor_role": "healer",
          "virtues": "compassion",
          "vice_risk": "imprudence",
          "circumstance": "ignores large efficacy gap",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "Good motive, poor judgment"
        }
      ]
    },
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "framework ranking retains an unresolved decision-critical dependency",
  "investigative_claim": "Would verifying this unestablished consequence reverse the admitted ranking: The medicine is a scarce resource that can be given to only one patient?",
  "committed_native_ledger": {
    "ledger_kind": "VIRTUE_CHARACTER_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "assessment_node_id": "VIRTUE_CHARACTER:virtue:A0",
        "framework": "VIRTUE",
        "specialist": "virtue",
        "cycle": 3,
        "canonical_action_id": "A0",
        "verdict": "EXEMPLIFIES",
        "actor_role": "healer",
        "virtues": "prudence, beneficence, compassion",
        "vice_risk": "NONE",
        "circumstance": "95 % vs 5 % survival under one-dose scarcity",
        "evidence_basis": "ACTION_GRAPH",
        "reason": "Uses skill wisely to save life",
        "ranking_basis": "PRACTICAL_WISDOM",
        "grounded_effect_ids": [
          "E5",
          "E2"
        ],
        "grounded_consequence_ids": [
          "A0:WORLD_EFFECT:E5",
          "A0:WORLD_EFFECT:E2"
        ],
        "epistemic_status": "FRAMEWORK_GROUNDED_WITH_ACTION_EFFECTS",
        "provenance": [
          "delegate:virtue",
          "cycle:3"
        ]
      },
      {
        "assessment_node_id": "VIRTUE_CHARACTER:virtue:A1",
        "framework": "VIRTUE",
        "specialist": "virtue",
        "cycle": 3,
        "canonical_action_id": "A1",
        "verdict": "MIXED",
        "actor_role": "healer",
        "virtues": "compassion",
        "vice_risk": "imprudence",
        "circumstance": "ignores large efficacy gap",
        "evidence_basis": "ACTION_GRAPH",
        "reason": "Good motive, poor judgment",
        "ranking_basis": "PRACTICAL_WISDOM",
        "grounded_effect_ids": [],
        "grounded_consequence_ids": [],
        "epistemic_status": "UNRESOLVED_ACTION_GROUNDING",
        "provenance": [
          "delegate:virtue",
          "cycle:3"
        ]
      }
    ]
  },
  "unsupported_assumption": "Higher efficacy is decisive for prudent stewardship",
  "normative_reversal_threshold": "Impartial equality overrides outcome maximisation",
  "factual_reversal_threshold": "Cara survival probability exceeds Ben’s",
  "model_argument_fields": null
}
```
