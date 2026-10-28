# Candidate arguments

Nonvoting argument overlays. These are not admitted world-state graphs.

## care

STAGED_PARTIAL_ARGUMENT

Critic's proposed remaining conclusion (check accepted graph below): Caring responsiveness to the most acute life-threatening need provisionally supports giving the blanket to Rowan, though the moral cost of breaking Ellis’s promise to Sasha remains a live concern.

```mermaid
flowchart TB
n0["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: Both hikers are stranded in the cold."]
n1["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If the blanket is given to Rowan, Rowan has an 80% chance of survival."]
n2["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If the blanket is given to Rowan, Sasha will die."]
n3["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If the blanket is given to Sasha, Sasha has a 60% chance of survival."]
n4["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If the blanket is given to Sasha, Rowan will die."]
n5["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: Ellis promised Sasha that Sasha would get the blanket."]
n6["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: Under a care perspective, responding to the more urgent threat to life normally has priority over an earlier but weaker relational assurance when the "]
n7["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: Keeping a promise is an important element of sustaining trust within caring relationships."]
n8["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Ellis has stronger reason of care to give the blanket to Rowan."]
n9["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Breaking the promise to Sasha is a morally relevant harm that counts against giving the blanket to Rowan."]
n1 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n3 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n2 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n4 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n6 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n5 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n9
n7 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n9
```

Remaining questions: Whether Ellis could have competently avoided promising before full knowledge and whether the lingering harm to caring relationships outweighs the 20-percentage-point survival difference


## Binding issues (not moral verdicts)

[
  {
    "kind": "DETACHED_COMPARISON",
    "record_id": "C1"
  }
]

## Comparison bookkeeping (not semantic validation)

{
  "version": "complaint-comparison/0.1",
  "findings": [
    {
      "code": "BURDEN_TRANSITION_MISMATCH",
      "record_id": "C1",
      "burden_id": "B1R",
      "claim": "relieved_burden_ids",
      "observed": [
        "PRESENT",
        "PRESENT"
      ],
      "expected": [
        "PRESENT",
        "ABSENT"
      ]
    },
    {
      "code": "BURDEN_TRANSITION_MISMATCH",
      "record_id": "C1",
      "burden_id": "B2S",
      "claim": "introduced_burden_ids",
      "observed": [
        "PRESENT",
        "ABSENT"
      ],
      "expected": [
        "ABSENT",
        "PRESENT"
      ]
    },
    {
      "code": "BURDEN_TRANSITION_MISMATCH",
      "record_id": "C1",
      "burden_id": "B3E",
      "claim": "retained_burden_ids",
      "observed": [
        "UNKNOWN",
        "UNKNOWN"
      ],
      "expected": [
        "PRESENT",
        "PRESENT"
      ]
    }
  ],
  "status": "REVIEW_REQUIRED",
  "matrix_authority": "MODEL_PROPOSED_NOT_SOURCE_VERIFIED",
  "semantic_support": "NOT_INDEPENDENTLY_ASSESSED",
  "moral_ranking_computed": false,
  "operative": false,
  "recommendation_status": "PROVISIONAL_PREFERENCE",
  "remaining_complaint_ids": [
    "R2"
  ]
}

## Burdens by option

| Burden / person | Option | Status |
| --- | --- | --- |
| Risk of death from exposure for Rowan / P1 | A | PRESENT |
| Risk of death from exposure for Sasha / P2 | A | PRESENT |
| Harm of promise being broken / P2 | A | PRESENT |
| Moral distress for Ellis at failing someone in need or breaking a promise / P3 | A | UNKNOWN |
| Risk of death from exposure for Rowan / P1 | B | PRESENT |
| Risk of death from exposure for Sasha / P2 | B | PRESENT |
| Harm of promise being broken / P2 | B | ABSENT |
| Moral distress for Ellis at failing someone in need or breaking a promise / P3 | B | UNKNOWN |


Full principles, reasons, comparisons and proposed attacks: [response.json](response.json).
