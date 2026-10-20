# Targeted interpretation selection

This review selects an existing admitted interpretation; it does not prove semantic correctness.

Status: **SELECTED**

```json
{
  "status": "SELECTED",
  "selected_candidate_id": "candidate_0",
  "errors": [],
  "origin": "MODEL_INTERPRETATION_REVIEW",
  "semantic_support": "reviewer_assessed_not_proven",
  "raw_response": "{\n  \"status\": \"SELECTED\",\n  \"selected_candidate_id\": \"candidate_0\",\n  \"resolutions\": [\n    {\n      \"difference_id\": \"difference_0\",\n      \"rationale\": \"Candidate 0 includes specific actions that Mira can take, which are supported by the source text that describes her options.\",\n      \"evidence\": [\n        {\n          \"clause_id\": \"C3\",\n          \"excerpt\": \"Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes.\"\n        }\n      ]\n    },\n    {\n      \"difference_id\": \"difference_1\",\n      \"rationale\": \"Candidate 0 provides causal links that explain the consequences of the actions, which are directly supported by the source text.\",\n      \"evidence\": [\n        {\n          \"clause_id\": \"C3\",\n          \"excerpt\": \"doing so may expose nearby staff to hazardous fumes\"\n        },\n        {\n          \"clause_id\": \"C4\",\n          \"excerpt\": \"the technician may die during that delay\"\n        }\n      ]\n    },\n    {\n      \"difference_id\": \"difference_2\",\n      \"rationale\": \"Candidate 0 outlines effects of the actions that are explicitly mentioned in the source text, supporting the interpretation of outcomes.\",\n      \"evidence\": [\n        {\n          \"clause_id\": \"C3\",\n          \"excerpt\": \"Mira can open the door now to reach the technician\"\n        },\n        {\n          \"clause_id\": \"C4\",\n          \"excerpt\": \"Stopping the experiment first takes ten minutes; the technician may die during that delay.\"\n        }\n      ]\n    },\n    {\n      \"difference_id\": \"difference_3\",\n      \"rationale\": \"Candidate 0 identifies parties involved in the scenario, which are explicitly mentioned in the source text.\",\n      \"evidence\": [\n        {\n          \"clause_id\": \"C2\",\n          \"excerpt\": \"A technician is trapped inside and needs urgent medical help.\"\n        },\n        {\n          \"clause_id\": \"C3\",\n          \"excerpt\": \"Mira can open the door now to reach the technician\"\n        }\n      ]\n    }\n  ],\n  \"unresolved_question\": \"\",\n  \"missing_content\": []\n}",
  "response": {
    "status": "SELECTED",
    "selected_candidate_id": "candidate_0",
    "resolutions": [
      {
        "difference_id": "difference_0",
        "rationale": "Candidate 0 includes specific actions that Mira can take, which are supported by the source text that describes her options.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."
          }
        ]
      },
      {
        "difference_id": "difference_1",
        "rationale": "Candidate 0 provides causal links that explain the consequences of the actions, which are directly supported by the source text.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "doing so may expose nearby staff to hazardous fumes"
          },
          {
            "clause_id": "C4",
            "excerpt": "the technician may die during that delay"
          }
        ]
      },
      {
        "difference_id": "difference_2",
        "rationale": "Candidate 0 outlines effects of the actions that are explicitly mentioned in the source text, supporting the interpretation of outcomes.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "Mira can open the door now to reach the technician"
          },
          {
            "clause_id": "C4",
            "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."
          }
        ]
      },
      {
        "difference_id": "difference_3",
        "rationale": "Candidate 0 identifies parties involved in the scenario, which are explicitly mentioned in the source text.",
        "evidence": [
          {
            "clause_id": "C2",
            "excerpt": "A technician is trapped inside and needs urgent medical help."
          },
          {
            "clause_id": "C3",
            "excerpt": "Mira can open the door now to reach the technician"
          }
        ]
      }
    ],
    "unresolved_question": "",
    "missing_content": []
  },
  "scope": "top_three_distinct_admitted_interpretations",
  "other_admitted_candidates_preserved": 1
}
```
