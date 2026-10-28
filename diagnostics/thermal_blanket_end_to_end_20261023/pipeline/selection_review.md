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
  "raw_response": "{\n  \"status\": \"SELECTED\",\n  \"selected_candidate_id\": \"candidate_0\",\n  \"resolutions\": [\n    {\n      \"difference_id\": \"difference_0\",\n      \"rationale\": \"The source explicitly states Ellis can give the blanket either to Rowan or to Sasha, so the two \\\"gives\\\" actions in candidate_0 are text-grounded.  The missing \\\"promise\\\" action is an omission, not a contradiction.\",\n      \"evidence\": [\n        {\"clause_id\": \"C1\", \"excerpt\": \"Rescuer Ellis can give the blanket to Rowan or Sasha, but not both.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_1\",\n      \"rationale\": \"Clauses C3–C5 describe causal results that follow from giving (survival chance or death).  Candidate_0’s causal_links encode this enabling relation, which is therefore supported.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.\"},\n        {\"clause_id\": \"C4\", \"excerpt\": \"If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival.\"},\n        {\"clause_id\": \"C5\", \"excerpt\": \"The hiker who does not get the blanket will die.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_2\",\n      \"rationale\": \"The commitment in candidate_0 records Ellis’s promise to Sasha that the blanket would be received, matching clause C6.  The more generic wording of the content proposition (\\\"would get the blanket\\\") still refers to Sasha via the commitment fields, so it is not contradicted.\",\n      \"evidence\": [\n        {\"clause_id\": \"C6\", \"excerpt\": \"Ellis promised Sasha that Sasha would get the blanket.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_3\",\n      \"rationale\": \"All effects listed in candidate_0 (resource transfer, 80 %/60 % survival chances, death of the non-recipient) are explicitly or conditionally stated in C3–C5, so they are supported.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"Rowan has an 80% chance of survival.\"},\n        {\"clause_id\": \"C4\", \"excerpt\": \"Sasha has a 60% chance of survival.\"},\n        {\"clause_id\": \"C5\", \"excerpt\": \"The hiker who does not get the blanket will die.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_4\",\n      \"rationale\": \"Candidate_0 lists all entities that appear in the text (Ellis, Rowan, Sasha, and the single thermal blanket).\",\n      \"evidence\": [\n        {\"clause_id\": \"C0\", \"excerpt\": \"A mountain rescue shelter has one thermal blanket.\"},\n        {\"clause_id\": \"C1\", \"excerpt\": \"Rescuer Ellis can give the blanket to Rowan or Sasha\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_5\",\n      \"rationale\": \"The proposition \\\"would get the blanket\\\" captures the promised content; linked commitment fields tie it to Sasha, so it is consistent with C6.\",\n      \"evidence\": [\n        {\"clause_id\": \"C6\", \"excerpt\": \"Sasha would get the blanket.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_6\",\n      \"rationale\": \"Compared with candidate_2, candidate_0 correctly models the giving actions with real participants (Rowan, Sasha) rather than turning probabilities into parties, matching clause C1.\",\n      \"evidence\": [\n        {\"clause_id\": \"C1\", \"excerpt\": \"give the blanket to Rowan or Sasha\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_7\",\n      \"rationale\": \"As in difference 1, the causal links in candidate_0 reflect the conditional consequences stated in C3–C5; candidate_2 provides none.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_8\",\n      \"rationale\": \"Candidate_0 includes the promise commitment required by C6; candidate_2 omits it.\",\n      \"evidence\": [\n        {\"clause_id\": \"C6\", \"excerpt\": \"Ellis promised Sasha that Sasha would get the blanket.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_9\",\n      \"rationale\": \"Candidate_2 invents separate condition nodes; their absence in candidate_0 is only an omission, not a contradiction, so candidate_0 remains text-consistent.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"If Ellis gives the blanket to Rowan…\"},\n        {\"clause_id\": \"C4\", \"excerpt\": \"If Ellis gives the blanket to Sasha…\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_10\",\n      \"rationale\": \"The detailed survival/death effects in candidate_0 mirror C3–C5, while candidate_2 records merely the transfers and loses the outcome information, so the effects set in candidate_0 is better supported.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"80% chance of survival\"},\n        {\"clause_id\": \"C4\", \"excerpt\": \"60% chance of survival\"},\n        {\"clause_id\": \"C5\", \"excerpt\": \"will die\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_11\",\n      \"rationale\": \"Candidate_2 misidentifies \\\"a 60% chance\\\" and \\\"an 80% chance\\\" as parties, contradicting the text.  Candidate_0’s parties match the named humans and the blanket.\",\n      \"evidence\": [\n        {\"clause_id\": \"C3\", \"excerpt\": \"Rowan has an 80% chance of survival.\"},\n        {\"clause_id\": \"C4\", \"excerpt\": \"Sasha has a 60% chance of survival.\"}\n      ]\n    },\n    {\n      \"difference_id\": \"difference_12\",\n      \"rationale\": \"Candidate_0 preserves the promised-content proposition; candidate_2 lacks any proposition node, so candidate_0 is the supported reading.\",\n      \"evidence\": [\n        {\"clause_id\": \"C6\", \"excerpt\": \"Sasha would get the blanket.\"}\n      ]\n    }\n  ],\n  \"unresolved_question\": \"\",\n  \"missing_content\": [\n    \"The promise is not encoded as an explicit 'promise' action in candidate_0 (only as a commitment node).\",\n    \"The two conditional \\\"if\\\" statements are not represented as separate condition nodes.\"\n  ]\n}",
  "response": {
    "status": "SELECTED",
    "selected_candidate_id": "candidate_0",
    "resolutions": [
      {
        "difference_id": "difference_0",
        "rationale": "The source explicitly states Ellis can give the blanket either to Rowan or to Sasha, so the two \"gives\" actions in candidate_0 are text-grounded.  The missing \"promise\" action is an omission, not a contradiction.",
        "evidence": [
          {
            "clause_id": "C1",
            "excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha, but not both."
          }
        ]
      },
      {
        "difference_id": "difference_1",
        "rationale": "Clauses C3–C5 describe causal results that follow from giving (survival chance or death).  Candidate_0’s causal_links encode this enabling relation, which is therefore supported.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival."
          },
          {
            "clause_id": "C4",
            "excerpt": "If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival."
          },
          {
            "clause_id": "C5",
            "excerpt": "The hiker who does not get the blanket will die."
          }
        ]
      },
      {
        "difference_id": "difference_2",
        "rationale": "The commitment in candidate_0 records Ellis’s promise to Sasha that the blanket would be received, matching clause C6.  The more generic wording of the content proposition (\"would get the blanket\") still refers to Sasha via the commitment fields, so it is not contradicted.",
        "evidence": [
          {
            "clause_id": "C6",
            "excerpt": "Ellis promised Sasha that Sasha would get the blanket."
          }
        ]
      },
      {
        "difference_id": "difference_3",
        "rationale": "All effects listed in candidate_0 (resource transfer, 80 %/60 % survival chances, death of the non-recipient) are explicitly or conditionally stated in C3–C5, so they are supported.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "Rowan has an 80% chance of survival."
          },
          {
            "clause_id": "C4",
            "excerpt": "Sasha has a 60% chance of survival."
          },
          {
            "clause_id": "C5",
            "excerpt": "The hiker who does not get the blanket will die."
          }
        ]
      },
      {
        "difference_id": "difference_4",
        "rationale": "Candidate_0 lists all entities that appear in the text (Ellis, Rowan, Sasha, and the single thermal blanket).",
        "evidence": [
          {
            "clause_id": "C0",
            "excerpt": "A mountain rescue shelter has one thermal blanket."
          },
          {
            "clause_id": "C1",
            "excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha"
          }
        ]
      },
      {
        "difference_id": "difference_5",
        "rationale": "The proposition \"would get the blanket\" captures the promised content; linked commitment fields tie it to Sasha, so it is consistent with C6.",
        "evidence": [
          {
            "clause_id": "C6",
            "excerpt": "Sasha would get the blanket."
          }
        ]
      },
      {
        "difference_id": "difference_6",
        "rationale": "Compared with candidate_2, candidate_0 correctly models the giving actions with real participants (Rowan, Sasha) rather than turning probabilities into parties, matching clause C1.",
        "evidence": [
          {
            "clause_id": "C1",
            "excerpt": "give the blanket to Rowan or Sasha"
          }
        ]
      },
      {
        "difference_id": "difference_7",
        "rationale": "As in difference 1, the causal links in candidate_0 reflect the conditional consequences stated in C3–C5; candidate_2 provides none.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival."
          }
        ]
      },
      {
        "difference_id": "difference_8",
        "rationale": "Candidate_0 includes the promise commitment required by C6; candidate_2 omits it.",
        "evidence": [
          {
            "clause_id": "C6",
            "excerpt": "Ellis promised Sasha that Sasha would get the blanket."
          }
        ]
      },
      {
        "difference_id": "difference_9",
        "rationale": "Candidate_2 invents separate condition nodes; their absence in candidate_0 is only an omission, not a contradiction, so candidate_0 remains text-consistent.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "If Ellis gives the blanket to Rowan…"
          },
          {
            "clause_id": "C4",
            "excerpt": "If Ellis gives the blanket to Sasha…"
          }
        ]
      },
      {
        "difference_id": "difference_10",
        "rationale": "The detailed survival/death effects in candidate_0 mirror C3–C5, while candidate_2 records merely the transfers and loses the outcome information, so the effects set in candidate_0 is better supported.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "80% chance of survival"
          },
          {
            "clause_id": "C4",
            "excerpt": "60% chance of survival"
          },
          {
            "clause_id": "C5",
            "excerpt": "will die"
          }
        ]
      },
      {
        "difference_id": "difference_11",
        "rationale": "Candidate_2 misidentifies \"a 60% chance\" and \"an 80% chance\" as parties, contradicting the text.  Candidate_0’s parties match the named humans and the blanket.",
        "evidence": [
          {
            "clause_id": "C3",
            "excerpt": "Rowan has an 80% chance of survival."
          },
          {
            "clause_id": "C4",
            "excerpt": "Sasha has a 60% chance of survival."
          }
        ]
      },
      {
        "difference_id": "difference_12",
        "rationale": "Candidate_0 preserves the promised-content proposition; candidate_2 lacks any proposition node, so candidate_0 is the supported reading.",
        "evidence": [
          {
            "clause_id": "C6",
            "excerpt": "Sasha would get the blanket."
          }
        ]
      }
    ],
    "unresolved_question": "",
    "missing_content": [
      "The promise is not encoded as an explicit 'promise' action in candidate_0 (only as a commitment node).",
      "The two conditional \"if\" statements are not represented as separate condition nodes."
    ]
  },
  "evidence_bindings": [
    {
      "difference_id": "difference_0",
      "clause_id": "C1",
      "raw_excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha, but not both.",
      "bound_excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha, but not both.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_1",
      "clause_id": "C3",
      "raw_excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.",
      "bound_excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_1",
      "clause_id": "C4",
      "raw_excerpt": "If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival.",
      "bound_excerpt": "If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_1",
      "clause_id": "C5",
      "raw_excerpt": "The hiker who does not get the blanket will die.",
      "bound_excerpt": "The hiker who does not get the blanket will die.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_2",
      "clause_id": "C6",
      "raw_excerpt": "Ellis promised Sasha that Sasha would get the blanket.",
      "bound_excerpt": "Ellis promised Sasha that Sasha would get the blanket.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_3",
      "clause_id": "C3",
      "raw_excerpt": "Rowan has an 80% chance of survival.",
      "bound_excerpt": "Rowan has an 80% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_3",
      "clause_id": "C4",
      "raw_excerpt": "Sasha has a 60% chance of survival.",
      "bound_excerpt": "Sasha has a 60% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_3",
      "clause_id": "C5",
      "raw_excerpt": "The hiker who does not get the blanket will die.",
      "bound_excerpt": "The hiker who does not get the blanket will die.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_4",
      "clause_id": "C0",
      "raw_excerpt": "A mountain rescue shelter has one thermal blanket.",
      "bound_excerpt": "A mountain rescue shelter has one thermal blanket.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_4",
      "clause_id": "C1",
      "raw_excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha",
      "bound_excerpt": "Rescuer Ellis can give the blanket to Rowan or Sasha",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_5",
      "clause_id": "C6",
      "raw_excerpt": "Sasha would get the blanket.",
      "bound_excerpt": "Sasha would get the blanket.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_6",
      "clause_id": "C1",
      "raw_excerpt": "give the blanket to Rowan or Sasha",
      "bound_excerpt": "give the blanket to Rowan or Sasha",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_7",
      "clause_id": "C3",
      "raw_excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.",
      "bound_excerpt": "If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_8",
      "clause_id": "C6",
      "raw_excerpt": "Ellis promised Sasha that Sasha would get the blanket.",
      "bound_excerpt": "Ellis promised Sasha that Sasha would get the blanket.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_9",
      "clause_id": "C3",
      "raw_excerpt": "If Ellis gives the blanket to Rowan…",
      "bound_excerpt": "If Ellis gives the blanket to Rowan",
      "method": "EXACT_PREFIX_WITH_TERMINAL_ELLIPSIS"
    },
    {
      "difference_id": "difference_9",
      "clause_id": "C4",
      "raw_excerpt": "If Ellis gives the blanket to Sasha…",
      "bound_excerpt": "If Ellis gives the blanket to Sasha",
      "method": "EXACT_PREFIX_WITH_TERMINAL_ELLIPSIS"
    },
    {
      "difference_id": "difference_10",
      "clause_id": "C3",
      "raw_excerpt": "80% chance of survival",
      "bound_excerpt": "80% chance of survival",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_10",
      "clause_id": "C4",
      "raw_excerpt": "60% chance of survival",
      "bound_excerpt": "60% chance of survival",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_10",
      "clause_id": "C5",
      "raw_excerpt": "will die",
      "bound_excerpt": "will die",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_11",
      "clause_id": "C3",
      "raw_excerpt": "Rowan has an 80% chance of survival.",
      "bound_excerpt": "Rowan has an 80% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_11",
      "clause_id": "C4",
      "raw_excerpt": "Sasha has a 60% chance of survival.",
      "bound_excerpt": "Sasha has a 60% chance of survival.",
      "method": "EXACT_SUBSTRING"
    },
    {
      "difference_id": "difference_12",
      "clause_id": "C6",
      "raw_excerpt": "Sasha would get the blanket.",
      "bound_excerpt": "Sasha would get the blanket.",
      "method": "EXACT_SUBSTRING"
    }
  ],
  "scope": "top_three_distinct_admitted_interpretations",
  "other_admitted_candidates_preserved": 0
}
```
