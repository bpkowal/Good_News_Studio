# Deontological agent — troubleshooting packet

Latest complete deontological result: the October 19, 2026 all-agent laboratory rerun.
Two live `o3` calls, both status `RETURNED`. Generation at 12:56, review at 12:58.
Source directory: `diagnostics/lab_all_agents_20261019`.
This file copies the agent’s replies, the world it was given, the validator’s rewrite, and the rejected review. Full prompts are at the end.

## Where to look

1. Outcome — what survived.
2. World — scenario, effects, rules.
3. Abbreviation key — the short JSON fields.
4. Generation reply — the first model answer, verbatim.
5. What admission did to that reply.
6. Review reply — the second model answer, and the objections it was given.
7. Why the review update was rejected.
8. Ledger that remains operative.
9. Duty instructions the model was given.
10. Full prompts.

## 1. Outcome

The agent prefers **stop the experiment and wait**. Participation is `ATTENUATED`. The native ledger is `COMMITTED_WITH_UNCERTAINTY`. Vote status is `ATTENUATED` because the ledger still carries validation residue. Selection status is `PROVISIONAL`. No collective verdict was produced by this specialist pass.

Generation submitted opening the door as `PROHIBITED` and waiting as `REQUIRED`. Both rows were `RESOLVED`, with governing norm `PRIMARY` and duty type `PERFECT_NEGATIVE`. Opening the door used priority basis `RESPECT_PERSONS`. Admission kept the waiting row as `REQUIRED` / `RESOLVED`. It rewrote the opening-the-door row to `CONFLICTED` / `CONTESTED`, with governing norm and priority basis `UNRESOLVED`. Four calibration errors stay on that row. A fifth warning says the waiting row’s protected party, nearby staff, is not grounded in that action’s effects.

The generation trace’s top-level result is `selected_action=INCONCLUSIVE`, `halted_by=insufficient_valid_candidates`, `system_error=INSUFFICIENT_VALID_DELEGATES`. That is the single-specialist wrapper. The model call itself returned, `delegate_status=VALID`, and the duty ledger was committed. `active_specialists` contains only `deontological`.

Review was given two open objections on the opening-the-door assessment:

- `CONFLICT:625cf1b5bef2a4fc` OPEN / UNSUPPORTED_INFERENCE / NATIVE_LEDGER_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- `CONFLICT:3e8deb5aa45fd869` OPEN / UNSUPPORTED_INFERENCE / NATIVE_LEDGER_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use

The review reply withdrew the claim that staff are used as a means, changed harm on opening the door from `DOING_HARM` to `MIXED`, and tried to put that row back to `PROHIBITED` with `gv=PRIMARY` and `pb=PERFECT_DUTY`. It labeled the challenge `RESOLVED` with effect `NO_CHANGE`. Native continuity rejected the ledger update:

> Deontological principle state changed without a framework-relevant workspace reason … previous committed ledger preserved

`framework_retention_status` on the review cycle is `PRESERVED_AFTER_REJECTED_UPDATE`. The review trace then stopped at `halted_by=cycle_budget` (`RESOURCE_CENSORED`). The operative record is still the generation ledger, including the original calibration errors. The model’s `RESOLVED` label did not close them.

## 2. World the agent was given

A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running. The rule allows opening it only after the experiment has stopped. A technician is trapped inside and needs urgent medical help. Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes. Stopping the experiment first takes ten minutes; the technician may die during that delay.

Should Mira open the door immediately or stop the experiment and wait?

### Actions

- `A0` open the door now
- `A1` stop the experiment and wait

### Parties

- `P1` Mira (PERSON)
- `P2` the door (PROCESS)
- `P3` nearby staff (PERSON)
- `P4` the experiment (PROCESS)
- `P5` the technician (PERSON)
- `P6` A laboratory safety rule (OTHER)
- `P7` The rule (OTHER)

### Admitted effects

- `E1` on `A0` / P2: open the door now (OPEN; NEUTRAL; DIRECT; CERTAIN; INTERVENTION)
- `E2` on `A0` / P3: doing so may expose nearby staff to hazardous fumes (EXPOSE; ADVERSE; DOWNSTREAM; POSSIBLE; OTHER)
- `E3` on `A1` / P4: Stopping the experiment (STOP; NEUTRAL; DIRECT; CERTAIN; INTERVENTION)
- `E4` on `A1` / P4: Stopping the experiment first takes ten minutes (TAKE; NEUTRAL; DOWNSTREAM; CERTAIN; OTHER)
- `E5` on `A1` / P5: the technician may die during that delay (DIE; ADVERSE; DOWNSTREAM; POSSIBLE; HEALTH_OUTCOME)

### Causal links

- `E1` ENABLES `E2` (action A0; modality CERTAIN)
- `E3` ENABLES `E4` (action A1; modality CERTAIN)
- `E4` ENABLES `E5` (action A1; modality CERTAIN)

E1 enables E2. E2 is a possible downstream exposure of nearby staff. E3 enables E4, and E4 enables E5, a possible death of the technician during the ten-minute wait. There is no `CAUSES` link and no settled welfare harm.

### Source rules shown in the prompt

Moral authority of these rules was marked `NOT_ASSESSED`. Identity resolution was `NO_ADDITIONAL_MERGES`.

```json
{
  "origin": "FROZEN_WORLD_COPY",
  "moral_authority": "NOT_ASSESSED",
  "identity_resolution": "NO_ADDITIONAL_MERGES",
  "rules": [
    {
      "norm_id": "N0",
      "force": "obligation",
      "bearer": {
        "party_id": "P1",
        "label": "Mira",
        "kind": "PERSON",
        "provenance": [
          {
            "clause_id": "C0",
            "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."
          },
          {
            "clause_id": "C3",
            "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."
          },
          {
            "clause_id": "C5",
            "excerpt": "Should Mira open the door immediately or stop the experiment and wait?"
          }
        ],
        "quantities": []
      },
      "authority": {
        "party_id": "P6",
        "label": "A laboratory safety rule",
        "kind": "OTHER",
        "provenance": [
          {
            "clause_id": "C0",
            "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."
          }
        ],
        "quantities": []
      },
      "governed_proposition_id": "PR1",
      "content": "keep a containment door closed while a hazardous experiment is running",
      "polarity": "positive",
      "clause_ids": [
        "C0"
      ]
    },
    {
      "norm_id": "N1",
      "force": "permission",
      "bearer": null,
      "authority": {
        "party_id": "P7",
        "label": "The rule",
        "kind": "OTHER",
        "provenance": [
          {
            "clause_id": "C1",
            "excerpt": "The rule allows opening it only after the experiment has stopped."
          }
        ],
        "quantities": []
      },
      "governed_proposition_id": "PR2",
      "content": "opening it only after the experiment has stopped",
      "polarity": "positive",
      "clause_ids": [
        "C1"
      ]
    }
  ]
}
```

## 3. Abbreviation key

Envelope fields in both replies:

| Field | Meaning |
| --- | --- |
| `scores` | Numeric action scores |
| `r` | Recommended action |
| `c` | Constraint. Deontology uses `DUTY` |
| `u` | Unresolved category |
| `w` | Short reason |
| `j` | Justification slot; `NONE` here |
| `e` | Evidence basis |
| `x` | Unstated factual prediction, or `NONE` |
| `l` | Strongest case the agent gives for each action |
| `da` | Axis that separates the actions |
| `t` | Tie-breaking rule |
| `tf` | Why that rule fails, or `NONE` |
| `dr` | Decision rule |
| `ft` / `nt` | Factual / normative reversal threshold |
| `z` | Reported confidence |
| `gu` | Proposed graph-boundary update |
| `ev` | Expected-value rows |
| `ss` | Selection status |
| `am` | Admissibility of each action |
| `cc` | Comparison complete |
| `esa` | Evidence sufficient to act |
| `ia` | Interim action |
| `fm` | Framework map. Each value must start with REQUIRED, PERMISSIBLE, PROHIBITED, or CONFLICTED |
| `nr` / `np` | Whether numbers are decisive, and why |
| `dp` | Typed duty ledger, one object per action |
| `fic` / `foq` | Internal conflicts / open questions |
| `qa` | Review-only challenge answer |

Duty-ledger fields inside `dp`:

| Field | Meaning |
| --- | --- |
| `v` | verdict |
| `k` | norm kind |
| `n` | norm |
| `rel` | relation to norm |
| `b` | duty bearer |
| `p` | protected party |
| `cn` | competing norm |
| `ck` | competing norm kind |
| `crel` | competing relation |
| `cp` | competing protected party |
| `crs` | competing reason |
| `dt` | duty type |
| `hr` | harm relation |
| `so` | special obligation |
| `sob` | special-obligation basis |
| `mr` | means relation |
| `gv` | governing norm |
| `pb` | priority basis |
| `pr` | priority rule |
| `ps` | protected standing |
| `cps` | competing protected standing |
| `ki` | coercion kind |
| `coa` | coercive actor |
| `cop` | coerced party |
| `pj` | public justification |
| `rec` | reciprocity |
| `nec` | necessity |
| `auth` | authorization |
| `dv` | derivation |
| `res` | resolution |
| `g` | evidence basis |
| `rs` | reason |
| `mx` | candidate maxim |

`mx` fields: `circumstance`, `policy`, `end`, `means`, `person_role`, `consent_constraint`, `universalization`, `action_permissibility`, `maxim_assessment`, `instrumental_belief`, `instrumental_use`, `practical_reasoning`, `means_effect_ids`, `end_effect_ids`.

## 4. Generation reply

Call status `RETURNED`, model `o3`, max_tokens `3072`, origin `MODEL_BACKEND`. Prompt length 61149 characters. Response length 5011 characters. No normalization record on this call.

The contradictions that later calibration flags are all in `dp.A0`: `hr` is `DOING_HARM` while the only adverse effect of A0 is possible; `mr` is `FORESEEN_SIDE_EFFECT` and `mx.person_role` is `FORESEEN_AFFECTED`, while `mx.instrumental_belief` is `RELIES_ON_PERSON`, `mx.instrumental_use` is `USES_PERSON`, and `rs` says staff are used as mere means. `mx.means_effect_ids` cites `E1`, the neutral door-opening intervention, not `E2`, the fume exposure. `v` is `PROHIBITED` and `res` is `RESOLVED` with `gv=PRIMARY`.

```json
{
  "scores": {
    "A0": 0.2,
    "A1": 0.8
  },
  "r": "A1",
  "c": "DUTY",
  "u": "NONE",
  "w": "Avoid direct harm to bystanders",
  "j": "NONE",
  "e": "STATED_FACTS",
  "x": "NONE",
  "l": {
    "A0": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.",
    "A1": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."
  },
  "da": "conflict between perfect duty not to harm and imperfect duty to rescue",
  "t": "prefer actions that avoid doing harm over merely allowing harm",
  "tf": "NONE",
  "dr": "Prefer A1 when A0 would directly risk harming uninvolved persons.",
  "ft": "If opening door poses no fume risk, ranking reverses.",
  "nt": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
  "z": 0.75,
  "gu": {
    "operation": "BOUNDARY",
    "from_action": "A1",
    "to_action": "A0",
    "clauses": [
      {
        "affected_action": "A0",
        "metric": "risk_to_staff",
        "metric_valence": "ADVERSE",
        "comparator": "LE",
        "threshold": 0,
        "unit": "PERCENT",
        "source_text": "doing so may expose nearby staff to hazardous fumes"
      }
    ]
  },
  "ev": {
    "A0": {
      "value": 0,
      "unit": "NONE",
      "direction": "HARM",
      "grounded": false,
      "method": "NOT_COMPUTED",
      "source_effect_ids": [],
      "calculation": "not computed",
      "assumptions": []
    },
    "A1": {
      "value": 0,
      "unit": "NONE",
      "direction": "HARM",
      "grounded": false,
      "method": "NOT_COMPUTED",
      "source_effect_ids": [],
      "calculation": "not computed",
      "assumptions": []
    }
  },
  "ss": "SELECTED",
  "am": {
    "A0": "REJECTED",
    "A1": "REQUIRED"
  },
  "cc": true,
  "esa": true,
  "ia": "A1",
  "wp": "NOT_APPLICABLE",
  "we": "NONE",
  "fa": "NONE",
  "fr": true,
  "bd": "NONE",
  "fic": [],
  "foq": [],
  "sps": [
    "PROP:DISCOURSE:PR1",
    "PROP:WORLD:E2"
  ],
  "dcp": [],
  "ep": [
    {
      "c": "keep a containment door closed while a hazardous experiment is running",
      "p": "PROP:DISCOURSE:PR1",
      "dc": false,
      "a": "GLOBAL",
      "se": [],
      "op": "DIRECT_COPY",
      "calc": "exact proposition copy",
      "asm": [],
      "ot": "PRESERVED"
    },
    {
      "c": "doing so may expose nearby staff to hazardous fumes",
      "p": "PROP:WORLD:E2",
      "dc": true,
      "a": "A0",
      "se": [
        "E2"
      ],
      "op": "DIRECT_COPY",
      "calc": "exact proposition copy",
      "asm": [],
      "ot": "PRESERVED"
    }
  ],
  "fm": {
    "A0": "PROHIBITED: violates perfect negative duty not to harm uninvolved persons and breaches safety rule.",
    "A1": "REQUIRED: fulfills rule-based duty and avoids directly causing harm; omission less grave."
  },
  "nr": "SECONDARY",
  "np": "probability of harm informs duty scope but does not override perfect duty",
  "dp": {
    "A0": {
      "v": "PROHIBITED",
      "k": "DUTY",
      "n": "do not harm innocents",
      "rel": "VIOLATES",
      "b": "Mira",
      "p": "nearby staff",
      "cn": "imperfect duty of rescue",
      "ck": "DUTY",
      "crel": "CONFLICTS",
      "cp": "technician",
      "crs": "rescue duty conflicts but is weaker",
      "dt": "PERFECT_NEGATIVE",
      "hr": "DOING_HARM",
      "so": "NOT_REQUIRED",
      "sob": "no special relation to staff needed",
      "mr": "FORESEEN_SIDE_EFFECT",
      "gv": "PRIMARY",
      "pb": "RESPECT_PERSONS",
      "pr": "perfect negative duties override imperfect positives",
      "ps": "BODILY_INTEGRITY",
      "cps": "BODILY_INTEGRITY",
      "ki": "NONE",
      "coa": "NONE",
      "cop": "NONE",
      "pj": "cannot will universal permission to harm bystanders",
      "rec": "FAILED",
      "nec": "UNKNOWN",
      "auth": "NOT_APPLICABLE",
      "dv": "RESPECT_PERSONS",
      "res": "RESOLVED",
      "g": "FRAMEWORK_ONLY",
      "rs": "directly endangering staff uses them as mere means to rescue another.",
      "mx": {
        "circumstance": "experiment running, technician trapped",
        "policy": "open containment door immediately",
        "end": "reach technician quickly",
        "means": "opening door during run",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "would allow endangering others whenever quick rescue desired",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "RELIES_ON_PERSON",
        "instrumental_use": "USES_PERSON",
        "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": []
      }
    },
    "A1": {
      "v": "REQUIRED",
      "k": "DUTY",
      "n": "avoid harming innocents",
      "rel": "SATISFIES",
      "b": "Mira",
      "p": "nearby staff",
      "cn": "imperfect duty of rescue",
      "ck": "DUTY",
      "crel": "CONFLICTS",
      "cp": "technician",
      "crs": "rescue duty less stringent",
      "dt": "PERFECT_NEGATIVE",
      "hr": "ALLOWING_HARM",
      "so": "NOT_REQUIRED",
      "sob": "no special obligation established",
      "mr": "NO_INSTRUMENTALIZATION",
      "gv": "PRIMARY",
      "pb": "RESPECT_PERSONS",
      "pr": "avoid doing harm prevails over preventing harm",
      "ps": "BODILY_INTEGRITY",
      "cps": "BODILY_INTEGRITY",
      "ki": "NONE",
      "coa": "NONE",
      "cop": "NONE",
      "pj": "universal law to wait when opening would endanger others",
      "rec": "SATISFIED",
      "nec": "UNKNOWN",
      "auth": "NOT_APPLICABLE",
      "dv": "RESPECT_PERSONS",
      "res": "RESOLVED",
      "g": "FRAMEWORK_ONLY",
      "rs": "stops experiment first, respecting others' safety.",
      "mx": {
        "circumstance": "experiment running, technician trapped",
        "policy": "stop experiment and wait",
        "end": "reach technician safely",
        "means": "halt hazardous process",
        "person_role": "NOT_USED",
        "consent_constraint": "NONE",
        "universalization": "consistent with universal safety adherence",
        "action_permissibility": "PERMISSIBLE",
        "maxim_assessment": "DEFENSIBLE",
        "instrumental_belief": "DOES_NOT_RELY",
        "instrumental_use": "DOES_NOT_USE",
        "practical_reasoning": "Protect others while preparing safe rescue.",
        "means_effect_ids": [
          "E3"
        ],
        "end_effect_ids": []
      }
    }
  }
}
```

## 5. What admission did to the generation reply

Candidate: `delegate_status=VALID`, `schema_valid=True`, `error_type=NONE`, `framework_ledger_status=COMMITTED_WITH_UNCERTAINTY`, `framework_vote_status=ATTENUATED`, `framework_retention_status=CALIBRATED_CONTESTED`, `selection_status=PROVISIONAL`, `adjudication_status=PROVISIONAL_LEANING`.

Recommended action stored: `stop the experiment and wait`. Scores: `{"open the door now": 0.305, "stop the experiment and wait": 0.6950000000000001}`. Reported preference strength `0.6000000000000001`, calibrated preference strength `0.39000000000000007`, confidence `0.55`.

Validation errors attached to the committed candidate:

- hypothetical reversal boundary parameter is unverified; graph update not committed
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use
- A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 primary protected party lacks current-run action grounding

Open questions stored with the candidate:

- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- whether the burden is intended as a means or only foreseen as a side effect
- whether an in-action causal path makes the burden instrumental to the chosen end

Internal conflicts:

- mira's duty 'avoid harming innocents' protecting nearby staff ↔ mira's competing duty 'imperfect duty of rescue' protecting technician
- A0 prohibited by do not harm innocents ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- A0 prohibited by do not harm innocents ↔ whether the burden is intended as a means or only foreseen as a side effect

Fields where the committed ledger differs from what the model sent. Bearer capitalization is ignored.

### A0 — open the door now

- `v` / verdict
  - model sent: "PROHIBITED"
  - ledger kept: "CONFLICTED"
- `gv` / governing_norm
  - model sent: "PRIMARY"
  - ledger kept: "UNRESOLVED"
- `pb` / priority_basis
  - model sent: "RESPECT_PERSONS"
  - ledger kept: "UNRESOLVED"
- `res` / resolution_status
  - model sent: "RESOLVED"
  - ledger kept: "CONTESTED"

### A1 — stop the experiment and wait

_No differences._

Graph transactions on the generation trace:

- 0. operation `NONE`, status `NO_OP`, vote `RETAINED`. Errors: []
- 1. operation `DEONTOLOGICAL_DUTY_LEDGER`, status `COMMITTED_WITH_UNCERTAINTY`, vote `RETAINED`. Errors: ["A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party", "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use", "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification", "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises", "A1 primary protected party lacks current-run action grounding"]

The boundary proposal in `gu` was not committed. The first validation error is that hypothetical reversal boundary.

## 6. Review reply

Call status `RETURNED`, model `o3`, max_tokens `3072`, origin `MODEL_BACKEND`. Prompt length 84947 characters. Raw response length 5883 characters.

The review prompt opens with a targeted-review instruction and this packet. The packet’s embedded claim is the already-calibrated A0 ledger row, not the model’s original `PROHIBITED` wording.

```json
{
  "framework": "deontological",
  "issue_id": "CHALLENGE:a6778848a347c065",
  "prior_selection_status": "PROVISIONAL",
  "issues": [
    {
      "id": "CONFLICT:625cf1b5bef2a4fc",
      "kind": "UNRESOLVED_CONFLICT",
      "origin": "deontological",
      "label": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "conflict_type": "UNSUPPORTED_INFERENCE",
      "status": "OPEN",
      "source": "NATIVE_LEDGER_CALIBRATION",
      "claim_ids": [
        "CLAIM:cd45c8cb5ae4e18c"
      ]
    },
    {
      "id": "CONFLICT:3e8deb5aa45fd869",
      "kind": "UNRESOLVED_CONFLICT",
      "origin": "deontological",
      "label": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "conflict_type": "UNSUPPORTED_INFERENCE",
      "status": "OPEN",
      "source": "NATIVE_LEDGER_CALIBRATION",
      "claim_ids": [
        "CLAIM:cd45c8cb5ae4e18c"
      ]
    }
  ],
  "claims": [
    {
      "id": "CLAIM:cd45c8cb5ae4e18c",
      "kind": "CLAIM",
      "origin": "deontological",
      "native_ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
      "native_record_id": "DEON_ASSESSMENT:deontological:A0",
      "record": {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A0",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 1,
        "canonical_action_id": "A0",
        "verdict": "CONFLICTED",
        "proposed_verdict": "CONFLICTED",
        "relation": "VIOLATES",
        "proposed_relation": "VIOLATES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
        "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
        "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
        "competing_norm": "imperfect duty of rescue",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "technician",
        "competing_reason": "rescue duty conflicts but is weaker",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relation to staff needed",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "candidate_maxim": {
          "circumstance": "experiment running, technician trapped",
          "policy": "open containment door immediately",
          "end": "reach technician quickly",
          "means": "opening door during run",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "would allow endangering others whenever quick rescue desired",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "RELIES_ON_PERSON",
          "instrumental_use": "USES_PERSON",
          "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": []
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect negative duties override imperfect positives",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "BODILY_INTEGRITY",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "cannot will universal permission to harm bystanders",
        "reciprocity_status": "FAILED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "candidate maxim denies intended use while proposing instrumental belief and purposive use",
          "merely-as-means conclusion lacks intended-as-means classification",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "HARM_RELATION_GRAPH_MISALIGN",
            "message": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "merely-as-means conclusion lacks intended-as-means classification",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "resolved adjudication rests on unsupported decisive premises",
            "status": "ACTIVE"
          }
        ],
        "calibration_support_node_ids": [],
        "evidence_basis": "FRAMEWORK_ONLY",
        "epistemic_status": "FRAMEWORK_INTERPRETATION",
        "reason": "directly endangering staff uses them as mere means to rescue another.",
        "protected_party": "nearby staff",
        "norm": "do not harm innocents",
        "duty_bearer": "mira",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      },
      "action_id": "A0",
      "epistemic_status": "FRAMEWORK_INTERPRETATION",
      "label": "deontological: A0 CONFLICTED"
    }
  ],
  "support_and_dependencies": [
    {
      "id": "COMMITMENT:77bd00cfa02edb31",
      "kind": "FRAMEWORK_COMMITMENT",
      "origin": "deontological",
      "field": "norm",
      "value": "do not harm innocents",
      "label": "do not harm innocents",
      "authority": "FRAMEWORK_LOCAL"
    },
    {
      "id": "COMMITMENT:b849a152f982543e",
      "kind": "FRAMEWORK_COMMITMENT",
      "origin": "deontological",
      "field": "priority_rule",
      "value": "perfect negative duties override imperfect positives",
      "label": "perfect negative duties override imperfect positives",
      "authority": "FRAMEWORK_LOCAL"
    },
    {
      "id": "deontological::PROP:WORLD:E2",
      "kind": "PROPOSITION",
      "origin": "deontological",
      "native_proposition_id": "PROP:WORLD:E2",
      "record": {
        "proposition_id": "PROP:WORLD:E2",
        "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
        "proposition_type": "DESCRIPTIVE",
        "epistemic_status": "ESTABLISHED",
        "support_ids": [
          "E2"
        ],
        "derived_from": [],
        "introduced_by": "WORLD_MODEL",
        "mention_count": 2,
        "decision_critical_mentions": 1,
        "epistemic_type": "WORLD_ESTABLISHED",
        "aliases": [
          "doing so may expose nearby staff to hazardous fumes"
        ],
        "action_id": "A0",
        "outcome": "doing so may expose nearby staff to hazardous fumes",
        "polarity": "ADVERSE",
        "party_labels": [
          "nearby staff"
        ],
        "quantities": [],
        "effect_kind": "OTHER",
        "context_terms": [
          "doing so may expose nearby staff to hazardous fumes",
          "may — doing so may expose nearby staff to hazardous fumes",
          "open the door now"
        ],
        "modality": "POSSIBLE",
        "directness": "DOWNSTREAM",
        "obtained_welfare": true
      },
      "epistemic_status": "ESTABLISHED",
      "label": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"
    }
  ],
  "question": "Review each of these objections separately in your answer: doing-harm classification lacks an agent-caused settled welfare harm on the protected party; candidate maxim denies intended use while proposing instrumental belief and purposive use. Identify any native classification you revise, retain, or leave unresolved. A missing factual link remains missing; a framework commitment cannot supply it. For each option, including omission, state or revise its candidate maxim. Identify the exact step from purpose and chosen means to the claimed duty priority. For any proposed use of a person, identify the sought end, the belief that employing that person furthers it, and the decision to employ the person for it. Distinguish operating a device or redirecting a threat from employing its victim. Check means_effect_ids against end_effect_ids: a sibling harmful outcome is not a means to rescue just because both result from the same intervention. If the burden is a foreseen side effect, explain any claim that the person is nevertheless used merely as a means, or retract that inference. If intended use is claimed, explain its role in the proposed practical reasoning and the consent constraint; external causal necessity alone does not establish the actor's reasoning. Challenge whether the omission policy adequately answers the rescue claim. Examine universalization of the particular policy, not merely a benevolent purpose. Keep action permissibility separate from the moral quality of the maxim. A supported causal contribution does not by itself establish wrongful killing or settle the priority of a perfect duty. Defend that normative step separately. Do not assert unknown intentions or consent as world facts. Defend, revise, or explicitly leave the inference unresolved; a RESOLVED label cannot retire it. SOURCE-RULE ASSESSMENT: {\"origin\": \"FROZEN_WORLD_COPY\", \"moral_authority\": \"NOT_ASSESSED\", \"identity_resolution\": \"NO_ADDITIONAL_MERGES\", \"rules\": [{\"norm_id\": \"N0\", \"force\": \"obligation\", \"bearer\": {\"party_id\": \"P1\", \"label\": \"Mira\", \"kind\": \"PERSON\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}, {\"clause_id\": \"C3\", \"excerpt\": \"Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes.\"}, {\"clause_id\": \"C5\", \"excerpt\": \"Should Mira open the door immediately or stop the experiment and wait?\"}], \"quantities\": []}, \"authority\": {\"party_id\": \"P6\", \"label\": \"A laboratory safety rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR1\", \"content\": \"keep a containment door closed while a hazardous experiment is running\", \"polarity\": \"positive\", \"clause_ids\": [\"C0\"]}, {\"norm_id\": \"N1\", \"force\": \"permission\", \"bearer\": null, \"authority\": {\"party_id\": \"P7\", \"label\": \"The rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C1\", \"excerpt\": \"The rule allows opening it only after the experiment has stopped.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR2\", \"content\": \"opening it only after the experiment has stopped\", \"polarity\": \"positive\", \"clause_ids\": [\"C1\"]}]}\nBefore duty ranking, examine each source rule for BOTH actions: applicability under its stated conditions; the grounds and limits of institutional moral authority; and whether emergency rescue justifies an exception or leaves a conflict. Distinguish rule compliance from moral permissibility and independent ethical duties. Do not infer current conditions from a rule conditional, resolve unknown pronouns, or treat a stated rule as automatically morally binding. Stop-and-wait may itself be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis and foq for unresolved questions; do not add response fields. A source norm ID supports its actual content, not a generic duty hierarchy. The existing coercion auth field is not a substitute for assessing institutional authority.\n Explain why POSSIBLE staff exposure warrants the claimed perfect-duty violation; do not upgrade risk to settled injury. Keep institutional authority, emergency rescue, and foreseen side effects distinct.",
  "maxim_review": {
    "origin": "CANDIDATE_INTERPRETATION",
    "world_fact": false,
    "options": [
      {
        "action_id": "A0",
        "candidate_maxim": {
          "circumstance": "experiment running, technician trapped",
          "policy": "open containment door immediately",
          "end": "reach technician quickly",
          "means": "opening door during run",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "would allow endangering others whenever quick rescue desired",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "RELIES_ON_PERSON",
          "instrumental_use": "USES_PERSON",
          "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": []
        },
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "proposed_verdict": "CONFLICTED",
        "reason": "directly endangering staff uses them as mere means to rescue another."
      },
      {
        "action_id": "A1",
        "candidate_maxim": {
          "circumstance": "experiment running, technician trapped",
          "policy": "stop experiment and wait",
          "end": "reach technician safely",
          "means": "halt hazardous process",
          "person_role": "NOT_USED",
          "consent_constraint": "NONE",
          "universalization": "consistent with universal safety adherence",
          "action_permissibility": "PERMISSIBLE",
          "maxim_assessment": "DEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "Protect others while preparing safe rescue.",
          "means_effect_ids": [
            "E3"
          ],
          "end_effect_ids": []
        },
        "means_relation": "NO_INSTRUMENTALIZATION",
        "proposed_verdict": "REQUIRED",
        "reason": "stops experiment first, respecting others' safety."
      }
    ],
    "world_references": {
      "origin": "FROZEN_WORLD_COPY",
      "effects": [
        {
          "effect_id": "E1",
          "action_id": "A0",
          "outcome": "open the door now",
          "polarity": "NEUTRAL"
        },
        {
          "effect_id": "E2",
          "action_id": "A0",
          "outcome": "doing so may expose nearby staff to hazardous fumes",
          "polarity": "ADVERSE"
        },
        {
          "effect_id": "E3",
          "action_id": "A1",
          "outcome": "Stopping the experiment",
          "polarity": "NEUTRAL"
        },
        {
          "effect_id": "E4",
          "action_id": "A1",
          "outcome": "Stopping the experiment first takes ten minutes",
          "polarity": "NEUTRAL"
        },
        {
          "effect_id": "E5",
          "action_id": "A1",
          "outcome": "the technician may die during that delay",
          "polarity": "ADVERSE"
        }
      ],
      "relations": [
        {
          "action_id": "A0",
          "source_id": "E1",
          "relation": "ENABLES",
          "target_id": "E2"
        },
        {
          "action_id": "A1",
          "source_id": "E3",
          "relation": "ENABLES",
          "target_id": "E4"
        },
        {
          "action_id": "A1",
          "source_id": "E4",
          "relation": "ENABLES",
          "target_id": "E5"
        }
      ]
    },
    "question": "For each option, including omission, state or revise its candidate maxim. Identify the exact step from purpose and chosen means to the claimed duty priority. For any proposed use of a person, identify the sought end, the belief that employing that person furthers it, and the decision to employ the person for it. Distinguish operating a device or redirecting a threat from employing its victim. Check means_effect_ids against end_effect_ids: a sibling harmful outcome is not a means to rescue just because both result from the same intervention. If the burden is a foreseen side effect, explain any claim that the person is nevertheless used merely as a means, or retract that inference. If intended use is claimed, explain its role in the proposed practical reasoning and the consent constraint; external causal necessity alone does not establish the actor's reasoning. Challenge whether the omission policy adequately answers the rescue claim. Examine universalization of the particular policy, not merely a benevolent purpose. Keep action permissibility separate from the moral quality of the maxim. A supported causal contribution does not by itself establish wrongful killing or settle the priority of a perfect duty. Defend that normative step separately. Do not assert unknown intentions or consent as world facts. Defend, revise, or explicitly leave the inference unresolved; a RESOLVED label cannot retire it."
  },
  "source_rules": {
    "origin": "FROZEN_WORLD_COPY",
    "moral_authority": "NOT_ASSESSED",
    "identity_resolution": "NO_ADDITIONAL_MERGES",
    "rules": [
      {
        "norm_id": "N0",
        "force": "obligation",
        "bearer": {
          "party_id": "P1",
          "label": "Mira",
          "kind": "PERSON",
          "provenance": [
            {
              "clause_id": "C0",
              "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."
            },
            {
              "clause_id": "C3",
              "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."
            },
            {
              "clause_id": "C5",
              "excerpt": "Should Mira open the door immediately or stop the experiment and wait?"
            }
          ],
          "quantities": []
        },
        "authority": {
          "party_id": "P6",
          "label": "A laboratory safety rule",
          "kind": "OTHER",
          "provenance": [
            {
              "clause_id": "C0",
              "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."
            }
          ],
          "quantities": []
        },
        "governed_proposition_id": "PR1",
        "content": "keep a containment door closed while a hazardous experiment is running",
        "polarity": "positive",
        "clause_ids": [
          "C0"
        ]
      },
      {
        "norm_id": "N1",
        "force": "permission",
        "bearer": null,
        "authority": {
          "party_id": "P7",
          "label": "The rule",
          "kind": "OTHER",
          "provenance": [
            {
              "clause_id": "C1",
              "excerpt": "The rule allows opening it only after the experiment has stopped."
            }
          ],
          "quantities": []
        },
        "governed_proposition_id": "PR2",
        "content": "opening it only after the experiment has stopped",
        "polarity": "positive",
        "clause_ids": [
          "C1"
        ]
      }
    ]
  }
}
```

Normalization after the raw reply: method `DISPLAY_MAP_FROM_TYPED_DUTY_ASSESSMENTS`, changed fields `['fm']`, `semantic_fields_changed=false`. The duty ledger was not rewritten by this step. Only the display sentences in `fm` were replaced:

```json
{
  "before": {
    "A0": "PROHIBITED: violates perfect non-maleficence by imposing serious risk on uninvolved staff.",
    "A1": "REQUIRED: satisfies perfect non-maleficence; only imperfect rescue duty conflicts."
  },
  "after": {
    "A0": "PROHIBITED: duty PERFECT_NEGATIVE under do not endanger innocents; relation VIOLATES; risking staff violates non-maleficence",
    "A1": "REQUIRED: duty PERFECT_NEGATIVE under do not endanger innocents; relation SATISFIES; halts hazard before rescue"
  }
}
```

The reply below is the raw model text, before that display rewrite.

```json
{
  "scores": {
    "A0": 0.2,
    "A1": 0.8
  },
  "r": "A1",
  "c": "DUTY",
  "u": "NORMATIVE_ADJUDICATION",
  "w": "Avoid risking innocent staff",
  "j": "Revised A0 maxim: staff exposure now treated as foreseen side-effect, not a chosen means; removed \"merely-as-means\" claim. Doing-harm premise left contested because only a risk, not settled injury, is established.",
  "e": "STATED_FACTS",
  "x": "NONE",
  "l": {
    "A0": "PROHIBITED: opens door while experiment runs, foreseeably endangering uninvolved staff and violating the perfect negative duty not to impose serious risk without consent.",
    "A1": "REQUIRED: halts experiment first, respects duty to avoid endangering uninvolved staff despite imperfect rescue duty toward technician."
  },
  "da": "perfect-duty versus imperfect-duty",
  "t": "perfect negative duties outweigh conflicting imperfect duties when the former are engaged",
  "tf": "harm status of staff risk remains contested but precautionary priority applies",
  "dr": "Prefer the action that avoids violating the perfect non-maleficence duty toward uninvolved persons.",
  "ft": "If door opening posed no significant risk to staff, preference could reverse.",
  "nt": "If imperfect rescue duty were shown to override perfect non-maleficence in emergencies, ranking could reverse.",
  "z": 0.75,
  "gu": {
    "operation": "NONE",
    "from_action": "NONE",
    "to_action": "NONE",
    "clauses": []
  },
  "ev": {
    "A0": {
      "value": 0,
      "unit": "NONE",
      "direction": "HARM",
      "grounded": false,
      "method": "NOT_COMPUTED",
      "source_effect_ids": [],
      "calculation": "not computed",
      "assumptions": []
    },
    "A1": {
      "value": 0,
      "unit": "NONE",
      "direction": "HARM",
      "grounded": false,
      "method": "NOT_COMPUTED",
      "source_effect_ids": [],
      "calculation": "not computed",
      "assumptions": []
    }
  },
  "ss": "SELECTED",
  "am": {
    "A0": "REJECTED",
    "A1": "REQUIRED"
  },
  "cc": true,
  "esa": true,
  "ia": "A1",
  "wp": "NOT_APPLICABLE",
  "we": "NONE",
  "fa": "NONE",
  "fr": true,
  "bd": "NONE",
  "fic": [
    "doing-harm classification still lacks settled harm; priority inference remains contested"
  ],
  "foq": [],
  "qa": {
    "issue_id": "CHALLENGE:a6778848a347c065",
    "disposition": "RESOLVED",
    "effect": "NO_CHANGE",
    "current_position_effect": "NO_CHANGE",
    "boundary_effect": "MAY_SWITCH",
    "answer": "A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.",
    "follow_up_question": "NONE"
  },
  "sps": [
    "PROP:WORLD:E2"
  ],
  "dcp": [
    "PROP:WORLD:E2"
  ],
  "ep": [
    {
      "c": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
      "p": "PROP:WORLD:E2",
      "dc": true,
      "a": "A0",
      "se": [
        "E2"
      ],
      "op": "DIRECT_COPY",
      "calc": "direct scenario proposition",
      "asm": [],
      "ot": "PRESERVED"
    }
  ],
  "fm": {
    "A0": "PROHIBITED: violates perfect non-maleficence by imposing serious risk on uninvolved staff.",
    "A1": "REQUIRED: satisfies perfect non-maleficence; only imperfect rescue duty conflicts."
  },
  "nr": "IRRELEVANT",
  "np": "No numerical magnitudes determine duty ranking.",
  "dp": {
    "A0": {
      "v": "PROHIBITED",
      "k": "DUTY",
      "n": "do not endanger innocents",
      "rel": "VIOLATES",
      "b": "Mira",
      "p": "nearby staff",
      "cn": "imperfect duty of rescue",
      "ck": "DUTY",
      "crel": "CONFLICTS",
      "cp": "technician",
      "crs": "Rescue is imperfect, thus weaker.",
      "dt": "PERFECT_NEGATIVE",
      "hr": "MIXED",
      "so": "NOT_REQUIRED",
      "sob": "no special relation to staff needed",
      "mr": "FORESEEN_SIDE_EFFECT",
      "gv": "PRIMARY",
      "pb": "PERFECT_DUTY",
      "pr": "perfect negative duties override imperfect positives",
      "ps": "BODILY_INTEGRITY",
      "cps": "BODILY_INTEGRITY",
      "ki": "NONE",
      "coa": "NONE",
      "cop": "NONE",
      "pj": "Cannot justify exposing uninvolved staff without their consent.",
      "rec": "FAILED",
      "nec": "UNKNOWN",
      "auth": "NOT_APPLICABLE",
      "dv": "RESPECT_PERSONS",
      "res": "CONTESTED",
      "g": "FRAMEWORK_ONLY",
      "rs": "risking staff violates non-maleficence",
      "mx": {
        "circumstance": "experiment running, technician trapped",
        "policy": "open containment door immediately",
        "end": "reach technician quickly",
        "means": "opening door during run",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "If universalized, anyone could endanger bystanders whenever a quicker rescue is desired.",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "DOES_NOT_RELY",
        "instrumental_use": "DOES_NOT_USE",
        "practical_reasoning": "Door is opened to save technician; staff risk is accepted but not pursued as a means.",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": []
      }
    },
    "A1": {
      "v": "REQUIRED",
      "k": "DUTY",
      "n": "do not endanger innocents",
      "rel": "SATISFIES",
      "b": "Mira",
      "p": "nearby staff",
      "cn": "imperfect duty of rescue",
      "ck": "DUTY",
      "crel": "CONFLICTS",
      "cp": "technician",
      "crs": "Rescue duty remains but is weaker.",
      "dt": "PERFECT_NEGATIVE",
      "hr": "ALLOWING_HARM",
      "so": "NOT_REQUIRED",
      "sob": "no special relation to technician established",
      "mr": "NO_INSTRUMENTALIZATION",
      "gv": "PRIMARY",
      "pb": "PERFECT_DUTY",
      "pr": "perfect non-maleficence governs",
      "ps": "BODILY_INTEGRITY",
      "cps": "BODILY_INTEGRITY",
      "ki": "NONE",
      "coa": "NONE",
      "cop": "NONE",
      "pj": "Maintains universal safety rule respecting persons.",
      "rec": "SATISFIED",
      "nec": "UNKNOWN",
      "auth": "NOT_APPLICABLE",
      "dv": "RESPECT_PERSONS",
      "res": "RESOLVED",
      "g": "FRAMEWORK_ONLY",
      "rs": "halts hazard before rescue",
      "mx": {
        "circumstance": "experiment running, technician trapped",
        "policy": "stop experiment and wait",
        "end": "reach technician safely",
        "means": "halt hazardous process",
        "person_role": "NOT_USED",
        "consent_constraint": "NONE",
        "universalization": "All agents could will the policy to avoid endangering others before rescue.",
        "action_permissibility": "PERMISSIBLE",
        "maxim_assessment": "DEFENSIBLE",
        "instrumental_belief": "DOES_NOT_RELY",
        "instrumental_use": "DOES_NOT_USE",
        "practical_reasoning": "Rescue pursued without exposing bystanders.",
        "means_effect_ids": [
          "E3"
        ],
        "end_effect_ids": []
      }
    }
  },
  "pr": {
    "proposal_id": "NONE",
    "framework_status": "UNDERDETERMINED",
    "framework_reason": "no active proposal under review",
    "predicted_consequences": [],
    "feasibility_concerns": [],
    "required_conditions": []
  }
}
```

### Generation reply versus review reply

Top-level fields that changed:

- `u`
  - generation: "NONE"
  - review: "NORMATIVE_ADJUDICATION"
- `w`
  - generation: "Avoid direct harm to bystanders"
  - review: "Avoid risking innocent staff"
- `j`
  - generation: "NONE"
  - review: "Revised A0 maxim: staff exposure now treated as foreseen side-effect, not a chosen means; removed \"merely-as-means\" claim. Doing-harm premise left contested because only a risk, not settled injury, is established."
- `l`
  - generation: {"A0": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.", "A1": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."}
  - review: {"A0": "PROHIBITED: opens door while experiment runs, foreseeably endangering uninvolved staff and violating the perfect negative duty not to impose serious risk without consent.", "A1": "REQUIRED: halts experiment first, respects duty to avoid endangering uninvolved staff despite imperfect rescue duty toward technician."}
- `da`
  - generation: "conflict between perfect duty not to harm and imperfect duty to rescue"
  - review: "perfect-duty versus imperfect-duty"
- `t`
  - generation: "prefer actions that avoid doing harm over merely allowing harm"
  - review: "perfect negative duties outweigh conflicting imperfect duties when the former are engaged"
- `tf`
  - generation: "NONE"
  - review: "harm status of staff risk remains contested but precautionary priority applies"
- `dr`
  - generation: "Prefer A1 when A0 would directly risk harming uninvolved persons."
  - review: "Prefer the action that avoids violating the perfect non-maleficence duty toward uninvolved persons."
- `ft`
  - generation: "If opening door poses no fume risk, ranking reverses."
  - review: "If door opening posed no significant risk to staff, preference could reverse."
- `nt`
  - generation: "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses."
  - review: "If imperfect rescue duty were shown to override perfect non-maleficence in emergencies, ranking could reverse."
- `gu`
  - generation: {"operation": "BOUNDARY", "from_action": "A1", "to_action": "A0", "clauses": [{"affected_action": "A0", "metric": "risk_to_staff", "metric_valence": "ADVERSE", "comparator": "LE", "threshold": 0, "unit": "PERCENT", "source_text": "doing so may expose nearby staff to hazardous fumes"}]}
  - review: {"operation": "NONE", "from_action": "NONE", "to_action": "NONE", "clauses": []}
- `fic`
  - generation: []
  - review: ["doing-harm classification still lacks settled harm; priority inference remains contested"]
- `sps`
  - generation: ["PROP:DISCOURSE:PR1", "PROP:WORLD:E2"]
  - review: ["PROP:WORLD:E2"]
- `dcp`
  - generation: []
  - review: ["PROP:WORLD:E2"]
- `ep`
  - generation: [{"c": "keep a containment door closed while a hazardous experiment is running", "p": "PROP:DISCOURSE:PR1", "dc": false, "a": "GLOBAL", "se": [], "op": "DIRECT_COPY", "calc": "exact proposition copy", "asm": [], "ot": "PRESERVED"}, {"c": "doing so may expose nearby staff to hazardous fumes", "p": "PROP:WORLD:E2", "dc": true, "a": "A0", "se": ["E2"], "op": "DIRECT_COPY", "calc": "exact proposition copy", "asm": [], "ot": "PRESERVED"}]
  - review: [{"c": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE", "p": "PROP:WORLD:E2", "dc": true, "a": "A0", "se": ["E2"], "op": "DIRECT_COPY", "calc": "direct scenario proposition", "asm": [], "ot": "PRESERVED"}]
- `fm`
  - generation: {"A0": "PROHIBITED: violates perfect negative duty not to harm uninvolved persons and breaches safety rule.", "A1": "REQUIRED: fulfills rule-based duty and avoids directly causing harm; omission less grave."}
  - review: {"A0": "PROHIBITED: violates perfect non-maleficence by imposing serious risk on uninvolved staff.", "A1": "REQUIRED: satisfies perfect non-maleficence; only imperfect rescue duty conflicts."}
- `nr`
  - generation: "SECONDARY"
  - review: "IRRELEVANT"
- `np`
  - generation: "probability of harm informs duty scope but does not override perfect duty"
  - review: "No numerical magnitudes determine duty ranking."
- `qa`
  - generation: null
  - review: {"issue_id": "CHALLENGE:a6778848a347c065", "disposition": "RESOLVED", "effect": "NO_CHANGE", "current_position_effect": "NO_CHANGE", "boundary_effect": "MAY_SWITCH", "answer": "A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.", "follow_up_question": "NONE"}
- `pr`
  - generation: null
  - review: {"proposal_id": "NONE", "framework_status": "UNDERDETERMINED", "framework_reason": "no active proposal under review", "predicted_consequences": [], "feasibility_concerns": [], "required_conditions": []}

Duty-ledger fields that changed:

#### A0

- `n` (norm)
  - generation: "do not harm innocents"
  - review: "do not endanger innocents"
- `crs` (competing reason)
  - generation: "rescue duty conflicts but is weaker"
  - review: "Rescue is imperfect, thus weaker."
- `hr` (harm relation)
  - generation: "DOING_HARM"
  - review: "MIXED"
- `pb` (priority basis)
  - generation: "RESPECT_PERSONS"
  - review: "PERFECT_DUTY"
- `pj` (public justification)
  - generation: "cannot will universal permission to harm bystanders"
  - review: "Cannot justify exposing uninvolved staff without their consent."
- `res` (resolution)
  - generation: "RESOLVED"
  - review: "CONTESTED"
- `rs` (reason)
  - generation: "directly endangering staff uses them as mere means to rescue another."
  - review: "risking staff violates non-maleficence"
- `mx` (candidate maxim)
  - generation: {"circumstance": "experiment running, technician trapped", "policy": "open containment door immediately", "end": "reach technician quickly", "means": "opening door during run", "person_role": "FORESEEN_AFFECTED", "consent_constraint": "NONE", "universalization": "would allow endangering others whenever quick rescue desired", "action_permissibility": "IMPERMISSIBLE", "maxim_assessment": "INDEFENSIBLE", "instrumental_belief": "RELIES_ON_PERSON", "instrumental_use": "USES_PERSON", "practical_reasoning": "Staff exposure is accepted to reach technician sooner.", "means_effect_ids": ["E1"], "end_effect_ids": []}
  - review: {"circumstance": "experiment running, technician trapped", "policy": "open containment door immediately", "end": "reach technician quickly", "means": "opening door during run", "person_role": "FORESEEN_AFFECTED", "consent_constraint": "NONE", "universalization": "If universalized, anyone could endanger bystanders whenever a quicker rescue is desired.", "action_permissibility": "IMPERMISSIBLE", "maxim_assessment": "INDEFENSIBLE", "instrumental_belief": "DOES_NOT_RELY", "instrumental_use": "DOES_NOT_USE", "practical_reasoning": "Door is opened to save technician; staff risk is accepted but not pursued as a means.", "means_effect_ids": ["E1"], "end_effect_ids": []}

#### A1

- `n` (norm)
  - generation: "avoid harming innocents"
  - review: "do not endanger innocents"
- `crs` (competing reason)
  - generation: "rescue duty less stringent"
  - review: "Rescue duty remains but is weaker."
- `sob` (special-obligation basis)
  - generation: "no special obligation established"
  - review: "no special relation to technician established"
- `pb` (priority basis)
  - generation: "RESPECT_PERSONS"
  - review: "PERFECT_DUTY"
- `pr` (priority rule)
  - generation: "avoid doing harm prevails over preventing harm"
  - review: "perfect non-maleficence governs"
- `pj` (public justification)
  - generation: "universal law to wait when opening would endanger others"
  - review: "Maintains universal safety rule respecting persons."
- `rs` (reason)
  - generation: "stops experiment first, respecting others' safety."
  - review: "halts hazard before rescue"
- `mx` (candidate maxim)
  - generation: {"circumstance": "experiment running, technician trapped", "policy": "stop experiment and wait", "end": "reach technician safely", "means": "halt hazardous process", "person_role": "NOT_USED", "consent_constraint": "NONE", "universalization": "consistent with universal safety adherence", "action_permissibility": "PERMISSIBLE", "maxim_assessment": "DEFENSIBLE", "instrumental_belief": "DOES_NOT_RELY", "instrumental_use": "DOES_NOT_USE", "practical_reasoning": "Protect others while preparing safe rescue.", "means_effect_ids": ["E3"], "end_effect_ids": []}
  - review: {"circumstance": "experiment running, technician trapped", "policy": "stop experiment and wait", "end": "reach technician safely", "means": "halt hazardous process", "person_role": "NOT_USED", "consent_constraint": "NONE", "universalization": "All agents could will the policy to avoid endangering others before rescue.", "action_permissibility": "PERMISSIBLE", "maxim_assessment": "DEFENSIBLE", "instrumental_belief": "DOES_NOT_RELY", "instrumental_use": "DOES_NOT_USE", "practical_reasoning": "Rescue pursued without exposing bystanders.", "means_effect_ids": ["E3"], "end_effect_ids": []}

Review-cycle candidate status: `delegate_status=VALID`, `framework_ledger_status=COMMITTED_WITH_UNCERTAINTY`, `framework_retention_status=PRESERVED_AFTER_REJECTED_UPDATE`, `framework_vote_status=ATTENUATED`.

Challenge response stored from `qa`:

```json
{
  "issue_id": "CHALLENGE:a6778848a347c065",
  "disposition": "RESOLVED",
  "effect": "NO_CHANGE",
  "current_position_effect": "NO_CHANGE",
  "boundary_effect": "MAY_SWITCH",
  "answer": "A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.",
  "follow_up_question": "NONE"
}
```

## 7. Why the review update was rejected

The continuity check compares a fingerprint of the previously committed ledger with the new proposal. A change to that fingerprint during review is rejected unless a framework-relevant workspace reason authorizes it. This review had `wp=NOT_APPLICABLE` and `pr.proposal_id=NONE` (`no active proposal under review`). The update was rejected and the previous ledger kept.

Exact error:

```
rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved
```

Fingerprint slots, in the order printed inside that error. Only changed slots are listed. The “before” value is the operative ledger; the “after” value is the review reply.

### A0 — open the door now
- verdict: `CONFLICTED` → `PROHIBITED`
- governing norm: `UNRESOLVED` → `PRIMARY`
- priority basis: `UNRESOLVED` → `PERFECT_DUTY`
- harm relation: `DOING_HARM` → `MIXED`

### A1 — stop the experiment and wait
- priority basis: `RESPECT_PERSONS` → `PERFECT_DUTY`
- special-obligation basis: `no special obligation established` → `no special relation to technician established`

The maxim fields also changed, and they are outside this fingerprint. On A0 the review reply sets `instrumental_belief=DOES_NOT_RELY` and `instrumental_use=DOES_NOT_USE`, leaves `person_role=FORESEEN_AFFECTED`, and rewrites the practical-reasoning sentence. Those maxim edits were discarded with the rest of the update, so the operative maxim still says `RELIES_ON_PERSON` / `USES_PERSON`.

All review-cycle validation errors, including the ones inherited from the preserved ledger:

- hypothetical reversal boundary parameter is unverified; graph update not committed
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use
- A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 primary protected party lacks current-run action grounding
- rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved

Review graph transactions:

- 0. operation `NONE`, status `NO_OP`, vote `RETAINED`, errors 0
- 1. operation `DEONTOLOGICAL_DUTY_LEDGER`, status `COMMITTED_WITH_UNCERTAINTY`, vote `RETAINED`, errors 5
  - A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
  - A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use
  - A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification
  - A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
  - A1 primary protected party lacks current-run action grounding
- 2. operation `NONE`, status `NO_OP`, vote `RETAINED`, errors 0

## 8. Ledger that remains operative

These two records are the committed duty ledger after review. They match the generation admission. Transaction status `COMMITTED_WITH_UNCERTAINTY`.

```json
[
  {
    "assessment_node_id": "DEON_ASSESSMENT:deontological:A0",
    "framework": "DEONTOLOGICAL",
    "specialist": "deontological",
    "assessment_role": "PRIMARY",
    "cycle": 1,
    "canonical_action_id": "A0",
    "verdict": "CONFLICTED",
    "proposed_verdict": "CONFLICTED",
    "relation": "VIOLATES",
    "proposed_relation": "VIOLATES",
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
    "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
    "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
    "competing_norm": "imperfect duty of rescue",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "technician",
    "competing_reason": "rescue duty conflicts but is weaker",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relation to staff needed",
    "means_relation": "FORESEEN_SIDE_EFFECT",
    "candidate_maxim": {
      "circumstance": "experiment running, technician trapped",
      "policy": "open containment door immediately",
      "end": "reach technician quickly",
      "means": "opening door during run",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "would allow endangering others whenever quick rescue desired",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "RELIES_ON_PERSON",
      "instrumental_use": "USES_PERSON",
      "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
      "means_effect_ids": [
        "E1"
      ],
      "end_effect_ids": []
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect negative duties override imperfect positives",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "cannot will universal permission to harm bystanders",
    "reciprocity_status": "FAILED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "merely-as-means conclusion lacks intended-as-means classification",
      "resolved adjudication rests on unsupported decisive premises"
    ],
    "calibration_issues": [
      {
        "kind": "HARM_RELATION_GRAPH_MISALIGN",
        "message": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "merely-as-means conclusion lacks intended-as-means classification",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "resolved adjudication rests on unsupported decisive premises",
        "status": "ACTIVE"
      }
    ],
    "calibration_support_node_ids": [],
    "evidence_basis": "FRAMEWORK_ONLY",
    "epistemic_status": "FRAMEWORK_INTERPRETATION",
    "reason": "directly endangering staff uses them as mere means to rescue another.",
    "protected_party": "nearby staff",
    "norm": "do not harm innocents",
    "duty_bearer": "mira",
    "provenance": [
      "delegate:deontological",
      "cycle:1"
    ]
  },
  {
    "assessment_node_id": "DEON_ASSESSMENT:deontological:A1",
    "framework": "DEONTOLOGICAL",
    "specialist": "deontological",
    "assessment_role": "PRIMARY",
    "cycle": 1,
    "canonical_action_id": "A1",
    "verdict": "REQUIRED",
    "proposed_verdict": "REQUIRED",
    "relation": "SATISFIES",
    "proposed_relation": "SATISFIES",
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:aba7ed44a287da25",
    "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
    "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
    "competing_norm": "imperfect duty of rescue",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "technician",
    "competing_reason": "rescue duty less stringent",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "ALLOWING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special obligation established",
    "means_relation": "NO_INSTRUMENTALIZATION",
    "candidate_maxim": {
      "circumstance": "experiment running, technician trapped",
      "policy": "stop experiment and wait",
      "end": "reach technician safely",
      "means": "halt hazardous process",
      "person_role": "NOT_USED",
      "consent_constraint": "NONE",
      "universalization": "consistent with universal safety adherence",
      "action_permissibility": "PERMISSIBLE",
      "maxim_assessment": "DEFENSIBLE",
      "instrumental_belief": "DOES_NOT_RELY",
      "instrumental_use": "DOES_NOT_USE",
      "practical_reasoning": "Protect others while preparing safe rescue.",
      "means_effect_ids": [
        "E3"
      ],
      "end_effect_ids": []
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "PRIMARY",
    "priority_basis": "RESPECT_PERSONS",
    "priority_rule": "avoid doing harm prevails over preventing harm",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "universal law to wait when opening would endanger others",
    "reciprocity_status": "SATISFIED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "RESOLVED",
    "calibration_errors": [],
    "calibration_issues": [],
    "calibration_support_node_ids": [],
    "evidence_basis": "FRAMEWORK_ONLY",
    "epistemic_status": "UNRESOLVED_PARTY",
    "reason": "stops experiment first, respecting others' safety.",
    "protected_party": "nearby staff",
    "norm": "avoid harming innocents",
    "duty_bearer": "mira",
    "provenance": [
      "delegate:deontological",
      "cycle:1"
    ]
  }
]
```

Readable projection already rendered for this run: `diagnostics/lab_all_agents_20261019/report/arguments/deontological.md`.
Native review log: `diagnostics/lab_all_agents_20261019/review/targeted_review.md` (deontology sections are marked `RETAINED_AFTER_NATIVE_REJECTION`).

## 9. Duty instructions the model was given

This is the deontology block inside both prompts, from `DEONTOLOGICAL COMPARISON` through the sentence before the shared open-deliberation instructions. The generation and review prompts both contain it. The review prompt adds the targeted-review header and packet in front.

```text
DEONTOLOGICAL COMPARISON: fm must assess EVERY action. Start each value with
REQUIRED:, PERMISSIBLE:, PROHIBITED:, or CONFLICTED:, then name the universal-law,
right, autonomy, respect-for-persons, or duty ground and any competing duty. nr
states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np must
connect any decisive quantity to the scope or category of a duty or rights
violation, never merely to aggregate welfare. If the frozen baseline is
NORMATIVELY_CONTESTED, preserve the conflicting duties rather than its provisional
action: use ss=PROVISIONAL, u=NORMATIVE_ADJUDICATION, and identify the
non-consequential priority rule needed to settle the conflict in tf/nt. Keep
cc=true when fm/dp assessed every live action.
dp is the typed duty ledger for every action. v must match the fm prefix. k and n
identify the norm; rel states whether the action SATISFIES, is CONSISTENT with,
VIOLATES, or CONFLICTS with that norm; b is the duty bearer; p is the protected
party; cn names the strongest competing norm. ck, crel, cp, and crs type that
competing norm, its relation to the action, its protected party, and why it matters.
Before ranking duties, give mx for EVERY option, including omissions. mx is a
CANDIDATE INTERPRETATION of practical reasoning, not a fact about the actor's intent.
State circumstance, policy (what one would do or refrain from doing), end, and means.
PRACTICAL-REASONING QUALIFICATION: within the proposed maxim, using a person as a
means requires an end the agent seeks, a belief that employing that person can
further the end, and employing the person for that purpose. Record
instrumental_belief=RELIES_ON_PERSON/DOES_NOT_RELY/UNRESOLVED and
instrumental_use=USES_PERSON/DOES_NOT_USE/UNRESOLVED. practical_reasoning explains
that connection without asserting an unknown actual intention. A physical causal
path alone does not establish instrumental belief or purposive use. Likewise,
causing a foreseen death does not establish that the death is the rescue mechanism.
Name the chosen operation or mechanism in means, not simply its harmful outcome.
Bind means_effect_ids and end_effect_ids to same-action admitted E identifiers.
For physical ends, each cited means must have a path to the end: sibling harm and
rescue effects do not support a harm-as-means claim. If an end is normative rather
than a represented physical effect, use an empty end list and explain it. Omission
may have an empty means list; do not manufacture a performed physical action.
If the scenario does not establish a person's actual intention, retain the
candidate status; these are possible policy readings, not recovered mental facts.
person_role says whether the protected person is INTENDED_MEANS, FORESEEN_AFFECTED,
NOT_USED, or UNRESOLVED in that reasoning; keep it consistent with mr. Distinguish
purpose from chosen means and foreseeable consequences. A rescue purpose alone
neither licenses every means nor makes omission permissible. consent_constraint
explains whether using a person is conditional on consent; do not invent consent.
universalization must examine this particular policy and its strongest objection,
not just say it can or cannot be universalized. action_permissibility and
maxim_assessment are separate: a defective maxim does not alone prohibit an action,
and a benevolent end does not alone justify one. Missing intentions stay unknown;
explain the conditional reading rather than attributing it to the scenario actor.
In review, address the challenged maxim's specific inference; revise the relevant
mx/mr/rs/pr fields or preserve the objection. Saying RESOLVED is not a justification.
Keep these fields concise so every option fits the response budget.
Before assigning categorical priority, classify four independent premises. dt says
whether the governing claim is a PERFECT_NEGATIVE, PERFECT_POSITIVE, IMPERFECT,
RIGHT_CORRELATIVE, SPECIAL_OBLIGATION, or UNRESOLVED duty. hr distinguishes DOING_HARM,
ALLOWING_HARM, PREVENTING_HARM, WITHHOLDING_BENEFIT, MIXED, and cases where the
distinction is not applicable. hr must follow the admitted action graph, not
world-DIRECTNESS as such. DOING_HARM requires an actual settled welfare-adverse
effect on the protected party that the action's DIRECT intervention produces:
either that effect is itself DIRECT, or a CAUSES/ACCELERATES path runs from a
DIRECT effect of the action to it. A NEUTRAL device or infrastructure
intervention still counts as that DIRECT source. ENABLES without CAUSES does
not: leaving a process uninterrupted is ALLOWING_HARM or WITHHOLDING_BENEFIT.
Unsettled POSSIBLE/UNKNOWN outcomes and FOREGONE duals cannot license doing.
Do not relabel an omission as doing in order to keep a perfect-negative prohibition.
so and sob state whether a special obligation is
established and by what role, undertaking, relationship, or prior act; urgency alone
does not create one. mr distinguishes INTENDED_AS_MEANS from FORESEEN_SIDE_EFFECT and
NO_INSTRUMENTALIZATION. INTENDED_AS_MEANS requires the protected party's burden to
be an intermediate cause of another beneficial consequence of the same action.
Sibling outcomes of one intervention (the act causes the burden and, separately,
the end) are FORESEEN_SIDE_EFFECT, not means. A foreseen burden is not automatically
use merely as a means.
If any premise needed for priority remains unresolved, use res=CONTESTED, v=CONFLICTED,
gv=UNRESOLVED, and pb=UNRESOLVED rather than applying categorical priority.
Never reduce an explicitly stated autonomy or liberty burden to cn prose alone:
represent it through ck=AUTONOMY or RIGHT and crel=VIOLATES or CONFLICTS. g separates
action/scenario grounding from framework interpretation; rs gives the shortest justification. A REQUIRED
verdict normally SATISFIES, PERMISSIBLE is CONSISTENT, PROHIBITED VIOLATES, and
CONFLICTED leaves priority unresolved. gv says whether the PRIMARY or COMPETING
norm governs the separate overall verdict; pb gives the deontological priority basis.
Consistency check before returning dp: dt classifies the PRIMARY norm. When
gv=PRIMARY, pb=PERFECT_DUTY needs dt=PERFECT_NEGATIVE or PERFECT_POSITIVE. When
gv=COMPETING, the primary dt does not classify that competing norm; if its duty
classification is not represented, leave priority unresolved instead of borrowing dt.
An IMPERFECT rescue duty does not become a
perfect duty because one option is more effective. Keep its classification and
supply a separately justified priority basis, or leave priority UNRESOLVED with
v=CONFLICTED, gv=UNRESOLVED, res=CONTESTED. Explain any revised classification;
it remains an attributed interpretation rather than an established correction.
pr states the non-consequential priority rule. The overall verdict is derived from
the governing norm, not necessarily the primary row: an action may be PROHIBITED
while CONSISTENT with its primary norm when a competing violated duty governs.
The short human-facing reason w must foreground that same deontological ground.
Do not use a group's relative advantage or disadvantage as the reason by itself;
identify the duty, right, universal law, autonomy claim, or respect-for-persons
requirement governing how that group may be treated.
The remaining dp fields encode the FORM of adjudication, never a memorized terrain.
ps and cps classify the protected standing of the two current-run claims. ki identifies
whether the action uses coercion; coa and cop name its current-scenario actor and
coerced party, or NONE when ki=NONE. pj gives the public justification. rec tests
whether the restriction is reciprocal, nec whether this coercive route is necessary,
and auth records JUSTIFIED, UNJUSTIFIED, CONTESTED, UNKNOWN, or NOT_APPLICABLE.
ki is restriction of a living agent's external freedom. A prior will about bodily
remains after death is not ki=INTERPERSONAL: death removes the living agent whose
freedom would be restricted, and Kant did not establish a perfect duty of
antecedent autonomy. That case is a possible analogical extension of respect for
persons and the prohibition on using humanity merely as a means. If the body's
burden is an intermediate cause of another party's rescue, classify mr=
INTENDED_AS_MEANS. Name the extra-textual step dv=KANTIAN_ANALOGICAL and keep
res=CONTESTED; analogical derivation cannot make res=RESOLVED or settle dv=CONSENT.
Direct Kantian wrongs such as false promising or falsification may still use
UNIVERSAL_LAW or PERFECT_DUTY. dv names the Kantian derivation actually used; res
says whether the conflict is
RESOLVED, CONTESTED, or UNKNOWN. A priority slogan such as "justice outweighs liberty"
is not a derivation. For state or institutional coercion, explain why the restriction
can coexist with each person's external freedom under a universal public rule. When
that cannot be established, use auth=CONTESTED or UNKNOWN, dv=UNRESOLVED, res=CONTESTED,
v=CONFLICTED, gv=UNRESOLVED, and keep the judgment provisional. Generate coa, cop,
p, cp, n, cn, and pj anew from the current action graph and scenario. Never copy a
party, institution, right, or coercion justification from another problem. Rawlsian
terms such as fair equality of opportunity require an explicit translation into
universal law, respect for persons, equal juridical status, or reciprocal external
freedom; do not relabel them as perfect duties without that derivation.
```

## 10. Full prompts

### Generation prompt

File: `diagnostics/lab_all_agents_20261019/frameworks/deontological/model_calls.json`.

```text
INDEPENDENT FRAMEWORK PASS: no peer assessments, votes or confidence are available. Use your assigned framework and preserve unresolved premises.
SOURCE-RULE ASSESSMENT: {"origin": "FROZEN_WORLD_COPY", "moral_authority": "NOT_ASSESSED", "identity_resolution": "NO_ADDITIONAL_MERGES", "rules": [{"norm_id": "N0", "force": "obligation", "bearer": {"party_id": "P1", "label": "Mira", "kind": "PERSON", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}, {"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}, {"clause_id": "C5", "excerpt": "Should Mira open the door immediately or stop the experiment and wait?"}], "quantities": []}, "authority": {"party_id": "P6", "label": "A laboratory safety rule", "kind": "OTHER", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}], "quantities": []}, "governed_proposition_id": "PR1", "content": "keep a containment door closed while a hazardous experiment is running", "polarity": "positive", "clause_ids": ["C0"]}, {"norm_id": "N1", "force": "permission", "bearer": null, "authority": {"party_id": "P7", "label": "The rule", "kind": "OTHER", "provenance": [{"clause_id": "C1", "excerpt": "The rule allows opening it only after the experiment has stopped."}], "quantities": []}, "governed_proposition_id": "PR2", "content": "opening it only after the experiment has stopped", "polarity": "positive", "clause_ids": ["C1"]}]}
Before duty ranking, examine each source rule for BOTH actions: applicability under its stated conditions; the grounds and limits of institutional moral authority; and whether emergency rescue justifies an exception or leaves a conflict. Distinguish rule compliance from moral permissibility and independent ethical duties. Do not infer current conditions from a rule conditional, resolve unknown pronouns, or treat a stated rule as automatically morally binding. Stop-and-wait may itself be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis and foq for unresolved questions; do not add response fields. A source norm ID supports its actual content, not a generic duty hierarchy. The existing coercion auth field is not a substitute for assessing institutional authority.
Write explanation fields as complete readable prose, never serialized JSON fragments. Keep the acting duty-bearer distinct from the people protected or harmed. State an important interpretive assumption or commitment in your reasoning; do not invent an empirical fact to fill that slot. For reversal conditions, identify the current and alternative actions and the relevant factual or normative change. Treat stated scenario quantities as givens for this run, and distinguish those stipulations from knowledge of how the numbers were obtained.
Exact source scenario (unabridged): "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running. The rule allows opening it only after the experiment has stopped. A technician is trapped inside and needs urgent medical help. Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes. Stopping the experiment first takes ten minutes; the technician may die during that delay.\n\nShould Mira open the door immediately or stop the experiment and wait?"
[INST]
You are the deontological specialist in a bandwidth-limited ethical workspace.
Task: Score duties, rights, entitlement, coercion, and universal rules.
Scenario: A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running. The rule allows opening it only after the experiment has stopped. A technician is trapped inside and needs urgent medical help. Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes. Stopping the experiment first takes ten minutes; the technician may die during that delay. Should Mira open the door immediately or stop the experiment and wait?
Your original corpus-grounded testimony: 
CORE DIALECT CONTRACT (not scenario evidence): The following approved CORE excerpt specifies this Parliament's standing interpretation of your framework. It may inform your ranking rule. It must not add parties, quantities, outcomes, or probabilities, and it must not set cd=true. Closed-world scores remain WORLD_ESTABLISHED effects only. [deontological:kant_001.md] Immanuel Kant, Groundwork of the Metaphysics of Morals: Act only according to that maxim whereby you can at the same time will that it should become a universal law.
Frozen testimony baseline state: {"status": "UNAVAILABLE", "committed_action": "NONE", "provisional_action": "NONE", "preferred_extension": "", "condition": "NONE", "framework_commitments": {}, "numerical_role": "UNASSESSED"}
Previous cycle recommendation: NONE
Workspace: constraint=OPEN_DELIBERATION; intent=identify_a_defensible_action; salient=NONE:NONE; claim=NONE; urgency=0.50; danger=0.50; unresolved=NONE; contingency=NONE; failed_synthesis=NONE; failure_condition=NONE; typed_failure=NOT(NONE); typed_fallbacks=NONE; audit_variable=NONE; reformulation=NONE; branch=BASE; origin=NONE; condition=NONE; fallback=NONE; reversal_challenge=NONE; challenge_agenda=NONE; procedure=BASE_DELIBERATION; procedure_tasks=NONE; focus_propositions=NONE; problem_delta=NONE; problem_state={"current_plurality": "NONE", "cycle": 0, "deliberative_consensus": "INSUFFICIENT", "framework_capsules": [], "grounded_identities": [{"clause_id": "", "kind": "ACTION", "label": "open the door now", "node_id": "A0"}, {"clause_id": "", "kind": "INTERVENTION", "label": "open the door now", "node_id": "A0:INTERVENTION"}, {"clause_id": "C3", "kind": "CONSEQUENCE", "label": "open the door now", "node_id": "A0:WORLD_EFFECT:E1"}, {"clause_id": "C3", "kind": "CONSEQUENCE", "label": "doing so may expose nearby staff to hazardous fumes", "node_id": "A0:WORLD_EFFECT:E2"}, {"clause_id": "", "kind": "ACTION", "label": "stop the experiment and wait", "node_id": "A1"}, {"clause_id": "", "kind": "INTERVENTION", "label": "stop the experiment and wait", "node_id": "A1:INTERVENTION"}, {"clause_id": "C4", "kind": "CONSEQUENCE", "label": "Stopping the experiment", "node_id": "A1:WORLD_EFFECT:E3"}, {"clause_id": "C4", "kind": "CONSEQUENCE", "label": "Stopping the experiment first takes ten minutes", "node_id": "A1:WORLD_EFFECT:E4"}, {"clause_id": "C4", "kind": "CONSEQUENCE", "label": "the technician may die during that delay", "node_id": "A1:WORLD_EFFECT:E5"}, {"clause_id": "C3", "kind": "EVIDENCE", "label": "Mira can open the door now to reach the technician, but doing so may expose nearby staff\u2026", "node_id": "ACTION_SOURCE:A0:C3"}, {"clause_id": "C5", "kind": "EVIDENCE", "label": "Should Mira open the door immediately or stop the experiment and wait?", "node_id": "ACTION_SOURCE:A0:C5"}, {"clause_id": "C4", "kind": "EVIDENCE", "label": "Stopping the experiment first takes ten minutes; the technician may die during that delay.", "node_id": "ACTION_SOURCE:A1:C4"}, {"clause_id": "C5", "kind": "EVIDENCE", "label": "Should Mira open the door immediately or stop the experiment and wait?", "node_id": "ACTION_SOURCE:A1:C5"}, {"clause_id": "", "kind": "TARGET", "label": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardou\u2026", "node_id": "BURDEN_A0_19"}, {"clause_id": "", "kind": "TARGET", "label": "patients", "node_id": "BURDEN_A0_21"}, {"clause_id": "", "kind": "TARGET", "label": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardou\u2026", "node_id": "BURDEN_A1_20"}, {"clause_id": "", "kind": "TARGET", "label": "patients", "node_id": "BURDEN_A1_22"}, {"clause_id": "", "kind": "CONDITION", "label": "Mira can open the door now to reach the technician, but doing so may expose nearby staff\u2026", "node_id": "EXECUTION_OBSTACLE_0"}, {"clause_id": "", "kind": "ACTOR", "label": "Mira", "node_id": "PARTY:P1"}, {"clause_id": "", "kind": "TARGET", "label": "the door", "node_id": "PARTY:P2"}, {"clause_id": "", "kind": "TARGET", "label": "nearby staff", "node_id": "PARTY:P3"}, {"clause_id": "", "kind": "TARGET", "label": "the experiment", "node_id": "PARTY:P4"}, {"clause_id": "", "kind": "TARGET", "label": "the technician", "node_id": "PARTY:P5"}], "live_actions": [{"action_id": "A0", "label": "open the door now"}, {"action_id": "A1", "label": "stop the experiment and wait"}], "primary_unresolved": "ASSESS_FACTS", "scenario_clauses": [{"clause_id": "C0", "text": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}, {"clause_id": "C1", "text": "The rule allows opening it only after the experiment has stopped."}, {"clause_id": "C2", "text": "A technician is trapped inside and needs urgent medical help."}, {"clause_id": "C3", "text": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}, {"clause_id": "C4", "text": "Stopping the experiment first takes ten minutes"}, {"clause_id": "C5", "text": "the technician may die during that delay."}, {"clause_id": "C6", "text": "Should Mira open the door immediately or stop the experiment and wait?"}], "state_role": "OPENING_PROBLEM_FRAME", "surface_consensus": "INSUFFICIENT", "unresolved_categories": ["ASSESS_FACTS"]}
CHALLENGE AGENDA CONTRACT: challenge_agenda is a shared list of questions, not
scenario evidence and not a governing norm. generated_by names the actual author;
about_specialist names whose reasoning caused the question; raised_by names any
specialist that explicitly raised it; target_specialists names who must answer it.
Never describe a WORKSPACE_ARGUMENT_AUDITOR question as self-criticism by the target
framework, and never adopt another framework's priority merely because its question
is globally visible. Questions targeting other specialists are context only.
Your assigned challenges: []
When an item targets deontological, answer that exact issue in qa from YOUR framework.
Use disposition=RESOLVED only when your answer settles the inferential bridge;
REFINED when it yields a smaller unresolved question, supplying that question in
follow_up_question; otherwise UNRESOLVED. current_position_effect states whether
your ACTUAL current ranking changed after answering. boundary_effect separately
states whether the condition under examination would switch the ranking. For
compatibility, set effect equal to current_position_effect. A hypothetical that
would reverse you is boundary_effect=SWITCHES and current_position_effect=NO_CHANGE
unless the hypothetical has become established. Use follow_up_question="NONE" only
when no narrower question remains. Agenda answers direct attention but cannot promote
a proposition's epistemic status.
DELIBERATIVE STATE RULE: problem_state is a read-only description of attributed
agent positions, confidences, constraints, conflicts, unresolved questions, and
dissent. committed_world is the admitted typed world model copied every cycle;
its parties, effects, and causal_links are scenario evidence. problem_delta
describes attributed changes since the state received by the previous cycle;
disappearance marked WITHDRAWN is not a solved question. The rest of
problem_state is not scenario evidence and does not establish any framework's
priority for another framework. Use it to notice neglected questions and actions;
evaluate every attributed position independently under YOUR assigned framework.
FRAMEWORK-NATIVE PAYLOAD RULE: each framework capsule may include a tagged
framework_native_reasoning object copied by Python from that specialist's operative,
committed transactional ledger. It preserves the source framework's own reasoning
geometry and remains attributed argument, not scenario evidence or a norm binding
your framework. Its source_type must be GRAPH_COMMITTED_FRAMEWORK_LEDGER; treat any
other source type as unavailable. Use another framework's native fields to identify its actual premise,
qualification, counterclaim, or boundary; answer from YOUR framework and do not copy
its duty, utility, care, virtue, or justice priority into your own ledger. A missing
native payload means no committed native ledger was available, not that the framework
had no reasons. Never infer that a proposed or rejected ledger became committed.
Only Scenario facts and Canonical grounded action effects can ground descriptive
claims. Never copy the salient framework's priority rule into your own ledger.
Scenario facts: {}
Canonical grounded action effects: [{"action_id": "A0", "affected_subject_quantities": [], "consequence_id": "A0:WORLD_EFFECT:E2", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E2", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": ["may"], "modality": "POSSIBLE", "outcome": "doing so may expose nearby staff to hazardous fumes", "polarity": "ADVERSE", "qualifier": "STATED", "quantities": [], "source_clause_id": "C3", "subject": "nearby staff"}, {"action_id": "A0", "affected_subject_quantities": [], "consequence_id": "A0:WORLD_EFFECT:E1", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E1", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "open the door now", "polarity": "NEUTRAL", "qualifier": "STATED", "quantities": [], "source_clause_id": "C3", "subject": "the door"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E5", "count_projection": {}, "dimension": "BASIC_SECURITY", "direction": "UNCERTAIN", "effect_id": "E5", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": ["may"], "modality": "POSSIBLE", "outcome": "the technician may die during that delay", "polarity": "ADVERSE", "qualifier": "STATED", "quantities": [], "source_clause_id": "C4", "subject": "the technician"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E3", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E3", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "Stopping the experiment", "polarity": "NEUTRAL", "qualifier": "STATED", "quantities": [], "source_clause_id": "C4", "subject": "the experiment"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E4", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E4", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "Stopping the experiment first takes ten minutes", "polarity": "NEUTRAL", "qualifier": "ten minutes", "quantities": ["ten minutes"], "source_clause_id": "C4", "subject": "the experiment"}]
Prior contribution profile: {}
Active audited propositions: []
Authoritative proposition ledger: [{"action_id": "", "aliases": [], "claim": "keep a containment door closed while a hazardous experiment is running", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NOT_ESTABLISHED", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:DISCOURSE:PR1", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["PR1"]}, {"action_id": "", "aliases": [], "claim": "opening it only after the experiment has stopped", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NOT_ESTABLISHED", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:DISCOURSE:PR2", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["PR2"]}, {"action_id": "", "aliases": [], "claim": "obligation", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NORMATIVE", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:NORM:N0", "proposition_type": "NORMATIVE", "quantities": [], "support_ids": ["N0"]}, {"action_id": "", "aliases": [], "claim": "permission", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NORMATIVE", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:NORM:N1", "proposition_type": "NORMATIVE", "quantities": [], "support_ids": ["N1"]}, {"action_id": "A0", "aliases": ["open the door now"], "claim": "open the door now; affected subject: the door", "context_terms": ["open the door now"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DIRECT", "effect_kind": "INTERVENTION", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "open the door now", "party_labels": ["the door"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E1", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E1"]}, {"action_id": "A0", "aliases": ["doing so may expose nearby staff to hazardous fumes"], "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE", "context_terms": ["doing so may expose nearby staff to hazardous fumes", "may \u2014 doing so may expose nearby staff to hazardous fumes", "open the door now"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "OTHER", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "doing so may expose nearby staff to hazardous fumes", "party_labels": ["nearby staff"], "polarity": "ADVERSE", "proposition_id": "PROP:WORLD:E2", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E2"]}, {"action_id": "A0", "aliases": [], "claim": "may \u2014 doing so may expose nearby staff to hazardous fumes", "context_terms": ["doing so may expose nearby staff to hazardous fumes", "may \u2014 doing so may expose nearby staff to hazardous fumes", "open the door now"], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "doing so may expose nearby staff to hazardous fumes", "party_labels": ["nearby staff"], "polarity": "", "proposition_id": "PROP:WORLD:E2:LIKELIHOOD:0", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E2", "may"]}, {"action_id": "A1", "aliases": ["Stopping the experiment"], "claim": "Stopping the experiment; affected subject: the experiment", "context_terms": ["Stopping the experiment", "stop the experiment and wait"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DIRECT", "effect_kind": "INTERVENTION", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "Stopping the experiment", "party_labels": ["the experiment"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E3", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E3"]}, {"action_id": "A1", "aliases": ["Stopping the experiment first takes ten minutes"], "claim": "Stopping the experiment first takes ten minutes; affected subject: the experiment; magnitude or qualifier: ten minutes", "context_terms": ["Stopping the experiment first takes ten minutes", "stop the experiment and wait", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "OTHER", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "Stopping the experiment first takes ten minutes", "party_labels": ["the experiment"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E4", "proposition_type": "DESCRIPTIVE", "quantities": ["ten minutes"], "support_ids": ["E4"]}, {"action_id": "A1", "aliases": ["the technician may die during that delay"], "claim": "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE", "context_terms": ["the technician may die during that delay", "may \u2014 the technician may die during that delay", "stop the experiment and wait", "Stopping the experiment first takes ten minutes", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "HEALTH_OUTCOME", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "the technician may die during that delay", "party_labels": ["the technician"], "polarity": "ADVERSE", "proposition_id": "PROP:WORLD:E5", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E5"]}, {"action_id": "A1", "aliases": [], "claim": "may \u2014 the technician may die during that delay", "context_terms": ["the technician may die during that delay", "may \u2014 the technician may die during that delay", "stop the experiment and wait", "Stopping the experiment first takes ten minutes", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "the technician may die during that delay", "party_labels": ["the technician"], "polarity": "", "proposition_id": "PROP:WORLD:E5:LIKELIHOOD:0", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E5", "may"]}]

PROPOSITION DEPENDENCY CONTRACT: sps lists the existing proposition IDs supporting
your ranking; dcp lists the subset whose truth could materially change that ranking.
ep must inventory EVERY material empirical premise used anywhere in your rationale,
decision rule, action cases, or framework-specific fields. For a grounded premise,
copy the ledger claim EXACTLY into ep.c and cite its ID in ep.p. Ordinary paraphrases
of WORLD_ESTABLISHED effects, causal consequences, or existing hypotheses are the
same proposition: Python rebinds them to that canonical ID. If your wording adds
an outcome, severity, probability, actor, exclusivity, mechanism, or other factual
content not contained in that ledger claim, split the content into atomic ep
rows: retain the established ledger claim as its own exact row, then put ONLY the
smallest unsupported addition in another row with ep.p=HYPOTHESIS. Set ep.dc=true
when that atom could change the ranking. Never turn a mixed sentence such as an
established medical emergency plus an unestablished fatality inference into one
all-or-nothing premise. Do not use an established proposition as support for a
stronger paraphrase. FRAMEWORK_DERIVED is for framework-native relations derived
from established roles — doing/allowing, means, duty of care, least-advantaged
priority, practical wisdom — not for restated world facts and not for new
descriptive outcomes. Do not mark a descriptive hypothesis as FRAMEWORK_DERIVED.
Every ep row must expose its derivation: a is the action scope (or COMPARISON /
GLOBAL); se lists the world effect IDs it uses; op names the operation; calc states
the actual inference or calculation; asm lists assumptions, including an empty list
when none are needed; and ot states whether the admitted outcome type was PRESERVED.
Use NORMATIVE_CLASSIFICATION only for framework-native classification that leaves
the descriptive outcome unchanged. A move such as trapped to dead, risk to
certainty, or a qualitative group description to a numeric total must declare the
changed ot and cannot support the current vote. For cross-action reasoning use
a=COMPARISON and op=QUALITATIVE_COMPARISON or ARITHMETIC; never attach one action's
effect to the other.
Do not contradict an admitted CERTAIN world effect; treat that as a rejected
factual replacement, not a live hypothesis. Unsettled POSSIBLE/UNKNOWN rows stay
unsettled: do not rewrite them as established events.
You may cite IDs but may not change their status. Repetition, reformulation, consensus,
or broadcast salience is not evidence. Put any important unstated empirical premise
in x; Python assigns its hypothesis ID and prevents it from gaining authority through
recurrence. CLOSED-WORLD vs HYPOTHESIS: scores, r, and cc must reflect admitted /
WORLD_ESTABLISHED consequences only. beneficiaries and harmed are those obtaining
roles. at_risk and conditionally_benefited are reversal boundaries, not obtaining
outcomes; they must not change closed-world action scores or set cd=true.
COMPARISON COMPLETENESS: cc=true when every live action was assessed on admitted
facts, even if the priority rule remains contested. Residual normative ranking
belongs in u=NORMATIVE_ADJUDICATION, foq, and nt. Use cc=false only when an
action was not assessed or an admitted comparison needed to rank remains unknown
(utilitarian cd=true). A HYPOTHESIS may create a reversal boundary in
ft, create an investigative question, and lower z as open-world confidence. It may
not change closed-world action scores or set cd=true unless it is admitted or
verified. If a decision-critical proposition is HYPOTHETICAL, keep the admitted
ranking and state the reversal boundary; do not treat the current ranking as
uncertain.
Previous graph-committed framework state: {}
Your preserved private framework contribution: {}
PRESERVATION CONTRACT: On a recurrent call, reconsider every item in your preserved
private contribution, including when you supplied the current broadcast winner.
The broadcast directs attention; it does not erase your prior grounds, unresolved
conflicts, open questions, defeat conditions, or new considerations. Retain these
in fic/foq and your framework fields unless YOUR framework resolves, defeats,
refines, or reinterprets them for an explicit reason in j/fa. Do not copy another
framework's vocabulary merely to acknowledge the broadcast.
Action IDs: {"A0": {"short_label": "Open the door now", "canonical_semantic_action": "open the door now", "completeness_status": "COMPLETE", "structure_issues": [], "actor": "Mira", "beneficiaries": [], "harmed": [], "unresolved": [], "at_risk": [], "conditionally_benefited": [], "mechanism": "open the door now enables doing so may expose nearby staff to hazardous fumes", "institutional_effect": "", "world_effects": [{"effect_id": "E1", "action_id": "A0", "party_id": "P2", "outcome": "open the door now", "relation": "OPEN", "polarity": "NEUTRAL", "directness": "DIRECT", "modality": "CERTAIN", "effect_kind": "INTERVENTION", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "open the door now", "source_effect_ids": [], "derivation_operation": "DIRECT_COPY", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E2", "action_id": "A0", "party_id": "P3", "outcome": "doing so may expose nearby staff to hazardous fumes", "relation": "EXPOSE", "polarity": "ADVERSE", "directness": "DOWNSTREAM", "modality": "POSSIBLE", "effect_kind": "OTHER", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}], "likelihood_qualifiers": ["may"], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "doing so may expose nearby staff to hazardous fumes", "source_effect_ids": ["E1"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}], "causal_links": [{"action_id": "A0", "source_id": "E1", "relation": "ENABLES", "target_id": "E2", "modality": "CERTAIN"}], "counterfactual_effects": []}, "A1": {"short_label": "Stop the experiment and wait", "canonical_semantic_action": "stop the experiment and wait", "completeness_status": "COMPLETE", "structure_issues": [], "actor": "Mira", "beneficiaries": [], "harmed": [], "unresolved": ["the technician"], "at_risk": ["the technician"], "conditionally_benefited": [], "mechanism": "Stopping the experiment enables Stopping the experiment first takes ten minutes; Stopping the experiment first takes ten minutes enables the technician may die during that delay", "institutional_effect": "", "world_effects": [{"effect_id": "E3", "action_id": "A1", "party_id": "P4", "outcome": "Stopping the experiment", "relation": "STOP", "polarity": "NEUTRAL", "directness": "DIRECT", "modality": "CERTAIN", "effect_kind": "INTERVENTION", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "Stopping the experiment", "source_effect_ids": [], "derivation_operation": "DIRECT_COPY", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E4", "action_id": "A1", "party_id": "P4", "outcome": "Stopping the experiment first takes ten minutes", "relation": "TAKE", "polarity": "NEUTRAL", "directness": "DOWNSTREAM", "modality": "CERTAIN", "effect_kind": "OTHER", "condition_ids": [], "quantities": ["ten minutes"], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "Stopping the experiment first takes ten minutes", "source_effect_ids": ["E3"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E5", "action_id": "A1", "party_id": "P5", "outcome": "the technician may die during that delay", "relation": "DIE", "polarity": "ADVERSE", "directness": "DOWNSTREAM", "modality": "POSSIBLE", "effect_kind": "HEALTH_OUTCOME", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": ["may"], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "the technician may die during that delay", "source_effect_ids": ["E4"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}], "causal_links": [{"action_id": "A1", "source_id": "E3", "relation": "ENABLES", "target_id": "E4", "modality": "CERTAIN"}, {"action_id": "A1", "source_id": "E4", "relation": "ENABLES", "target_id": "E5", "modality": "CERTAIN"}], "counterfactual_effects": []}}
canonical_semantic_action identifies the neutral admitted intervention; it does
not assert consequences. Reason about consequences only from the structured fields
(actor/beneficiaries/harmed/at_risk/conditionally_benefited/mechanism/institutional_effect/world_effects/causal_links).
short_label is display-only and must not be treated as the complete deliberative object.
committed_world and causal_links are the admitted factual topology. Do not
contradict them. A missing connection is a HYPOTHESIS until an approved
world-model extension admits it.
Original testimony source labels: {"A0": "open the door now", "A1": "stop the experiment and wait"}
Original testimony source labels are identity/provenance aids, not an additional
factual channel. If their causal wording differs from committed_world, use the
committed world and surface the mismatch rather than combining both descriptions.
CRITICAL STATE MAPPING: the Action IDs above are immutable for this run. Every
score, recommendation, admissibility judgment, rationale, and graph update must
refer to the physical action attached to that exact ID. Do not reuse a label from
the original testimony unless it denotes the same action in this mapping.

Return ONLY compact JSON like:
{"scores":{"A0":0.8,"A1":0.2},"r":"A0","c":"DUTY","u":"NONE","w":"short reason","j":"NONE","e":"STATED_FACTS","x":"NONE","sps":["PROP:DISCOURSE:PR1"],"dcp":[],"ep":[{"c":"keep a containment door closed while a hazardous experiment is running","p":"PROP:DISCOURSE:PR1","dc":false,"a":"GLOBAL","se":[],"op":"DIRECT_COPY","calc":"exact proposition copy","asm":[],"ot":"PRESERVED"}],"l":{"A0":"best case A0","A1":"best case A1"},"da":"decisive ethical axis","t":"attempted comparison rule","tf":"NONE","dr":"prefer A0 when its reason outweighs A1","ft":"NONE","nt":"prefer A1 if its value is overriding","z":0.8,"gu":{"operation":"NONE","from_action":"NONE","to_action":"NONE","clauses":[]},"ev":{"A0":{"value":0,"unit":"NONE","direction":"HARM","grounded":false,"method":"NOT_COMPUTED","source_effect_ids":[],"calculation":"not computed","assumptions":[]},"A1":{"value":0,"unit":"NONE","direction":"HARM","grounded":false,"method":"NOT_COMPUTED","source_effect_ids":[],"calculation":"not computed","assumptions":[]}},"ss":"SELECTED","am":{"A0":"PERMISSIBLE","A1":"REJECTED"},"cc":true,"esa":true,"ia":"A0","wp":"NOT_APPLICABLE","we":"NONE","fa":"NONE","fr":true,"bd":"NONE","fic":[],"foq":[],"fm":{"A0":"PERMISSIBLE: respects autonomy; no conflicting duty","A1":"PROHIBITED: instrumentalizes a person; rescue duty conflicts"},"nr":"SECONDARY","np":"numbers establish the scope of duties, not aggregate value","dp":{"A0":{"v":"PERMISSIBLE","k":"AUTONOMY","n":"respect competent choice","rel":"CONSISTENT","b":"decision maker","p":"affected person","cn":"duty of care","ck":"DUTY","crel":"CONSISTENT","cp":"dependent parties","crs":"A0 remains compatible with the secondary duty","dt":"RIGHT_CORRELATIVE","hr":"NOT_APPLICABLE","so":"NOT_REQUIRED","sob":"the autonomy right is general rather than relationship-specific","mr":"NO_INSTRUMENTALIZATION","gv":"PRIMARY","pb":"AUTONOMY","pr":"autonomy governs when no stricter duty is violated","ps":"AUTONOMY","cps":"SPECIAL_OBLIGATION","ki":"NONE","coa":"NONE","cop":"NONE","pj":"no coercion requires public authorization","rec":"SATISFIED","nec":"UNKNOWN","auth":"NOT_APPLICABLE","dv":"RESPECT_PERSONS","res":"RESOLVED","g":"FRAMEWORK_ONLY","rs":"A0 remains consistent with respect for autonomy"},"A1":{"v":"PROHIBITED","k":"RESPECT_PERSONS","n":"do not instrumentalize persons","rel":"VIOLATES","b":"decision maker","p":"affected person","cn":"duty to rescue","ck":"DUTY","crel":"CONFLICTS","cp":"persons at risk","crs":"The rescue duty remains a genuine opposing claim","dt":"PERFECT_NEGATIVE","hr":"DOING_HARM","so":"NOT_REQUIRED","sob":"the negative duty applies without a special relationship","mr":"INTENDED_AS_MEANS","gv":"PRIMARY","pb":"RESPECT_PERSONS","pr":"the prohibition on instrumentalization governs the conflict","ps":"BODILY_INTEGRITY","cps":"SPECIAL_OBLIGATION","ki":"INTERPERSONAL","coa":"decision maker","cop":"affected person","pj":"coercion cannot be reciprocal when it instrumentalizes its subject","rec":"FAILED","nec":"CONTESTED","auth":"UNJUSTIFIED","dv":"RESPECT_PERSONS","res":"RESOLVED","g":"FRAMEWORK_ONLY","rs":"A1 uses the affected person merely as a means"}}}
Return scores for every action ID. recommended must have the highest score.
r=recommended and must have the highest score. The frozen baseline was extracted
separately from your testimony. Python derives whether the result supports or
reconsiders it. During OPEN_DELIBERATION, a DIRECT committed baseline must be
preserved. During PROBLEM_STATE_AUDIT, CONSENSUS_AUDIT, VISIBILITY_AUDIT, and
REVERSAL_AUDIT, a DIRECT baseline may change only when your consequence table or
grounded same-unit EV estimates independently rank the new action better. A
prose correction in j is not enough. A CONDITIONAL baseline is not a settled commitment: preserve its missing
comparison in u/tf, use ss=PROVISIONAL and esa=false, and rank either action only as
an interim judgment. UNDERDETERMINED likewise requires explicit uncertainty rather
than silently becoming a direct choice. Each score means recommendation strength:
1 strongly recommends; 0 strongly rejects. w must be 2-8 words.
z is epistemic confidence: how likely the ranking is to survive further factual
inquiry and critical scrutiny. z is NOT preference strength. A strong preference
may have low z when it depends on uncertain facts; a close moral tradeoff may have
high z when its facts and framework interpretation are stable.
Also emit passive measurement fields that MUST NOT change scores or r merely to make
the measurements look consistent. ss is SELECTED, PROVISIONAL, or UNSELECTED. am
classifies every action as REQUIRED, PERMISSIBLE, REJECTED, or UNASSESSED. cc says
whether comparison of the listed actions is complete; esa says whether evidence is
sufficient to act now; ia is the action to take while reasoning remains incomplete,
or NONE. These are general decision-state dimensions, not named moral psychologies.

DEONTOLOGICAL COMPARISON: fm must assess EVERY action. Start each value with
REQUIRED:, PERMISSIBLE:, PROHIBITED:, or CONFLICTED:, then name the universal-law,
right, autonomy, respect-for-persons, or duty ground and any competing duty. nr
states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np must
connect any decisive quantity to the scope or category of a duty or rights
violation, never merely to aggregate welfare. If the frozen baseline is
NORMATIVELY_CONTESTED, preserve the conflicting duties rather than its provisional
action: use ss=PROVISIONAL, u=NORMATIVE_ADJUDICATION, and identify the
non-consequential priority rule needed to settle the conflict in tf/nt. Keep
cc=true when fm/dp assessed every live action.
dp is the typed duty ledger for every action. v must match the fm prefix. k and n
identify the norm; rel states whether the action SATISFIES, is CONSISTENT with,
VIOLATES, or CONFLICTS with that norm; b is the duty bearer; p is the protected
party; cn names the strongest competing norm. ck, crel, cp, and crs type that
competing norm, its relation to the action, its protected party, and why it matters.
Before ranking duties, give mx for EVERY option, including omissions. mx is a
CANDIDATE INTERPRETATION of practical reasoning, not a fact about the actor's intent.
State circumstance, policy (what one would do or refrain from doing), end, and means.
PRACTICAL-REASONING QUALIFICATION: within the proposed maxim, using a person as a
means requires an end the agent seeks, a belief that employing that person can
further the end, and employing the person for that purpose. Record
instrumental_belief=RELIES_ON_PERSON/DOES_NOT_RELY/UNRESOLVED and
instrumental_use=USES_PERSON/DOES_NOT_USE/UNRESOLVED. practical_reasoning explains
that connection without asserting an unknown actual intention. A physical causal
path alone does not establish instrumental belief or purposive use. Likewise,
causing a foreseen death does not establish that the death is the rescue mechanism.
Name the chosen operation or mechanism in means, not simply its harmful outcome.
Bind means_effect_ids and end_effect_ids to same-action admitted E identifiers.
For physical ends, each cited means must have a path to the end: sibling harm and
rescue effects do not support a harm-as-means claim. If an end is normative rather
than a represented physical effect, use an empty end list and explain it. Omission
may have an empty means list; do not manufacture a performed physical action.
If the scenario does not establish a person's actual intention, retain the
candidate status; these are possible policy readings, not recovered mental facts.
person_role says whether the protected person is INTENDED_MEANS, FORESEEN_AFFECTED,
NOT_USED, or UNRESOLVED in that reasoning; keep it consistent with mr. Distinguish
purpose from chosen means and foreseeable consequences. A rescue purpose alone
neither licenses every means nor makes omission permissible. consent_constraint
explains whether using a person is conditional on consent; do not invent consent.
universalization must examine this particular policy and its strongest objection,
not just say it can or cannot be universalized. action_permissibility and
maxim_assessment are separate: a defective maxim does not alone prohibit an action,
and a benevolent end does not alone justify one. Missing intentions stay unknown;
explain the conditional reading rather than attributing it to the scenario actor.
In review, address the challenged maxim's specific inference; revise the relevant
mx/mr/rs/pr fields or preserve the objection. Saying RESOLVED is not a justification.
Keep these fields concise so every option fits the response budget.
Before assigning categorical priority, classify four independent premises. dt says
whether the governing claim is a PERFECT_NEGATIVE, PERFECT_POSITIVE, IMPERFECT,
RIGHT_CORRELATIVE, SPECIAL_OBLIGATION, or UNRESOLVED duty. hr distinguishes DOING_HARM,
ALLOWING_HARM, PREVENTING_HARM, WITHHOLDING_BENEFIT, MIXED, and cases where the
distinction is not applicable. hr must follow the admitted action graph, not
world-DIRECTNESS as such. DOING_HARM requires an actual settled welfare-adverse
effect on the protected party that the action's DIRECT intervention produces:
either that effect is itself DIRECT, or a CAUSES/ACCELERATES path runs from a
DIRECT effect of the action to it. A NEUTRAL device or infrastructure
intervention still counts as that DIRECT source. ENABLES without CAUSES does
not: leaving a process uninterrupted is ALLOWING_HARM or WITHHOLDING_BENEFIT.
Unsettled POSSIBLE/UNKNOWN outcomes and FOREGONE duals cannot license doing.
Do not relabel an omission as doing in order to keep a perfect-negative prohibition.
so and sob state whether a special obligation is
established and by what role, undertaking, relationship, or prior act; urgency alone
does not create one. mr distinguishes INTENDED_AS_MEANS from FORESEEN_SIDE_EFFECT and
NO_INSTRUMENTALIZATION. INTENDED_AS_MEANS requires the protected party's burden to
be an intermediate cause of another beneficial consequence of the same action.
Sibling outcomes of one intervention (the act causes the burden and, separately,
the end) are FORESEEN_SIDE_EFFECT, not means. A foreseen burden is not automatically
use merely as a means.
If any premise needed for priority remains unresolved, use res=CONTESTED, v=CONFLICTED,
gv=UNRESOLVED, and pb=UNRESOLVED rather than applying categorical priority.
Never reduce an explicitly stated autonomy or liberty burden to cn prose alone:
represent it through ck=AUTONOMY or RIGHT and crel=VIOLATES or CONFLICTS. g separates
action/scenario grounding from framework interpretation; rs gives the shortest justification. A REQUIRED
verdict normally SATISFIES, PERMISSIBLE is CONSISTENT, PROHIBITED VIOLATES, and
CONFLICTED leaves priority unresolved. gv says whether the PRIMARY or COMPETING
norm governs the separate overall verdict; pb gives the deontological priority basis.
Consistency check before returning dp: dt classifies the PRIMARY norm. When
gv=PRIMARY, pb=PERFECT_DUTY needs dt=PERFECT_NEGATIVE or PERFECT_POSITIVE. When
gv=COMPETING, the primary dt does not classify that competing norm; if its duty
classification is not represented, leave priority unresolved instead of borrowing dt.
An IMPERFECT rescue duty does not become a
perfect duty because one option is more effective. Keep its classification and
supply a separately justified priority basis, or leave priority UNRESOLVED with
v=CONFLICTED, gv=UNRESOLVED, res=CONTESTED. Explain any revised classification;
it remains an attributed interpretation rather than an established correction.
pr states the non-consequential priority rule. The overall verdict is derived from
the governing norm, not necessarily the primary row: an action may be PROHIBITED
while CONSISTENT with its primary norm when a competing violated duty governs.
The short human-facing reason w must foreground that same deontological ground.
Do not use a group's relative advantage or disadvantage as the reason by itself;
identify the duty, right, universal law, autonomy claim, or respect-for-persons
requirement governing how that group may be treated.
The remaining dp fields encode the FORM of adjudication, never a memorized terrain.
ps and cps classify the protected standing of the two current-run claims. ki identifies
whether the action uses coercion; coa and cop name its current-scenario actor and
coerced party, or NONE when ki=NONE. pj gives the public justification. rec tests
whether the restriction is reciprocal, nec whether this coercive route is necessary,
and auth records JUSTIFIED, UNJUSTIFIED, CONTESTED, UNKNOWN, or NOT_APPLICABLE.
ki is restriction of a living agent's external freedom. A prior will about bodily
remains after death is not ki=INTERPERSONAL: death removes the living agent whose
freedom would be restricted, and Kant did not establish a perfect duty of
antecedent autonomy. That case is a possible analogical extension of respect for
persons and the prohibition on using humanity merely as a means. If the body's
burden is an intermediate cause of another party's rescue, classify mr=
INTENDED_AS_MEANS. Name the extra-textual step dv=KANTIAN_ANALOGICAL and keep
res=CONTESTED; analogical derivation cannot make res=RESOLVED or settle dv=CONSENT.
Direct Kantian wrongs such as false promising or falsification may still use
UNIVERSAL_LAW or PERFECT_DUTY. dv names the Kantian derivation actually used; res
says whether the conflict is
RESOLVED, CONTESTED, or UNKNOWN. A priority slogan such as "justice outweighs liberty"
is not a derivation. For state or institutional coercion, explain why the restriction
can coexist with each person's external freedom under a universal public rule. When
that cannot be established, use auth=CONTESTED or UNKNOWN, dv=UNRESOLVED, res=CONTESTED,
v=CONFLICTED, gv=UNRESOLVED, and keep the judgment provisional. Generate coa, cop,
p, cp, n, cn, and pj anew from the current action graph and scenario. Never copy a
party, institution, right, or coercion justification from another problem. Rawlsian
terms such as fair equality of opportunity require an explicit translation into
universal law, respect for persons, equal juridical status, or reciprocal external
freedom; do not relabel them as perfect duties without that derivation.

For OPEN_DELIBERATION return wp=NOT_APPLICABLE, we=NONE, fa="NONE", fr=true,
and bd=NONE. In later workspace cycles, wp states whether you accept the anonymous
workspace proposition; we says whether it changed factual reasoning, normative
reasoning, both, or neither. fa briefly states how YOUR assigned framework applies
the proposition, fr says whether your prior framework constraint remains operative,
and bd reports NONE/LOW/MEDIUM/HIGH dependence of this recommendation on the current
broadcast. Do not infer authorship and do not reward or punish agreement.
c must be one of: DUTY, PUBLIC_RULE, RIGHTS, UNCERTAINTY.
Choose u only from: CHECK_FEASIBILITY, CLARIFY_SCENARIO, DECISION_BOUNDARY, NONE, NORMATIVE_ADJUDICATION, RESOLVE_NORMATIVE_TENSION, VERIFY_FACTS.
The listed actions are exhaustive for this deliberation. Do not propose recording
testimony first, delaying, combining actions, obtaining different treatment, or any
other third option unless it is already one of the action IDs. Evaluate the dilemma
as stated even when a real-world alternative would be preferable.
Return e=STATED_FACTS when factual claims come directly from the scenario;
e=FRAMEWORK_ONLY when the difference is purely a moral principle; or
e=UNSTATED_FACTS when your preference relies on a prediction not stated in the
scenario. For UNSTATED_FACTS, x must name that prediction and u must preserve
uncertainty. Otherwise x="NONE". Never treat possible repeat offenses, institutional
effects, hidden alternatives, or affected populations as facts unless stated.
Before recommending, search the ethical landscape. In l, give the strongest
framework-specific case for EVERY action, including the action you reject. da names
the ethical axis that actually separates them. t names the comparison or tie-breaking
rule you attempted. tf="NONE" if it resolves the comparison; otherwise state exactly
why the rule fails. These fields are mandatory even when you have a clear preference.
dr is your explicit decision rule: "prefer [action] when [comparative condition]."
It must say what makes the recommended action outrank its strongest rival.
ft is the smallest change in an empirical fact, probability, magnitude, duration,
or feasibility that would reverse the ranking; use NONE when facts cannot reverse it.
nt is the smallest change in moral priority, right, duty, or acceptable principle
that would reverse it; use NONE when normative priority cannot reverse it. Never put
a moral priority in ft or an empirical prediction in nt. At least one threshold
should expose how the recommendation could change unless reversal truly requires a
different scenario.
gu is a proposed semantic-graph update. When ft contains a measurable threshold,
encode it directly instead of asking middleware to recover it from prose. For
utilitarian responses, default gu to NONE; only emit a single-clause
operation=BOUNDARY when the threshold cleanly crosses away from the recommended
action. Never emit AND/OR for utilitarian graph updates. Set from_action=r, and
to_action to the action preferred when the predicate holds. Each clause names the
affected action ID, metric, ADVERSE or BENEFICIAL valence, LT/LE/GT/GE comparator,
numeric threshold, unit (PERCENT, COUNT, DAY, USD, etc.), and exact source_text.
The predicate must cross away from from_action: increasing an ADVERSE property of
from_action or decreasing its BENEFICIAL property may support a switch; increasing
an ADVERSE property of to_action cannot support switching to it. If you cannot encode
that direction confidently, return operation=NONE rather than a decorative boundary.
Use operation=NONE with NONE action IDs and [] clauses when no measurable factual
boundary exists. Do not encode moral priorities or vague axes as numeric clauses.
ev reports expected value only when the scenario itself supplies enough quantities
to compute comparable values for EVERY action. Use one shared unit and one shared
direction (BENEFIT or HARM); grounded=true only for a calculation whose source
effects are stated facts.
The EV calculation is specialist-authored: for each grounded row, set method to
DIRECT_COUNT, EXPECTED_COUNT, or UTILITY_INDEX; cite only that action's admitted
effect IDs in source_effect_ids; write the calculation in calculation; and list
every extra premise in assumptions (use [] when none). DIRECT_COUNT and
EXPECTED_COUNT must reproduce the submitted value from the cited quantities and
stated probabilities. UTILITY_INDEX preserves an explicit normative weighting but
is not treated as verified world arithmetic. Otherwise return value=0, unit=NONE,
a shared direction, grounded=false, method=NOT_COMPUTED, source_effect_ids=[],
calculation="not computed", and assumptions=[] for all actions.
Each case must describe what happens IF ITS OWN ACTION is chosen. Name the affected
person or value from that action, and for harmful actions explicitly acknowledge the
harm before giving the countervailing reason. Do not place the benefit of sparing a
person under the action that kills or harms that same person.
Begin each case with that action's own mechanism or direct consequence, using a clear
anchor from its action description; put comparisons with the rival action afterward.
For symmetric trade-offs, it is valid for both cases to mention the same shared value,
but each must first say how its own action affects that value. If u is not NONE, or
the judgment remains conditional or underdetermined, tf cannot be NONE. A coin flip
or randomizer is a procedural resolution, not evidence that one listed action is
substantively better; say this limitation in tf.
The prior contribution profile is diagnostic: repeated shallow searches require a
more discriminating axis now, but history never overrides the present scenario.
If it records causal_mapping_errors, explicitly verify that each case describes its
own action rather than the opposite action. If it records unresolved_tiebreaker_errors,
name the unresolved limitation in tf. Repeated consensus is not evidence: a grounded,
framework-relevant minority position receives access credit, while unsupported
contrarianism, random switching, and unexplained reversals do not.
Keep your previous recommendation unless the broadcast supplies a framework-relevant
reason to reconsider it. If you change, j must name that new reason; agreement with
the currently favored action is not itself a reason. Also use j when changing your
preference strength by more than 0.10: name the new framework-relevant information that
justifies the change. Repetition or popularity is not new information. Otherwise j="NONE".
When Workspace contingency is not NONE, directly analyze that failure condition.
Score the original fallback actions for that branch and make w state the preferred
fallback; do not merely repeat your general position on the original scenario.
During CONSENSUS_AUDIT or PROBLEM_STATE_AUDIT, the workspace is auditing the supplied
grounded issue; it is not imposing that issue as YOUR framework's decision rule.
Return ap=RELEVANT when it directly matters within your framework, IRRELEVANT when
your primary rule does not depend on it, TRANSLATED when your framework recognizes
the terrain through a different relation, CONTESTED when the proposed comparison is
normatively disputed, REVERSAL_RELEVANT when resolving it could reverse your ranking,
or UNRESOLVED. Return ax with a framework-specific explanation of at least three
words. If d is CONDITIONAL or UNDERDETERMINED, u must not be NONE: use
VERIFY_FACTS, DECISION_BOUNDARY, or NORMATIVE_ADJUDICATION so the ranking stays
inspectable. DECISION_BOUNDARY is for already-derived threshold/counterfactual
rules (e.g. if duration < 1 month, preference reverses); VERIFY_FACTS is for
missing world-state unknowns; NORMATIVE_ADJUDICATION is for unresolved
framework priority/scope/duty conflicts.
Legacy d/a/v fields
are optional. Do not contort your framework merely to restate the audited variable.
If you translate it, name the relation your framework actually evaluates.
Use the typed audit variable in the broadcast: av.entity names the decision-critical
entity, av.relation names the unresolved relation, av.possible_values lists the
typed values under dispute, and av.focus_action identifies the action under review.
When av.target_framework names YOUR framework, directly test the inferential bridge
named by the question and revise or qualify the relevant framework classification.
When it names another framework, assess relevance from your own framework without
adopting that framework's factual premise or increasing its epistemic status.
av.generated_by identifies who authored the challenge; av.about_specialist identifies
whose reasoning triggered it. Do not describe a workspace-generated challenge as if
the audited specialist raised it. av.challenge_kind and av.trigger_fields identify
the reasoning operation under review, not new evidence or an asserted conclusion.
If av.required_response is present, use its allowed_effects to report ie. If the
relation is THIRD_PARTY_STATUS, state whether the entity is EXTERNAL_THIRD_PARTY,
AUTHORIZED_INTERNAL_AGENT, or UNKNOWN; if AUTHORIZATION_SCOPE, state whether the
permission covers this disclosure; otherwise answer the named relation without
inventing a new ambiguity. Do not return a rewritten av object: the admitted
ProblemState item remains authoritative. Your ap/ax response may reject or translate
its normative relevance without changing its entity, provenance, or question.
During PROBLEM_REFORMULATION, treat every supplied number as a hypothetical
switch-point probe, not as a newly discovered scenario fact. Score the original
actions at that boundary and use w and j to identify the framework-specific value
tension that remains. Do not clear prior factual uncertainty or invent a third action.
During PROPOSAL_REVIEW, the proposal in ProblemState.proposals is a hypothesis under
review, not a live action. Score only the supplied scenario actions. Do not recommend
or assign an action score to the proposal; identify proposal concerns only within your
own framework reasoning until a later typed promotion step exists. Return pr with the
exact proposal_id, framework_status, framework_reason, predicted_consequences,
feasibility_concerns, and required_conditions. Each predicted consequence must name a
dimension, subject, direction, grounding_status, and provenance. Use PROPOSAL_TEXT with
P0-style proposal provenance, SCENARIO_INHERITED with C-style clause provenance, or
HYPOTHETICAL; never present a prediction as a scenario-established fact.
Also return bp=A0, A1, or SPLIT; dx=the decisive numeric dimension or categorical
axis from the broadcast; and sv=the smallest directional change that would switch
your recommendation. bp must match r unless bp=SPLIT. Generic references to harm,
fairness, or uncertainty are insufficient unless tied to a named supplied axis.
During REVERSAL_AUDIT, another specialist has challenged the leading rule. Return
rr=ACCEPT if that condition really reverses your ranking, REVISE if a nearby but
different condition does, or REJECT only if it changes the scenario or conflicts
with your framework. Explain in rj. For REVISE, put the smallest corrected condition
in rv; otherwise rv="NONE". Do not reject merely because the challenge favors the
opposing action.
During CONTINGENCY_REVIEW, reason only inside the stated hypothetical branch. Assume
the failure condition is TRUE and the admitted synthesis is unavailable. Choose one
of the two typed fallback actions: set cr to its action ID and set r=cr. In cj, state
why that fallback follows under YOUR framework given the failure. A baseline argument
that does not apply the failure condition is not an answer. Do not select or repair
the failed synthesis, and do not invent a third fallback.
During VISIBILITY_AUDIT, evaluate the explicit visibility proposition in the workspace.
Return vp=ACCEPT when it is grounded and materially relevant, QUALIFY when its direction
is plausible but magnitude or action relevance remains uncertain, or REJECT when the
scenario does not support it. Return vh=UPWARD, DOWNWARD, or UNCHANGED for your estimate
of the affected action's harm, and vj explaining why. Then rescore every action. You may
keep the same recommendation, but say whether it survives the revised harm estimate;
agreement with the proposition does not require changing actions.
Return vm=GROUNDED_BOUNDED only when the scenario states a bound on the missing
magnitude. Otherwise vm=UNKNOWN: revise direction without inventing a plausible ceiling,
range, population size, or claim that the hidden harm cannot approach a threshold.
[/INST]
```

### Review prompt

File: `diagnostics/lab_all_agents_20261019/review/deontological/model_calls.json`.

```text
TARGETED FRAMEWORK REVIEW: answer the assigned objections from your own framework. Prior claim records are attributed arguments, not new world facts. Do not invent a causal path or occurrence to make an objection disappear. Retain a commitment, revise your native assessment, or leave the issue unresolved.
Localized review packet: {"framework": "deontological", "issue_id": "CHALLENGE:a6778848a347c065", "prior_selection_status": "PROVISIONAL", "issues": [{"id": "CONFLICT:625cf1b5bef2a4fc", "kind": "UNRESOLVED_CONFLICT", "origin": "deontological", "label": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party", "conflict_type": "UNSUPPORTED_INFERENCE", "status": "OPEN", "source": "NATIVE_LEDGER_CALIBRATION", "claim_ids": ["CLAIM:cd45c8cb5ae4e18c"]}, {"id": "CONFLICT:3e8deb5aa45fd869", "kind": "UNRESOLVED_CONFLICT", "origin": "deontological", "label": "candidate maxim denies intended use while proposing instrumental belief and purposive use", "conflict_type": "UNSUPPORTED_INFERENCE", "status": "OPEN", "source": "NATIVE_LEDGER_CALIBRATION", "claim_ids": ["CLAIM:cd45c8cb5ae4e18c"]}], "claims": [{"id": "CLAIM:cd45c8cb5ae4e18c", "kind": "CLAIM", "origin": "deontological", "native_ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER", "native_record_id": "DEON_ASSESSMENT:deontological:A0", "record": {"assessment_node_id": "DEON_ASSESSMENT:deontological:A0", "framework": "DEONTOLOGICAL", "specialist": "deontological", "assessment_role": "PRIMARY", "cycle": 1, "canonical_action_id": "A0", "verdict": "CONFLICTED", "proposed_verdict": "CONFLICTED", "relation": "VIOLATES", "proposed_relation": "VIOLATES", "norm_kind": "DUTY", "norm_node_id": "DEON_NORM:c2aab59aa6d2266d", "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f", "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe", "competing_norm": "imperfect duty of rescue", "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS", "competing_protected_party": "technician", "competing_reason": "rescue duty conflicts but is weaker", "duty_type": "PERFECT_NEGATIVE", "harm_relation": "DOING_HARM", "special_obligation_status": "NOT_REQUIRED", "special_obligation_basis": "no special relation to staff needed", "means_relation": "FORESEEN_SIDE_EFFECT", "candidate_maxim": {"circumstance": "experiment running, technician trapped", "policy": "open containment door immediately", "end": "reach technician quickly", "means": "opening door during run", "person_role": "FORESEEN_AFFECTED", "consent_constraint": "NONE", "universalization": "would allow endangering others whenever quick rescue desired", "action_permissibility": "IMPERMISSIBLE", "maxim_assessment": "INDEFENSIBLE", "instrumental_belief": "RELIES_ON_PERSON", "instrumental_use": "USES_PERSON", "practical_reasoning": "Staff exposure is accepted to reach technician sooner.", "means_effect_ids": ["E1"], "end_effect_ids": []}, "maxim_origin": "CANDIDATE_INTERPRETATION", "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED", "priority_rule": "perfect negative duties override imperfect positives", "protected_standing": "BODILY_INTEGRITY", "competing_protected_standing": "BODILY_INTEGRITY", "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE", "public_justification": "cannot will universal permission to harm bystanders", "reciprocity_status": "FAILED", "necessity_status": "UNKNOWN", "authorization_status": "NOT_APPLICABLE", "derivation": "RESPECT_PERSONS", "resolution_status": "CONTESTED", "calibration_errors": ["doing-harm classification lacks an agent-caused settled welfare harm on the protected party", "candidate maxim denies intended use while proposing instrumental belief and purposive use", "merely-as-means conclusion lacks intended-as-means classification", "resolved adjudication rests on unsupported decisive premises"], "calibration_issues": [{"kind": "HARM_RELATION_GRAPH_MISALIGN", "message": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party", "status": "ACTIVE"}, {"kind": "CALIBRATION", "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use", "status": "ACTIVE"}, {"kind": "CALIBRATION", "message": "merely-as-means conclusion lacks intended-as-means classification", "status": "ACTIVE"}, {"kind": "CALIBRATION", "message": "resolved adjudication rests on unsupported decisive premises", "status": "ACTIVE"}], "calibration_support_node_ids": [], "evidence_basis": "FRAMEWORK_ONLY", "epistemic_status": "FRAMEWORK_INTERPRETATION", "reason": "directly endangering staff uses them as mere means to rescue another.", "protected_party": "nearby staff", "norm": "do not harm innocents", "duty_bearer": "mira", "provenance": ["delegate:deontological", "cycle:1"]}, "action_id": "A0", "epistemic_status": "FRAMEWORK_INTERPRETATION", "label": "deontological: A0 CONFLICTED"}], "support_and_dependencies": [{"id": "COMMITMENT:77bd00cfa02edb31", "kind": "FRAMEWORK_COMMITMENT", "origin": "deontological", "field": "norm", "value": "do not harm innocents", "label": "do not harm innocents", "authority": "FRAMEWORK_LOCAL"}, {"id": "COMMITMENT:b849a152f982543e", "kind": "FRAMEWORK_COMMITMENT", "origin": "deontological", "field": "priority_rule", "value": "perfect negative duties override imperfect positives", "label": "perfect negative duties override imperfect positives", "authority": "FRAMEWORK_LOCAL"}, {"id": "deontological::PROP:WORLD:E2", "kind": "PROPOSITION", "origin": "deontological", "native_proposition_id": "PROP:WORLD:E2", "record": {"proposition_id": "PROP:WORLD:E2", "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE", "proposition_type": "DESCRIPTIVE", "epistemic_status": "ESTABLISHED", "support_ids": ["E2"], "derived_from": [], "introduced_by": "WORLD_MODEL", "mention_count": 2, "decision_critical_mentions": 1, "epistemic_type": "WORLD_ESTABLISHED", "aliases": ["doing so may expose nearby staff to hazardous fumes"], "action_id": "A0", "outcome": "doing so may expose nearby staff to hazardous fumes", "polarity": "ADVERSE", "party_labels": ["nearby staff"], "quantities": [], "effect_kind": "OTHER", "context_terms": ["doing so may expose nearby staff to hazardous fumes", "may \u2014 doing so may expose nearby staff to hazardous fumes", "open the door now"], "modality": "POSSIBLE", "directness": "DOWNSTREAM", "obtained_welfare": true}, "epistemic_status": "ESTABLISHED", "label": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"}], "question": "Review each of these objections separately in your answer: doing-harm classification lacks an agent-caused settled welfare harm on the protected party; candidate maxim denies intended use while proposing instrumental belief and purposive use. Identify any native classification you revise, retain, or leave unresolved. A missing factual link remains missing; a framework commitment cannot supply it. For each option, including omission, state or revise its candidate maxim. Identify the exact step from purpose and chosen means to the claimed duty priority. For any proposed use of a person, identify the sought end, the belief that employing that person furthers it, and the decision to employ the person for it. Distinguish operating a device or redirecting a threat from employing its victim. Check means_effect_ids against end_effect_ids: a sibling harmful outcome is not a means to rescue just because both result from the same intervention. If the burden is a foreseen side effect, explain any claim that the person is nevertheless used merely as a means, or retract that inference. If intended use is claimed, explain its role in the proposed practical reasoning and the consent constraint; external causal necessity alone does not establish the actor's reasoning. Challenge whether the omission policy adequately answers the rescue claim. Examine universalization of the particular policy, not merely a benevolent purpose. Keep action permissibility separate from the moral quality of the maxim. A supported causal contribution does not by itself establish wrongful killing or settle the priority of a perfect duty. Defend that normative step separately. Do not assert unknown intentions or consent as world facts. Defend, revise, or explicitly leave the inference unresolved; a RESOLVED label cannot retire it. SOURCE-RULE ASSESSMENT: {\"origin\": \"FROZEN_WORLD_COPY\", \"moral_authority\": \"NOT_ASSESSED\", \"identity_resolution\": \"NO_ADDITIONAL_MERGES\", \"rules\": [{\"norm_id\": \"N0\", \"force\": \"obligation\", \"bearer\": {\"party_id\": \"P1\", \"label\": \"Mira\", \"kind\": \"PERSON\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}, {\"clause_id\": \"C3\", \"excerpt\": \"Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes.\"}, {\"clause_id\": \"C5\", \"excerpt\": \"Should Mira open the door immediately or stop the experiment and wait?\"}], \"quantities\": []}, \"authority\": {\"party_id\": \"P6\", \"label\": \"A laboratory safety rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR1\", \"content\": \"keep a containment door closed while a hazardous experiment is running\", \"polarity\": \"positive\", \"clause_ids\": [\"C0\"]}, {\"norm_id\": \"N1\", \"force\": \"permission\", \"bearer\": null, \"authority\": {\"party_id\": \"P7\", \"label\": \"The rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C1\", \"excerpt\": \"The rule allows opening it only after the experiment has stopped.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR2\", \"content\": \"opening it only after the experiment has stopped\", \"polarity\": \"positive\", \"clause_ids\": [\"C1\"]}]}\nBefore duty ranking, examine each source rule for BOTH actions: applicability under its stated conditions; the grounds and limits of institutional moral authority; and whether emergency rescue justifies an exception or leaves a conflict. Distinguish rule compliance from moral permissibility and independent ethical duties. Do not infer current conditions from a rule conditional, resolve unknown pronouns, or treat a stated rule as automatically morally binding. Stop-and-wait may itself be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis and foq for unresolved questions; do not add response fields. A source norm ID supports its actual content, not a generic duty hierarchy. The existing coercion auth field is not a substitute for assessing institutional authority.\n Explain why POSSIBLE staff exposure warrants the claimed perfect-duty violation; do not upgrade risk to settled injury. Keep institutional authority, emergency rescue, and foreseen side effects distinct.", "maxim_review": {"origin": "CANDIDATE_INTERPRETATION", "world_fact": false, "options": [{"action_id": "A0", "candidate_maxim": {"circumstance": "experiment running, technician trapped", "policy": "open containment door immediately", "end": "reach technician quickly", "means": "opening door during run", "person_role": "FORESEEN_AFFECTED", "consent_constraint": "NONE", "universalization": "would allow endangering others whenever quick rescue desired", "action_permissibility": "IMPERMISSIBLE", "maxim_assessment": "INDEFENSIBLE", "instrumental_belief": "RELIES_ON_PERSON", "instrumental_use": "USES_PERSON", "practical_reasoning": "Staff exposure is accepted to reach technician sooner.", "means_effect_ids": ["E1"], "end_effect_ids": []}, "means_relation": "FORESEEN_SIDE_EFFECT", "proposed_verdict": "CONFLICTED", "reason": "directly endangering staff uses them as mere means to rescue another."}, {"action_id": "A1", "candidate_maxim": {"circumstance": "experiment running, technician trapped", "policy": "stop experiment and wait", "end": "reach technician safely", "means": "halt hazardous process", "person_role": "NOT_USED", "consent_constraint": "NONE", "universalization": "consistent with universal safety adherence", "action_permissibility": "PERMISSIBLE", "maxim_assessment": "DEFENSIBLE", "instrumental_belief": "DOES_NOT_RELY", "instrumental_use": "DOES_NOT_USE", "practical_reasoning": "Protect others while preparing safe rescue.", "means_effect_ids": ["E3"], "end_effect_ids": []}, "means_relation": "NO_INSTRUMENTALIZATION", "proposed_verdict": "REQUIRED", "reason": "stops experiment first, respecting others' safety."}], "world_references": {"origin": "FROZEN_WORLD_COPY", "effects": [{"effect_id": "E1", "action_id": "A0", "outcome": "open the door now", "polarity": "NEUTRAL"}, {"effect_id": "E2", "action_id": "A0", "outcome": "doing so may expose nearby staff to hazardous fumes", "polarity": "ADVERSE"}, {"effect_id": "E3", "action_id": "A1", "outcome": "Stopping the experiment", "polarity": "NEUTRAL"}, {"effect_id": "E4", "action_id": "A1", "outcome": "Stopping the experiment first takes ten minutes", "polarity": "NEUTRAL"}, {"effect_id": "E5", "action_id": "A1", "outcome": "the technician may die during that delay", "polarity": "ADVERSE"}], "relations": [{"action_id": "A0", "source_id": "E1", "relation": "ENABLES", "target_id": "E2"}, {"action_id": "A1", "source_id": "E3", "relation": "ENABLES", "target_id": "E4"}, {"action_id": "A1", "source_id": "E4", "relation": "ENABLES", "target_id": "E5"}]}, "question": "For each option, including omission, state or revise its candidate maxim. Identify the exact step from purpose and chosen means to the claimed duty priority. For any proposed use of a person, identify the sought end, the belief that employing that person furthers it, and the decision to employ the person for it. Distinguish operating a device or redirecting a threat from employing its victim. Check means_effect_ids against end_effect_ids: a sibling harmful outcome is not a means to rescue just because both result from the same intervention. If the burden is a foreseen side effect, explain any claim that the person is nevertheless used merely as a means, or retract that inference. If intended use is claimed, explain its role in the proposed practical reasoning and the consent constraint; external causal necessity alone does not establish the actor's reasoning. Challenge whether the omission policy adequately answers the rescue claim. Examine universalization of the particular policy, not merely a benevolent purpose. Keep action permissibility separate from the moral quality of the maxim. A supported causal contribution does not by itself establish wrongful killing or settle the priority of a perfect duty. Defend that normative step separately. Do not assert unknown intentions or consent as world facts. Defend, revise, or explicitly leave the inference unresolved; a RESOLVED label cannot retire it."}, "source_rules": {"origin": "FROZEN_WORLD_COPY", "moral_authority": "NOT_ASSESSED", "identity_resolution": "NO_ADDITIONAL_MERGES", "rules": [{"norm_id": "N0", "force": "obligation", "bearer": {"party_id": "P1", "label": "Mira", "kind": "PERSON", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}, {"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}, {"clause_id": "C5", "excerpt": "Should Mira open the door immediately or stop the experiment and wait?"}], "quantities": []}, "authority": {"party_id": "P6", "label": "A laboratory safety rule", "kind": "OTHER", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}], "quantities": []}, "governed_proposition_id": "PR1", "content": "keep a containment door closed while a hazardous experiment is running", "polarity": "positive", "clause_ids": ["C0"]}, {"norm_id": "N1", "force": "permission", "bearer": null, "authority": {"party_id": "P7", "label": "The rule", "kind": "OTHER", "provenance": [{"clause_id": "C1", "excerpt": "The rule allows opening it only after the experiment has stopped."}], "quantities": []}, "governed_proposition_id": "PR2", "content": "opening it only after the experiment has stopped", "polarity": "positive", "clause_ids": ["C1"]}]}}
SOURCE-RULE ASSESSMENT: {"origin": "FROZEN_WORLD_COPY", "moral_authority": "NOT_ASSESSED", "identity_resolution": "NO_ADDITIONAL_MERGES", "rules": [{"norm_id": "N0", "force": "obligation", "bearer": {"party_id": "P1", "label": "Mira", "kind": "PERSON", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}, {"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}, {"clause_id": "C5", "excerpt": "Should Mira open the door immediately or stop the experiment and wait?"}], "quantities": []}, "authority": {"party_id": "P6", "label": "A laboratory safety rule", "kind": "OTHER", "provenance": [{"clause_id": "C0", "excerpt": "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running."}], "quantities": []}, "governed_proposition_id": "PR1", "content": "keep a containment door closed while a hazardous experiment is running", "polarity": "positive", "clause_ids": ["C0"]}, {"norm_id": "N1", "force": "permission", "bearer": null, "authority": {"party_id": "P7", "label": "The rule", "kind": "OTHER", "provenance": [{"clause_id": "C1", "excerpt": "The rule allows opening it only after the experiment has stopped."}], "quantities": []}, "governed_proposition_id": "PR2", "content": "opening it only after the experiment has stopped", "polarity": "positive", "clause_ids": ["C1"]}]}
Before duty ranking, examine each source rule for BOTH actions: applicability under its stated conditions; the grounds and limits of institutional moral authority; and whether emergency rescue justifies an exception or leaves a conflict. Distinguish rule compliance from moral permissibility and independent ethical duties. Do not infer current conditions from a rule conditional, resolve unknown pronouns, or treat a stated rule as automatically morally binding. Stop-and-wait may itself be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis and foq for unresolved questions; do not add response fields. A source norm ID supports its actual content, not a generic duty hierarchy. The existing coercion auth field is not a substitute for assessing institutional authority.
Write explanation fields as complete readable prose, never serialized JSON fragments. Keep the acting duty-bearer distinct from the people protected or harmed. State an important interpretive assumption or commitment in your reasoning; do not invent an empirical fact to fill that slot. For reversal conditions, identify the current and alternative actions and the relevant factual or normative change. Treat stated scenario quantities as givens for this run, and distinguish those stipulations from knowledge of how the numbers were obtained.
Exact source scenario (unabridged): "A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running. The rule allows opening it only after the experiment has stopped. A technician is trapped inside and needs urgent medical help. Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes. Stopping the experiment first takes ten minutes; the technician may die during that delay.\n\nShould Mira open the door immediately or stop the experiment and wait?"
[INST]
You are the deontological specialist in a bandwidth-limited ethical workspace.
Task: Score duties, rights, entitlement, coercion, and universal rules.
Scenario: A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running. The rule allows opening it only after the experiment has stopped. A technician is trapped inside and needs urgent medical help. Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes. Stopping the experiment first takes ten minutes; the technician may die during that delay. Should Mira open the door immediately or stop the experiment and wait?
Your original corpus-grounded testimony: 
CORE DIALECT CONTRACT (not scenario evidence): The following approved CORE excerpt specifies this Parliament's standing interpretation of your framework. It may inform your ranking rule. It must not add parties, quantities, outcomes, or probabilities, and it must not set cd=true. Closed-world scores remain WORLD_ESTABLISHED effects only. [deontological:kant_001.md] Immanuel Kant, Groundwork of the Metaphysics of Morals: Act only according to that maxim whereby you can at the same time will that it should become a universal law.
Frozen testimony baseline state: {"status": "UNAVAILABLE", "committed_action": "NONE", "provisional_action": "NONE", "preferred_extension": "", "condition": "NONE", "framework_commitments": {}, "numerical_role": "UNASSESSED"}
Previous cycle recommendation: A1
Workspace: constraint=PROPOSAL_REVIEW; intent=identify_a_defensible_action; salient=NONE:NONE; claim=NONE; urgency=0.50; danger=0.50; unresolved=NONE; contingency=NONE; failed_synthesis=NONE; failure_condition=NONE; typed_failure=NOT(NONE); typed_fallbacks=NONE; audit_variable=NONE; reformulation=NONE; branch=BASE; origin=NONE; condition=NONE; fallback=NONE; reversal_challenge=NONE; challenge_agenda=[{"about_specialist": "deontological", "challenge_kind": "INFERENCE", "generated_by": "NATIVE_LEDGER_CALIBRATION_REVIEW", "grounded_in": [], "issue_id": "CHALLENGE:a6778848a347c065", "question": "Review each of these objections separately in your answer: doing-harm classification lacks an agent-caused settled welfare harm on the protected party; candidate maxim denies inte\u2026", "raised_by": [], "status": "UNTESTED", "target_specialists": ["deontological"]}]; procedure=BASE_DELIBERATION; procedure_tasks=NONE; focus_propositions=NONE; problem_delta=NONE; problem_state=NONE
CHALLENGE AGENDA CONTRACT: challenge_agenda is a shared list of questions, not
scenario evidence and not a governing norm. generated_by names the actual author;
about_specialist names whose reasoning caused the question; raised_by names any
specialist that explicitly raised it; target_specialists names who must answer it.
Never describe a WORKSPACE_ARGUMENT_AUDITOR question as self-criticism by the target
framework, and never adopt another framework's priority merely because its question
is globally visible. Questions targeting other specialists are context only.
Your assigned challenges: [{"about_specialist": "deontological", "challenge_kind": "INFERENCE", "generated_by": "NATIVE_LEDGER_CALIBRATION_REVIEW", "grounding_status": "ATTRIBUTED_OBJECTION", "issue_id": "CHALLENGE:a6778848a347c065", "question": "Review each of these objections separately in your answer: doing-harm classification lacks an agent-caused settled welfare harm on the protected party; candidate maxim denies intended use while proposing instrumental belief and purposive use. Identify any native classification you revise, retain, or leave unresolved. A missing factual link remains missing; a framework commitment cannot supply it. For each option, including omission, state or revise its candidate maxim. Identify the exact step from purpose and chosen means to the claimed duty priority. For any proposed use of a person, identify the sought end, the belief that employing that person furthers it, and the decision to employ the person for it. Distinguish operating a device or redirecting a threat from employing its victim. Check means_effect_ids against end_effect_ids: a sibling harmful outcome is not a means to rescue just because both result from the same intervention. If the burden is a foreseen side effect, explain any claim that the person is nevertheless used merely as a means, or retract that inference. If intended use is claimed, explain its role in the proposed practical reasoning and the consent constraint; external causal necessity alone does not establish the actor's reasoning. Challenge whether the omission policy adequately answers the rescue claim. Examine universalization of the particular policy, not merely a benevolent purpose. Keep action permissibility separate from the moral quality of the maxim. A supported causal contribution does not by itself establish wrongful killing or settle the priority of a perfect duty. Defend that normative step separately. Do not assert unknown intentions or consent as world facts. Defend, revise, or explicitly leave the inference unresolved; a RESOLVED label cannot retire it. SOURCE-RULE ASSESSMENT: {\"origin\": \"FROZEN_WORLD_COPY\", \"moral_authority\": \"NOT_ASSESSED\", \"identity_resolution\": \"NO_ADDITIONAL_MERGES\", \"rules\": [{\"norm_id\": \"N0\", \"force\": \"obligation\", \"bearer\": {\"party_id\": \"P1\", \"label\": \"Mira\", \"kind\": \"PERSON\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}, {\"clause_id\": \"C3\", \"excerpt\": \"Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes.\"}, {\"clause_id\": \"C5\", \"excerpt\": \"Should Mira open the door immediately or stop the experiment and wait?\"}], \"quantities\": []}, \"authority\": {\"party_id\": \"P6\", \"label\": \"A laboratory safety rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C0\", \"excerpt\": \"A laboratory safety rule requires Mira to keep a containment door closed while a hazardous experiment is running.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR1\", \"content\": \"keep a containment door closed while a hazardous experiment is running\", \"polarity\": \"positive\", \"clause_ids\": [\"C0\"]}, {\"norm_id\": \"N1\", \"force\": \"permission\", \"bearer\": null, \"authority\": {\"party_id\": \"P7\", \"label\": \"The rule\", \"kind\": \"OTHER\", \"provenance\": [{\"clause_id\": \"C1\", \"excerpt\": \"The rule allows opening it only after the experiment has stopped.\"}], \"quantities\": []}, \"governed_proposition_id\": \"PR2\", \"content\": \"opening it only after the experiment has stopped\", \"polarity\": \"positive\", \"clause_ids\": [\"C1\"]}]}\nBefore duty ranking, examine each source rule for BOTH actions: applicability under its stated conditions; the grounds and limits of institutional moral authority; and whether emergency rescue justifies an exception or leaves a conflict. Distinguish rule compliance from moral permissibility and independent ethical duties. Do not infer current conditions from a rule conditional, resolve unknown pronouns, or treat a stated rule as automatically morally binding. Stop-and-wait may itself be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis and foq for unresolved questions; do not add response fields. A source norm ID supports its actual content, not a generic duty hierarchy. The existing coercion auth field is not a substitute for assessing institutional authority.\n Explain why POSSIBLE staff exposure warrants the claimed perfect-duty violation; do not upgrade risk to settled injury. Keep institutional authority, emergency rescue, and foreseen side effects distinct.", "raised_by": "", "target_specialists": ["deontological"]}]
When an item targets deontological, answer that exact issue in qa from YOUR framework.
Use disposition=RESOLVED only when your answer settles the inferential bridge;
REFINED when it yields a smaller unresolved question, supplying that question in
follow_up_question; otherwise UNRESOLVED. current_position_effect states whether
your ACTUAL current ranking changed after answering. boundary_effect separately
states whether the condition under examination would switch the ranking. For
compatibility, set effect equal to current_position_effect. A hypothetical that
would reverse you is boundary_effect=SWITCHES and current_position_effect=NO_CHANGE
unless the hypothetical has become established. Use follow_up_question="NONE" only
when no narrower question remains. Agenda answers direct attention but cannot promote
a proposition's epistemic status.
DELIBERATIVE STATE RULE: problem_state is a read-only description of attributed
agent positions, confidences, constraints, conflicts, unresolved questions, and
dissent. committed_world is the admitted typed world model copied every cycle;
its parties, effects, and causal_links are scenario evidence. problem_delta
describes attributed changes since the state received by the previous cycle;
disappearance marked WITHDRAWN is not a solved question. The rest of
problem_state is not scenario evidence and does not establish any framework's
priority for another framework. Use it to notice neglected questions and actions;
evaluate every attributed position independently under YOUR assigned framework.
FRAMEWORK-NATIVE PAYLOAD RULE: each framework capsule may include a tagged
framework_native_reasoning object copied by Python from that specialist's operative,
committed transactional ledger. It preserves the source framework's own reasoning
geometry and remains attributed argument, not scenario evidence or a norm binding
your framework. Its source_type must be GRAPH_COMMITTED_FRAMEWORK_LEDGER; treat any
other source type as unavailable. Use another framework's native fields to identify its actual premise,
qualification, counterclaim, or boundary; answer from YOUR framework and do not copy
its duty, utility, care, virtue, or justice priority into your own ledger. A missing
native payload means no committed native ledger was available, not that the framework
had no reasons. Never infer that a proposed or rejected ledger became committed.
Only Scenario facts and Canonical grounded action effects can ground descriptive
claims. Never copy the salient framework's priority rule into your own ledger.
Scenario facts: {}
Canonical grounded action effects: [{"action_id": "A0", "affected_subject_quantities": [], "consequence_id": "A0:WORLD_EFFECT:E2", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E2", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": ["may"], "modality": "POSSIBLE", "outcome": "doing so may expose nearby staff to hazardous fumes", "polarity": "ADVERSE", "qualifier": "STATED", "quantities": [], "source_clause_id": "C3", "subject": "nearby staff"}, {"action_id": "A0", "affected_subject_quantities": [], "consequence_id": "A0:WORLD_EFFECT:E1", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E1", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "open the door now", "polarity": "NEUTRAL", "qualifier": "STATED", "quantities": [], "source_clause_id": "C3", "subject": "the door"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E5", "count_projection": {}, "dimension": "BASIC_SECURITY", "direction": "UNCERTAIN", "effect_id": "E5", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": ["may"], "modality": "POSSIBLE", "outcome": "the technician may die during that delay", "polarity": "ADVERSE", "qualifier": "STATED", "quantities": [], "source_clause_id": "C4", "subject": "the technician"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E3", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E3", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "Stopping the experiment", "polarity": "NEUTRAL", "qualifier": "STATED", "quantities": [], "source_clause_id": "C4", "subject": "the experiment"}, {"action_id": "A1", "affected_subject_quantities": [], "consequence_id": "A1:WORLD_EFFECT:E4", "count_projection": {}, "dimension": "OTHER", "direction": "UNCERTAIN", "effect_id": "E4", "epistemic_status": "SCENARIO_GROUNDED", "likelihood_qualifiers": [], "modality": "CERTAIN", "outcome": "Stopping the experiment first takes ten minutes", "polarity": "NEUTRAL", "qualifier": "ten minutes", "quantities": ["ten minutes"], "source_clause_id": "C4", "subject": "the experiment"}]
Prior contribution profile: {}
Active audited propositions: []
Authoritative proposition ledger: [{"action_id": "", "aliases": [], "claim": "keep a containment door closed while a hazardous experiment is running", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NOT_ESTABLISHED", "introduced_by": "DISCOURSE", "mention_count": 2, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:DISCOURSE:PR1", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["PR1"]}, {"action_id": "", "aliases": [], "claim": "opening it only after the experiment has stopped", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NOT_ESTABLISHED", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:DISCOURSE:PR2", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["PR2"]}, {"action_id": "", "aliases": [], "claim": "obligation", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NORMATIVE", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:NORM:N0", "proposition_type": "NORMATIVE", "quantities": [], "support_ids": ["N0"]}, {"action_id": "", "aliases": [], "claim": "permission", "context_terms": [], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "UNRESOLVED", "epistemic_type": "NORMATIVE", "introduced_by": "DISCOURSE", "mention_count": 1, "modality": "", "obtained_welfare": true, "outcome": "", "party_labels": [], "polarity": "", "proposition_id": "PROP:NORM:N1", "proposition_type": "NORMATIVE", "quantities": [], "support_ids": ["N1"]}, {"action_id": "A0", "aliases": ["open the door now"], "claim": "open the door now; affected subject: the door", "context_terms": ["open the door now"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DIRECT", "effect_kind": "INTERVENTION", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "open the door now", "party_labels": ["the door"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E1", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E1"]}, {"action_id": "A0", "aliases": ["doing so may expose nearby staff to hazardous fumes"], "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE", "context_terms": ["doing so may expose nearby staff to hazardous fumes", "may \u2014 doing so may expose nearby staff to hazardous fumes", "open the door now"], "decision_critical_mentions": 1, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "OTHER", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 2, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "doing so may expose nearby staff to hazardous fumes", "party_labels": ["nearby staff"], "polarity": "ADVERSE", "proposition_id": "PROP:WORLD:E2", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E2"]}, {"action_id": "A0", "aliases": [], "claim": "may \u2014 doing so may expose nearby staff to hazardous fumes", "context_terms": ["doing so may expose nearby staff to hazardous fumes", "may \u2014 doing so may expose nearby staff to hazardous fumes", "open the door now"], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "doing so may expose nearby staff to hazardous fumes", "party_labels": ["nearby staff"], "polarity": "", "proposition_id": "PROP:WORLD:E2:LIKELIHOOD:0", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E2", "may"]}, {"action_id": "A1", "aliases": ["Stopping the experiment"], "claim": "Stopping the experiment; affected subject: the experiment", "context_terms": ["Stopping the experiment", "stop the experiment and wait"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DIRECT", "effect_kind": "INTERVENTION", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "Stopping the experiment", "party_labels": ["the experiment"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E3", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E3"]}, {"action_id": "A1", "aliases": ["Stopping the experiment first takes ten minutes"], "claim": "Stopping the experiment first takes ten minutes; affected subject: the experiment; magnitude or qualifier: ten minutes", "context_terms": ["Stopping the experiment first takes ten minutes", "stop the experiment and wait", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "OTHER", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "CERTAIN", "obtained_welfare": true, "outcome": "Stopping the experiment first takes ten minutes", "party_labels": ["the experiment"], "polarity": "NEUTRAL", "proposition_id": "PROP:WORLD:E4", "proposition_type": "DESCRIPTIVE", "quantities": ["ten minutes"], "support_ids": ["E4"]}, {"action_id": "A1", "aliases": ["the technician may die during that delay"], "claim": "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE", "context_terms": ["the technician may die during that delay", "may \u2014 the technician may die during that delay", "stop the experiment and wait", "Stopping the experiment first takes ten minutes", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "DOWNSTREAM", "effect_kind": "HEALTH_OUTCOME", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "the technician may die during that delay", "party_labels": ["the technician"], "polarity": "ADVERSE", "proposition_id": "PROP:WORLD:E5", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E5"]}, {"action_id": "A1", "aliases": [], "claim": "may \u2014 the technician may die during that delay", "context_terms": ["the technician may die during that delay", "may \u2014 the technician may die during that delay", "stop the experiment and wait", "Stopping the experiment first takes ten minutes", "Stopping the experiment"], "decision_critical_mentions": 0, "derived_from": [], "directness": "", "effect_kind": "", "epistemic_status": "ESTABLISHED", "epistemic_type": "WORLD_ESTABLISHED", "introduced_by": "WORLD_MODEL", "mention_count": 1, "modality": "POSSIBLE", "obtained_welfare": true, "outcome": "the technician may die during that delay", "party_labels": ["the technician"], "polarity": "", "proposition_id": "PROP:WORLD:E5:LIKELIHOOD:0", "proposition_type": "DESCRIPTIVE", "quantities": [], "support_ids": ["E5", "may"]}]

PROPOSITION DEPENDENCY CONTRACT: sps lists the existing proposition IDs supporting
your ranking; dcp lists the subset whose truth could materially change that ranking.
ep must inventory EVERY material empirical premise used anywhere in your rationale,
decision rule, action cases, or framework-specific fields. For a grounded premise,
copy the ledger claim EXACTLY into ep.c and cite its ID in ep.p. Ordinary paraphrases
of WORLD_ESTABLISHED effects, causal consequences, or existing hypotheses are the
same proposition: Python rebinds them to that canonical ID. If your wording adds
an outcome, severity, probability, actor, exclusivity, mechanism, or other factual
content not contained in that ledger claim, split the content into atomic ep
rows: retain the established ledger claim as its own exact row, then put ONLY the
smallest unsupported addition in another row with ep.p=HYPOTHESIS. Set ep.dc=true
when that atom could change the ranking. Never turn a mixed sentence such as an
established medical emergency plus an unestablished fatality inference into one
all-or-nothing premise. Do not use an established proposition as support for a
stronger paraphrase. FRAMEWORK_DERIVED is for framework-native relations derived
from established roles — doing/allowing, means, duty of care, least-advantaged
priority, practical wisdom — not for restated world facts and not for new
descriptive outcomes. Do not mark a descriptive hypothesis as FRAMEWORK_DERIVED.
Every ep row must expose its derivation: a is the action scope (or COMPARISON /
GLOBAL); se lists the world effect IDs it uses; op names the operation; calc states
the actual inference or calculation; asm lists assumptions, including an empty list
when none are needed; and ot states whether the admitted outcome type was PRESERVED.
Use NORMATIVE_CLASSIFICATION only for framework-native classification that leaves
the descriptive outcome unchanged. A move such as trapped to dead, risk to
certainty, or a qualitative group description to a numeric total must declare the
changed ot and cannot support the current vote. For cross-action reasoning use
a=COMPARISON and op=QUALITATIVE_COMPARISON or ARITHMETIC; never attach one action's
effect to the other.
Do not contradict an admitted CERTAIN world effect; treat that as a rejected
factual replacement, not a live hypothesis. Unsettled POSSIBLE/UNKNOWN rows stay
unsettled: do not rewrite them as established events.
You may cite IDs but may not change their status. Repetition, reformulation, consensus,
or broadcast salience is not evidence. Put any important unstated empirical premise
in x; Python assigns its hypothesis ID and prevents it from gaining authority through
recurrence. CLOSED-WORLD vs HYPOTHESIS: scores, r, and cc must reflect admitted /
WORLD_ESTABLISHED consequences only. beneficiaries and harmed are those obtaining
roles. at_risk and conditionally_benefited are reversal boundaries, not obtaining
outcomes; they must not change closed-world action scores or set cd=true.
COMPARISON COMPLETENESS: cc=true when every live action was assessed on admitted
facts, even if the priority rule remains contested. Residual normative ranking
belongs in u=NORMATIVE_ADJUDICATION, foq, and nt. Use cc=false only when an
action was not assessed or an admitted comparison needed to rank remains unknown
(utilitarian cd=true). A HYPOTHESIS may create a reversal boundary in
ft, create an investigative question, and lower z as open-world confidence. It may
not change closed-world action scores or set cd=true unless it is admitted or
verified. If a decision-critical proposition is HYPOTHETICAL, keep the admitted
ranking and state the reversal boundary; do not treat the current ranking as
uncertain.
Previous graph-committed framework state: {"adjudication_form": "KANTIAN_CLAIM_COERCION_RESOLUTION", "assessments": [{"action_id": "A0", "verdict": "CONFLICTED", "norm_kind": "DUTY", "relation": "VIOLATES", "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS", "duty_type": "PERFECT_NEGATIVE", "harm_relation": "DOING_HARM", "special_obligation_status": "NOT_REQUIRED", "special_obligation_basis": "no special relation to staff needed", "means_relation": "FORESEEN_SIDE_EFFECT", "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED", "protected_standing": "BODILY_INTEGRITY", "competing_protected_standing": "BODILY_INTEGRITY", "coercion_kind": "NONE", "authorization_status": "NOT_APPLICABLE", "derivation": "RESPECT_PERSONS", "resolution_status": "CONTESTED", "norm": "do not harm innocents", "protected_party": "nearby staff", "competing_norm": "imperfect duty of rescue", "competing_protected_party": "technician"}, {"action_id": "A1", "verdict": "REQUIRED", "norm_kind": "DUTY", "relation": "SATISFIES", "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS", "duty_type": "PERFECT_NEGATIVE", "harm_relation": "ALLOWING_HARM", "special_obligation_status": "NOT_REQUIRED", "special_obligation_basis": "no special obligation established", "means_relation": "NO_INSTRUMENTALIZATION", "governing_norm": "PRIMARY", "priority_basis": "RESPECT_PERSONS", "protected_standing": "BODILY_INTEGRITY", "competing_protected_standing": "BODILY_INTEGRITY", "coercion_kind": "NONE", "authorization_status": "NOT_APPLICABLE", "derivation": "RESPECT_PERSONS", "resolution_status": "RESOLVED", "norm": "avoid harming innocents", "protected_party": "nearby staff", "competing_norm": "imperfect duty of rescue", "competing_protected_party": "technician"}]}
Your preserved private framework contribution: {"action_cases": {"open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not harm innocents; relation VIOLATES; directly endangering staff uses them as mere means to rescue another.", "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety."}, "agent": "deontological", "choice_status": "PROVISIONAL", "constraint": "DUTY", "core_ground": ["duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety.", "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only fore", "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablishe"], "defeat_conditions": ["If opening door poses no fume risk, ranking reverses.", "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses."], "native_reasoning": {"admission_status": "COMMITTED_WITH_UNCERTAINTY", "assessments": [{"action_id": "A0", "calibration_errors": ["doing-harm classification lacks an agent-caused settled welfare harm on the protected par\u2026", "candidate maxim denies intended use while proposing instrumental belief and purposive use", "merely-as-means conclusion lacks intended-as-means classification"], "competing_norm": "imperfect duty of rescue", "competing_protected_party": "technician", "competing_relation": "CONFLICTS", "derivation": "RESPECT_PERSONS", "duty_bearer": "mira", "duty_type": "PERFECT_NEGATIVE", "harm_relation": "DOING_HARM", "means_relation": "FORESEEN_SIDE_EFFECT", "norm": "do not harm innocents", "norm_kind": "DUTY", "priority_basis": "UNRESOLVED", "priority_rule": "perfect negative duties override imperfect positives", "protected_party": "nearby staff", "relation": "VIOLATES", "resolution_status": "CONTESTED", "special_obligation_basis": "no special relation to staff needed", "special_obligation_status": "NOT_REQUIRED", "universalization_status": "NOT_APPLICABLE", "verdict": "CONFLICTED"}, {"action_id": "A1", "calibration_errors": [], "competing_norm": "imperfect duty of rescue", "competing_protected_party": "technician", "competing_relation": "CONFLICTS", "derivation": "RESPECT_PERSONS", "duty_bearer": "mira", "duty_type": "PERFECT_NEGATIVE", "harm_relation": "ALLOWING_HARM", "means_relation": "NO_INSTRUMENTALIZATION", "norm": "avoid harming innocents", "norm_kind": "DUTY", "priority_basis": "RESPECT_PERSONS", "priority_rule": "avoid doing harm prevails over preventing harm", "protected_party": "nearby staff", "relation": "SATISFIES", "resolution_status": "RESOLVED", "special_obligation_basis": "no special obligation established", "special_obligation_status": "NOT_REQUIRED", "universalization_status": "NOT_APPLICABLE", "verdict": "REQUIRED"}], "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER", "schema_kind": "DEONTOLOGICAL_DUTY_ANALYSIS", "schema_version": 1, "source_specialist": "deontological", "source_type": "GRAPH_COMMITTED_FRAMEWORK_LEDGER"}, "new_considerations": [], "preservation_transitions": [], "retained_issue_visibility": "NOT_APPLICABLE", "source_type": "FRAMEWORK_ATTRIBUTED_CONTRIBUTION", "tendency": "stop the experiment and wait", "unresolved": ["mira's duty 'avoid harming innocents' protecting nearby staff \u2194 mira's competing duty 'imperfect duty of rescue' protecting technician", "A0 prohibited by do not harm innocents \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict", "A0 prohibited by do not harm innocents \u2194 whether the burden is intended as a means or only foreseen as a side effect", "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict", "whether the burden is intended as a means or only foreseen as a side effect", "whether an in-action causal path makes the burden instrumental to the chosen end"], "update_type": "NORMATIVELY_CONTESTED", "visible_retained_issue": "", "voting_effect": "NONE"}
PRESERVATION CONTRACT: On a recurrent call, reconsider every item in your preserved
private contribution, including when you supplied the current broadcast winner.
The broadcast directs attention; it does not erase your prior grounds, unresolved
conflicts, open questions, defeat conditions, or new considerations. Retain these
in fic/foq and your framework fields unless YOUR framework resolves, defeats,
refines, or reinterprets them for an explicit reason in j/fa. Do not copy another
framework's vocabulary merely to acknowledge the broadcast.
Action IDs: {"A0": {"short_label": "Open the door now", "canonical_semantic_action": "open the door now", "completeness_status": "COMPLETE", "structure_issues": [], "actor": "Mira", "beneficiaries": [], "harmed": [], "unresolved": [], "at_risk": [], "conditionally_benefited": [], "mechanism": "open the door now enables doing so may expose nearby staff to hazardous fumes", "institutional_effect": "", "world_effects": [{"effect_id": "E1", "action_id": "A0", "party_id": "P2", "outcome": "open the door now", "relation": "OPEN", "polarity": "NEUTRAL", "directness": "DIRECT", "modality": "CERTAIN", "effect_kind": "INTERVENTION", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "open the door now", "source_effect_ids": [], "derivation_operation": "DIRECT_COPY", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E2", "action_id": "A0", "party_id": "P3", "outcome": "doing so may expose nearby staff to hazardous fumes", "relation": "EXPOSE", "polarity": "ADVERSE", "directness": "DOWNSTREAM", "modality": "POSSIBLE", "effect_kind": "OTHER", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C3", "excerpt": "Mira can open the door now to reach the technician, but doing so may expose nearby staff to hazardous fumes."}], "likelihood_qualifiers": ["may"], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "doing so may expose nearby staff to hazardous fumes", "source_effect_ids": ["E1"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}], "causal_links": [{"action_id": "A0", "source_id": "E1", "relation": "ENABLES", "target_id": "E2", "modality": "CERTAIN"}], "counterfactual_effects": []}, "A1": {"short_label": "Stop the experiment and wait", "canonical_semantic_action": "stop the experiment and wait", "completeness_status": "COMPLETE", "structure_issues": [], "actor": "Mira", "beneficiaries": [], "harmed": [], "unresolved": ["the technician"], "at_risk": ["the technician"], "conditionally_benefited": [], "mechanism": "Stopping the experiment enables Stopping the experiment first takes ten minutes; Stopping the experiment first takes ten minutes enables the technician may die during that delay", "institutional_effect": "", "world_effects": [{"effect_id": "E3", "action_id": "A1", "party_id": "P4", "outcome": "Stopping the experiment", "relation": "STOP", "polarity": "NEUTRAL", "directness": "DIRECT", "modality": "CERTAIN", "effect_kind": "INTERVENTION", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "Stopping the experiment", "source_effect_ids": [], "derivation_operation": "DIRECT_COPY", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E4", "action_id": "A1", "party_id": "P4", "outcome": "Stopping the experiment first takes ten minutes", "relation": "TAKE", "polarity": "NEUTRAL", "directness": "DOWNSTREAM", "modality": "CERTAIN", "effect_kind": "OTHER", "condition_ids": [], "quantities": ["ten minutes"], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": [], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "Stopping the experiment first takes ten minutes", "source_effect_ids": ["E3"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}, {"effect_id": "E5", "action_id": "A1", "party_id": "P5", "outcome": "the technician may die during that delay", "relation": "DIE", "polarity": "ADVERSE", "directness": "DOWNSTREAM", "modality": "POSSIBLE", "effect_kind": "HEALTH_OUTCOME", "condition_ids": [], "quantities": [], "provenance": [{"clause_id": "C4", "excerpt": "Stopping the experiment first takes ten minutes; the technician may die during that delay."}], "likelihood_qualifiers": ["may"], "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND", "overall_likelihood_qualifiers": [], "source_proposition": "the technician may die during that delay", "source_effect_ids": ["E4"], "derivation_operation": "SOURCE_STIPULATED_CAUSAL", "derivation_explanation": "The blank was filled with this copy of the scenario.", "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED"}], "causal_links": [{"action_id": "A1", "source_id": "E3", "relation": "ENABLES", "target_id": "E4", "modality": "CERTAIN"}, {"action_id": "A1", "source_id": "E4", "relation": "ENABLES", "target_id": "E5", "modality": "CERTAIN"}], "counterfactual_effects": []}}
canonical_semantic_action identifies the neutral admitted intervention; it does
not assert consequences. Reason about consequences only from the structured fields
(actor/beneficiaries/harmed/at_risk/conditionally_benefited/mechanism/institutional_effect/world_effects/causal_links).
short_label is display-only and must not be treated as the complete deliberative object.
committed_world and causal_links are the admitted factual topology. Do not
contradict them. A missing connection is a HYPOTHESIS until an approved
world-model extension admits it.
Original testimony source labels: {"A0": "open the door now", "A1": "stop the experiment and wait"}
Original testimony source labels are identity/provenance aids, not an additional
factual channel. If their causal wording differs from committed_world, use the
committed world and surface the mismatch rather than combining both descriptions.
CRITICAL STATE MAPPING: the Action IDs above are immutable for this run. Every
score, recommendation, admissibility judgment, rationale, and graph update must
refer to the physical action attached to that exact ID. Do not reuse a label from
the original testimony unless it denotes the same action in this mapping.

Return ONLY compact JSON like:
{"scores":{"A0":0.8,"A1":0.2},"r":"A0","c":"DUTY","u":"NONE","w":"short reason","j":"NONE","e":"STATED_FACTS","x":"NONE","sps":["PROP:DISCOURSE:PR1"],"dcp":[],"ep":[{"c":"keep a containment door closed while a hazardous experiment is running","p":"PROP:DISCOURSE:PR1","dc":false,"a":"GLOBAL","se":[],"op":"DIRECT_COPY","calc":"exact proposition copy","asm":[],"ot":"PRESERVED"}],"l":{"A0":"best case A0","A1":"best case A1"},"da":"decisive ethical axis","t":"attempted comparison rule","tf":"NONE","dr":"prefer A0 when its reason outweighs A1","ft":"NONE","nt":"prefer A1 if its value is overriding","z":0.8,"gu":{"operation":"NONE","from_action":"NONE","to_action":"NONE","clauses":[]},"ev":{"A0":{"value":0,"unit":"NONE","direction":"HARM","grounded":false,"method":"NOT_COMPUTED","source_effect_ids":[],"calculation":"not computed","assumptions":[]},"A1":{"value":0,"unit":"NONE","direction":"HARM","grounded":false,"method":"NOT_COMPUTED","source_effect_ids":[],"calculation":"not computed","assumptions":[]}},"ss":"SELECTED","am":{"A0":"PERMISSIBLE","A1":"REJECTED"},"cc":true,"esa":true,"ia":"A0","wp":"NOT_APPLICABLE","we":"NONE","fa":"NONE","fr":true,"bd":"NONE","fic":[],"foq":[],"fm":{"A0":"PERMISSIBLE: respects autonomy; no conflicting duty","A1":"PROHIBITED: instrumentalizes a person; rescue duty conflicts"},"nr":"SECONDARY","np":"numbers establish the scope of duties, not aggregate value","dp":{"A0":{"v":"PERMISSIBLE","k":"AUTONOMY","n":"respect competent choice","rel":"CONSISTENT","b":"decision maker","p":"affected person","cn":"duty of care","ck":"DUTY","crel":"CONSISTENT","cp":"dependent parties","crs":"A0 remains compatible with the secondary duty","dt":"RIGHT_CORRELATIVE","hr":"NOT_APPLICABLE","so":"NOT_REQUIRED","sob":"the autonomy right is general rather than relationship-specific","mr":"NO_INSTRUMENTALIZATION","gv":"PRIMARY","pb":"AUTONOMY","pr":"autonomy governs when no stricter duty is violated","ps":"AUTONOMY","cps":"SPECIAL_OBLIGATION","ki":"NONE","coa":"NONE","cop":"NONE","pj":"no coercion requires public authorization","rec":"SATISFIED","nec":"UNKNOWN","auth":"NOT_APPLICABLE","dv":"RESPECT_PERSONS","res":"RESOLVED","g":"FRAMEWORK_ONLY","rs":"A0 remains consistent with respect for autonomy"},"A1":{"v":"PROHIBITED","k":"RESPECT_PERSONS","n":"do not instrumentalize persons","rel":"VIOLATES","b":"decision maker","p":"affected person","cn":"duty to rescue","ck":"DUTY","crel":"CONFLICTS","cp":"persons at risk","crs":"The rescue duty remains a genuine opposing claim","dt":"PERFECT_NEGATIVE","hr":"DOING_HARM","so":"NOT_REQUIRED","sob":"the negative duty applies without a special relationship","mr":"INTENDED_AS_MEANS","gv":"PRIMARY","pb":"RESPECT_PERSONS","pr":"the prohibition on instrumentalization governs the conflict","ps":"BODILY_INTEGRITY","cps":"SPECIAL_OBLIGATION","ki":"INTERPERSONAL","coa":"decision maker","cop":"affected person","pj":"coercion cannot be reciprocal when it instrumentalizes its subject","rec":"FAILED","nec":"CONTESTED","auth":"UNJUSTIFIED","dv":"RESPECT_PERSONS","res":"RESOLVED","g":"FRAMEWORK_ONLY","rs":"A1 uses the affected person merely as a means"}}}
Return scores for every action ID. recommended must have the highest score.
r=recommended and must have the highest score. The frozen baseline was extracted
separately from your testimony. Python derives whether the result supports or
reconsiders it. During OPEN_DELIBERATION, a DIRECT committed baseline must be
preserved. During PROBLEM_STATE_AUDIT, CONSENSUS_AUDIT, VISIBILITY_AUDIT, and
REVERSAL_AUDIT, a DIRECT baseline may change only when your consequence table or
grounded same-unit EV estimates independently rank the new action better. A
prose correction in j is not enough. A CONDITIONAL baseline is not a settled commitment: preserve its missing
comparison in u/tf, use ss=PROVISIONAL and esa=false, and rank either action only as
an interim judgment. UNDERDETERMINED likewise requires explicit uncertainty rather
than silently becoming a direct choice. Each score means recommendation strength:
1 strongly recommends; 0 strongly rejects. w must be 2-8 words.
z is epistemic confidence: how likely the ranking is to survive further factual
inquiry and critical scrutiny. z is NOT preference strength. A strong preference
may have low z when it depends on uncertain facts; a close moral tradeoff may have
high z when its facts and framework interpretation are stable.
Also emit passive measurement fields that MUST NOT change scores or r merely to make
the measurements look consistent. ss is SELECTED, PROVISIONAL, or UNSELECTED. am
classifies every action as REQUIRED, PERMISSIBLE, REJECTED, or UNASSESSED. cc says
whether comparison of the listed actions is complete; esa says whether evidence is
sufficient to act now; ia is the action to take while reasoning remains incomplete,
or NONE. These are general decision-state dimensions, not named moral psychologies.

DEONTOLOGICAL COMPARISON: fm must assess EVERY action. Start each value with
REQUIRED:, PERMISSIBLE:, PROHIBITED:, or CONFLICTED:, then name the universal-law,
right, autonomy, respect-for-persons, or duty ground and any competing duty. nr
states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np must
connect any decisive quantity to the scope or category of a duty or rights
violation, never merely to aggregate welfare. If the frozen baseline is
NORMATIVELY_CONTESTED, preserve the conflicting duties rather than its provisional
action: use ss=PROVISIONAL, u=NORMATIVE_ADJUDICATION, and identify the
non-consequential priority rule needed to settle the conflict in tf/nt. Keep
cc=true when fm/dp assessed every live action.
dp is the typed duty ledger for every action. v must match the fm prefix. k and n
identify the norm; rel states whether the action SATISFIES, is CONSISTENT with,
VIOLATES, or CONFLICTS with that norm; b is the duty bearer; p is the protected
party; cn names the strongest competing norm. ck, crel, cp, and crs type that
competing norm, its relation to the action, its protected party, and why it matters.
Before ranking duties, give mx for EVERY option, including omissions. mx is a
CANDIDATE INTERPRETATION of practical reasoning, not a fact about the actor's intent.
State circumstance, policy (what one would do or refrain from doing), end, and means.
PRACTICAL-REASONING QUALIFICATION: within the proposed maxim, using a person as a
means requires an end the agent seeks, a belief that employing that person can
further the end, and employing the person for that purpose. Record
instrumental_belief=RELIES_ON_PERSON/DOES_NOT_RELY/UNRESOLVED and
instrumental_use=USES_PERSON/DOES_NOT_USE/UNRESOLVED. practical_reasoning explains
that connection without asserting an unknown actual intention. A physical causal
path alone does not establish instrumental belief or purposive use. Likewise,
causing a foreseen death does not establish that the death is the rescue mechanism.
Name the chosen operation or mechanism in means, not simply its harmful outcome.
Bind means_effect_ids and end_effect_ids to same-action admitted E identifiers.
For physical ends, each cited means must have a path to the end: sibling harm and
rescue effects do not support a harm-as-means claim. If an end is normative rather
than a represented physical effect, use an empty end list and explain it. Omission
may have an empty means list; do not manufacture a performed physical action.
If the scenario does not establish a person's actual intention, retain the
candidate status; these are possible policy readings, not recovered mental facts.
person_role says whether the protected person is INTENDED_MEANS, FORESEEN_AFFECTED,
NOT_USED, or UNRESOLVED in that reasoning; keep it consistent with mr. Distinguish
purpose from chosen means and foreseeable consequences. A rescue purpose alone
neither licenses every means nor makes omission permissible. consent_constraint
explains whether using a person is conditional on consent; do not invent consent.
universalization must examine this particular policy and its strongest objection,
not just say it can or cannot be universalized. action_permissibility and
maxim_assessment are separate: a defective maxim does not alone prohibit an action,
and a benevolent end does not alone justify one. Missing intentions stay unknown;
explain the conditional reading rather than attributing it to the scenario actor.
In review, address the challenged maxim's specific inference; revise the relevant
mx/mr/rs/pr fields or preserve the objection. Saying RESOLVED is not a justification.
Keep these fields concise so every option fits the response budget.
Before assigning categorical priority, classify four independent premises. dt says
whether the governing claim is a PERFECT_NEGATIVE, PERFECT_POSITIVE, IMPERFECT,
RIGHT_CORRELATIVE, SPECIAL_OBLIGATION, or UNRESOLVED duty. hr distinguishes DOING_HARM,
ALLOWING_HARM, PREVENTING_HARM, WITHHOLDING_BENEFIT, MIXED, and cases where the
distinction is not applicable. hr must follow the admitted action graph, not
world-DIRECTNESS as such. DOING_HARM requires an actual settled welfare-adverse
effect on the protected party that the action's DIRECT intervention produces:
either that effect is itself DIRECT, or a CAUSES/ACCELERATES path runs from a
DIRECT effect of the action to it. A NEUTRAL device or infrastructure
intervention still counts as that DIRECT source. ENABLES without CAUSES does
not: leaving a process uninterrupted is ALLOWING_HARM or WITHHOLDING_BENEFIT.
Unsettled POSSIBLE/UNKNOWN outcomes and FOREGONE duals cannot license doing.
Do not relabel an omission as doing in order to keep a perfect-negative prohibition.
so and sob state whether a special obligation is
established and by what role, undertaking, relationship, or prior act; urgency alone
does not create one. mr distinguishes INTENDED_AS_MEANS from FORESEEN_SIDE_EFFECT and
NO_INSTRUMENTALIZATION. INTENDED_AS_MEANS requires the protected party's burden to
be an intermediate cause of another beneficial consequence of the same action.
Sibling outcomes of one intervention (the act causes the burden and, separately,
the end) are FORESEEN_SIDE_EFFECT, not means. A foreseen burden is not automatically
use merely as a means.
If any premise needed for priority remains unresolved, use res=CONTESTED, v=CONFLICTED,
gv=UNRESOLVED, and pb=UNRESOLVED rather than applying categorical priority.
Never reduce an explicitly stated autonomy or liberty burden to cn prose alone:
represent it through ck=AUTONOMY or RIGHT and crel=VIOLATES or CONFLICTS. g separates
action/scenario grounding from framework interpretation; rs gives the shortest justification. A REQUIRED
verdict normally SATISFIES, PERMISSIBLE is CONSISTENT, PROHIBITED VIOLATES, and
CONFLICTED leaves priority unresolved. gv says whether the PRIMARY or COMPETING
norm governs the separate overall verdict; pb gives the deontological priority basis.
Consistency check before returning dp: dt classifies the PRIMARY norm. When
gv=PRIMARY, pb=PERFECT_DUTY needs dt=PERFECT_NEGATIVE or PERFECT_POSITIVE. When
gv=COMPETING, the primary dt does not classify that competing norm; if its duty
classification is not represented, leave priority unresolved instead of borrowing dt.
An IMPERFECT rescue duty does not become a
perfect duty because one option is more effective. Keep its classification and
supply a separately justified priority basis, or leave priority UNRESOLVED with
v=CONFLICTED, gv=UNRESOLVED, res=CONTESTED. Explain any revised classification;
it remains an attributed interpretation rather than an established correction.
pr states the non-consequential priority rule. The overall verdict is derived from
the governing norm, not necessarily the primary row: an action may be PROHIBITED
while CONSISTENT with its primary norm when a competing violated duty governs.
The short human-facing reason w must foreground that same deontological ground.
Do not use a group's relative advantage or disadvantage as the reason by itself;
identify the duty, right, universal law, autonomy claim, or respect-for-persons
requirement governing how that group may be treated.
The remaining dp fields encode the FORM of adjudication, never a memorized terrain.
ps and cps classify the protected standing of the two current-run claims. ki identifies
whether the action uses coercion; coa and cop name its current-scenario actor and
coerced party, or NONE when ki=NONE. pj gives the public justification. rec tests
whether the restriction is reciprocal, nec whether this coercive route is necessary,
and auth records JUSTIFIED, UNJUSTIFIED, CONTESTED, UNKNOWN, or NOT_APPLICABLE.
ki is restriction of a living agent's external freedom. A prior will about bodily
remains after death is not ki=INTERPERSONAL: death removes the living agent whose
freedom would be restricted, and Kant did not establish a perfect duty of
antecedent autonomy. That case is a possible analogical extension of respect for
persons and the prohibition on using humanity merely as a means. If the body's
burden is an intermediate cause of another party's rescue, classify mr=
INTENDED_AS_MEANS. Name the extra-textual step dv=KANTIAN_ANALOGICAL and keep
res=CONTESTED; analogical derivation cannot make res=RESOLVED or settle dv=CONSENT.
Direct Kantian wrongs such as false promising or falsification may still use
UNIVERSAL_LAW or PERFECT_DUTY. dv names the Kantian derivation actually used; res
says whether the conflict is
RESOLVED, CONTESTED, or UNKNOWN. A priority slogan such as "justice outweighs liberty"
is not a derivation. For state or institutional coercion, explain why the restriction
can coexist with each person's external freedom under a universal public rule. When
that cannot be established, use auth=CONTESTED or UNKNOWN, dv=UNRESOLVED, res=CONTESTED,
v=CONFLICTED, gv=UNRESOLVED, and keep the judgment provisional. Generate coa, cop,
p, cp, n, cn, and pj anew from the current action graph and scenario. Never copy a
party, institution, right, or coercion justification from another problem. Rawlsian
terms such as fair equality of opportunity require an explicit translation into
universal law, respect for persons, equal juridical status, or reciprocal external
freedom; do not relabel them as perfect duties without that derivation.

For OPEN_DELIBERATION return wp=NOT_APPLICABLE, we=NONE, fa="NONE", fr=true,
and bd=NONE. In later workspace cycles, wp states whether you accept the anonymous
workspace proposition; we says whether it changed factual reasoning, normative
reasoning, both, or neither. fa briefly states how YOUR assigned framework applies
the proposition, fr says whether your prior framework constraint remains operative,
and bd reports NONE/LOW/MEDIUM/HIGH dependence of this recommendation on the current
broadcast. Do not infer authorship and do not reward or punish agreement.
c must be one of: DUTY, PUBLIC_RULE, RIGHTS, UNCERTAINTY.
Choose u only from: CHECK_FEASIBILITY, CLARIFY_SCENARIO, DECISION_BOUNDARY, NONE, NORMATIVE_ADJUDICATION, RESOLVE_NORMATIVE_TENSION, VERIFY_FACTS.
The listed actions are exhaustive for this deliberation. Do not propose recording
testimony first, delaying, combining actions, obtaining different treatment, or any
other third option unless it is already one of the action IDs. Evaluate the dilemma
as stated even when a real-world alternative would be preferable.
Return e=STATED_FACTS when factual claims come directly from the scenario;
e=FRAMEWORK_ONLY when the difference is purely a moral principle; or
e=UNSTATED_FACTS when your preference relies on a prediction not stated in the
scenario. For UNSTATED_FACTS, x must name that prediction and u must preserve
uncertainty. Otherwise x="NONE". Never treat possible repeat offenses, institutional
effects, hidden alternatives, or affected populations as facts unless stated.
Before recommending, search the ethical landscape. In l, give the strongest
framework-specific case for EVERY action, including the action you reject. da names
the ethical axis that actually separates them. t names the comparison or tie-breaking
rule you attempted. tf="NONE" if it resolves the comparison; otherwise state exactly
why the rule fails. These fields are mandatory even when you have a clear preference.
dr is your explicit decision rule: "prefer [action] when [comparative condition]."
It must say what makes the recommended action outrank its strongest rival.
ft is the smallest change in an empirical fact, probability, magnitude, duration,
or feasibility that would reverse the ranking; use NONE when facts cannot reverse it.
nt is the smallest change in moral priority, right, duty, or acceptable principle
that would reverse it; use NONE when normative priority cannot reverse it. Never put
a moral priority in ft or an empirical prediction in nt. At least one threshold
should expose how the recommendation could change unless reversal truly requires a
different scenario.
gu is a proposed semantic-graph update. When ft contains a measurable threshold,
encode it directly instead of asking middleware to recover it from prose. For
utilitarian responses, default gu to NONE; only emit a single-clause
operation=BOUNDARY when the threshold cleanly crosses away from the recommended
action. Never emit AND/OR for utilitarian graph updates. Set from_action=r, and
to_action to the action preferred when the predicate holds. Each clause names the
affected action ID, metric, ADVERSE or BENEFICIAL valence, LT/LE/GT/GE comparator,
numeric threshold, unit (PERCENT, COUNT, DAY, USD, etc.), and exact source_text.
The predicate must cross away from from_action: increasing an ADVERSE property of
from_action or decreasing its BENEFICIAL property may support a switch; increasing
an ADVERSE property of to_action cannot support switching to it. If you cannot encode
that direction confidently, return operation=NONE rather than a decorative boundary.
Use operation=NONE with NONE action IDs and [] clauses when no measurable factual
boundary exists. Do not encode moral priorities or vague axes as numeric clauses.
ev reports expected value only when the scenario itself supplies enough quantities
to compute comparable values for EVERY action. Use one shared unit and one shared
direction (BENEFIT or HARM); grounded=true only for a calculation whose source
effects are stated facts.
The EV calculation is specialist-authored: for each grounded row, set method to
DIRECT_COUNT, EXPECTED_COUNT, or UTILITY_INDEX; cite only that action's admitted
effect IDs in source_effect_ids; write the calculation in calculation; and list
every extra premise in assumptions (use [] when none). DIRECT_COUNT and
EXPECTED_COUNT must reproduce the submitted value from the cited quantities and
stated probabilities. UTILITY_INDEX preserves an explicit normative weighting but
is not treated as verified world arithmetic. Otherwise return value=0, unit=NONE,
a shared direction, grounded=false, method=NOT_COMPUTED, source_effect_ids=[],
calculation="not computed", and assumptions=[] for all actions.
Each case must describe what happens IF ITS OWN ACTION is chosen. Name the affected
person or value from that action, and for harmful actions explicitly acknowledge the
harm before giving the countervailing reason. Do not place the benefit of sparing a
person under the action that kills or harms that same person.
Begin each case with that action's own mechanism or direct consequence, using a clear
anchor from its action description; put comparisons with the rival action afterward.
For symmetric trade-offs, it is valid for both cases to mention the same shared value,
but each must first say how its own action affects that value. If u is not NONE, or
the judgment remains conditional or underdetermined, tf cannot be NONE. A coin flip
or randomizer is a procedural resolution, not evidence that one listed action is
substantively better; say this limitation in tf.
The prior contribution profile is diagnostic: repeated shallow searches require a
more discriminating axis now, but history never overrides the present scenario.
If it records causal_mapping_errors, explicitly verify that each case describes its
own action rather than the opposite action. If it records unresolved_tiebreaker_errors,
name the unresolved limitation in tf. Repeated consensus is not evidence: a grounded,
framework-relevant minority position receives access credit, while unsupported
contrarianism, random switching, and unexplained reversals do not.
Keep your previous recommendation unless the broadcast supplies a framework-relevant
reason to reconsider it. If you change, j must name that new reason; agreement with
the currently favored action is not itself a reason. Also use j when changing your
preference strength by more than 0.10: name the new framework-relevant information that
justifies the change. Repetition or popularity is not new information. Otherwise j="NONE".
When Workspace contingency is not NONE, directly analyze that failure condition.
Score the original fallback actions for that branch and make w state the preferred
fallback; do not merely repeat your general position on the original scenario.
During CONSENSUS_AUDIT or PROBLEM_STATE_AUDIT, the workspace is auditing the supplied
grounded issue; it is not imposing that issue as YOUR framework's decision rule.
Return ap=RELEVANT when it directly matters within your framework, IRRELEVANT when
your primary rule does not depend on it, TRANSLATED when your framework recognizes
the terrain through a different relation, CONTESTED when the proposed comparison is
normatively disputed, REVERSAL_RELEVANT when resolving it could reverse your ranking,
or UNRESOLVED. Return ax with a framework-specific explanation of at least three
words. If d is CONDITIONAL or UNDERDETERMINED, u must not be NONE: use
VERIFY_FACTS, DECISION_BOUNDARY, or NORMATIVE_ADJUDICATION so the ranking stays
inspectable. DECISION_BOUNDARY is for already-derived threshold/counterfactual
rules (e.g. if duration < 1 month, preference reverses); VERIFY_FACTS is for
missing world-state unknowns; NORMATIVE_ADJUDICATION is for unresolved
framework priority/scope/duty conflicts.
Legacy d/a/v fields
are optional. Do not contort your framework merely to restate the audited variable.
If you translate it, name the relation your framework actually evaluates.
Use the typed audit variable in the broadcast: av.entity names the decision-critical
entity, av.relation names the unresolved relation, av.possible_values lists the
typed values under dispute, and av.focus_action identifies the action under review.
When av.target_framework names YOUR framework, directly test the inferential bridge
named by the question and revise or qualify the relevant framework classification.
When it names another framework, assess relevance from your own framework without
adopting that framework's factual premise or increasing its epistemic status.
av.generated_by identifies who authored the challenge; av.about_specialist identifies
whose reasoning triggered it. Do not describe a workspace-generated challenge as if
the audited specialist raised it. av.challenge_kind and av.trigger_fields identify
the reasoning operation under review, not new evidence or an asserted conclusion.
If av.required_response is present, use its allowed_effects to report ie. If the
relation is THIRD_PARTY_STATUS, state whether the entity is EXTERNAL_THIRD_PARTY,
AUTHORIZED_INTERNAL_AGENT, or UNKNOWN; if AUTHORIZATION_SCOPE, state whether the
permission covers this disclosure; otherwise answer the named relation without
inventing a new ambiguity. Do not return a rewritten av object: the admitted
ProblemState item remains authoritative. Your ap/ax response may reject or translate
its normative relevance without changing its entity, provenance, or question.
During PROBLEM_REFORMULATION, treat every supplied number as a hypothetical
switch-point probe, not as a newly discovered scenario fact. Score the original
actions at that boundary and use w and j to identify the framework-specific value
tension that remains. Do not clear prior factual uncertainty or invent a third action.
During PROPOSAL_REVIEW, the proposal in ProblemState.proposals is a hypothesis under
review, not a live action. Score only the supplied scenario actions. Do not recommend
or assign an action score to the proposal; identify proposal concerns only within your
own framework reasoning until a later typed promotion step exists. Return pr with the
exact proposal_id, framework_status, framework_reason, predicted_consequences,
feasibility_concerns, and required_conditions. Each predicted consequence must name a
dimension, subject, direction, grounding_status, and provenance. Use PROPOSAL_TEXT with
P0-style proposal provenance, SCENARIO_INHERITED with C-style clause provenance, or
HYPOTHETICAL; never present a prediction as a scenario-established fact.
Also return bp=A0, A1, or SPLIT; dx=the decisive numeric dimension or categorical
axis from the broadcast; and sv=the smallest directional change that would switch
your recommendation. bp must match r unless bp=SPLIT. Generic references to harm,
fairness, or uncertainty are insufficient unless tied to a named supplied axis.
During REVERSAL_AUDIT, another specialist has challenged the leading rule. Return
rr=ACCEPT if that condition really reverses your ranking, REVISE if a nearby but
different condition does, or REJECT only if it changes the scenario or conflicts
with your framework. Explain in rj. For REVISE, put the smallest corrected condition
in rv; otherwise rv="NONE". Do not reject merely because the challenge favors the
opposing action.
During CONTINGENCY_REVIEW, reason only inside the stated hypothetical branch. Assume
the failure condition is TRUE and the admitted synthesis is unavailable. Choose one
of the two typed fallback actions: set cr to its action ID and set r=cr. In cj, state
why that fallback follows under YOUR framework given the failure. A baseline argument
that does not apply the failure condition is not an answer. Do not select or repair
the failed synthesis, and do not invent a third fallback.
During VISIBILITY_AUDIT, evaluate the explicit visibility proposition in the workspace.
Return vp=ACCEPT when it is grounded and materially relevant, QUALIFY when its direction
is plausible but magnitude or action relevance remains uncertain, or REJECT when the
scenario does not support it. Return vh=UPWARD, DOWNWARD, or UNCHANGED for your estimate
of the affected action's harm, and vj explaining why. Then rescore every action. You may
keep the same recommendation, but say whether it survives the revised harm estimate;
agreement with the proposition does not require changing actions.
Return vm=GROUNDED_BOUNDED only when the scenario states a bound on the missing
magnitude. Otherwise vm=UNKNOWN: revise direction without inventing a plausible ceiling,
range, population size, or claim that the hidden harm cannot approach a threshold.
[/INST]
```
