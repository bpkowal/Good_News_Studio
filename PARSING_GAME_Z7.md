# Z7: explicit condition content across selection paths

The subsequent [Z8 increment](PARSING_GAME_Z8.md) separates ellipsis proposition
identities before applying these condition-content rules.

Use `parsing_game_Z7.export_candidate_graph(text, package_id=None)`. Z7 wraps Z6;
Z6 and earlier exporters remain unchanged. The shared validator accepts schema
0.3 as well as legacy 0.1/0.2 packages.

For “If Maria decides to pull the lever, the trolley will stop,” a selection of
the stopping predication, prediction, subject role, or conditional link retains:

- The decide and pull predications and their complement link.
- Maria as the decision subject and the lever as the pulling object.
- The candidate controller of pulling, including its unresolved reference question.

The stopping reading therefore stays provisional until the controller question
is resolved. Merely selecting a controller candidate does not resolve the question.
A simple “If Maria pulls the lever…” condition can complete without a controller
question. A first-use definite reference question still does not block its role.

## Schema 0.3 addition

The envelope requires `condition_contents`, including an empty list for texts with
no represented condition. Each record has exactly:

```json
{
  "id": "condition_content_0",
  "condition_proposition_id": "p2",
  "required_candidate_ids": ["c0", "c1", "c5", "c6", "c8", "c9"],
  "question_ids": ["q0"],
  "evidence_ids": ["e0"]
}
```

IDs are illustrative. They resolve within the original immutable package. Record
IDs share the package namespace. Exactly one record must exist for each condition
referenced by a conditional scope context or `CONDITIONAL_ON`. Required candidates
must include a predication anchor for that condition. Duplicate records, dangling
references, missing condition records, and use of this field under older schemas
are rejected. Selections must use the same schema version as their package.

Required content includes represented, compatible participants, their explicit
quantities, recursive complement/attempt structure, and unique modal readings.
Questions affecting that content are recorded explicitly. Alternatives are not
silently conjoined: ambiguous roles/links/predications and represented embedded
predicates without retained links produce blocking interpretation gaps. Those gap
questions have no selectable resolution in this increment. A later complete reading
protocol is needed to resolve them safely.

## Selection behavior

Z7 adds safe ordinary dependencies to conditional consumers for convenience.
Independently, the validator checks the explicit content record on **every**
candidate carrying the condition, including the explicit conditional link.
Missing required content, an unanswered recorded question, or provisional required
content keeps that consumer provisional. Provisional status propagates to dependent
candidates and through nested conditions until stable. Different conditions retain
separate records; an open controller question in one does not block another.

An ordinary omitted `requires` dependency still makes a selection invalid. If a
package exposes only the condition-content record without a convenience dependency,
the missing content makes the selection provisional instead. Neither mechanism
promotes a hypothetical event to actual occurrence. The validator does not mutate
inputs, and no world-state commitment is authorized.

## Validation and limits

155 tests passed: the 145-test baseline and 10 new tests covering direct consequence
paths, controller/modal resolution, missing-role bypasses, quantities, nested links,
negative scope, independent conditions, ambiguity, and malformed records.

```sh
.venv/bin/python -m unittest test_parsing_game_Z7 -v
```

This establishes completeness against the package's represented content, not
complete linguistic understanding. Missing unrecognized predicates or roles can
still escape the exporter. It does not add intention semantics, repair a wrong
parse, verify identity, or prove semantic support. The caller must retain the
original package; the validator cannot detect wholesale rewriting of its trusted
content records. Re-export old packages through Z7 for the new guarantee.
