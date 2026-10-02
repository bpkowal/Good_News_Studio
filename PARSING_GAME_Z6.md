# Z6: condition complement completeness

For explicit participant/quantity/controller content and independent selection
validation, use the subsequent [Z7 increment](PARSING_GAME_Z7.md).

Use `parsing_game_Z6.export_candidate_graph(text, package_id=None)` for this increment.
Z6 wraps Z5; earlier exporters and the shared validator are unchanged.

For “If Maria decides to pull the lever, the trolley will stop,” every candidate
carrying the conditional scope now requires the represented decide→pull complement.
This includes direct selection of the stopping predication, its roles, and its modal,
as well as the explicit conditional link. Nested unambiguous complement/attempt
links are included recursively. Their existing dependencies retain child anchors;
selecting them does not assert that those events occurred.

Competing event readings, event-versus-nominal readings, unsupported link meanings,
or dependency back references generate blocking scope questions. Partial selections
remain valid but provisional. These questions have no candidate resolutions in this
increment: selecting one fragment does not establish a complete interpretation.
This deliberately abstains rather than forcing incompatible dependencies or cycles.

The change preserves scope, nodes, evidence, and candidate arguments. Simple
conditions and sentences without conditional content expansion match Z5 apart from
producer metadata. It uses existing `requires` and question fields; no schema
extension is introduced.

The guarantee covers represented complement structure, not complete semantic
understanding. It does not infer missing links, resolve controllers, require every
participant role, or add intention semantics. Generic coverage questions remain.
Historical Z5 packages must be re-exported through Z6 to receive these dependencies.

```sh
.venv/bin/python -m unittest test_parsing_game_Z6 -v
```

Tests cover all consequence selection paths, missing-dependency rejection, nested
complements, ambiguity abstention, unchanged baseline graphs, scope/evidence
preservation, and cycle avoidance.
