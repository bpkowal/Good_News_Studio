# Typed trolley slots and compact representation

Implemented Cloze version 0.12. Omission has 15 blanks versus allocation's 19.
The added blanks identify each branch's patient and explicit count, and the operated
instrument. Actor/process/entity fills yield minimal copied noun phrases; actions
and harmful consequences have separate boundaries. Raw completions and hypotheses
remain inspectable. Older saved answers acquire role bindings through parser
recovery, including supporting candidate IDs. Unknown counts remain unknown.

The graph builder consumes typed patients/counts/instrument without changing the
Parliament world schema. Coordinated harms and benefits remain distinct. Relative
mechanisms stay available through original source and physical-scene parsing.

The viewer's compact projection bundles matching action/direct-effect records and
moves ungoverned scene assertions into inspectable context. It preserves the frozen
world and full graph. Browser checks showed 17 compact trolley boxes versus 22 full
boxes, and selecting A0 exposed its A0/E1 records. Medicine bundles its matching
pairs as well. Console inspection found no errors or warnings.

Validation: 64 tests ran; 60 passed and four skipped. See [regression.log](regression.log).
These cover typed spans, wrong-branch rejection, legacy recovery, projection,
existing Cloze fixtures, and viewer export. `git diff --check` passes.

An additional native-dependent trolley run could not execute admission: the
`/tmp/parliament-smoke-614af0c/global_workspace` directory contains only cache files,
so the runner fails importing `global_workspace.action_identity`. This is not a
successful native regression run. No provider calls were made, and no existing
admitted worlds were replaced. Live generation quality has not been measured by
this saved-answer replay.

Inspect [typed roles](typed_role_summary.json), [replayed candidates](saved_answer_replay.json),
and [candidate graphs](candidate_graphs.md). These new candidates have not received
fresh native admission. Reader projection is implemented; replacing native event
expansion with a lean semantic-event compiler remains a later increment.
