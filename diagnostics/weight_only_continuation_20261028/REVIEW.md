# What continued weight updates changed

The resumed weight-only policy improved on the reused validation monitoring
cases compared with a copy of the exact same starting model whose weights were
frozen. No retrieval was used. Each pair received the same additional training
schedule and simulator random draws; only the updating copy learned from its
chosen actions' observed outcomes.

| Monitoring result | Frozen weights | Updated weights |
| --- | --- | --- |
| Validation toy-optimal choices | 70% | 90% |
| Validation expected functional success | 0.7485 | 0.7935 |
| Validation expected regret | 0.0765 | 0.0315 |
| Test toy-optimal choices | 100% | 100% |
| Test expected functional success | 0.9500 | 0.9500 |

The validation expected-score change by seed was -0.0075, 0, +0.1575, +0.0750
and 0. Two seeds improved, one worsened slightly, and two were unchanged. The
mean improvement is therefore not evidence of a universal gain from additional
updates. Constant-rate learning from stochastic outcomes can regress.

This comparison improves on the first experiment's weak heuristic control:
the control here is the same previously trained feature model. The only intended
intervention is enabling weight updates. Both policies remain frozen during
monitoring, and the final updated checkpoint is saved without selecting an
intermediate checkpoint based on monitoring results.

All five pairs started at revision 512. Frozen copies stayed there, while
updating copies reached revision 1024. The checkpoints can be restored again
without resetting weights. Individual changes and contextual value contributions
are inspectable in [the weight CSV](weight_changes.csv) and
[the value explanations](weight_explanations.json).

The cases and artificial reward model are unchanged from the earlier report.
Their results had already been inspected, so they are monitoring sets, not a
fresh test of generalization. There are still only two validation and two test
feature combinations with repeated wording variants. These results do not show
that Parliament's moral reasoning improved or that learned values can yet
select an operative Parliament action.

Twenty-seven learner, continuation and provenance tests passed. The
[verification](verification.json) confirms matched starts and schedules,
zero retrieval records, zero monitoring updates and resumable final weights.
The earlier experiment's manifest also still verifies. No API calls, agent
edits, world changes or live policy changes were made.

The next useful increment is a small shadow connection to supported cues and
candidate actions from admitted Parliament cases, with independent feedback.
Keep the exact frozen-weight control and the weight-only mechanism so that any
benefit can be attributed to learning instead of changing several components
at once.
