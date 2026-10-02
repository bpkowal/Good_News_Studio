# Z9: modal negation in ellipsis

`parsing_game_Z9.export_candidate_graph` retains Z8's separate reconstructed
propositions and adds a blocking scope question for modal-bearing reconstructions:
`NOT MODAL(P)` versus `MODAL(NOT P)`. The reconstructed predication, participants,
and modal candidates have unresolved polarity; the spoken antecedent is unchanged.
Selecting an unresolved candidate cannot itself resolve this operator question.
The question is created before condition-content analysis, so conditional consumers
also retain incompleteness. Attribution and local modal alternatives survive.

This implements explicit abstention, not two executable operator-order graphs.
Schema 0.4 and the existing validator suffice. Five tests cover direct selections,
false resolutions, can/may/must/will, unchanged antecedents, attribution, recipients,
multiple reconstructions and whitespace. Z10 incorporates this behavior and adds
complete alternative participant bundles; see [PARSING_GAME_Z10.md](PARSING_GAME_Z10.md).
