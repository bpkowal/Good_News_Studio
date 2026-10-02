# Files for the latest parsing game (Z5)

The latest exporter is `parsing_game_Z10.py`; see [PARSING_GAME_Z10.md](PARSING_GAME_Z10.md).
It preserves complete alternative role bundles, explicit modal-negation ambiguity,
separate reconstruction identities and Z6/Z7 condition checks. Construction-level
coverage is measured by `eval_ellipsis_coverage.py` using frozen semantic fixtures.
The Z5 file inventory below remains applicable to its unchanged baseline.

Run a sentence by calling `export_candidate_graph` in `parsing_game_Z5.py`. The installed spaCy model `en_core_web_sm` is also required. It is not a file in this repository. The Hoosier corpus under `resources/thec_eng/` is not read during a parse.

## Code

`parsing_game_Z5.py`  
Baseline exporter. It builds the Z4 graph, then adds an unresolved negative copy when a sentence has "but" or "and," then "not," then a nominal. Each sentence is judged on its own. The original sentence is not rewritten. Proposal generation and stripping export now share the exact original Doc, including whitespace tokens.

`parsing_game_Z4.py`  
Builds the candidate graph Z5 starts from. That graph has predications, roles, modals, conditionals, coordination, clause boundaries, quantities, and scenario choices. It does not add a stripping copy.

`parsing_game_S.py`  
Tokenizes the sentence with spaCy. It proposes predicate frames, complements, attempt links, and "to" attachments. Z4 reads those frames. It does not choose the final graph.

`parsing_references.py`  
Adds unresolved same-referent candidates for pronouns, definites, and proper names. Mentions stay unmerged. The scores and lemmas come from the reference policy file.

`candidate_validation.py`  
Checks a finished package against the candidate contract: ids, roles, scope, questions, and choice sets. Z4 imports it. Export does not select a graph or commit world state.

`ellipsis_proposer.py`  
Offers a copy after the detector accepts a sentence. A verb-phrase remnant copies a verb phrase. A stripping remnant copies the left verb onto the spoken nominal, with negative polarity.

`ellipsis_detector.py`  
Scores whether a sentence still has a gap. A sentence with no cue stays absent. It does not name the missing words. Stripping runs only after a positive decision.

`ellipsis_corpus.py`  
Loads Hoosier ellipsis pairs and assigns train, development, and test groups for fitting. Its surface cleaner is used for corpus preparation and comparison, not to produce runtime proposal indices. The corpus files are not opened during export.

## Data read during a parse

`resources/ellipsis_detector_v1.json`  
Saved detector weights, bias, and threshold. The proposer loads this file before it decides whether a gap is present.

`resources/ellipsis_proposer_v1.json`  
Saved weights and margin for choosing a verb-phrase copy. The stripping copy does not use these weights. The file is still read when a sentence is proposed.

`resources/T_reference_policy.json`  
Pronouns, human and nonhuman lemmas, and the score cutoffs used to propose a same-referent link.

`resources/to_attachment_valency.json`  
Lemmas and weights that classify "to" as a destination, an infinitive, or another reading.

`resources/verbnet_attempt_adapter.json`  
The try and attempt lemmas and the frame test. A matching infinitive can become an attempt link. "intend" stays a complement.

`resources/reduced_passive_Q.json`  
Heads that separate a passive "by" phrase from a temporal one, and that mark a reduced passive. Read when a clause has no object or contains "by."
