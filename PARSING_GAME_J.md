# Parsing game J

J is a local sentence-to-claim experiment for the Parliament STA layer. It is
importable without starting training, loading spaCy, opening logs, or redirecting
terminal streams. It does not write to a graph.

## Run

Use the project's environment, which needs NumPy, spaCy, and `en_core_web_sm`:

```sh
.venv/bin/python parsing_game_J.py
.venv/bin/python -m unittest test_parsing_game_J -v
```

In a terminal, every normal run first asks for one new sentence. Press Enter to
skip adding one. The latest **five submissions** are retained, oldest first,
and all five are re-evaluated on each run. A sixth submission deletes the oldest
entry and its previous output. Repeated submissions count as separate entries.

The terminal prints each probe's extracted relations, assertion statuses, lexical
evidence, eligibility, and reasons for withholding commitment. These are unlabelled
exploratory probes, not accuracy-scored tests. They never enter CEM training or
select its weights. A parser exception is displayed for that probe while the
remaining probes continue.

```sh
.venv/bin/python parsing_game_J.py --add-sentence "A wave of rainfall was followed by flooding."
.venv/bin/python parsing_game_J.py --no-prompt
```

`--no-prompt` replays saved inputs; `--add-sentence` adds one without prompting.
Noninteractive stdin also skips the prompt. Existing `--sentence` remains a
one-off inspection and does not add that text to the rolling suite.

The suite and its latest detailed results live only in
`diagnostics/parsing_game_J_user_probes.json`, relative to this script. Override
with `--user-suite PATH`. User probes are deliberately excluded from dated logs,
so rotation does not leave their old input/results archived there. Terminal
scrollback or files you explicitly capture are outside this retention mechanism.
A corrupt suite produces an error rather than silently replacing your data.

Use these probes to challenge the syntax/semantics boundary: paraphrases,
temporal-only language, implicit relations, and counterfactuals. A successful
parse is not a correctness judgment; a missing relation is not proof of absence.
Neither syntax nor the fixed lexicon establishes real-world causation, and small
phrasing changes can still change the analysis. This feature adds observability,
not LLM inference, discourse reasoning, or new example-specific semantic rules.

Each experiment saves a `parsing_game_J_<timestamp>.policy.json`, JSONL evidence
log, and text summary under `diagnostics/`. Options include `--seed`,
`--population`, `--generations`, and `--output-dir`. To inspect a sentence with an
existing policy:

```sh
.venv/bin/python parsing_game_J.py --load-policy diagnostics/YOUR_RUN.policy.json --sentence "Disease was not induced by exposure."
```

The Python API accepts raw text; hand-labelled entity fields never enter parsing:

```python
from parsing_game_J import CEMPolicy, parse_sentence, parse_claims, train_policy

policy, history = train_policy(seed=42)
claim = parse_sentence("Disease was not induced by exposure.", policy)
# relation_type: causal
# direction: second_to_first
# source: exposure; target: Disease
# assertion_status: denied
# eligible_for_world_state: False

policy.save("my_policy.json")
restored = CEMPolicy.load("my_policy.json")

claims = parse_claims(
    "Heavy rain causes severe flooding and strong winds trigger power outages.",
    policy,
)
# Heavy rain -> severe flooding
# strong winds -> power outages
```

`parse_claims()` is the API for multiple relations and multiple sentences. Each
claim has its own assertion status, evidence, validation reasons, and eligibility.
`parse_sentence()` preserves the single-claim interface; for multiple relations it
returns a primary diagnostic view with the complete `claims` list and disables
commitment at that top level. The CLI's `--sentence` output is now a claim list.

## Responsibilities and claim contract

1. spaCy supplies tokens, lemmas, POS, dependencies, and argument proposals.
2. An explicit local lexicon proposes relation meanings. Surface forms and
   lemma matches can survive incorrect POS tags. Unknown meanings stay unknown.
3. CEM learns weights over 11 reusable semantic/structural channels, producing
   four meaning/direction actions. Features contain no predicate identities.
4. Separate rules annotate assertion scope: asserted, denied, possible,
   conditional, attributed, questioned, quoted, or unresolved. Multiple statuses
   are retained, so "might not" records possibility and negative polarity.
5. A validation gate checks the proposed action against semantic and directional
   evidence and blocks qualified or unsupported claims from positive causal
   commitment. It exposes disagreement; it does not silently replace the learned
   action with a rule-based answer.

`source` and `target` describe the proposition, even when denied. Only an
unqualified causal claim without validation failures receives
`eligible_for_world_state=True`. This is eligibility for further integration,
not proof that the proposition is true. Parliament still needs its own graph
schema, identity resolution, validation, and treatment of claims versus facts.
The parser performs no graph insertion, including for denied or attributed claims.

`relation_type` distinguishes `causal`, `association`, `explicit_no_relation`, and
`unresolved`. An association does not prove that causation is absent. Unsupported
input, missing arguments, ties, and policy/semantic disagreements abstain.

Each result contains token offsets, the candidate proposals, selected predicate,
argument provenance, feature values, assertion cues, per-action scores and feature
contributions, and validation reasons. Scores and margins are **uncalibrated**;
they are not probabilities. No candidate produces a consistent abstention result
with an empty entity list and `decision=None`.

Arguments include full bounded mention text, head text/index, token and character
offsets, and an entity/event distinction. `source_entity` and `target_entity`
identify the particular mention, so two occurrences of "rain" need not be merged.
`find_entity_spans(doc, text)` locates every case-insensitive exact token-sequence
match, including noun phrases and gerunds. It does not match "rain" inside
"rainfall" or silently select the first repeated mention. Automatic extraction
uses dependency roles and bounded spans; it does not require supplied entity names.

FIRST/SECOND remain text-order labels. Subject/object roles are recovered before
sorting, and semantic/passive evidence determines the direction channel. Thus
"Flooding was caused by rain" and "By rain, flooding was caused" both express
rain causing flooding, with different text-order action labels. Taking min/max
alone would fix a span boundary but would not establish those roles.

## Experiment changes and evaluation

The original example suites are retained. Human labels for negated causal
sentences now preserve direction and annotate denial separately; `legacy_action`
retains their former action. Consequently, old accuracy numbers and 17-feature
weights are not directly comparable or compatible. The old Gym wrapper was
replaced by vectorized CEM over feature arrays computed once per training run.
This extension uses policy schema **2** because role-aware direction changes the
feature contract. Saved schema-1 policies are rejected with a retraining message;
run the experiment again to create a current policy.

Training uses only training labels, with training margin as an accuracy tie-break.
Holdout results never select weights or stop training. Every generation is logged.
Representation collisions are reported separately and do not feed training.

The added lexical holdout has content lemmas absent from training but available in
the independent semantic lexicon. A separate unknown-meaning suite tests abstention,
including a fabricated predicate. An assertion suite tests scoped claims with
held-out content. Regression tests check disjoint lexical content, parser mistakes,
scope, policy persistence, import behavior, and trace reconstruction.

Metrics report direction-action accuracy, argument-pair accuracy, assertion-status
accuracy, and their joint correctness after policy validation. These are small
development/regression suites, not an untouched external benchmark. Some original
suites intentionally reuse training sentences to test particular structures.
CEM is stochastic: a perfect default-seed score does not establish generalization.
Legacy single-label metrics still compare argument heads. The separate
`multi_relation_holdout` compares the entire ordered list of full source/target
spans, assertion statuses, and eligibility flags, catching missing and extra claims.
Neither this suite nor the added multi-clause regression cases trains CEM.

## Boundaries

Supported structures include independent coordinated clauses, semicolons,
multiple sentences, shared subjects, active/passive combinations, simple local
relative antecedents, and bounded nominal/gerund arguments. Shared modal scope
is propagated when appropriate; a new clause subject does not inherit the previous
clause's negation or modal. Ambiguous shared negation blocks commitment.

Coordinated arguments remain groups: "rain and snow" is not automatically expanded
into separate claims that each independently causes flooding. Collective arguments,
disjunction, and unresolved pronouns are flagged. General coreference, quantifier
logic, gapping, noncontiguous arguments, and arbitrary nested propositions remain
unsupported. Scope for conditions, quotes, and some attribution cues remains
conservatively sentence-wide. Full mention extraction is bounded and heuristic,
not a guarantee of recovering every noun phrase modifier or every relation.
Unknown intervening content can be proposed tentatively even when POS is wrong;
this is not proof a relation exists.

Scope rules deliberately block uncertain embeddings and quantified arguments.
They are inspectable heuristics, not a complete language understanding system.
The lexicon is intentionally finite; extending it should add semantic knowledge,
not sentence-specific features. No network semantic service or LLM is invoked.
