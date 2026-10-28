# A game for learning how Parliament should investigate and deliberate

Branch: `Learning_Deliberation_Game`, based on non-simulation application
checkpoint `298a8eb`. The baseline remains on `Parliament_Frontend`; exact native
source is backed up separately at `backup/parliament-native-20261028`.
Commit `b5dc414` preserves the earlier simulations and their failure archives.

This direction learns how to use reasoning resources in particular contexts.
It does not assign an objective moral truth score to a theory. A helpful
Contractualist clarification can matter without producing a final vote; another
investigation round can be valuable or wasteful depending on the question.

## First playable increment

The command-line game starts from an actual admitted Parliament world and its
five recorded framework arguments. The default case is the thermal-blanket
promise case. Native replay checks the exact scenario/world and trace hash.
Preparation makes no API calls:

```bash
.runtime/venv/bin/python -B run_deliberation_game.py start \
  --session diagnostics/deliberation_game_blanket_20261028
```

Play in a terminal; each reasoning move makes one bounded live o3 request:

```bash
.runtime/venv/bin/python -B run_deliberation_game.py play \
  --session diagnostics/deliberation_game_blanket_20261028
```

The session defaults to two additional reasoning rounds; `start --rounds N`
allows one to four. SDK retries are disabled. Moves are:

| Move | What happens |
| --- | --- |
| `investigate` | A selected perspective examines a specific missing premise and asks decision-relevant questions |
| `self_counterfactual` | That perspective tests its own argument against a counterexample, alternative or reversal condition |
| `deliberate` | One perspective addresses a named opposing perspective's actual recorded argument |
| `answer` | The player supplies attributed game information in response to an investigation; native world records stay unchanged |
| `cues` | The player annotates factual, normative, causal, relational and outcome uncertainty/concerns, or leaves them unassessed |
| `weights` | Set requested perspective emphasis and counterfactual attention for subsequent reasoning |
| `stop` | Record a player choice among existing actions, or leave the decision unresolved |
| `quit` | Leave the episode open for later continuation |

The game records the real model reply, source quotations, unresolved questions,
hypotheses, provider-call count, elapsed time and any failed/refused attempt.
Unsupported quotations and unknown action proposals stay flagged, and those
flags travel with any reply supplied to a peer. Exact quotations establish
traceability, not semantic entailment. New arguments, player information and
counterfactuals stay in the game record, outside the native admitted world.
No collective verdict or operative Parliament action is fabricated.

The player rates whether a move was worth its cost, from 0 to 1, with an
identified rater and explanation. This is subjective process feedback. A
response's own `resolved` label, structural admission or agreement with another
agent supplies no reward. Leaving a move unrated makes no update. A duplicate
rating cannot update its move twice. Stopping can also be rated; the system
does not assume that more rounds are always better.

Only the **round-allocation control policy** learns in this increment. It has
four available control actions and additive contextual weights. It reuses the
numeric learning mechanism, not the simulation's disclosure/protection weights:
those action meanings differ and cannot be relabeled. Starting control values
are untrained priors until real player feedback arrives. There is no predefined
payoff table or generated moral score in this game.

Question cues are player annotations in this first version. Each has a separate
"assessed" indicator, so an unassessed cue is not conflated with an assessed
zero. The weights learn relationships between those cues, remaining budget and
the rated usefulness of control moves. This is not yet automatic discovery of
question categories. Unknown cue annotations remain visible.

The policy recommends moves; the player dispatches them. This first local,
single-player experiment does not claim an unbiased comparison between policies
or automatic autonomous improvement. Player choices, ratings and theory
preferences can bias the observed experience. Later matched comparisons must
use independent assessments and preserve frozen controls.

## Preserve, inspect and continue

The session contains `case.json`, the original report and frozen trace,
`game.sqlite3`, per-move provider requests/responses, `game_record.json`, a
readable `GAME_REPORT.md`, and `policy_checkpoint.json`. SQLite retains the
configuration history, attempted moves and before/after feedback updates.
Failed moves consume a reserved round and remain visible; they receive no
automatic reward. Existing run provenance records in-place session revisions,
retaining their input database bytes and source snapshots. A later session
revision can supersede an earlier database hash; inspect the recorded revision.

Inspect without API calls or another learning update:

```bash
.runtime/venv/bin/python -B run_deliberation_game.py show \
  --session diagnostics/deliberation_game_blanket_20261028
```

Resume the learned controller in a new episode without resetting it:

```bash
.runtime/venv/bin/python -B run_deliberation_game.py start \
  --session diagnostics/deliberation_game_next \
  --report path/to/another/DEMO_REPORT.json \
  --resume-policy diagnostics/deliberation_game_blanket_20261028/policy_checkpoint.json
```

An additional supported report must retain a validated frozen trace and its
hash. A new scenario still enters through the existing construction/admission
pipeline first. Game observations do not bypass that boundary.

## Subsequent increments, one learning target at a time

1. **Investigation and stopping.** Collect real episodes and human feedback;
   compare a frozen controller with an updating copy at matched resource cost.
   Measure whether an extra round answered the actual question, preserved
   support, changed the justified conclusion or merely elaborated it.
2. **Perspective emphasis.** Learn context-dependent attention to each
   perspective. Compare the reasons exposed and objections handled, not just
   which theory's recommendation wins. Preserve independent contributions and
   abstentions. The existing game emphasis settings provide the intervention;
   their weights are manual until this experiment is isolated.
3. **Self-counterfactual influence.** Compare omission, limited attention and
   stronger attention to a perspective's own counterfactuals. Zero currently
   excludes those replies from subsequent peer input; positive settings request
   a degree of attention. A requested number is not a measured percentage of
   influence inside an LLM. Counterfactuals remain hypothetical evidence tests.
4. **Inter-agent deliberation.** Test which question contexts benefit from
   challenge/reply rounds, including when they increase contamination or
   conformity. Record both directions explicitly, failures and marginal benefit
   per round; do not reward consensus alone.
5. **Question categorization.** Audit and expand the cue inventory, compare
   player labels with model-proposed classifications, and learn useful feature
   weights on newly held-out question combinations. Keep disagreements about
   categorization visible. Do not disguise a subjective labeling policy as an
   objective map of ethical problem types.

Only after those comparisons should multiple control families adapt together.
That preserves the ability to identify which intervention helped and prevents
an apparent improvement from hiding a changed reward, category system and
reasoning budget at the same time.
