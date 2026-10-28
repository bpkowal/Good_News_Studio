# Contextual Moral Generalization v0.1

Offline simulated terminal-action weighting. No API calls, agent edits, live decisions or world-state mutations.

48 synthetic scenarios in 12 cue-combination groups: 32 training, 8 validation, 8 test.
Each held-out split has two unseen context groups and four surface variants per group.
The variants are not independent contexts. Features and outcome contingencies are hand specified.

## Held-out results

| Split / arm | Expected functional score | Regret | Toy-optimal action rate | Observed success | Process proxy |
| --- | --- | --- | --- | --- | --- |
| validation / fixed_heuristic | 0.675 | 0.150 | 50.0% | 0.625 | 0.500 |
| validation / learned_values | 0.748 | 0.077 | 70.0% | 0.675 | 0.800 |
| validation / learned_values_memory | 0.748 | 0.077 | 70.0% | 0.675 | 0.800 |
| test / fixed_heuristic | 0.035 | 0.915 | 0.0% | 0.050 | 1.000 |
| test / learned_values | 0.950 | 0.000 | 100.0% | 1.000 | 1.000 |
| test / learned_values_memory | 0.950 | 0.000 | 100.0% | 1.000 | 1.000 |

Expected score and regret use the stipulated simulator, not a moral truth metric.
Observed success is sampled: a good expected choice may fail, and a poor one may succeed.
Only observed outcome feedback updates weights. Decision and process scores stay separate.
5 seeds estimate sensitivity to training exposure and luck; they do not create new independent ethical cases.
The fixed control is a coercion-based heuristic, not a fixed LLM or Parliament baseline.

## Counterexamples to a familiar cue

| Test context / arm | Expected score | Selected actions across seeds and surface variants |
| --- | --- | --- |
| harmful-c0-u1 / fixed_heuristic | 0.020 | disclose: 20 |
| protective-c1-u1 / fixed_heuristic | 0.050 | protect: 20 |
| harmful-c0-u1 / learned_values | 0.950 | protect: 20 |
| protective-c1-u1 / learned_values | 0.950 | disclose: 20 |
| harmful-c0-u1 / learned_values_memory | 0.950 | protect: 20 |
| protective-c1-u1 / learned_values_memory | 0.950 | disclose: 20 |

## Controls and interpretation

- Split, configuration, evaluator source and input hashes were frozen before the first trial.
- All arms received the same scenario schedule and common simulator random draws for each seed.
- The policy sees only the shared feature schema and the three available actions; no case IDs, labels or oracle alternatives.
- Retrieval uses only observed training outcomes from the same arm and seed. Success/failure examples remain inspectable.
- Held-out evaluation is greedy, makes no updates, and preserves the exact frozen model checkpoint.
- Retrieval and feature interactions were specified in advance; neither validation nor test tuned this run.
- The experiment has one disclosure domain and only two test contexts. Results do not establish improved Parliament reasoning, moral correctness, or broad transfer.
- Missing cues, new actions, learned agent recruitment and live Parliament integration are later increments.

[Interactive record](report.html) · [results](results.json) · [all held-out selections](evaluation.csv) · [SQLite experiences](experience.sqlite3) · [frozen design](design.json)
