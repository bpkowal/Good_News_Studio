# Research memory

This directory records what the project has actually tested, what conclusions
those tests support, and the conditions under which rejected ideas should be
reconsidered.

The records deliberately separate:

- `experiments/`: comparisons, metrics, limitations, and artifacts;
- `decisions/`: current engineering decisions derived from experiments;
- `baselines/`: named configurations used as controls;
- `benchmarks/`: immutable benchmark identity and provenance;
- `project_state/CURRENT.json`: the current evidence-backed project summary.

Experiment outcomes are `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`,
`INCONCLUSIVE`, `REGRESSION`, or `NO_DEMONSTRATED_VALUE`. Every experiment must
state `intentionally_not_compared`; every decision must state `reconsider_if`.

Validate the ledger:

```bash
.venv/bin/python -m research.ledger validate
```

Search both experiments and decisions:

```bash
.venv/bin/python -m research.ledger search spacy
.venv/bin/python -m research.ledger search evidence repair
```

Live stochastic runs must say so in `limitations`. A result should not be
described as causal unless the relevant upstream artifacts were frozen.

