# First real-case game session

Prepared from the native-replay-validated five-perspective blanket report.
No API calls, player moves, ratings or control-weight updates have occurred.
The world is identical to the original report. The controller has fresh additive
control weights; simulation disclosure/protection values were not imported.

The game and learning/provenance suite ran 43 tests successfully; a final
10-test game subset also passed after adding elapsed-time recording.
The non-simulation backup ran 210 application tests (four skipped) and
40 native reasoning tests in the appropriate Python environments.

Play interactively:

```bash
.runtime/venv/bin/python -B run_deliberation_game.py play \
  --session diagnostics/deliberation_game_blanket_20261028
```

Each reasoning move consumes one of the two available additional rounds and
one provider request. Human ratings alone update the allocation weights.
No claim of improved game performance is made before real play and evaluation.
