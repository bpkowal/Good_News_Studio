# Optional deeper review of revised arguments

Implemented opt-in follow-ups that can return to the same framework's revised
argument. The current ten-worker trolley revision triggers an offer asking why
a foreseen side effect falls under the governing prohibition against intentional
killing, while distinguishing intention of the intervention, harm as an end,
harm as a means, and foresight. This is a question, not a finding that the agent's
view is necessarily inconsistent.

The demo defaults to `--review-depth ask` when ordinary review is requested.
Interactive terminals ask before additional calls; noninteractive execution
preserves the offer and spends nothing extra. `--review-depth deep` is explicit
advance opt-in. The cap is two additional rounds, one adapter call per round.

Each additional round consumes the latest admitted revision and prior challenge
reply. Native replay verifies that the retained response reproduces the committed
duty state before a model call. Earlier arguments and the admitted world remain
unchanged. Rejection, unchanged structured justification, or the round limit stops
the invocation. A model's `RESOLVED` label is not independent verification of the
answer. Open advisory questions do not themselves veto an admitted recommendation.

## Verification

- Nine focused tests pass, including real native replay and two sequential
  follow-ups with controlled replies.
- Seventy-one related review, partial-review, presentation, demo, provenance and
  runtime tests pass (80 together with the focused tests).
- Thirteen maxim, means-end and negation regressions pass separately.
- The second controlled follow-up receives the first admitted revision.
- An unchanged reply stops after one call while leaving the question visible.
- A rejected reply retains the prior candidate. A mismatched prior response is
  detected before a follow-up call. Source files and the shared world are unchanged.
- Report output preserves the offer and the chosen depth in execution provenance.

No additional live provider calls were made. The actual trolley offer was
generated in read-only mode; its manifest records `SAVED_REPLAY_NO_API` and zero
follow-up adapter calls. The native tests use controlled backends, not new live
arguments or a normative quality evaluation.

This first increment supports the deontological application gap above and retained
deontological premise errors. It is not a general detector of all serious ethical
issues. Full RelEnt deliberation and the read-only browser viewer are unchanged.

- [Offer for the current trolley argument](offer/reasoning_followup.md)
- [User-facing report with the offer](report/DEMO_REPORT.md)
- [Read-only execution manifest](offer/run_manifest.json)
- [Usage and limits](../../DEEPER_REASONING_REVIEW.md)

To opt in later, use a new output folder:

```bash
.runtime/venv/bin/python logic_puzzles_followup.py --trace diagnostics/causal_means_reasoning_20261015/live_review/targeted_review.json --output-dir diagnostics/tram_deeper_review_live --execute
```
