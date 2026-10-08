# Resource recognition and duty-generation increment

Resource stock is now recognized from an admitted RESOURCE party's own original
evidence. It does not depend on a transfer, choice, or exclusive-allocation
complement. The quantity card cites the resource identity and its source clause;
it does not infer scarcity, control, exclusivity or benefit. Tests cover inventory
with no effects, missing evidence, wrong endpoint kind, reported/modal stock and
quantity mismatch. The world-state schema remains unchanged.

The saved medicine duty failure was not a missing reference. All three cycles
reported perfect-duty priority without a perfect-duty classification, for both
actions. The agent described an imperfect rescue duty while asserting perfect-duty
priority. See [saved diagnosis](duty_diagnosis.json).

The native deontological generation prompt now explicitly explains this
cross-field requirement and the unresolved-priority representation. It instructs
the model to retain its actual duty classification rather than changing it to
pass validation. The rejection rule is unchanged. A synthetic native probe checks
that the contradiction is rejected and that a complete IMPERFECT/CONFLICTED ledger
with unresolved priority is representable. It also checks the actual generated
prompt. This does not establish that a fresh model response will follow the
guidance, or that any particular moral verdict should pass.

The [saved medicine replay](replay.json) preserves the prior probability and
source-rule improvements: 95% → 0.95, 5% → 0.05, and two death-rule claims rebind
to the stipulated source rule. Five recognition records remain. The
[evidence graph](recognition_graph.md) includes the independent resource proof.
No API calls were made, no judgment was recomputed, and the saved inputs are
unchanged.

**Regression: 125 tests run; 124 passed; one pre-existing trolley failure.**
All 14 recognition tests pass, including the native duty probe. The trolley
omission fixture still fails admission for AV3/AV6's unestablished assumptions
and POLARITY_INVERTED transformations. No new failures appeared. See the
[regression log](regression.log). The additive integration patch was applied to
the saved pre-increment files in a temporary directory and reproduced all five
installed native files byte-for-byte.

Remaining work: test fresh duty generation against this guidance; broaden stock
language beyond the current explicit has/have construction; handle supported
derivations separately from exact source recognition. Welfare magnitude, duration
and comparability remain unresolved. A named-person count is not automatically
the welfare magnitude of that person's survival, and a quantity fragment does
not establish an entire strengthened stock/benefit claim.
