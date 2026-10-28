# Bounded argument-support implementation

Implemented October 23, 2026 as a versioned, nonvoting argument path. Parliament's world schema and previously saved worlds/ledgers are unchanged. A narrowly scoped native calibration patch is active through the pinned runtime. This is not an automatic replacement of native framework generation or a new operative deliberation mode.

## What now runs

**Canonical statements and revisions.** `argument_support.py` computes immutable statement IDs, support IDs and hash-linked events. Withdrawal, supersession and qualification preserve old records. A changed proposition gets a new identity. Withdrawing a support route need not withdraw its conclusion if another independently reviewed route survives. Source bindings and explicit assumption roots are compiled by code rather than repeated manually by the model. A citation does not automatically establish semantic support.

**Canonical citations.** Bare IDs, exact existing ID-plus-pointer aliases, and structured `{evidence_id, value_path, quote}` citations resolve to the original source with the raw form preserved. `inference_review.assess_response` now uses that resolver. A nonexistent pointer or fabricated quotation still fails. Matching a source location does not establish the claim's entailment. Archived provider responses are never rewritten.

**Polarity and scope.** Canonical statements keep polarity, modality, hypothesis scope and lifecycle separate. Affirmative practical-reasoning requirements cannot be satisfied by an explicitly denied belief. Scope propagates transitively along each support route, and withdrawing a hypothesis reopens claims scoped to it. An independently supported route can remain unconditional. UNKNOWN is not treated as an asserted scope promotion. Legacy free-text negation is not silently parsed or guessed; old annotations without polarity retain that limitation. Malformed legacy records now produce controlled ValueError failures.

**Proposal, compilation and critique.** `argument_construction.py` requests one or at most two small arguments, compiles them, and requests one isolated critique for each. Critics propose local semantic operations; code handles IDs, revisions and projections. Partial construction keeps valid statements and routes plus explicit construction problems. An invalid critique preserves the prior candidate rather than discarding it. The result and Mermaid report remain nonvoting. Critics' support assessments are attributed model judgments, not independent proof.

**Native scope correction.** `argument_support_scope.patch` separates a coherent candidate instrumental account from actual physical efficacy. A proposed maxim can contain a mistaken belief without an actual means-to-end path. Explicit same-action means/end effect citations still require graph support. Legacy intended-means claims without a coherent candidate account retain the old path requirement. Practical-reasoning consistency, harm checks and other native invariants remain. No schema field or LLM-supplied repair category authorizes a blanket bypass.

The patch is the twelfth ordered integration patch in `pipeline.lock.json`; fresh reconstruction matches the locked tree. Both compact and full RelEnt preflights are ready. The native patch can affect future calibration of qualifying maxims; it is not merely a passive report change.

## Run commands

The new two-case text-source experiment can be run with:

```bash
.runtime/venv/bin/python run_argument_construction.py \
  --output diagnostics/my_argument_construction_run
```

This makes at most six provider requests, using the existing OpenAI environment. It is an experimental argument path; its text-source fixtures are not newly admitted worlds.

To consume an already admitted world instead, use the same native frozen-world validation boundary as framework generation:

```bash
.runtime/venv/bin/python run_argument_construction.py \
  --trace diagnostics/deontic_choice_composition_20261016/live_final/frozen_world_trace.json \
  --output diagnostics/my_frozen_world_argument_run
```

That creates at most two interpretations and three requests. It validates and preserves the frozen source; it does not submit its argument overlay as a native operative ledger. Outputs include `design.json`, raw calls, candidate attempts, explicit missing constructions, canonical argument records, support assessments and `candidate_arguments.md` Mermaid diagrams. Design records include code and runtime-lock hashes.

## Evidence from this increment

See the [results and argument graphs](diagnostics/argument_support_20261023/RESULT.md). Two new author-written cases received five total provider requests within a six-request cap. One privacy proposal initially failed construction. Its unchanged saved proposal was then passed to one critique after partial compilation was implemented. That critique exposed a missing support-withdrawal operation; exact saved fields were replayed offline after adding it. Three proposals now compile as staged arguments. This corrects representation/application mechanics, not the model's moral reasoning.

The volunteer means account retained source-stated purposeful employment despite failed efficacy. But another volunteer argument introduced an unstated training status, and the privacy argument overstated necessity. The critic did not fully correct those semantic issues. Its handling of duty priority often made an assumption explicit rather than independently justifying that assumption. Neither a compiled candidate nor an attributed SUPPORTED assessment certifies the argument.

Regression results: 73 argument/protocol/native tests plus 71 application tests, of which two were skipped: **142 passed, two skipped**. The native suite includes frozen-world framework/targeted-review replays and genuine means/end path controls. The new live argument cases were not admitted through a full native Parliament deliberation, and no live RelEnt/RAG run was performed. Runtime readiness is reported separately from live behavioral evidence.

## Remaining boundary

Operation-specific construction and local critique are working prototypes, not a complete semantic verifier. Labels can still be wrong, source entailment can be overstated, and normative priority remains contestable. The old adversarial collection improved from four to seven expected behaviors out of ten because transitive scope, UNKNOWN handling and malformed input handling changed. Its remaining misses involve semantic relabeling, unannotated negation and qualifier overreach; that adversarial count is not an estimated real-world success rate.

CALCULATION is registered as an operation but requires a numeric verifier before its route can count as supported. Full integration with existing native numerical ledgers, independent/blinded content evaluation, and routing these argument proposals into operative native recommendations remain future work. No claim is made that the new path has demonstrated generally better ethical conclusions.

## Next increment: source applicability before priority critique

`premise_audit.py` now provides an opt-in source review before local critique. It inspects the conclusion's support ancestry, including conditions inside maxims and normative statements. The review packet excludes framework instructions, support explanations, earlier judgments and the preferred-conclusion designation. The claim wording remains visible, so this is reduced context, not a blinded evaluation.

The reviewer separates source-supported components, contradictions, information not established, normative assumptions, and hypothetical factual preconditions. Normative assumptions need argument rather than quotations pretending to establish their truth. An unknown applicability condition remains unknown. Exact source quotations bind by their declared location, or by a unique exact-quote location if the model invents an alias; the raw alias and resolution method remain saved. Paraphrased claim components are explicitly labeled. None of this proves semantic entailment.

Run with `--premise-audit` to add one review per candidate, at most five provider requests per case including generation. The default path retains its three-request maximum. To inspect just one saved proposal:

```bash
.runtime/venv/bin/python run_argument_construction.py \
  --case ineffective_volunteer \
  --saved-generation diagnostics/argument_construction_live_20261023/ineffective_volunteer/calls.json \
  --candidate-index 1 --premise-audit \
  --output diagnostics/my_premise_audit_run
```

That requires at most two requests. `--saved-audit path/to/calls.json` reuses the exact source/argument-bound audit and requests only the revision. SDK retries and provider retries are disabled in this experimental runner. The audited mode preserves each valid local operation transactionally if a later operation fails, records rejected operations, and keeps dependent claims requiring reassessment. Supersession/qualification now honors an attached support operation instead of silently ignoring it. Old statements remain inspectable.

The four-request live probe corrected specific overclaims but did not establish a dependable improvement in moral reasoning. Privacy retained its missing factual condition and withdrew the categorical recommendation. The volunteer correction removed “untrained,” but its attempted conditional conclusion referenced an undeclared hypothesis; that edit remains rejected. Duty priority and complete canonical support remain unresolved. See the [content review and exact replay graphs](diagnostics/premise_audit_20261023/RESULT.md). No native world or operative ledger was modified.
