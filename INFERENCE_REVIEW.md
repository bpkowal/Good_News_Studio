# Bounded inference review

Implemented October 20, 2026. This is a reusable review protocol with a deontology repair-card adapter. It is not a new world schema or an automatic detector of philosophical error.

## Systematic target

Structurally consistent fields can rest on an unsupported bridge. Another full argument often elaborates that bridge instead of testing it. The review unit is therefore one claim and its dependencies, rather than a scenario-specific prohibition or an entire delegate response.

`inference_review.py` accepts a reviewer-identified exact passage from an immutable saved reply, a defect category, the missing premise, evidence records and affected fields. The same functions cover unsupported factual bounds, missing explanatory bridges, scope overreach, normative priority and treating absent evidence as evidence of absence. The reviewer objection itself is attributed and can be wrong.

The agent receives that packet and its prior reply. It must:

1. Separate source-supported premises, interpretive assumptions, normative assumptions and unknowns.
2. Describe the bridge to its conclusion.
3. Defend, qualify, withdraw or leave the inference unresolved.
4. State the strongest surviving conclusion, remaining questions and permitted local changes.

An explicit interpretive or normative assumption can support a conditional argument; the protocol does not require every normative claim to become a world fact, nor demand direct access to an actor's mind before considering a candidate maxim. The target is undisclosed or unjustified premises, not the existence of assumptions.

## What happens automatically

Source-reply hashes and exact quotations bind the review to a saved argument. The deontology adapter also binds its evidence and edit scope to the original trusted repair card. The model cannot expand that scope. The request and raw response are retained; one provider call is allowed and SDK retries are disabled.

Bookkeeping checks require an explicit qualification, named withdrawal or remaining question, as appropriate, and resolve citation IDs. They do not prove that the cited evidence supports the claim. A model's DEFENDED or QUALIFIED label leaves the effective issue OPEN_PENDING_CONTENT_REVIEW.

The deontology adapter restages the permitted local changes from the immutable original ledger, records the previous probe as a parent artifact, reopens dependent closure and performs whole-ledger native validation. The staged result cannot vote. No existing saved run, world, action score or other-action record is rewritten. A failed response remains inspectable rather than triggering another generation automatically.

## First evidence

The [laboratory follow-up](diagnostics/deontology_inference_review_live_20261020/RESULT.md) used one fresh o3 response. It changed unsupported negative instrumental-belief/use claims to UNRESOLVED, with the correction retained by native validation. It still inadequately justified the side-effect interpretation. That is a partial correction, not a verified moral conclusion or a measured general improvement.

Six generic protocol tests exercise different empirical, causal and duty-priority claims, stale inputs, unsupported citations and self-reported closure. Existing local-repair/native tests cover scope isolation, both repair orders and dependency reopening. These are interface and regression tests, not independent evaluations of argument quality.

## Boundary and next evaluation

### Typed boundary cards and precise support locations

The October 23 increment adds `inference_boundaries.py`. A challenge may optionally carry `inference_map`: typed claims, interpretation modes, precise evidence bindings and explicit bridge dependencies. Old challenge IDs and saved histories remain valid when this optional field is absent.

Claims distinguish physical contribution, purpose, belief, purposive employment, intended use, a candidate maxim, source rules and normative claims. Interpretation modes distinguish source-stated content, attributed inference, conditional hypotheses and unknowns. These labels describe argument annotations, not additions to the admitted world. A support record binds an evidence ID plus a JSON pointer relative to its exact value and a matching quotation. Provenance excerpts can therefore support a purpose even when the effect predicate alone does not.

The causal/practical-reasoning card exposes whether the proposed inference identifies an end, instrumental belief and purposive employment. Unknown premises cannot silently provide that support. A conditional interpretation can remain useful; actual causal efficacy is not required to express a stipulated mistaken instrumental belief. Companion generic rules expose intended-use-to-prohibition and source-rule-to-priority bridges without deciding their normative validity.

Typed challenges request `reasoning_revision` alongside the existing local response. Existing claim identities must survive; newly introduced claims use new IDs. The revision is checked against the original evidence locations and retained in the ordinary response history, using the existing scope controls and two-round budget. The offline replay also carries a reasoning revision into a next challenge when available. No full-response schema, native validator or world schema changed.

Findings are investigative questions, not automatic rejection of the whole candidate. Quote and type checks do not prove entailment, consent, actual psychology or moral priority. A valid citation can still be misused to support “solely”; that limitation is explicitly included in the fixtures. A model can also put a false statement into an apparently appropriate type, so semantic review remains necessary.

Run the offline contrast export with:

```bash
.runtime/venv/bin/python replay_inference_boundaries.py \
  --output diagnostics/my_boundary_card_replay
```

Inspect [the eight contrast cards](diagnostics/inference_boundaries_20261023/RESULT.md) and the [saved laboratory annotation](diagnostics/inference_boundaries_20261023/lab_boundary_card.json). The laboratory annotation preserves its exhausted review budget and authorizes no further follow-up. These are authored fixtures and offline annotations, not new model results. Native admission of a mistaken-means interpretation remains a known limitation pending a separately bounded native correction.

### Persistent dependencies and bounded revisions

`inference_history.py` now stores a separate argument-review graph and hash-linked response/reviewer events. Claim identities persist through reviewed revisions; newly introduced claims receive new identities. Every response retains its native validation result separately from its content assessment. A source hash, world hash and exact prior-reply hash bind the sequence to its inputs. Hashes detect accidental alteration; they are not signatures or proof of reviewer competence.

Reviewers record corrected points, newly introduced unsupported claims, the remaining gap and claim-level support judgments. These are attributed judgments about arguments, never additions to the world. A withdrawn or unknown premise marks dependent supported conclusions REQUIRES_REASSESSMENT transitively. Conditional premises propagate conditional status. Independent supported claims retain their status. This projection does not select an action or infer a new moral verdict.

The request context now includes that history and asks only about the remaining gap. There are at most two response rounds per issue. A reviewed correction or defense stops the sequence, as does reviewed lack of progress. A response awaiting content review cannot silently trigger another request. A model's own disposition cannot stop or resolve the issue. Resuming the probe requires its history; changing the world, resuming from a stale reply or expanding the original evidence/edit scope fails before a provider request.

`replay_inference_history.py` builds a new history from a saved probe plus an explicit content-review file without API calls. Existing artifacts are never overwritten. For example:

```bash
.runtime/venv/bin/python replay_inference_history.py \
  --probe diagnostics/deontology_inference_review_live_20261020/live_probe.json \
  --review fixtures/inference_history_lab_content_review_20261020.json \
  --output diagnostics/my_inference_history_replay
```

The [saved laboratory history](diagnostics/inference_history_20261020/RESULT.md) records one used response and one remaining. Its next challenge concerns conditional interpretation; it preserves the prior correction rather than requesting a new whole argument. `run_deontology_repair_probe.py` accepts `--history`, `--prior-probe` and `--inference-challenge` to consume that remaining round. No second live round was run during this engineering increment.

Ten history tests cover persistence, transitive invalidation, conditional support, mistaken-reviewer defense, both stop conditions, latest-reply/world/scope checks, event tampering, new claim identity and the saved-case replay. The mistaken-reviewer case is a synthetic state-transition test, not evidence that an agent reliably detects mistaken criticism.

Issue discovery and semantic content assessment still require a reviewer. No keyword rule for this laboratory sentence was added. This mechanism can consume future native or independent reviewer findings, but it does not pretend to discover arbitrary unsupported inferences yet.

The next evaluation should reuse this exact protocol on several already-saved arguments with distinct defect types. Compare withdrawal or defended qualification, surviving unsupported bridges and uncertainty propagation separately from native admission. Include defensible claims where the reviewer is wrong, so success does not mean automatic agreement with criticism. Do not add a new scenario-specific validator for each example.
