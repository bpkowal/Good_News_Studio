# Contractualist specialist: comparison before roster replacement

Status: neutral adapter implemented; a first four-framework cooling probe is
saved in `diagnostics/contractualist_pilot_live_20261024_network`. The full
four-case comparison using burden-preserving comparisons is recorded separately
in `diagnostics/contractualist_comparison_v03_live_20261025`; consult that run's
RESULT.md and content review for completion and findings.
The active fifth specialist remains `rawlsian`; saved runs keep their original
framework identities. This document records the proposed roster change and its
separate exploratory adapter.

The current increment makes [critical unanswered questions](CONTRACTUALIST_CRITICAL_QUESTIONS.md)
a visible contribution in fresh generations and consequence-first follow-ups.
The agent can clarify the decision without recommending an action; it may still
recommend one when its comparison is defended. Existing results remain frozen.

## Purpose

Evaluate a Scanlonian contractualist specialist as the proposed fifth member,
alongside Utilitarian, Deontological, Care, and Virtue specialists. Retain Rawls
as an optional institutional-analysis perspective. Do not relabel Rawlsian
outputs, scores, or ledger records as contractualist ones.

The contractualist asks which general principles could be justified to each
affected individual in light of the alternatives. It does not maximize a minimum
welfare number, require actual unanimous approval, or treat any expressed
objection as an automatic veto. Reasonable rejection and comparisons of claims
remain contestable moral judgments, not validator-certified facts.

Prioritarian and sufficientarian analyses may be explicitly named tools, with
their assumptions and any sufficiency threshold stated. They neither replace
individual complaints nor supply automatic contractualist rankings.

## Proposed reasoning task

1. Formulate general candidate principles permitting or requiring the available
   actions, including their relevant conditions and exceptions. More than one
   principle may permit an action, and a principle may permit multiple actions.
   Avoid names or arbitrary case details that build the desired winner into a rule.
2. Identify affected persons and relevant positions, including nonrecipients,
   involuntary contributors, and promisees. Group counts cannot replace the
   reasons of an individual member. Generic positions may represent unnamed
   people; never fabricate personal histories or preferences for them.
3. State each material complaint against each principle. Separate factual grounds,
   evaluative warrants, and unresolved applicability conditions. A plausible
   complaint does not establish that a principle is reasonably rejectable.
4. Compare complaints against alternative principles, explaining relevance and
   comparative force. Do not sum complaints into a welfare score. If numbers are
   morally relevant, explain the disputed argument for considering them. Do not
   silently replace comparison with a fixed strongest-complaint algorithm.
5. Give the strongest defensible conclusion: a justified preference, several
   defensible principles, or residual disagreement. Identify the specific premise
   on which the judgment depends and a relevant reversal condition.

For risk, explicitly distinguish ex ante prospects from ex post burdens and name
the adopted interpretation. For promises, distinguish an unfulfilled assurance
from the further conclusion that its nonfulfillment is wrongful betrayal. Actual
reliance, distress, deceit, and the agent's exclusive intentions require evidence.

## Argument architecture

Use the existing immutable statements, source citations, support routes,
conditional assumptions, attributed assessments, and explicit revisions wherever
their meanings fit. A complaint is an argument offered on behalf of an affected
person, not a source-world atom or a report of what that person actually said.

The inspectable chain should be:

`source grounds + normative warrant -> individual complaint -> comparison with
alternative complaints -> judgment about a principle -> action assessment`

A factual objection attacks the grounds. An applicability objection challenges
whether the warrant covers this situation. A countervailing reason can rebut the
conclusion without denying the grounds. These are proposed Toulmin/Pollock-style
uses of the architecture, not already implemented attack semantics.

Preserve the person, principle, alternative, grounds, warrant, comparison, and
remaining defeaters as linked argument records. The later adapter must not force
these into Rawlsian position scores or deontological duty classifications.
Structural validity never proves reasonable rejection or normative correctness.

## Frozen exploratory cases

Inputs: [contractualist_pilot_cases.json](fixtures/contractualist_pilot_cases.json).
Each exact source text has its own SHA-256 digest. These are frozen text fixtures,
not admitted Parliament worlds. `design_notes`, pairing, and review targets are
evaluator metadata and must be withheld from generation prompts.

| Case | Contrast |
| --- | --- |
| Cooling plans | Three feasible actions: more beneficiaries with displacement; fewer beneficiaries without that burden; protection of the most heat-exposed person with an uncompensated burden on another |
| Cooling with consent | Same actions and outcomes, but the contributor to the third plan freely agrees; tests sensitivity to the grounds of that complaint |
| Blanket with promise | 80% versus 60% rescue prospects, zero for the excluded person, prior assurance to the latter |
| Blanket without promise | Same rescue prospects, explicit absence of an assurance; tests whether the assurance changes the argument rather than merely its vocabulary |

The cooling case stipulates beneficiary counts, not total welfare or a uniquely
correct fairness ranking. Calling the third plan protection of the worst-off in
every morally relevant respect would prejudge its burden on the shopkeeper.
No expected recommendation is frozen for any case. Agreement among specialists
is compatible with distinct reasoning; disagreement earns no bonus.

## Matched procedure

The isolated, nonvoting text-case adapter accepts all three cooling options
without passing them through a two-action native world runner.
This tests argument contribution; it cannot establish end-to-end world admission.

Compare Utilitarian, Deontological, Care, and Contractualist reasoning on the same
four inputs: 16 initial calls, one per framework/case, no repairs or peer exposure.
Freeze the final prompts, source hashes, framework instructions, provider model,
reasoning setting, token ceiling, raw responses, and transport errors before
comparison. Use the same common argument envelope and budget for every framework;
framework-specific instructions and source excerpts are explicit experimental
differences, not claimed identical controls. Do not silently retry weak arguments.

`argument_frameworks.py` separates the common task from framework-local
instructions. `argument_construction.construct()` uses that separation for both
generation and critique, while retaining legacy free-text framework commitments.
`argument_support.py` adds descriptive premises, interpretive assumptions,
principles, personal reasons and comparisons without remapping them to maxims.
Existing deontological domain types remain available.

`contractualist_pilot.py` uses the same output schema for all four specialists,
allowing aggregate reasons to have no person/principle binding when appropriate
to another framework. Missing contractualist bindings remain visible issues,
not grounds for deleting its argument. Support ancestry, references, complete
option coverage and terminal assessments are checked independently of ethical
defensibility. The adapter does not compute normative acceptance or defeat.

Prepare all 16 requests with zero API calls:

```bash
.runtime/venv/bin/python run_contractualist_pilot.py \
  --output diagnostics/contractualist_pilot_prepared
```

Run one fresh case through all four frameworks (four calls maximum):

```bash
.runtime/venv/bin/python run_contractualist_pilot.py \
  --case cooling_three_options \
  --output diagnostics/contractualist_cooling_live --execute
```

Omit `--case` to execute all four cases in a fresh directory. Defaults are o3,
low reasoning effort and an 8192-token completion ceiling including hidden
reasoning. Every selected cell has one attempt, no SDK retries, repairs or peer
input. Interrupted attempts remain saved and are not automatically rerun. Only
allowlisted source fields enter generation requests; evaluator notes stay out.
The protocol, exact requests and implementation bytes freeze before calls.
Raw outputs, compiled Mermaid argument graphs, and partial/error records remain
inspectable. Reusing an attempted directory is an error, not an implicit retry.

See [research reuse and attribution](CONTRACTUALIST_RESEARCH.md) for the paper's
contribution, our adaptations, and formal semantics not claimed as implemented.

The next increment is specified in
[CONTRACTUALIST_COMPARISON_INCREMENT.md](CONTRACTUALIST_COMPARISON_INCREMENT.md).
It adds option-specific burden positions, explicitly scoped pair comparisons,
and a distinction between provisional preference and unique defensibility. Its
annotations are proposed model interpretations, not a deterministic moral ranking.
In particular, retained risk can have different probabilities under two actions;
PRESENT in both does not mean equal severity. Comparison findings appear beside
binding issues and never silently change an agent's recommendation.

No collective vote or topography follows this exploratory comparison. Virtue
remains in the proposed production roster but is outside this first four-way
discrimination study. Rawls needs a separate institutional case, not pressure to
answer the same rescue question under a new name.

## Content review

Review saved outputs in randomized order without model/framework labels where
practical; theoretical language can still reveal identity, so do not claim perfect
blinding. Reviewers should record uncertainty and disagreements. An assistant-only
review is explicitly labeled as such. No authoritative automated moral oracle is
assumed.

Score each dimension separately on a 0–2 scale (missing/wrong; partial; explicit
and adequately defended for this case):

- Fidelity: affected people, outcomes, probabilities, consent, and assurance match
  the source; unsupported factual inferences are identified.
- Principle quality: generality, relevant conditions, and connection to available
  alternatives; no tailor-made winner or forced one-action/one-principle mapping.
- Complaint quality: a particular person's grounds and a defended normative
  warrant, rather than a relabeled aggregate or minimum welfare score.
- Comparative justification: explains why an objection survives or is outweighed
  by objections to alternatives; does not merely name a serious burden.
- Defeater handling: addresses a substantive objection or retains the exact
  unresolved premise; confidence and self-declared resolution are not evidence.
- Contrast sensitivity: explains what the consent/assurance change alters in
  reasons, including a defensible unchanged action preference.

Separately assess marginal contribution across the four saved arguments in each
case: absent, restatement, or distinct defensible contribution. A contribution is
a supported warrant, objection, response, or comparison that changes the argument
map, not a new word, longer explanation, or different selected action. Record an
exact passage and the closest argument in another framework; shared reasons are
allowed. Do not add these ratings into a single moral-quality score.

This is one sample per cell and four designed cases. It cannot establish reliable
superiority, coverage of aggregation problems, or general framework independence.
If useful contributions appear, pre-register a second small held-out study,
including many-small-claims versus one-severe-claim and an institutional case,
before deciding to replace the fifth production specialist.

## Integration map after that evidence

- Compact generation: `logic_puzzles_frameworks.FRAMEWORKS`, quote packs, and
  native `CompactLocalSpecialist` prompts/proposals.
- Native assessment and review: framework-specific ledgers, engine handling,
  conflict projection, and targeted review; preserve source-world immutability.
- Presentation: report/export allowlists, argument preservation, viewer labels
  and colors, and any framework assumptions in topography or Decisions prompts.
- Full RelEnt/RAG: specialist registration and retrieval corpus identity; compact
  support alone does not deliver this second pipeline.
- Provenance: new framework/version identity and resource hashes, unchanged old
  runs, updated runtime lock/profile only when actual dependencies change.

These are migration work items, not implemented support. No active roster,
world-state schema, native validator, source corpus, or report is changed here.

## Reading basis and limits

- [Scanlon, What We Owe to Each Other (1998)](https://doi.org/10.2307/j.ctv134vmrn).
- [Stanford Encyclopedia: Contractualism](https://plato.stanford.edu/entries/contractualism/),
  especially reasonable rejection, reasons beyond well-being, aggregation, risk,
  and overlap with other nonconsequentialist views. These are contested issues.
- [Marcos-Vidal, Abbo, and Belpaeme (2026)](https://arxiv.org/abs/2608.01937)
  propose a contractualist argumentation framework using ASPIC+ and value-based
  filtering. Its worked example supports investigating this connection; it is
  not evidence that our agent works or that formal semantics settle moral priority.
