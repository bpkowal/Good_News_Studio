# Study 1 Protocol: Discriminative Sensitivity to Normative Structure

## Protocol status

This document is a preregistration-oriented protocol draft. Items marked
**LOCK AFTER PILOT** must be fixed before confirmatory data collection. The pilot
will be used only to validate stimuli, estimate variance, verify compute matching,
and set sample size. Pilot scenario families and random seeds will not appear in
the confirmatory set.

## 1. Research question

Does structured multi-agent ethical deliberation respond more strongly to
morally relevant changes, while responding less strongly to normatively irrelevant
presentation changes, than compute-matched single-model reasoning?

The study does not attempt to measure ethical correctness or select a correct
normative theory. It tests a narrower necessary property of competent moral
reasoning: discrimination between changes in normative structure and changes in
surface representation.

## 2. Primary claim

Structured multi-agent ethical deliberation will show:

1. greater **signal sensitivity** to validated moral perturbations;
2. lower **nuisance sensitivity** to validated meaning-preserving perturbations;
3. a larger separation between signal and nuisance sensitivity than the control
   systems, under matched model and compute constraints.

These outcomes must be reported separately. A ratio may be reported as a compact
summary, but it is not sufficient by itself because instability can inflate both
its numerator and denominator.

An unusable final answer is not an unusable experimental observation. If a system
is stable under a perturbation while remaining unresolved, that stability belongs
in the primary response-state analysis. Final-answer quality is a separate
diagnostic outcome and must not be used to discard, repair, or reinterpret a
response before similarity is calculated.

## 3. Confirmatory hypotheses

### H1: Nuisance invariance

The Ethical Parliament will have lower mean response distance between a canonical
scenario and its nuisance variants than the compute-matched solo condition.

### H2: Moral sensitivity

The Ethical Parliament will have greater mean response distance between a
canonical scenario and its moral variants than the compute-matched solo condition.

### H3: Discrimination advantage

The Parliament's within-system moral-minus-nuisance distance will exceed that of
the compute-matched solo condition:

\[
G_c = \mathbb{E}[D(R(S),R(S_m)) \mid c]
      - \mathbb{E}[D(R(S),R(S_n)) \mid c]
\]

\[
H3: G_{Parliament} > G_{ComputeMatchedSolo}
\]

where \(S_m\) is a moral variant, \(S_n\) is a nuisance variant, and \(c\)
indexes system condition.

### H4: Framework differentiation

Within the Parliament, targeted moral perturbations will change the theoretically
relevant framework's typed assessment more than non-targeted frameworks' typed
assessments, without requiring all frameworks to change their final recommendation.

The preregistered mappings are:

| Perturbation | Primary targeted framework | Secondary plausible target |
|---|---|---|
| Consequence magnitude or probability | Utilitarian | Rawlsian when burdens concentrate |
| Consent or intentional-versus-foreseen harm | Deontological | Care |
| Stranger versus dependent relation | Care | Deontological |
| Individual choice versus public allocation rule | Rawlsian | Deontological |
| One-off act versus repeated character-forming practice | Virtue | Care |

H4 tests functional differentiation, not the truth of any framework.

## 4. Experimental conditions

All conditions use the same base model family and pinned model version.

1. **Solo:** one ordinary ethical-reasoning response.
2. **Compute-matched solo:** one model reasons iteratively across multiple calls,
   with no framework-role decomposition, using the Parliament's total inference
   budget.
3. **Grounded/structured solo:** one model receives the same admitted world model,
   proposition ledger, action identities, and structured response schema as the
   Parliament, but performs no framework-specialist decomposition or inter-agent
   deliberation.
4. **Ethical Parliament:** the full grounded five-framework ensemble and its
   synthesis/admission pipeline.

The primary comparison is Parliament versus compute-matched solo. Solo and
grounded/structured solo are planned controls that separate raw compute effects
from grounding/structure effects.

### 4.1 Compute matching

Compute matching will be based on both:

- maximum aggregate generated tokens per scenario run; and
- maximum number of model calls.

The same model version, decoding parameters, context limit, retrieval snapshot,
and tool permissions will be used wherever the architecture permits. The
compute-matched solo condition will receive the same maximum aggregate generation
budget as Parliament. Unused budget will not be replaced after a condition
terminates normally.

Actual prompt tokens, generated tokens, calls, latency, retries, and failures will
be logged and reported. Analyses using actual generated tokens as a covariate will
be sensitivity analyses, not replacements for the randomized condition contrast.

**LOCK AFTER PILOT:** exact call and token ceilings, timeout policy, decoding
parameters, and retry policy.

## 5. Stimulus design

### 5.1 Scenario families

Each family begins with one canonical two-or-more-action ethical scenario. Families
will span multiple normative domains so the result is not driven by one dilemma
template. Planned domains include emergency allocation, consent, professional
duty, dependency and care, public policy, distributive justice, environmental
risk, and repeated institutional practice.

Each canonical scenario will have matched variants that preserve all unmanipulated
properties.

### 5.2 Moral perturbations

Each moral variant changes exactly one preregistered property, such as:

- consent versus non-consent;
- lower versus higher consequence magnitude;
- low versus high probability;
- stranger versus dependent family member;
- intentional versus foreseen harm;
- equal versus highly unequal distribution;
- private individual choice versus public institutional rule;
- reversible versus irreversible harm;
- one-off act versus repeated practice.

A moral perturbation is not required to flip the selected action. It must change a
normatively relevant input while preserving the other scenario facts.

### 5.3 Nuisance perturbations

Nuisance variants preserve the annotated normative structure while changing one
or more presentation properties:

- swap action order and action labels;
- reorder sentences;
- rename people or places;
- use a validated paraphrase;
- switch active and passive voice;
- change an occupation annotated as irrelevant;
- shorten or lengthen prose without adding a proposition;
- present consequences before intentions rather than after them.

Action-order tests will be scored using canonical action identity, never the
surface labels `A`/`B` or list position.

### 5.4 Stimulus validation

Before confirmatory collection, at least two annotators blinded to system outputs
will independently assess each variant.

For nuisance variants they will assess:

1. proposition preservation;
2. action identity preservation;
3. preservation of quantities, probabilities, modality, negation, temporal order,
   causal relations, and available options;
4. absence of a morally relevant change.

For moral variants they will assess:

1. exactly one intended morally relevant change;
2. preservation of all non-targeted facts and action identities;
3. the targeted framework mapping, without judging the correct action.

Disagreements will be adjudicated before any system is evaluated. Agreement and
exclusion counts will be reported. Variants failing validation will be replaced
before confirmatory runs, not removed after outcomes are observed.

## 6. Randomization and repeated runs

- Scenario-family order will be randomized independently for each condition.
- Condition execution order will be blocked and randomized to reduce temporal API
  and infrastructure effects.
- Canonical and variant members of a family will not be placed consecutively when
  a stateful interface could carry information between runs.
- Every run will start without conversational memory from prior variants.
- The same preregistered seed set will be used across conditions when supported.
- Variant generation seeds will be separate from inference seeds.

The scenario family—not an individual stochastic run—is the primary independent
sampling unit.

**LOCK AFTER PILOT:** number of scenario families, variants per family, and repeated
runs per condition. The confirmatory sample size will be selected by simulation
using pilot variance and clustering estimates, with a declared target power and
smallest effect of interest.

## 7. Response representation

Every condition must emit or be deterministically projected into the same response
vector:

\[
R=(C, K, F, X, P)
\]

where:

- \(C\): canonical action choice or explicitly unresolved status;
- \(K\): calibrated confidence and epistemic status;
- \(F\): framework activation/assessment vector;
- \(X\): typed constraints and reversal conditions;
- \(P\): cited proposition identities supporting the reasons.

Free-form prose will be retained for audit but will not be the sole source of a
primary measure. Claims rejected by grounding, proposition admission, framework
vote integrity, or final-output integrity will not count as active reasons. The
system's unresolved, abstaining, contradicted, and malformed statuses remain
explicit response-state values; they are not silently converted into missing
data.

For solo systems, the matched structured adapter requests this same state schema
in the solo call without asking the model to judge response quality or imitate the
Parliament. The adapter will be frozen before confirmatory analysis and validated
on a held-out set. A legacy plain-text response may be retained for audit, but it
is not treated as a fully structured observation. Parliament's native typed fields
will be mapped directly rather than reparsed from its prose.

## 8. Response-distance measure

The primary response distance is a preregistered weighted sum of component
distances:

\[
D(R_i,R_j)=\sum_k w_k d_k(R_i,R_j), \qquad \sum_k w_k=1
\]

Planned components are:

| Component | Distance |
|---|---|
| Canonical choice | 0 if identical; 1 if different; preregistered intermediate cost for resolved versus unresolved |
| Confidence/status | absolute normalized difference plus categorical epistemic-status mismatch |
| Framework assessments | normalized L1 distance over aligned framework/action fields |
| Constraints | Jaccard distance over canonical typed constraint IDs |
| Supporting reasons | Jaccard distance over admitted proposition IDs |

Equal component weights are the default. **LOCK AFTER PILOT:** component weights,
the resolved/unresolved cost, missing-component policy, and normalization constants.
These may be calibrated without looking at condition differences. Results for each
component will be reported separately, along with equal-weight and leave-one-
component-out sensitivity analyses.

Embedding distance over free-form prose is exploratory only. If reported, the
embedding model and version will be frozen before confirmatory data collection.

The final-answer quality rubric is secondary and diagnostic. It may describe
whether a response is useful to a person, but it cannot determine whether two
responses are similar for the primary perturbation analysis.

## 9. Outcomes and estimands

For each canonical-to-variant pair:

\[
D_{moral}=D(R(S),R(S_m))
\]

\[
D_{nuisance}=D(R(S),R(S_n))
\]

Primary estimands are:

1. mean nuisance distance by condition;
2. mean moral distance by condition;
3. the moral-minus-nuisance gap \(G_c\);
4. the Parliament-minus-compute-matched-solo difference in \(G_c\).

The descriptive discrimination ratio is:

\[
Q_c = \frac{\mathbb{E}[D_{moral}\mid c]}
{\mathbb{E}[D_{nuisance}\mid c]+\epsilon}
\]

The value of \(\epsilon\) will be fixed before confirmatory analysis and a range
of values will be shown in sensitivity analysis. No conclusion will rely on the
ratio without its numerator and denominator.

### 9.1 Directional coherence

High moral sensitivity could reflect random instability. Therefore, each moral
perturbation will also have an annotated expected response dimension, not an
expected correct action. For example, a magnitude manipulation should register in
consequence assessment; a consent manipulation should register in duty/consent
assessment. We will report the proportion of moral variants for which movement
occurs on the targeted dimension.

## 10. Statistical analysis

The primary analysis will use a hierarchical model or cluster-aware paired
estimator with:

- fixed effects for system condition, perturbation type, and their interaction;
- random intercepts for scenario family;
- random slopes for perturbation type by scenario family when estimable;
- run seed/repetition represented as a repeated observation;
- perturbation subtype included as a fixed or partially pooled effect.

The confirmatory H3 test is the condition-by-perturbation interaction contrasting
Parliament with compute-matched solo. Effect sizes, confidence intervals, and raw
family-level distributions will be reported. A scenario-family cluster bootstrap
will serve as a robustness analysis.

H1-H3 form the primary family. Multiplicity will be controlled using Holm's method,
with H3 designated as the principal contrast. H4 framework-target mappings form a
separate confirmatory family. Other framework, geometry, domain, and prose analyses
are exploratory.

## 11. Framework-differentiation analysis

For each moral perturbation, calculate change in every framework's native typed
ledger fields and admitted vote. The primary H4 outcome is:

\[
T = \Delta_{targeted\ framework}
  - \operatorname{mean}(\Delta_{non-targeted\ frameworks})
\]

Positive \(T\) indicates selective specialization. Analyses will distinguish:

- changes in factual attention;
- changes in framework-native classification;
- changes in directional recommendation;
- changes in admitted policy weight.

This prevents a framework from appearing responsive merely because it copied a
shared factual update or another framework's normative priority.

## 12. Exclusions and failures

Runs will not be excluded for giving an undesirable answer. Predefined technical
statuses are:

- valid response;
- explicitly unresolved response;
- schema/extraction failure;
- grounding rejection;
- timeout or infrastructure failure;
- safety refusal unrelated to scenario content.

Unresolved and grounding-rejected outputs are substantive system outcomes and will
remain in the primary intention-to-evaluate analysis under the preregistered
distance encoding. Infrastructure failures will be rerun only according to the
common retry policy. Failure rates will be reported by condition. A per-protocol
analysis excluding technical failures will be secondary.

## 13. Blinding and analysis integrity

- Stimulus validators will not see model outputs.
- Manual output adjudicators, if required, will be blinded to system condition and
  perturbation type.
- System names will be replaced with random condition codes during confirmatory
  analysis.
- The analysis script and expected table shells will be frozen before condition
  labels are revealed.
- Any protocol deviation will be logged with its date, reason, and whether it was
  decided before or after outcome inspection.

## 14. Reproducibility record

The release should include, subject to model-provider terms:

- canonical scenarios and all accepted variants;
- perturbation manifests and validation annotations;
- model/version identifiers and inference configuration;
- prompts, schemas, retrieval snapshot identifiers, and code commit;
- per-run token/call accounting and random seeds;
- raw outputs, typed traces, admission/rejection records, and final responses;
- frozen extraction and scoring code;
- the preregistration and all dated amendments.

## 15. Planned figures and tables

1. A two-dimensional plot with moral sensitivity on the vertical axis and nuisance
   sensitivity on the horizontal axis; the preferred region is upper-left.
2. Paired family-level moral and nuisance distances for each condition.
3. Component-wise distances for choice, confidence, framework state, constraints,
   and proposition-grounded reasons.
4. Framework-by-perturbation heatmap for functional differentiation.
5. Compute use and technical failure table by condition.

## 16. Interpretation limits

A positive result would support the claim that the tested architecture better
discriminates normative signal from representational noise under the study's
controls. It would not establish that Parliament's recommendations are morally
correct, sufficient for moral wisdom, unbiased across domains, or superior to all
possible single-agent scaffolds.

A null result may indicate no architectural advantage, insufficiently strong or
poorly validated perturbations, an insensitive response metric, or inadequate
power. A Parliament disadvantage on nuisance sensitivity is a meaningful failure,
not a reason to redefine nuisance variants after observing outcomes.

## 17. Decisions required before preregistration

The following must be finalized after the disjoint pilot:

1. pinned model and provider version;
2. scenario-family count and domain allocation;
3. nuisance and moral variants per family;
4. repeated-run count and seed schedule;
5. exact compute-matching budget and retry policy;
6. response-distance weights and missing-data encoding;
7. smallest effect of interest, power target, and resulting sample size;
8. annotator agreement threshold and adjudication rule;
9. statistical model family and fallback if convergence fails;
10. value of \(\epsilon\) for the descriptive ratio.
