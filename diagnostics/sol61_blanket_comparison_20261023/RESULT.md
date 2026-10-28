# Blind and Parliament-informed GPT-6.1 Sol reviews

Open the [side-by-side browser report](http://127.0.0.1:7860/data/sol61_blanket_comparison/index.html), the [blind response](blind.md), the [informed response](informed.md), or the [frozen model comparison](MODEL_COMPARISON.md).

## Result

Both Sol reviews recommend giving the blanket to Rowan. The blind review already supplies the essential 0.8-versus-0.6 expected-survivor arithmetic, recognizes Sasha's directed promise claim, states a defeasible priority judgment, addresses a serious objection, and identifies reliance-based reversal conditions. It did not see Parliament.

The informed review preserves that recommendation and improves several distinctions. It rejects Parliament's strict-maximin calculation, challenges the source-level arithmetic objection, distinguishes a sincere promise later overridden from deceptive promising, and avoids calling a 60% rescue attempt waste or incompetence. It also explains that an omission-classification problem does not remove Sasha's independent promise claim.

The third, separate Sol call assesses this as a **modest argument improvement**, rather than a new decision basis. That is an attributed model comparison, not an independent normative verdict. Its 24 evidence quotations all match the corresponding frozen source strings.

| Condition | Recommendation/result | Main strength | Main weakness |
| --- | --- | --- | --- |
| Latest Parliament | FRAMEWORK_PARTIAL; no governing verdict. Virtue fully favors Rowan, Deontology attenuated toward Sasha; Care and Rawlsian retain Rowan preferences while abstaining; Utilitarian withholds full ranking. | Inspectable world, multiple framework claims, explicit objections and retained uncertainty. | Incorrect Rawlsian worst-case argument, an arithmetic admission barrier despite clear source-level survivor arithmetic, and insufficiently defended duty priority. |
| Blind Sol | Rowan | Clear, usable source-grounded argument that takes Sasha's promise seriously. | The priority given to a 20-point survival advantage remains a contestable normative judgment. |
| Parliament-informed Sol | Rowan | Explicitly diagnoses several Parliament errors and better bounds the promise exception. | Same unresolved priority judgment; “otherwise unassigned” risks prejudging what the promise already allocated. |

## Assistant content assessment

On this example, I find the blind Sol response clearer and more useful as a practical ethical argument than Parliament's aggregate presentation. This is based on its correct reasoning and substantive engagement, not merely on selecting an action while Parliament remains partial. A well-defended Sasha recommendation could be strong; there is no stipulated uniquely correct moral verdict.

Parliament's useful contribution to the informed reply is chiefly a collection of concrete arguments and failure points to examine. Its context yields better critical distinctions, but does not solve the central normative weighting. In particular, identifying a false-deceptive-promising analogy weakens one argument for absolute fidelity without establishing that this promise must lose.

The informed review's improvement is therefore limited. The shared claim that a 20-percentage-point advantage is “substantial” needs a more developed account of its priority against a directed lifesaving promise. The context also introduces an assumption about initial, otherwise-unassigned aid. That assumption is partly disclosed but does not settle whether the promise itself creates priority.

The comparator additionally identifies a procedure neither review adequately examines: if randomization over the two allocations is admissible, giving Rowan the blanket with probability 3/7 equalizes both hikers' survival prospects at 12/35. This arithmetic is correct. Whether that procedure is ethically appropriate, or within the intended decision procedure, is a separate normative question. It is not a newly admitted world action.

## Isolation and freezing

- The model was exactly `gpt-6.1-sol`, confirmed by provider responses and the [official model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
- The protocol and evaluation rubric were frozen before the first call.
- Blind input contained only the scenario and common review instructions/schema. No Parliament material, tools, conversation continuation or prior messages were supplied.
- The blind response was frozen before the informed request began. The informed request contained the exact frozen public Parliament viewer packet, including all five recorded arguments, world, objections, statuses and latest topography. It did not receive the blind answer or assistant editorial diagnoses.
- Both reviews used high reasoning effort, the same schema and the same 16,384-token completion ceiling. SDK retries were disabled. No responses were regenerated or substituted.
- Only after both reviews were frozen did a fresh third Sol call receive them and Parliament's packet for comparison.
- All protocol/response/request/provider hashes and timestamp ordering pass [verification](verification.json). Four isolation/ordering tests pass in [the test log](protocol_tests.log).
- Raw responses and freeze receipts remain intact. The experiment changed no shared world, operative ledger or Parliament preference.

## Limits and efficiency

This is one scenario and one sampled response per condition. Different input lengths and ordinary model variation prevent interpreting the observed additions as a reliable treatment effect. The comparison call uses the same model family and sees condition labels; it is not an independent or blinded grading panel. The two Sol replies are not subject to Parliament's native admission requirements, and Parliament used different models and more calls. This is not a budget-matched architecture benchmark.

The blind request used **477 input tokens**; the informed request used **75,305**. Both had the same output ceiling. The informed packet deliberately includes the full viewer record, including lengthy plotting/provenance material. This tests access to that complete packet, not the efficiency of a compact Parliament summary. The extra input produced useful criticism but no changed recommendation in this sample.

The next implication to investigate is whether a compact, evidence-bound set of competing claims can preserve those critical gains at much lower input cost. That experiment has not been run here.

## Frozen artifacts

- [Protocol and preregistered rubric](protocol.json), [protocol freeze](protocol.freeze.json).
- [Blind request](blind.request.json), [blind provider response](blind.provider_response.json), [blind freeze](blind.freeze.json).
- [Informed request](informed.request.json), [informed provider response](informed.provider_response.json), [informed freeze](informed.freeze.json).
- [Comparison request](comparison.request.json), [comparison provider response](comparison.provider_response.json), [comparison freeze](comparison.freeze.json).
- [Exact frozen Parliament packet](parliament_packet.json), [comparison quotation checks](comparison_quote_checks.json).
