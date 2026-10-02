# Integration_Alpha

Alpha is the step after transport. The Z10 packet already reaches Parliament's grounding prompts, and two live `o3` calls completed. Both were rejected. Alpha asks whether that packet changes admission, and it keeps Z10 frozen while it asks.

The original scenario stays the source text in every arm. Parser evidence stays advisory. No reading is selected. No sentence is rewritten. Deliberation stays unwired. The held-out set stays unused.

## What the live calls showed

Both sentences were stored as one clause, `C0`. Both actions were grounded to that same clause, so each failed the distinguishing-clause check.

| Scenario | Actions | Admission | What else failed |
| --- | --- | --- | --- |
| Maria can save the child, but not the dog. | save the child; save the dog | REJECTED | The dog was given a welfare polarity. A direct intervention on an animal must stay NEUTRAL. The repair pass then dropped Maria, the child, and the dog. |
| Lila gives Omar the medicine, but not Nora. | give the medicine to Omar; give the medicine to Nora | REJECTED | The world model committed, then the repair pass dropped Lila, Omar, Nora, and the medicine. |

`parser_advisory.semantic_use` is `not_assessed` on both. Two model calls mean the packet was offered on the first pass and the repair pass. They do not show that the model used the reconstructed roles.

Records: `diagnostics/z10_live_save_child_dog.json`, `diagnostics/z10_live_medicine_nora.json`.

## Path

1. **Pair the arms.** For each development scenario, run raw text and the Z10 advisory packet with the same model, the same actions, and the same attempt limit. Keep the source text identical. Record admission, clause count, party survival, and the animal-polarity error separately.

2. **Read the worlds, not only the status.** A committed world model can still be rejected, as the medicine call was. For each arm, note whether the two actions received different clauses, whether the dog stayed NEUTRAL, and whether the repair pass kept parties the error list did not name.

3. **Score semantic use by hand on these two scenarios first.** Check whether the admitted or rejected world contains the reconstructed roles: Maria did not save the dog; Nora is a competing recipient, object, or subject. If the roles are absent from both arms, the packet is not yet doing work.

4. **Change the bridge only if the pair shows a repeated miss.** The first candidate hint is that competing readings are separate clauses and must not share one citation. That hint stays advisory. It does not pick a reading, and it does not replace the source text. Re-run the same pair after the hint.

5. **Leave Parliament's admission rules in place.** The animal-polarity rule and the repair-party rule are Parliament checks. Alpha records them. It does not weaken them to obtain a COMMITTED status.

## Stop

Alpha stops when the paired runs say whether the packet changes clause split, party survival, and role recovery on the development scenarios. A rejected world that preserves those roles is a useful result. A committed world that collapses both actions into `C0` is not success.
