# Parliament backups and the new learning branch

Repository: [bpkowal/Good_News_Studio](https://github.com/bpkowal/Good_News_Studio).

| Component | Location | Contents |
| --- | --- | --- |
| Non-simulation Parliament application | [Parliament_Frontend](https://github.com/bpkowal/Good_News_Studio/tree/Parliament_Frontend) | October 28 checkpoint: arguments, Contractualist adapter, parser/blueprints, viewer, traces and tests |
| Exact native Parliament source | [backup/parliament-native-20261028](https://github.com/bpkowal/Good_News_Studio/tree/backup/parliament-native-20261028) | Commit `3eb9d2ea331eb936e522d5a7132a14bd98ef2217`, matching the runtime lock |
| Weight-learning experiments and subsequent game work | `Learning_Deliberation_Game` | New branch based on the preserved application; the earlier simulations remain explicitly labeled |

The [checkpoint description](CHECKPOINT_20261028.md) records validation and
limitations. Older backup branches remain available. Branch names identify
development lines; run manifests and source archives identify the bytes used by
a particular experiment. No previous failed or partial analysis is relabeled as
successful by backing it up.
