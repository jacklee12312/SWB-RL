# Current research state

This document is the handoff point for continuing the checked-in research. It
separates published evidence from work that was still running when the public
repository was prepared.

## Software and data contract

- Source baseline: private-research commit `a1f9a8928e15ea171fc7991b4c0b696bee120a10`.
- Policy architecture: `entity_action_v1`.
- Observation contract: `observation-v4.1`.
- Action contract: `action-112-v2`.
- Match setup: official seven-class, seven-deck distribution.
- Card catalog: 826 records, including 735 collectible and 91 generated cards.
- Database SHA-256: `df069e713a97493c885266b72f303874035beea571147ba14b77c57c9e631376`.
- Structured rule files: 119.

The exact action-layout, observation-schema, card-vocabulary, rulebook, and
training-pool hashes are embedded in every schema-v2 checkpoint and in the
generation population manifests. Training refuses incompatible manifests or
checkpoints instead of silently continuing under a changed contract.

## League design

The active population contains six independent lineages with policy seeds
`20260903` through `20260908`. A generation trains every lineage serially for
approximately 250,000 additional agent steps. Opponents are sampled from a
shared cross-seed pool using payoff-aware Hard PFSP with a bounded historical
archive.

Every published generation must pass:

1. all six training jobs;
2. all 15 unique active-policy matchups at 196 games per pair;
3. six parent-validation matchups;
4. safety checks for illegal actions, mask mismatches, NaN/Inf, and truncation;
5. the scheduled historical-forgetting audit;
6. the registered generation stopping gate.

Intermediate output is never treated as a published generation merely because
its six training jobs finished.

## Latest stable checked-in generation metadata

Generation 1 is the first complete evolving-PFSP generation represented in the
repository metadata. It passed every publication gate on 2026-08-05.

| Seed | Agent steps | Checkpoint SHA-256 |
| --- | ---: | --- |
| 20260903 | 1,251,592 | `c3b7b689cd9d8ebece200edec37f666488e55dfe4dcaf6f58ca547f67c39a5ae` |
| 20260904 | 1,250,138 | `261dc0daaef9c12cd595aea37ec6f78563c2f802694897d9e964f65f190e3f12` |
| 20260905 | 1,252,604 | `b967adbb6dc1234002d089f8d14aec9bbbfd2a6f3b8ccdfe1940fa82e34792bd` |
| 20260906 | 1,251,765 | `8c3da19d22b8a3618c52963c17c27c7fcb8fa45a0be52ddd2aba7f7236911e4b` |
| 20260907 | 1,251,450 | `37d81b4984f1fddc1db4ca2a35c861059526bc4306db84449b50ffa4eba687c4` |
| 20260908 | 1,251,217 | `ed82c9230e4a62bb20ada59ce6c307f0a1f34575bee46b16fd3e30fbb2f528bc` |

Generation 1 active evaluation used 2,940 games: 15 unique pairs times 196
games. The uniform-population worst-case score was `0.4600`. Parent validation
had a mean score of `0.5238`, a minimum lineage score of `0.4949`, and all six
lineages passed the non-degradation gate. Training reported zero illegal
actions, zero action-mask mismatches, zero NaN/Inf values, and zero truncated
episodes.

The full machine-readable records are under
`data/reports/league_training/generations/generation_001/`.

## Installing the continuation snapshot

The repository intentionally keeps large `.pt` files out of normal Git. The
latest release manifest lists every asset and checksum required to restore the
published state:

```powershell
python -m scripts.bootstrap --install --with-ui --release latest
```

The release contains complete schema-v2 checkpoints, not inference-only
weights. Each active learner includes model parameters, optimizer state,
trainer configuration, current agent steps, recurrent state, opponent state,
and Python/NumPy/PyTorch/CUDA RNG state. The release also includes every
historical opponent referenced by the lineage manifests.

After installation, continue the registered queue with:

```powershell
python -m scripts.run_ppo_league_generations --max-target-generation 8
```

The runner reconstructs its queue from existing immutable generation reports,
skips validated completed jobs, and resumes a partial checkpoint when one is
present. Use a feature branch and a separate report/checkpoint root for a new
algorithm rather than overwriting this baseline.

## Known limitations

- Strong tactical play has not yet been established. Early models understand
  basic legal play and pressure but still show weak long-horizon resource and
  evolution planning.
- The current tactical suite is small and should grow from reviewed human
  replays and deterministic synthetic cases.
- The six-lineage population is a research baseline, not evidence that PFSP is
  superior to NFSP, DeepNash, reward prediction, or oracle-guided variants.
- Card artwork and audio are not distributed. The simulator displays locally
  supplied images when available.
- The community card dataset includes underlying game content that is not
  licensed under MIT; see `THIRD_PARTY_NOTICES.md`.
