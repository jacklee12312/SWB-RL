# Contributing

Thank you for helping improve SWB RL. Accuracy and reproducibility take
priority over broad but unverified card coverage.

## Development setup

```powershell
git clone https://github.com/jacklee12312/SWB-RL.git
cd SWB-RL
python -m scripts.bootstrap --install --with-ui
```

Before submitting a change, run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

For changes to legal actions, resolution, cards, targets, combat, turns, or
observations, also run:

```powershell
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

## Rules contributions

- Express behavior through generic commands, events, targets, and operations.
- Put card-specific mappings in `data/rules/`; do not add large card-ID switch
  statements to the shared engine.
- Add normal, illegal, and boundary-path tests.
- Keep unsupported or uncertain behavior explicit.
- Cite the source or reproduction used to settle ambiguous timing and wording.

## Pull requests

Keep changes focused and include:

- the behavior being changed;
- tests and commands actually run;
- determinism or checkpoint compatibility impact;
- remaining unsupported semantics.

Do not commit card artwork, match histories, training logs, local checkpoints,
credentials, or generated caches.
