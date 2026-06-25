# SWB RL

This repository turns `shadowverse_cards.json` into a normalized SQLite card
database and provides a small two-player environment suitable for early
reinforcement-learning experiments.

## Current scope

The database preserves all 740 cards, localized names, skill metadata, printed
skill text, texture references, and the original JSON record. Rule support is
tracked separately so that database completeness is not confused with engine
completeness.

Cards from set `90000` are marked non-collectible. They remain available for
future generated-card effects, but are excluded from training/deck-building
pools and rejected if passed directly in an initial deck.

Abilities are normalized in two relational tables:

- `abilities`: canonical keyword, implementation status, trigger events, aliases
- `card_abilities`: card-to-ability mapping plus the original matched keyword

`rule_support.keywords` remains the raw extracted text for auditing. Queries and
the engine use the normalized tables, so aliases such as `毁灭 -> 必杀` do not
need special handling at runtime.

The first engine version supports:

- four-card opening hands; the first player then gains one mana and draws
- 20 health, mana growth, draw, hand and board limits
- playing followers
- follower combat and simultaneous damage
- summoning sickness
- `守护`, `疾驰`, and `突进`
- follower evolution: +2/+2 and rush, once per turn, with configurable unlock
  timing/points
- basic super-evolution commands and same-turn protection from damage/effect
  destruction; the full resource model is still pending
- simple unconditional fanfares: draw, leader damage/heal, self buff, mana restore
- machine-authored spells, including effects that pause for a target choice
- amulets sharing the five board slots with followers
- amulet play effects, countdown reduction, destruction, and explicit last words
- deck exhaustion damage
- fixed-size observations and discrete action masks
- deterministic reset and shuffle by seed

Generated cards, complete super-evolution resource rules, broad conditional
fanfares, and most triggered abilities are preserved in the database but are not
executed yet.

## Build the database

From the repository root:

```powershell
python -m swb.db.import_cards
```

The default output is `data/cards.sqlite3`. Re-running the command replaces the
imported rows in one transaction.

## Refresh from SVA

The canonical card data endpoint used by this project is:
`https://sva.hypd.asia/data/cards.json`.

Download the latest JSON, back up the current JSON/database, rebuild SQLite,
and record the source URL, timestamp, card count, and SHA-256:

```powershell
python -m scripts.fetch_sva_cards
```

Backups are written to `data/backups`. The richer source format is normalized
into localization, flavor text, alternate mode, card reference, ability, and
extra asset metadata tables. The complete source record is also retained in
`cards.raw_json`.

## Run tests

```powershell
python -m unittest discover -v
```

## Random self-play smoke test

```powershell
python -m scripts.random_self_play --games 100
```

Enable runtime state-invariant checks during a smoke run with:

```powershell
python -m scripts.random_self_play --games 100 --validate-invariants
```

To print one deterministic match:

```powershell
python -m scripts.demo_match --output data/demo_match.log
```

To inspect the implementation status of every registered ability:

```powershell
python -m scripts.ability_status
```

The ability registry and its no-op extension points live in
`swb/engine/abilities.py`. Placeholder handlers record matching lifecycle
events in `info["placeholder_ability_events"]` without changing game state.

## Engine architecture

The rules core is command based and independent of RL action numbers:

- `state.py`: mutable match, player, unit, zone, and pending-choice state
- `commands.py`: play, attack, evolve, end-turn, and choice commands
- `events.py`: ordered lifecycle events emitted by rule execution
- `effects.py`: reusable primitive effect operations and target kinds
- `card_rules.py`: JSON rule loading keyed by card ID and trigger
- `resolution.py`: command validation, event resolution, stabilization, death,
  graveyard, and winner handling
- `environment.py`: compatibility RL adapter for action encoding, observations,
  masks, and rewards

New rule code should target `GameEngine.apply(command)`. It should not modify
RL observations or integer action encoding directly.

Machine-authored card rules live in `data/rules`. Explicit rules are preferred
over the temporary Chinese-text fanfare parser; the parser remains only as a
compatibility fallback while card coverage is migrated.

The environment has 111 actions:

- `0`: end turn
- `1..9`: play a hand slot
- `10..39`: five attacker slots times six target slots
- `40..44`: evolve a board slot
- `45..60`: resolve one of up to 16 pending choice options
- `61..78`: graveyard choice paging and slots
- `79..105`: special play modes for hand slots
- `106..110`: super-evolve a board slot

Always apply `info["action_mask"]` before sampling or selecting an action.
By default, `info()` is public and redacts debug transcripts/events. Use
`ShadowverseEnv(..., debug_info=True)` or `env.info(debug=True)` only for
diagnostics.

The currently authored spell/amulet examples are in
`data/rules/spells_and_amulets.json`. They cover:

- `智慧光辉`: draw without a target
- `龙人碎击`: choose an enemy follower and deal damage
- `魔女的炼金炉`: amulet play effect
- `祥和的教会`: countdown and last words
- `坚固的雾卷花`: targeted amulet play effect

Run a mixed-card RL smoke match and write its transcript:

```powershell
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

The mixed-card smoke match also accepts `--validate-invariants` for debugging
state consistency regressions.

## Suggested next milestones

1. Add explicit, machine-authored effect definitions instead of parsing Chinese
   skill text at runtime.
2. Add evolution and target-selection continuation states.
3. Wrap the environment for PettingZoo or Gymnasium after the multi-agent reward
   convention is finalized.
4. Add deck validation and format/card-set constraints.
5. Establish heuristic and search-based baselines before training PPO/DQN.
