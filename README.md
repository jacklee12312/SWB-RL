# SWB RL

This repository turns `shadowverse_cards.json` into a normalized SQLite card
database and provides a small two-player environment suitable for early
reinforcement-learning experiments.

## Current Scope

The database currently preserves 826 cards from `shadowverse_cards.json`,
including 735 collectible cards and 91 non-collectible/generated cards from set
`90000`. The normalized SQLite database stores localized names, skill metadata,
printed skill text, flavor text, alternate modes, card references, texture
references, extra asset metadata, source import records, and the original JSON
record. Rule support is tracked separately so database completeness is not
confused with engine completeness.

Cards from set `90000` are marked non-collectible. They remain available for
future generated-card effects, but are excluded from training/deck-building
pools and rejected if passed directly in an initial deck. Deck validation
requires exactly 40 collectible cards from the selected class and/or neutral.

Abilities are normalized in two relational tables:

- `abilities`: canonical keyword, implementation status, trigger events, aliases
- `card_abilities`: card-to-ability mapping plus the original matched keyword

`rule_support.keywords` remains the raw extracted text for auditing. Queries and
the engine use the normalized tables, so aliases such as `毁灭 -> 必杀` do not
need special handling at runtime.

## Implemented Engine Surface

The deterministic rules core supports:

- seeded reset, shuffle, draw, turn progression, deck exhaustion damage,
  reproducible command sequences, and deterministic full-state fingerprints for
  replay diagnostics;
- player health, mana growth, hand limits, board limits, graveyards, banished
  cards, stable entity IDs, and origin metadata;
- follower, spell, and amulet play through `GameEngine.apply(command)`;
- follower combat, leader attacks, guard targeting, summoning sickness,
  simultaneous combat damage, and state-based deaths;
- implemented combat keywords including `守护`, `疾驰`, `突进`, `必杀`, `潜行`,
  `吸血`, and `屏障`;
- normal evolution and manual super-evolution, including independent
  super-evolution resources, unlock timing, once-per-turn limits, and same-turn
  protection from effect damage/destruction only on the turn the follower
  super-evolved, but not combat damage, even when an opponent-controlled
  trigger resolves during that turn;
- structured effects for damage, healing, draw, summon, destroy, banish, return,
  discard, transform, explicit target-set binding, stat changes, keyword
  changes, cost modifiers, and attack or targeting restrictions;
- selected, random, all, implicit, leader, board, all-board, hand, and
  graveyard target flows, including pending choices, target-leaves-play safety,
  and target revalidation when a pending target changes controller or no longer
  matches the original target filter; ordinary play choices cover selected
  board targets moving to the graveyard or changing controller, selected hand
  cards leaving hand, and selected graveyard cards moving to hand before
  resolution resumes; `target_key` stores an ordered tuple of selected board
  targets and `previous_target` chains revalidate each member against the
  original bound target filter before later operations resolve;
- true multi-target pending choices through `target_count` or
  `target_count_expr`, with explicit `allow_duplicate_targets` /
  `allow_duplicates` policy, candidate-shortage handling, command-level
  selection progress, ordered multi-target `target_key` bindings, and
  per-target revalidation before resolution; real card `10351120` demonstrates
  selecting and destroying two enemy followers, while partial real rule
  `10474120` demonstrates reusing the same selected set for later damage;
- field-backed `土之印` stacks and structured `土之秘术` payment, including
  Sigil entry/merge/depletion, effect-destroy protection, opposing manual-target
  protection, generated `大地之魔片`, nested post-payment operations, and
  controller/opponent count conditions and expressions;
- command-level `融合` from hand, including structured material filters and
  count limits, variable-count selection with explicit confirmation, once-per-
  card-per-turn tracking, atomic hand-zone revalidation, a distinct consumed
  material zone, inherited material identity, and `card_fused` events; real
  card `10213310` demonstrates an exact Elf-material fusion rule and a play
  effect that draws two cards after fusion instead of one;
- structured `瞬念召唤` at turn start before the normal draw, with persistent
  match evolution counts, seeded random candidate ordering weighted by copies,
  one copy per card definition per timing, board-full handling, summon-event and pending-choice
  continuations, and explicit `card_invoked` events; real card `10404110`
  invokes after six evolutions, gains its countdown crest, and returns to hand;
- follower healing for selected, random, and all-unit target flows, used by the
  invoked Sandalphon crest to heal all allied followers alongside the leader;
- structured `target_exists` no-target branches that reuse normal target
  candidate generation before queuing a then/else effect branch, including
  unit-or-leader fallback targets when no target-dependent condition is present;
- countdown amulets, explicit last words, fanfare/play rules, attack/clash,
  evolve/super-evolve, turn-start/turn-end triggers, and trigger continuations
  that can pause for choices;
- death-batch event diagnostics that expose the active-player-first,
  left-to-right order used by destroyed, left-play, and Last Words lifecycle
  events, including follower/amulet composition for mixed death batches;
- `death_batch_end` emblem triggers that fire after a death batch's Last Words
  complete, with any new deaths collected into a later death batch;
- recursive resolution-loop diagnostics for events, effects, death batches,
  active emblem batches, recent emblem triggers, and suspended continuations;
- partial higher-level mechanics and primitives for cooperation, `觉醒`, `连击`,
  necromancy, reanimate, spellboost-style hand cost changes, emblems, optional
  decisions, choose-one decisions, play modes, and runtime modifiers.

The RL adapter provides a fixed 111-action space, 257-feature public
observation, action mask, terminal reward, graveyard choice paging, special
hand actions for fusion/play modes, and super-evolve actions. `info()` is public by default and
redacts debug transcripts/events unless `debug_info=True` or
`info(debug=True)` is used, including pending-choice and graveyard-page returns.
Public observations and default info are regression-tested not to depend on
opponent hand identity or deck identity/order while a real-card pending choice
is awaiting resolution. The public observation includes explicit
controller/opponent `觉醒` flags derived from maximum mana and public
controller/opponent `连击` counts for the current turn, plus pending multi-target
choice size and progress.
The final two features expose the controller and opponent's public Earth Sigil
totals. Sigils are board amulets rather than player-side counters: entering
Sigils merge into the newest amulet, merged Sigils are banished, and a depleted
Sigil is destroyed. This follows the
[official Worlds Beyond mechanic description](https://beginner.shadowverse-wb.com/ja/deck_shindan/result04/).
Fusion adds own-hand fused-material counts and current-turn availability plus
public board fused-material counts without exposing opponent hand identities.
The material transition follows the
[official Fusion glossary](https://shadowverse-wb.com/chs/deck/cardslist/card/?card_id=10021110):
Fusion is usable from hand once per turn, an unspecified count permits multiple
materials at once, and consumed materials do not enter the graveyard.
The observation also exposes both players' public number of follower evolutions
this match. Invocation itself is automatic and therefore adds no RL action.
The current implementation follows the
[official Skybound Dragons mechanic overview](https://shadowverse-wb.com/chs/cards/pack/skybound-dragons/)
and [official card glossary](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10404110):
Sandalphon is the unique Invoke card; Invoke enters from the deck when its
condition is met, orders simultaneous candidates randomly, and limits duplicate
copies to one while letting their copy count affect selection probability.

## Unsupported Or Partial

The engine still does not model the full SWB ruleset. Unsupported behavior must
remain visible instead of silently behaving as implemented.

Known broad gaps include:

- exact semantics for many real cards and most generated-card workflows;
- full `策动`, `信仰`, and `奥义` semantics, plus broader
  real-card coverage for `土之秘术`, `觉醒`, and `连击` beyond the currently
  authored examples;
- Fusion cards that transform in hand when fused and abilities on other cards
  that trigger from a fusion event remain unsupported; the command-level
  material transition and source-card fused-count conditions are covered;
- `10404110` remains partial only because its Fanfare `解放奥义` damage sequence
  is deferred to the Union Burst slice; its Invocation and crest clauses are
  structured and tested;
- non-manual super-evolution edge semantics;
- remaining trigger-ordering edge cases beyond the current death-batch
  ordering diagnostics and `death_batch_end` boundary triggers, including
  unsupported `death_batch_start` emblem triggers, plus broad real-card
  coverage audits;
- `10474120` remains partial: its selected-set damage is covered, but making
  the selected followers lose all abilities and applying persistent
  leader-damage amplification remain unsupported primitives;
- Earth Sigil `策动` remains unsupported as its own command-level slice; this
  keeps `10031210` partial even though its draw and Earth Sigil board semantics
  are covered. `10032310` and `10732120` are exact consume/gain demos;
- keyword registry status is intentionally conservative: handlers and generic
  primitives may exist before a keyword is marked fully implemented.

See `docs/roadmap.md` for the current implementation roadmap.

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
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

## Random self-play smoke test

```powershell
python -m scripts.random_self_play --games 100
```

Enable runtime state-invariant checks during a smoke run with:

```powershell
python -m scripts.random_self_play --games 100 --validate-invariants
```

Runtime invariants cover zone/entity consistency, pending-choice shape including
target/leader choice identity, and effect-stack frame structure so corrupted
suspended effects fail explicitly. Illegal-command no-mutation tests compare
full engine fingerprints, including hidden zones and suspended effects.

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
- `commands.py`: play, fusion, attack, evolve, end-turn, and choice commands
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
- `79..105`: fusion or special play-mode actions for hand slots
- `106..110`: super-evolve a board slot

Always apply `info["action_mask"]` before sampling or selecting an action.
By default, `info()` is public and redacts debug transcripts/events. Use
`ShadowverseEnv(..., debug_info=True)` or `env.info(debug=True)` only for
diagnostics. Public `info()` is intentionally a small whitelist and does not
include pending-choice option labels, entity IDs, player objects, deck lists, or
event/log transcripts.

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

## Roadmap

See `docs/roadmap.md` for the current priority order, known gaps, and suggested
vertical slices. Keep that file current when implementation status changes.
