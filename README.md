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

Cards from set `90000` are marked non-collectible. They remain resolvable by
generated-card effects, but are excluded from training/deck-building pools and
rejected if passed directly in an initial deck. Deck validation requires
exactly 40 collectible cards from the selected class and/or neutral. The
auditable [token report](data/reports/token_audit.md) distinguishes database
references from executable producer paths and behavior completeness for all 91
cards: 10 complete entries, 11 partial entries, and 70 with no authored entry.

Abilities are normalized in two relational tables:

- `abilities`: canonical keyword, implementation status, trigger events, aliases
- `card_abilities`: card-to-ability mapping plus the original matched keyword

`rule_support.keywords` remains the raw extracted text for auditing. Queries and
the engine use the normalized tables, so aliases such as `毁灭 -> 必杀` do not
need special handling at runtime.

The [ability registry audit](data/reports/ability_audit.md) records a reason
and test evidence for all 34 statuses (18 implemented, 5 partial, 11
placeholder). Primitive availability is reported independently: all 34 have a
covered generic boundary, but that never upgrades a partial or placeholder
keyword whose full tagged-card semantics still require structured rules.

The [rule coverage report](data/reports/rule_coverage.md) now includes a
clause-audit layer without changing its legacy coverage categories. Of 87
nominally exact collectible cards, 23 currently have explicit implemented text
and direct test evidence; 64 are flagged `unverified_exact`. Blockers are
separately typed as missing rule/schema/primitive/targeting, unclear timing or
text, external blocker, or unverified audit. Rule metadata supports version and
errata fields, and the report records the imported source snapshot hash so card
database refreshes are visible.

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
  `威慑`, `灵气`, `吸血`, and `屏障`;
- official `威慑` attack legality: an opposing follower cannot attack an
  Intimidate follower, but abilities can still select and damage it. A follower
  with both Ward and Intimidate does not enforce Ward because Intimidate makes
  it unavailable as an attack target; any other visible Ward still applies.
  Runtime add/remove, temporary removal, evolution, super evolution, transform,
  legal commands, and RL masks share the same effective-keyword state. Real
  `10451120` exactly demonstrates static Intimidate plus its summon-and-self-
  damage Last Words. This follows the [official help glossary](https://shadowverse-wb.com/ja/help?tab=tab0)
  and [official Bazarraga card page](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10451120);
- official `灵气` / Aura targeting legality: opposing manually selected
  abilities cannot choose an Aura follower or amulet, while its controller's
  abilities, random effects, all-target effects, and follower attacks remain
  legal. Runtime add/remove, temporary removal, transform, pending-choice
  revalidation, and RL choice masks share centralized target candidates.
  Structured `non_intrinsic_keyword` annotations stop conditional or random
  Aura mentions from becoming initial keywords. Real `10161140` exactly
  combines intrinsic Aura with a board listener that grants itself `+1/+0`
  until turn end whenever its controller activates an amulet. This follows the
  [official Aura glossary](https://shadowverse-wb.com/ja/help?tab=tab0);
- normal evolution and manual super-evolution, including correct `+2/+2` and
  `+3/+3` stat changes, independent resources, unlock timing, once-per-turn
  limits, all-damage/effect-destroy protection during every turn owned by that
  follower's controller, and the 1-damage leader bonus when its attack destroys
  the attacked follower (including destruction by an attack-time ability);
- structured `super_evolve_unit` effects that grant the complete super-
  evolution state without spending SEP or consuming the manual once-per-turn
  action; effect-caused super evolution counts as an evolution but, unlike an
  SEP action, does not fire `进化时` or `超进化时` keyword abilities;
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
  selecting and destroying two enemy followers. Exact `10474120` reuses the
  same selected set for ability removal followed by damage, including
  candidate shortage and targets leaving before resolution;
- structured `remove_all_abilities` suppresses a follower's printed keywords,
  future printed triggers, Last Words, and board listeners while clearing
  runtime-granted keywords and ability restrictions without changing identity,
  stats, damage, evolution, or origin. Queued effects continue, transform and
  re-entry restore the new/current printed ability set, and later runtime
  grants remain possible;
- leader damage-taken modifiers support deterministic additive stacking,
  increases and reductions, permanent/turn/source-in-play lifetimes, effect,
  combat, and self-damage, with source leave/transform/control revalidation,
  fingerprint coverage, and auditable damage metadata. Exact `10474120`
  permanently gives the opposing leader damage taken +1;
- structured ordinary-card event listeners sourced from the board, hand, or
  shared leader area. Rules can observe `amulet_activated`, `card_fused`,
  follower summon/evolution/destruction, amulet destruction, entity leave-play,
  card play, and turn boundaries; filter the event card by type, original cost,
  class, Trait ID/name, card ID/name, or runtime keyword; scope to
  owner/opponent/any events and
  turns; target the exact `event_source`; and enforce per-turn or lifetime
  activation limits. Simultaneous listeners snapshot in active-player-first,
  board/hand/leader-area order, revalidate their source before activation, and
  preserve pending-choice continuations. Real `10443110` now exactly gains
  `守护` when another allied original-cost-2 follower enters play. Its structured
  `non_intrinsic_keyword` annotation also prevents the database's full-text
  keyword audit field from granting that Ward before the listener fires.
  Repository `CardDefinition` values now retain normalized Trait metadata and
  fingerprints include it. Trait-aware real rules exactly cover `10311120` and
  `10511120` reacting to any Fairy-trait follower rather than only a card named
  Fairy; exact `10402110` uses the ending player's event snapshot for own-turn
  healing, and `10632110` heals once for each named follower actually summoned.
  A second exact batch (`10122110`, `10122120`, `10122130`, `10123140`) covers
  Soldier-Trait listeners and multi-operation `event_source` buffs. Temporary
  `until_end_of_turn` modifiers now expire for the actual active player at the
  triggering boundary, matching the official `10122110` Q&A for soldiers that
  enter during the opponent's turn. A cross-class exact batch adds named-Bat
  Storm/self-damage for `10151110`, Ocean-Trait leader healing for `10541120`,
  and Artifact-Trait healing plus evolve summon for `10771110`;
- field-backed `土之印` stacks and structured `土之秘术` payment, including
  Sigil entry/merge/depletion, effect-destroy protection, opposing manual-target
  protection, generated `大地之魔片`, nested post-payment operations, and
  controller/opponent count conditions and expressions;
- command-level `策动` for field amulets, with structured activation costs,
  once-per-amulet-per-turn state, required-target prevalidation, pending-choice
  revalidation, source-leaves-play safety, and explicit `amulet_activated`
  events that structured emblems can observe. Generic `reduce_countdown` clamps
  at zero and expires an amulet through the normal death/Last Words pipeline.
  Exact real rules cover `10031210` adding an Earth Sigil, `10161210` paying 1
  PP to advance its countdown, and `10563210` destroying itself before cycling
  two seeded-random hand cards and drawing two;
- structured `信仰` leader-area state created from the initial deck without
  removing physical card copies, with same-name deduplication, stable identity,
  deterministic fingerprints, and public value-change events; the first
  verified trigger counts both normal and super evolution for real card
  `10614120`. Its Fanfare now atomically pays 10 Faith and generates token
  `90014330` with auditable origin, then grants a stacking evolution trigger
  that deals 1 damage to the opposing leader after Faith progression;
- generic `consume_faith` costs identify a stable Faith instance, require the
  full value without clamping, emit explicit success/failure diagnostics, and
  queue nested payoff operations only after an atomic successful payment;
- generic Faith abilities store structured triggers and operations on the
  stable leader-area instance, support explicit unique/stacking policies,
  resolve in Faith-placement then grant order, preserve pending-choice event
  continuations, and participate in fingerprints, invariants, and diagnostics;
- Faith progression also accepts owner-scoped amulet-destruction events in
  active-player death-batch order. Exact `10664120` owns a separate persistent
  Faith, advances it when its controller's amulets are destroyed, and at turn
  end atomically pays 10 to generate `90064320` with token origin while present.
  Bound selected-board snapshots retain auditable owner/type metadata after a
  target leaves play; generated `90064320` uses that snapshot to destroy any
  selected board card, damage only for an allied amulet, and replace itself;
- generic effect-caused normal evolution selects an unevolved follower, spends
  no EP or once-per-turn manual evolution allowance, updates public evolution
  counters/gauges, and emits the normal evolution event with evolve abilities;
  generated spell `90014330` now uses this primitive exactly;
- generic signed maximum-PP changes clamp the configured 0–10 range and clamp
  current PP after reductions, emit requested/applied resource diagnostics,
  and immediately update derived Overflow state; exact `10042310` raises max
  PP and conditionally draws after reaching 10;
- authored evolve, super-evolve, attack, and clash rules dispatch even when the
  normalized source text omitted the corresponding keyword tag, without
  double-running tagged abilities and while still respecting ability removal;
  exact `10143120` auto-evolves at Overflow before its evolve rule raises max PP;
- command-level `融合` from hand, including structured material filters and
  count limits, variable-count selection with explicit confirmation, once-per-
  card-per-turn tracking, atomic hand-zone revalidation, a distinct consumed
  material zone, inherited material identity, and `card_fused` events; real
  card `10213310` demonstrates an exact Elf-material fusion rule and a play
  effect that draws two cards after fusion instead of one;
- ordered post-Fusion hand transforms can branch on cumulative material count,
  cost, distinct card identities, and all/any material filters. Transform is
  atomically prevalidated, preserves stable hand identity and nested material
  lineage by default (or explicitly resets it), resets per-card runtime state,
  adopts the replacement's cost/type/passives/rules/play modes, and may fuse
  again when the replacement has its own definition. Exact `10171110`
  generates Past Core; Past/Future Core and their Castle/Attack Artifact forms
  demonstrate Artifact filters and cumulative-cost transformation;
- structured `瞬念召唤` at turn start before the normal draw, with persistent
  match evolution counts, seeded random candidate ordering weighted by copies,
  one copy per card definition per timing, board-full handling, summon-event and pending-choice
  continuations, and explicit `card_invoked` events; real card `10404110`
  invokes after six evolutions, gains its countdown crest, and returns to hand;
- structured `奥义` / `解放奥义` definitions with fixed 10/15 thresholds,
  per-hand-card evolution acceleration, deterministic activation events, and
  seeded random enemy-follower-or-leader targeting that revalidates between
  repeated hits; real `10404110` now resolves all five 2-damage hits exactly;
- follower healing for selected, random, and all-unit target flows, used by the
  invoked Sandalphon crest to heal all allied followers alongside the leader;
- structured `target_exists` no-target branches that reuse normal target
  candidate generation before queuing a then/else effect branch, including
  unit-or-leader fallback targets when no target-dependent condition is present;
- selected board targets support explicit dynamic source exclusion across
  follower/amulet mixed zones and multi-target choices. The same filtered
  candidate set drives play legality, commands, pending choices, revalidation,
  and RL masks; `10664120` demonstrates its verified three-other-card Fanfare;
- structured `grant_attacks_per_turn` capacity preserves attacks already used,
  supports permanent and turn-scoped grants, does not bypass summoning
  sickness, refreshes at turn start, and is removed by ability removal or
  transform. Remaining capacity drives command legality and RL masks. Exact
  `10162120` deals 1 leader damage on clash and gains two attacks per turn on
  super evolution;
- exact `10011130` now uses the normal post-play Combo count to evolve itself
  on a Combo-3 Fanfare without spending EP, then heals its leader for 2 on
  each attack;
- exact `10214110` summons Fairy and gains its emblem on Fanfare; its Evolve
  choice now transforms an enemy follower into Fairy while preserving stable
  entity identity and recording transformed origin, and safely skips when no
  enemy follower exists;
- countdown amulets, explicit last words, fanfare/play rules, attack/clash,
  evolve/super-evolve, turn-start/turn-end triggers, and trigger continuations
  that can pause for choices. Exact `10713110` uses a source-in-play turn-end
  rule to draw only when the ending player's current Combo count is at least 3;
- death-batch event diagnostics that expose the active-player-first,
  left-to-right order used by destroyed, left-play, and Last Words lifecycle
  events, including follower/amulet composition for mixed death batches;
- `death_batch_end` emblem triggers that fire after a death batch's Last Words
  complete, with any new deaths collected into a later death batch;
- recursive resolution-loop diagnostics for events, effects, death batches,
  active card-listener/emblem batches, recent listener triggers, and suspended
  continuations;
- partial higher-level mechanics and primitives for cooperation, `觉醒`, `连击`,
  necromancy, reanimate, spellboost-style hand cost changes, emblems, optional
  decisions, choose-one decisions, play modes, and runtime modifiers.

The RL adapter keeps the original fixed 111-action space and 290-feature public
observation as the default `observation_version="v1"`, together with an action
mask, terminal reward, graveyard choice paging, special
hand actions for fusion/play modes, and super-evolve actions. `info()` is public by default and
redacts debug transcripts/events unless `debug_info=True` or
`info(debug=True)` is used, including pending-choice and graveyard-page returns.
Public observations and default info are regression-tested not to depend on
opponent hand identity or deck identity/order while a real-card pending choice
is awaiting resolution. The public observation includes explicit
controller/opponent `觉醒` flags derived from maximum mana and public
controller/opponent `连击` counts for the current turn, plus pending multi-target
choice size and progress.

An opt-in `observation_version="v2"` returns a structured, fixed-shape mapping
for full-card training without changing any action IDs. Callers should pass a
stable catalog-wide `card_vocabulary`; index 0 is reserved for padding or an
unknown card. V2 adds categorical own-hand and public-board identities, initial
deck-composition and public graveyard/banished histograms, origin and runtime
modifier features, board keyword bits, Faith/emblem identity and values,
parameterized-choice references, the legal action mask, and a bounded public
event history. It never emits raw entity IDs, opponent hand identity, fusion
materials hidden in the opponent hand, or remaining deck order. The default
derived vocabulary is convenient for small fixtures, but production training
must configure one shared vocabulary so shapes and indices are stable across
matches. `observation_v2_spec()` exposes every shape and categorical ordering;
`recurrent_observation()` supplies the same public v2 input for a caller-owned
recurrent or belief state. Existing v1 consumers require no migration. V2
consumers select the version explicitly and replace scalar card features with
categorical embeddings while retaining `continuous_v1` during transition.
Each public board slot appends Intimidate and Aura flags, migrating the
observation from 270 to 280 and then 290 features without changing action IDs;
attack and selected-effect mask entries come from the same command legality as
`GameEngine`.
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
Activate reuses the evolution action slot for an amulet in the same board
position, because followers and amulets are mutually exclusive there. Its
current-turn usage flag occupies an existing public amulet board feature, so
it does not expand the action space or amulet feature width. This follows the
[official Activate glossary](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10114110):
a field amulet can activate once per turn, and a specified cost is paid only
when enough PP remains.
The observation also exposes both players' public number of follower evolutions
this match. Invocation itself is automatic and therefore adds no RL action.
The current implementation follows the
[official Skybound Dragons mechanic overview](https://shadowverse-wb.com/chs/cards/pack/skybound-dragons/)
and [official card glossary](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10404110):
Sandalphon is the unique Invoke card; Invoke enters from the deck when its
condition is met, orders simultaneous candidates randomly, and limits duplicate
copies to one while letting their copy count affect selection probability.
The final four observation features expose both players' public Faith counts
and total Faith values. This follows the
[official Faith glossary](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10723110):
an initial-deck Faith is placed in the leader area at battle start, same-name
Faiths do not duplicate, and each Faith has its own visible value.
Each visible own-hand slot also exposes a normalized `奥义` gauge only when
that card has a structured Union Burst definition. The gauge is the current
player turn number plus evolutions completed while that specific card remained
in hand; entering hand resets its evolution contribution. This adds nine
features without changing the 111-action layout.
Automatic super evolution adds no RL action: its public board state reuses the
existing evolved/super-evolved features, while the unaffected follower slots
retain their normal manual super-evolution actions. Manual SEP evolution now
resolves `进化时` before `超进化时`, preserving pending choices and intervening
`follower_evolved` emblem listeners. These rules follow the
[official super-evolution overview](https://shadowverse-wb.com/chs/system/cardbattle/battle/)
and the [official Olivia Q&A](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10104110).

## Unsupported Or Partial

The engine still does not model the full SWB ruleset. Unsupported behavior must
remain visible instead of silently behaving as implemented.

Known broad gaps include:

- exact semantics for many real cards and most generated-card workflows; the
  token audit keeps the remaining 11 partial and 70 unimplemented entries
  explicit instead of treating non-collectible classification as coverage;
- remaining `信仰` progression/payoff semantics, plus broader real-card coverage for `策动`,
  `土之秘术`, `觉醒`, and `连击` beyond the currently authored examples;
- ordinary board, hand, and leader-area listeners now receive
  `amulet_activated` and `card_fused`; remaining cards that use those events
  still need individual structured rules and official-text verification;
- Faith currently supports the verified `follower_evolved` progression trigger
  for normal and super evolution, owner amulet-destruction progression, atomic
  value spending, and dynamically gained structured abilities. Mode selection,
  Enhance play, named-follower entry, and the shared five-slot leader-area
  limit remain explicit unsupported edges; other generated cards remain
  individually classified in the token/generated-card audit;
- Fusion-driven hand transforms and refusion are implemented. Other cards can
  listen to `card_fused`, but their individual reactions and later generated
  Artifact end-form abilities still require audited structured rules;
- additional `奥义` cards still require explicit structured definitions; the
  generic gauge, thresholds, activation event, and repeated random target flow
  are implemented, while unsupported card-specific clauses remain visible;
- additional cards that cause normal evolution, restore SEP, or super-evolve
  multiple/selected followers still need their own structured rules; real
  `10443110` now exactly covers both its `奥义` self-super-evolution and its
  cost-2-follower Ward listener;
- source-backed continuous stat or keyword derivation remains unsupported;
  source-backed leader damage modifiers are implemented and revalidate entry,
  leave, transform, and control changes;
- remaining trigger-ordering edge cases beyond the current death-batch
  ordering diagnostics and `death_batch_end` boundary triggers, including
  unsupported `death_batch_start` emblem triggers, plus broad real-card
  coverage audits;
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
