# SWB Engine Roadmap

Last refreshed: 2026-07-10.

This file tracks implementation priorities and known gaps. Treat executable
code and tests as the source of truth when this file drifts.

## Current Baseline

- Database: 826 cards, 735 collectible cards, 91 non-collectible/generated
  cards from set `90000`.
- Latest SVA source: `https://sva.hypd.asia/data/cards.json`.
- Tests: `python -m unittest discover -s tests -v` currently runs 937 tests.
- RL adapter: fixed 111-action space and 257-feature observation.
- Ability registry status: 14 implemented, 4 partial, 16 placeholder.
- Explicit card and demo rules live in `data/rules/`; the current coverage
  report classifies 97 card IDs with explicit rules, passives, fusion, or
  invocation definitions.

## Stable Priorities

Unless the user gives a different priority, choose the next coherent vertical
slice in this order:

1. Harden correctness invariants: illegal-command no-mutation coverage, entity
   uniqueness, zone ownership, pending choice consistency, and deterministic
   replay checks.
2. Protect RL public information boundaries: default `Env.info()` and
   observations must not leak hidden hand identity, deck order, transcripts, or
   debug-only events.
3. Implement core class/mechanic conditions one slice at a time. The `觉醒`
   and `连击` primitive slices are in place; targeting edge cases are next.
4. Expand targeting and effect edge cases: multi-target choices, no-target
   branches, duplicate-target policy, source/target leaving play, and zone
   changes during suspended effects.
5. Stabilize trigger semantics: ordering, simultaneous deaths, last words,
   pending choices during triggers, and recursive trigger loop diagnostics.
6. Finish remaining super-evolution edge semantics, class resources, and class
   mechanics as official text or real-card rules require them.
7. Expand real-card coverage only after the necessary generic primitive exists
   and has focused tests.

## Implemented Or Substantially Covered

- Deck validation for 40 collectible class/neutral cards.
- Deterministic reset, shuffle, draw, turns, mana, hand/board limits, and deck
  exhaustion damage.
- Followers, spells, amulets, combat, evolution, manual super-evolution, and
  shared board slots.
- Manual super-evolution protection uses the actual turn player rather than
  effect controller, so opponent-controlled triggers during the owner's turn
  cannot bypass same-turn effect damage/destruction prevention. Combat damage
  is not prevented, and protection is stamped to the exact turn the follower
  super-evolved so later own turns do not retain it. Real `10012120`
  super-evolve flows cover both combat damage after its structured trigger
  resolves and effect damage on a later own turn.
- Core combat keywords: `守护`, `疾驰`, `突进`, `必杀`, `潜行`, `吸血`, `屏障`.
- Target candidate generation for board, leader, hand, and graveyard choices.
- `all_board` target support for effects that need one simultaneous candidate
  set across both players' followers and amulets.
- Choice suspension/resume for targeted effects and several trigger paths.
- Pending board and graveyard choices validate stale entity IDs against the
  requested zone before resolution resumes.
- Stale board-target choices are covered through ordinary effects,
  choose-one branches, turn-end trigger continuation, and emblem continuation.
- `requires_target` JSON is limited to targets with explicit candidate sets;
  implicit, previous-target, and unit-or-leader fallback targets are rejected
  as ambiguous.
- True multi-target choices support `target_count`, `target_count_expr`, and
  explicit `allow_duplicate_targets` / `allow_duplicates` policy in normal,
  nested, emblem-trigger, and emblem-expiration operations. Choices accumulate
  through deterministic `Choose` commands, reduce to the available distinct
  candidate count when duplicates are forbidden, and revalidate each selected
  target before execution. The fixed RL action space exposes each selection
  through the existing choice actions and adds public count/progress features.
  Real card `10351120` demonstrates selecting and destroying two enemy
  followers before its self-damage resolves.
- `select_targets` can bind an ordered selected board-entity tuple to one
  `target_key`; later `previous_target` operations reuse that set in selection
  order and revalidate every member against the original target specification.
  Missing runtime bindings caused by a no-candidate branch skip safely. Partial
  real rule `10474120` demonstrates selecting two enemy followers and applying
  later damage to the same set without claiming support for its ability-loss or
  leader-damage-amplification clauses.
- `土之印` is modeled on `Amulet.earth_sigil_count`, not as a player-side
  integer. A Sigil enters at 1, banishes and merges all other friendly Sigils
  into the newest entity, cannot be destroyed by abilities, cannot be manually
  selected by opposing abilities, and is destroyed through the normal
  death-batch pipeline when its count reaches 0. Nonselecting banish remains
  legal. `add_earth_sigils` increments the existing stack or creates token
  `90031210`; a full board with no existing stack explicitly skips creation.
- `earth_rite` pays a positive structured cost only when the full amount is
  available, emits public resource/activation events, and queues nested effects
  after payment and any depletion death batch. It supports pending choices
  without repeat payment. Controller/opponent Sigil conditions, expressions,
  invariant checks, deterministic fingerprints, and public RL totals are
  covered. Exact real rules `10032310` and `10732120` demonstrate consume/gain;
  `10031210` remains partial solely because `策动` is not implemented.
- `BeginFusion` exposes Fusion before RL action encoding. A structured
  `fusions` definition filters eligible hand materials and sets optional
  minimum/maximum counts. The pending choice supports variable-count selection
  and explicit confirmation, revalidates the source card and every selected
  material before one atomic transition, and preserves illegal-command
  fingerprints when a card leaves hand. Each Fusion card can fuse once per
  turn; multiple Fusion cards may each do so.
- Consumed Fusion materials are retained in a distinct identity-bearing zone,
  do not enter the graveyard or increase shadows, and keep nested material
  relations if a previously fused card is consumed. Hand and board cards carry
  their material IDs through play and return-to-hand transitions. Structured
  source fusion-count conditions are available to normal and nested effects,
  and `CARD_FUSED` makes the transition auditable. The fixed 111-action layout
  reuses existing special-hand and choice slots; the 257-feature observation
  exposes own-hand and public-board Fusion state without revealing the
  opponent's hidden hand.
- Exact real rule `10213310` (`花园的指引`) accepts Elf-class Fusion materials
  and draws two cards if it was fused, otherwise one. These core semantics
  follow the official Fusion glossary; hand transformation and other-card
  Fusion-event triggers remain explicit unsupported edges.
- `InvocationDefinition` provides deck-active `turn_start` conditions and the
  `invoke` trigger. At the timing boundary, candidate instances are snapshotted,
  seeded RNG selects their order with duplicate copies contributing their proper
  probability weight, and one copy per card ID can enter. Conditions and board
  space are revalidated, and eligible followers enter from
  deck before the normal draw without spending mana, counting as played, or
  firing Fanfare. `CARD_INVOKED` and `FOLLOWER_SUMMONED` (`via=invocation`) make
  the transition auditable.
- Invocation participates in normal summon-event ordering and can suspend for
  an event or on-invoke target choice before the remaining candidates and normal
  draw resume. Full boards leave candidates in deck; malformed non-follower
  definitions remain visible through Invocation placeholder reporting instead
  of silently executing. Match evolution counts include normal and super
  evolution, are fingerprinted/invariant-checked, and are public in the
  257-feature observation without changing the 111 actions.
- Real `10404110` (`天司长的继承者·圣德芬`) is the current unique official
  Invocation card. Its six-evolution condition, deck summon, countdown-2 crest,
  return to hand, and crest turn-end healing of all allied followers and leader
  are structured and tested. The new `heal_unit` primitive emits actual-heal
  events and clamps to maximum health. The card remains `covered_partial` only
  because its Fanfare `解放奥义` damage sequence awaits the Union Burst slice.
- `target_exists` provides structured no-target branch semantics for explicit
  candidate-set targets and unit-or-leader fallback targets, including board
  target-dependent filters, then/else effect branches, and existing RL
  choice-mask/decode behavior when the chosen branch asks for a target. Real
  cards `10012310` and `10153310` demonstrate the random-target and
  unit-or-leader paths.
- Source-dependent effect operations now safely skip if their source board
  entity leaves play before resolution resumes.
- Generic effects for damage, healing, draw, summon, destroy, banish, return,
  discard, transform, stat changes, keyword changes, cost changes, and attack
  or targeting restrictions.
- Countdown amulets, last words, fanfare/play rules, attack/clash,
  evolve/super-evolve, turn-start/turn-end triggers, and emblem triggers.
- Simultaneous-death tests now cover same-batch Last Words ordering, pending
  choices during Last Words, stale Last Words choice targets, and defer deaths
  caused during a Last Words batch into the following death batch.
- Death-batch diagnostics now attach `batch_id` metadata to destroyed,
  left-play, and Last Words start/complete events; mixed follower/amulet death
  batches are covered before Last Words resolution begins.
- Death-batch diagnostics now also expose `active_player`,
  `batch_order_index`, `batch_record_count`, and a start-event
  `ordered_records` summary so active-player-first, left-to-right destroyed
  and Last Words ordering is auditable across synthetic and real-card tests.
- Mixed follower/amulet death-batch diagnostics expose follower/amulet totals
  and per-owner composition, including a real-card demonstration with
  `10052110` and `10161210`.
- `follower_destroyed` emblem triggers are supported from structured JSON and
  can pause for a pending choice before same-batch Last Words continue; multiple
  destroyed-event emblems, including cross-player `any_event` triggers for
  multiple destroyed followers, finish before that death batch's Last Words
  begin. `EMBLEM_TRIGGERED` is recorded when the trigger is accepted, so
  diagnostics show the trigger before its suspended choice effect.
- `amulet_destroyed` emblem triggers are supported from structured JSON and
  fire from the same death-batch event stream as amulet Last Words. Pending
  amulet-destroyed choices resolve before the destroyed amulet's Last Words,
  no-target trigger operations skip cleanly, and the real `祥和的教会`
  countdown-expiry rule demonstrates the ordering.
- `death_batch_end` emblem triggers are supported from structured JSON and fire
  only after a death batch's Last Words finish; their effects can create later
  death batches without re-entering the completed batch. Trigger diagnostics
  carry the originating death `batch_id`.
- Recursive resolution loops are bounded by a step limit and raise
  `ResolutionLoopError` with structured, deterministic diagnostics for recent
  events, queued events, effect frames, death batches, active emblem batches,
  recent emblem triggers, suspended continuations, and log tail.
- Runtime invariant checks validate pending-choice shape, target/leader choice
  identity, and effect-stack frame structure, including operation payloads,
  source-card identity, target-id sentinels, and emblem activation/expiration
  field consistency. Illegal-command no-mutation tests snapshot suspended
  action, event, death, emblem, and spellboost internals.
- Deterministic replay diagnostics expose a full-state engine fingerprint,
  including hidden deck order and RNG state, for same-seed command replay tests.
- Illegal-command no-mutation coverage uses the full-state fingerprint,
  including a real-card pending-choice path, so invalid commands preserve
  suspended effects, RNG, hidden zones, and diagnostics.
- Special play modes: `爆能强化`, `激奏`, and `结晶`.
- Partial primitives and demos for cooperation, `觉醒`, `连击`, necromancy, reanimate,
  spellboost-style costs, emblems, optional decisions, and choose-one decisions.
- RL action mask, action decoding, public observation, terminal reward, pending
  choices, graveyard paging, special play modes, and super-evolve actions.
- Public `Env.info()` uses a small redacted key set by default, including when
  reset/step return info during pending choices; debug transcripts, event
  objects, pending-choice labels/entity IDs, player objects, and deck lists are
  excluded unless explicitly requested through debug info where applicable.
  Real-card pending-choice observations/default info are covered against
  opponent hidden hand/deck identity changes, and graveyard page-turn info
  redacts option labels/entity IDs.
- Illegal RL actions restore environment choice-page bookkeeping as well as
  preserving core state, logs, events, RNG, and suspended engine internals.

## Known Partial Or Unsupported Areas

- `觉醒` has condition/expression support, public observation flags, and one
  real-card rule demo; broader real-card coverage and future max-mana ramp
  interactions remain partial.
- `连击` has natural per-turn counting, condition/expression support,
  `add_combo`, public observation counts, and real-card demos; broader real-card
  coverage remains partial.
- `策动`, `信仰`, and `奥义` remain placeholder or incomplete.
  `土之秘术` / `土之印` core semantics are implemented, while broader
  real-card coverage remains intentionally incremental.
- `融合` has the command-level material transition, state, event, structured
  filters/counts, source fused-count conditions, RL exposure, and one exact
  real-card demo. It remains partial because Fusion-driven hand transformations
  and triggers owned by other hand/board cards are not yet modeled.
- `瞬念召唤` is implemented for its sole current official card and marked
  implemented in the ability registry. Sandalphon as a whole stays partial
  until `解放奥义` is implemented; this does not downgrade the Invocation
  transition itself.
- The ability registry is conservative; a generic primitive can exist before a
  keyword is marked fully implemented.
- Many real cards are intentionally uncovered or only partly covered by
  structured rules.
- Non-manual super-evolution edge semantics remain pending.
- Trigger ordering and simultaneous-death edge cases beyond the current
  death-batch ordering and mixed-composition diagnostics need broader coverage.
- Emblem event trigger coverage is intentionally limited to explicit trigger
  names; `death_batch_start` emblem triggers remain unsupported until ordering
  semantics are specified by a real rule.
- Source-leaves-play handling currently covers `target: self` board operations,
  `source_attack`/`source_health` expressions, source-specific conditions, and
  source-dependent multi-target counts. Multi-target bindings preserve their
  selected identity when a source leaves, while source-dependent later
  operations still skip through the normal source revalidation path.
- `target_exists` intentionally rejects implicit and previous-target targets;
  target-dependent conditions are only supported for board candidates, so
  leader fallback targets do not satisfy target-specific predicates.
- Target-leaves-play handling covers ordinary board choices, graveyard choices,
  choose-one branches, a turn-end trigger choice, and emblem continuation where
  the selected board entity moved to the graveyard. Pending board/hand/graveyard
  choices are also revalidated against their current candidate set before
  resolution resumes, so targets that changed controller or stopped matching the
  original filter skip cleanly instead of resolving against the wrong object;
  ordinary play choices include real-card coverage for a selected target moving
  to the graveyard, fingerprint no-mutation for invalid choices after that zone
  change, synthetic coverage for a selected target changing controller, real-card
  hand-target coverage for a selected card leaving hand, and graveyard-target
  coverage for a selected graveyard card moving to hand before resolution.
  `previous_target` target-key chains retain the original binding operation and
  revalidate its ordered single- or multi-target tuple against that operation's
  current candidate set before later operations resolve, including targets that
  left play, changed controller, or no longer match the original board filter.
- Coverage tooling should continue distinguishing covered text from
  unsupported text; never hide unsupported card text behind `covered_exact`.

## Next Coherent Slices

### 1. Activate

- Model `策动` as its own command against an eligible amulet, including costs,
  once-per-turn or countdown semantics only where verified by real text, stale
  entity revalidation, action-mask agreement, and one real-card demo.

### 2. Faith / Union Burst

- Implement `信仰` and `奥义` as separate command/state/rule slices in that
  order, each tied to verified real text and one real-card demo. Completing
  Union Burst should promote the remaining Sandalphon Fanfare clause.

### 3. Targeting Edge Cases

- Extend no-target branch coverage only when real cards need graveyard/hand
  target-dependent filters or additional fallback target semantics.
- Audit real rules that combine selected hand/graveyard sets with later
  operations before expanding `target_key` beyond board-entity tuples.

### 4. Trigger Loop Diagnostics

- Add broader trigger ordering tests around any future `death_batch_start`
  boundary semantics and real-card recursive trigger combinations as coverage
  expands.

### 5. Super-Evolution Edge Semantics

- Keep non-manual super-evolution unsupported until official text or a real
  structured rule needs it; add it as a separate vertical slice with command,
  event, protection, and RL coverage.

### 6. Coverage Reporting

- Refresh rule coverage reports after database updates.
- Make reports surface newly added cards and newly unsupported keyword text.
- Keep real-card expansion tied to implemented generic primitives.

## Verification Policy

For every implementation slice, run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

For changes affecting legal actions, resolution, cards, targets, combat, turns,
or observations, also run:

```powershell
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

Use 1000 self-play games for broad shared-engine changes.
