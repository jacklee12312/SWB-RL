# SWB Engine Roadmap

Last refreshed: 2026-07-14.

This file tracks implementation priorities and known gaps. Treat executable
code and tests as the source of truth when this file drifts.

## Current Baseline

- Database: 826 cards, 735 collectible cards, 91 non-collectible/generated
  cards from set `90000`.
- Latest SVA source: `https://sva.hypd.asia/data/cards.json`.
- Tests: `python -m unittest discover -s tests -v` currently runs 1234 tests.
- RL adapter: fixed 111-action space; the default v1 observation remains 290
  floats, while opt-in v2 provides fixed-shape categorical/public state without
  changing action IDs.
- Ability registry status: 18 implemented, 5 partial, 11 placeholder.
- Explicit card and demo rules live in `data/rules/`; the current coverage
  report classifies 145 card IDs with explicit rules, passives, fusion,
  invocation, activation, Faith, Union Burst, or listener definitions. Current
  collectible coverage is 113 exact, 0 partial, 604 supported-but-missing-rule,
  0 missing-primitive, and 18 text-unclear cards.
- `data/reports/token_audit.{json,md}` audits all 91 non-collectible/generated
  cards independently of collectible coverage. Database `card_references` and
  executable structured producers are reported separately: 11 tokens have a
  complete executable entry/behavior path, 11 have an executable entry but
  partial behavior, and 69 have no authored producer. No token is currently
  classified text-unclear or externally blocked; those categories remain
  explicit and can be assigned through `data/audits/token_overrides.json`.
- `data/reports/ability_audit.{json,md}` validates all 34 runtime/database
  registry entries against `data/audits/ability_registry.json`. Every status
  has a precise conservative reason and test evidence. Primitive support is a
  separate column and is covered for all 34; it does not automatically promote
  the 5 partial or 11 placeholder keyword statuses.
- Coverage now has a backward-compatible clause audit. The legacy summary still
  reports 113 exact collectible cards, and all 113 now have an explicit exact
  text mapping plus named direct test evidence. The versioned sibling registry
  at `data/audits/rule_clauses.json` hashes every imported primary and
  alternate-mode clause; changed source text or stale test evidence invalidates
  the audit. Synthetic `999xxx` rules are counted but are not ordinary
  consistency failures. The report includes source import hash/count/timestamp,
  rule version and errata metadata, structured trigger/operation evidence,
  explicit unsupported text, and a stable blocker taxonomy. There are zero
  unverified exact entries and no known missing schema, primitive, targeting,
  or timing blocker among authored rules. The 604 supported-but-missing-rule
  cards are now the per-card structured-content backlog; 18 unclear texts
  remain explicit.
- RL observation v2 is available behind an explicit version switch. A
  configured card vocabulary supplies stable categorical indices for own-hand,
  public-board, initial-deck composition, and public graveyard/banished state;
  runtime modifier, keyword, origin, transform/Fusion, Faith/emblem, choice,
  action-mask, and bounded public-history inputs have fixed shapes. Index 0 is
  padding/unknown. Privacy tests lock out opponent hand identity, hidden Fusion
  state, raw entity IDs, and remaining deck order. V1 stays the default at
  111 actions and 290 floats, and the recurrent/belief-state interface leaves
  model hidden state under caller ownership.

## Stable Priorities

Unless the user gives a different priority, choose the next coherent vertical
slice in this order:

1. Harden correctness invariants: illegal-command no-mutation coverage, entity
   uniqueness, zone ownership, pending choice consistency, and deterministic
   replay checks.
2. Protect RL public information boundaries: default `Env.info()` and
   observations must not leak hidden hand identity, deck order, transcripts, or
   debug-only events.
3. Extend targeting only when a verified real rule needs a new hand,
   graveyard, fallback, or chained-target boundary; true multi-target choices,
   duplicate policy, and suspended-choice revalidation are in place.
4. Add new trigger/death boundaries only when a verified real rule requires
   them; simultaneous deaths, pending trigger choices, and bounded recursive
   trigger diagnostics are covered.
5. Extend Earth Rite, Fusion, Invocation, Activate, Faith, and Union Burst one
   real-card slice at a time without weakening their explicit unsupported
   clauses.
6. Extend super-evolution effects only for verified real-card variants; manual
   and single/all-follower effect semantics, protection, event ordering, and
   attack-destruction bonus behavior are covered.
7. Expand real-card coverage only after the necessary generic primitive exists
   and has focused tests.

## Implemented Or Substantially Covered

- Deck validation for 40 collectible class/neutral cards.
- Deterministic reset, shuffle, draw, turns, mana, hand/board limits, and deck
  exhaustion damage.
- Followers, spells, amulets, combat, evolution, manual and effect-caused
  super-evolution, and shared board slots.
- Manual super evolution spends SEP, adds `+3/+3`, grants follower-only attack
  access, and emits `FOLLOWER_EVOLVED` before `FOLLOWER_SUPER_EVOLVED` so
  `进化时` completes before `超进化时`. A choice in the first trigger suspends
  the second event, while `follower_evolved` emblem listeners finish between
  them. Normal evolution remains `+2/+2`.
- Super-evolution protection uses the actual turn player rather than effect
  controller. Every own turn of a super-evolved follower prevents all damage,
  including combat damage, and effect destruction; opponent turns remain
  unprotected. `super_evolved_turn` is retained as an audit timestamp rather
  than an expiry condition. Real `10012120` covers combat protection and later
  own-turn effect protection.
- A super-evolved follower that attacks and destroys the attacked follower
  deals 1 damage to the enemy leader. The attack context survives pending
  trigger choices and recognizes both combat destruction and destruction by an
  attack-time ability, while return/banish/nonlethal paths do not fabricate the
  bonus. Context state is invariant-checked, fingerprinted, and included in
  loop diagnostics.
- `super_evolve_unit` is a structured, follower-only effect. It applies the
  complete super-evolution stats/protection/attack state, increments match and
  hand-gauge evolution counts, and leaves SEP/manual-turn availability intact.
  Official Q&A semantics are explicit: effect-caused super evolution does not
  fire `进化时` or `超进化时` keyword abilities. Real `10443110` demonstrates
  `奥义` self-super-evolution plus an exact structured cost-2-follower Ward
  listener. RL reuses existing public board flags and manual action slots.
- Core combat keywords: `守护`, `疾驰`, `突进`, `必杀`, `潜行`, `威慑`, `灵气`,
  `吸血`, `屏障`.
- `威慑` removes that follower from opposing follower attack targets without
  affecting ability targets. It takes precedence over Ward on the same
  follower, while other visible Wards remain mandatory. Legal commands, direct
  command validation, and RL masks share one target predicate; add/remove,
  temporary removal, evolution, super evolution, and transform reuse normal
  runtime-keyword lifecycle. A public Intimidate bit per board slot migrates
  observation width from 270 to 280 while preserving all 111 action IDs.
  Structured `non_intrinsic_keyword` annotations prevent conditional/random
  card-text mentions from becoming initial keywords. Real `10451120` exactly
  covers static Intimidate and its summon/self-damage Last Words, based on the
  official help glossary and card page.
- `灵气` removes followers and amulets from opposing manually selected ability
  targets, while controller-owned selections, random effects, all-target
  effects, and attacks remain unaffected. The normal target-candidate path
  drives play legality, pending-choice revalidation, and RL masks. Dynamic
  add/remove, temporary removal, and transform reuse runtime keyword state;
  intrinsic Aura amulets use their normalized ability definition. A public
  Aura bit per board slot migrates observation width from 280 to 290 without
  changing any of the 111 action IDs. `non_intrinsic_keyword` annotations
  prevent effect-granted/random full-text mentions from becoming initial Aura.
  Real `10161140` is exact through static Aura plus an `amulet_activated`
  listener that stacks `+1/+0` until turn end. These semantics follow the
  official help glossary.
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
- `CardListenerDefinition` provides ordinary board, hand, and leader-area event
  listeners for amulet activation, Fusion, follower summon/evolution/
  destruction, amulet destruction, entity leave-play, card play, and turn
  boundaries. Event-card filters cover original cost, card type, class, Trait
  ID/name, card ID/name, and runtime keyword; event/turn scopes and self/other
  relation are explicit;
  `once_per_turn` and `max_activations` are fingerprinted and invariant-checked.
  Listener snapshots use active-player-first then board/hand/leader-area stable
  ordering, revalidate later sources, preserve nested `event_source` identity,
  and resume remaining listeners/emblems after choices. Loop diagnostics expose
  active batches and recent accepted listener triggers. Real `10443110` is now
  `covered_exact`; a structured `non_intrinsic_keyword` passive distinguishes
  its conditionally granted Ward from the database's full-text keyword audit.
  `CardDefinition` now retains normalized database Trait identity and includes
  it in deterministic fingerprints. Exact real cards `10311120` and `10511120`
  demonstrate Fairy-Trait listeners, while `10632110` demonstrates a named-card
  listener. Turn-start/end listener scope and ordering use the event player's
  timing snapshot rather than the already-switched active player; exact
  `10402110` covers own-turn-end all-follower/leader healing.
  Exact Royal cards `10122110`, `10122120`, `10122130`, and `10123140` cover
  Soldier-Trait listeners, named token summons, and multiple ordered
  `event_source` operations. `until_end_of_turn` modifier expiry uses the
  active player at effect resolution rather than assuming the effect
  controller is active, matching the official opponent-turn Q&A for
  `10122110`. Exact cross-class cards `10151110`, `10541120`, and `10771110`
  cover named-card event filters, Ocean-Trait healing, Artifact-Trait healing,
  listener-granted Storm, and evolve-triggered generated-card summon.
- `select_targets` can bind an ordered selected board-entity tuple to one
  `target_key`; later `previous_target` operations reuse that set in selection
  order and revalidate every member against the original target specification.
  Missing runtime bindings caused by a no-candidate branch skip safely. Exact
  real rule `10474120` selects up to two available enemy followers, removes
  their abilities, damages each remaining valid target, and installs a
  permanent opposing-leader damage-taken modifier.
- `remove_all_abilities` separates removed printed abilities from later runtime
  grants, suppresses board listeners/turn triggers/Last Words, preserves queued
  effects and non-ability state, and resets on transform or re-entry. Generic
  leader damage modifiers stack increases/reductions and support permanent,
  turn-scoped, and source-in-play lifetimes with deterministic source
  revalidation and fingerprint/event diagnostics. The default RL v1 interface
  remains 111 actions/290 observations; these public fields are reserved for
  the later versioned observation migration rather than breaking v1.
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
  exact real rule `10031210` combines Earth Sigil entry with command-level
  `策动` to increment that field-backed stack.
- `ActivateAmulet` exposes `策动` before RL encoding. Structured `activations`
  definitions supply non-negative PP costs and require a paired non-empty
  `activate` trigger rule. An eligible field amulet can activate once per turn;
  legality checks controller, current entity, cost, and required targets before
  any mutation. The paid continuation reuses normal effects and choices, so
  targets that leave or change controller are revalidated without a second
  payment, and source-independent effects can finish after the amulet leaves.
- `AMULET_ACTIVATED` makes activation auditable and is available to structured
  emblem triggers. Deterministic fingerprints and invariants include the
  amulet's activation-turn stamp. RL reuses the same board position's evolve
  slot for amulets and an existing public amulet feature for current-turn use,
  retaining the fixed 111 actions and current observation width. Generic
  `reduce_countdown` clamps at zero and sends expired amulets through ordinary
  death batches and Last Words. Exact real rules now cover `10031210` spending
  1 PP to add one Earth Sigil, `10161210` spending 1 PP to reduce its countdown,
  and `10563210` destroying itself before two seeded-random hand-to-deck moves
  and two draws.
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
  reuses existing special-hand and choice slots; the public observation
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
  public observation without changing the 111 actions.
- `FaithDefinition` creates persistent leader-area `FaithInstance` state from
  the initial deck before opening draws. Physical deck copies remain in the
  deck, repeated copies and same-ID Faith definitions deduplicate by stable
  `faith_id`, and generated cards do not retroactively create Faith. Instances
  carry stable entity/sequence identity, non-negative public values, reset
  deterministically, and emit `FAITH_PLACED` / `FAITH_VALUE_CHANGED` events.
- Faith accepts `follower_evolved`; both normal and super
  evolution increment only the evolving player's matching Faith before later
  event listeners resolve. Real `10614120` (`古旧天枪·萨莎妮德`) starts at 0
  and increments by 1. Its Fanfare atomically spends 10 and generates
  `90014330` with token origin, then grants its Faith a stacking evolution
  trigger that damages the opposing leader after value progression. The full
  source-card text is now `covered_exact`.
- Owner `amulet_destroyed` events provide a second Faith progression trigger
  and preserve active-player-first death-batch order. Exact `10664120`
  (`古旧天书·莲妥丝`) owns a distinct `ancient_tome` Faith, advances it for
  its controller's destroyed amulets, and while present at turn end atomically
  spends 10 to generate `90064320` with token origin. The generated spell
  selects and destroys any board card, then uses the immutable selected-card
  owner/type snapshot to deal 2 leader damage only for an allied amulet before
  adding another token-origin copy to hand. It is unplayable with no board
  target, and illegal play preserves the complete deterministic fingerprint.
- Faith count and aggregate value for both players add four public observation
  features, migrating the fixed observation from 257 to 261 while leaving the
  111-action layout unchanged. The database importer now includes alternate-mode
  text in keyword extraction, so Faith-only alternate modes remain visible in
  normalized `card_abilities` and coverage reports.
- `consume_faith` is a structured atomic cost boundary: it resolves a stable
  `faith_id`, requires the complete value, never clamps below zero, emits
  success or missing/insufficient diagnostics, and only schedules nested
  operations after successful payment. Sasanid's generated-card payoff now
  uses this boundary.
- `grant_faith_ability` stores structured trigger/operation payloads on the
  stable Faith instance with explicit unique or stacking semantics. Matching
  abilities execute in Faith-placement then grant order after normal Faith
  progression; grant and trigger events, fingerprints, invariants, loop
  diagnostics, and pending-choice event continuation cover the dynamic state.
- `evolve_unit` provides effect-caused normal evolution without consuming EP
  or the manual once-per-turn allowance. It updates match/hand evolution
  counters, emits `FOLLOWER_EVOLVED` with `cause=effect`, and resolves normal
  evolve abilities. Generated `90014330` (`天枪深渊`) selects an unevolved
  allied follower and now demonstrates the complete primitive.
- `change_max_mana` applies signed maximum-PP changes within the configured
  0–10 bounds and clamps current PP after reductions. `MAX_MANA_CHANGED`
  records requested/applied deltas and before/after values; Overflow and public
  observation derive immediately from the new maximum. Exact `10042310`
  (`龙之启示`) demonstrates post-ramp conditional draw at 10 max PP.
- Structured evolve/super-evolve/attack/clash rules no longer depend on a
  redundant normalized keyword tag: if a rule exists but extraction omitted
  the tag, event dispatch executes it directly; tagged rules retain their
  existing single handler path and removed abilities remain disabled. Exact
  `10143120` (`荣弦的天宫·龙芙`) demonstrates Overflow-gated effect evolution
  followed by its untagged evolve rule raising max PP.
- `exclude_source` is a structured selected-board targeting policy carried by
  the centralized candidate generator. It works across mixed follower/amulet
  zones and multi-target choices; legality, target-exists, pending choices,
  stale revalidation, fingerprints/invariants, and RL masks share the filtered
  set. Real `10664120` demonstrates selecting three other board cards. Its
  turn-end Faith payment/generated-card clause is now exact through the
  owner-scoped amulet-destruction Faith trigger.
- Selected board bindings retain immutable identity, controller, card type,
  name, and printed-cost snapshots alongside their live entity IDs. A
  `conditional` may explicitly reference a preceding single-target binding;
  ordinary `previous_target` operations continue to revalidate live targets,
  while post-removal clauses can safely inspect the original selection.
- `UnionBurstDefinition` provides structured `奥义` and `解放奥义` operations
  at fixed gauges 10 and 15. Every hand card independently records evolutions
  completed while it remains in hand; both normal and super evolution count,
  and a card entering hand starts with no retained evolution bonus. Invalid
  evolution commands preserve that state and the full engine fingerprint.
- On normal follower play, all satisfied definitions activate in threshold
  order after the summon event and before their effects resolve. The generic
  `random_enemy_unit_or_leader` damage target includes the enemy leader even on
  an empty board and recalculates the follower pool between repeated hits, so
  dead or otherwise removed followers cannot be selected again. Activation is
  auditable through `UNION_BURST_ACTIVATED` and all randomness remains on the
  engine-owned seeded RNG.
- Own-hand Union Burst gauges add one feature per visible hand slot, migrating
  the fixed observation from 261 to 270 without changing the 111 actions.
  Cards without a structured definition expose zero, and hidden opponent hand
  identity remains absent.
- Real `10404110` (`天司长的继承者·圣德芬`) is the current unique official
  Invocation card. Its six-evolution condition, deck summon, countdown-2 crest,
  return to hand, and crest turn-end healing of all allied followers and leader
  are structured and tested. The new `heal_unit` primitive emits actual-heal
  events and clamps to maximum health. Its `解放奥义` now performs five
  independently seeded 2-damage hits against random enemy followers or the
  enemy leader, promoting the complete card rule to `covered_exact`.
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
- Exact file `data/rules/real_basic_spells_batch.json` adds 12 audited real
  spells across Neutral, Forestcraft, Swordcraft, Runecraft, Dragoncraft,
  Abysscraft, and Havencraft. Direct tests cover repeated Token summoning and
  board capacity, draw, unit-or-leader targets, simultaneous all-follower
  deaths, ally/enemy-wide damage, seeded hand cycling and random damage,
  follower/amulet banish, multi-target shortage and stale-target revalidation,
  illegal no-target fingerprints, whole-hand Spellboost, and RL mask parity.
  `10101310` is now an executable producer for vanilla Token `90001110`, which
  moves that Token from database-only to behavior-complete without an override.
- Filtered hand targeting uses one centralized printed-definition candidate
  path for selected, random, and all-own-hand operations, including source-ID
  exclusion, required-target play/activation legality, pending options, stale
  revalidation, and RL masks. The strict `hand_filter` schema supports type,
  class, printed cost, identity/name, and trait fields. Exact `10333310` selects
  only a follower in hand, permanently adds 1 to its cost, then destroys a
  seeded-random enemy follower; direct tests cover no candidates, invalid and
  stale choices, no enemy target, determinism, and random/all filter reuse.
- Exact file `data/rules/real_basic_followers_batch.json` adds 13 audited
  followers with no alternate modes or referenced cards. Repository-backed
  tests cover Fanfare and Last Words destruction, follower/card banish,
  unconditional follow-up healing, another-follower source exclusion,
  deterministic filtered spell draw, simultaneous enemy-wide damage, exact
  death-draw counts, normalized static and granted Barrier keyword state, and
  RL mask parity. A `non_intrinsic_keyword` annotation ensures `10412120` starts
  without Barrier before its Fanfare grants exactly one charge to every allied
  follower, including itself.

## Known Partial Or Unsupported Areas

- `觉醒` has condition/expression support, public observation flags, and one
  real-card rule demo; broader real-card coverage and future max-mana ramp
  interactions remain partial.
- `连击` has natural per-turn counting, condition/expression support,
  `add_combo`, public observation counts, and real-card demos. Exact
  `10713110` evaluates Combo 3 at its controller's turn-end boundary and only
  while the source remains in play; broader real-card coverage remains partial.
- Union Burst core gauge and threshold semantics are implemented, but every
  additional real card still needs an explicit structured definition. Faith
  leader-area initialization, evolution progression, atomic value spending,
  and gained structured abilities are implemented, while mode-selection,
  Enhance, and named-follower progression triggers remain
  explicit partial semantics. `策动` and `土之秘术` /
  `土之印` core semantics are implemented, while broader real-card coverage
  remains intentionally incremental.
- Faith and emblems are both represented in the leader area, but the official
  shared five-slot capacity is not enforced until a real conflict case requires
  its ordering semantics.
- `AMULET_ACTIVATED` and `CARD_FUSED` reach ordinary board, hand, and
  leader-area listeners. Individual real-card reactions still require authored
  rules and tests.
- `融合` has the command-level material transition, state, event, structured
  filters/counts, source fused-count conditions, ordered post-Fusion hand
  transforms, material lineage preservation/reset policy, refusion, RL
  exposure, and exact real-card demos. Other hand/board cards can listen to
  Fusion; later generated Artifact end-form abilities remain incremental
  structured-rule content.
- `瞬念召唤` is implemented for its sole current official card and marked
  implemented in the ability registry. Sandalphon now combines exact
  Invocation, crest, return-to-hand, and `解放奥义` rules.
- The generated-card audit is deterministic and covers all 91 database tokens.
  It does not equate a database reference with an executable entry, and it only
  marks behavior complete for vanilla cards, fully implemented keyword-only
  cards, or explicitly exact structured rules. Current output is 11 complete,
  11 partial, and 69 database-only/no-entry; this is the content backlog for
  later card-rule entry, not a claim of broad generated-card support.
- The ability registry is conservative and fully audited: all 34 entries have
  a reason and test evidence, and a covered generic primitive does not promote
  a keyword until its real tagged-card semantics and boundaries are complete.
- Many real cards are intentionally uncovered or only partly covered by
  structured rules.
- Additional effect-driven normal-evolution cards, SEP restoration, and cards
  that super-evolve selected/deck-summoned groups remain incremental real-card
  coverage work; the generic single/all follower super-evolution operation is
  available without card-ID branches.
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

### 1. Further Basic Real-Card Batches

- Continue 5–15 card batches whose complete primary/alternate/reference text
  fits existing primitives. Audit every referenced Token with its producer;
  do not select `10373310` until generated amulet `90074210` is itself exact.

### 2. Source-Backed Continuous Modifiers

- Add source-backed derived modifiers with deterministic stacking and automatic
  recomputation for entry, leave-play, transform, return, banish, and control
  changes when a verified real card requires them. This is not the official
  `灵气` keyword, whose manual-target protection is implemented.

### 3. Targeting Edge Cases

- Extend no-target branch coverage only when real cards need graveyard/hand
  target-dependent filters or additional fallback target semantics.
- Audit real rules that combine selected hand/graveyard sets with later
  operations before expanding `target_key` beyond board-entity tuples.

### 4. Trigger Loop Diagnostics

- Add broader trigger ordering tests around any future `death_batch_start`
  boundary semantics and real-card recursive trigger combinations as coverage
  expands.

### 5. Incremental Real-Card Coverage

- Add further Faith progression/payoff and Union Burst cards only when their
  complete generic operations can be represented without card-ID branches.

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
