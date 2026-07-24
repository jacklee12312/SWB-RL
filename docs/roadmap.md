# SWB Engine Roadmap

Last refreshed: 2026-07-24.

This file tracks implementation priorities and known gaps. Treat executable
code and tests as the source of truth when this file drifts.

## Current Baseline

- Database: 826 cards, 735 collectible cards, 91 non-collectible/generated
  cards from set `90000`.
- Latest SVA source: `https://sva.hypd.asia/data/cards.json`.
- Tests: `python -m unittest discover -s tests -v` discovers 2400 behavioral
  contracts (2026-07-24 thirteenth random/same-name/deck-cost verification).
- RL adapter: fixed 111-action space; the default v1 observation is 294
  floats, opt-in v2 preserves the structured compatibility mapping, and v3
  supplies fixed-dtype NumPy arrays plus a Gymnasium observation space without
  changing action IDs. Hidden decklists are the v3 default.
- The exact-audit `TrainableCardCatalog` currently admits all 683 exact
  collectible cards, preloads all 826 definitions for SQLite-free matches, and
  provides deterministic class-valid deck sampling. Self-play no longer uses
  the legacy 2-to-5-card follower pool.
- `SWBAECEnv` is a PettingZoo AEC wrapper with per-agent rewards and done state.
  Rules endings and sampling truncations are mutually exclusive, and both game
  turn and agent-step safety limits are explicit.
- The reproducible RL baseline now includes state-version caches, bounded-log
  training mode, immutable spawn-safe worker assets, formal Observation/Action
  and trajectory versions, deterministic single/multi-worker rollout,
  recurrent masked PPO with stable card embeddings, persistent multiprocess
  fixed-policy sampling, atomic mid-episode checkpoint/resume, a four-kind
  opponent league, fixed mirrored evaluation, `SWBGymEnv`, and deterministic
  snapshot/restore/clone. PPO training now supports a deterministic seven-class
  7x7 ordered matchup cycle, and the fixed evaluation suite defaults to two
  seeded exact deck pairs per class with mirrored sides. Official PettingZoo
  and Gymnasium checks pass.
- The saved embedding/vector CPU smoke is deliberately not a strength claim:
  its 2-worker whole-episode batches requested 1,024 agent steps, completed
  1,304 steps/16 episodes, resumed to 1,571 steps/20 episodes without episode-ID
  gaps, and ran a 16-game fixed-seed mirrored evaluation with zero illegal
  actions or mask mismatches. Whole-episode collection may pass the requested
  step boundary.
- The 2026-07-22 distribution audit runs two complete 49-episode class cycles:
  every ordered matchup appears twice, learner/opponent counts are 14 per
  class, all three card types appear, and all 588 exact cards enter at least one
  sampled deck. The new 28-game same-class fixed suite covers all seven classes
  with two deck pairs and mirrored sides; all games rule-terminate with zero
  truncations, illegal actions, or action-mask mismatches. These are environment
  and distribution acceptance results, not policy-strength evidence.
- Ability registry status: 18 implemented, 5 partial, 11 placeholder.
- Explicit card and demo rules live in `data/rules/`; the current coverage
  report classifies 787 card IDs with explicit rules, passives, fusion,
  invocation, activation, Faith, Union Burst, or listener definitions. Current
  collectible coverage is 683 exact (92.93% of 735), 0 partial, 36 supported-but-missing-rule,
  0 missing-primitive, and 16 text-unclear cards.
- `data/reports/token_audit.{json,md}` audits all 91 non-collectible/generated
  cards independently of collectible coverage. Database `card_references` and
  executable structured producers are reported separately: all 91 tokens have a
  complete executable entry/behavior path, no token has a partial authored
  behavior, and none lacks an authored producer. No token is currently
  classified text-unclear or externally blocked; those categories remain
  explicit and can be assigned through `data/audits/token_overrides.json`.
- `data/reports/ability_audit.{json,md}` validates all 34 runtime/database
  registry entries against `data/audits/ability_registry.json`. Every status
  has a precise conservative reason and test evidence. Primitive support is a
  separate column and is covered for all 34; it does not automatically promote
  the 5 partial or 11 placeholder keyword statuses.
- Coverage now has a backward-compatible clause audit. The legacy summary still
  reports 683 exact collectible cards, and all 683 now have an explicit exact
  text mapping plus named direct test evidence. The versioned sibling registry
  at `data/audits/rule_clauses.json` hashes every imported primary and
  alternate-mode clause; changed source text or stale test evidence invalidates
  the audit. Synthetic `999xxx` rules are counted but are not ordinary
  consistency failures. The report includes source import hash/count/timestamp,
  rule version and errata metadata, structured trigger/operation evidence,
  explicit unsupported text, and a stable blocker taxonomy. There are zero
  unverified exact entries and no known missing schema, primitive, targeting,
  or timing blocker among authored rules. The 36 supported-but-missing-rule
  cards are now the per-card structured-content backlog; 16 unclear texts
  remain explicit.
- RL observation v2 is available behind an explicit version switch. A
  configured card vocabulary supplies stable categorical indices for own-hand,
  public-board, initial-deck composition, and public graveyard/banished state;
  runtime modifier, keyword, origin, transform/Fusion, Faith/emblem, choice,
  action-mask, and bounded public-history inputs have fixed shapes. Index 0 is
  padding/unknown. Privacy tests lock out opponent hand identity, hidden Fusion
  state, raw entity IDs, and remaining deck order. V1 stays the default at
  111 actions and 294 floats, and the recurrent/belief-state interface leaves
  model hidden state under caller ownership.
  V2 also exposes both public leader maximum-health values without changing v1
  width or action IDs.

## Stable Priorities

The P0 platform gate and the requested P1/P2 reproducible baseline in
[the RL architecture audit](rl_architecture_audit.md) are complete: exact-audit
catalog, AEC/Gym protocols, no-leak Observation v3, termination/truncation,
version caches, training mode, vector workers, seed derivation, recurrent
masked PPO, stable card embeddings, multiprocess PPO sampling, atomic
checkpoint/resume, opponent snapshots, fixed evaluation,
and performance reports are executable and tested.

The remaining RL priorities are explicitly beyond this baseline: profile and
optimize snapshot/clone payload cost before high-branching search; scale worker
and learner topology only when a real experiment needs it; add an adaptive
curriculum or a 7x7 cross-class strength matrix without weakening deterministic manifests;
and treat every saved training/evaluation number as smoke until a separately
designed policy-strength experiment exists. A full MCTS/search algorithm and a
distributed learner are not implemented. Multiprocess PPO currently freezes
one current policy generation per collection batch and supports self-play only;
mixed random/fixed/historical opponents remain on the single-process collector.

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
   duplicate policy, multi-mode collection, and suspended-choice revalidation
   are in place.
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
- Hand-follower stat modifiers are structured, filterable, duration-aware,
  fingerprinted, visible in own-hand observations, and inherited on ordinary
  follower play. Exact `10172130`, `10343110`, and `10772310` cover generated
  Puppets, class/trait filtering, live source-health thresholds, source leaving
  play, Super Evolution unlock gating, and deterministic replay.
- Physical hand-card cost filters now use current rather than printed cost.
  Generic `summon_hand_copy` retains the selected hand entity, clones its
  attack/health modifiers with fresh modifier identities, and binds only
  successful board outputs. `requires_full_target_count` distinguishes an
  exact distinct-card prerequisite from the existing up-to-N shortage path.
  Followers can carry owner-turn or opponent-turn self-destruction grants;
  these resolve before the player flip, are cleared by ability removal, respect
  Super-Evolution effect-destroy protection, participate in fingerprints and
  invariants, and occupy two explicit public v2 board-runtime bits. Exact
  `10172320`, `10173140`, `10174130`, `10261120`, `10271210`, `10274120`, and
  `10572110` cover the complete real-card slice, including Core/Artifact Token
  production, Evolution/Super-Evolution, Engage, board shortage, stale
  multi-hand choices, RL masks, and official FAQ boundaries.
- Filtered controller-hand count thresholds now share `HandFilter` semantics
  with selected/random/all hand targets. Exact `10112210`, `10521120`,
  `10741120`, and `10853110` cover Combo-gated Token production, target-required
  Engage after source destruction, spell-only thresholds, non-intrinsic Ward,
  selected hand buffs, stale hand-target revalidation, and RL choice masks.
- `controller_hand_count` expressions now accept the same type/class/cost/
  identity/Trait filters. The bounded `repeat` meta-effect snapshots its count
  once, rejects static or dynamic counts above 100, and queues each nested
  operation in sequential resolution order so random candidates, triggers,
  deaths, and state-based checks fully resolve before the next iteration. Empty
  random candidate sets do not consume RNG. Exact `10114130` covers filtered
  Fanfare stats, intrinsic Ward, dynamic Evolve hits, same-target reselection,
  no-target behavior, seeded replay, and RL evolution; exact `10313310` covers
  Combo-driven repeated random health loss and between-iteration death checks.
- `add_shadows` is a structured leader operation with constant or dynamic
  amounts and auditable gain events carrying the source, target player, and
  before/after values. Exact `10152210` covers entry gain, zero-cost Engage,
  source destruction, ordinary graveyard gain, board-capacity-aware Ghost
  summons, Token provenance, deterministic replay, and RL activation. Exact
  `10153120` covers Fanfare gain, Ward, insufficient/sufficient Necromancy,
  one-time payment, sequential random hits, empty candidates, seeded replay,
  and RL evolution. JSON `summon` now schema-requires the implicit
  `own_leader` destination, preventing accidental board-choice targets from
  silently turning a summon into a no-op.
- `choose_one` remains backward compatible for one-mode rules and now accepts
  `choose_count` for multi-mode cards. Distinct modes accumulate through
  sequential `Choose` commands, no selected ability resolves before the full
  set is collected, and the chosen abilities execute in printed declaration
  order. Modes remain selectable when their current target set is empty and
  safely no-op at execution, matching the official Mode tip. Duplicate
  attempts are atomic; request IDs, fixed RL masks, and public selection
  progress advance at each step. Exact `10852310` covers all six pairs of its
  four modes, random/no-target behavior, spell graveyard Shadow, seeded replay,
  and RL continuation.
- Exact file `data/rules/real_existing_primitives_followup_batch.json` adds ten
  collectibles without new engine branches: `10151140`, `10652110`, and
  `10352120` exercise Reanimate and Enhance ordering; `10113140`, `10501110`,
  and `10253120` cover Combo, other-card printed-cost existence, and other-own-
  follower counts; `10761120`, `10251110`, `10233110`, and `10532120` cover
  filtered draw/live hand counts, two-target destruction, Clash/Evolve draws,
  and Last Words Spellboost/cost reduction. Tests cover normal and shortage
  paths, source exclusion, board capacity, seeded replay/fingerprints, and RL
  mode plus two-step choice masks.
- Exact file `data/rules/real_deck_cost_transform_token_batch.json` adds
  copy-specific runtime deck cost modifiers plus cost-bearing hand transforms.
  Only physical deck copies present at resolution are modified; modifiers
  stack, floor at zero, survive shuffle/draw, enter deterministic fingerprints,
  and are invariant-checked. Exact `10334120` gives every current deck follower
  `-3` cost, transforms one seeded-random hand spell into `90034310`, and sets
  the replacement to 0 until turn end. Exact generated spell `90034310`
  permanently raises every allied hand follower's cost by 1 before one
  simultaneous enemy-follower destruction batch. Tests cover filters,
  stacking, later deck entries, empty-candidate RNG preservation, expiry,
  illegal-play atomicity, deterministic replay, producer auditing, and RL mask
  parity.
- Exact file `data/rules/real_attack_history_emblem_countdown_token_batch.json`
  adds a fingerprinted and invariant-checked public follower-attack counter,
  the `controller_follower_attacks_this_turn_at_most` condition, and an ordered
  `all_own_emblems` countdown-increase target. Legal attack declaration counts
  even if its attack trigger removes the attacker; illegal attacks remain
  atomic, and the count resets at the turn boundary. Exact `10364120` produces
  exact generated spell `90064310`, gains its nonstacking crest on evolution,
  and at turn end distributes the live allied-crest count among all enemies
  only when no allied follower attacked. The spell seeded-randomly banishes one
  enemy follower before increasing every allied countdown crest by 1. Tests
  cover no-attack/attacked branches, source removal, empty-target RNG
  preservation, permanent/opposing crest exclusion, event ordering, producer
  audit, and v1/v2 public observation. The two attack counters intentionally
  migrate v1 from 292 to 294 while all 111 action IDs remain unchanged.
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
- Random own/enemy follower and board targets consume the same count fields
  automatically. Default selection is seeded and distinct, candidate shortage
  uses every available target, explicit duplicates use selection with
  replacement, and the selected batch resolves before one state-based check.
  Fixed printed repetitions may remain separate operations; dynamic repetitions
  use `repeat`. Both forms independently reselect and stabilize between
  iterations.
- `CardListenerDefinition` provides ordinary board, hand, and leader-area event
  listeners for amulet activation, card draw, Earth Rite, Fusion, follower summon/evolution/
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
  remains 111 actions/294 observations; detailed modifier identity and lifetime
  fields remain in the explicit v2 observation rather than leaking entity IDs.
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
  `activate` trigger rule. Such a definition also makes an otherwise effectless,
  non-countdown amulet normally playable, with matching command and RL masks.
  An eligible field amulet can activate once per turn;
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
  and two draws. Exact follow-up rules add eight activation cards spanning
  activation-only play, self-destruction, targeted buffs/debuffs or keyword
  removal, healing, hand cycling, Countdown, and all-follower resolution.
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
- `exclude_source` is a structured selected/random/all-board targeting policy
  carried by the centralized candidate generator. It works across mixed
  follower/amulet zones, multi-target choices, and automatic all-target batches;
  legality, target-exists, pending choices, stale revalidation,
  fingerprints/invariants, and RL masks share the filtered set. Real `10664120`
  demonstrates selecting three other board cards, while exact `10121120`
  demonstrates an all-other-followers Fanfare/Evolve buff.
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
- Last Words are the explicit off-board exception: each death record captures
  an immutable source-state snapshot (evolution/super-evolution state,
  effective keywords, attack, and health), which source conditions and
  expressions can read through nested pending choices. The snapshot never
  makes the departed entity a valid `self` target, and it is covered by event
  metadata, deterministic fingerprints, loop diagnostics, and invariants.
- Generic effects for damage, healing, draw, summon, destroy, banish, return,
  discard, transform, stat changes, keyword changes, cost changes, and attack
  or targeting restrictions.
- `distribute_damage` implements the official oldest-first follower procedure:
  earlier followers receive no more than their current health, and the final
  remainder goes to the last follower or an explicitly included leader.
  Allocations are determined before prevention is applied. No-follower paths
  are an explicit no-op for follower-only effects and assign the full amount to
  the leader when included.
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
- Emblem triggers now accept `card_fused` and structured event-card filters.
  Fusion filters inspect all consumed material definitions but enqueue only one
  activation per Fusion event, matching Octrice's official two-Treasure FAQ.
  `emblem_self` is an explicit countdown-change target, preserving `self` as
  the triggering card for existing emblem rules. Same-name crests still use
  their declared stacking policy instead of duplicating silently.
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
- Special play modes: `爆能强化`, `激奏`, and `结晶`. Enhance preserves the
  card's original follower/spell/amulet route, appends mode operations by
  default, and supports explicit `replace_base_operations` for printed
  replacement clauses. Legality evaluates only the effective operation set;
  enhanced spells still resolve once into the graveyard and emit the selected
  mode ID without adding RL actions.
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
- Exact file `data/rules/real_overflow_batch.json` adds 9 audited Dragoncraft
  cards at the 6/7-max-mana Overflow boundary. Direct tests cover conditional
  Storm and Intimidate provenance, healing and draws, source-excluding +2/+2 to
  +3/+3 replacement, required-target atomicity, Dragoncraft-follower filtered
  draw, two independently seeded random hits, no-enemy fallthrough, and RL mask
  parity.
- Exact file `data/rules/real_evolution_batch.json` adds 12 audited followers
  with no alternate modes or referenced cards. Direct tests cover repeated
  Fanfare/Evolve effects, evolve plus Last Words draws, all-other source
  exclusion, selected no-target fallthrough, simultaneous damage/debuff deaths,
  all-hand Spellboost, self-damage/healing order, static Guard/Ambush, and a
  source-excluding Super-Evolve Storm choice with RL mask parity.
- Exact file `data/rules/real_evolution_followup_batch.json` adds 12 more
  audited followers with no alternate modes or referenced cards. Direct tests
  cover type/class-filtered draws, PP restoration, repeated selected and
  enemy-wide health reduction, damage, destruction, Combo gain with and without
  a target, whole-hand Spellboost, normal and Super-Evolve self buffs or keyword
  grants, intrinsic Ambush/Guard, lethal state checks, a temporary attack lock,
  and RL mask parity. Cards whose text permanently buffs a follower in hand
  remain unsupported until a persistent hand-stat primitive is implemented.
- Exact file `data/rules/real_core_primitives_batch.json` adds 12 audited cards
  with no alternate modes or referenced cards. Direct tests cover per-boost
  cost reduction, Combo expressions and conditions, effect evolution without
  EP, Enhance grants with non-intrinsic Storm provenance, selected hand return
  or discard, source-bound turn-end draw, all-ally unevolved filtering,
  source-excluding cross-controller targets, sufficient/insufficient
  Necromancy, Reanimate after selected destruction, and no-target fallthrough.
- Exact file `data/rules/real_activate_followup_batch.json` adds 8 audited
  amulets with no alternate modes or referenced cards. Direct tests cover zero-
  and one-PP activation, activation-only normal play, once-per-turn legality,
  source destruction before pending choices, selected Ward removal, buffs and
  health reduction, healing, repeated hand cycling, Countdown, simultaneous
  all-follower damage/destruction, and real-card RL mask parity.
- Exact file `data/rules/real_random_multi_target_batch.json` adds 5 audited
  cards with complete localized-text, keyword, alternate-mode, reference, and
  source-hash evidence. Direct tests cover seeded distinct selection, two-hit
  reselection, Combo replacement, candidate shortage, one simultaneous death
  batch, leader damage, whole-hand Spellboost, replay determinism, and RL mask
  parity without exposing an automatic random choice.
- Exact file `data/rules/real_spell_enhance_batch.json` adds 6 audited
  collectible spells and generated spell `90021350`. Direct tests cover normal
  versus enhanced costs and effects, append/replacement provenance, random
  target-count replacement, simultaneous damage, draw/heal order, board and
  hand capacity, Token origin, normal Mode choices, Enhance-all execution,
  filtered selected/all banishment, spell graveyard identity, and RL mask
  parity. The Token audit now recognizes Glittering Gold as an exact generated
  card with an executable producer and both printed branches.
- Exact file `data/rules/real_random_effect_followup_batch.json` adds 6 audited
  collectible followers plus generated follower `90054130`. Direct tests cover
  both Mode branches, Token capacity/origin/keywords/Last Words, filtered
  two-amulet thresholds, dynamic other-card target counts, mixed-zone
  source-excluding destruction, ordered own-turn-end effects, repeated random
  hits, conditional effect evolution without EP, determinism, death batches,
  and RL Mode-mask parity. The Token audit now recognizes One-Tailed Fox as an
  exact generated card with an executable producer and full printed behavior.
- Exact file `data/rules/real_last_words_source_snapshot_batch.json` adds
  `10203120` as the first real consumer of death-time source snapshots. Direct
  tests distinguish unevolved, manual-evolved, and Enhance-evolved deaths;
  cover EP/mana provenance, no-target skipping, seeded random replay, imported
  source hash, and RL Enhance-mask parity; and keep the conditional Last Words
  source off the board throughout resolution.
- Exact file `data/rules/real_royal_gilded_last_words_batch.json` adds Royal
  followers `10321110` and `10322110` plus exact generated spells `90021310`
  through `90021340`. Direct tests cover Fanfare/Last Words Token origin,
  full-hand overflow, required own-follower legality, one-target stat/keyword
  binding, stale pending targets, follower-or-leader damage, capped healing,
  imported source hashes, producer auditing, and RL action-mask parity. The
  Token audit promotes all four gilded spells from database-only to complete.
- Exact file `data/rules/real_generated_hand_last_words_batch.json` adds five
  collectible followers whose Last Words generate a collectible amulet or
  non-collectible cards in hand. Direct tests cover imported names, stats,
  aliases, references and source hashes; generated versus Token origin;
  printed two-card order and second-card full-hand overflow; normal versus
  Enhance-4 Rush/Bane state; zero-cost Skeleton play; Forest's Mystery healing;
  Fairy and Ancient Artifact Rush; Bat Drain; deterministic replay; and RL
  normal/Enhance mask parity. Non-intrinsic annotations prevent the Enhance
  keywords on `10152110` from becoming initial abilities. The Token audit now
  promotes `90011310` and `90071140` to complete and records exact audited
  behavior plus the new executable producers for all five generated cards.
- Exact file `data/rules/real_intrinsic_keyword_pairs_batch.json` introduces a
  first-class `intrinsic_keywords` declaration for cards whose complete text is
  only printed follower keywords. The loader normalizes aliases, rejects empty,
  duplicate, unknown, malformed, and cross-file duplicate declarations, and
  exposes the declaration as coverage evidence without inventing a no-op
  trigger. Six real followers directly cover Storm/Bane, Bane/Ward,
  Storm/Intimidate, Ward/Barrier, Ambush/Bane, and Ward/Aura initial state,
  source hashes, database aliases, combat legality, deterministic rule loading,
  and RL Storm attack-mask parity.
- Exact file `data/rules/real_exact_token_followup_batch.json` adds five
  collectible cards whose complete text uses already-audited generic effects
  and complete Tokens: `10011110`, `10112310`, `10151120`, `10151130`, and
  `10512120`. Direct tests cover two-card Fanfare generation and second-card
  overflow, add-before-draw ordering and overdraw, evolve-time double Bat
  summoning, double-Skeleton Last Words, board-capacity shortage, Token origins
  and producer records, Rush/Bane/Drain combat, imported source hashes,
  deterministic replay, and RL action-mask parity.
- Exact file `data/rules/real_complete_token_board_batch.json` adds nine
  collectible cards whose complete text reuses exact Fairy, Knight, Iron
  Knight, Orca, Skeleton, and Ancient Artifact Tokens. Direct tests cover
  Fanfare/Evolve repetition, summon-before-add and summon-before-buff order,
  draw-before-summon, board and hand capacity, selected evolve damage,
  no-target and stale-target revalidation, Knight/Skeleton Last Words, illegal
  evolution fingerprint preservation, Token origins/keywords/producers,
  deterministic replay, imported source hashes, and RL target-choice masks.
- Exact file `data/rules/real_super_evolution_unlock_batch.json` adds five
  collectible cards gated by each controller's configured super-evolution
  unlock turn: `10121150`, `10202110`, `10401120`, `10471110`, and `10872310`.
  The generic condition context is shared by trigger and target-candidate
  evaluation and exposes both controller and opponent unlock state. Direct
  tests cover asymmetric first/second-player boundaries, conditional
  Bane/Barrier/stat buffs/healing/effect evolution, intrinsic Rush/Ward/Storm,
  actual combat and Barrier consumption, no-EP effect evolution, health caps,
  deterministic replay, imported source hashes, and RL play/attack-mask parity.
- Exact file `data/rules/real_evolution_replacement_batch.json` adds five
  followers whose ordinary Evolve effect is replaced, not repeated, by their
  Super Evolve text: `10002110`, `10042110`, `10062110`, `10072120`, and
  `10541110`. A generic `source_super_evolved` condition reads live state or an
  immutable source snapshot and guards the ordinary branch. Direct tests cover
  exact 2-versus-4 healing, selected-versus-all damage and health-filtered
  banish, one-versus-two collectible self-summons with intrinsic Ward, board
  and health caps, no-target evolution, insufficient multi-target candidates,
  duplicate rejection, leave-play revalidation, deterministic replay, imported
  source hashes, and RL two-choice mask parity.
- Exact file `data/rules/real_evolution_resource_recovery_batch.json` adds five
  cards around capped EP/SEP restoration: `10414110`, `10653310`, `10801120`,
  `10804120`, and `10854110`. The new generic effects emit actual-versus-
  requested restoration events with configured maxima and schema-reject
  non-positive amounts or non-controller targets. Direct tests cover the 9/10
  Union Burst boundary, intrinsic Rush/Ward, full-resource no-overflow, both
  spell modes, both Fanfare modes, both Evolve modes, leader/heal ordering,
  simultaneous deaths, draw, mana recovery, illegal-choice immutability,
  deterministic continuations, source hashes, and RL masks. Target
  availability now distinguishes selected targets from random/all effects:
  empty random/all candidate sets safely resolve as no-ops unless
  `requires_target` explicitly prohibits play.
- Exact file `data/rules/real_selected_hand_exchange_batch.json` adds eight
  followers whose Fanfare selects another own-hand card to return or discard:
  `10131120`, `10621120`, `10641110`, `10642110`, `10703110`, `10711120`,
  `10811130`, and `10842120`. Existing optional selected-hand semantics allow
  later effects to continue when no other hand card exists. Direct tests cover
  return/discard ordering, plain and Royal-follower-filtered draws, Earth
  Sigils, Last Words, simultaneous enemy-wide damage, exact Fairy generation,
  full-hand behavior, repeated Fanfare text on Evolve, health caps and EP
  spending, intrinsic Rush/Barrier/Ward combat, stale/illegal target handling,
  deterministic replay, imported source hashes, Token producer auditing, and
  RL choice-mask parity.
- Exact file `data/rules/real_existing_condition_direct_batch.json` adds ten
  cards using established Combo, evolved-board, amulet-count, target-health,
  hand-count, Union Burst, Enhance, and direct-effect primitives: `10012110`,
  `10111110`, `10111130`, `10152140`, `10433310`, `10451110`, `10452120`,
  `10512310`, `10672120`, and `10762110`. The shared play path now carries
  precomputed Union Burst definitions into ordinary spells and amulets, keeps
  base → Burst → mode ordering, and emits the same activation metadata used by
  followers. Direct tests cover exact and below thresholds, source exclusion,
  draw-before-hand-count healing, no-target continuation, Earth Sigils,
  effect evolution without EP, intrinsic Rush/Aura/Ambush, Combo replacement,
  health-filtered banish, three-amulet targeting, deterministic replay,
  imported hashes, and RL masks.
- Exact file `data/rules/real_evolve_and_burst_direct_batch.json` adds eleven
  cards using established evolution, turn-end, board-listener, Earth Rite,
  draw, destruction, and target primitives: `10231310`, `10411120`, `10432110`,
  `10442110`, `10453110`, `10473310`, `10571120`, `10622120`, `10642120`,
  `10742120`, and `10832110`. `UnionBurstDefinition.replace_base_operations`
  is a validated opt-in for text whose active Super Skybound Art replaces the
  base effect; its default remains additive, and follower/spell/amulet play plus
  suspended follower choices preserve the selected behavior. Direct tests
  cover exact 3-versus-6 damage, below/at-threshold behavior, max-mana effect
  evolution, turn-end evolution and healing, distinct multi-target shortages,
  owner-only spell listeners, exact-health destruction, intrinsic Drain/Storm,
  asymmetric draws, simultaneous effects, deterministic replay, imported
  hashes, and RL masks.
- Exact file `data/rules/real_board_state_filter_batch.json` adds ten cards:
  `10201110`, `10204120`, `10222110`, `10231110`, `10242110`, `10261110`,
  `10341120`, `10441120`, `10462110`, and `10863210`. `BoardFilter` now has
  schema-validated `super_evolved` and `damaged` booleans for board-presence
  conditions and manual/random/all board targets. Super evolution remains
  distinct from ordinary evolution; damaged means current health is strictly
  below maximum health and therefore never matches amulets. The batch covers a
  conditional emblem, hand-zone super-evolution listener with printed-cost
  filtering and end-of-turn expiry, exact damaged-target revalidation,
  Enhance, Countdown, Ward, Earth Sigils, draw/heal replacement, deterministic
  random selection, imported source and alternate-mode hashes, and RL choice
  masks.
- Exact file `data/rules/real_extreme_candidate_batch.json` adds eight
  collectible cards (`10103310`, `10341310`, `10503310`, `10552310`,
  `10613310`, `10743110`, `10822310`, `10832320`) plus complete generated
  followers `90011120` and `90022110`. `CandidateExtreme` filters otherwise
  legal board candidates by highest/lowest current Attack or Health while
  retaining every tie; random selection remains seeded within that reduced
  set, and all-target resolution snapshots all tied candidates. `all_leaders`
  supplies the corresponding current-Health-only set semantics for one or both
  leaders. Direct tests cover schema rejection, no-target continuation,
  simultaneous snapshots, Overflow, Enhance, Earth Rite, Mode ordering,
  turn-end effect evolution, intrinsic Ward/Rush, source hashes, and RL masks.
- Exact file `data/rules/real_deck_summon_batch.json` adds five collectible
  cards: `10164110`, `10264110`, `10322120`, `10462120`, and `10813310`.
  The generic `summon_from_deck` operation filters physical deck cards by type,
  class, cost, ID, name, or Trait; chooses through the engine RNG with physical
  duplicate copies contributing their official increased probability; excludes
  an already-selected card name for the rest of the same operation; and caps
  removals to available board slots. Followers and amulets retain Deck origin,
  emit ordinary entry/Cooperation events without Fanfare or play effects, and
  preserve Countdown/Earth Sigil initialization. Direct tests cover schema
  rejection, candidate/board shortages, copy weighting, distinctness,
  deterministic replay, the official no-other-hand Rodeo Q&A, class filters,
  Evolve/Super Evolve follow-ups, hashes, and unchanged RL action IDs.
- Exact file `data/rules/real_summon_output_trait_batch.json` adds seven
  collectible cards (`10124120`, `10224110`, `10653110`, `10724120`,
  `10753110`, `10754110`, and `10834120`) plus exact generated-card follow-up
  for `90031120` and `90051140`. `BoardFilter` now accepts class ID/name and
  Trait ID/name for centralized selected/random/all board candidates and board
  conditions. `summon` and `summon_from_deck` can bind the stable IDs of their
  successful outputs to `target_key`; full-board or no-candidate outcomes bind
  empty, while direct-deck multi-output bindings retain only entities that
  actually entered. Follow-up `previous_target` operations skip outputs that
  have since left play and never fall back to older same-card entities. Direct
  tests cover single and multi-output identity, capacity failure, Cooperation,
  Royal/Nightmare filtering, entry listeners, dynamic Rush/Storm, Spellboost,
  automatic and manual evolution, intrinsic Ward, and Rotten Zombie's
  terminating Last Words replacement.
- Exact file `data/rules/real_basic_existing_primitives_batch.json` adds ten
  collectibles (`10002210`, `10123120`, `10124110`, `10162220`, `10164120`,
  `10403120`, `10551110`, `10662210`, `10762210`, and `10814110`) without new
  engine primitives or referenced generated cards. Four additive Enhance modes
  cover filtered high-cost follower draw, PP restoration, all-enemy damage,
  double attacks, source buffs, and granted keywords; normal-mode tests prove
  those operations do not leak into base play. Three Engage amulets verify
  activation costs, source destruction before pending board choices, damage,
  healing, and immediate same-turn activation. Countdown expiry, Last Words,
  intrinsic keyword separation, source-excluding buffs, and ordered selected
  destruction are directly tested with imported-text hashes.
- Exact file `data/rules/real_listener_evolution_existing_batch.json` adds
  fifteen collectibles (`10022210`, `10062120`, `10112130`, `10131130`,
  `10132110`, `10133120`, `10161110`, `10461110`, `10463110`, `10612310`,
  `10622110`, `10652120`, `10712110`, `10862120`, and `10871120`). Six
  structured listeners cover owner amulet activation, exact follower entry,
  stable hand-card normal evolution, and stable hand-card Super Evolve events.
  The remaining rules reuse Combo, Enhance, mixed-board return, filtered draw
  and hand choice, Spellboost, Earth Sigils, PP restoration, intrinsic keyword,
  and Evolve/Super Evolve target flows. Direct tests distinguish below/at Combo
  threshold, normal and super event dispatch, source exclusion, listener owner
  scope, exact hand identity, and ordered target follow-ups.
- Exact file `data/rules/real_repeated_evolve_listener_batch.json` adds fifteen
  collectibles (`10104110`, `10113110`, `10232120`, `10253110`, `10372110`,
  `10444110`, `10461120`, `10552120`, `10561120`, `10562120`, `10643110`,
  `10661210`, `10731120`, `10841110`, and `10861110`). `CARD_DRAWN` is now a
  supported ordinary listener event and carries the new hand entity as its
  filterable event source. Two board listeners demonstrate owner-event plus
  owner-turn scope, including a Fanfare/Enhance draw that immediately triggers
  its own in-play listener. The remaining rules reuse established selection,
  random/all target, output binding, ability removal, discard, Earth Sigil,
  attack restriction, Combo, evolution, Countdown activation, Last Words, and
  intrinsic keyword primitives. Direct tests cover empty-target continuations,
  expiry and threshold boundaries, repeated Fanfare/Evolve paths, event turn
  scope, deterministic random effects, and exact imported-text hashes.
- Exact file `data/rules/real_spell_modes_and_earth_listener_batch.json` adds
  fifteen spells (`10122310`, `10132310`, `10211310`, `10241310`, `10323310`,
  `10332310`, `10432310`, `10542310`, `10611310`, `10621310`, `10731310`,
  `10732310`, `10733310`, `10803310`, and `10823310`). Ordinary listeners now
  accept the already-audited `EARTH_RITE_ACTIVATED` event, enabling two hand
  cards to stack temporary cost reduction and expire it at the correct turn
  boundary. Enhance validation can inherit top-level base-operation output
  bindings, matching runtime concatenation and allowing `10621310` to buff
  only the soldiers that actually entered under board-cap shortage. Five Mode
  cards cover explicit choice and threshold-driven all-branch resolution;
  remaining tests cover Treasure Fusion, Overflow, exact referenced summons,
  dynamic total-follower damage, draw/damage ordering, distinct seeded random
  targets, Reanimate, insufficient Earth Rite payment, hashes, and audit state.
- Exact file `data/rules/real_token_producer_completion_batch.json` closes nine
  collectibles (`10133320`, `10141140`, `10154110`, `10161310`, `10163210`,
  `10331120`, `10371310`, `10373310`, and `10831120`) together with twelve
  generated cards (`90031130`, `90031140`, `90031210`, `90031310`, `90041110`,
  `90054110`, `90054120`, `90061110`, `90061120`, `90071110`, `90074210`, and
  `90074220`). Tests cover exact producer paths, Shikigami Spellboost Last
  Words, simultaneous Mimi/Coco deaths, Necromancy source exclusion, repeated
  Reanimate, opponent-turn Puppet destruction, reciprocal nonrecursive
  White/Black Countdown cycles, Earth Sigil activation, activation-driven
  Tiger summon, and intrinsic Storm/Rush/Aura separation.
- Exact file `data/rules/real_artifact_fusion_completion_batch.json`, together
  with the structured definitions in `data/rules/fusion.json`, completes eight
  generated Artifact cards (`90071210`, `90071220`, `90072110`, `90072120`,
  `90073110`, `90073120`, `90073130`, and `90074110`). `DeckFilter.card_ids`
  provides an audited OR whitelist without card-ID conditionals in resolution.
  Tests cover unplayable cores, legal/illegal Artifact materials, stable hand
  identity and inherited material state, all cumulative-cost branches,
  duplicate-kind non-transformation, β+γ transformation to Ω, own-turn-end
  effects, Ω Fanfare ordering, and intrinsic Rush/Ward/Storm/Aura.
- Exact file `data/rules/real_final_partial_token_completion_batch.json`
  completes generated Ghost (`90051130`) and Improved Puppet (`90071120`). The
  generic `banish_on_leave` passive replaces destruction, return-to-hand, and
  return-to-deck destinations with banishment, but is disabled after printed
  abilities are removed; transformation remains correctly outside leave-play.
  Tests verify no graveyard, Shadow, Last Words, destroyed-follower/Reanimate
  history, or RNG use on replaced returns, plus simultaneous ordinary deaths,
  real producers, Storm/Rush, and own/opponent turn-end expiry.
- Exact file `data/rules/real_token_producer_modes_activation_batch.json`
  closes six collectibles (`10042120`, `10141110`, `10143210`, `10144130`,
  `10061210`, and `10821110`) and six generated cards (`90021130`, `90041120`,
  `90043110`, `90044110`, `90044120`, and `90061130`). Tests cover normal versus
  Enhance play and RL mode masks, paid activation and selected discard,
  summon-before-discard ordering, Countdown expiry and Last Words slot release,
  ordered dual summons, exact card-ID Super Evolution keyword grants, repeated
  summon-then-damage resolution, intrinsic Token keywords, and board-capacity
  continuation.
- Exact file `data/rules/real_generated_spell_and_follower_chain_batch.json`
  closes six collectible producers (`10223120`, `10134120`, `10872120`,
  `10873110`, `10354120`, and `10374120`) and six generated cards (`90023110`,
  `90034130`, `90071160`, `90054310`, `90054320`, and `90074310`). The generic
  `cannot_be_destroyed_by_effects` passive and `effect_destroy_prevented` event
  implement printed immunity without a card-ID branch. Direct tests cover
  ability removal, banish and zero-Health boundaries; exact-card-ID conditions;
  summon-before-Spellboost and destroy-before-summon order; opponent-turn-end
  expiry; both Mode choices and RL mask parity; bound successful summon outputs;
  board/hand capacity; and illegal no-target fingerprint/RNG preservation.
- Exact file `data/rules/real_generated_burst_spell_batch.json` closes six
  collectible producers (`10304120`, `10824110`, `10654120`, `10314120`,
  `10834110`, and `10114120`) and seven generated spells (`90004330`,
  `90024330`, `90054330`, `90014320`, `90034340`, `90034350`, and `90014310`).
  Generic additions are an opponent-max-PP condition, an immutable play-time
  Spellboost value carried by effect frames and deterministic fingerprints,
  printed `attacks_per_turn` passives, and filtered all-hand transformation.
  Direct tests cover both 10-PP players; 9/10/19/20 Spellboost thresholds;
  Cooperation and Necromancy success/failure; full-board continuation; two
  independently sampled random hits; own-turn hand cost listeners; static
  attack capacity and ability removal; follower/leader targets; atomic illegal
  spell play; RL mask parity; and compatibility with source-leave semantics.
- Exact file `data/rules/real_generated_entry_listener_batch.json` closes four
  collectible producers (`10874120`, `10773110`, `10272110`, and `10264120`)
  and three generated followers (`90071130`, `90072130`, and `90064110`). It
  reuses self-related follower-entry listeners for Analyzing Artifact's draw,
  Artifact-trait filtering for Wild Announcer's Rush grant, filtered selected
  hand transformation for Fia, and a persistent emblem represented by an
  existing leader-area listener for Wilbert's Ward-entry buff. Tests cover
  normal/Enhance output order, RL Enhance masks, source and evolved-target
  exclusion, no-candidate continuation, stable hand identity, Ambush/Last
  Words return, crest persistence, non-Ward exclusion, and board shortage.
- Exact file `data/rules/real_generated_spellboost_growth_batch.json` closes
  three collectible producers (`10814120`, `10232110`, and `10133310`) and
  three generated cards (`90014110`, `90032110`, and `90033310`). The generic
  listener surface now exposes positive follower stat-increase events, and
  hand listeners can consume their own immutable Spellboost count and transform
  the exact event source while preserving entity identity. Direct tests cover
  Tia's Enhance-wide buff, own-turn once-per-turn reset, evolution/debuff and
  ability-removal exclusions, Eve's Storm/Ward, Basset's dual summon and
  recurring crest production, Onion's attack-time all-hand Spellboost, the
  exact fifth-Spellboost transformation threshold, empty-enemy continuation,
  board capacity, deterministic event metadata, and RL Enhance-mask parity.
- Exact file `data/rules/real_generated_damage_countdown_batch.json` closes
  three collectible producers (`10344120`, `10434120`, and `10764110`) and
  three generated cards (`90044320`, `90034320`, and `90064210`). Generic
  additions are a filterable follower-damaged-but-survived event and positive
  amulet Countdown changes with auditable before/after metadata. Tests cover
  independent source/crest once-per-turn reactions, lethal and opponent-turn
  exclusions, Enhance Storm, all-board damage, gauge 9/10/15 Union Burst
  boundaries, Earth Rite success/shortage, follower/leader choices, exact-card
  random amulet filtering, Aura legality, the 2/3-amulet threshold, full-board
  production failure, and safe no-target evolution.
- Exact file `data/rules/real_generated_hidden_copy_discard_batch.json` closes
  four collectible producers (`10244110`, `10644120`, `10573310`, and
  `10674120`) and four generated cards (`90044310`, `90044330`, `90074140`,
  and `90074320`). Generic additions are `all_enemy_hand`, stable post-zone
  `discarded` rules, `copy_to_hand` with hidden metadata and a copy-local cost
  modifier, amulet-to-follower transformation, and an auditable board-transform
  event. Tests cover Overflow 6/7, damaged-only Mode destruction, opponent-turn
  cost expiration, discard event identity, full-board summon failure, ordinary
  versus Super Evolution replacement, play/discard spell parity, required
  any-board targeting, stable amulet entity identity, original-cost filtering,
  generated copy origin, non-public logs, and full-hand copy burn.
- Exact file `data/rules/real_generated_forced_target_strike_batch.json` closes
  four collectibles (`10174120`, `10274110`, `10273310`, and `10173120`) plus
  generated Puppetry followers `90074120` and `90074130`. Generic additions
  are a removable source-backed enemy ability-selection lock and implicit
  `attack_target` frame identity. Tests cover unit/leader and mixed candidate
  forcing, filters, multiple Lloyds, random/all exclusions, ability removal,
  leaving play, current-attack pre-combat damage, leader exclusion, both real
  token producers, Puppet-trait entry grants, board shortage, and the official
  Sylvia sequence that destroys Lloyd before selecting Orchis.
- Exact file `data/rules/real_generated_enhance_faith_shikigami_batch.json`
  closes collectibles `10624120` (Yidmetra, Eld Sword) and `10134110` (Kuon,
  Fivefold Master) plus generated `90024320`, `90034110`, and `90034120`.
  `card_enhanced` is a verified Faith trigger derived from the registered play
  mode rather than a mode-name guess, and excludes normal, Accelerate, and
  Crystallize plays. Destroyed-follower records now retain the global turn;
  filtered current-turn expressions sum the printed attack/health of matching
  owned followers. Tests cover insufficient/exact Faith payment, gained
  abilities, required-target atomicity, base/Enhance damage replacement,
  ordered Shikigami summons and simultaneous destruction, three Last Words,
  printed-stat Noble growth, prior-turn/opponent/non-Shikigami exclusions,
  board shortage, Super-Evolve targeting, and deterministic replay.
- Exact file `data/rules/real_crystal_faith_random_distribution_batch.json`
  closes collectible `10634120` and generated `90034330`. Faith progression
  now accepts a structured card filter for owner `follower_summoned` events;
  retained hand copies reuse ordinary listener cost modifiers. Generic
  `random_distribute` reads a named Faith without consuming it, independently
  assigns each point through the engine RNG to one of at least two buckets,
  exposes the bucket count through `distributed_value`, emits an auditable
  result event, and retains nested target bindings plus deterministic
  fingerprints. The official three-way FAQ law, live post-summon value, two
  Storm grants, board capacity, zero-value RNG preservation, and seeded replay
  are directly tested.
- Exact file `data/rules/real_generated_distributed_damage_crest_batch.json`
  closes eight collectibles (`10113120`, `10154130`, `10324120`, `10363210`,
  `10511310`, `10514120`, `10673110`, `10753310`) and generated `90024310`.
  It combines oldest-first damage distribution, dynamic hand/emblem-count
  expressions, Octrice's filtered Treasure crest and expiry payoff, Choose,
  Necromancy, Accelerate, evolution, and deterministic replay. Tests lock
  Barrier allocation, no-follower/one-follower remainder, nonduplicating
  crests, multi-material Fusion ticking once, and all alternate play paths.
- Exact file `data/rules/real_portal_core_puppet_producer_batch.json` closes ten
  Portalcraft collectibles (`10071110`, `10071120`, `10071310`, `10072110`,
  `10072210`, `10171120`, `10171130`, `10172110`, `10172120`, `10173110`).
  It adds executable Future/Past Core, Puppet/Improved Puppet, and Attack/Castle
  Artifact producer paths using existing generic operations. Tests cover
  ordered outputs, intrinsic Rush, target-required spell immutability,
  Countdown owner-turn timing, evolution repeat/target/no-target paths, board
  shortage, Token origin, producer audits, and deterministic replay.
- Exact file `data/rules/real_portal_artifact_listener_followup_batch.json`
  closes ten more Portalcraft collectibles (`10171140`, `10173210`, `10271110`,
  `10371120`, `10372210`, `10374110`, `10471130`, `10571110`, `10671120`,
  `10672310`). Existing listener, Fusion-event, output-binding, play-mode,
  mixed-board, and destruction-immunity primitives implement the full texts.
  Tests lock the official Automata Assassin simultaneous-entry first-only FAQ,
  one Heritage Barrage activation per multi-material Fusion, filtered evolution
  choice, optional no-target draw continuation, real effect-destroy prevention,
  pre-destruction board counting, Last Words, normal/Enhance output sets,
  board shortage, producer audit paths, and deterministic replay.
- Exact file `data/rules/real_portal_artifact_entry_history_batch.json` adds a
  public, deterministic record for every successful follower entry and generic
  filtered condition/expression primitives that count different printed names.
  Seven Portalcraft collectibles (`10771120`, `10771310`, `10772120`,
  `10773310`, `10774110`, `10774120`, `10873310`) use Artifact-Trait history.
  Tests cover same-name deduplication across different IDs, opponent/non-Artifact
  exclusion, leave-play persistence, reset, fingerprints, invariants, zero and
  nonzero dynamic damage, no-target continuation, required-target atomicity,
  Mode/RL masks, board shortage, EP caps, seeded replay, and Myuu checking the
  threshold after its evolution summon. V1/v2 continuous public observations
  expose both players' normalized counts, intentionally migrating v1 from 290
  to 292 floats while preserving all 111 action IDs.
- Exact file `data/rules/real_portal_dynamic_evolution_batch.json` closes five
  Portalcraft collectibles (`10371110`, `10472110`, `10573110`, `10672110`,
  `10874110`). Generic live-board count expressions now accept compact
  `BoardFilter` payloads, and both Attack and Clash trigger frames can bind the
  current opposing follower as `attack_target`. The rules cover dynamic
  other-card buff/destruction, follower-only PP recovery, Last Words draw,
  target-gated Skybound Art double effect evolution, attacker/defender Clash,
  same-card summon output binding, Accelerate 3, non-intrinsic Ward/Storm,
  Enhance 9 effect evolution, and random two-Ward destruction. Direct tests
  cover empty targets, evolved-target exclusion, mixed follower/amulet boards,
  simultaneous cleanup, board shortage, no-EP effect evolution, illegal-mode
  no-mutation, seeded random replay, and RL mode/choice action masks.
- Exact file `data/rules/real_mjerrabaine_deck_batch.json` closes collectible
  `10304110` and the executable producer/behavior chain for generated spell
  `90004320`. Generic `replace_deck`, `set_empty_deck_outcome`, crest `on_gain`,
  and hand-filter exclusion semantics implement the reviewed 76-card special
  deck, seeded shuffle, Victory Card end condition, end-of-turn discard/draw,
  and selected destroy spell without a card-ID branch. The ordinary empty-deck
  rule is corrected from fatigue damage to immediate defeat. Fingerprints,
  events, invariants, ignored duplicate crest acquisition, stale target
  revalidation, illegal-play no-mutation, RL masks, and v2 public victory-mode
  state have direct tests.
- Exact file `data/rules/real_spellboost_recycle_listener_batch.json` closes
  six collectible cards (`10131320`, `10831310`, `10341110`, `10551310`,
  `10843310`, `10752310`). The played hand card's frozen Spellboost count now
  reaches spell effect frames and is available as a generic value expression;
  `add_card_to_deck` resolves a specified definition before one seeded hidden-
  position insertion and emits a distinct auditable event. The real rules also
  exercise survived-damage self listeners, owner-turn gating, filtered
  Dragoncraft-follower draw, follower-or-leader selection, Overflow ordering,
  ordered summon-output evolution, and board shortage. Direct tests cover
  lethal/opponent-turn listener skips, no-target atomic rollback, deterministic
  deck order and fingerprints, schema rejection, RL masks, database references,
  and both identical Storm Blast printings. Token audit producer lists for
  Ghost, Bat, and Skeleton now include the executable `10752310` path.
- Exact file `data/rules/real_zone_output_copy_batch.json` closes five
  collectible cards (`10853310`, `10541310`, `10802310`, `10443310`,
  `10652310`). Draw and filtered-draw operations can now bind every physical
  output, including an overdraw snapshot, while `bound_card_cost` reads one
  named output's frozen current cost. Missing output skips dependent effects
  before random target selection. Generic `random_enemy_hand`, snapshot-backed
  `copy_to_hand`, and `summon_copy` support ordered unrevealed hand copies and
  banish-then-copy flows without retaining a live board dependency or adding
  card-ID branches. Direct tests cover post-unlock cost reduction, deck cost
  inheritance, seeded random targets, no-candidate RNG stability, overdraw,
  stale pending choices, full boards, base-state copy summoning, hidden
  opponent observations, schema rejection, clause hashes, and unchanged
  111-action/v1-294 observation compatibility.
- Exact file `data/rules/real_destroyed_history_source_cost_batch.json` closes
  five collectible cards (`10572310`, `10871130`, `10641310`, `10643310`,
  `10331110`). Effect frames now retain the physical hand card's current cost
  when it is played or discarded, and expose it through `source_cost_equals`
  and `source_cost`; nested frames inherit that frozen value while non-card
  emblem/Faith sources safely default to zero. Generic destroyed-follower
  history copying filters allied definitions, samples by destroyed instance or
  distinct name using only the engine RNG, emits unrevealed generated cards,
  and safely handles empty history, fewer candidates, and full hands. Direct
  tests cover actual simultaneous destruction, deterministic replay,
  Artifact-only filtering, hidden opponent observations, optional empty hand
  selection, cost-4 and 7-to-5-to-3 discard replacement, modified-cost area
  damage/healing, required-target rollback, schema rejection, source hashes,
  and unchanged 111-action/v1-294 compatibility.
- Exact file `data/rules/real_listener_spellboost_followup_batch.json` closes
  eleven collectibles (`10121130`, `10131310`, `10212120`, `10232310`,
  `10252110`, `10353310`, `10361120`, `10612110`, `10673310`, `10721110`,
  `10812110`). `HandFilter` now accepts a normalized ability keyword, allowing
  selected-hand Spellboost effects to share ordinary candidate generation,
  play legality, stale-choice handling, fingerprints, and RL masks. Existing
  summon, Super-Evolution, positive-stat, and hand listeners express the other
  cards without card-ID branches. Direct tests cover the official Rainbow
  Miracle no-target prohibition and Holy Knight SET_STATS distinction, frozen
  Spellboost distribution, all six distinct pairs of four modes, Token
  production and capacity, listener self/other and turn scopes, temporary cost
  stacking/expiry, seeded random evolution/damage, source hashes, and unchanged
  111-action/v1-294 compatibility.
- Exact file `data/rules/real_hand_exact_copy_turn_end_destroy_batch.json`
  closes seven collectibles (`10172320`, `10173140`, `10174130`, `10261120`,
  `10271210`, `10274120`, `10572110`). Generic hand-copy summoning retains the
  selected physical card, copies current stat modifiers with fresh identities,
  and output-binds only successful summons. Full-count availability separates
  the exact-two spell prerequisite from Loramia's official up-to-three shortage
  behavior. Entity-attached owner/opponent-turn destruction resolves before
  the player flip and therefore respects Super-Evolution protection; ability
  removal, fingerprints, invariants, and public v2 state include the grant.
  Direct tests cover current-cost filters, two-step hand selection, duplicate
  prevention, stale targets, board shortage, Engage source destruction,
  Core/Artifact Token production, Evolution/Super-Evolution, RL masks, official
  FAQs, and clause hashes while preserving 111-action/v1-294 compatibility.
- Exact file `data/rules/real_apocalypse_deck_batch.json` closes collectible
  `10104120` and the final four database-only generated cards (`90004110`,
  `90004120`, `90004130`, `90004310`). Generic player maximum-health state,
  healing caps, `set_leader_max_health`, event emission, fingerprints,
  invariants, and v2 public fields implement Astaroth without treating the
  change as damage. The source Fanfare atomically replaces and seed-shuffles
  the official ten-card Apocalypse Deck; its three copies of each follower and
  one spell are authored as structured data. Direct tests cover exact deck
  composition, replay, illegal-play no-mutation, Storm, printed vanilla stats,
  up-to-two selected targets, the official zero-target FAQ, stale second
  targets, health clamping, later healing caps, database text, clause hashes,
  and unchanged 111-action/v1-294 compatibility. The token audit now reports
  all 91 generated cards complete with no partial or database-only entry.
- Exact file `data/rules/real_royal_bishop_existing_primitives_batch.json`
  closes twelve collectibles (`10723310`, `10263310`, `10062210`, `10362220`,
  `10122140`, `10421120`, `10421130`, `10821130`, `10761110`, `10662120`,
  `10562110`, `10762120`). The slice reuses cooperation replacement,
  follower-filtered board counts, countdown/activation, emblem expiration,
  summon-output bindings, intrinsic keywords, evolution, banish, and healing.
  Direct tests lock target-required illegal-command immutability, zero-target
  Rally legality, stale choice continuation, Token references and printed
  order, board capacity, paired/self producer chains, Last Words, deterministic
  replay, RL action masks, clause hashes, and zero unverified exact entries.
- Exact file `data/rules/real_royal_rune_bishop_mixed_batch.json` closes fifteen
  collectibles (`10121140`, `10322210`, `10623110`, `10822110`, `10031110`,
  `10331310`, `10531110`, `10631120`, `10633110`, `10731110`, `10833310`,
  `10463210`, `10562210`, `10662110`, `10763110`). `BoardFilter` and filtered
  board-count expressions now accept an optional keyword and evaluate live
  entity state, so both printed Ward and Ward granted at runtime are counted.
  The slice also reuses Enhance, Engage, choose-one, source destruction,
  once-per-owner-turn spell listeners, Earth Rite, crests, Super Evolve,
  summon-output filtering, Crystallize, Countdown, Last Words, and intrinsic
  keyword declarations. A generic `source_card_type_is` condition reads live
  sources or immutable death snapshots and restricts Crystallize Last Words to
  the amulet form, preventing same-ID follower death loops. Direct tests cover
  targetless/stale choices, board and
  hand capacity, insufficient resources, deterministic random effects, source
  hashes, Token/mode references, backward-compatible filter parsing, and RL
  action-mask parity.
- Exact file `data/rules/real_spell_amulet_crest_batch.json` closes eleven
  collectibles (`10412310`, `10441310`, `10451310`, `10712310`, `10713310`,
  `10233310`, `10352210`, `10633310`, `10413310`, `10332110`, `10631110`).
  The slice reuses Combo, countdown crests, owner-turn listeners, Earth Rite,
  Engage, choose-one, Enhance replacement, Super Skybound Art, source-cost
  conditions, automatic repeat, and summon-output bindings. Direct tests cover
  all primary and crest-mode clauses, paired/self generation, no-target paths,
  hand/board capacity, modified play cost, deterministic replay, RL action-mask
  parity, source hashes, and zero unverified exact entries.
- Exact file `data/rules/real_low_coverage_token_amulet_batch.json` closes twelve
  collectibles (`10143130`, `10741110`, `10641120`, `10651120`, `10252120`,
  `10152130`, `10552110`, `10851120`, `10154120`, `10262310`, `10113210`,
  `10011210`). Generic `bound_target_health` reads one earlier selected live
  follower after intervening operations, and safely becomes zero if that entity
  has left play. The slice reuses self-copy Enhance, intrinsic Rush/Ward,
  Last Words, ordered Token production, evolution, Follower Strike, three-attack
  capacity, Combo, Engage, Countdown, Trait listeners, and keyword-filtered
  targets. Direct tests cover schema rejection, post-buff defense damage,
  no-target illegal-command immutability, stale choice continuation, seeded
  random targets, simultaneous deaths, hand/board capacity, Trait mismatch,
  Token references, clause hashes, and RL action-mask parity.
- Exact file `data/rules/real_balanced_trigger_resource_batch.json` closes
  fifteen collectibles (`10411110`, `10212110`, `10811110`, `10611110`,
  `10614110`, `10213110`, `10123110`, `10522120`, `10522110`, `10134310`,
  `10632120`, `10132130`, `10533110`, `10542110`, `10661110`). The slice
  balances Forest, Royal, Runecraft, Dragoncraft, and Bishop while reusing
  attack capacity, intrinsic/runtime keywords, Trait-filtered buffs, seeded
  repeat, draw and follower-entry listeners, Spellboost cost/value operations,
  Earth Rite, Enhance, Crystallize, Countdown, Last Words, and source-form
  guards. Direct tests cover capacity shortage, simultaneous deaths, no-target
  and stale-target continuation, cost expiry, deterministic recycling/shuffle,
  resource shortage, referenced Token behavior, clause hashes, and RL masks.
  Card `10333110` remains unsupported because its “exact copy” clauses require
  a board-entity clone primitive; ordinary definition-based summon is not used
  as a substitute.
- Exact file `data/rules/real_ward_marine_crest_listener_batch.json` closes
  twelve collectibles (`10162110`, `10361110`, `10362110`, `10563110`,
  `10663110`, `10142110`, `10241110`, `10342120`, `10542120`, `10424110`,
  `10512110`, `10613110`). Ordinary card listeners now accept the generic
  `leader_healed` event, allowing actual nonzero leader healing to trigger
  board abilities with owner-event and owner-turn scopes. The slice reuses
  destroyed-follower keyword snapshots, countdown crests, attack history,
  random target bindings, hand cost listeners, Overflow, Marine Trait entry
  listeners, discard triggers, summon-output bindings, Enhance, Combo,
  Crystallize, Last Words, evolution events, and explicit non-intrinsic keyword
  declarations. Direct tests cover no-target and stale-target evolution,
  simultaneous filtered deaths, full-health non-events, opponent-turn scope,
  board capacity, seeded randomness, Token references, source-form guards,
  clause hashes, and RL masks. Card `10263110` remains unsupported because its
  random attack trigger must exclude the current combat target; the existing
  random-enemy primitive has no generic bound-target exclusion filter, so it is
  not used as an inexact substitute.
- Exact file `data/rules/real_leftmost_golem_bat_crest_batch.json` closes twelve
  collectibles (`10032110`, `10132120`, `10433110`, `10732110`, `10452130`,
  `10752110`, `10852110`, `10852120`, `10423310`, `10524120`, `10723110`,
  `10603210`). Generic candidate extremes now support filtered board-order
  `leftmost` selection without consuming RNG, and `BoardFilter` supports a
  schema-validated `exclude_tribe_name` boundary for all non-Encroacher
  followers. Bound targets explicitly changed by generic effect evolution stay
  available to subsequent operations on the same selected entity. The slice
  reuses Earth Rite, Spellboost, Golem/Bat/Knight/Skeleton production, ability
  removal, two countdown crests, choose-one, PP/EP recovery, simultaneous
  damage, source exclusion, and card-name listeners. Direct tests cover every
  Mode, Token reference, target departure, empty candidates, hand/board
  capacity, Countdown expiry, seeded replay, backward JSON compatibility,
  clause hashes, and all four RL Mode-mask choices.
- Exact file `data/rules/real_crest_token_activation_exact_batch.json` closes
  eleven collectibles (`10403110`, `10702110`, `10114110`, `10513110`,
  `10423110`, `10321120`, `10644110`, `10841130`, `10163130`, `10462210`,
  `10174110`), bringing collectible exact coverage to 588/735 (80.00%). The
  slice balances Neutral, Forestcraft, Swordcraft, Dragoncraft, Bishop, and
  Portalcraft while covering a spell-like Mode decision on a follower, an
  Engage amulet, four crest/Token chains, selected discard, filtered draw,
  Skybound Art, Enhance-all replacement, effect evolution, and one-shot
  self-replacement Last Words. Generic emblem startability now recursively
  follows the selected branch of `conditional`, including a valid `else`
  branch when its condition is false. Direct tests cover random determinism,
  empty and stale targets, hand/board capacity, countdown timing, dynamic
  non-intrinsic keywords, source/reference hashes, illegal-action immutability,
  and RL Mode, Engage, and target-choice masks.
- Exact file `data/rules/real_existing_primitives_completion_batch.json`
  closes seven collectibles (`10052120`, `10153110`, `10211120`, `10504110`,
  `10812120`, `10813110`, `10861120`), bringing current collectible exact
  coverage to 595/735 (80.95%). The slice reuses ordinary and super evolution,
  summon/draw output bindings, owner-turn listeners, Combo conditions,
  reciprocal referenced followers, full-hand discard, and choose-one. Direct
  tests cover normal and replacement branches, no/stale targets, simultaneous
  deaths, board capacity, dynamic Drain/Barrier/Storm, discard-before-draw,
  cost floors, deterministic replay, illegal command and RL-action
  immutability, action-mask agreement, multilingual text, references, clause
  hashes, and the unchanged 91/91 complete Token Audit.
- Exact file `data/rules/real_existing_primitives_second_completion_batch.json`
  closes twelve collectibles (`10113130`, `10133130`, `10272120`, `10351110`,
  `10461210`, `10523110`, `10654110`, `10733110`, `10734120`, `10803110`,
  `10842110`, `10844120`), bringing current collectible exact coverage to
  607/735 (82.59%). The slice composes hand, board, and turn listeners, Earth
  Rite, Mode, Engage, destroyed-history copies, multi-target discard,
  Accelerate, intrinsic/static traits, and summon/copy bindings. The generic
  board transform primitive now supports follower-to-amulet replacement while
  preserving stable identity, source origin, fused materials, and countdown
  state without firing an enter-play event. Direct tests cover illegal/no/stale
  target paths, hand/board capacity, simultaneous damage, deterministic random
  replay, hidden observation parity, multilingual Mode/reference clauses,
  command/RL action-mask agreement, clause hashes, and the unchanged 91/91
  complete Token Audit.
- Exact file `data/rules/real_existing_primitives_third_completion_batch.json`
  closes ten collectibles (`10234110`, `10234120`, `10352110`, `10353110`,
  `10534120`, `10544110`, `10564110`, `10764120`, `10824120`, `10854120`),
  bringing collectible exact coverage to 617/735 (83.95%). The slice reuses
  Earth Rite, Necromancy, Mode, multi-target choice, summon output bindings,
  random filtered evolution, class-filtered hand cost changes, conditional
  owner-turn effects, board-wide stat changes, and Trait-filtered entry
  listeners. Direct tests cover every Mode and reference, insufficient
  resources, no/stale/duplicate targets, hand/board capacity, simultaneous
  deaths, seeded replay, intrinsic keywords, multilingual source hashes,
  Clause/Token consistency, and command/RL action-mask agreement. Cards whose
  Strike/Clash clauses need a damaged or typed `attack_target` remain in the
  visible backlog until that generic binding boundary is implemented.
- Exact file `data/rules/real_existing_primitives_fourth_completion_batch.json`
  closes eight collectibles (`10153130`, `10323110`, `10413110`, `10442120`,
  `10624110`, `10674110`, `10754120`, `10823110`), bringing collectible exact
  coverage to 625/735 (85.03%). The slice reuses Necromancy, cumulative
  Enhance, Skybound/Super Skybound Art, original-cost and Trait-filtered board
  listeners, Fusion/play event filters, dynamic board counts, repeat, and
  attack-target operations without adding shared-engine code or card-ID
  branches. Direct tests cover every referenced card and embedded Enhance
  clause, resource shortage, no/stale/duplicate targets, hand/board capacity,
  source departure, fixed-seed random replay, multilingual source hashes,
  Clause/Token consistency, and command/RL action-mask agreement.
- Exact file `data/rules/real_crest_listener_burst_existing_fifth_batch.json`
  closes eight collectibles (`10124130`, `10133110`, `10243110`, `10414120`,
  `10714120`, `10724110`, `10833110`, `10864120`), bringing collectible exact
  coverage to 633/735 (86.12%). The slice reuses Countdown crests, Earth Rite,
  Combo, Rally, Trait-filtered summon/spell listeners, dynamic Earth Sigil
  damage, repeated random distribution, selected evolution, Choose One, and
  intrinsic/runtime keywords without shared-engine changes or card-ID branches.
  Direct tests cover every reference and alternate clause, no-target
  continuation, stale/illegal choices, hand/board capacity, source departure,
  once-per-turn behavior, fixed-seed replay, multilingual source hashes,
  Clause/Token consistency, and command/RL action-mask agreement.
- Exact file `data/rules/real_listener_condition_output_binding_sixth_batch.json`
  closes seven collectibles (`10363110`, `10454110`, `10544120`, `10553110`,
  `10744110`, `10843110`, `10851130`), bringing collectible exact coverage to
  640/735 (87.07%). Generic additions include attack-declared listener/emblem
  dispatch with event-source targeting, strict controller-versus-opponent
  leader-health conditions, printed-life deck filters, reanimate output
  bindings, and explicit opponent-emblem targeting. Direct tests cover every
  main/evolved/crest clause and reference, required/no-target behavior,
  stale-choice no-mutation, command/RL masks, full hand/board, simultaneous
  deaths, source departure, fixed-seed replay, multilingual source hashes, and
  Clause/Token audit consistency without card-ID branches.
- Exact file `data/rules/real_keyed_crest_existing_seventh_batch.json` closes
  seven collectibles (`10333110`, `10432120`, `10453310`, `10454120`,
  `10521110`, `10734110`, `10744120`), bringing collectible exact coverage to
  647/735 (88.03%). Generic additions include selected-hand snapshot bindings,
  cross-scope summon-output bindings inside Earth Rite, source-cost propagation
  through conditional frames, keyed crest countdown changes, and a distinct
  crest Last Words lifecycle that fires on both countdown expiry and effect
  destruction without changing legacy `on_expire` semantics. Direct tests
  cover every main/evolved/crest clause and reference, no/stale target paths,
  hand/board capacity, simultaneous deaths, source cost changes, fixed-seed
  replay, command/RL masks, multilingual source hashes, and Clause/Token audit
  consistency without card-ID branches.
- Exact file `data/rules/real_hand_runtime_existing_eighth_batch.json` closes
  seven collectibles (`10302110`, `10303110`, `10022120`, `10722310`,
  `10271120`, `10471120`, `10223110`), bringing collectible exact coverage to
  654/735 (88.98%). Generic additions include persistent and temporary hand
  keyword state, keyword inheritance on ordinary play and exact hand-copy
  summon, successful `add_card` output bindings, filtered Skybound Art gauge
  increments, and implicit-target board-filter revalidation. Direct tests
  cover opponent versus owner Super-Evolution listeners, same-name filtered
  draws and overdraw, Rally and hand capacity, generated Token keyword/stat
  grants, targets leaving hand, gauge events, Enhance legality and zero-cost
  output, damaged versus undamaged Follower Strike targets, fixed-seed replay,
  multilingual hashes, Token Audit continuity, and RL mask parity without
  card-ID branches.
- Exact file `data/rules/real_existing_primitives_ninth_batch.json` closes nine
  collectibles (`10131110`, `10144110`, `10254120`, `10313110`, `10434110`,
  `10532110`, `10634110`, `10862110`, `10871110`), bringing collectible exact
  coverage to 663/735 (90.20%). Generic `summon_exact_copy` preserves live
  runtime stats, evolution, keyword/ability state, restrictions, and selective
  Last Words removal, applies an optional stat delta before the summon event,
  and therefore supports bounded recursive entry chains without card-ID
  branches. Generic `remove_last_words` removes only Last Words while retaining
  Ward, Aura, and all unrelated abilities. Hand-listener
  `buff_hand_card(target=self)` is valid only in its hand-zone schema context.
  Observation v2 adds a public `last_words_removed` bit per board slot, changing
  its public-board runtime vector from 150 to 160 values without changing the
  111 action IDs; the derived formal schema is bumped to
  `observation-v3.1`. Direct tests cover all primary/crest/Mode/reference clauses,
  no/stale/illegal targets, hand/board capacity, source departure, simultaneous
  deaths and trigger order, fixed-seed replay, multilingual hashes, Clause/
  Token/Ability consistency, and RL action-mask parity.
- Exact file `data/rules/real_damage_replacement_binding_tenth_batch.json`
  closes five collectibles (`10163110`, `10401110`, `10464120`, `10711110`,
  `10804110`), bringing collectible exact coverage to 668/735 (90.88%).
  Generic destroy output binding records only successful destruction and
  `bound_target_count` makes its cardinality available to subsequent effects.
  A structured incoming-damage replacement passive resolves before Barrier and
  respects printed-ability removal. Generic `all_board` target filtering,
  `remove_all_emblems`, and exact-copy Rush/Storm readiness cover global Mode
  choices and copied followers without card-ID branches. Direct tests cover
  protected destruction, X=0, simultaneous deaths, threshold/prevention order,
  fixed-seed distinct random targets, multi-target pending choices, no/stale/
  illegal targets, board capacity, source retention, multilingual hashes,
  Clause/Token/Ability consistency, and command/RL action-mask parity without
  changing observation or action schemas.
- Exact file
  `data/rules/real_listener_enhance_random_keyword_eleventh_batch.json` closes
  five collectibles (`10224120`, `10424120`, `10574120`, `10603110`,
  `10622310`), bringing collectible exact coverage to 673/735 (91.56%).
  Generic summons can place the resolved card on either leader's board while
  retaining correct ownership and event attribution. Card and emblem event
  sources are snapshotted at emission, preventing later sources from reacting
  retroactively. Event filters can distinguish Enhance plays; filtered draws
  can require different printed names; Super Skybound Art can explicitly
  replace lower active tiers; and `add_random_keywords` samples a fixed number
  of distinct canonical runtime abilities through engine-owned RNG. Direct
  tests cover enemy-board capacity and trigger order, attack-restriction
  expiry, source departure, 9/10/15 burst thresholds, no/stale/illegal hand
  choices, hand overdraw, first-versus-existing crest activation, multiple
  countdown crests, deterministic replay, multilingual/reference hashes,
  Clause/Token/Ability consistency, and command/RL action-mask parity without
  card-ID branches or observation/action schema changes.
- Exact file `data/rules/real_selected_hand_grants_twelfth_batch.json` closes
  five collectibles (`10111140`, `10272310`, `10273110`, `10412110`,
  `10473110`), bringing collectible exact coverage to 678/735 (92.24%).
  Generic dynamic filtered-draw cost expressions, bound hand-card attack,
  granted Last Words, granted effect-destroy immunity, and physical
  `summon_from_hand` resolution cover the shared behavior without card-ID
  branches. The two granted abilities survive ordinary play and exact copy,
  reset on ability removal/transform/return where appropriate, enter state
  fingerprints and invariants, and are visible in structured observations.
  Direct tests cover Combo timing, Artifact/follower filtering, required and
  optional hand targets, no/stale/illegal choices, hand/board capacity,
  Fanfare suppression, stable hand-to-board identity, simultaneous deaths,
  ability removal, deterministic clone/replay, multilingual/reference hashes,
  Clause/Token consistency, and command/RL action-mask parity. V1 stays at 294
  floats and 111 action IDs; v2/v3 runtime shapes intentionally migrate under
  `observation-v3.2`.
- Exact file `data/rules/real_random_same_name_cost_thirteenth_batch.json`
  closes five collectibles (`10173130`, `10244120`, `10263110`, `10334110`,
  `10532310`), bringing collectible exact coverage to 683/735 (92.93%).
  Generic `halve_round_up` deck modifiers operate sequentially on physical
  current costs; `random_choice` samples distinct structured branches through
  engine-owned RNG; random follower targeting can explicitly exclude the
  current attack target; and `banish_same_name` consumes a bound definition
  snapshot after the selected entity leaves play. Fennie's odd-cost rounding
  and repeated halving are recorded against its official card-page FAQ.
  Direct tests cover linked Puppetry and Clay Golem Tokens, board/hand
  transfer and capacity, simultaneous Last Words, amulet thresholds, attacks
  against followers and leaders, no/stale/illegal targets, ordinary versus
  Super-Evolve behavior, deterministic branch order, Earth Rite shortage,
  multilingual/reference/raw-source hashes, Clause/Token consistency, and
  command/RL action-mask parity without card-ID branches or observation/action
  schema changes.

## Known Partial Or Unsupported Areas

- `repeat` snapshots and sequentially resolves automatic repetitions, but its
  schema rejects nested `requires_target`. Repeated manual selections that must
  affect initial play legality need an explicit command-level availability
  rule before being enabled; empty automatic/random iterations safely no-op.
- Random hand, graveyard, and follower-or-leader targets remain single-target
  operations. Multi-target count fields are schema-rejected for those target
  kinds until a verified real rule requires explicit zone or mixed leader
  selection semantics.
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
  gained structured abilities, Enhance-card progression, and filtered
  named-follower entry progression are implemented, while mode-selection
  progression remains explicit partial semantics. `策动` and `土之秘术` /
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
  Fusion; the current Core → Attack/Castle → α/β/γ → Ω generated Artifact
  chain is exact, while other Fusion cards remain incremental content.
- `瞬念召唤` is implemented for its sole current official card and marked
  implemented in the ability registry. Sandalphon now combines exact
  Invocation, crest, return-to-hand, and `解放奥义` rules.
- The generated-card audit is deterministic and covers all 91 database tokens.
  It does not equate a database reference with an executable entry, and it only
  marks behavior complete for vanilla cards, fully implemented keyword-only
  cards, or explicitly exact structured rules. Current output is 91 complete,
  0 partial, and 0 database-only/no-entry. This closes the imported generated-
  card entry/behavior backlog without implying that uncovered collectible cards
  are implemented.
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
  fits existing primitives. Audit every referenced Token with its producer and
  prefer slices that convert related database-only Tokens into executable
  producer-and-behavior workflows.

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
