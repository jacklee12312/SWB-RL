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
cards: 91 complete entries, 0 partial entries, and no database-only entry gap.

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
clause-audit layer without changing its legacy coverage categories. Of 617
exact collectible cards (83.95% of 735), all 617 have explicit implemented text and named direct
test evidence. The sibling `data/audits/rule_clauses.json` registry hashes every
imported skill and alternate-mode clause, so a database text change or stale
test reference invalidates the audit instead of silently retaining exact
status. The current report has no unverified exact entry or missing generic
schema, primitive, targeting, or timing blocker. Its remaining collectible gaps
are 102 missing per-card structured rules plus 16 explicitly unclear texts.
Rule metadata also supports version and errata fields, and the report
records the complete imported source snapshot hash.

## Reproducible RL Platform

The P1/P2 platform-hardening slice is implemented around the deterministic
rules core:

- Observation v3 and the 111-action layout have named schemas and stable
  SHA-256 manifests. Catalog, card vocabulary, training pool, coverage report,
  RuleBook, observation, action, and seed-derivation versions travel with every
  trajectory and checkpoint; incompatible checkpoints are rejected by field
  name.
- `GameEngine` and `ShadowverseEnv` expose monotonic state/transition versions,
  deterministic snapshot/restore/clone, and version-keyed command, mask,
  observation, and public-zone histogram caches. Retained mutable debug
  references are explicit cache-invalidation boundaries.
- Training mode suppresses unbounded Chinese text logs, bounds diagnostic event
  history, and preserves complete per-transition events and deterministic rule
  behavior. An immutable spawn-safe worker snapshot loads the Catalog and
  RuleBook once, so worker match hot paths do not access SQLite or rule files.
- `VectorRollout` provides persistent Windows/Linux spawn workers, stable
  master/worker/episode/deck/engine/policy seed derivation, ordered trajectory
  identity, timeout/error propagation, graceful shutdown, and a formal
  recurrent trajectory schema. `PolicyVectorRollout` additionally keeps those
  workers alive across PPO updates and collects trajectories under one frozen
  policy generation.
- The CPU baseline is shared-parameter recurrent masked PPO with separate
  player hidden states, sparse terminal reward, terminated/truncated bootstrap,
  recurrent sequence batching, PPO clipping, gradient clipping, and finite-
  value guards. Stable vocabulary indices in hand and public-board slots feed a
  trainable card embedding instead of being treated as dynamically normalized
  ordinal numbers. Atomic schema-v2 checkpoints include the model, optimizer, live
  environment, RNGs, progress, versions, dirty git state, and opponent league.
- The configurable opponent league includes current, historical checkpoint,
  random-legal, and fixed-first-legal policies with reproducible selection,
  periodic snapshots, and bounded retention. New PPO runs use a deterministic
  seven-class 7x7 ordered matchup cycle, while legacy callers retain their
  single-class default. Fixed-seed mirrored evaluation defaults to two exact
  deck pairs for each of the seven classes and reports win/side rates,
  confidence interval, relative Elo, duration, done split, illegal/mask
  consistency, invariant checking, deck/checkpoint/version hashes, and visited
  cards/classes/mechanisms/resources without mutating training state.
- `SWBAECEnv` and the one-learner `SWBGymEnv` wrappers pass the official
  PettingZoo API test and Gymnasium environment checker, respectively.

The checked-in reports under `data/reports/` are reproducibility and smoke
artifacts, not policy-strength claims. The 2026-07-21 embedding/vector CPU
smoke requested 1,024 agent steps and completed 1,304 steps/16 whole episodes
with finite metrics, then resumed to 1,571 steps/20 episodes without reusing or
skipping episode IDs. Whole-episode vector batches may intentionally pass the
requested step boundary. Its 16-game mirrored evaluation is likewise a pipeline
check, not evidence of a strong policy. The 2026-07-22 seven-class smoke adds a
98-episode pre-training distribution audit whose two complete schedule cycles
sample every ordered class matchup twice, balance learner/opponent class counts
at 14 each, and include all 588 exact cards. A separate 28-game evaluation uses
two fixed deck pairs per class with mirrored sides; all 28 games terminated by
rules with zero truncations, illegal actions, or action-mask mismatches. The
current environment
benchmark records 143.79 step/s, 34.75x cached-mask speedup, 21.75x cached-v3-
observation speedup, 24.61 snapshots/s, and 7.23 clones/s on the recorded
machine; the four-worker report records 369.22 rollout steps/s.

Install the optional training stack and run the reproducible entry points with:

```powershell
python -m pip install -e ".[rl,train]"
python -m scripts.vector_rollout --workers 4 --episodes 16
python -m scripts.audit_rl_distribution --episodes 98 --workers 2
python -m scripts.train_ppo --total-agent-steps 10000
python -m scripts.train_ppo --rollout-workers 4 --total-agent-steps 10000 --opponent-current-weight 1 --opponent-random-weight 0 --opponent-fixed-weight 0 --opponent-historical-weight 0
python -m scripts.evaluate_ppo data/checkpoints/ppo_smoke.pt
python -m scripts.benchmark_rl_env
```

Still unsupported: this is a baseline PPO and league/evaluation system, not a
distributed learner, a policy-strength result, or a complete MCTS
implementation. Multiprocess PPO currently uses current-policy self-play;
random, fixed, and historical opponent mixing remains on the single-process
collector. The balanced class schedule is not an adaptive curriculum, and the
same-class fixed evaluation suite is not yet a 7x7 cross-class policy-strength
matrix. Snapshot/clone is the search foundation only. Card-rule
coverage also remains deliberately separate: 102 collectible cards still lack
per-card structured rules and 16 card texts remain explicitly unclear; neither
group enters the exact training catalog or counts as supported.

## Implemented Engine Surface

The deterministic rules core supports:

- generic leader maximum-health state, healing caps, invariant/fingerprint
  coverage, v2 public observation, and an auditable `set_leader_max_health`
  operation. Exact `10104120` replaces its controller's deck with the official
  ten-card Apocalypse Deck (three copies each of `90004110`, `90004120`, and
  `90004130`, plus `90004310`) using the engine RNG. The four generated cards
  cover Storm, vanilla stats, up-to-two selected follower damage with the
  official zero-target leader-damage behavior, and setting the enemy leader's
  maximum health to 1 without dealing damage. Seeded replay, stale targets,
  board shortage, healing after the cap change, official FAQ semantics, and
  unchanged v1/action layouts are directly tested;
- existing cooperation, filtered board-count, countdown/activation, emblem
  expiry, intrinsic keyword, summon-output, evolution, banish, and healing
  primitives close exact rules for twelve Royal/Bishop cards (`10723310`,
  `10263310`, `10062210`, `10362220`, `10122140`, `10421120`, `10421130`,
  `10821130`, `10761110`, `10662120`, `10562110`, `10762120`). Direct tests
  cover target-required no-mutation and RL masks, the Rally replacement's
  zero-target path, stale selections, printed summon order, Token references,
  board shortage, paired/self producer chains, Last Words, and seeded replay;
- a keyword-aware `BoardFilter` and board-count expression inspect both printed
  and runtime-granted keywords. Fifteen mixed Royal/Runecraft/Bishop exact cards
  (`10121140`, `10322210`, `10623110`, `10822110`, `10031110`, `10331310`,
  `10531110`, `10631120`, `10633110`, `10731110`, `10833310`, `10463210`,
  `10562210`, `10662110`, `10763110`) exercise Enhance, Engage choices,
  spell listeners, Earth Rite, crests, Super Evolve, filtered Token buffs,
  Crystallize, Countdown, Last Words, deterministic random effects, and RL masks.
  A source-card-type condition keeps alternate-form Last Words on the amulet
  form and prevents the same-ID follower from inheriting them;
- an 11-card exact spell/amulet/crest batch covers `10412310`, `10441310`,
  `10451310`, `10712310`, `10713310`, `10233310`, `10352210`, `10633310`,
  `10413310`, `10332110`, and referenced follower `10631110`. It reuses Combo,
  countdown crests, Earth Rite, Engage, Mode/Enhance, Super Skybound Art,
  source-cost conditions, repeated summons, and successful-output bindings.
  Direct tests cover paired/self card generation, owner-turn scope, empty
  random targets, hand/board capacity, modified play cost, deterministic replay,
  and RL action-mask parity;
- a 12-card exact Dragon/Nightmare Token-chain, Forest amulet, and Bishop spell
  batch covers `10143130`, `10741110`, `10641120`, `10651120`, `10252120`,
  `10152130`, `10552110`, `10851120`, `10154120`, `10262310`, `10113210`,
  and `10011210`. A generic `bound_target_health` expression reads the current
  live defense of one earlier selected target after intervening buffs, while
  existing Enhance, intrinsic keyword, Last Words, Trait listener, Countdown,
  Engage, Combo, attack-capacity, and output primitives express the remaining
  clauses. Direct tests cover Token order and references, simultaneous deaths,
  hand/board capacity, no-target atomicity, stale selections, post-buff damage,
  deterministic random targets, non-matching Traits, and RL action-mask parity;
- a 15-card balanced exact batch covers Forest attack/Token chains, Royal draw
  and turn-end interactions, Runecraft Spellboost and Earth Rite resources, a
  Dragoncraft targeted summon, and a Bishop Crystallize chain (`10411110`,
  `10212110`, `10811110`, `10611110`, `10614110`, `10213110`, `10123110`,
  `10522120`, `10522110`, `10134310`, `10632120`, `10132130`, `10533110`,
  `10542110`, `10661110`). It reuses attack capacity, Trait filters, seeded
  repeat, hand and board listeners, source Spellboost values, Enhance,
  Crystallize, Countdown, Last Words, and source-form guards. Direct tests cover
  no-target and stale-target continuation, hand/board capacity, simultaneous
  deaths, cost expiry, deterministic shuffling, resource shortage, Token
  references, clause hashes, and RL mode masks;
- a 12-card exact Rune/Nightmare/Royal/Neutral batch covers `10032110`,
  `10132120`, `10433110`, `10732110`, `10452130`, `10752110`, `10852110`,
  `10852120`, `10423310`, `10524120`, `10723110`, and `10603210`. Generic
  `leftmost` candidate selection preserves filtered board order without RNG,
  `exclude_tribe_name` expresses all non-Encroacher followers, and effect-
  evolved target bindings remain valid for printed follow-up buffs. The slice
  exercises Earth Rite, Spellboost, Token chains, four- and two-way Modes,
  ability removal, crests, Countdown, simultaneous damage, EP/PP recovery,
  hand/board capacity, stale/no-target paths, seeded replay, and RL choice masks;
- an 11-card exact cross-class crest, Token, Mode, and Engage batch covers
  `10403110`, `10702110`, `10114110`, `10513110`, `10423110`, `10321120`,
  `10644110`, `10841130`, `10163130`, `10462210`, and `10174110`. It reuses
  Skybound Art, filtered draw, effect evolution, self-replacement Last Words,
  selected discard, Enhance-all replacement, hand-zone Engage listeners,
  Trait-filtered entry crests, and countdown expiry. Generic emblem preflight
  now evaluates the executable `else` branch of a false `conditional` instead
  of silently suppressing it. Direct tests cover every primary and crest text,
  Token references, no-target continuation, illegal Engage immutability, stale
  targets, hand/board capacity, seeded random damage, countdown timing, and RL
  Mode/target masks;
- a 7-card exact existing-primitive batch covers `10052120`, `10153110`,
  `10211120`, `10504110`, `10812120`, `10813110`, and `10861120`. It composes
  evolution/super-evolution replacement, summon and draw output bindings,
  Combo replacement, owner-turn listeners, reciprocal referenced followers,
  whole-hand discard, and a two-way Mode without card-ID engine branches.
  Direct tests cover stale/no-target paths, simultaneous deaths, one-slot and
  full-board capacity, dynamic keywords, discard/draw ordering, illegal command
  rollback, fixed-seed replay, clause hashes, Token Audit, and RL action-mask
  agreement;
- a 12-card exact cross-class existing-primitive batch covers `10113130`,
  `10133130`, `10272120`, `10351110`, `10461210`, `10523110`, `10654110`,
  `10733110`, `10734120`, `10803110`, `10842110`, and `10844120`. The generic
  board transform operation now accepts follower-to-amulet replacement while
  preserving stable entity identity, source origin, fused materials, and
  countdown state without emitting a false enter-play event. Structured rules
  compose hand/board/turn listeners, Earth Rite, Mode, Engage, random destroyed-
  history copies, multi-target discard, alternate Accelerate, intrinsic/static
  traits, and summon/copy bindings. Direct tests cover no/stale targets,
  illegal-command rollback, hand/board capacity, simultaneous damage, seeded
  replay and hidden-information parity, multilingual/Mode/reference clauses,
  Token Audit continuity, and command/RL action-mask agreement;
- a 10-card exact cross-class existing-primitive batch covers `10234110`,
  `10234120`, `10352110`, `10353110`, `10534120`, `10544110`, `10564110`,
  `10764120`, `10824120`, and `10854120`. It composes Earth Rite and
  Necromancy gates, repeated Mode abilities, multi-target selection, summon
  output binding, random filtered evolution, class-filtered hand cost changes,
  owner-turn branches, intrinsic traits, and a Soldier-entry listener without
  card-ID engine branches. Direct tests cover all modes and referenced cards,
  insufficient resources, no/stale/duplicate targets, hand/board capacity,
  simultaneous deaths, fixed-seed randomness, multilingual clause hashes,
  Token Audit continuity, and command/RL action-mask agreement;
- seeded reset, shuffle, draw, turn progression, official immediate defeat when
  drawing from an empty deck, an auditable alternate victory outcome, atomic
  seeded replacement-deck shuffling,
  reproducible command sequences, and deterministic full-state fingerprints for
  replay diagnostics;
- generic crest `on_gain` operations, exclusion filters shared by hand targeting
  and hand-count expressions, and public v2 observation of each leader's empty-
  deck outcome. Exact `10304110` replaces its controller's deck on evolution
  with the 76 other cards from imported set `10003`, changes empty-deck draw to
  victory, discards every non-`90004320` hand card and draws six at each owner
  turn end. Its Fanfare produces exact target-required destroy spell `90004320`.
  Empty-target prohibition, stale target revalidation, seeded replay, ignored
  duplicate crest gain, RL masks, and the normal defeat path are directly tested.
  The end condition follows the [official Victory Card glossary](https://shadowverse-wb.com/ja/help?tab=tab0)
  and the crest text on the [official card page](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=10304110);
- generic frozen source-Spellboost value expressions and seeded hidden-position
  `add_card_to_deck` operations. Exact `10131320` and `10831310` deal their
  printed 2 plus the played copy's Spellboost count; exact `10551310` and
  `10843310` recycle one copy after damage, with the latter drawing afterward
  only in Overflow. Exact `10341110` listens for its own survived damage only
  during its controller's turn and draws a Dragoncraft follower, while exact
  `10752310` summons and effect-evolves Ghost, Bat, and Skeleton in printed
  order. Required-target rollback, seeded deck order, board-capacity shortage,
  listener turn/lethal boundaries, RL masks, official references, and both
  reprints are directly tested;
- generic draw-output identity bindings retain the physical card and its frozen
  current cost across hand entry or overdraw, while named bound-card expressions
  skip dependent random effects without consuming RNG when no output exists.
  `random_enemy_hand`, snapshot-backed `copy_to_hand`, and `summon_copy` preserve
  hidden information and printed effect order after a selected target leaves
  play. Exact `10853310`, `10541310`, `10802310`, `10443310`, and `10652310`
  cover post-draw cost changes, drawn-cost damage, unrevealed opponent-hand
  copying, banish-then-copy-to-hand, and banish-then-copy-summon. Direct tests
  cover Super-Evolution unlock timing, overdraw, empty candidates, seeded
  replay, stale choices, full boards, opponent observations, and unchanged RL
  action/observation layouts;
- generic effect frames freeze a played or discarded hand card's physical
  current cost, exposing it to structured conditions and value expressions
  without retaining a live hand dependency. Destroyed-follower history can be
  filtered and sampled either by destroyed instance or by distinct card name,
  with empty-history RNG stability, hidden generated copies, and hand-overflow
  handling. Exact `10572310`, `10871130`, `10641310`, `10643310`, and
  `10331110` cover distinct-name revival, Artifact-only revival, cost-gated
  discard replacement, a 7-to-5-to-3 discard chain, current-cost area damage,
  and nonprinted-cost healing. Direct tests cover simultaneous destroyed
  history, deterministic replay, opponent observation privacy, no discard
  candidate, illegal target rollback, and unchanged RL layouts;
- generic hand filters can select by normalized ability keyword in addition to
  card type, class, current physical cost, identity, and Trait. Exact `10131310`
  uses the new boundary to prohibit play without an On-Spellboost hand target, while
  `10232310` resolves its frozen Spellboost value through the established
  distributed-damage procedure. Exact `10121130`, `10212120`, `10252110`,
  `10361120`, `10673310`, `10721110`, and `10812110` extend the existing
  summon, Super-Evolution, and positive-stat listener boundaries; exact
  `10353310` exercises every pair of its four modes, and `10612110` adds its
  referenced Token. Direct tests cover all six mode pairs, official no-target
  and set-stat FAQ semantics, once-per-own-turn reset, temporary cost expiry,
  seeded random evolution/damage, Token capacity, and unchanged RL layouts;
- generic `summon_hand_copy` keeps the selected physical card in hand, copies
  its current attack/health modifiers into a new follower, and output-binds
  only entities that actually entered play. Selected hand sets support explicit
  full-count availability, so cards can distinguish "must select exactly N"
  from "select up to N" without card-ID conditionals. Granted owner-turn or
  opponent-turn self-destruction is entity state, is removed by ability
  removal, and resolves before the active-player flip so Super-Evolution's
  own-turn effect-destroy protection remains correct. Exact `10172320`,
  `10173140`, `10174130`, `10261120`, `10271210`, `10274120`, and `10572110`
  cover two- and three-card hand choices, current-cost filtering, copied stat
  modifiers, board shortage, Core/Artifact token production, Engage, and both
  Evolution paths. Official shortage/no-target/effect-destroy-protection rulings,
  stale multi-hand choices, deterministic state, RL masks, and v2 public board
  ability bits are directly tested;
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
- source-backed ability-target forcing: while Lloyd is in play with its printed
  ability intact, opposing manually selected abilities can choose only Lloyd
  among opposing cards and cannot choose the opposing leader. Own-side targets,
  attacks, random effects, and all-target effects remain unchanged. Multiple
  Lloyds, target filters, ability removal, leaving play, and pending choices all
  re-evaluate the centralized candidates;
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
  discard, hidden board-card copying, transform, explicit target-set binding,
  stat changes, keyword
  changes, cost modifiers, and attack or targeting restrictions;
- official oldest-first damage distribution across enemy followers, with each
  earlier follower capped by its current health and any remainder assigned to
  the last follower or, when explicitly included, the enemy leader. Allocation
  is fixed before Barrier or other prevention resolves, so prevented damage is
  not redistributed. The rule follows the
  [official help procedure](https://shadowverse-wb.com/ja/help?tab=tab0);
- selected, random, all, implicit, leader, board, all-board, hand, and
  graveyard target flows, including pending choices, target-leaves-play safety,
  and target revalidation when a pending target changes controller or no longer
  matches the original target filter; ordinary play choices cover selected
  board targets moving to the graveyard or changing controller, selected hand
  cards leaving hand, and selected graveyard cards moving to hand before
  resolution resumes; `target_key` stores an ordered tuple of selected board
  targets or the exact entities successfully created by `summon`,
  `summon_hand_copy`, and `summon_from_deck`. `previous_target` chains revalidate selected members
  against the original target filter and created members by stable entity ID;
  failed summons bind an empty tuple, and multi-output deck summons preserve
  only the entities that actually entered play;
- structured hand filters match printed type/class/identity/name/Trait and the
  physical hand card's current cost. Selected, random, and all-hand flows share the same centralized
  candidates with play legality, stale-choice revalidation, and RL masks.
  Exact `10333310` uses the type filter to expose only followers before adding
  1 to the selected card's cost and destroying a seeded-random enemy follower;
- structured hand-follower stat modifiers retain auditable duration and
  identity, appear in own-hand observations, and transfer to the follower when
  it is played; hand transform, deck return, and board return reset runtime stat
  changes through their normal zone semantics. Exact `10172130` adds and then
  buffs three Puppets, `10343110` gates a Dragoncraft hand-wide +1/+1 on its
  live end-of-turn defense, and `10772310` gates a follower hand-wide +1/+0 on
  Super Evolution unlock without exposing opponent hand contents;
- filtered hand-count conditions reuse the same type/class/identity/trait
  definition filters as hand targeting. Exact `10521120` counts only spells
  before conditionally gaining +1/+1 and Ward, while exact `10741120` and
  `10853110` expose selected follower-only hand buffs through existing command
  choices and RL masks. Exact `10112210` additionally covers Combo-gated Token
  generation and a target-required zero-cost Engage that continues after its
  source amulet destroys itself;
- filtered hand-count expressions use those same definition filters for dynamic
  values. A bounded `repeat` meta-effect evaluates its count once (maximum 100),
  then resolves the nested operation sequence one iteration at a time. Random
  candidates are rebuilt with the engine RNG and events, deaths, and other
  state-based checks stabilize before the next iteration; an empty candidate
  set is a no-op and consumes no randomness. Exact `10114130` counts Pixie
  followers for both its Fanfare stats and independently reselected Evolve
  hits, while exact `10313310` repeats random health loss using current Combo;
- structured `add_shadows` changes either leader's public shadow count, supports
  dynamic expressions, and emits the same auditable `SHADOWS_CHANGED` boundary
  used by ordinary graveyard entry and Necromancy spending. Exact `10152210`
  gains two shadows on entry, destroys itself through zero-cost Engage, and
  summons up to two board-capacity-limited Ghosts; exact `10153120` combines
  Fanfare shadow gain and Ward with a single Necromancy payment followed by two
  independently resolved random hits. The rule loader also requires `summon`
  to use its implicit `own_leader` destination so an accidental board-choice
  target cannot silently skip an otherwise authored summon;
- a 10-card existing-primitives follow-up adds three exact Reanimate followers,
  Combo and allied-board dynamic self buffs, an other-card printed-cost gate,
  filtered amulet/follower draws, live hand-type damage, two-target evolution
  destruction, Clash and Last Words hand Spellboost, and per-Spellboost cost
  reduction. Direct tests cover normal versus Enhance modes, ordered summons,
  board capacity, both battlefields, empty and two-step choices, seeded ties,
  fingerprints, intrinsic keywords, and RL mode/choice masks;
- physical deck-card cost modifiers are copy-specific, stackable, floor at
  zero, survive shuffle and draw, and participate in fingerprints and
  invariants. A hand transform can atomically attach a duration-aware cost
  modifier to its replacement. Exact `10334120` gives every follower currently
  in its deck `-3` cost, transforms one seeded-random hand spell into exact
  generated spell `90034310`, and sets it to 0 cost until turn end. That spell
  permanently gives every allied hand follower +1 cost before destroying all
  enemy followers, and its executable transform producer promotes it to a
  behavior-complete generated card;
- public per-turn follower-attack history increments on every legal attack
  declaration, remains true when an attack trigger removes the attacker, resets
  at the turn boundary, participates in fingerprints/invariants, and is exposed
  to RL without changing action IDs. Structured conditions can gate effects on
  that count, while `all_own_emblems` applies an ordered countdown increase to
  every allied crest that actually has a countdown. Exact `10364120` generates
  exact spell `90064310`, gains a nonstacking evolution crest, and distributes
  damage equal to the live allied-crest count at turn end only if no allied
  follower attacked. The spell seeded-randomly banishes one enemy follower,
  then increases all allied crest countdowns by 1; empty enemy candidates do
  not consume RNG and do not stop the countdown operation;
- selected multi-target pending choices through `target_count` or
  `target_count_expr`, with explicit `allow_duplicate_targets` /
  `allow_duplicates` policy, candidate-shortage handling, command-level
  selection progress, ordered multi-target `target_key` bindings, and
  per-target revalidation before resolution; real card `10351120` demonstrates
  selecting and destroying two enemy followers. Exact `10474120` reuses the
  same selected set for ability removal followed by damage, including
  candidate shortage and targets leaving before resolution;
- random board operations also consume `target_count` or a dynamic
  `target_count_expr`. They select one seeded batch before execution, default to
  distinct targets, cap to available candidates, and defer state-based checks
  until the batch completes. Explicit duplicate selection remains available;
  fixed printed repetitions can stay separate operations and dynamic
  repetitions use `repeat`, with both forms reselecting and stabilizing between
  iterations;
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
  shared leader area. Rules can observe `amulet_activated`, `card_drawn`, `card_fused`,
  `earth_rite_activated`,
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
  events that structured emblems can observe. An activation definition is also
  a valid ordinary-play anchor for an otherwise effectless amulet, with matching
  command and RL masks. Generic `reduce_countdown` clamps at zero and expires an
  amulet through the normal death/Last Words pipeline. Eight additional exact
  real rules cover activation-only play, self-destruction, targeted buffs,
  keyword removal, healing, hand cycling, countdown state, and all-follower or
  selected damage;
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
- command-level `融合` from hand, including structured material filters (single
  card IDs or explicit card-ID whitelists) and
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
  demonstrate Artifact filters and cumulative-cost transformation. The exact
  α/β/γ chain additionally verifies β-or-γ-only fusion, distinct material-kind
  counting, and the Ω end-form transform;
- emblem event filters can inspect each material consumed by one `card_fused`
  event while accepting at most one trigger for that Fusion action. Explicit
  `emblem_self` targeting lets a crest change its own countdown without
  changing the established event-source meaning of `self`. Exact `10324120`
  uses those boundaries for Octrice's Treasure-play/Fusion countdown and
  generated Remnant payoff, including the official one-decrement rule when two
  Treasure cards are fused together;
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
- selected, random, and all board targets support explicit dynamic source
  exclusion across follower/amulet mixed zones. The same filtered candidate set
  drives play legality, commands, pending choices, revalidation, and RL masks;
  `10664120` demonstrates a three-other-card choice and `10121120` demonstrates
  an automatic all-other-followers effect;
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
- direct exact-card audits lock `10041130` leader damage, `10051120` self
  damage, `10061110` heal/Ward, `10132320` health-setting and timed attack
  restriction, `10161130` draw-plus-heal, `10431120` source-Attack-scaled
  whole-hand Spellboost plus its permanent post-Evolve attack restriction,
  `10551120` Reanimate, and `10642310` staged discard/destroy choices;
- a 12-card exact basic-spell batch covers repeated Token summons, plain and
  filtered target damage, leader healing, simultaneous all-follower damage,
  selected destruction/banish, seeded random hand cycling, multi-target buffs,
  and explicit whole-hand Spellboost. `10101310` now supplies the executable
  producer for vanilla Token `90001110`, including board-capacity and Token
  origin tests; `10751310` directly locks command legality, stale target
  revalidation, candidate shortage, and the matching RL action mask;
- a 13-card exact basic-follower batch covers Fanfare and Last Words destruction,
  banish, leader healing, source-excluding ally buffs, filtered spell draw,
  simultaneous enemy-wide damage, and one- or three-card death draws. Direct
  repository-backed tests also lock each card's normalized Ward, Storm, Rush,
  Bane, Intimidate, or granted Barrier state and the source-excluding RL choice
  mask;
- a 9-card exact Dragoncraft Overflow batch covers the 6/7-max-mana boundary,
  conditional Storm and Intimidate grants, conditional healing and draws,
  source-excluding replacement buffs, class/type-filtered draw, and independent
  seeded random hits. Direct tests also prohibit a selected-target spell when no
  allied follower exists and lock the corresponding RL mask;
- a 12-card exact evolution batch covers Fanfare/Evolve replay, Last Words plus
  Evolve draws, all-other-follower buffs, all-hand Spellboost, conditional
  self-damage/healing, selected destruction and damage, simultaneous damage and
  health-reduction deaths, and Super-Evolve draws or Storm grants. Static Guard
  and Ambush plus the real Super-Evolve target/RL mask are audited directly;
- a 12-card exact evolution follow-up batch covers filtered draws, PP recovery,
  single and enemy-wide health reductions, repeated damage or destruction,
  explicit Combo gain even without an enemy target, whole-hand Spellboost,
  normal and Super-Evolve stat/ability triggers, and a temporary attack lock.
  Direct tests also lock intrinsic Ambush/Guard, evolve-only Storm provenance,
  lethal state checks, and RL mask parity;
- a 12-card exact core-primitives batch covers per-Spellboost cost reduction,
  Combo-based draw and automatic evolution, Enhance buffs and keyword grants,
  selected hand cycling/discard, own-turn-end draw, all-ally effect evolution,
  source-excluding cross-controller targeting, Necromancy healing, and
  destruction followed by Reanimate. Direct tests cover insufficient-resource
  and no-target fallthrough paths plus intrinsic Ambush/Rush provenance;
- a 6-card exact intrinsic-keyword batch introduces a schema-validated
  `intrinsic_keywords` declaration for followers whose complete text is only
  printed runtime keywords, without inventing empty trigger effects. Real-card
  tests cover Storm/Bane, Bane/Ward, Storm/Intimidate, Ward/Barrier,
  Ambush/Bane, and Ward/Aura initial state, combat legality, aliases, source
  hashes, coverage evidence, and RL attack-mask parity;
- a 5-card exact complete-Token follow-up batch covers two-Fairy Fanfare hand
  generation, ordered Forest's Mystery plus draw, evolve-time double Bat
  summoning, double-Skeleton Last Words, and a Rush/Bane/Drain follower. Direct
  tests lock hand and board capacity boundaries, Token origins and producer
  evidence, manual evolution, combat behavior, deterministic replay, source
  hashes, and RL attack-mask parity;
- a 9-card exact complete-Token board batch covers repeated Fanfare/Evolve
  summons and hand generation, summon-before-buff ordering, draw-before-summon,
  selected evolve damage, and Knight/Skeleton Last Words. Direct tests cover
  full hand and board continuation, no-target and stale-target paths, illegal
  evolution no-mutation, Token origins and keywords, deterministic replay,
  source hashes, producer auditing, and RL target-choice mask parity;
- a 5-card exact super-evolution-unlock batch adds config-aware controller and
  opponent unlock conditions shared by ordinary and target-dependent rule
  evaluation. Real-card tests cover first/second-player boundaries, conditional
  Bane/Barrier/stat buffs/healing/effect evolution, intrinsic Rush/Ward/Storm,
  combat resolution, health caps, deterministic replay, source hashes, and RL
  play/attack-mask parity;
- a 5-card exact evolution-replacement batch adds a source-snapshot-aware
  `source_super_evolved` condition so ordinary Evolve effects are skipped when
  Super Evolve text says “instead.” Direct tests cover exact 2-versus-4 healing,
  selected-versus-all damage and banish, one-versus-two self-summons, Ward,
  board and health caps, health-filtered/no-target paths, distinct target
  selection, target leave-play revalidation, deterministic replay, hashes, and
  RL choice-mask parity;
- a 5-card exact evolution-resource batch adds capped, event-audited EP and SEP
  restoration primitives. It covers Union Burst threshold recovery, intrinsic
  Rush/Ward, spell and follower Mode choices, leader and simultaneous board
  damage, healing, draw, mana recovery, illegal-choice immutability,
  deterministic continuations, source hashes, and RL masks. Random and
  all-target effects now remain executable as safe no-ops on empty candidate
  sets, while selected and explicit `requires_target` effects stay prohibited;
- an 8-card exact selected-hand exchange batch covers optional hand return or
  discard followed by filtered/plain draw, Earth Sigils, simultaneous enemy
  board damage, exact Fairy generation, leader healing, and repeated Evolve
  text. Direct tests lock effect ordering, no-other-hand continuation, hand
  capacity, Last Words, Rush/Barrier/Ward combat, stale and illegal choices,
  deterministic replay, source hashes, Token producer evidence, and RL masks;
- a 10-card exact existing-condition/direct-effect batch covers post-play Combo
  thresholds, evolved-ally and three-amulet conditions, filtered banish,
  draw-then-hand-count healing, damage/heal continuation, Union Burst,
  Enhance effect evolution, and intrinsic Rush/Aura/Ambush. Ordinary spells
  and amulets now consume their precomputed Union Burst operations and emit the
  same threshold events as followers. Direct tests cover exact-threshold and
  below-threshold paths, target filters, simultaneous damage, health/resource
  caps, source exclusion, deterministic replay, hashes, and RL masks;
- an 11-card exact evolution/Burst/direct-effect batch covers turn-end and
  max-mana effect evolution, multi-target Fanfare damage, Earth Rite, intrinsic
  Drain/Storm, spell-triggered board listeners, exact-health mass destruction,
  asymmetric draws, and evolved-state healing. Structured Super Skybound Art
  can now explicitly replace a card's base operations instead of being forced
  to append to them; the ordinary append behavior remains the default. Direct
  tests lock the 3-versus-6 damage replacement, threshold boundaries,
  simultaneous effects, candidate shortages, deterministic replay, imported
  source hashes, and RL choice/action-mask parity;
- a 10-card exact board-state-filter batch adds reusable `super_evolved` and
  `damaged` dimensions to board filters used by conditions and selected,
  random, or all-target operations. It covers super-evolved-board Fanfare and
  turn-end gates, a conditional emblem, an original-cost-three hand listener,
  damaged-only destruction, Enhance health, Countdown draw/heal replacement,
  Ward, Earth Sigils, and a referenced exact amulet. Direct tests distinguish
  ordinary evolution from super evolution and full health from damaged state,
  and lock illegal-choice immutability, seeded random selection, hashes, and RL
  choice-mask parity;
- an 8-card exact extreme-candidate batch adds schema-validated highest/lowest
  current Attack or Health filtering after ordinary target legality. Random
  effects select only among tied extrema, all-target effects snapshot every tie,
  and the new all-leaders target can affect one or both leaders when their
  current Health is tied. Two referenced Tokens become complete executable
  entries with Ward/turn-end evolution and Rush. Direct tests cover no-target
  continuation, seeded ties, simultaneous target snapshots, Overflow, Enhance,
  Earth Rite, Mode choices, source self-evolution, hashes, and RL masks;
- a 5-card exact direct-deck-summon batch adds filtered follower/amulet entry
  from the physical deck. Selection follows the official direct-summon rule:
  duplicate copies increase a name's probability, but only one copy of the same
  name can enter during one operation; selected cards and their entry order
  remain seeded, distinct, board-cap limited, and carry Deck origin. Direct
  tests cover candidate and board shortages, no-Fanfare entry, Countdown,
  class/type/cost filters, Cooperation events, follow-up Evolve/Super Evolve
  choices and buffs, deterministic replay, source hashes, and unchanged RL IDs;
- a 7-card exact summon-output/Trait batch adds class- and Trait-aware board
  filters plus identity-safe summon result bindings. It covers Royal and
  Nightmare board-wide effects, generated followers that must immediately gain
  Rush/evolve, entry listeners, per-entry Spellboost, and a terminating Rotten
  Zombie Last Words that summons an exact replacement and removes its printed
  abilities. Guardian Golem's Ward and the previously vanilla Steelclad Knight
  are audited alongside their producers;
- a 10-card exact established-primitives batch adds four additive Enhance
  followers, three Engage amulets, one Countdown/Last Words amulet, and two
  ordered Fanfare followers. It directly covers normal-versus-Enhance behavior,
  filtered draws and PP restoration, double attacks, source-excluding buffs,
  self-destruction before suspended Engage choices, all-enemy damage, healing,
  intrinsic versus granted keywords, and Countdown expiry;
- a 15-card exact listener/evolution batch adds owner amulet-activation,
  follower-entry, hand evolution and hand super-evolution listeners alongside
  Combo, Enhance, Evolve, Super Evolve, Earth Sigil and Spellboost effects.
  Direct tests lock owner scope, stable hand identity, normal-versus-super
  evolution event behavior, source exclusion, filtered hand choices, intrinsic
  versus granted keywords, and ordered follow-up effects;
- a 15-card exact repeated-evolution/listener batch adds filterable draw-event
  subjects and owner-turn draw listeners alongside repeated Fanfare/Evolve
  effects. It covers self-excluding grants, automatic and selected Super
  Evolution, ability removal with bound targets, optional sacrifice/discard
  continuations, Earth Sigils, attack-lock expiry, Countdown activation and
  Last Words, Combo thresholds, intrinsic keywords, and turn-end all-unit
  damage;
- a 15-card exact spell/mode batch adds ordinary `earth_rite_activated` hand
  listeners and permits Enhance operations to consume target bindings produced
  by their base spell operations. It covers five Mode spells, Treasure Fusion,
  Overflow filtered draw, dynamic all-board damage, exact multi-summons,
  draw-before-random-damage ordering, Reanimate, stacked turn-scoped hand cost
  reduction, and board-cap-safe Enhance buffs of only successful summons;
- a generated-card completion batch closes 12 Token behaviors and nine real
  producer cards. It covers reciprocal Countdown amulets, opponent-turn-end
  Puppet destruction, Shikigami Last Words Spellboost, Mimi/Coco Last Words,
  canonical Earth Sigil activation, Storm/Rush/Aura tokens, Necromancy and
  repeated Reanimate, while every Token now has an executable producer path;
- an 8-Token exact Artifact fusion batch completes Future/Past Core,
  Attack/Castle Artifact, Destruction Artifact α/β/γ, and Superior Artifact Ω.
  It covers unplayable cores, exact Artifact material filters, cumulative-cost
  branching, β/γ card-ID whitelisting, two-distinct-kind transformation,
  end-of-turn heal/damage, Ω Fanfare damage/heal, and Rush/Ward/Storm/Aura;
- a 10-card Portalcraft producer batch closes the early Future/Past Core,
  Puppet/Improved Puppet, and Attack/Castle Artifact entry chains. It covers
  ordered multi-card generation, intrinsic Rush, required-target atomicity,
  Countdown and owner-turn timing, evolution repeat effects, targeted evolution
  with a no-target skip, board-capacity failure, generated origins, and seeded
  replay without adding new engine or RL branches;
- a second 10-card Portalcraft follow-up closes Puppet/Artifact listeners,
  additional producer chains, optional other-card destruction, and output-bound
  keyword/stat grants. Official Automata Assassin ordering gives Bane only to
  the first of two Puppets entering in one turn; multi-material Fusion triggers
  Heritage Barrage once. Tests also cover evolution filters, mixed-board Super
  Evolution destruction, effect-destroy immunity, Enhance output shortage,
  Last Words, generated origins, and deterministic replay;
- a 7-card Portalcraft Artifact-history batch records every successful follower
  entry as public deterministic match history and counts matching followers by
  different printed names. Structured conditions and dynamic expressions filter
  that history by card type and Trait. Dope Dancer, Street Run, Bold Painter,
  Teleport Slash, Scarlet, Myuu, and The Journey Ahead cover threshold branches,
  Mode/RL choice masks, dynamic all-enemy damage, required-target atomicity,
  board shortage, evolution-listener ordering, and EP recovery. The final two
  v1 observation features expose both players' public Artifact-kind counts;
  the action layout remains unchanged. The semantics follow the official
  [Street Run](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10771310),
  [Teleport Slash](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=10773310),
  and [Myuu](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10774120)
  card pages;
- a 5-card Portalcraft dynamic-evolution batch adds filtered live-board count
  expressions and lets `clash` rules bind the opposing follower through the
  existing `attack_target` identity. Assertion of Destruction counts every
  other allied board card before destroying them; Neural Blocker counts only
  other followers, excluding amulets. Eustace resolves target-gated Skybound
  Art double evolution and damages the opposing follower whether attacking or
  defending; Substandard Puppet covers normal self-copy double evolution plus
  Accelerate 3; Ashray & Lishenya separates non-intrinsic Ward/Storm mentions,
  grants Ward to the selected enemy, and on Enhance 9 evolves before randomly
  destroying up to two current Ward followers. Tests cover no-target skips,
  board shortage, illegal-mode fingerprint immutability, seeded replay, and RL
  mode/target masks. The rules follow the official
  [Eustace](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=10472110),
  [Substandard Puppet](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=10672110),
  and [Ashray & Lishenya](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=10874110)
  card pages;
- a final partial-Token batch completes Ghost and Improved Puppet. The generic
  `banish_on_leave` passive replaces destruction, return-to-hand, and
  return-to-deck destinations with banishment while respecting removed printed
  abilities; it emits auditable replacement causes and adds no graveyard,
  Shadow, Last Words, or Reanimate history. Real producers and both turn-end
  expiry boundaries are directly tested;
- a six-producer/six-Token exact batch adds Roaring Dragoneer, Cloudsea Dragon
  Rider, Otohime Fan, Wolong, Projected Bird Statue, and Nahato & Vincent. It
  closes the Giantwing Dragon, Otohime Guard, Gold Dragon, Silver Dragon,
  Majestic Falcon, and Nahato's Private Soldier workflows with Enhance, paid
  activation, selected discard, Countdown/Last Words, Super Evolution replay,
  exact card-ID keyword grants, board-capacity continuation, and RL Mode-mask
  parity;
- a generated follower/spell-chain batch closes six more collectible producers
  and six Tokens: Prim/Norga, Anne & Grea/Anne's Great Spirit, two Brilliant
  Artifact producers, Lulunai & Valnareik's two Mode spells, and Lishenna/Solo.
  The generic `cannot_be_destroyed_by_effects` printed passive blocks only
  ability destruction, emits an explicit prevention event, and is disabled by
  ability removal without blocking banish or zero-Health death. Tests cover
  exact-name conditions, ordered Spellboost, opponent-turn expiry, Mode/RL
  choice parity, successful-summon output binding, no-target atomicity, and
  destroy-prevention continuation;
- a generated burst-spell batch closes six collectible producers and seven
  generated spells. It adds auditable opponent-max-PP and play-time source-
  Spellboost conditions, permanent printed attack capacity, and filtered
  all-hand transformation across card types. Direct tests cover 9/10/19/20
  Spellboost thresholds, both-player 10-PP gating, Cooperation and Necromancy
  boundaries, own-turn hand cost listeners, independently resampled repeated
  random damage, full-board continuation, source-leave compatibility, atomic
  no-target rejection, and RL target-mask parity;
- a generated entry-listener batch closes four collectible producers and three
  generated followers. It composes self-related entry draw, Artifact-trait Rush
  grants, Enhance summons, filtered Puppet hand transformation, Ambush/Last
  Words return chains, and a persistent Ward-entry crest implemented through
  the existing leader-area listener boundary. Tests cover source exclusion,
  evolved-target filtering, no-candidate continuation, board shortage, crest
  persistence, non-Ward exclusion, and RL Enhance-mask parity;
- a hidden-copy/discard batch closes four collectible producers and four
  generated cards. Generic support now includes opponent-wide hidden-hand cost
  changes, post-zone `discarded` triggers with stable source identity,
  board-card-to-follower transformation with an auditable event, and hidden
  board-copy generation with an attached cost modifier. Direct tests cover
  Overflow 6/7, both Mode branches, exact opponent-turn expiration, ordinary
  versus Super Evolution replacement, full-board/full-hand failures, amulet
  transformation, no-target atomicity, hidden metadata/logging, and generated
  origin;
- a forced-target/follower-strike batch closes four collectibles (`10174120`,
  `10274110`, `10273310`, `10173120`) and two generated Puppetry followers
  (`90074120`, `90074130`). A structured static passive implements Lloyd's
  enemy ability-selection lock, while implicit `attack_target` identity lets
  Victoria deal its current attack to the attacked follower before combat.
  Orchis and Zwei use trait-filtered entry listeners; the official
  Lloyd-then-Orchis Sylvia Super-Evolve sequence verifies state-based
  stabilization between consecutive choices;
- an Enhance-Faith/Shikigami batch closes `10624120` and `10134110` plus
  generated cards `90024320`, `90034110`, and `90034120`. Genuine Enhance
  `CARD_PLAYED` events now advance matching Faiths and can fire dynamically
  granted abilities without treating normal, Accelerate, or Crystallize modes
  as Enhance. Destroyed-follower records retain their destruction turn, and
  filtered expressions can sum printed attack/health for the current turn.
  This implements Yidmetra's Faith/payment/Depths chain and Kuon's ordered
  Shikigami destruction, Spellboost Last Words, Noble growth, and filtered
  Super-Evolve Storm grant;
- a Crystalspawn/Faith random-distribution slice closes collectible `10634120`
  and generated spell `90034330`. Faith triggers now accept audited card filters
  on allied `follower_summoned` events, while the existing hand-listener path
  permanently reduces each retained copy's cost. Generic `random_distribute`
  assigns every point of a named Faith independently and uniformly to one of
  its structured buckets; nested `distributed_value` expressions drive the
  resulting follower buff, leader healing, and leader damage without consuming
  Faith. Tests cover two ordered summons with Storm, board shortage, live
  post-summon Faith, full-board continuation, zero-total RNG preservation,
  state fingerprints, and seeded replay. The draw law follows the
  [official Depths of the Eld Crystals FAQ](https://shadowverse-wb.com/en/deck/cardslist/card/?card_id=90034330);
- an oldest-first distributed-damage/crest batch closes eight collectibles
  (`10113120`, `10154130`, `10324120`, `10363210`, `10511310`, `10514120`,
  `10673110`, and `10753310`) plus generated `90024310`. Tests cover follower
  age and health capping, last-target overkill, leader remainder, Barrier
  allocation, dynamic hand/emblem counts, Octrice's nonduplicating crest,
  Treasure play/Fusion filters and one-tick multi-material Fusion, countdown
  expiration, Choose modes, Necromancy, Accelerate, evolution, and replay;
- an 8-card exact Activate follow-up batch covers zero- and one-PP activations,
  activation-only amulet play, self-destruction before queued choices, repeatable
  once-per-turn effects, selected keyword removal/buffs/debuffs, healing, hand
  cycling, Countdown, simultaneous all-follower damage/destruction, and matching
  real-card command/RL masks;
- a 5-card exact random multi-target batch covers Combo replacement of one
  target by three distinct targets, two independent Combo hits, random two- and
  three-follower damage, candidate shortage, simultaneous deaths, subsequent
  leader damage, whole-hand Spellboost, deterministic replay, and automatic RL
  resolution without adding a target-selection action;
- a 6-card exact random-effect follow-up batch covers two Mode followers,
  filtered allied-amulet thresholds, a dynamic other-card count followed by
  source-excluding self-board destruction, ordered turn-end damage, conditional
  effect evolution, distinct random batches versus repeated independent hits,
  and RL Mode-mask parity. It also promotes generated follower `90054130` to an
  exact executable Token with Rush, Ward, and its card-ID-filtered Last Words;
- Enhance keeps the original card type: enhanced spells resolve as spells and
  enhanced amulets enter as amulets. Mode operations append to base operations
  by default, while explicit `replace_base_operations` models printed “instead”
  clauses without text inference. Replacement legality, origin/graveyard flow,
  event metadata, and the unchanged special-mode RL action slots are tested;
- a 6-card exact spell-Enhance batch covers append versus replacement modes,
  random one-to-three target replacement, all-follower damage followed by draw
  and heal, Token summoning with board shortage, generated hand cards, Mode-to-
  all-abilities replacement, and selected-to-all filtered banishment. It also
  promotes generated spell `90021350` to an exact executable Token with both
  printed Mode branches;
- countdown amulets, explicit last words, fanfare/play rules, attack/clash,
  evolve/super-evolve, turn-start/turn-end triggers, and trigger continuations
  that can pause for choices. Exact `10713110` uses a source-in-play turn-end
  rule to draw only when the ending player's current Combo count is at least 3;
- death-batch event diagnostics that expose the active-player-first,
  left-to-right order used by destroyed, left-play, and Last Words lifecycle
  events, including follower/amulet composition for mixed death batches;
- Last Words effect frames retain an immutable death-time source snapshot for
  evolution state, effective keywords, attack, and health. Source conditions
  and expressions remain deterministic through nested pending choices after
  the entity leaves play, while board-mutating `self` targets still require a
  live source; snapshots participate in event diagnostics, fingerprints, and
  runtime invariants. Exact `10203120` uses this path for an evolved-only random
  damage Last Words after normal, manual-evolution, or Enhance play;
- `death_batch_end` emblem triggers that fire after a death batch's Last Words
  complete, with any new deaths collected into a later death batch;
- recursive resolution-loop diagnostics for events, effects, death batches,
  active card-listener/emblem batches, recent listener triggers, and suspended
  continuations;
- structured Mode decisions support `choose_count`, collect distinct choices
  through sequential commands without resolving early, and execute the chosen
  abilities in printed option order. The existing fixed RL choice actions and
  public observation fields expose both selection steps. Exact `10852310`
  covers all six two-of-four pairs, no-target random behavior, duplicate-choice
  atomicity, seeded replay, and spell graveyard Shadow gain;
- partial higher-level mechanics and primitives for cooperation, `觉醒`, `连击`,
  necromancy, reanimate, spellboost-style hand cost changes, emblems, optional
  decisions, play modes, and runtime modifiers.

The RL adapter keeps the fixed 111-action space and a 294-feature public
observation as the default `observation_version="v1"`, together with an action
mask, terminal reward, graveyard choice paging, special
hand actions for fusion/play modes, and super-evolve actions. `info()` is public by default and
redacts debug transcripts/events unless `debug_info=True` or
`info(debug=True)` is used, including pending-choice and graveyard-page returns.
Public observations and default info are regression-tested not to depend on
opponent hand identity or deck identity/order while a real-card pending choice
is awaiting resolution. The public observation includes explicit
controller/opponent `觉醒` flags derived from maximum mana and public
controller/opponent `连击` counts and follower-attack counts for the current
turn, plus pending multi-target choice size and progress.

An opt-in `observation_version="v2"` returns a structured, fixed-shape mapping
for full-card training without changing any action IDs. Callers should pass a
stable catalog-wide `card_vocabulary`; index 0 is reserved for padding or an
unknown card. V2 adds categorical own-hand and public-board identities, initial
deck-composition and public graveyard/banished histograms, origin and runtime
modifier features, board keyword bits, Faith/emblem identity and values,
granted owner/opponent-turn self-destruction bits, parameterized-choice
references, the legal action mask, and a bounded public
event history. It never emits raw entity IDs, opponent hand identity, fusion
materials hidden in the opponent hand, or remaining deck order. The default
derived vocabulary is convenient for small fixtures, but production training
must configure one shared vocabulary so shapes and indices are stable across
matches. `observation_v2_spec()` exposes every shape and categorical ordering;
`recurrent_observation()` supplies the same public v2 input for a caller-owned
recurrent or belief state. V1 consumers created before the attack-history
slice must migrate their input width from 292 to 294; action IDs are unchanged. V2
consumers select the version explicitly and replace scalar card features with
categorical embeddings while retaining `continuous_v1` during transition.
Each public board slot appends Intimidate and Aura flags, historically migrating
the observation from 270 to 280 and then 290 features. Public Artifact-kind
history historically raised the width to 292; the two public follower-attack
counters now raise it to 294 without changing action IDs;
attack and selected-effect mask entries come from the same command legality as
`GameEngine`.
The structured leader-area section also exposes both public leader maximum
health values; this does not change the fixed v1 width or any action ID.

For training, `observation_version="v3"` converts the same public state into
fixed-shape NumPy arrays with an explicit Gymnasium `spaces.Dict`. Hidden
decklists are the default: the learner receives its own initial-deck histogram,
while the opponent histogram is zero unless `open_decklists=True` is selected.
V3 observations for a non-acting player expose an all-zero action mask. V1 and
V2 remain compatibility interfaces; V3 is the supported input boundary for new
full-card training code.

`swb.rl.TrainableCardCatalog` builds the training pool from exact collectible
entries in `data/reports/rule_coverage.json`, rather than the legacy
follower-only database support flag. It preloads every database card for
SQLite-free match resolution and exposes the coverage-report hash/source
snapshot for experiment metadata. Its seeded deck sampler produces legal
40-card class/neutral decks with a configurable copy limit.

`swb.rl.SWBAECEnv` wraps the deterministic engine as a PettingZoo AEC
environment with `player_0` and `player_1`, per-agent rewards and done flags,
Gymnasium spaces, and decision ownership that follows pending choices. Rules
victories set `terminated`; `max_game_turns` and `max_agent_steps` set only
`truncated` and never manufacture a health-based winner. Page-navigation
actions count toward the agent-step limit. A single step reuses its computed
next-state action mask for the returned observation and info. The AEC wrapper
requires an explicit shared `card_vocabulary`; production callers should pass
`catalog.card_vocabulary` so every worker and deck has identical shapes and
card indices, including generated cards.

The RL dependencies are installed with the project metadata:

```powershell
python -m pip install -e .
```
Two public features expose the controller and opponent's Earth Sigil totals.
Sigils are board amulets rather than player-side counters: entering
Sigils merge into the newest amulet, merged Sigils are banished, and a depleted
Sigil is destroyed. This follows the
[official Worlds Beyond mechanic description](https://beginner.shadowverse-wb.com/ja/deck_shindan/result04/).
The final two features expose the controller and opponent's different-name
Artifact follower kinds that entered play this match. The history survives
leave-play, ignores transform as a new entry, and is reset with the match.
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

- `repeat` currently supports automatic/optional nested targeting and rejects
  nested `requires_target`; a future card whose repeated sequence must prohibit
  play when its first manual target set is empty needs explicit command-level
  availability semantics before that schema can be enabled;
- multi-target count fields remain intentionally unsupported for random hand,
  graveyard, and follower-or-leader targets; the rule loader rejects those
  combinations instead of silently applying single-target semantics;
- exact semantics for many real collectible cards; all 91 imported generated
  cards now have complete executable producer/behavior paths in the independent
  token audit, with no partial or database-only entry remaining;
- remaining `信仰` progression/payoff semantics, plus broader real-card coverage for `策动`,
  `土之秘术`, `觉醒`, and `连击` beyond the currently authored examples;
- ordinary board, hand, and leader-area listeners now receive
  `amulet_activated` and `card_fused`; remaining cards that use those events
  still need individual structured rules and official-text verification;
- Faith currently supports verified `follower_evolved`, `amulet_destroyed`,
  genuine Enhance-card progression, and filtered named-follower entry, plus
  atomic value spending, non-consuming random payoff reads, and dynamically
  gained structured abilities. Mode-selection progression and the shared
  five-slot leader-area limit remain explicit unsupported edges; other generated cards remain
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

Both `shadowverse_cards.json` and the built `data/cards.sqlite3` snapshot are
tracked repository files. After the current branch is committed and pushed, a
fresh clone on another computer already contains the database; rebuilding it is
only needed after refreshing the source JSON or changing the import schema.

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
vertical slices. The dedicated
[RL architecture audit](docs/rl_architecture_audit.md) records the training-stack
risks and the gate that should be completed before large-scale self-play.
Keep both files current when implementation status changes.
