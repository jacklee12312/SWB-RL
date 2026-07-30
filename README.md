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

## Local Match Simulator

The repository includes a local human-versus-PPO match interface. It shows both
boards and public resources, reveals only the human hand, supports mulligan and
all currently legal command/choice actions, and serves the downloaded card
images without copying them into the frontend build.

Every action is atomically persisted under ignored `data/match_history/` JSON
records. Starting another match marks an unfinished record as abandoned instead
of overwriting it. The UI can browse those records and their action-by-action
resolution timelines. Schema-v2 records retain complete private snapshots,
unredacted logs/events, and every legal action at each decision. AI decisions
also include logits, normalized probabilities, the selected action, and the
value estimate. Privacy is applied only to the live/UI presentation while a
match is ongoing; completed matches expose the saved AI hand and policy
distribution in the history drawer for diagnosis. Schema-v1 records remain
readable, although information already redacted from them cannot be recovered.

The leader panels keep deck, hand, graveyard, class resources, and a fixed
five-slot shared Faith/emblem area visible. Identical emblems cannot coexist,
and hand cards with structured Union Burst rules show their live `奥义` /
`解放奥义` gauge and ready state. Structured engine events generate a
lightweight resolution presentation for attacks, damage, spells, amulets,
summons, evolution, destruction, healing, and game results; the raw text log
remains available as a diagnostic fallback.

Install the frontend dependencies once, then start the Python inference service
and UI together:

```powershell
cd simulator-ui
npm install
cd ..
python -m scripts.run_match_simulator
```

The default opponent is the 1,206,159-decision Havencraft specialist at
`data/checkpoints/ppo_haven_specialist_8deck_1200k.pt`. The toolbar can start
either side with any validated fixed-deck QR profile and can select a different
PPO checkpoint before starting a new match. By default the server recursively
discovers `.pt` files under `data/checkpoints`; training-history, tuning,
initialization, and preflight artifacts are hidden. Model IDs come only from
that startup catalog, so the HTTP API cannot load an arbitrary submitted path.
The selected model ID and both exact deck manifests are retained in match
history. Use `--checkpoint` to choose the initial model,
`--checkpoint-directory` to change the selectable catalog, `--device cuda` for
local GPU inference, and `--frontend-port` / `--port` when the default ports are
occupied.

The current specialist checkpoints were updated only from Havencraft-side
trajectories, so selecting another AI deck is useful for engine/UI testing but
is not evidence that the policy specializes in that deck. The model's declared
specialist profile remains the default for both sides.

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
clause-audit layer without changing its legacy coverage categories. All 735
collectible cards are exact (100%), and all 735 have explicit implemented text
and named direct test evidence. The sibling
`data/audits/rule_clauses.json` registry hashes every
imported skill and alternate-mode clause, so a database text change or stale
test reference invalidates the audit instead of silently retaining exact
status. The current report has no unverified exact entry or missing generic
schema, primitive, targeting, timing, text-clarity, external-evidence, or
per-card structured-rule blocker. Rule metadata also supports version, errata,
official-source URL, retrieval date, and ruling fields, and the report records
the complete imported source snapshot hash.

The checklist 1.8
[zone/resource audit](data/reports/card_bug_audit/zone_resource_audit.md)
cross-checks the 826-card database and structured rules against official
capacity and resource semantics. It inventories 611 production cards that
touch zones, capacity, leader-area state, or class resources, including 107 in
the recursive eight-deck training closure. Direct contracts cover 0/8/9-card
hand and 0/4/5-card board boundaries, unique entity ownership across zone
changes, countdown/destruction/banish/Act distinctions, the shared five-slot
Faith/crest area, empty-deck outcomes, resource timing, successful-draw
listeners, and exact public graveyard/banished histograms. The checked-in
report has zero inventory, evidence, matrix, or behavioral-contract failures.

The checklist 1.9
[combat/endgame/RNG audit](data/reports/card_bug_audit/combat_endgame_random_audit.md)
inventories 643 combat-, damage-, endgame-, or randomness-related cards across
the full 826-card database, including 122 in the recursive eight-deck training
closure. Ten executable contracts cover attack targeting and timing, super
evolution, damage/healing/replacement/caps, distinct leave-play causes,
terminal outcomes, queue termination, deterministic replay, event-visible
random choices, and RNG-neutral illegal or skipped branches. Its AST inventory
finds all 30 engine RNG callsites use the engine-owned RNG or an explicitly
passed targeting RNG. This scan found the P0 `SWB-CARD-0003`: Bane was
incorrectly skipped after zero combat damage or Barrier prevention. Commit
`60d1c2f` aligns the shared mechanic with the official glossary and makes Bane
reuse the generic effect-destruction protection path. The checked-in ledger
has no open P0/P1 entry.

The checklist 1.10
[RL interface/privacy audit](data/reports/card_bug_audit/rl_interface_privacy_audit.md)
locks all nine command types to the continuous, non-overlapping 112-action
layout and directly exercises true-mask routing, sampled false-mask atomicity,
bounded graveyard pagination, public action/target/zone history, and the formal
v3.6/v4.1 shape, dtype, version, and privacy contracts. It also found the P0
`SWB-CARD-0004`: the online history endpoint exposed the unredacted local
schema-v2 replay, including the AI hand and complete policy distribution.
Commit `82bd251` keeps the complete private record on disk for offline review
while redacting the online history response. The report includes the explicit
decision that neither this transport fix nor the Bane semantics fix changes
Observation fields or the action layout, so no Observation migration is
required. The checked-in ledger again has no open P0/P1 entry.

Checklist 1.11 adds an opt-in, non-semantic runtime coverage recorder through
`audit_runtime_coverage=True`. It consumes structured engine events and stable
`card:<id>/.../operation:<index>` rule-tree IDs rather than localized logs,
records lifecycle and alternate-mode execution, condition branches, operation
execution, target/no-target/capacity/random-candidate paths, and all required
runtime diagnostics. The recorder is disabled by default and excluded from
snapshots and deterministic fingerprints. Run
`python -m scripts.report_runtime_coverage` to regenerate the
[JSON](data/reports/card_bug_audit/runtime_coverage.json) and
[Markdown](data/reports/card_bug_audit/runtime_coverage.md) reports, including
card/mechanic/deck/matchup aggregations. The saved deterministic fixed-deck
smoke ended normally after 44 agent steps with 0 placeholder, unsupported,
resolution-limit, illegal-action/command, or action-mask-mismatch diagnostics.
It catalogues 458 structured operation clauses in the 147-card training
closure: 15 triggered and executed, 3 triggered without operation execution,
and 440 not triggered. Those 440 remain open coverage work for 1.12 and are
explicitly not reported as passing. This slice also removed a diagnostic false
positive where a structured countdown emblem still emitted the generic
`COUNTDOWN` placeholder; match-state semantics and the Observation/action
schemas did not change.

Multi-session aggregation merges each stable clause by its highest observed
coverage state, so a later execution cannot remain counted as globally
untriggered. Snapshot/clone operation round-trips also retain the original
rule-tree clause ID instead of falling back to a dynamic ID.

The 1.11 shared-engine gate also passed the full 2,784-test suite (one skip),
compileall, the required 100-game invariant smoke, the mixed RL match, and a
1,000-game invariant self-play run with zero draws, truncations, or action-mask
mismatches. The machine-readable 1,000-game result is saved at
[`runtime_coverage_self_play_1000.json`](data/reports/card_bug_audit/runtime_coverage_self_play_1000.json).
This broad regression smoke is not the eight-deck matchup matrix or the
coverage-guided scenarios still required by checklist 1.12.

Checklist 1.12 now closes that gap. Nine minimal public-interface fixtures
cover cost, target, board/hand capacity, resources, ordinary/super evolution,
turn start/end, and simultaneous death; every necessary direct state setup is
followed by invariant validation. The
[forced-scenario report](data/reports/card_bug_audit/forced_scenario_audit.md)
maps 2,793 applicable mechanism assignments across the 147-card training
closure and the complete 735 collectible plus 91 generated-card catalog with
no unexplained clause or missing test file.

The rebuilt eight-deck
[matrix](data/reports/card_bug_audit/training_matrix_1000.json) passes 1,024
games (960 random-legal, 64 frozen-checkpoint policy) and all 1,024
deterministic replays. The stratified
[full-pool gate](data/reports/card_bug_audit/full_pool_sampling_10000.json)
passes 10,000 games (9,804 random-legal, 196 policy), places all 735 exact
collectible cards in sampled decks, encounters 824 cards, and completes
909,158 action-mask checks. Both reports have zero placeholders, mask
mismatches, illegal actions, exceptions, or truncations. The separate
[long-game/Myuu report](data/reports/card_bug_audit/long_truncation_myuu_distribution.md)
retains reproducible manifests for 95 long games and 240 Myuu matchups; neither
accepted source gate has a truncation.

Coverage-guided sampling found and fixed `SWB-CARD-0005`/`0006`, where
structured Last Words and other keyword sources were incorrectly reported as
unsupported, and P0 `SWB-CARD-0007`, where a lethal Last Words child effect
could expose the parent effect's next choice before state-based deaths were
processed. The latter fix bounds nested effect continuation by stack depth,
so new deaths stabilize before the parent frame resumes; it adds no card-ID
branch. The required 1,000-game shared-engine gate then found P0
`SWB-CARD-0008`: after Gilnelise's `+2/-2`, Snow Awake set Medusa's health to
1, but `SET_STATS` retained the older hidden `-2` health modifier. Super
evolution therefore produced an invalid 8/4 follower with maximum health 2.
An initial invariant-only correction produced 8/2 and was reopened because it
contradicted the direct
[official Snow Awake Q&A](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10132320),
which says a follower set to 1 health evolves to 3 health and super-evolves to
4 health. The final generic `Unit.set_stats()` implementation supersedes
earlier modifiers only in assigned dimensions, while preserving unassigned
dimensions and allowing later modifiers to layer normally; it contains no
card-ID branch. The exact saved 107-action prefix now replays to 8/4 with
maximum health 4 and no illegal action.

The official-source-first ruling ledger is saved at
[`data/audits/card_ruling_reviews.json`](data/audits/card_ruling_reviews.json).
It records queries, access dates, official URLs, conclusions, and evidence
scope. The lower-frequency question of how an older temporary modifier expires
after a later specific-value assignment has no direct official Q&A yet; its
provisional behavior remains explicitly `ruling_uncertain` pending a client
reproduction rather than being presented as officially confirmed.

After that fix, the same frozen configuration was rerun: the 1,024-game
eight-deck matrix and 10,000-game full-pool gate passed again, as did
[100-game](data/reports/card_bug_audit/stage_1_12_0008_official_random_self_play_100.json)
and
[1,000-game](data/reports/card_bug_audit/stage_1_12_0008_official_random_self_play_1000.json)
invariant self-play. The 1,000-game run had zero draws, truncations, illegal
actions, or mask mismatches. Random self-play now writes the failing game
index, per-game seed, decks, full action sequence, mask, board state, and
fingerprint to its requested JSON output before re-raising an exception. The
bug ledger has no open P0/P1 entry.

Checklist 1.13 turns that incident into a portable regression workflow. The
[`SWB-CARD-0008` package](data/reports/card_bug_audit/repros/SWB-CARD-0008.json)
contains the database and rule hashes, exact decks and seed, a JSON-native
pre-command state, decoded command, full 112-action mask and legal commands,
transition events, and official expected versus historical actual results. A
prefix-first delta debugger made 755 candidate replays and reduced the original
107-action match to an 86-action legal natural reproduction; the same final
command still super-evolves Medusa from 5/1/1 to 8/4/4. A separate
[synthetic fixture](data/reports/card_bug_audit/repros/SWB-CARD-0008-synthetic.json)
keeps the primitive boundary directly testable without replacing the real-match
evidence.

The accompanying
[checkpoint impact report](data/reports/card_bug_audit/repros/checkpoint_impact.json)
read 52 local `.pt` manifests without modifying the files. All 52 predate
`b6f1d95` and remain preserved for historical reconstruction only; they must
not be mixed with post-fix models for fair strength conclusions. This engine
change did not alter `data/rules`, so a matching rulebook hash alone is not
enough to establish behavioral compatibility. The consolidated 1.13 result is
saved in
[`stage_1_13_repro_closure.json`](data/reports/card_bug_audit/stage_1_13_repro_closure.json).
Its final gate passed 2,823 tests with one conditional skip, compileall, and the
deterministic 1,546-case RL interface/privacy report.

## Reproducible RL Platform

The P1/P2 platform-hardening slice is implemented around the deterministic
rules core:

- Observation v3 and the 112-action layout have named schemas and stable
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
- The compatibility baseline is shared-parameter recurrent masked PPO with separate
  player hidden states, sparse terminal reward, terminated/truncated bootstrap,
  recurrent sequence batching, PPO clipping, gradient clipping, and finite-
  value guards. Stable vocabulary indices in hand and public-board slots feed a
  trainable card embedding instead of being treated as dynamically normalized
  ordinal numbers. Atomic schema-v2 checkpoints include the model, optimizer, live
  environment, RNGs, progress, versions, dirty git state, and opponent league.
  Training accepts `--device cpu` or `--device cuda`; policy sampling and PPO
  minibatch permutation use a generator on the selected device, with a
  CUDA-conditional collection/update regression test.
- New training runs default to `entity_action_v1`: a 6.5M-parameter policy
  with 128-dimensional card embeddings, a 256-wide four-layer/8-head
  Transformer over the public hand/board entities, a 512-wide recurrent
  memory, and a source/target-conditioned action scorer. Entity-target choice
  scores follow the candidate when option ordering changes instead of attaching
  one independent learned weight to each choice position. Legacy
  `legacy_gru_v1` checkpoints remain loadable and keep their original network.
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
- Training-facing AEC/Gym adapters, single- and multi-process PPO rollout,
  fixed evaluation, and match scripts default to `match_setup="official"`.
  The setup name is stored in trajectory/checkpoint experiment versions;
  pre-integration schema-v2 checkpoints without the field resume as `legacy`
  instead of silently changing their opening distribution.
- Named fixed-deck training includes
  `official_qr_evolve_haven_20260727`, the 40-card official QR
  super-evolution Havencraft list. Fixed mode uses the exact same deck for both
  players by default. Specialist mode keeps that learner deck fixed while
  `--opponent-decks` deterministically cycles named opposing decks and both
  player positions. The current policy still acts for both sides, but only
  learner-deck transitions enter the PPO update. Checkpoints and reports store
  the complete schedule, assignments, and per-opponent results. Both modes
  retain seeded shuffle, mulligan, random first player, and all ordinary match
  randomness. Source hashes and immutable deck SHA-256 values are stored in
  training reports and checkpoints. Official deck QR images can
  also be decoded locally with `scripts.import_deck_qr`. International
  `shadowverse-wb.com` URLs and NetEase's nested `163.com` wrapper share the
  same four-character card-token codec, so no official-server request is
  required once the image is available. Imports are saved under `data/decks/`
  with the raw payload, 40 card IDs, content hash, database snapshot, and
  per-card exact-rule coverage. Only manifests with zero validation issues are
  automatically exposed as named fixed training/evaluation decks.

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
python -m pip install -e ".[rl,train,qr]"
python -m scripts.import_deck_qr path/to/deck-qr.png --name my_deck --display-name "My Deck" --require-trainable
python -m scripts.vector_rollout --workers 4 --episodes 16
python -m scripts.audit_rl_distribution --episodes 98 --workers 2
python -m scripts.train_ppo --total-agent-steps 10000
python -m scripts.train_ppo --training-deck official_qr_evolve_haven_20260727 --device cuda --rollout-workers 4 --total-agent-steps 100000 --opponent-current-weight 1 --opponent-random-weight 0 --opponent-fixed-weight 0 --opponent-historical-weight 0
python -m scripts.train_ppo --training-deck official_qr_evolve_haven_20260727 --opponent-decks international_qr_forest_20260728 international_qr_sword_20260728 international_qr_runecraft_20260728 international_qr_dragon_20260728 international_qr_nightmare_20260728 international_qr_portal_myuu_20260728 international_qr_portal_lishenna_20260728 --device cuda --rollout-workers 4 --total-agent-steps 1200000 --opponent-current-weight 1 --opponent-random-weight 0 --opponent-fixed-weight 0 --opponent-historical-weight 0
python -m scripts.evaluate_deck_matchups data/checkpoints/ppo_haven_specialist_8deck_1200k.pt --opponent-checkpoint data/checkpoints/ppo_haven_specialist_8deck_10k.pt --learner-deck official_qr_evolve_haven_20260727 --opponent-decks international_qr_forest_20260728 international_qr_sword_20260728 international_qr_runecraft_20260728 international_qr_dragon_20260728 international_qr_nightmare_20260728 international_qr_portal_myuu_20260728 international_qr_portal_lishenna_20260728 --seed-count 20 --max-agent-steps 512 --master-seed 20260808 --device cuda
python -m scripts.evaluate_ppo data/checkpoints/ppo_evolve_haven_smoke.pt --training-deck official_qr_evolve_haven_20260727 --seed-count 250 --master-seed 20260801
python -m scripts.train_ppo --rollout-workers 4 --total-agent-steps 10000 --opponent-current-weight 1 --opponent-random-weight 0 --opponent-fixed-weight 0 --opponent-historical-weight 0
python -m scripts.evaluate_ppo data/checkpoints/ppo_smoke.pt
python -m scripts.benchmark_rl_env
python -m scripts.profile_ppo_training --checkpoint data/checkpoints/ppo_evolve_haven_entity_action_4m.pt --additional-agent-steps 100000 --device cuda --output data/reports/ppo_evolve_haven_entity_action_4m_profile_100k.json
```

Fixed evaluation can mirror a named training deck on both sides and records
the immutable deck manifest in its evaluation-suite hash. Historical-opponent
reports also hash the opponent checkpoint. In the 2026-07-27 held-out
fixed-deck experiment, the 1,024-step checkpoint scored 93.8% against the
random-legal baseline (500 games, 95% CI 91.3%-95.6%); after continuing to
100,096 steps, it scored 99.4% against random legal (98.3%-99.8%) and 72.4%
against the 1,024-step checkpoint (68.3%-76.1%, +167.5 relative Elo). All
1,500 games terminated normally with zero illegal actions or mask mismatches.
This establishes relative learning on the mirror, not ladder strength.
The 2026-07-27 entity/action-conditioned Havencraft run used the RTX 4080 as
the learner and four CPU rollout actors. It reached 102,067 agent steps across
1,236 games and 46 PPO updates in 661.9 seconds for the 2k-to-100k continuation
(150.98 agent steps/s). On 100 held-out mirrored games it scored 100% against
random legal and 100% against its random initialization, with zero illegal
actions, truncations, or mask mismatches. Against its own 50,957-step snapshot
it scored 55% (95% CI 45.2%-64.4%, +34.9 relative Elo), which is evidence of a
working training path but not statistically significant evidence that the
second half of this short run materially improved policy strength.
The 2026-07-28 eight-deck Havencraft-specialist run reached 1,206,159 total
environment decisions, 15,664 whole games, and 545 updates. Its main
10k-to-1.2M continuation sustained 257.19 environment decisions/s. In a
held-out 280-game suite against the frozen 10k policy, the final checkpoint
scored 87.3% (95% CI 82.9%-90.7%, +335.2 relative Elo); the 10k checkpoint
scored 48.6% on the identical decks and seeds, a gain of 38.8 percentage
points. Every opponent improved, with final per-deck rates from 77.5% to
95.0%. Across both 280-game suites there were zero illegal actions or
action-mask mismatches; the baseline had two 512-step truncations and the
final policy had one.
The non-destructive PPO profiler reports parent-process rollout phases, worker
policy construction/loading, environment setup, observation encoding, CPU
inference, engine steps, trajectory packaging, and learner advantage, batching,
device transfer, forward/loss, backward, optimizer, and validation phases. It
updates the loaded policy only in memory and verifies that the source checkpoint
was not modified. A 101,183-step RTX 4080/i7-13700KF run over 45 updates
measured 153.39 agent steps/s; after excluding two warm-up updates, rollout
accounted for 52.5% of measured wall time and the CUDA learner for 47.5%.
Within the four rollout workers, batch-one CPU policy inference consumed 75.1%
of worker episode time versus 22.1% for rules-engine steps. The collector also
broadcast the roughly 24.8 MiB policy about 6.74 times per PPO update; the four
workers collectively rebuilt/reloaded about 27 model copies and received about
670 MiB of policy payload per update.
The central-inference collector keeps those engine workers persistent but owns
the only rollout policy in the learner process. Workers send observations and
receive actions; the learner batches requests for 0.5 ms, evaluates them on the
same CUDA model, and keeps independent recurrent state and seeded sampling per
episode. A like-for-like 100,716-step profile from the same 4M checkpoint used
four workers with two Torch threads each and reached 193.28 agent steps/s,
26.0% above the previous 153.39-step/s path. Steady rollout cost fell from
3.39 to 2.07 ms/step (38.9%), median/P95 rollout time fell from 7.70/8.13 to
4.64/5.01 seconds, and policy transmission fell from 28.1 GiB to zero. Learner
update cost remained effectively unchanged at 3.03 ms/step, confirming that
the PPO update boundary and workload were not shortened to produce the gain.
Still unsupported: this is a baseline PPO and league/evaluation system, not a
distributed learner, a policy-strength result, or a complete MCTS
implementation. Multiprocess PPO currently uses current-policy self-play;
random, fixed, and historical opponent mixing remains on the single-process
collector. The balanced class schedule is not an adaptive curriculum, and the
same-class fixed evaluation suite is not yet a 7x7 cross-class policy-strength
matrix. Snapshot/clone is the search foundation only. Card-rule coverage also
remains deliberately separate from policy strength: the current frozen
database snapshot has 735/735 exact collectible rules, and all 735 enter the
exact training catalog.

## Implemented Engine Surface

The deterministic rules core supports:

- official match setup and shared limits: four-card opening hands, an optional
  interactive 0-to-4-card mulligan that cannot redraw the same physical card,
  seeded random first-player selection, the second player's one-use Extra PP
  with a one-time refresh at the start of its sixth turn, first-player turn-5
  versus second-player turn-4 normal evolution, nine-card hand capacity, and a
  five-slot leader area shared by Faiths and emblems. Pass
  `match_setup="official"` for seeded random first-player assignment and the
  interactive mulligan. Low-level deterministic fixtures retain the
  `match_setup="legacy"` default and may still override `starting_player` or
  `enable_mulligan` explicitly;
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
- a seven-card listener/condition/output-binding batch covers `10363110`,
  `10454110`, `10544120`, `10553110`, `10744110`, `10843110`, and `10851130`.
  Generic attack-declared listeners and event-source emblem targeting, strict
  leader-health comparison, printed-life deck filters, reanimate output
  bindings, and opponent-emblem targeting express the rules without card-ID
  branches. Direct tests cover target/no-target paths, stale selections,
  hand/board capacity, simultaneous source death, source departure, fixed-seed
  replay, and command/action-mask parity;
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
- an 8-card exact cross-class existing-primitive batch covers `10153130`,
  `10323110`, `10413110`, `10442120`, `10624110`, `10674110`, `10754120`,
  and `10823110`. It composes Necromancy, cumulative Enhance tiers, Skybound
  Art and Super Skybound Art, original-cost and Trait entry listeners, dynamic
  board-count damage, Fusion/play event filters, repeated random damage, and
  Follower Strike through existing generic primitives. Direct tests cover all
  referenced cards and embedded tiers, resource shortage, hand/board capacity,
  no/stale/duplicate targets, source departure, fixed-seed replay, multilingual
  clause hashes, Token Audit continuity, and command/RL action-mask agreement;
- an 8-card exact crest/listener batch covers `10124130`, `10133110`,
  `10243110`, `10414120`, `10714120`, `10724110`, `10833110`, and `10864120`.
  It composes Countdown crests, Earth Rite, Combo, Rally, Trait-filtered summon
  and spell listeners, dynamic Earth Sigil damage, repeated distributed random
  damage, selected evolution, Choose One, and intrinsic/runtime keywords using
  existing generic primitives. Direct tests cover all referenced Tokens and
  alternate clauses, no-target continuation, stale and illegal choices,
  hand/board capacity, source departure, once-per-turn listeners, fixed-seed
  replay, multilingual clause hashes, Token Audit continuity, and command/RL
  action-mask agreement;
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
  the attacked follower (including destruction by an attack-time ability).
  Resource-gated `进化时`/`超进化时` keywords are distinct from ordinary
  `本随从进化时`/`本随从超进化时` state-change clauses: effect evolution fires
  only the latter clauses;
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
- runtime hand keywords retain permanent and duration-aware add/remove state,
  expire through the common modifier boundary, survive ordinary follower play
  and exact hand-copy summoning, and reset on hand transform. `add_card` now
  output-binds the exact generated hand entity, `add_union_burst_gauge`
  increments only cards with structured Skybound Art definitions, and implicit
  attack targets revalidate their centralized board filters at resolution.
  Exact `10302110`, `10303110`, `10022120`, `10722310`, `10271120`, `10471120`,
  and `10223110` cover opponent Super-Evolution hand listeners, same-name draws,
  generated-card keyword/stat grants, Rally capacity, gauge increments,
  Enhance draw/set-cost, damaged-target Follower Strike, stale outputs,
  deterministic replay, and RL command-mask parity;
- generic `summon_exact_copy` copies a live follower's runtime stats,
  evolution state, keyword/ability modifiers, restrictions, and selective
  Last Words-removal state into a newly entered entity before emitting its
  summon event; optional stat deltas allow recursive exact-copy clauses to
  observe the already-adjusted copy. Generic `remove_last_words` suppresses
  only the follower's Last Words while preserving Ward, Aura, and unrelated
  printed/runtime abilities. Hand-listener `buff_hand_card(target=self)` is
  schema-scoped to the hand-listener context. Exact `10131110`, `10144110`,
  `10254120`, `10313110`, `10434110`, `10532110`, `10634110`, `10862110`,
  and `10871110` cover hand Spellboost stats, bound discard cost, four crests,
  Reanimate/Trait Ward, recursive exact copies, both Super-Evolve Modes,
  simultaneous crest damage, a referenced follower chain, and self-replacing
  Last Words that retain non-Last-Words abilities. Direct tests cover
  no/stale/illegal targets, hand/board capacity, source departure,
  simultaneous deaths, deterministic replay, multilingual hashes, Clause and
  Token audits, and action-mask parity. Observation v2 now exposes the public
  per-follower Last-Words-removal bit (160 public-board runtime values instead
  of 150); the derived formal observation manifest is versioned as
  `observation-v3.1` without changing the 111 action IDs;
- generic destroy output bindings retain only targets actually destroyed, and
  `bound_target_count` exposes that successful-result cardinality to later
  effects. Structured incoming-damage replacement applies before Barrier and
  is disabled by printed-ability removal. Global `all_board` filters and
  `remove_all_emblems` cover both leaders without card-ID branches, while exact
  Rush/Storm copies receive the normal same-turn attack readiness. Exact
  `10163110`, `10401110`, `10464120`, `10711110`, and `10804110` cover
  destroyed-amulet damage, thresholded damage replacement, seeded distinct
  Union Burst targets, selected double banish, Super Skybound Art, exact-copy
  readiness, and all three global banish Modes. Direct tests cover prevention,
  zero/no/stale/illegal targets, board capacity, simultaneous deaths,
  deterministic randomness, Clause/Token audits, pending choices, and
  command/action-mask parity without changing observation or action schemas;
- event listener and emblem eligibility is snapshotted when an event is
  emitted, so a follower or crest entering later cannot retroactively react to
  an earlier event. Generic enemy-side summon ownership, Enhance-only event
  filters, distinct-name filtered draws, replacement of lower Union Burst
  tiers, and seeded distinct random-keyword grants are structured engine
  primitives. Exact `10224120`, `10424120`, `10574120`, `10603110`, and
  `10622310` cover enemy Knights, temporary attack prohibition, direct
  Super-Evolution replacement, selected discard and evolved spell listeners,
  different-name 1-cost spell draws, six-way random ability selection, and
  the Majestic Conquest crest/Enhance chain. Direct tests cover no/stale/
  illegal choices, source departure, board/hand capacity, event order,
  deterministic RNG, multilingual hashes, Clause/Token audits, and
  command/action-mask parity;
- dynamic filtered-draw costs can use the controller's current Combo, selected
  hand followers can bind their current attack, followers can receive
  structured granted Last Words or effect-destroy immunity in hand, and
  `summon_from_hand` moves the same physical entity to the board without
  running Fanfare. Exact `10111140`, `10272310`, `10273110`, `10412110`, and
  `10473110` cover Combo-cost draw, Artifact Rush/Last Words, Ward/effect
  protection, Enhance hand summoning and return, selected-attack area damage,
  and both referenced Artifact Tokens. Direct tests cover no/stale/illegal
  targets, hand/board capacity, ability removal, simultaneous deaths,
  deterministic replay, multilingual hashes, Clause/Token audits, and
  command/action-mask parity. V1 remains 294 floats and 111 action IDs;
  structured v2/v3 hand and board runtime vectors now expose granted Last
  Words and effect-destroy immunity, with the formal schema version advanced
  to `observation-v3.2`;
- generic `halve_round_up` deck-cost modifiers, seeded distinct
  `random_choice` branches, attack-target exclusion for random follower
  effects, and bound-snapshot `banish_same_name` resolution close exact
  `10173130`, `10244120`, `10263110`, `10334110`, and `10532310`.
  Fennie's odd-cost rounding and repeated-current-cost behavior are grounded
  in its official card-page FAQ. Direct tests cover linked Puppetry and Clay
  Golem Tokens, simultaneous granted Last Words, board capacity, amulet
  thresholds, attack-target exclusion, evolve/Super-Evolve pending choices,
  stale and illegal targets, seeded branch uniqueness, Earth Rite shortage,
  multilingual hashes, Clause/Token consistency, and command/action-mask
  parity without changing observation or action schemas;
- generic deck-duplicate banish, exact leftmost-hand copy, printed-cost top-N
  comparison, enemy-deck random hand transform, and owner-turn-end banish
  primitives close exact Neutral cards `10303210`, `10502110`, `10502120`,
  `10602210`, and `10704110`, bringing exact collectible coverage to 693/735
  (94.29%). The two existing public turn-end-removal runtime slots encode
  destroy and banish as a bitmask without changing their shape; the formal
  schema advances to `observation-v3.3` while all 111 action IDs remain
  stable. Direct tests cover empty decks and target shortages, physical
  deck-cost modifiers, complete hidden hand-copy state, silence, source/target
  departure, crest expiry, hand/board capacity, deterministic replay,
  multilingual/Mode/reference hashes, Clause/Token audits, illegal choices,
  and command/action-mask parity without card-ID branches;
- generic distinct Fusion-material-name expressions, CARD_PLAYED current/base
  cost-change filters, exact random physical enemy-deck copying, combined
  follower-and-leader healing by the follower's actual restored amount, and
  safe bound hand-target continuation close exact `10324110`, `10514110`,
  `10332210`, `10342110`, and `10364110`, bringing exact collectible coverage
  to 698/735 (94.97%). Sinciro counts differently named fused Loot cards;
  Wolfraud copies exact hidden physical deck cards with fresh modifier IDs;
  Truth's Research Facility exposes its Engage selection and cost-change
  listener through commands and action masks; Worshipper of Disdain preserves
  its Rush/Ward and healing order; and stacked Himeka crests use seeded random
  filtering plus permanent attack locks and owner-turn-end banish. Direct
  tests cover duplicate names, no/stale/illegal targets, source departure,
  empty decks without RNG consumption, hand capacity, simultaneous effects,
  deterministic replay, multilingual/Mode/reference/raw-source hashes,
  Clause/Token consistency, and command/action-mask parity without changing
  Observation `observation-v3.3` or the 111 action IDs;
- generic per-follower attacked-this-turn board filtering, drawn-card current
  cost-set event filters, same-current-cost hand grouping, enemy-hand follower
  stat changes, and random any-follower-or-either-leader damage close exact
  `10464110`, `10474110`, `10524110`, `10553310`, and `10564120`, bringing
  exact collectible coverage to 703/735 (95.65%). Galleon evolves only an
  eligible follower that did not attack; Lu Woh's countdown crest modifies a
  Storm attacker before leader combat; Oluon rebuilds its mixed random
  candidates between hits and excludes itself; Rigor evaluates the post-draw
  physical hand; and Kukishiro routes current-cost draws to the correct board
  owner. Direct tests cover unlock and turn gates, no-target and full-board
  paths, source departure, runtime cost modifiers, sequential deaths, seeded
  replay, multilingual/Mode/reference/raw-source hashes, and Clause/Token
  consistency without changing Observation `observation-v3.3` or the 111
  action IDs;
- persistent destroyed-amulet history, highest-base-cost and distinct-name
  history selection, current-cost filtered bulk deck banish with actual-count
  follow-ups, whole-hand redraw, and simultaneous enemy-follower/leader damage
  close exact `10162130`, `10663210`, `10664110`, `10543110`, and `10554120`,
  bringing exact collectible coverage to 708/735 (96.33%). Direct tests cover
  source and selected-target departure, no candidates without RNG consumption,
  board and hand capacity, current-cost deck modifiers, simultaneous deaths,
  seeded replay, multilingual/reference/raw-source hashes, Clause/Token
  consistency, illegal-choice atomicity, and command/action-mask parity.
  Observation `observation-v3.3` and all 111 action IDs remain unchanged;
- frozen listener event base-cost and activation-count context, public leader
  Barrier charges, ordered leader-damage replacement, and generic source
  Fanfare replay close exact `10362210`, `10444120`, `10503210`, `10604110`,
  and `10703210`, bringing exact collectible coverage to 713/735 (97.01%).
  Temple of Repose advances by the live crest count and grants a one-hit leader
  Barrier; enhanced Zooey sets max defense to 1 and replaces positive incoming
  damage through the opponent-turn boundary; World of Games compares the
  played card's frozen base cost against every other field card; Omegotep uses
  distinct seeded random choices and bounded recursive Fanfare replay; and
  Babelon consumes its persistent listener count for the printed three-step
  sequence. Direct tests cover no-target/source-leave paths, hand capacity,
  illegal-choice atomicity, seeded recursion, countdown/turn boundaries,
  command/action-mask parity, multilingual/raw-source hashes, and Clause/Token
  consistency. Public leader Barrier and damage-replacement mode advance the
  formal schema to `observation-v3.4`; v1 remains 294 floats and all 111 action
  IDs remain stable;
- follower stat-decrease events, persistent granted turn-end abilities,
  permanent physical deck-follower stat modifiers, Faith Mode-selection
  bonuses, non-repeating random branch histories, cross-board random follower
  targeting, signed `negate` expressions, and three generic follower-play
  emblem passives close exact `10214120`, `10314110`, `10354110`, `10554110`,
  `10574110`, and `10714110`. The slice covers Lymaga's two-target attack lock
  and delayed self-damage, Krulle's once-per-turn recovery and opposing crest,
  Sham & Nacha's Faith-driven extra Mode choice and copy, Milteo & Luzen's
  Reanimate/destroy-six/play-suppression crest, Slaus's seeded non-repeating
  positive and negative wheels, and Thestae's Combo/deck buff. Direct tests
  cover atomic multi-choice and RL masks, no-target/capacity paths, source
  departure, simultaneous deaths, current hand/deck stats, fixed-seed replay,
  hidden-deck privacy, strict schema rejection, source hashes, and
  Clause/Token consistency. Exact collectible coverage reaches 719/735
  (97.82%), with no supported-but-missing rule or generic blocker. The added
  public runtime fields advance the formal schema to `observation-v3.5`; v1
  remains 294 floats and all 111 action IDs remain stable;
- a strict audited `vanilla_cards` declaration plus the existing intrinsic
  keyword registry closes 11 official-source basic followers: two cards with
  no printed ability (`10002120`, `10422120`) and nine Ward, Storm, Rush, or
  Ambush followers (`10001130`, `10021110`, `10021130`, `10041120`,
  `10061120`, `10143110`, `10211110`, `10221120`, `10612120`). Loader
  validation rejects missing provenance notes, duplicate vanilla declarations,
  and any overlap with behavior-bearing definitions. Direct tests cover all
  real stats and multilingual/raw source data, official-source metadata,
  no-target and full-board paths, Ward target forcing, Storm/Rush legality,
  Ambush attack/manual-target protection, illegal-command atomicity, fixed-seed
  replay, RL mask parity, and Clause/Token/Ability consistency. Exact
  collectible coverage reaches 730/735 (99.32%); 91 generated cards remain
  complete, and Observation/action versions are unchanged;
- official English/Japanese/Simplified-Chinese card pages and FAQs close the
  final five complex collectibles (`10201310`, `10533310`, `10572120`,
  `10741310`, `10851110`). Generic additions cover Lloyd-style forced manual
  targeting, independent-with-replacement whole-board exact-copy transforms
  from the owner's physical deck, filtered whole-deck replacement, spell-play
  self-evolution listeners, and a removable passive that ignores Ward.
  Direct tests cover normal and illegal paths, no legal target, source/target
  departure, hand and board capacity, physical deck modifiers, replacement
  reset semantics, fixed-seed replay, Ward removal, and command/action-mask
  parity. Coverage reaches 735/735 exact (100%); Clause Audit has 735 mapped
  entries, Token Audit keeps all 91 generated cards complete, and
  Observation/action versions remain unchanged;
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

For legacy training, `observation_version="v3"` converts the same public state into
fixed-shape NumPy arrays with an explicit Gymnasium `spaces.Dict`. Hidden
decklists are the default: the learner receives its own initial-deck histogram,
while the opponent histogram is zero unless `open_decklists=True` is selected.
V3 observations for a non-acting player expose an all-zero action mask. V1 and
V2 remain compatibility interfaces. V3.6 is now frozen for existing checkpoint
compatibility.

Observation v4.0 is retained as the audited migration format that removed the
known v3.6 state collisions. New command-line training now selects
`observation_version="v4.1"`. V4.1 keeps the same information and privacy
boundary but replaces v4.0's broad one-hot/flat input with typed categorical
rows, sparse card/count pairs, compact semantic tokens, and action-centered
candidate rows. In the current schema, 15,757 numeric values and 1,290 shared
card-vocabulary indices are assembled into 93 meaningful Transformer tokens:
one match token, two player tokens, 19 hand/board entities, 20 Faith/emblem
slots, 13 zone summaries, 32 public events, and six rule-record summaries.
Local modifiers, granted effects, fusion materials, and listener state are
pooled into their owning entity instead of becoming unrelated global values.
The standard model remains 256-wide with four Transformer layers and a
512-wide GRU, but falls to about 5.58M parameters with the current 826-card
vocabulary.

V4.1 still excludes opponent hidden hands, unknown deck contents/order, future
randomness, and raw entity IDs. It preserves the 112-action layout and requires
the entity/action policy; v3.6 and v4.0 checkpoint loading remains unchanged.
See [`docs/observation_v4_1_design.md`](docs/observation_v4_1_design.md) for
the v4.1 field groups, token construction, limits, and migration rules, and
[`docs/observation_v4_field_audit.md`](docs/observation_v4_field_audit.md) for
the original information audit.

A completed three-seed follow-up tuned v4.1 and trained both v3.6 and v4.1
from scratch to about 500k learner decisions per seed. V4.1 averaged 84.52%
against the common frozen reference versus 82.02% for v3.6, but the
seed-level difference interval included zero. Their 600-game direct matchup
was effectively tied at 49.33% for v4.1 (95% CI 45.35%-53.33%), while v4.1
cost 2.72 times as much wall-clock training time. The current direction is
therefore to retain v4.1's richer state contract while optimizing its token
and inference cost. See
[`docs/observation_v3_6_v4_1_learning_ablation.md`](docs/observation_v3_6_v4_1_learning_ablation.md)
for the staged tuning, per-seed results, integrity checks, and limitations.

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
features without changing the pre-existing action IDs.
The final ten v1 fields expose first-player identity, mulligan progress, and
both players' public Extra PP availability, use count, and active-turn state.
This moves v1 to 304 floats and `observation-v3.6`. Extra PP is appended as
action 111, moving the layout to 112 actions without renumbering actions
0 through 110. During an interactive mulligan, the existing 16 choice actions
represent all subsets of the four-card opening hand.
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

All cards in the current 826-card snapshot are audited: 735/735 collectibles
are exact and all 91 generated cards have complete executable producer and
behavior paths. The remaining broad engine limits are guarded schema surfaces
or future-content risks, not uncovered clauses in the current catalog:

The deeper runtime card-bug audit is tracked separately from structural exact
coverage. P0-P3 definitions, training-blocking policy, and the validated ledger
contract are generated by `scripts/report_card_bug_audit.py` into
`data/reports/card_bug_audit/bug_ledger.{json,md}`. An empty ledger is not a
claim that the eight-deck or full-catalog runtime gates have passed; follow
`docs/card_bug_audit_and_training_speed_checklist.md` for their evidence.
The frozen audit baseline and recursive training-deck closure are generated by
`scripts/report_card_bug_audit_baseline.py` into
`data/reports/card_bug_audit/baseline.json` and
`training_deck_card_closure.json`: the eight decks contain 111 unique
collectibles and currently expand to 147 database-resolved, audit-resolved
cards. This is an inventory gate, not yet the per-clause runtime gate.
`scripts/report_card_clause_matrix.py` turns that closure into 147 card rows
and 161 independent main/alternate-mode clause rows. Its five-state dimensions
keep 1,224 applicable runtime checks explicitly untested, so structural exact
coverage cannot silently authorize training.
`scripts/report_card_source_alignment.py` independently checks those 147 cards
against each preserved raw import record: all 161 Chinese/English/Japanese
clauses, printed fields and base-keyword records align, and all 76 references
resolve to the named target version. The deterministic ruling queue is
currently empty for source-to-structure ambiguities, but that source-only
result does not close any of the remaining runtime or official/client evidence
dimensions.
The first runtime gate now covers every PP-based alternate play route in the
current snapshot. `scripts/report_play_mode_boundary_audit.py` scans 54 cards
and 55 Enhance/Accelerate/Crystallize modes across 1,546 cost-boundary cases
plus 55 full-board cases, including temporary and permanent cost changes.
The saved report has zero command/action-mask mismatches, illegal-mutation
failures, or execution failures; the two resulting P0 bugs are fixed in
`f895051` and closed in the card-bug ledger. Mode and Invocation remain
separate non-PP route families covered by their dedicated behavioral tests.
The next runtime gate, generated by
`scripts/report_keyword_entry_audit.py`, inventories every keyword and
attack/target restriction source across the current 735 collectibles and 91
generated cards. It finds 321 source cards (59 in the training closure) and
executes nine runtime keywords through twelve play, summon, copy, transform,
and evolution paths. Its saved matrix has zero inventory, execution, or RL
command/mask failures. A real Zooey regression keeps Storm exclusive to
Enhance 10 and cites the official card page used for verification.
The target and pending-choice gate is generated by
`scripts/report_target_choice_audit.py`. It inventories 477 production
target/choice source cards (90 in the training closure), keeps eight explicitly
named synthetic demo sources outside the 826-card catalog, and executes all 14
manual target domains plus cardinality, source exclusion, restriction,
multi-target, stale-target, source-leave, mixed selected/random/all,
no-candidate, snapshot/restore, and 112-action ordering contracts. The saved
report has zero inventory or behavioral failures.
The trigger timing and batch gate is generated by
`scripts/report_trigger_timing_audit.py`. It inventories 770 production trigger
source cards (131 in the training closure), isolates 26 explicitly named demo
sources, and maps turn start/end, attack, clash, survived damage, entry,
evolution, super evolution, Last Words, countdown, emblem, and Faith timing.
Eleven executable contracts retain official turn-boundary evidence, Marwynn's
real-card priority regression, death batching, source and condition snapshots,
pending-choice continuation, terminal-result stopping, and deterministic
20,000-step loop diagnostics. The saved report has zero inventory, evidence, or
behavioral failures; `death_batch_start` emblem triggers remain an explicitly
rejected future-content boundary with no current production source.

- `repeat` currently supports automatic/optional nested targeting and rejects
  nested `requires_target`; a future card whose repeated sequence must prohibit
  play when its first manual target set is empty needs explicit command-level
  availability semantics before that schema can be enabled;
- multi-target count fields remain intentionally unsupported for random hand,
  graveyard, and follower-or-leader targets; the rule loader rejects those
  combinations instead of silently applying single-target semantics;
- future `信仰`, `策动`, `土之秘术`, `觉醒`, and `连击` variants still require
  explicit structured definitions and direct verification before a later
  database snapshot can classify them exact;
- ordinary board, hand, and leader-area listeners now receive
  `amulet_activated` and `card_fused`; remaining cards that use those events
  still need individual structured rules and official-text verification;
- Faith currently supports verified `follower_evolved`, `amulet_destroyed`,
  genuine Enhance-card progression, and filtered named-follower entry, plus
  atomic value spending, non-consuming random payoff reads, and dynamically
  gained structured abilities. Mode-selection progression remains an explicit
  unsupported edge; Faiths and emblems now enforce the official shared
  five-slot leader-area limit. Other generated cards remain individually
  classified in the token/generated-card audit;
- Fusion-driven hand transforms and refusion are implemented. Other cards can
  listen to `card_fused`, but their individual reactions and later generated
  Artifact end-form abilities still require audited structured rules;
- future `奥义` cards still require explicit structured definitions; the
  generic gauge, thresholds, activation event, and repeated random target flow
  are implemented;
- future cards that cause normal evolution, restore SEP, or super-evolve
  multiple/selected followers will still need their own structured rules; real
  `10443110` now exactly covers both its `奥义` self-super-evolution and its
  cost-2-follower Ward listener;
- source-backed continuous stat or keyword derivation remains schema-rejected
  until a verified card requires it;
  source-backed leader damage modifiers are implemented and revalidate entry,
  leave, transform, and control changes;
- remaining trigger-ordering edge cases beyond the current death-batch
  ordering diagnostics and `death_batch_end` boundary triggers, including
  unsupported `death_batch_start` emblem triggers; no current exact rule uses
  that boundary;
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

Random self-play defaults to the official setup and reports first-player
distribution, completed mulligans, replaced cards, Extra PP use, illegal
actions, and reported/executable mask mismatches. The optional curve baseline
replaces opening cards above the configured printed-cost threshold. Run the
full acceptance gate with:

```powershell
python -m scripts.random_self_play --games 1000 --mulligan-policy curve --validate-invariants --assert-official-acceptance --output data/reports/official_self_play_acceptance.json
```

The checked-in seed-7 acceptance run completed all 1,000 mulligans, sampled
first player 499/501, made exactly 2,000 mulligan decisions, exercised Extra PP
1,885 times, and had zero truncations, illegal actions, or mask mismatches.

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

The environment has 112 actions:

- `0`: end turn
- `1..9`: play a hand slot
- `10..39`: five attacker slots times six target slots
- `40..44`: evolve a board slot
- `45..60`: resolve one of up to 16 pending choice options
- `61..78`: graveyard choice paging and slots
- `79..105`: fusion or special play-mode actions for hand slots
- `106..110`: super-evolve a board slot
- `111`: use the second player's Extra PP when available

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
