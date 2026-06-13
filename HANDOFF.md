# HANDOFF - SWB Engine

**Updated:** 2026-06-13
**Branch:** `deepseek/effect-engine`

## Current Baseline

The graveyard, shadows, Necromancy, and Reanimate slice is implemented and
audited.

Current verified behavior includes:

- centralized graveyard entry and shadow gain;
- structured shadow gain/spend events;
- destroyed-follower history with deterministic death sequence;
- Necromancy payment and nested effect continuation;
- Reanimate candidate selection, deterministic tie-breaking, new entity
  creation, cooperation updates, and no Fanfare;
- shadow conditions and value expressions;
- public shadow features in the RL observation;
- structured real-card rules for `10051130` and `10551120`;
- repository-backed real-card coverage tests;
- explicit placeholder reporting only when a card has no structured
  Necromancy/Reanimate implementation.

## Verification

```text
python -m unittest discover -s tests -v
261/261 tests passed

python -m compileall -q swb scripts tests
passed

python -m scripts.random_self_play --games 1000 --seed 7
games=1000 wins=[611,389] draws=0 mean_turns=23.3

python -m scripts.necromancy_reanimate_scenario
passed, including deterministic replay
```

Deterministic RL logs:

- `data/rl_necromancy_audited_a.log`
- `data/rl_necromancy_audited_b.log`

Both produced:

```text
F6DE58F029D55605002AEF66EF40DCC1C353F56FA519C8BA47AE52BA577CB828
```

Generated logs and `data/swb.db` remain intentionally untracked.

## Important Semantics

- Cards entering the graveyard increase shadows exactly once.
- Banish, return, deck return, and transform do not increase shadows.
- Necromancy checks and consumes the controlling player's shadows once before
  executing its nested operation frame.
- A nested selected-target operation may pause and resume without repeating
  payment or losing its controller.
- Reanimate uses destroyed-follower history, not the graveyard list.
- Reanimated followers receive a new entity ID and reset runtime state.
- Reanimate does not trigger Fanfare.
- Equal-cost Reanimate candidates use the engine RNG.
- Shadow counts are public RL observation features.

## Known Limits

- Exact SWB treatment of derived/token followers in destroyed history still
  needs verification.
- Graveyard recovery and graveyard-card selection are not implemented.
- Reanimate currently selects by the implemented cost/history rules only;
  card-specific exclusions require verified structured conditions.
- Multi-card graveyard browsing would require a new player-choice model and RL
  action encoding.

## Recommended Next Slice

Harden graveyard-zone interactions before starting another class resource:

1. introduce stable graveyard entries with entity identity;
2. add generic graveyard count/filter expressions and conditions;
3. implement return-from-graveyard to hand and summon-from-graveyard;
4. define token/derived-card eligibility explicitly;
5. support selected graveyard targets through commands and pending choices;
6. expose any new decision through the RL adapter without leaking hidden state;
7. add verified real-card demonstrations and deterministic scenario tests.

Do not parse card text at runtime or add card-ID branches to
`resolution.py`.

## Working Method

Read `AGENTS.md`, inspect the current commit and tests, and treat executable
code as the source of truth. Preserve generated logs and the local database.
Do not create a commit unless explicitly requested.
