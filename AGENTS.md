# AGENTS.md

## Project Goal

This repository builds a deterministic Shadowverse: Worlds Beyond (SWB)
rules engine and an RL environment on top of a normalized card database.

The long-term goals are:

1. Accurately model SWB match rules.
2. Keep the deterministic game engine independent from RL action encoding.
3. Represent card-specific behavior as structured, auditable rules.
4. Expand card coverage without turning the engine into card-ID conditionals.
5. Keep seeded matches reproducible for testing and training.

Accuracy is more important than claiming broad card coverage. Unsupported
behavior must remain visible and must never silently behave as implemented.

## Source Of Truth

Before starting a task, inspect the code and tests instead of relying on a
static project summary. This file is intentionally durable guidance rather than
a precise feature snapshot.

Treat these as the current facts of record:

- Executable code and tests define supported engine behavior.
- `README.md` summarizes the user-facing current state.
- `docs/roadmap.md` tracks implementation priorities and known remaining work.
- `data/rules/` contains auditable card-specific rules.
- The SQLite database and `shadowverse_cards.json` determine current card
  counts; do not hard-code those counts in tests or documentation unless the
  task is explicitly about a fixed fixture.

If documentation and executable behavior disagree, trust the executable
behavior, then update the documentation in the same change when practical.

## Architecture Boundaries

- `swb/db/`: database schema, import, and card queries.
- `swb/engine/state.py`: mutable match state and zone/entity models.
- `swb/engine/commands.py`: player intentions accepted by the rules core.
- `swb/engine/events.py`: facts emitted during resolution.
- `swb/engine/effects.py`: reusable effect and target primitives.
- `swb/engine/card_rules.py`: structured card-rule loading.
- `swb/engine/abilities.py`: keyword registry and generic keyword behavior.
- `swb/engine/resolution.py`: command validation and deterministic resolution.
- `swb/engine/environment.py`: RL action encoding, masks, observations, rewards.
- `data/rules/`: card-specific machine-readable rules.
- `scripts/`: database, match, reporting, and smoke-test entry points.
- `tests/`: behavioral contracts.

`GameEngine` must not depend on RL integer action IDs. New game behavior should
first be expressible through commands, events, state, targets, and effects.
Only then should `ShadowverseEnv` expose it to RL.

## Required Working Method

For every task:

1. Read the relevant implementation and tests before editing.
2. Check `git status` and preserve user changes.
3. State the files and behavior you intend to change.
4. Make the smallest coherent implementation.
5. Add focused tests for normal behavior and edge cases.
6. Run the complete test suite and compile check.
7. Run an appropriate deterministic match or self-play smoke test.
8. Report changed files, test results, and remaining unsupported behavior.

Do not stop after writing a plan when implementation is requested.

## Mandatory Verification

Run from the repository root:

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

Increase self-play to 1000 games for broad shared-engine changes.

Never claim a command passed unless it was actually run successfully.

## Rules-Engine Invariants

- A seeded RNG owned by the engine is the only source of game randomness.
- The same seed, decks, and command sequence must reproduce the same result.
- Illegal commands and illegal RL actions must not mutate state.
- Entity identity uses stable `entity_id`; board indexes are presentation slots.
- State-based checks run after effect resolution where required.
- Simultaneous deaths must be collected before triggered effects resolve.
- Cards must belong to exactly one zone unless represented by a board entity.
- Board, hand, mana, health, countdowns, and resources must remain valid.
- Non-collectible cards cannot appear in initial decks.
- A deck must contain exactly 40 cards from its class and/or neutral.
- Hidden opponent hand contents and deck order must not enter RL observations.
- `action_mask()` and executable legal commands must agree.
- Unsupported abilities must emit explicit placeholder/coverage information.

Add invariant checks where they reduce the chance of silent state corruption.

## Card Rule Design

Prefer generic engine primitives plus structured rules:

```text
trigger -> conditions -> target specification -> operations
```

Do not implement card behavior through large `if card_id == ...` blocks in
`resolution.py` or `environment.py`.

Card IDs are acceptable as keys in `data/rules/*.json`. Generic mechanics such
as damage, summon, destroy, banish, transform, draw, return, cost changes,
keyword changes, and targeting belong in the engine.

Do not parse Chinese skill text at runtime as the final rule system. Text
parsing may produce reviewable rule drafts, but uncertain text must remain
unsupported until verified.

Maintain backward compatibility with existing rule JSON unless a migration is
implemented and all existing rules/tests are updated together.

## Targeting and Choices

- Centralize target candidate generation and target legality.
- Distinguish selected, random, all, and implicit targets.
- A target requirement with no legal candidates must follow explicit rule
  semantics: prohibit play, skip the operation, or use an alternate branch.
- Pending choices must safely handle targets that leave play.
- Multi-target choices must define whether duplicate targets are allowed.
- Target selection must be available through commands before RL action encoding.

## Events and Effects

Events describe what happened; effects describe requested state changes. Avoid
using events as mutable game state.

Resolution must have a deterministic and bounded order. If recursive triggers
can loop, enforce a maximum resolution-step limit and raise an error containing
enough event/effect history to diagnose the loop.

Do not mark an ability `IMPLEMENTED` merely because a handler exists. It is
implemented only when its real game-state behavior and edge cases are tested.

## RL Requirements

- Keep sparse terminal reward as the default.
- Any shaped reward must be optional and disabled by default.
- Update observation-size tests whenever observation features change.
- Encode decision-relevant public state, including class resources when added.
- Do not feed raw `entity_id` values as meaningful continuous features.
- Keep action encoding stable where practical; document intentional migrations.

## Testing Expectations

Tests should cover behavior, not private implementation details. Shared engine
changes normally require tests for:

- legal and illegal paths;
- no-target and target-required paths;
- source/target leaving play;
- simultaneous effects or deaths;
- deterministic seeded behavior;
- action-mask compatibility when exposed to RL;
- placeholder reporting for unsupported cases.

Use small synthetic `CardDefinition` fixtures for engine tests. Use the real
SQLite database for repository and end-to-end coverage tests.

## Git Safety

- Do not reset, checkout, delete, or overwrite unrelated user changes.
- Do not amend, rebase, push, or force-push unless explicitly requested.
- Do not create a commit unless explicitly requested.
- Before broad changes, recommend or create a feature branch only when asked.
- Keep generated Python caches and `data/backups/` untracked.
- Do not assume the current branch or initial local snapshot from old
  documentation; inspect `git status --short --branch`.

## Scope Control

Work in vertical slices. A good slice implements:

1. one generic mechanic;
2. its state and resolution behavior;
3. structured rule support;
4. tests;
5. one real-card demonstration;
6. RL exposure only if a player decision is involved.

Do not attempt to implement every card or every keyword in one change. If the
requested scope is too broad, complete the highest-priority coherent slice and
report the remaining work precisely.

## Priority And Roadmap

Unless the user gives a different priority, choose the next coherent slice from
`docs/roadmap.md`. Keep that roadmap current when a task changes implementation
status, test coverage, database assumptions, or known unsupported behavior.

Do not treat the roadmap as permission for broad rewrites. Work in vertical
slices and keep each slice independently reviewable.

## Completion Report

At the end of each task, provide:

- concise implementation summary;
- changed files;
- verification commands and results;
- generated log/report paths;
- known limitations and unsupported semantics;
- recommended next coherent slice.
