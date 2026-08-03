# Tactical replay evaluation cases

This directory contains deterministic, checkpoint-independent policy-quality
cases extracted from completed simulator matches. A case is intended to answer
one narrow question such as “which follower should receive Super Evolution?”
It is not an engine-rules test and it does not assert that the annotated move
must eventually win the game.

## Case contract

Schema version 1 records:

- provenance: source history path/hash, match ID, and source checkpoint;
- setup: exact fixed-deck names and hashes, player roles, match seed, and
  official match setup;
- prefix: every action before the target plus a stable action-trace hash;
- replay guard: a hash of the complete private decision state at the target;
- decision: semantic preferred and disfavored action selectors;
- reference policy: the original checkpoint's logits and probabilities.

The prefix retains integer action IDs because it must reproduce the historical
command sequence exactly. The target annotation does not: it resolves the
current legal action by kind, source card ID, and same-card board occurrence.
This avoids treating transient entity IDs or presentation board slots as policy
semantics. The stored target-state hash makes any incompatible rules, deck, RNG,
or action-prefix change fail loudly.

The evaluator teacher-forces the recorded prefix. At every earlier decision of
the evaluated player it still runs the candidate checkpoint once to advance
that checkpoint's recurrent hidden state, then executes the recorded action.
Top-1 disagreement within the prefix is reported as a diagnostic but does not
change the state trajectory.

## Annotation policy

Add a case only when the desired comparison is narrow and reviewable. Prefer:

- `pairwise_preference` when one move is clearly better than one observed bad
  alternative;
- explicit category, objective, confidence, and rationale;
- statements about immediate tactical consequences rather than unsupported
  claims about eventual win probability;
- semantic source/target selectors rather than entity IDs.

Initially this suite is evaluation-only. If tactical cases later inform
training data or curricula, keep a separate held-out subset so a rising suite
score cannot be explained by direct memorization.

## Extraction and evaluation

Extract the first case from its local schema-v2 match history:

```powershell
python -m scripts.extract_tactical_scenario `
  data/match_history/20260802T081708672593Z-0ca7034d.json `
  --target-sequence 73 `
  --preferred-action-id 106 `
  --disfavored-action-id 107 `
  --case-id TACT-SE-0001 `
  --title "空场时优先超进化可立即攻击的疾驰随从" `
  --category super_evolution_target `
  --objective prefer_immediate_leader_pressure `
  --rationale "对方空场时，优先把超进化用于本回合可立即攻击的疾驰随从。" `
  --output data/tactical_scenarios/TACT-SE-0001-empty-board-storm.json
```

Evaluate one or more checkpoints by repeating `--case` and `--checkpoint`:

```powershell
python -m scripts.evaluate_tactical_suite `
  --case data/tactical_scenarios/TACT-SE-0001-empty-board-storm.json `
  --checkpoint data/checkpoints/example/final.pt `
  --device cuda `
  --output data/reports/tactical_suite/evaluation.json
```

The JSON report is machine-readable; a Markdown summary is written beside it.
