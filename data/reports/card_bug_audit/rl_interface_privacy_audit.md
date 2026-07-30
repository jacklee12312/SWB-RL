# RL Interface, Observation, and Privacy Audit

This report is the executable evidence index for checklist section 1.10.

## Summary

- Action layout: `action-112-v2` (112 actions)
- Command types: 9
- Checklist contracts: 10
- Observation schemas: 2
- v3.6 fields: 18
- v4.1 fields: 61
- Full-pool cost boundary cases: 1546
- Command/mask mismatches: 0
- Illegal atomicity failures: 0
- Observation migration required: false
- Failures: 0
- Result: PASS

## Command matrix

| Command | Tests | Result |
|---|---:|:---:|
| `activate_amulet` | 1 | PASS |
| `attack` | 2 | PASS |
| `begin_fusion` | 1 | PASS |
| `choose` | 2 | PASS |
| `end_turn` | 1 | PASS |
| `evolve` | 1 | PASS |
| `play_card` | 2 | PASS |
| `super_evolve` | 2 | PASS |
| `use_extra_pp` | 1 | PASS |

## Checklist contracts

| Contract | Tests | Result |
|---|---:|:---:|
| `command_has_one_expected_action` | 2 | PASS |
| `true_mask_executes_expected_command` | 1 | PASS |
| `false_mask_is_atomic` | 2 | PASS |
| `pagination_complete_unique_bounded` | 2 | PASS |
| `opponent_hand_identity_hidden` | 1 | PASS |
| `opponent_deck_identity_and_order_hidden` | 1 | PASS |
| `public_history_tracks_actions_targets_and_zones` | 1 | PASS |
| `persistent_private_online_redacted` | 2 | PASS |
| `v3_6_and_v4_1_shape_dtype_version` | 2 | PASS |
| `observation_migration_decision_explicit` | 1 | PASS |

## Observation schemas

| Environment | Formal version | Fields | Manifest SHA-256 | Result |
|---|---|---:|---|:---:|
| `v3` | `observation-v3.6` | 18 | `380bba38c548b392ab4e993e574bc9357d36bf38c6c4415332f68d133c1afcef` | PASS |
| `v4.1` | `observation-v4.1` | 61 | `bba4f4b923de6de1e5144b2725efe97907379d88b157a4b232bac3bf203b54b6` | PASS |

## Migration decisions

| Change | Scope | Migration | Reason |
|---|---|:---:|---|
| `60d1c2f` | 必杀战斗结算语义 | no | 修复只改变既有战斗结果和既有事件内容，没有增加、删除或重解释策略输入字段；规则库哈希继续区分旧 checkpoint。 |
| `82bd251` | 在线复盘 API 脱敏 | no | 修复位于 simulator 在线序列化边界，不进入 GameEngine、ShadowverseEnv 或 PPO Observation。 |

## Privacy finding

- `SWB-CARD-0004`: P0 / fixed / fix `82bd251`
