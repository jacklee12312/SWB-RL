# Runtime Coverage

本报告只读取结构化事件和稳定 card/clause ID，不解析中文运行日志。

## Summary

- Sessions: 1
- Not triggered: 440
- Triggered and passed: 15
- Triggered but not executed: 3

## Diagnostics

| Kind | Count |
| --- | ---: |
| `action_mask_mismatch` | 0 |
| `illegal_action` | 0 |
| `illegal_command` | 0 |
| `placeholder` | 0 |
| `resolution_step_limit` | 0 |
| `unsupported` | 0 |

## Aggregations

JSON 产物包含 `by_card`、`by_mechanic`、`by_deck` 和 `by_matchup` 四个机器可读聚合。

## Interpretation

`not_triggered` 与 `triggered_passed` 是不同状态；前者不能作为能力通过的证据。1.12 将使用强制场景和随机对局继续消除未解释空白。

## 1.11 Instrumentation Smoke

- Acceptance: `pass`
- Matchup: `international_qr_dragon_20260728__vs__international_qr_forest_20260728`
- Seed: 111
- Agent steps: 44
- Terminated: true
- Truncated: false
- Structured operation clauses in the training closure: 458

该 smoke 只验证 1.11 采集链路；强制场景、八套卡组对阵矩阵和 1,000 局采样属于 1.12，因此当前 `not_triggered` 不能解释为通过。
