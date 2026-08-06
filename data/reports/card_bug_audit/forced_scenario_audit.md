# 1.12 Forced Scenario Audit

- Acceptance: `pass`
- Minimum public-interface fixtures: 9/9 passed
- Direct mutations / invariant checks: 17 / 35
- Scope: 8 decks, 147 closure cards, 735 collectible + 91 generated cards
- Runtime clauses: 2151; unexplained 0

## Minimum Fixtures

| Scenario | Category | Status | Invariant checks |
|---|---|---:|---:|
| `minimum_cost_threshold` | `cost` | passed | 4 |
| `minimum_selected_target` | `target` | passed | 5 |
| `minimum_board_capacity` | `capacity` | passed | 4 |
| `minimum_resource_threshold` | `resource` | passed | 4 |
| `minimum_ordinary_evolution` | `ordinary_evolution` | passed | 4 |
| `minimum_super_evolution` | `super_evolution` | passed | 4 |
| `minimum_turn_end` | `turn_end` | passed | 3 |
| `minimum_turn_start` | `turn_start` | passed | 3 |
| `minimum_simultaneous_death` | `simultaneous_death` | passed | 4 |

## Evidence Boundary

Runtime 未触发的 clause 只记录为“由已重新执行的直接测试解释”，不会被改写为 runtime passed。完整 1,000/10,000 局分布由独立采样报告保存。
