# Card Bug Audit Ledger

Schema version: `1`

## Severity And Training Policy

| Severity | Definition | Training policy |
|---|---|---|
| P0 | 非法动作、费用或替代模式合法性、攻击权限、隐藏信息泄漏、胜负、伤害、奖励，或 action mask 与 command 不一致。 | 立即暂停正式训练，修复并通过完整门禁前不得恢复。 |
| P1 | 八套训练卡组及其递归衍生闭包中常见卡牌的效果、目标、时机、区域移动或职业资源结算错误。 | 允许最小复现和修复验证；正式长训练前必须清零。 |
| P2 | 低频卡牌或罕见组合的状态或规则错误；当前训练轨迹中出现概率低，但仍须登记、保存复现并增加回归测试。 | 不阻塞小规模试验，可与修复工作并行。 |
| P3 | 只影响 UI、动画建议或文字日志，不改变引擎状态、合法动作、奖励或策略输入。 | 不阻塞训练，但仍须登记影响和处置结果。 |

P0 立即暂停正式训练；P1 必须在正式长训练前清零；P2 可与小规模试验并行；P3 不阻塞训练。

## Ledger Contract

| Field | Type | Required | Meaning |
|---|---|---|---|
| `bug_id` | `string` | yes | 稳定编号，格式为 SWB-CARD-0001。 |
| `severity` | `enum[P0,P1,P2,P3]` | yes | 按本报告固化定义分级。 |
| `status` | `enum[open,ruling_uncertain,fixed,closed_not_bug]` | yes | 当前处理状态。 |
| `card` | `object{card_id: integer|null, name: string}` | yes | 受影响卡牌；通用机制缺陷允许 card_id 为 null。 |
| `mechanic` | `string` | yes | 受影响的通用机制或规则族。 |
| `discovery_commit` | `string` | yes | 保存最小复现时的 Git HEAD。 |
| `minimal_seed` | `integer|null` | yes | 最小复现 seed；确定性 fixture 无随机性时为 null。 |
| `reproduction_file` | `string` | yes | 仓库相对路径，指向可移植最小复现包。 |
| `expected` | `string` | yes | 由卡牌文字和外部证据支持的预期结果。 |
| `actual` | `string` | yes | 修复前实际结果。 |
| `impact` | `string` | yes | 对规则、轨迹、训练或展示的明确影响。 |
| `affected_decks` | `array[string]` | yes | 受影响的固定训练卡组名称，稳定排序且不得重复。 |
| `fix_commit` | `string|null` | yes | 兼容字段：修复提交，或用户禁止提交时明确标注的工作树 diff hash；fixed 状态必须填写。 |
| `regression_tests` | `array[string]` | yes | 永久回归测试路径；fixed 状态必须非空。 |
| `notes` | `string` | yes | 裁定、关闭理由、checkpoint 影响或其他审计说明。 |

## Summary

- Total entries: 4
- Open P0 blockers: 1
- Open P1 blockers: 0
- Ledger P0 clear: false
- Ledger P0/P1 clear: false

These ledger flags do not authorize training by themselves; the eight-deck and full-catalog checklist gates must also pass.

## Entries

| ID | Severity | Status | Card | Mechanic | Reproduction |
|---|---|---|---|---|---|
| SWB-CARD-0001 | P0 | fixed | 10661110 崇奉的懦者 | Crystallize high-PP mode exclusivity and RL action legality | `data/reports/card_bug_audit/reproductions/SWB-CARD-0001.json` |
| SWB-CARD-0002 | P0 | fixed | 10424110 真红与群青·塞达&贝阿朵丽丝 | Enhance mandatory cost substitution and RL action legality | `data/reports/card_bug_audit/reproductions/SWB-CARD-0002.json` |
| SWB-CARD-0003 | P0 | fixed | 通用必杀机制 | Bane activation after zero combat damage and Barrier prevention | `data/reports/card_bug_audit/reproductions/SWB-CARD-0003.json` |
| SWB-CARD-0004 | P0 | open | 在线复盘隐私边界 | Ongoing match history API privacy boundary | `data/reports/card_bug_audit/reproductions/SWB-CARD-0004.json` |
