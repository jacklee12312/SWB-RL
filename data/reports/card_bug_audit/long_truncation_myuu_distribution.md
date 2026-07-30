# Checklist 1.12 长局、截断与 Myuu 分布

## 结论

- 验收：`pass`；来源对局 11024，长局 95，截断 0，Myuu 对局 240。
- 完整卡池：10000 局，回合 p99/max 44/58，agent steps p99/max 189/1023。
- 八套矩阵：1024 局，回合 p99/max 37/49，agent steps p99/max 181/360。
- Myuu：240 局，截断 0，回合 p99/max 37/46，agent steps p99/max 175/360。

长局定义：`turn >= 45 (accepted full-pool p99 44 + 1) or agent_steps >= 257`。全部复现 manifest 保存在 JSON。

## 最长样本

| scope | game_id | sampling | turn | steps | engine_seed |
| --- | ---: | --- | ---: | ---: | ---: |
| full_pool | 9858 | current_policy | 19 | 1023 | 10063143868823724012 |
| full_pool | 9970 | current_policy | 15 | 582 | 15432816034253107362 |
| full_pool | 9976 | current_policy | 17 | 390 | 14514588313321710121 |
| full_pool | 9996 | current_policy | 24 | 378 | 1555596015839967122 |
| fixed_matrix | 988 | current_policy | 21 | 360 | 12189098270321849494 |
| full_pool | 8214 | random_legal | 55 | 297 | 14514777448238280512 |
| fixed_matrix | 275 | random_legal | 49 | 288 | 6368678457801101522 |
| full_pool | 9830 | current_policy | 16 | 272 | 17550382784711056645 |
| full_pool | 8289 | random_legal | 52 | 267 | 7919478159280191964 |
| full_pool | 9395 | random_legal | 51 | 253 | 382524615485880703 |

## 证据

- `data/reports/card_bug_audit/training_matrix_1000.json`
- `data/reports/card_bug_audit/full_pool_sampling_10000.json`
- JSON 报告保存全部长局、截断和 Myuu 对局的 seed、卡组哈希、动作轨迹哈希及终局指纹。
