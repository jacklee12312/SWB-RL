# PPO 截断对局动作循环诊断

## 摘要

- 来源截断局：35
- 精确重放一致：35
- 涉及超越者：28
- 重放 agent steps：17920
- 分类：`{"fusion_or_choice_state_cycle": 35}`
- 两步/三步状态环：{'2': 31, '3': 4}
- 循环入口牌：`[{"card_id": 90071220, "card_name": "过往核心", "games": 17, "actor_model_games": {"seed_20260902": 10, "seed_20260831": 7}}, {"card_id": 90071210, "card_name": "未来核心", "games": 9, "actor_model_games": {"seed_20260902": 7, "seed_20260831": 2}}, {"card_id": 10213310, "card_name": "花园的指引", "games": 6, "actor_model_games": {"seed_20260831": 5, "seed_20260902": 1}}, {"card_id": 10324110, "card_name": "篡夺的继承者·辛瑟莱兹", "games": 2, "actor_model_games": {"seed_20260831": 2}}, {"card_id": 90072120, "card_name": "城堡创造物", "games": 1, "actor_model_games": {"seed_20260831": 1}}]`
- 主循环执行模型：`{"seed_20260902": 18, "seed_20260831": 17}`

## 主循环聚类

| 局数 | 分类 | 动作循环 | 示例源局 | 可读动作 |
|---:|---|---|---:|---|
| 17 | fusion_or_choice_state_cycle | `fusion:90071220 -> choice:fusion:取消融合` | 25 | 开始融合 过往核心 → 为 过往核心 选择融合材料: 取消融合 |
| 8 | fusion_or_choice_state_cycle | `fusion:90071210 -> choice:fusion:取消融合` | 24 | 开始融合 未来核心 → 为 未来核心 选择融合材料: 取消融合 |
| 3 | fusion_or_choice_state_cycle | `fusion:10213310 -> choice:fusion:取消融合` | 26 | 开始融合 花园的指引 → 为 花园的指引 选择融合材料: 取消融合 |
| 2 | fusion_or_choice_state_cycle | `fusion:10324110 -> choice:fusion:取消融合` | 33 | 开始融合 篡夺的继承者·辛瑟莱兹 → 为 篡夺的继承者·辛瑟莱兹 选择融合材料: 取消融合 |
| 1 | fusion_or_choice_state_cycle | `fusion:10213310 -> choice:fusion:引路船工 -> choice:fusion:取消融合` | 142 | 开始融合 花园的指引 → 为 花园的指引 选择融合材料: 引路船工 → 为 花园的指引 选择融合材料: 取消融合 |
| 1 | fusion_or_choice_state_cycle | `fusion:10213310 -> choice:fusion:森林的游行 -> choice:fusion:取消融合` | 85 | 开始融合 花园的指引 → 为 花园的指引 选择融合材料: 森林的游行 → 为 花园的指引 选择融合材料: 取消融合 |
| 1 | fusion_or_choice_state_cycle | `fusion:10213310 -> choice:fusion:翅翼女王·提泰妮娅 -> choice:fusion:取消融合` | 31 | 开始融合 花园的指引 → 为 花园的指引 选择融合材料: 翅翼女王·提泰妮娅 → 为 花园的指引 选择融合材料: 取消融合 |
| 1 | fusion_or_choice_state_cycle | `fusion:90071210 -> choice:fusion:过往核心 -> choice:fusion:取消融合` | 176 | 开始融合 未来核心 → 为 未来核心 选择融合材料: 过往核心 → 为 未来核心 选择融合材料: 取消融合 |
| 1 | fusion_or_choice_state_cycle | `fusion:90072120 -> choice:fusion:取消融合` | 139 | 开始融合 城堡创造物 → 为 城堡创造物 选择融合材料: 取消融合 |

## 逐局结果

| 源局 | 对阵 | 截断回合 | 分类 | 最大同状态访问 | 尾部翻页占比 | 主循环执行模型 |
|---:|---|---:|---|---:|---:|---|
| 24 | 精灵 vs 超越者 | 2 | fusion_or_choice_state_cycle | 253 | 0.00% | {"seed_20260831": 2} |
| 25 | 精灵 vs 超越者 | 2 | fusion_or_choice_state_cycle | 254 | 0.00% | {"seed_20260831": 2} |
| 26 | 精灵 vs 超越者 | 2 | fusion_or_choice_state_cycle | 253 | 0.00% | {"seed_20260902": 2} |
| 27 | 精灵 vs 超越者 | 10 | fusion_or_choice_state_cycle | 236 | 0.00% | {"seed_20260831": 2} |
| 30 | 皇家护卫 vs 精灵 | 8 | fusion_or_choice_state_cycle | 243 | 0.00% | {"seed_20260831": 2} |
| 31 | 皇家护卫 vs 精灵 | 1 | fusion_or_choice_state_cycle | 171 | 0.00% | {"seed_20260831": 3} |
| 33 | 皇家护卫 vs 皇家护卫 | 12 | fusion_or_choice_state_cycle | 231 | 0.00% | {"seed_20260831": 2} |
| 81 | 巫师 vs 超越者 | 6 | fusion_or_choice_state_cycle | 250 | 0.00% | {"seed_20260831": 2} |
| 82 | 巫师 vs 超越者 | 9 | fusion_or_choice_state_cycle | 242 | 0.00% | {"seed_20260831": 2} |
| 83 | 巫师 vs 超越者 | 22 | fusion_or_choice_state_cycle | 212 | 0.00% | {"seed_20260831": 2} |
| 84 | 龙族 vs 精灵 | 1 | fusion_or_choice_state_cycle | 255 | 0.00% | {"seed_20260831": 2} |
| 85 | 龙族 vs 精灵 | 1 | fusion_or_choice_state_cycle | 171 | 0.00% | {"seed_20260831": 3} |
| 136 | 梦魇 vs 超越者 | 1 | fusion_or_choice_state_cycle | 255 | 0.00% | {"seed_20260831": 2} |
| 137 | 梦魇 vs 超越者 | 2 | fusion_or_choice_state_cycle | 253 | 0.00% | {"seed_20260831": 2} |
| 139 | 梦魇 vs 超越者 | 18 | fusion_or_choice_state_cycle | 219 | 0.00% | {"seed_20260831": 2} |
| 142 | 主教 vs 精灵 | 2 | fusion_or_choice_state_cycle | 169 | 0.00% | {"seed_20260831": 3} |
| 159 | 主教 vs 梦魇 | 15 | fusion_or_choice_state_cycle | 225 | 0.00% | {"seed_20260831": 2} |
| 170 | 超越者 vs 精灵 | 2 | fusion_or_choice_state_cycle | 254 | 0.00% | {"seed_20260902": 2} |
| 173 | 超越者 vs 皇家护卫 | 15 | fusion_or_choice_state_cycle | 225 | 0.00% | {"seed_20260902": 2} |
| 174 | 超越者 vs 皇家护卫 | 5 | fusion_or_choice_state_cycle | 251 | 0.00% | {"seed_20260902": 2} |
| 175 | 超越者 vs 皇家护卫 | 12 | fusion_or_choice_state_cycle | 236 | 0.00% | {"seed_20260902": 2} |
| 176 | 超越者 vs 巫师 | 2 | fusion_or_choice_state_cycle | 170 | 0.00% | {"seed_20260902": 3} |
| 177 | 超越者 vs 巫师 | 16 | fusion_or_choice_state_cycle | 223 | 0.00% | {"seed_20260902": 2} |
| 181 | 超越者 vs 龙族 | 11 | fusion_or_choice_state_cycle | 235 | 0.00% | {"seed_20260902": 2} |
| 183 | 超越者 vs 龙族 | 11 | fusion_or_choice_state_cycle | 239 | 0.00% | {"seed_20260902": 2} |
| 184 | 超越者 vs 梦魇 | 7 | fusion_or_choice_state_cycle | 247 | 0.00% | {"seed_20260902": 2} |
| 185 | 超越者 vs 梦魇 | 14 | fusion_or_choice_state_cycle | 230 | 0.00% | {"seed_20260902": 2} |
| 186 | 超越者 vs 梦魇 | 10 | fusion_or_choice_state_cycle | 240 | 0.00% | {"seed_20260902": 2} |
| 187 | 超越者 vs 梦魇 | 19 | fusion_or_choice_state_cycle | 216 | 0.00% | {"seed_20260902": 2} |
| 188 | 超越者 vs 主教 | 7 | fusion_or_choice_state_cycle | 247 | 0.00% | {"seed_20260902": 2} |
| 189 | 超越者 vs 主教 | 10 | fusion_or_choice_state_cycle | 240 | 0.00% | {"seed_20260902": 2} |
| 190 | 超越者 vs 主教 | 11 | fusion_or_choice_state_cycle | 239 | 0.00% | {"seed_20260902": 2} |
| 191 | 超越者 vs 主教 | 7 | fusion_or_choice_state_cycle | 249 | 0.00% | {"seed_20260902": 2} |
| 192 | 超越者 vs 超越者 | 2 | fusion_or_choice_state_cycle | 254 | 0.00% | {"seed_20260831": 2} |
| 193 | 超越者 vs 超越者 | 12 | fusion_or_choice_state_cycle | 235 | 0.00% | {"seed_20260902": 2} |

完整逐步动作、状态哈希、选择概率和卡牌信息见同名 JSON。
