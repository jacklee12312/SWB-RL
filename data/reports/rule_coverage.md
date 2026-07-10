# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 98 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 63 |
| supported_missing_rule | 608 |
| text_unclear | 19 |
| missing_primitive | 38 |
| covered_partial | 7 |
| token_or_non_collectible | 91 |

## Rule Consistency Issues

- **999001**: card_id_not_in_database — Card 999001 has rules but is not in the database
- **999002**: card_id_not_in_database — Card 999002 has rules but is not in the database
- **999101**: card_id_not_in_database — Card 999101 has rules but is not in the database
- **999102**: card_id_not_in_database — Card 999102 has rules but is not in the database
- **999103**: card_id_not_in_database — Card 999103 has rules but is not in the database
- **999801**: card_id_not_in_database — Card 999801 has rules but is not in the database
- **999802**: card_id_not_in_database — Card 999802 has rules but is not in the database
- **999803**: card_id_not_in_database — Card 999803 has rules but is not in the database
- **999804**: card_id_not_in_database — Card 999804 has rules but is not in the database
- **999805**: card_id_not_in_database — Card 999805 has rules but is not in the database
- **999806**: card_id_not_in_database — Card 999806 has rules but is not in the database
- **999901**: card_id_not_in_database — Card 999901 has rules but is not in the database
- **999902**: card_id_not_in_database — Card 999902 has rules but is not in the database
- **999903**: card_id_not_in_database — Card 999903 has rules but is not in the database
- **999910**: card_id_not_in_database — Card 999910 has rules but is not in the database
- **999911**: card_id_not_in_database — Card 999911 has rules but is not in the database
- **999912**: card_id_not_in_database — Card 999912 has rules but is not in the database
- **999913**: card_id_not_in_database — Card 999913 has rules but is not in the database
- **999914**: card_id_not_in_database — Card 999914 has rules but is not in the database
- **999950**: card_id_not_in_database — Card 999950 has rules but is not in the database
- **999951**: card_id_not_in_database — Card 999951 has rules but is not in the database
- **999952**: card_id_not_in_database — Card 999952 has rules but is not in the database
- **999953**: card_id_not_in_database — Card 999953 has rules but is not in the database
- **999954**: card_id_not_in_database — Card 999954 has rules but is not in the database
- **999955**: card_id_not_in_database — Card 999955 has rules but is not in the database
- **999956**: card_id_not_in_database — Card 999956 has rules but is not in the database

## Primitive Keyword Map

| Keyword | Primitive | Covered |
|---|---|---|
| 入场曲 | FANFARE trigger | True |
| 谢幕曲 | LAST_WORDS trigger | True |
| 进化时 | EVOLVE trigger | True |
| 超进化 | SUPER_EVOLVE trigger | True |
| 攻击时 | ATTACK trigger | True |
| 交战时 | CLASH trigger | True |
| 连击 | COMBO condition / expression / add_combo | True |
| 觉醒 | OVERFLOW condition / expression | True |
| 策动 | ActivateAmulet command / ACTIVATE trigger | True |
| 威慑 | INTIMIDATE (placeholder) | False |
| 灵气 | AURA (placeholder) | False |
| 瞬念召唤 | Invocation deck scan / INVOKE trigger | True |
| 奥义 | UNION_BURST (placeholder) | False |
| 回合开始 | TURN_START trigger / Emblem | True |
| 回合结束 | TURN_END trigger / Emblem | True |
| 倒数 | COUNTDOWN / countdown | True |
| 抽取 | DRAW / DRAW_FILTERED | True |
| 将.*加入手牌 | ADD_CARD | True |
| 回复 | HEAL_LEADER / HEAL_UNIT | True |
| 造成.*伤害 | DAMAGE_LEADER / DAMAGE_UNIT | True |
| 破坏 | DESTROY | True |
| 消失 | BANISH | True |
| 召唤 | SUMMON | True |
| 返回手牌 | RETURN_TO_HAND | True |
| 返回牌 | RETURN_TO_DECK | True |
| 亡者召还 | REANIMATE | True |
| 舍弃 | DISCARD | True |
| 死灵术|唤灵 | NECROMANCY | True |
| 魔力增幅 | SPELLBOOST_HAND / passive | True |
| 无法使用 | cannot_be_played passive | True |
| 协作 | COOPERATION value / conditions | True |
| 纹章 | GAIN_EMBLEM / EMBLEM system | True |
| 土之秘术|土之印 | EARTH_RITE / ADD_EARTH_SIGILS / Earth Sigil board state | True |
| 融合 | BeginFusion command / Fusion material state | True |
| 信仰 | Faith leader-area state / evolution trigger | True |
| 必杀 | BANE keyword | True |
| 吸血 | DRAIN keyword | True |
| 屏障 | BARRIER keyword | True |
| 不能攻击|无法攻击 | ADD_ATTACK_RESTRICTION | True |
| 不能被指定|无法被能力指定 | ADD_TARGETING_RESTRICTION | True |
| 变形 | TRANSFORM | True |
| 爆能强化 | ENHANCE play mode | True |
| 激奏 | ACCELERATE play mode | True |
| 结晶 | CRYSTALLIZE play mode | True |
| 选择一项|模式 | CHOOSE_ONE / OPTIONAL | True |

## Top 20 Recommended Cards

| # | Card ID | Name | Class | Cost | Type | Confidence | Why |
|---|---|---|---|---|---|---|---|
| 1 | 10572310 | 苏生调律 | 超越者 | 1 | 法术 | high | Covered keywords: 将.*加入手牌, 破坏, 舍弃 |
| 2 | 10751310 | 灵魂调律 | 梦魇 | 1 | 法术 | high | Covered keywords: 抽取 |
| 3 | 10333310 | 虚假的术式 | 巫师 | 2 | 法术 | high | Covered keywords: 破坏 |
| 4 | 10341110 | 侮蔑的肯定者 | 龙族 | 2 | 随从 | high | Covered keywords: 抽取, 破坏 |
| 5 | 10373310 | 歼灭的歌声 | 超越者 | 2 | 法术 | high | Covered keywords: 破坏, 召唤 |
| 6 | 10441120 | 梅格的挚友·玛丽亲 | 龙族 | 2 | 随从 | high | Covered keywords: 进化时, 超进化, 回合结束 |
| 7 | 10511310 | 虫风花的飞翔 | 精灵 | 2 | 法术 | high | Covered keywords: 将.*加入手牌 |
| 8 | 10551310 | 奥夜花的开战 | 梦魇 | 2 | 法术 | high | Covered keywords: 造成.*伤害 |
| 9 | 10552310 | 残虐的炸裂 | 梦魇 | 2 | 法术 | high | Covered keywords: 造成.*伤害 |
| 10 | 10772310 | 闪光一瞬 | 超越者 | 2 | 法术 | high | Covered keywords: 超进化 |
| 11 | 10832310 | 其乐融融的团聚 | 巫师 | 2 | 法术 | high | Covered keywords: 回复, 魔力增幅 |
| 12 | 10853310 | 改变的流向 | 梦魇 | 2 | 法术 | high | Covered keywords: 超进化, 抽取, 返回牌 |
| 13 | 10042310 | 龙之启示 | 龙族 | 3 | 法术 | high | Covered keywords: 抽取 |
| 14 | 10102310 | 炽天使的福音 | 中立 | 3 | 法术 | high | Covered keywords: 抽取 |
| 15 | 10112310 | 薰交的思慕 | 精灵 | 3 | 法术 | high | Covered keywords: 抽取, 将.*加入手牌 |
| 16 | 10541310 | 波摇花的裁决 | 龙族 | 3 | 法术 | high | Covered keywords: 抽取, 造成.*伤害 |
| 17 | 10561310 | 雾卷花的激愤 | 主教 | 3 | 法术 | high | Covered keywords: 抽取, 返回牌 |
| 18 | 10673310 | 恶劣的天斧 | 超越者 | 3 | 法术 | high | Covered keywords: 回合结束, 造成.*伤害 |
| 19 | 10773310 | 瞬移斩击 | 超越者 | 3 | 法术 | high | Covered keywords: 造成.*伤害 |
| 20 | 10802310 | 救世的英姿 | 中立 | 3 | 法术 | high | Covered keywords: 抽取 |
