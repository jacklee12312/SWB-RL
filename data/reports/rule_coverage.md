# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 740 |
| Cards with rules | 62 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 32 |
| supported_missing_rule | 558 |
| text_unclear | 23 |
| missing_primitive | 41 |
| covered_partial | 3 |
| token_or_non_collectible | 83 |

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
| 回合开始 | TURN_START trigger / Emblem | True |
| 回合结束 | TURN_END trigger / Emblem | True |
| 倒数 | COUNTDOWN / countdown | True |
| 抽取 | DRAW | True |
| 将.*加入手牌 | ADD_CARD | True |
| 回复 | HEAL_LEADER | True |
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
| 土之秘术|土之印 | EARTH_RITE (placeholder) | False |
| 融合 | FUSION (placeholder) | False |
| 信仰 | FAITH (placeholder) | False |
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
| 1 | 10012310 | 昆虫的忠告 | 精灵 | 1 | 法术 | high | Covered keywords: 造成.*伤害, 返回手牌 |
| 2 | 10151310 | 死神挥刀 | 梦魇 | 1 | 法术 | high | Covered keywords: 破坏 |
| 3 | 10171320 | 创造物充能 | 超越者 | 1 | 法术 | high | Covered keywords: 将.*加入手牌 |
| 4 | 10572310 | 苏生调律 | 超越者 | 1 | 法术 | high | Covered keywords: 将.*加入手牌, 破坏, 舍弃 |
| 5 | 10632310 | 正常的侵蚀 | 巫师 | 1 | 法术 | high | Covered keywords: 抽取, 破坏 |
| 6 | 10711310 | 人格切换 | 精灵 | 1 | 法术 | high | Covered keywords: 抽取, 返回牌 |
| 7 | 10751310 | 灵魂调律 | 梦魇 | 1 | 法术 | high | Covered keywords: 抽取 |
| 8 | 10021310 | 女仆的礼仪 | 皇家护卫 | 2 | 法术 | high | Covered keywords: 抽取, 返回牌 |
| 9 | 10031320 | 召唤真理 | 巫师 | 2 | 法术 | high | Covered keywords: 召唤 |
| 10 | 10153310 | 蛇神之怒 | 梦魇 | 2 | 法术 | high | Covered keywords: 造成.*伤害 |
| 11 | 10172310 | 生命的奔流 | 超越者 | 2 | 法术 | high | Covered keywords: 将.*加入手牌, 造成.*伤害 |
| 12 | 10221310 | 商谈成立 | 皇家护卫 | 2 | 法术 | high | Covered keywords: 抽取 |
| 13 | 10251310 | 诅咒派对 | 梦魇 | 2 | 法术 | high | Covered keywords: 将.*加入手牌 |
| 14 | 10252310 | 使唤蝙蝠 | 梦魇 | 2 | 法术 | high | Covered keywords: 召唤 |
| 15 | 10333310 | 虚假的术式 | 巫师 | 2 | 法术 | high | Covered keywords: 破坏 |
| 16 | 10341110 | 侮蔑的肯定者 | 龙族 | 2 | 随从 | high | Covered keywords: 抽取, 破坏 |
| 17 | 10373310 | 歼灭的歌声 | 超越者 | 2 | 法术 | high | Covered keywords: 破坏, 召唤 |
| 18 | 10411310 | 彗星 | 精灵 | 2 | 法术 | high | Covered keywords: 抽取, 造成.*伤害 |
| 19 | 10441120 | 梅格的挚友·玛丽亲 | 龙族 | 2 | 随从 | high | Covered keywords: 进化时, 超进化, 回合结束 |
| 20 | 10442310 | 至爱狂轰 | 龙族 | 2 | 法术 | high | Covered keywords: 回合结束, 造成.*伤害, 不能攻击|无法攻击 |
