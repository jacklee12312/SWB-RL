# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 218 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 184 |
| text_unclear | 18 |
| supported_missing_rule | 533 |
| token_or_non_collectible | 91 |

### Clause Audit

| Clause status | Count |
|---|---:|
| mapped_exact | 184 |
| unverified_exact | 0 |
| partial | 0 |
| missing_rule | 533 |
| missing_primitive | 0 |
| text_unclear | 18 |
| token_separate_audit | 91 |

### Blocker Types

| Blocker | Count |
|---|---:|
| missing_rule | 533 |
| missing_schema | 0 |
| missing_primitive | 0 |
| missing_targeting | 0 |
| timing_unclear | 0 |
| text_unclear | 18 |
| external_blocker | 0 |
| audit_unverified | 0 |

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
| 威慑 | INTIMIDATE attack-target legality | True |
| 灵气 | AURA manual enemy-effect target legality | True |
| 瞬念召唤 | Invocation deck scan / INVOKE trigger | True |
| 奥义 | Union Burst hand gauge / threshold operations | True |
| 回合开始 | TURN_START trigger / Emblem | True |
| 回合结束 | TURN_END trigger / Emblem | True |
| 倒数 | COUNTDOWN / countdown | True |
| 抽取 | DRAW / DRAW_FILTERED | True |
| 将.*加入手牌 | ADD_CARD | True |
| 回复 | HEAL_LEADER / HEAL_UNIT | True |
| 造成.*伤害 | DAMAGE_LEADER / DAMAGE_UNIT | True |
| 失去所有能力 | REMOVE_ALL_ABILITIES | True |
| 受到的伤害[+＋] | ADD_LEADER_DAMAGE_MODIFIER | True |
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
| 2 | 10341110 | 侮蔑的肯定者 | 龙族 | 2 | 随从 | high | Covered keywords: 抽取, 破坏 |
| 3 | 10373310 | 歼灭的歌声 | 超越者 | 2 | 法术 | high | Covered keywords: 破坏, 召唤 |
| 4 | 10441120 | 梅格的挚友·玛丽亲 | 龙族 | 2 | 随从 | high | Covered keywords: 进化时, 超进化, 回合结束 |
| 5 | 10511310 | 虫风花的飞翔 | 精灵 | 2 | 法术 | high | Covered keywords: 将.*加入手牌 |
| 6 | 10551310 | 奥夜花的开战 | 梦魇 | 2 | 法术 | high | Covered keywords: 造成.*伤害 |
| 7 | 10552310 | 残虐的炸裂 | 梦魇 | 2 | 法术 | high | Covered keywords: 造成.*伤害 |
| 8 | 10772310 | 闪光一瞬 | 超越者 | 2 | 法术 | high | Covered keywords: 超进化 |
| 9 | 10853310 | 改变的流向 | 梦魇 | 2 | 法术 | high | Covered keywords: 超进化, 抽取, 返回牌 |
| 10 | 10112310 | 薰交的思慕 | 精灵 | 3 | 法术 | high | Covered keywords: 抽取, 将.*加入手牌 |
| 11 | 10541310 | 波摇花的裁决 | 龙族 | 3 | 法术 | high | Covered keywords: 抽取, 造成.*伤害 |
| 12 | 10673310 | 恶劣的天斧 | 超越者 | 3 | 法术 | high | Covered keywords: 回合结束, 造成.*伤害 |
| 13 | 10773310 | 瞬移斩击 | 超越者 | 3 | 法术 | high | Covered keywords: 造成.*伤害 |
| 14 | 10802310 | 救世的英姿 | 中立 | 3 | 法术 | high | Covered keywords: 抽取 |
| 15 | 10872310 | 纯净无垢的日常 | 超越者 | 3 | 法术 | high | Covered keywords: 超进化, 抽取, 回复 |
| 16 | 10873310 | 开辟未来 | 超越者 | 3 | 法术 | high | Covered keywords: 回复, 造成.*伤害 |
| 17 | 10071310 | 来自异次元的枪击 | 超越者 | 4 | 法术 | high | Covered keywords: 将.*加入手牌, 破坏 |
| 18 | 10103310 | 神之雷霆 | 中立 | 4 | 法术 | high | Covered keywords: 造成.*伤害, 破坏 |
| 19 | 10332310 | 双重创造 | 巫师 | 4 | 法术 | high | Covered keywords: 召唤 |
| 20 | 10371310 | 丝线突袭 | 超越者 | 4 | 法术 | high | Covered keywords: 将.*加入手牌, 破坏 |
