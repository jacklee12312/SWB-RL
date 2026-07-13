# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 119 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 87 |
| supported_missing_rule | 630 |
| text_unclear | 18 |
| token_or_non_collectible | 91 |

### Clause Audit

| Clause status | Count |
|---|---:|
| unverified_exact | 64 |
| missing_rule | 630 |
| text_unclear | 18 |
| mapped_exact | 23 |
| token_separate_audit | 91 |

### Blocker Types

| Blocker | Count |
|---|---:|
| missing_rule | 630 |
| missing_schema | 0 |
| missing_primitive | 0 |
| missing_targeting | 0 |
| timing_unclear | 0 |
| text_unclear | 18 |
| external_blocker | 0 |
| audit_unverified | 64 |

## Exact-Coverage Clause Audit Issues

- **10001110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10011120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10011130**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text, test_evidence
- **10012120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10012310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10021310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10022110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10031210**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10031310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10031320**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10032310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10041110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10041130**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10041310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10051120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10051130**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10051310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10052110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10052310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10061110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text, test_evidence
- **10111310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10112120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10121310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10132320**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10151310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10153140**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10153310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10161130**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text, test_evidence
- **10162120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text, test_evidence
- **10171310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10171320**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10172310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10213310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10214110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10221310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10231120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10243310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10251120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10251310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10252310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10301310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10311310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10321310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10351120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10404110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10411310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10431120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text, test_evidence
- **10442310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10472310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10521310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10531310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10551120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10571310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10601110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10631310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10632310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10642310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10651110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10661310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10671110**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10671310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10711310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10721310**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text
- **10732120**: covered_exact_without_clause_evidence — missing explicit_exact_annotation, implemented_text

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
| 13 | 10102310 | 炽天使的福音 | 中立 | 3 | 法术 | high | Covered keywords: 抽取 |
| 14 | 10112310 | 薰交的思慕 | 精灵 | 3 | 法术 | high | Covered keywords: 抽取, 将.*加入手牌 |
| 15 | 10541310 | 波摇花的裁决 | 龙族 | 3 | 法术 | high | Covered keywords: 抽取, 造成.*伤害 |
| 16 | 10561310 | 雾卷花的激愤 | 主教 | 3 | 法术 | high | Covered keywords: 抽取, 返回牌 |
| 17 | 10673310 | 恶劣的天斧 | 超越者 | 3 | 法术 | high | Covered keywords: 回合结束, 造成.*伤害 |
| 18 | 10773310 | 瞬移斩击 | 超越者 | 3 | 法术 | high | Covered keywords: 造成.*伤害 |
| 19 | 10802310 | 救世的英姿 | 中立 | 3 | 法术 | high | Covered keywords: 抽取 |
| 20 | 10872310 | 纯净无垢的日常 | 超越者 | 3 | 法术 | high | Covered keywords: 超进化, 抽取, 回复 |
