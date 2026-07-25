# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 823 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 719 |
| text_unclear | 16 |
| token_or_non_collectible | 91 |

### Clause Audit

| Clause status | Count |
|---|---:|
| mapped_exact | 714 |
| unverified_exact | 5 |
| partial | 0 |
| missing_rule | 0 |
| missing_primitive | 0 |
| text_unclear | 16 |
| token_separate_audit | 91 |

### Blocker Types

| Blocker | Count |
|---|---:|
| missing_rule | 0 |
| missing_schema | 0 |
| missing_primitive | 0 |
| missing_targeting | 0 |
| timing_unclear | 0 |
| text_unclear | 16 |
| external_blocker | 0 |
| audit_unverified | 5 |

## Exact-Coverage Clause Audit Issues

- **10214120**: covered_exact_without_clause_evidence — missing test_evidence
- **10314110**: covered_exact_without_clause_evidence — missing test_evidence
- **10554110**: covered_exact_without_clause_evidence — missing test_evidence
- **10574110**: covered_exact_without_clause_evidence — missing test_evidence
- **10714110**: covered_exact_without_clause_evidence — missing test_evidence

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
| 回复自己\d+点超进化点 | RESTORE_SUPER_EVOLUTION_POINTS | True |
| 回复自己\d+点进化点 | RESTORE_EVOLUTION_POINTS | True |
| 回复自己\d+点能量点 | RESTORE_MANA | True |
| 回复(?!自己\d+点(?:超进化点|进化点|能量点)) | HEAL_LEADER / HEAL_UNIT | True |
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
