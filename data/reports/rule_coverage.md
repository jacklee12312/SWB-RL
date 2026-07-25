# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 802 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 698 |
| text_unclear | 16 |
| supported_missing_rule | 21 |
| token_or_non_collectible | 91 |

### Clause Audit

| Clause status | Count |
|---|---:|
| mapped_exact | 698 |
| unverified_exact | 0 |
| partial | 0 |
| missing_rule | 21 |
| missing_primitive | 0 |
| text_unclear | 16 |
| token_separate_audit | 91 |

### Blocker Types

| Blocker | Count |
|---|---:|
| missing_rule | 21 |
| missing_schema | 0 |
| missing_primitive | 0 |
| missing_targeting | 0 |
| timing_unclear | 0 |
| text_unclear | 16 |
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

## Top 20 Recommended Cards

| # | Card ID | Name | Class | Cost | Type | Confidence | Why |
|---|---|---|---|---|---|---|---|
| 1 | 10524110 | 威猛的《战车》·奥辂昂 | 皇家护卫 | 9 | 随从 | high | Covered keywords: 回合结束, 造成.*伤害 |
| 2 | 10464110 | 土之法则·伽莱翁 | 主教 | 3 | 随从 | high | Covered keywords: 超进化, 回合结束, 不能攻击|无法攻击 |
| 3 | 10543110 | 破灭屠戮者 | 龙族 | 7 | 随从 | high | Covered keywords: 进化时, 超进化, 消失 |
| 4 | 10503210 | 大游戏世界 | 中立 | 1 | 护符 | high | Covered keywords: 谢幕曲, 倒数, 抽取 |
| 5 | 10703210 | 巴别隆城 | 中立 | 1 | 护符 | high | Covered keywords: 策动, 回合结束, 倒数 |
| 6 | 10162130 | 大地守护神·米维 | 主教 | 7 | 随从 | high | Covered keywords: 谢幕曲, 破坏, 召唤 |
| 7 | 10214120 | 缠绕密林·丽梅格 | 精灵 | 7 | 随从 | high | Covered keywords: 入场曲, 进化时, 超进化 |
| 8 | 10604110 | 恐惧的象征·欧米伽奥提普 | 中立 | 9 | 随从 | high | Covered keywords: 入场曲, 进化时, 超进化 |
| 9 | 10554120 | 奥夜花·释藤 | 梦魇 | 10 | 随从 | high | Covered keywords: 入场曲, 进化时, 超进化 |
| 10 | 10553310 | 严酷的奥夜花 | 梦魇 | 2 | 法术 | high | Covered keywords: 回合结束, 倒数, 抽取 |
| 11 | 10574110 | 转动的《命运之轮》·斯洛士 | 超越者 | 3 | 随从 | high | Covered keywords: 回合开始, 回合结束, 倒数 |
| 12 | 10663210 | 崇高的天书 | 主教 | 4 | 护符 | high | Covered keywords: 入场曲, 谢幕曲, 倒数 |
| 13 | 10664110 | 崇高的憎恶·康蒂玛 | 主教 | 4 | 随从 | high | Covered keywords: 入场曲, 谢幕曲, 进化时 |
| 14 | 10444120 | 世界的伙伴·佐伊 | 龙族 | 5 | 随从 | high | Covered keywords: 入场曲, 回合结束, 爆能强化 |
| 15 | 10564120 | 雾卷花·茎白 | 主教 | 7 | 随从 | high | Covered keywords: 入场曲, 抽取, 召唤 |
| 16 | 10354110 | 混融的继承者·莎木·纳克雅 | 梦魇 | 2 | 随从 | medium | Covered keywords: 入场曲, 进化时, 超进化 |
| 17 | 10362210 | 安息的神殿 | 主教 | 3 | 护符 | medium | Covered keywords: 谢幕曲, 策动, 倒数 |
| 18 | 10554110 | 充实的《恋人与节制》·米路缇欧&卢泽 | 梦魇 | 7 | 随从 | medium | Covered keywords: 入场曲, 进化时, 超进化 |
| 19 | 10314110 | 不弑的继承者·库露露 | 精灵 | 4 | 随从 | medium | Covered keywords: 入场曲, 进化时, 超进化 |
| 20 | 10714110 | 操量的安纳提玛·达斯特迪兹 | 精灵 | 4 | 随从 | medium | Covered keywords: 入场曲, 进化时, 连击 |
