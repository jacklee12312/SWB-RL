# Rule Coverage Report

**Database**: `data/cards.sqlite3`
**Rules**: `data/rules`

## Summary

| Metric | Count |
|---|---|
| Total cards in DB | 826 |
| Cards with rules | 744 |
| Test/synthetic IDs with rules | 26 |

### Coverage Categories

| Category | Count |
|---|---|
| covered_exact | 640 |
| text_unclear | 16 |
| supported_missing_rule | 79 |
| token_or_non_collectible | 91 |

### Clause Audit

| Clause status | Count |
|---|---:|
| mapped_exact | 640 |
| unverified_exact | 0 |
| partial | 0 |
| missing_rule | 79 |
| missing_primitive | 0 |
| text_unclear | 16 |
| token_separate_audit | 91 |

### Blocker Types

| Blocker | Count |
|---|---:|
| missing_rule | 79 |
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
| 2 | 10532310 | 魔猫戏法 | 巫师 | 1 | 法术 | high | Covered keywords: 回复(?!自己\d+点(?:超进化点|进化点|能量点)), 召唤, 土之秘术|土之印 |
| 3 | 10412110 | 美妆少女·克洛伊 | 精灵 | 2 | 随从 | high | Covered keywords: 召唤, 返回手牌, 爆能强化 |
| 4 | 10464110 | 土之法则·伽莱翁 | 主教 | 3 | 随从 | high | Covered keywords: 超进化, 回合结束, 不能攻击|无法攻击 |
| 5 | 10602210 | 被侵略的世界 | 中立 | 3 | 护符 | high | Covered keywords: 策动 |
| 6 | 10502120 | 手持军配团扇的伟丈夫 | 中立 | 4 | 随从 | high | Covered keywords: 进化时, 破坏 |
| 7 | 10502110 | 星辉女神 | 中立 | 5 | 随从 | high | Covered keywords: 进化时, 将.*加入手牌, 舍弃 |
| 8 | 10163110 | 终焉的白骨圣堂之主 | 主教 | 6 | 随从 | high | Covered keywords: 入场曲, 造成.*伤害, 破坏 |
| 9 | 10521110 | 好施的名人 | 皇家护卫 | 6 | 随从 | high | Covered keywords: 入场曲, 回复(?!自己\d+点(?:超进化点|进化点|能量点)), 舍弃 |
| 10 | 10543110 | 破灭屠戮者 | 龙族 | 7 | 随从 | high | Covered keywords: 进化时, 超进化, 消失 |
| 11 | 10224120 | 雷维翁超越者·尤里乌斯 | 皇家护卫 | 8 | 随从 | high | Covered keywords: 入场曲, 回合结束, 回复(?!自己\d+点(?:超进化点|进化点|能量点)) |
| 12 | 10244120 | 绚丽凤凰·小凤 | 龙族 | 8 | 随从 | high | Covered keywords: 入场曲 |
| 13 | 10242210 | 炎龙之剑 | 龙族 | 1 | 护符 | high | Covered keywords: 谢幕曲, 策动, 破坏 |
| 14 | 10271120 | 猫偶 | 超越者 | 1 | 随从 | high | Covered keywords: 入场曲, 超进化, 将.*加入手牌 |
| 15 | 10272310 | 伊卡洛斯的飞翔 | 超越者 | 1 | 法术 | high | Covered keywords: 谢幕曲, 抽取 |
| 16 | 10503210 | 大游戏世界 | 中立 | 1 | 护符 | high | Covered keywords: 谢幕曲, 倒数, 抽取 |
| 17 | 10703210 | 巴别隆城 | 中立 | 1 | 护符 | high | Covered keywords: 策动, 回合结束, 倒数 |
| 18 | 10722310 | 无音的包围 | 皇家护卫 | 1 | 法术 | high | Covered keywords: 将.*加入手牌, 协作 |
| 19 | 10302110 | 抗拒叹息之人 | 中立 | 2 | 随从 | high | Covered keywords: 进化时, 超进化, 必杀 |
| 20 | 10471120 | 爆燃老大·翼 | 超越者 | 2 | 随从 | high | Covered keywords: 入场曲, 奥义 |
