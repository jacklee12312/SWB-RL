# Ability Registry Audit

Audit source: `data/audits/ability_registry.json`

## Summary

| Status | Count |
|---|---:|
| implemented | 18 |
| partial | 5 |
| placeholder | 11 |

## Abilities

| Ability | Registry | Primitive | Handler | Audit reason |
|---|---|---|---|---|
| 连击 | placeholder | covered | `handle_combo` | 回合计数、条件、表达式与增减原语已实现；带有连击标签但未编写结构化规则的卡牌仍必须显式报告未覆盖能力。 |
| ↳ tests |  |  |  | `tests/test_combo.py` |
| 协作 | placeholder | covered | `handle_cooperation` | 公开协作值和条件原语已实现；协作标签本身没有可统一执行的效果，未录入的逐卡条款继续产生占位报告。 |
| ↳ tests |  |  |  | `tests/test_cooperation.py` |
| 魔力增幅 | placeholder | covered | `handle_spellboost` | 手牌增幅次数、费用减免和结构化增幅效果已实现；未录入的增幅收益和触发条款仍是逐卡行为。 |
| ↳ tests |  |  |  | `tests/test_unit_state.py` |
| 土之秘术 | implemented | covered | `handle_earth_rite` | 土之印消耗、原子支付、嵌套效果和无资源分支均由结构化原语执行并有真实数据库测试。 |
| ↳ tests |  |  |  | `tests/test_earth_rite.py` |
| 土之印 | implemented | covered | `handle_earth_sigil` | 规范土之印生成、堆叠、离场和策动交互均作为公开战场状态实现。 |
| ↳ tests |  |  |  | `tests/test_earth_rite.py`, `tests/test_activate.py` |
| 觉醒 | placeholder | covered | `handle_overflow` | 最大PP派生状态、条件和表达式已实现；觉醒后的具体卡牌条款没有统一处理器，缺少结构化规则时保持可见。 |
| ↳ tests |  |  |  | `tests/test_overflow.py`, `tests/test_mana_resources.py` |
| 死灵术 | placeholder | covered | `handle_necromancy` | 墓场数原子支付和嵌套效果已实现；每张卡的支付值与收益必须由规则声明，未声明时报告占位。 |
| ↳ tests |  |  |  | `tests/test_necromancy.py` |
| 亡者召还 | placeholder | covered | `handle_reanimate` | 按费用与来源资格复活的原语已实现；标签未携带完整数值和逐卡目标语义，因此不能自动视为完整。 |
| ↳ tests |  |  |  | `tests/test_necromancy.py`, `tests/test_graveyard.py` |
| 疾驰 | implemented | covered | `handle_storm` | 入场回合即可攻击随从或主战者，且与攻击次数、限制和能力消除的交互均已测试。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_ability_removal_and_leader_modifiers.py` |
| 突进 | implemented | covered | `handle_rush` | 入场回合仅可攻击随从的合法性、限制和能力消除交互均已测试。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_unit_state.py` |
| 守护 | implemented | covered | `handle_ward` | 攻击目标强制、多个守护、离场和能力消除后的重新计算均由通用战斗合法性处理。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_ability_removal_and_leader_modifiers.py` |
| 必杀 | implemented | covered | `handle_bane` | 造成战斗伤害后的破坏语义、屏障和超进化保护顺序均已实现。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_super_evolution.py` |
| 潜行 | implemented | covered | `handle_ambush` | 手动攻击与敌方指定限制、攻击后失效和能力消除交互均已实现。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_ability_removal_and_leader_modifiers.py` |
| 吸血 | implemented | covered | `handle_drain` | 以实际造成的攻击伤害回复主战者，包含伤害修正、零伤害和生命上限钳制。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_ability_removal_and_leader_modifiers.py` |
| 倒数 | partial | covered | `handle_countdown` | 回合开始减值、归零破坏和谢幕曲顺序已实现；不同卡牌的启动、加减倒数及衍生条款仍需结构化定义。 |
| ↳ tests |  |  |  | `tests/test_last_words.py`, `tests/test_activate.py` |
| 威慑 | implemented | covered | `handle_intimidate` | 攻击目标合法性按威慑攻击力门槛集中计算，并覆盖多威慑、进化和能力消除。 |
| ↳ tests |  |  |  | `tests/test_intimidate.py` |
| 灵气 | implemented | covered | `handle_aura` | 敌方手动能力指定限制与随机、全体、己方效果的区别已实现，并覆盖能力消除。 |
| ↳ tests |  |  |  | `tests/test_aura.py`, `tests/test_targeting_and_zones.py` |
| 屏障 | implemented | covered | `handle_barrier` | 一次伤害防止、层数消耗、必杀和伤害修正顺序均由统一伤害流程处理。 |
| ↳ tests |  |  |  | `tests/test_keywords.py`, `tests/test_ability_removal_and_leader_modifiers.py` |
| 瞬念召唤 | implemented | covered | `handle_invocation` | 回合开始牌组扫描、种子随机顺序、每定义一次、战场满和事件继续均已实现并有真实卡完整示例。 |
| ↳ tests |  |  |  | `tests/test_invocation.py` |
| 入场曲 | partial | covered | `handle_fanfare` | 正常打出与替代模式的触发边界、来源和结构化规则调度已实现；未录入的逐卡入场曲内容仍显式占位。 |
| ↳ tests |  |  |  | `tests/test_core_engine.py`, `tests/test_play_modes.py` |
| 谢幕曲 | partial | covered | `handle_last_words` | 同时死亡收集、主动玩家顺序、来源快照和选择继续已实现；未录入的逐卡谢幕曲内容仍显式占位。 |
| ↳ tests |  |  |  | `tests/test_last_words.py`, `tests/test_triggers.py` |
| 进化时 | implemented | covered | `handle_on_evolve` | 手动与效果进化均发出统一事件并调度结构化进化规则，包含来源离场和选择继续。 |
| ↳ tests |  |  |  | `tests/test_triggers.py`, `tests/test_faith.py` |
| 超进化时 | implemented | covered | `handle_on_super_evolve` | 手动与效果超进化事件、资源区别和结构化规则调度均已实现。 |
| ↳ tests |  |  |  | `tests/test_super_evolution.py`, `tests/test_triggers.py` |
| 攻击时 | implemented | covered | `handle_on_attack` | 攻击确认后的结构化触发、来源离场和待决选择继续均已实现。 |
| ↳ tests |  |  |  | `tests/test_triggers.py` |
| 交战时 | implemented | covered | `handle_on_clash` | 双方交战来源的确定性触发顺序、伤害前结算和来源离场均已实现。 |
| ↳ tests |  |  |  | `tests/test_triggers.py`, `tests/test_keywords.py` |
| 爆能强化 | placeholder | covered | `handle_enhance` | 模式费用和操作框架已存在；未核对卡牌的基础模式替换、目标和全文条款不能由标签自动推断。 |
| ↳ tests |  |  |  | `tests/test_play_modes.py`, `tests/test_play_modes_audit.py` |
| 激奏 | placeholder | covered | `handle_accelerate` | 替代费用、卡牌类型和操作框架已存在；每张卡的激奏文本仍要求显式模式定义。 |
| ↳ tests |  |  |  | `tests/test_play_modes.py`, `tests/test_play_modes_audit.py` |
| 结晶 | placeholder | covered | `handle_crystallize` | 替代护符模式、费用和倒数框架已存在；逐卡结晶能力与谢幕曲必须显式定义。 |
| ↳ tests |  |  |  | `tests/test_play_modes.py`, `tests/test_play_modes_audit.py` |
| 模式 | placeholder | covered | `handle_choose` | 选择一项和可选确认命令框架已实现；选项、标签、目标和操作均为逐卡结构化内容。 |
| ↳ tests |  |  |  | `tests/test_decisions.py` |
| 融合 | partial | covered | `handle_fusion` | 材料选择、原子换区、继承、手牌变身、再融合、事件和RL曝光已实现；未录入的过滤器和融合反应仍逐卡缺失。 |
| ↳ tests |  |  |  | `tests/test_fusion.py` |
| 策动 | implemented | covered | `handle_activate` | 命令合法性、费用、每回合一次、倒数、目标、事件和RL动作掩码均已实现。 |
| ↳ tests |  |  |  | `tests/test_activate.py` |
| 纹章 | placeholder | covered | `handle_emblem` | 主战者区域状态、触发次数、期限和事件监听已实现；每个纹章定义及尚未需要的death_batch_start时序仍必须显式审计。 |
| ↳ tests |  |  |  | `tests/test_emblems.py`, `tests/test_emblems_advanced.py` |
| 信仰 | partial | covered | `handle_faith` | 实例、公开数值、进度、原子消费和动态能力已实现；命名随从、强化模式和共享五槽等真实语义仍未完成。 |
| ↳ tests |  |  |  | `tests/test_faith.py` |
| 奥义 | implemented | covered | `handle_union_burst` | 固定10/15阈值、手牌进化加速、顺序、事件和重复随机目标已实现并有真实卡完整示例。 |
| ↳ tests |  |  |  | `tests/test_union_burst.py` |
