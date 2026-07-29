# Observation v4 字段设计与审计结论

最后更新：2026-07-28。

## 结论

Observation v3.6 冻结为旧模型兼容格式；本文件记录的
`observation_version="v4"` / `observation-v4.0` 也已冻结为字段审计和旧模型
兼容格式。新训练使用其结构化后继 v4.1，详见
[`observation_v4_1_design.md`](observation_v4_1_design.md)。v4 不改变引擎
`GameState`，也不改变 112 个动作编号，只负责把当前玩家依法可见、会影响后续
转移或最优决策的状态编码给策略。

在当前 826 张卡的共享词表下，v4 有 49 个固定字段：55,992 个数值输入和
418 个卡牌词表索引。实体/候选/卡牌身份走共享 embedding；逐卡动态状态和动作
候选走专用编码器；全局投影只接收 28,989 个真正属于全局上下文的数值，不为
已经分流的字段分配无效权重。标准 entity-action 配置约 12.85M 参数。

## 从审计问题到修复

| 原审计结论 | v3.6 问题 | v4 修复 | 验证 |
|---|---|---|---|
| 不通过：手牌动态能力 | 手牌卡获得/失去关键词后可能与未变化卡产生相同观察 | 每个手牌槽编码当前有效关键词、永久增删和临时增删；临时项包含持续时间与到期方 | `test_hand_keyword_collision_is_removed` |
| 不通过：临时效果语义 | 只编码结果或数量，永久与“直到回合结束”等状态会碰撞 | 费用、攻防、攻击次数、攻击/指定限制、关键词修改均逐项编码类型、数值、持续时间和到期方，并提供容量溢出位 | `test_cost_modifier_duration_collision_is_removed`、`test_board_restriction_kind_and_duration_are_distinguishable` |
| 不通过：墓场候选身份 | 墓场翻页/选择动作没有把对应卡牌身份送给策略 | 每页 16 个槽分别携带卡牌 embedding、来源、派生/衍生标志和离场原因 | `test_graveyard_choice_slots_carry_card_identity_and_origin` |
| 有条件：牌库状态 | 只有初始牌表和牌库数量 | 加入己方当前剩余牌组的卡名计数及实体运行时合计；不输入抽牌顺序 | `test_exact_opponent_max_mana_and_current_own_deck_are_exposed` |
| 有条件：公共资源 | 部分资源只有派生标志，缺少精确基础值 | 双方生命上限、最大 PP、EP/SEP、Extra PP、职业资源和累计公开计数均显式编码 | 同上及 schema-space 测试 |
| 有条件：规则历史 | 粗粒度最近事件不能表达已破坏/登场历史 | 双方已破坏随从、护符和随从登场记录增加全局卡名直方图；每方最近 16 条还保留卡名、来源、回合和原因 | `test_public_history_includes_card_identity_but_not_raw_entity_id` |
| 有条件：公开事件 | 事件类型缺少来源卡、目标卡和具体公开结果 | 最近 32 个公开事件包含事件 one-hot、行动方、数值、公共区域引用、来源/目标卡牌 embedding 和稳定结构语义 | 同上及隐藏抽牌测试 |
| 有条件：纹章/信仰/监听器 | 只有身份、数量或当前值，缺少会改变下一次触发的运行时状态 | 加入信仰动态能力、纹章触发次数/每回合已用/倒数前值/随机历史、监听器触发次数与每回合已用位 | `test_emblem_and_listener_activation_runtime_is_exposed` |
| 有条件：融合与授予能力 | 只有融合数量、授予能力数量 | 己方手牌与公开场面保留融合素材卡牌 embedding；获得的谢幕曲/回合结束能力保留稳定结构语义 | schema、flattener 和策略前向测试 |
| 有条件：奥义 | 只有当前进度，种类、阈值和就绪状态需由卡名间接推断 | 手牌槽显式编码奥义/解放奥义种类、10/15 阈值、当前进度和各自就绪位 | `test_union_burst_kind_threshold_progress_and_ready_state_are_explicit` |
| 有条件：类别编码 | 来源、选择类型等曾以小数化 ordinal 输入，暗示不存在的大小关系 | 类别改为 one-hot；卡牌 ID 始终作为离散词表索引进入 embedding | `test_flattener_preserves_one_hot_semantics_and_embeds_auxiliary_cards` |
| 有条件：动作比较 | 候选身份混入全局平面输入，网络容易学习“第几个按钮” | 普通动作继续按来源/目标实体构造；选择和墓场动作直接绑定各自候选状态及卡牌 embedding | `test_entity_action_policy_accepts_v4_and_choice_scores_follow_candidate` |

## 49 个字段的正式定义

下表中的“我方”始终指观察视角；“对方”始终指另一玩家，而不是固定
`player_0`。所有空槽以 0 填充。卡牌索引 0 专用于空槽/未知卡。

| 字段 | 详细规定 |
|---|---|
| `player_state` | 依次编码我方、对方的生命/上限、PP/上限、牌库/手牌数、EP/SEP、已开始回合数、本回合进化/超进化状态、累计进化、本回合出牌/攻击/破坏、协作、墓影、土之印、信仰/纹章数、Extra PP、疲劳、主战者屏障、空牌库胜利规则及创造物登场种类数。 |
| `player_class_bits` | 我方和对方各 7 位职业 one-hot。 |
| `match_state` | 回合数、当前行动方关系、先后手关系、阶段 one-hot、双方换牌完成状态、是否待选择、墓场当前页和总页数。 |
| `own_hand_cards` | 我方 9 个手牌槽的共享卡牌词表索引。 |
| `public_board_cards` | 我方 5 个、对方 5 个场上槽的共享卡牌词表索引。 |
| `leader_area_cards` | 我方/对方各 5 个信仰与各 5 个纹章的来源卡索引。 |
| `graveyard_page_cards` | 当前墓场选择页 16 个候选槽的卡牌索引。 |
| `choice_option_cards` | 普通待选择的 16 个候选槽中，依法公开且可解析的卡牌索引。 |
| `history_source_cards` | 最近 32 个公开事件的来源卡索引；隐藏抽牌等不公开身份的事件为 0。 |
| `history_target_cards` | 最近 32 个公开事件的目标卡索引。 |
| `destroyed_follower_cards` | 我方最近 16 个、对方最近 16 个被破坏随从的卡牌索引。 |
| `destroyed_amulet_cards` | 我方最近 16 个、对方最近 16 个被破坏护符的卡牌索引。 |
| `follower_entry_cards` | 我方最近 16 个、对方最近 16 个登场随从的卡牌索引。 |
| `own_hand_fusion_cards` | 每个己方手牌槽最多 9 个融合素材的卡牌索引。 |
| `public_board_fusion_cards` | 每个公开场面槽最多 9 个融合素材的卡牌索引。 |
| `leader_modifier_source_cards` | 双方主战者伤害修改器的公开来源卡索引，每方最多 8 项。 |
| `own_initial_deck` | 我方初始牌表按共享词表统计的数量直方图。 |
| `opponent_initial_deck` | 默认全 0；只有显式 `open_decklists=True` 时才提供对方初始牌表。 |
| `own_current_deck` | 我方当前牌库中每种卡还剩几张；不提供顺序。 |
| `own_current_deck_runtime` | 按卡名聚合己方当前牌库实体数、动态费用、攻、体；用于牌库内变身或永久修改。 |
| `public_graveyards` | 我方、对方墓场卡名数量直方图。 |
| `public_banished` | 我方、对方消失区卡名数量直方图。 |
| `destroyed_follower_histograms` | 双方整局被破坏随从的卡名数量直方图，不受最近 16 条窗口限制。 |
| `destroyed_amulet_histograms` | 双方整局被破坏护符的卡名数量直方图。 |
| `follower_entry_histograms` | 双方整局登场随从的卡名数量直方图。 |
| `own_hand_origin_bits` | 每个己方手牌槽的当前来源与原始来源 one-hot。 |
| `public_board_origin_bits` | 每个公开场面槽的当前来源与原始来源 one-hot。 |
| `own_hand_state` | 存在位、当前费用、法术增幅及减费、融合数/本回合已融合、在手进化数、禁止打出、可收集性、效果破坏免疫、当前攻体、奥义进度、修改器/授予能力计数，以及奥义/解放奥义种类、阈值和就绪位。 |
| `own_hand_keyword_bits` | 每个己方手牌槽当前实际生效的运行时关键词位。 |
| `own_hand_modifier_state` | 费用、攻防和关键词修改器的逐项结构；每项含模式/数值、持续时间、到期方及必要的恢复标志，同时保留总量归一值和超限位。 |
| `own_hand_effect_bits` | 每个己方手牌槽最多 4 组动态获得谢幕曲的稳定结构语义及超限位。 |
| `public_board_state` | 卡牌类型、攻/当前体力/最大体力、倒数、土之印、攻击次数/可攻击性、疾突限制、进化/超进化、屏障、潜行、登场回合、能力移除、待破坏、策动、本回合进场、融合、破坏免疫及回合结束破坏/消失时机。 |
| `public_board_keyword_bits` | 每个公开场面槽当前实际生效的运行时关键词位。 |
| `public_board_modifier_state` | 场上攻防、攻击次数、攻击限制、被指定限制、关键词增删的逐项类型、数值、持续时间和到期方，并提供各列表超限位。 |
| `public_board_effect_bits` | 动态获得的谢幕曲、回合结束能力和随机选择历史的稳定结构语义。 |
| `leader_area_state` | 信仰值、动态能力、模式奖励；纹章当前/触发前倒数、逐触发次数和每回合已用位；主战者伤害修改器的数值、模式、持续时间、到期方、控制者及公开来源引用。 |
| `listener_state` | 己方手牌、双方场面、双方信仰/纹章来源的监听器触发次数、每回合已用位和定义数超限位；不扫描对方隐藏手牌。 |
| `choice_state` | 是否存在待选择、选择种类 one-hot、候选总数、要求数量、已选数量和是否允许重复。非决策方全 0。 |
| `choice_option_state` | 每个普通候选的公开实体引用 one-hot、主战者关系、已选位和选择语义；非决策方全 0。 |
| `graveyard_option_state` | 每个墓场候选的存在位、当前/原始来源、派生/衍生标志和进入墓场原因语义。 |
| `history_event_bits` | 最近 32 个公开事件的存在位和事件类型 one-hot。 |
| `history_actor_bits` | 最近 32 个公开事件行动方相对观察者的关系 one-hot。 |
| `history_amounts` | 最近 32 个公开事件的公开数值。 |
| `history_reference_bits` | 最近 32 个公开事件来源/目标在当前公开区域中的位置 one-hot；对象离场后可以为 0。 |
| `history_semantic_bits` | 最近 32 个事件中经隐私过滤后的模式、原因、来源等结构数据的稳定语义位。 |
| `destroyed_follower_state` | 与最近被破坏随从卡索引对齐的来源、派生/衍生、破坏回合、原因、模式和倒数。 |
| `destroyed_amulet_state` | 与最近被破坏护符卡索引对齐的来源、派生/衍生、破坏回合、原因、模式和倒数。 |
| `follower_entry_state` | 与最近登场随从卡索引对齐的登场回合和登场原因。 |
| `action_mask` | 与 112 动作布局严格对齐的合法动作位；非当前决策方全 0。 |

## 信息边界与固定长度

- 永远不输入对手隐藏手牌、未知牌库内容、双方牌库顺序、未来 RNG 或原始
  `entity_id`。事件元数据也要先按公开性过滤。
- “己方当前牌库组成”合法，因为玩家知道自己带了什么、已经公开/抽到了什么；
  牌库顺序仍未知。
- 完整累计规则事实优先使用直方图或引擎累计计数；最近记录用于表达顺序和原因。
  最近公共事件固定为 32 条，跨更长时间的策略信息由循环网络隐藏状态承接。
- 所有可变列表都有固定容量。费用/攻防/关键词/限制等列表同时编码真实数量和
  溢出位；历史列表另有全局直方图，因此超过最近窗口不会丢失卡名累计数量。
- 结构化效果身份使用去除实体 ID 后的规范 JSON 的 SHA-256 前 32 位。它比
  ordinal 有稳定语义，但理论上仍存在哈希碰撞；一旦未来卡池中实测碰撞，必须
  升级 schema 版本或改用审计词表，不能静默接受。

## 迁移规则

- v4.0 继续可由 PPO、Gym/AEC、向量采样和模拟器显式选择；新训练 CLI 默认
  使用 v4.1。
- v3.6 checkpoint 没有 `observation_version` 配置时，加载器根据存档中的
  `observation-v3.*` 清单自动恢复 v3，不会拿 v4 输入硬套旧权重。
- v4 改变输入层和 entity-action 字段布局，现有 v3.6 权重不能直接继续训练。
  112 动作含义没有变化。
- v4.1 再次改变输入层和 token 布局，因此 v4.0 权重也不能由 checkpoint
  加载器直接续训；两个旧版本仍按原 schema 严格恢复。
