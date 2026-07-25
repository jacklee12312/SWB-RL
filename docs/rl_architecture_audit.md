# SWB RL 底层技术栈架构审计

审计日期：2026-07-17
P0 修复状态更新：2026-07-21
P1/P2 平台验收更新：2026-07-21
七职业评估与训练分布更新：2026-07-22
全卡规则覆盖推进更新：2026-07-25

## 2026-07-21 修复结果

本次平台硬化已经完成审计中的四项 P0，并完成目标内的 P1/P2 训练平台闭环：

- `TrainableCardCatalog` 以 `rule_coverage.json` 的 `covered_exact` 为准，
  当前纳入 708 张可收集卡；全部 826 张定义在 worker 启动时进入只读内存，
  对局解析不再访问 SQLite；
- Observation v3 使用固定 shape/dtype 的 NumPy 数组和 Gymnasium space，
  默认隐藏对手初始牌组，只有 `open_decklists=True` 才公开；
- 第九批规则将“仅失去谢幕曲”作为公开随从状态加入 Observation v2/v3，
  公共场面运行时向量由每实体15维增至16维，正式 schema 版本更新为
  `observation-v3.1`；111个动作编号保持不变；
- 第十批新增成功破坏结果绑定、伤害替换、全场护符目标和双方纹章移除，
  均位于通用规则核心；Observation `observation-v3.1` 与111个动作编号保持不变；
- 第十一批新增事件监听来源快照、敌方战场召唤、爆能强化事件过滤、
  不同名过滤抽牌、低阶奥义替代和固定种子随机能力，均位于通用规则核心；
  Observation `observation-v3.1` 与111个动作编号保持不变；
- 第十二批新增动态连击费用过滤、手牌攻击力绑定、授予谢幕曲、授予能力破坏
  免疫及从手牌召唤实体迁移。V1 继续保持294维与111个动作编号；结构化
  v2/v3 的手牌运行时每槽由10维增至12维、公开场面每槽由16维增至18维，
  正式 schema 更新为 `observation-v3.2`，旧检查点会通过版本/hash明确拒绝；
- 第十三批新增牌组当前费用向上取整减半、固定种子互异随机分支、随机随从
  排除本次攻击目标，以及基于目标快照的同名批量消失。5张真实卡通过现有
  command/pending-choice/action-mask 边界接入；V1、v2/v3 Observation 与111个
  动作编号均不变；
- 第十四批新增重复随从入场次数、来源已损失生命、主战者最大生命增减，以及
  攻击目标上下文向纹章触发帧传播。5张真实卡通过既有 command、纹章、
  pending-choice 与 action-mask 边界接入；V1、v2/v3 Observation 与111个动作
  编号均不变；
- 第十五批新增牌库重复物理副本放逐、完整运行时手牌精确复制、双方手牌原始
  费用前三高比较、敌方牌库随机精确变身及随从控制者回合末放逐。5张中立真实
  卡通过既有 command、pending-choice、纹章与 action-mask 边界接入；公开场面
  的回合末移除槽以位掩码同时表达破坏/放逐且 shape 不变，正式 schema 更新为
  `observation-v3.3`，111个动作编号保持不变；
- 第十六批新增融合素材不同名计数、出牌当前/原始费用变化事件过滤、敌方牌库
  随机物理卡完整复制、随从实际回复量联动主战者回复，以及失效手牌绑定目标
  的安全续解。5张真实卡通过既有 command、pending-choice、纹章与 action-mask
  边界接入；Observation `observation-v3.3` 与111个动作编号保持不变；
- 第十七批新增随从本回合攻击历史过滤、抽牌物理当前费用集合过滤、同当前费用
  手牌分组、敌方手牌随从属性变化，以及“任意其他随从或双方主战者”随机目标。
  5张真实卡通过通用事件、条件、目标与效果边界接入；Observation
  `observation-v3.3` 与111个动作编号保持不变；
- 第十八批新增持久护符破坏历史、最高原始费用与不同名历史筛选、按牌组物理卡
  当前费用批量放逐并传递实际数量、整手返回牌组等量重抽，以及敌方全体随从与
  主战者同时伤害。5张真实卡通过通用状态、条件、目标与效果边界接入；
  Observation `observation-v3.3` 与111个动作编号保持不变；
- `SWBAECEnv` 提供 PettingZoo AEC 双智能体接口，并通过官方 `api_test`；
- 规则胜负与训练上限已经分离：前者 `terminated`，后者 `truncated`；
  独立 `max_agent_steps` 也覆盖不推进回合的墓地翻页动作；
- 单次 `action_mask()` 只扫描一次 `legal_commands()`；`step()` 计算出的
  下一状态 mask 会复用于 observation 和 info；跨调用的合法命令、mask、
  Observation v3 和公共直方图均按单调版本缓存；
- training mode 关闭无界中文日志、限制诊断事件历史，同时保留完整
  `CoreTransition.events` 和确定性；Catalog/RuleBook 由父进程构建不可变、
  可序列化的 spawn-safe worker 快照，对局热路径不访问 SQLite 或规则文件；
- Observation v3、111-action、trajectory、Catalog、词表、训练池、覆盖报告、
  RuleBook 和种子派生都具有正式版本或稳定 SHA-256；
- persistent spawn vector worker、稳定卡牌 embedding、共享参数 recurrent masked
  PPO、PPO 多 worker 策略采样、原子中局
  checkpoint/resume、四类对手池、固定种子镜像评估、Gym 单学习方包装和
  GameEngine snapshot/restore/clone 均已实现并有正常/边界测试。
- PPO 新训练入口默认启用七职业确定性 7×7 有序循环；完整 49-episode 周期
  覆盖每个职业对阵一次，两个周期配合 episode 奇偶翻转学习方先后手。旧 API
  和旧检查点未声明职业集时仍保持单职业兼容。98-episode 分布审计对每个有序
  对局采样 2 次，学习方和对手方每职业各 14 次，并采样到全部 588 张 exact 卡。
- 固定评估默认每职业使用两组种子牌组并镜像先后手，共 28 局；2026-07-22
  smoke 的 28 局全部规则终止，0 截断、0 非法动作、0 action-mask mismatch，
  同时记录牌组、检查点、规则、Catalog 和环境版本哈希。该结果仍不用于推断
  策略强度。

V1/V2 和原 `ShadowverseEnv` 均保留兼容。当前状态是“可复现训练基线闭环”，
不是“已证明策略强度”或“已完成大规模分布式训练系统”。保存的 2-worker
CPU 训练、恢复、16 局镜像评估和吞吐数据都只作为 smoke/回归证据。

## 状态分级与验收证据

| 状态 | 范围 | 证据或限制 |
| --- | --- | --- |
| 已完成 | P0、state-version 缓存、training mode、正式版本/hash、稳定卡牌 embedding、单/多 worker PPO rollout、recurrent masked PPO、原子恢复、四类对手池、固定评估、AEC/Gym、snapshot/clone | 聚焦测试、官方 wrapper 检查、2-worker CPU smoke 训练到 1,304 步并恢复到 1,571 步、环境与 4-worker 随机 rollout 基准 |
| 部分完成 | 性能优化与评估覆盖广度 | 已有稳定基准和阈值；snapshot 约 0.82 MB/24.61 次每秒、clone 7.23 次每秒，足以作为正确性基础但仍需在高分支搜索前优化；七职业固定套件已落地，职业间 7×7 策略强度矩阵和长期统计实验尚未开始 |
| 尚未实现 | 分布式 learner、完整 MCTS、策略强度实验、自适应训练 curriculum | 不属于本次可复现 baseline；确定性七职业轮转只保证采样均衡，不是按学习难度动态调整的 curriculum，也不得从 smoke 胜率推断策略强度 |
| 明确不支持 | 11 张缺少 per-card 结构化规则的可收集卡、16 张文本不明确卡 | 不进入 exact 训练目录，不得视为已支持；规则覆盖与 RL 平台状态继续分开报告 |

## 结论

当前项目的规则引擎底座方向正确，不需要推倒重来；审计指出的 RL 环境协议、
训练数据入口和首个可复现采样/训练闭环已经完成。后续工作是扩大实验规模与
覆盖面，而不是重新设计规则核心。

可以将当前状态概括为：

- 规则引擎：成熟的确定性规则核心；
- RL 适配器：通过 AEC/Gym 官方检查的版本化接口；
- 训练系统：已形成可复现的 PPO、采样、对手池、评估和检查点 baseline，
  但尚不是分布式 learner 或策略强度实验平台。

因此，开始长期训练前不再需要另一轮底层重写；应先扩大固定评估职业和牌组，
再用真实实验数据决定 learner 拓扑、curriculum 和性能优化优先级。

## 审计范围

本次审计检查了以下边界：

- `GameEngine` 与 RL 动作编码的依赖方向；
- 状态、命令、事件、效果和结构化卡牌规则之间的职责；
- `ShadowverseEnv` 的动作、掩码、观察、奖励和终止语义；
- 数据库卡牌目录到训练牌组的入口；
- 隐藏信息、双人轮流决策和历史状态表达；
- 单进程采样成本、资源加载方式和未来并行化条件；
- 对 PPO/R2D2 类前向自对战和 MCTS 类搜索算法的适配能力。

本次审计不修改比赛规则，也不把尚未验证的卡牌语义视为已实现。

## 已有架构中应当保留的部分

### 确定性规则核心

- `GameEngine` 接受命令对象，不接受 RL 整数动作；
- 引擎持有唯一的 seeded RNG，同种子、牌组和命令序列可以复现；
- 非法命令不应改变状态，并由完整状态指纹测试保护；
- 状态稳定化、同时死亡、触发器、待选择和循环上限有显式边界；
- 事件表达已经发生的事实，效果表达请求的状态变化；
- 卡牌专属行为优先由结构化规则和通用原语组合，而不是写入卡牌 ID 分支。

这是整个项目最重要的资产。RL 层应继续包装规则核心，而不应把训练框架依赖
反向引入 `GameEngine`。

### 动作合法性和信息边界

- `legal_commands()` 是规则侧合法动作来源；
- RL 侧有固定动作编号和 action mask；
- 默认 `info()` 隐藏日志、事件对象、选项标签和实体 ID；
- v1/v2 都有对手手牌身份和剩余牌库顺序的隐私回归测试；
- 稀疏终局奖励是默认策略，没有把未经验证的塑形奖励写进规则核心。

这些设计适合作为后续标准多智能体环境的语义基础。

## 主要问题

### P0：训练卡池与规则覆盖脱节（已修复）

`CardRepository.training_pool()` 当前只选择：

- 卡牌类型为随从；
- 数据库 `support_level` 为 `basic` 或 `keyword`；
- 可收藏且属于中立或指定职业。

本地数据库测得每个职业实际只有 2 至 5 张不同卡进入该池，法术和护符为 0。
大量已经通过条款审计的精确结构化规则在数据库中仍保留旧的
`support_level=unsupported`，因此也不会进入训练池。

直接后果是：当前随机自对战使用少量基础随从反复组成 40 张牌组，既不能代表
完整卡池，也不能验证大部分已实现规则在训练分布中的行为。

建议：

1. 新建 `TrainableCardCatalog`，以规则覆盖报告中的 `mapped_exact` 为主要准入条件；
2. 把 `training_pool()` 改名或保留为明确的 legacy/basic smoke pool；
3. 提供合法牌组生成器，显式定义职业、中立、重复张数和精确信息策略；
4. 将训练池版本、规则快照哈希和牌组生成种子写入实验元数据。

实现：`swb/rl/catalog.py` 保存覆盖报告 SHA-256 和源数据快照，提供按职业的
exact pool、可复现的 40 张牌组采样和默认最多 3 张同名卡约束。旧
`training_pool()` 暂留为兼容 smoke API，训练/自对战脚本已切换到新目录。

### P0：v1 观察不能表达完整卡牌策略（已修复）

v1 手牌特征只有存在、费用、攻防、类型和少量运行时数值，没有卡牌身份。
费用和身材相同但能力不同的卡可能得到相同输入，因此 v1 不适合完整卡池训练。

v1 应冻结为兼容和冒烟接口；正式训练应基于 v2 的卡牌类别索引，进一步形成
固定 dtype、固定 shape 的 Observation v3。

v2 当前存在以下工程问题：

- 返回嵌套 Python `dict` / `tuple`，没有标准 observation space；
- 每次观察重新构建双方初始牌组、墓地和消失区的全卡表直方图；
- 826 张词表下，仅上述六个直方图就包含 4956 个计数；
- action mask 同时出现在观察和 `info()` 路径，容易重复计算；
- 双方初始牌组组成都会暴露，需要明确是公开牌表赛制还是隐藏牌表赛制。

Observation v3 已使用 NumPy 数组；PPO 将手牌和公共场面卡牌的稳定词表索引
送入可训练 embedding，而不是把 ID 当连续数值动态缩放，并通过
`open_decklists` 配置区分公开牌表与普通隐藏信息对局。隐藏牌表模式只允许
己方初始牌组、已公开对手卡牌和模型自身维护的 belief state。

实现：`observation_version="v3"` 输出 NumPy-only 字典并提供对应
Gymnasium `spaces.Dict`。默认 `open_decklists=False`，对手初始牌组直方图固定
为零；非当前决策方的 action mask 也固定为零。

### P0：双人环境协议尚未标准化（已修复 AEC 层）

`decision_player` 已经表达了“当前应由谁处理普通行动或待选择”，这与轮流行动的
Agent Environment Cycle 很接近，但当前接口仍是自定义 `StepResult`，没有标准：

- `action_space` 和 `observation_space`；
- 每个 agent 独立的 reward、termination 和 truncation；
- agent iteration、死亡步骤和环境一致性检查；
- 面向向量化采样器的固定 NumPy 边界。

建议在现有 `ShadowverseEnv` 外新增 PettingZoo AEC 兼容包装层，并保留当前类作为
向后兼容适配器。PettingZoo AEC 官方接口专门覆盖轮流行动、多智能体和 action
mask：<https://pettingzoo.farama.org/main/api/aec/>。

单智能体训练框架需要时，再提供“一个学习方 + 一个内置对手策略”的 Gymnasium
包装层。Gymnasium 要求明确的动作/观察空间以及
`observation, reward, terminated, truncated, info` 步进协议：
<https://gymnasium.farama.org/api/env/>。

实现：`swb/rl/aec_env.py` 在规则核心之外跟踪 `player_0/player_1`、逐 agent
reward/termination/truncation、死亡步骤、agent selection 和固定空间。单学习方
Gymnasium 对手包装仍是后续工作。

### P0：终止和截断语义混合（已修复）

当前超过最大回合时，引擎会按生命值决定胜者并结束比赛；环境随后可能同时返回
`terminated=True` 与 `truncated=True`。这会影响价值函数是否 bootstrap，并把
安全上限隐式变成“生命领先即获胜”的训练目标。

此外，墓地翻页动作不推进游戏回合，策略可以通过来回翻页形成无限 episode；
单纯依赖最大游戏回合无法截断这种轨迹。

建议：

- 正式比赛胜负只返回 `terminated`；
- 外部采样步数、诊断上限或墙钟上限只返回 `truncated`；
- 增加独立的 `max_agent_steps`；
- 在有限时域被定义为任务本身时，把剩余时域加入观察；
- 不使用生命领先裁决，除非它被明确规定为训练任务的一部分。

Gymnasium 对 termination/truncation 的区分及其 bootstrap 影响有专门说明：
<https://gymnasium.farama.org/main/tutorials/handling_time_limits/>。

实现：核心默认不设置人为最大回合；环境用 `max_game_turns` 和
`max_agent_steps` 截断且不产生胜者。规则胜负优先，因此同一步不会同时返回
`terminated=True` 和 `truncated=True`。

### P1：每个策略动作重复计算合法性和观察（已完成）

当前典型调用链可能在一个动作上重复生成合法命令：

1. 策略先调用 `action_mask()`；
2. `step()` 再次调用 `action_mask()`；
3. `info()` 再次调用 `action_mask()`；
4. v2 观察还会请求自己的 action mask；
5. 待选择状态下，一次 mask 构建可能分别扫描非选择和选择命令。

审计时旧路径（legacy follower pool）在本机固定牌组、关闭运行时不变量的
微基准结果：

- 规则核心约 627 action/s；
- 完整 v1 环境约 404 action/s；
- 826 卡词表的一次 v2 观察约 0.98 ms；
- 1000 局现有随机自对战约 217.5 秒完成。

这些数据只用于保留修复前基线和确定优化方向，不代表 exact 全卡池的当前吞吐，
也不是跨机器性能承诺。

建议给引擎状态增加单调递增 `state_version`，按版本缓存合法命令、动作掩码和
观察中的公共增量。一次 `step()` 应只生成一次下一状态观察与掩码。

实现：`GameEngine.state_version` 只在成功 reset/command 后递增；非法命令不递增。
`ShadowverseEnv.transition_version` 还覆盖墓地翻页、待选择同步和显式外部失效。
合法命令、action mask、完整 Observation v3 与墓地/消失区直方图按版本缓存，
初始牌组直方图只构建一次；返回观察为防御性副本。公开可变 `core/players` 访问是
显式 invalidation boundary，debug fingerprint 可检测保留引用造成的未声明修改。
最终基准记录 mask/观察缓存加速 34.75x/21.75x，阈值检查通过。

### P1：运行时资源和诊断路径不适合大规模 worker（已完成基线）

- 默认构造每个环境时可以重新加载 RuleBook；
- `CardRepository.get()` 每次打开 SQLite 连接，生成卡牌时也可能进入该路径；
- 即使 `debug_info=False`，引擎仍完整构造中文日志并累计全部事件历史；
- 没有进程级共享的不可变 CardCatalog/RuleBook 工厂；
- 没有 worker、episode 和环境种子的统一派生策略。

建议每个采样进程启动时一次性加载不可变 CardCatalog 和 RuleBook，比赛过程中
不访问 SQLite。训练模式应关闭完整文本日志，使用短环形公共历史和按需诊断，
同时保留命令级 `CoreTransition.events` 与规则所需的专用历史状态。

实现：Catalog 使用单 SQLite 连接/批量查询构建，不再通过 826 次单卡连接；
父进程生成不可变、pickle/spawn-safe 的 Catalog/RuleBook 快照。worker 启动只
反序列化一次，对局热路径不访问 SQLite 或重新读取/解析规则。training mode
关闭文本日志、以环形上限保留诊断事件但不裁剪当前 transition events。
SHA-256 domain separation 固定 master→worker→episode→deck/engine/policy 种子；
单/多 worker 确定性、异常传播、优雅关闭和无残留进程均有测试。

### P1：规则解析与解析器已经成为维护单体

当前 `resolution.py` 约 1.3 万行，`card_rules.py` 约 4800 行。现有测试使其仍然
可控，但继续扩张会增加代码审查、性能定位和并行开发成本。

建议在不改变行为的前提下逐段提取：

- command execution；
- effect runtime；
- trigger/listener runtime；
- death/stabilization；
- fingerprint/invariants；
- state snapshot/serialization。

不建议进行一次性重写。每次只迁移一个已有测试充分覆盖的边界。

### P2：面向搜索算法的确定性状态分支（基础已完成）

`GameEngine.snapshot/restore/clone` 与环境级对应接口现已公开：RNG、pending
choice、效果/触发队列、实体 ID、批次状态、事件历史和所有决定未来行为的可变
状态都会复制；RuleBook、resolver 等不可变资产共享。相同 snapshot 与命令序列
会得到相同事件、winner 和 fingerprint，非法分支与 clone 可变状态相互隔离。

这完成了搜索基础设施，不等于实现完整 MCTS，也还不是“廉价 undo”。当前基准
payload 约 0.82 MB，24.61 snapshot/s、7.23 clone/s；在高分支搜索前仍应分析
增量/结构共享方案。PPO 前向基线不依赖该优化。

## 推荐目标架构

```text
Card database + clause audit
            |
            v
Immutable CardCatalog + RuleBook
            |
            v
Deterministic GameEngine  <---- replay/fingerprint/invariants
            |
            v
SWB AEC environment  ---- action mask / Observation v3 / agent rewards
            |
            v
Vector rollout workers ---- fixed-policy PPO sampling / seeded episode identity
            |
            v
Recurrent masked policy --- checkpoints / opponent league / evaluation
```

规则核心继续保持零训练框架依赖。NumPy、PettingZoo、Gymnasium 和模型框架只存在于
环境、采样和训练包中。

## 分阶段执行顺序

### 阶段 A：训练语义冻结（已完成）

1. 定义公开牌表与隐藏牌表模式；
2. 定义逐 agent 的奖励归属和终止/截断语义；
3. 建立 exact-audit 驱动的训练卡池与合法牌组生成；
4. 冻结 Observation v3 和动作编号迁移规则。

### 阶段 B：标准环境和吞吐（已完成基线）

1. 新增 PettingZoo AEC 包装层；
2. 使用 NumPy observation/action mask；
3. 引入 state-version 缓存；
4. 预加载 CardCatalog/RuleBook；
5. 增加 training mode、向量 worker 和稳定性能基准。

### 阶段 C：首个可复现实验闭环（已完成 smoke）

1. 参数共享的 recurrent masked-policy 基线；
2. 对手快照池与固定基准对手；
3. Elo/胜率、非法动作率、平均回合和卡牌覆盖指标；
4. 保存代码提交、规则快照、训练池、种子和超参数；
5. 固定评估牌组与跨版本回归。

### 阶段 D：按算法需要扩展（部分完成）

- PPO/R2D2 路线：PPO 已接入持久多进程策略采样；下一步扩展独立 learner
  拓扑、跨职业策略强度矩阵和自适应 curriculum；确定性七职业训练轮转与
  七职业固定评估已完成；
- MCTS 路线：补充廉价状态分支、批量推理和隐藏信息处理方案；
- 模型式路线：建立公开状态与完整状态的明确隔离，防止训练标签泄漏到策略输入。

阶段 D 目前完成了可验证 snapshot/clone 搜索基础、稳定卡牌 embedding 和
PPO 多进程固定权重采样；完整 MCTS、分布式 learner、跨 worker 批量推理和
策略强度训练尚未实现。多进程 PPO 当前仅支持 current-policy self-play，
四类对手池仍由单进程采样路径完整支持。

## 进入大规模训练前的验收条件

- 训练卡池来自可审计 exact 规则，而不是 legacy follower pool；
- 法术、护符、待选择、融合、进化和超进化进入训练分布；
- 标准 AEC/Gym 环境检查通过；
- 普通模式不暴露对手隐藏牌表、手牌或牌库顺序；
- action mask 与可执行命令保持一致；
- termination/truncation 不混淆且翻页不能制造无限 episode；
- 相同实验种子能复现牌组、对局和采样轨迹；
- 单 worker 和多 worker 吞吐有稳定基准与回归阈值；
- 首个训练脚本能保存、恢复并评估检查点；
- 固定对手与镜像自对战结果能够被独立复跑。

## 最终建议

不需要替换 Python 或重写规则引擎。阶段 A/B/C 的可复现 baseline 已在确定性
命令核心之外完成；下一步应由真实实验需求驱动：若走 PPO 路线，先扩展评估职业、
curriculum 与 worker/learner 吞吐；若走搜索路线，先降低 snapshot/clone 成本。
任何策略结论都应另设长期训练和统计设计。本次 smoke 不改变规则覆盖事实：仍有
11 张缺规则卡和 16 张文本不明确卡，它们必须继续保持 unsupported 可见性。
