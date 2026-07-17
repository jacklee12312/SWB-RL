# SWB RL 底层技术栈架构审计

审计日期：2026-07-17

## 结论

当前项目的规则引擎底座方向正确，不需要推倒重来；真正需要补齐的是
RL 环境协议、训练数据入口和大规模采样基础设施。

可以将当前状态概括为：

- 规则引擎：成熟的确定性规则核心；
- RL 适配器：可验证的原型接口；
- 训练系统：尚未形成完整的模型、采样、对手池、评估和实验管理闭环。

因此，在开始大规模训练前，应先完成一次独立的“RL 平台硬化”切片。
继续增加卡牌规则不会解决训练分布、观察语义、双人协议或吞吐问题。

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

### P0：训练卡池与规则覆盖脱节

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

### P0：v1 观察不能表达完整卡牌策略

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

建议 Observation v3 使用 NumPy 数组和卡牌 embedding 输入，并通过
`open_decklists` 配置区分公开牌表与普通隐藏信息对局。隐藏牌表模式只允许
己方初始牌组、已公开对手卡牌和模型自身维护的 belief state。

### P0：双人环境协议尚未标准化

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

### P0：终止和截断语义混合

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

### P1：每个策略动作重复计算合法性和观察

当前典型调用链可能在一个动作上重复生成合法命令：

1. 策略先调用 `action_mask()`；
2. `step()` 再次调用 `action_mask()`；
3. `info()` 再次调用 `action_mask()`；
4. v2 观察还会请求自己的 action mask；
5. 待选择状态下，一次 mask 构建可能分别扫描非选择和选择命令。

本机固定牌组、关闭运行时不变量的微基准结果：

- 规则核心约 627 action/s；
- 完整 v1 环境约 404 action/s；
- 826 卡词表的一次 v2 观察约 0.98 ms；
- 1000 局现有随机自对战约 217.5 秒完成。

这些数据只用于确定优化方向，不是跨机器性能承诺。

建议给引擎状态增加单调递增 `state_version`，按版本缓存合法命令、动作掩码和
观察中的公共增量。一次 `step()` 应只生成一次下一状态观察与掩码。

### P1：运行时资源和诊断路径不适合大规模 worker

- 默认构造每个环境时可以重新加载 RuleBook；
- `CardRepository.get()` 每次打开 SQLite 连接，生成卡牌时也可能进入该路径；
- 即使 `debug_info=False`，引擎仍完整构造中文日志并累计全部事件历史；
- 没有进程级共享的不可变 CardCatalog/RuleBook 工厂；
- 没有 worker、episode 和环境种子的统一派生策略。

建议每个采样进程启动时一次性加载不可变 CardCatalog 和 RuleBook，比赛过程中
不访问 SQLite。训练模式应关闭完整文本日志，使用短环形公共历史和按需诊断，
同时保留命令级 `CoreTransition.events` 与规则所需的专用历史状态。

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

### P2：没有面向搜索算法的廉价状态分支

当前引擎支持确定性重放和指纹，但没有公开的 `clone`、`snapshot/restore` 或 undo
接口。对 PPO、R2D2 一类只需要前向采样的算法，这不是阻塞项；对 AlphaZero/MCTS
类每个决策需要大量分支模拟的算法，这是硬性缺口。

建议先确定首个训练基线。考虑到当前为隐藏信息卡牌游戏，优先建设参数共享、
action-mask、循环策略的前向自对战基线；只有明确选择搜索路线后，再为搜索设计
可验证的快照或撤销协议，避免过早重构状态核心。

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
Vector rollout workers ---- seeded deck curriculum / opponent assignment
            |
            v
Recurrent masked policy --- checkpoints / opponent league / evaluation
```

规则核心继续保持零训练框架依赖。NumPy、PettingZoo、Gymnasium 和模型框架只存在于
环境、采样和训练包中。

## 分阶段执行顺序

### 阶段 A：训练语义冻结

1. 定义公开牌表与隐藏牌表模式；
2. 定义逐 agent 的奖励归属和终止/截断语义；
3. 建立 exact-audit 驱动的训练卡池与合法牌组生成；
4. 冻结 Observation v3 和动作编号迁移规则。

### 阶段 B：标准环境和吞吐

1. 新增 PettingZoo AEC 包装层；
2. 使用 NumPy observation/action mask；
3. 引入 state-version 缓存；
4. 预加载 CardCatalog/RuleBook；
5. 增加 training mode、向量 worker 和稳定性能基准。

### 阶段 C：首个可复现实验闭环

1. 参数共享的 recurrent masked-policy 基线；
2. 对手快照池与固定基准对手；
3. Elo/胜率、非法动作率、平均回合和卡牌覆盖指标；
4. 保存代码提交、规则快照、训练池、种子和超参数；
5. 固定评估牌组与跨版本回归。

### 阶段 D：按算法需要扩展

- PPO/R2D2 路线：优先扩展采样吞吐、序列批次和对手多样性；
- MCTS 路线：补充廉价状态分支、批量推理和隐藏信息处理方案；
- 模型式路线：建立公开状态与完整状态的明确隔离，防止训练标签泄漏到策略输入。

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

不需要替换 Python 或重写规则引擎。应保留当前确定性命令核心，在其外增加标准
多智能体协议和训练基础设施。下一项最高优先级不再是继续扩大卡牌数量，而是完成
阶段 A 与阶段 B，使已经实现的规则真正进入正确、无泄漏且可扩展的训练分布。
