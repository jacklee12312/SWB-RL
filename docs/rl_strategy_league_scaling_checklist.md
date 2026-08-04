# PPO 策略能力、League 自博弈与人类数据下一阶段 Checklist

最后更新：2026-08-04

## 目标与结论边界

本阶段的目标不是继续用更多 seed 做“摸奖”，而是把已经训练出的不同策略
组织成可审计、可复现、能抑制循环克制的训练种群，并验证这种训练是否真的
提高了跨对手、跨职业和人类对局中的策略质量。

本 checklist 将“好结果”分成三个层级：

1. **工程可用**：零非法动作、零 action-mask mismatch、无异常截断，UI 对局
   能稳定完成。
2. **社区有挑战性**：面对冻结模型池和普通玩家时能稳定表现出资源保留、
   长程规划和多种打法，不依赖某个幸运 seed 或单一克制关系。
3. **顶尖人类水平**：需要真实人类天梯数据与更大规模验证；当前阶段不作承诺。

阶段结论必须来自冻结对手、公共评估 seed、完整 7x7 职业矩阵、战术回放集和
人类留出集。训练时的自博弈胜率、单个 Elo 或挑出的最佳 seed 都不能单独作为
成功证据。

## 论文依据

- [OpenAI Five](https://openai.com/index/dota-2-with-large-scale-deep-reinforcement-learning/)
  证明 PPO、当前策略自博弈和历史策略混合可以扩展到很强的策略，但其结果依赖
  极大规模经验，不能据此假定 1M/3M steps 自然足够。
- [AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
  使用 Prioritized Fictitious Self-Play（PFSP）、主模型、主模型克星和 League
  克星处理循环克制与遗忘；还使用人类回放初始化和维持策略多样性。
- [PSRO](https://papers.neurips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html)
  指出独立 RL 会过拟合共同训练的对手，并用 payoff matrix、策略混合和近似
  best response 提高泛化。
- [NFSP](https://arxiv.org/abs/1603.01121) 与
  [DeepNash](https://arxiv.org/abs/2206.15378) 表明不完全信息游戏中的普通
  自博弈可能发散或围绕均衡循环，需要历史平均、正则化或其他博弈论约束。
- [DouZero](https://proceedings.mlr.press/v139/zha21a.html) 表明结构化动作表示、
  并行 actor 和从零自博弈能在复杂牌类游戏中取得强结果。
- [Suphx](https://arxiv.org/abs/2003.13590) 表明先学习顶尖人类行动，再进行
  自博弈强化，是复杂牌类长程规划的有效路线。
- [Deep RL at the Edge of the Statistical Precipice](https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
  要求少量 seed 实验报告区间、不确定性和稳健聚合指标，不能只比较点估计。
- [ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
  作为模型策略在 League 与人类数据后仍然平台化时的搜索路线；本阶段不提前
  实现不完全信息搜索。

这些论文支持的是总体方法和决策门，不提供适用于 SWB 的固定超参数。本文中
的百分比阈值是项目预注册的工程验收标准，必须通过本项目实验确认。

## 当前事实基线

- Observation v4.1、`entity_action_v1`、循环隐藏状态、112-action mask、完整
  7x7 职业调度和 sparse terminal reward 是下一阶段的冻结语义基线。
- 当前模型约 5.58M 参数。阶段开始时不同时扩大网络、修改 reward、修改 PPO
  batch/epoch/GAE 或引入搜索。
- 六个新 seed `20260903`–`20260908` 已分别训练到约 1M agent steps；每个
  约完成 13,000–14,000 局，训练速度约 154–158 agent steps/s。
- 三个旧 seed `20260831`、`20260901`、`20260902` 已到约 3M steps，但它们
  经历过较旧的 Extra PP 行为，优先作为冻结锚点，不能和新规则下的续训模型
  混称为严格同条件重复实验。
- 当前 `OpponentPool` 支持当前策略、自己的历史 checkpoint、random legal 和
  fixed first legal，并按静态权重选择；尚不支持共同的跨 seed 外部池、payoff
  更新、PFSP 或 League 角色。
- 当前 33 组、每组 196 局的 seed matrix 正在写入
  `data/reports/ppo_7x7_seed_matrix_20260804/`。只有
  `launcher_status.json.state == "completed"` 后才能生成最终矩阵结论。
- `docs/roadmap.md` 当前存在用户改动；本阶段不得覆盖或擅自提交它。

## Checklist 使用规则

- 严格按照 3.0–3.10 顺序执行。只有被明确标注为可并行的人类数据收集工作
  可以和模型训练并行。
- 每项先保存冻结基线和失败复现，再做最小实现，再测量；外部共同池、PFSP、
  exploiter、人类预训练和搜索不得在一个实验中同时引入。
- 所有实验必须预先写明训练 seed、评估 seed、checkpoint、规则/数据库/
  Observation/action hash、职业调度、训练步数、对手池 manifest 和主指标。
- 所有训练 run 都进入汇总，包括失败、退化和中断 run；不得删除坏 seed 后只
  汇报最佳结果。
- 196 局用于筛查明显差异；影响默认训练方案的近距离比较使用至少 10 个公共
  match seed 覆盖完整 7x7 和双方位置，即至少 980 局/模型对。
- 所有胜率报告包含 95% CI。多 seed 汇总至少报告逐 seed 值、中位数、IQM、
  paired bootstrap CI 和 performance profile。
- 单一 Elo 只作辅助指标；主指标必须包含冻结锚点表现、payoff matrix 下的
  population/worst-case 指标、职业最差项和战术回放准确率。
- League 训练保持 sparse terminal reward 为默认。人类数据通过行为克隆或
  明确的辅助损失进入，不能暗中变成手写 shaped reward。
- 每个编号阶段完成后更新 checkbox、保存机器可读报告，并创建独立 Git
  checkpoint；不得混入用户的 `docs/roadmap.md` 或正在运行的评估产物。

---

# 阶段 3：从独立 seed 扩展到可验证的 League Scaling

## 3.0 冻结实验合同与成功指标

- [ ] 运行 `git status --short --branch`，记录当前 commit、分支和全部用户改动。
- [ ] 冻结 SQLite、RuleBook、Catalog、训练池、八套固定牌组、7x7 实际牌组
  调度、Observation、action schema、policy architecture 和 seed derivation
  的版本/hash。
- [ ] 冻结六个新 1M checkpoint 和三个旧 3M 锚点的 SHA-256、训练步数、规则
  版本、Extra PP 语义和来源报告。
- [ ] 建立训练 seed、调参 seed、最终评估 seed 三个不重叠集合；最终评估 seed
  不得用于 PFSP 权重更新。
- [ ] 把以下项目预注册为主指标：
  - [ ] 对冻结锚点池的配对平均胜率及 95% CI；
  - [ ] payoff matrix 中 meta-strategy 的最差期望收益/可利用性代理；
  - [ ] 7x7 职业格子的最差项和第 10 百分位；
  - [ ] 战术回放集的 preferred-action top-1、top-k 和 preference margin；
  - [ ] 零非法动作、零 mask mismatch 和截断率。
- [ ] 将训练吞吐、GPU/CPU/RAM、episode 长度、entropy、value loss、policy loss、
  clip fraction、KL、explained variance、grad norm 和 opponent selection 分布
  注册为诊断指标，不把其中任何一个单独当作策略变强。
- [ ] 保存字节稳定的冻结 manifest，并增加字段完整性和 hash 失效测试。

产物：

- `data/reports/league_training/baseline_manifest.json`
- `data/reports/league_training/evaluation_protocol.json`
- `data/reports/league_training/checkpoint_registry.json`

完成标准：

- 同一工作区重复生成 manifest 字节一致。
- 任一规则、牌库、Observation、action、checkpoint 字节变化都会使旧实验合同
  明确失效，而不是静默继续。

## 3.1 完成并审计当前 seed payoff matrix

- [ ] 等待现有 33-pair 队列自然完成；不得重启、覆盖或混合修改运行中产物。
- [ ] 验证 33/33 组、6468/6468 局均存在，且无缺失、重复或错误 checkpoint。
- [ ] 验证全部对局非法动作和 action-mask mismatch 为 0；逐组报告截断。
- [ ] 记录每组胜率、Wilson 95% CI、双方位置胜率、相对 Elo、平均局长和 7x7
  职业格子。
- [ ] 明确把旧 3M 对新 1M 标为跨规则历史锚点比较，不能解释成纯 steps ablation。
- [ ] 对 95% CI 覆盖 50% 的接近模型不强行排序。
- [ ] 对可能影响后续池组成的接近对局补跑到至少 980 局。
- [ ] 生成一个机器可读的 antisymmetric payoff matrix；验证交换行列时符号相反。
- [ ] 检测三策略循环：若 A>B、B>C、C>A 的每条边均超过预注册阈值（首轮
  使用 55%），保存循环和各边 CI，不用单一 Elo 抹平该关系。

产物：

- `data/reports/league_training/seed_payoff_matrix.json`
- `data/reports/league_training/seed_payoff_matrix.md`
- `data/reports/league_training/close_pair_confirmations/`

决策门 3.1：

- 矩阵未完整、存在非法动作/mask mismatch 或 checkpoint 来源无法确认时，
  不进入正式 League 训练。
- 未检测到显著循环不代表可以丢弃策略池；仍继续 3.2 检查覆盖和遗忘。

## 3.2 建立 population 评估与 meta-strategy

- [ ] 从 payoff matrix 计算并保存：
  - [ ] uniform mixture 的期望收益；
  - [ ] 零和 meta-game 的 Nash/线性规划 mixture；
  - [ ] 每个 checkpoint 在 mixture 下的期望收益；
  - [ ] effective population size 和每个策略的 mixture 权重；
  - [ ] 相对当前 population 的 best-response/exploitability proxy；
  - [ ] 全局与逐职业的循环克制图。
- [ ] 对求解器设置确定性排序和固定容差；重复运行结果字节一致。
- [ ] 对奇异、重复、全平和非严格反对称矩阵提供显式诊断，不静默返回权重。
- [ ] 使用 paired bootstrap 对 class cell 和 match seed 重采样，报告各 population
  指标的 95% CI。
- [ ] 增加性能 profile 和 IQM 汇总，避免均值被一个 seed 或一个职业支配。
- [ ] 保存“候选对手为何进入池”的证据：Nash 权重、独特克制边、历史锚点、
  风格/职业覆盖或固定基准；不得只因为它的 Elo 高。

产物：

- `scripts/report_ppo_league_meta_game.py`
- `data/reports/league_training/meta_game.json`
- `data/reports/league_training/meta_game.md`
- `tests/test_ppo_league_meta_game.py`

完成标准：

- 合成石头剪刀布、完全传递、重复策略和不完整矩阵测试全部通过。
- 真实矩阵可重生成，所有输入 report 和 checkpoint hash 可追溯。

## 3.3 实现共同冻结对手池，先只做均匀抽样

本节只验证“跨 seed 共同池”本身，不同时加入 PFSP。

- [ ] 扩展 `OpponentEntry`，支持带来源 hash、策略 seed、训练步数、generation、
  role 和规则版本的外部冻结 checkpoint；不得伪装成自己的 historical snapshot。
- [ ] 对手池从只读 manifest 加载外部 checkpoint，并在启动时一次性完成 schema、
  policy architecture、Observation/action、Catalog/RuleBook 和文件 hash 校验。
- [ ] 一个 generation 内 manifest 不可变；串行训练的后启动 seed 不得看到先完成
  seed 的新权重。
- [ ] 外部对手始终 `eval()`、禁用梯度且不进入 optimizer；只收集 learner 一侧
  transition。
- [ ] 保持每个 episode 独立 recurrent hidden state、policy RNG、先后手分配、
  action 归属和 PPO policy generation 边界。
- [ ] 对手 checkpoint 缓存有确定性 key 和显式内存上限；淘汰缓存不能改变 RNG
  或 opponent selection 序列。
- [ ] checkpoint/resume 保存共同池 manifest hash、选择计数和下一次选择所需
  状态；中断续训后的选择序列与不中断运行一致。
- [ ] 共同池先使用 uniform 权重，构造 Generation 0：
  - [ ] 六个新规则下的 1M final checkpoint；
  - [ ] 各新 seed 具有代表性的 250k/500k/750k 自身历史，按 3.2 结果去重；
  - [ ] 三个旧 3M checkpoint 仅作为带旧规则标签的冻结锚点；
  - [ ] random/fixed 仅作评估锚点，正式 multiprocess PPO 默认权重为 0。
- [ ] 对池内每个模型至少命中一次的固定 seed 测试覆盖双方位置和 7x7 调度。
- [ ] 运行 10k 和 100k agent-step smoke，记录吞吐、显存、缓存命中、模型切换
  时间、选择分布、episode 长度和异常。

建议修改文件：

- `swb/rl/opponents.py`
- `swb/rl/ppo.py`
- `swb/rl/vector_rollout.py`
- `scripts/train_ppo.py`
- `tests/test_opponents.py`
- `tests/test_ppo.py`
- `tests/test_train_ppo.py`

产物：

- `data/reports/league_training/generation_000_manifest.json`
- `data/reports/league_training/uniform_pool_smoke.json`

决策门 3.3：

- 非法动作或 mask mismatch 必须为 0；resume、固定 seed 选择和 trajectory 合同
  必须逐项等价。
- 100k smoke 的截断率不得比冻结基线高 1 个百分点以上。
- median agent steps/s 不得低于同配置自有历史池基线的 90%；若低于，先优化
  checkpoint 缓存/推理合批，再开始三 seed 学习实验。

## 3.4 实现 payoff-aware PFSP

- [ ] PFSP 只读取训练专用 evaluator 产生的 payoff 快照，不读取最终评估 seed。
- [ ] 第一版预注册三种 sampler，按顺序单变量比较：
  - [ ] `uniform`：共同池均匀抽取，作为 3.3 基线；
  - [ ] `variance`：权重与 `p * (1 - p)` 成正比，优先约 50% 胜率对手；
  - [ ] `hard`：权重与 `(1 - p)^alpha` 成正比，`alpha` 首轮固定为 1。
- [ ] 所有 sampler 使用 epsilon floor，避免永久遗忘低权重对手；首轮固定
  `epsilon=0.02`，不得在看到最终结果后修改。
- [ ] 单个对手的采样概率默认封顶 35%；只有候选不足时才能超过，并在报告中
  显式记录。
- [ ] 增加 forgotten-opponent 标记：先前胜率不低于 70%，新快照降到 40% 以下
  时，进入独立的遗忘优先队列。
- [ ] payoff 更新只发生在 generation 边界；同一 PPO rollout 和同一 generation
  内不得动态改变分布。
- [ ] 保存每次分布计算的输入矩阵、公式、参数、归一化前后权重和选择计数。
- [ ] 合成矩阵测试覆盖全胜、全败、全平、缺失 CI、重复模型、只有一个候选、
  epsilon floor、概率上限和固定 seed 重现。

产物：

- `data/reports/league_training/pfsp_sampler_scan.json`
- `data/reports/league_training/generation_001_manifest.json`
- `tests/test_pfsp_sampling.py`

决策门 3.4：

- 先用 3 个训练 seed、每配置 100k steps 做实现筛查；吞吐与语义不通过的配置
  立即拒绝。
- 进入 500k 学习实验的 sampler 最多两个，必须是 uniform 和一个稳定胜出的
  payoff-aware 配置，避免无归因能力的超参数笛卡尔积。
- 500k 后只有满足以下任一主条件且无安全退化才进入 1M 确认：
  - 冻结锚点池配对平均胜率相对独立自博弈基线提高至少 3 个百分点；或
  - worst-case/meta-game exploitability proxy 相对改善至少 10%。
- 同时要求至少 2/3 seed 的主指标不退化，aggregate paired bootstrap CI 下界
  不低于 -2 个百分点，且任何职业格子不下降超过 5 个百分点。

## 3.5 有条件地加入 Main / Exploiter 角色

只有 3.1 检出循环/系统盲点，或 3.4 的 PFSP 在 500k–1M 明显平台化时才实施。

- [ ] 保存需要 exploiter 的证据：显著三循环、低 worst-case payoff、重复遗忘
  或稳定的人类/战术反例。
- [ ] 先实现角色为 manifest 和采样目标，不复制不同网络结构：
  - [ ] `main`：当前策略＋PFSP 全历史＋forgotten 队列；
  - [ ] `main_exploiter`：只针对指定 main 和它的近期快照；
  - [ ] `league_exploiter`：针对全池低覆盖/高可利用性区域。
- [ ] 单卡串行训练采用分代同步：本代所有角色只看上一代冻结模型；所有角色
  完成或明确失败后才生成下一代，避免训练顺序偏差。
- [ ] exploiter 的成功标准是发现可复现漏洞，不要求成为最高 Elo；不能因其
  综合胜率低就删除。
- [ ] main 的评估同时面对 exploiters、历史 mains、冻结锚点和未参与训练的
  validation policies。
- [ ] 首轮不做 PBT 超参数变异；角色收益确认后，PBT 才能作为独立 C 类实验。

产物：

- `data/reports/league_training/role_ablation.json`
- `data/reports/league_training/generation_002_manifest.json`

决策门 3.5：

- 与纯 PFSP 做三 seed、相同新增 steps 的配对比较。
- 只有 main 对 exploiter 和冻结 validation pool 的最差收益改善，且冻结锚点
  平均胜率非劣（95% CI 下界不低于 -2 个百分点），角色 League 才成为默认。
- 没有触发证据或没有明确收益时，以有证据的不适用/延期关闭，不照搬 AlphaStar。

## 3.6 扩充通用战术回放评估集

- [ ] 从当前 1 个案例扩到至少 50 个经人工复核的决策点，覆盖：
  - [ ] Extra PP 保留与费用规划；
  - [ ] 普通进化/超进化对象选择；
  - [ ] 当回合斩杀、两回合斩杀和防守存活；
  - [ ] 出牌、攻击、融合、取消和结束回合顺序；
  - [ ] 资源保留、爆牌、场面空间和墓场/职业资源；
  - [ ] 七职业和先后手；
  - [ ] 隐藏信息下允许多个合理行动的局面。
- [ ] 每个案例保存规则/checkpoint 无关的语义动作标签、状态 hash、动作前缀、
  preferred set、disfavored set、允许并列的理由和复核人/依据。
- [ ] 不把结果不确定的单一玩家意见标成唯一正确动作；使用多个 preferred
  actions 或先保持 `review_required`。
- [ ] 训练、调参和最终留出案例按完整对局分割，同一对局的相邻局面不得跨集合。
- [ ] 报告 top-1、top-k、preferred mass、preferred-vs-disfavored margin、value
  calibration 和逐类别结果。
- [ ] 所有 recurrent checkpoint 使用完整 teacher-forced 前缀重建 hidden state。

产物：

- `data/tactical_scenarios/`
- `data/reports/tactical_scenarios/suite_manifest.json`
- `data/reports/tactical_scenarios/league_comparison.json`

完成标准：

- 50 个案例全部可确定性重放，状态 hash 和合法动作一致。
- League 候选不得通过降低总体 preferred-action 指标来换取单一对阵胜率。

## 3.7 建立可公开 UI 的人类对局数据合同

本节的数据合同和本地采集可与 3.4–3.6 并行，但人类数据进入正式模型必须等
留出集、隐私和许可门禁完成。

- [ ] 在公开 UI 前写明参与者告知、许可范围、删除请求、保留期和开源数据范围。
- [ ] 原始记录只保存训练必要内容；公开数据移除 IP、账户、自由文本和可识别
  时间戳，玩家 ID 使用不可逆且可轮换的匿名 ID。
- [ ] 每局保存引擎/rule/database/Observation/action/checkpoint hash、双方牌组、
  seed、完整合法动作、玩家动作、结果、截断/断线和客户端版本。
- [ ] 记录 AI 行动时的 policy logits/probabilities/value，但不得把未公开的对手
  手牌泄漏进人类 observation 或训练输入。
- [ ] 区分正常完成、认输、断线、超时、规则错误和 UI 错误；后四类默认不进入
  行为克隆训练。
- [ ] 去重机器人刷局、同一回放重传和明显脚本行为；保留过滤原因审计。
- [ ] 按玩家而非单决策随机分割 train/validation/test，防止同一玩家风格泄漏。
- [ ] 玩家质量只用于分层报告和可选采样，不直接把胜者的每个动作都当成正确。
- [ ] 建立可从 schema 版本迁移、重新验证和删除单个玩家数据的工具与测试。

产物：

- `docs/human_match_data_policy.md`
- `docs/human_match_dataset_schema.md`
- `data/human_matches/raw/`（本地、默认不提交）
- `data/human_matches/processed/`（本地、默认不提交）
- `data/reports/human_matches/dataset_audit.json`

## 3.8 行为克隆预训练与 RL 接续实验

- [ ] 在没有足够人类数据前不启动正式 BC；先预注册最小数据门槛和目标。
- [ ] 首轮门槛建议为至少 500 个正常完成对局、50 个独立玩家、七职业均有数据；
  未达到时只用于评估和错误发现。
- [ ] BC 只学习 human-observable state 下的合法动作分布；所有非法 action logit
  继续被 mask，不使用私有日志字段作为输入。
- [ ] 保存行为频率、职业/先后手/胜负/玩家层级分布和 long-tail action 覆盖。
- [ ] 首轮只比较三个单变量配置：
  - [ ] 当前 RL checkpoint 直接续训；
  - [ ] BC 初始化后使用纯 sparse-reward League RL；
  - [ ] BC 初始化且早期保留逐步衰减的 imitation loss。
- [ ] imitation loss 的权重和衰减日程在训练前固定；不得根据最终测试集调整。
- [ ] 人类留出集报告 NLL、top-k、preferred mass 和 calibration，不只报告 top-1。
- [ ] RL 后同时验证冻结模型池、meta-game、战术案例和人类留出集，防止 BC
  提高模仿但降低获胜能力，或 RL 完全遗忘人类策略。

决策门 3.8：

- BC 模型必须在按玩家切分的留出集上显著优于 RL-only 基线的 NLL，并保持
  零非法动作/mask mismatch。
- 经相同新增 500k steps League RL 后，至少 2/3 seed 的冻结锚点或 meta-game
  主指标改善，且战术回放与人类留出指标不退化，才采用 BC 初始化。
- 数据不足或玩家分布失衡时延期，不用少量数据得出强结论。

## 3.9 三 seed 归因实验与 Scaling 决策门

- [ ] 所有学习比较使用相同起点 checkpoint、三个训练 seed、硬件、总新增
  agent steps、PPO 超参数、7x7 调度和评估 seed。
- [ ] 首轮实验矩阵严格限制为：

| 组 | 唯一变化 | 100k 筛查 | 500k 比较 | 1M 确认 |
|---|---|---:|---:|---:|
| A0 | 当前策略＋自己的历史池 | 3 seed | 3 seed | 仅作最终基线 |
| A1 | 共同冻结池＋uniform | 3 seed | 3 seed | 有收益才继续 |
| A2 | 共同冻结池＋胜出的 PFSP | 3 seed | 3 seed | 通过 3.4 才继续 |
| A3 | Main/Exploiter 角色 | 条件触发 | 条件触发 | 通过 3.5 才继续 |
| B1/B2 | 人类 BC 路线 | 数据门触发 | 3 seed | 通过 3.8 才继续 |

- [ ] 每 100k 保存可恢复 checkpoint，但 generation 对手池只在预注册边界刷新。
- [ ] 每个 checkpoint 对相同冻结 anchor mixture 做低成本筛查；最终候选才做
  980 局/对的确认，避免评估吞掉主要训练预算。
- [ ] 报告学习曲线而非只报告 final，并对性能倒退、entropy collapse、value
  发散、grad norm 异常和职业遗忘标记首次出现的 step。
- [ ] 选择默认方案时使用 population 主指标和稳健区间，不以单个 seed 最强为准。

按当前 154–158 agent steps/s 粗估，单 seed 的 100k/500k/1M 新增训练分别约
11 分钟、54 分钟和 1 小时 48 分；这只是训练本体预算，不包括 980 局确认评估。

Scaling 决策门：

- 只有通过 1M 三 seed 确认的方案才能把新规则下模型继续扩到总计 3M。
- 3M 之后每新增 1M 都重新检查主指标斜率；连续两个 generation 的提升小于
  预注册最小效应（首轮 3 个胜率百分点或 10% exploitability proxy），停止
  纯步数 scaling。
- 如果 League 指标提高但战术/人类规划不提高，优先推进人类数据和 credit
  assignment；如果两者都平台化，再单独评估模型容量或 ReBeL/SoG 类搜索。
- 不在同一实验中同时扩大网络、改 reward、改 PPO 更新语义和加入搜索。

## 3.10 最终验收与下一路线

- [ ] 最终候选至少完成一次 1M+ 连续稳定续训和三 seed 独立重复。
- [ ] 对冻结锚点、未参与训练的 validation policies、全部主要 exploiters、
  7x7 职业矩阵、战术留出集和人类留出集完成统一评估。
- [ ] 报告逐 seed 结果、中位数、IQM、paired bootstrap CI、performance profile、
  worst-case/meta-game 指标和完整失败 run。
- [ ] 验证零非法动作、零 mask mismatch、无异常截断、无 hidden-state 串局、
  无 checkpoint 泄漏、无最终评估 seed 污染和无残留 worker。
- [ ] 保存最终选中策略的训练谱系：起点、每代共同池、payoff 快照、角色、
  sampler、全部 checkpoint hash 和每次决策门结论。
- [ ] 在真实玩家盲测中记录模型版本，不向玩家暴露本局面对的是哪个候选；
  报告胜率之外的主观异常标签和复现 replay。
- [ ] 明确最终结论属于工程可用、社区有挑战性还是更高层级，不以论文中的
  工业级结果替代本项目证据。
- [ ] 生成阶段 3 综合报告，并据证据选择下一条唯一主路线：
  - [ ] 继续 generation League scaling；
  - [ ] 扩大并清洗人类数据；
  - [ ] 单独评估网络容量/长程 credit assignment；
  - [ ] 启动不完全信息搜索可行性研究；
  - [ ] 关闭无收益路线并保存拒绝证据。

产物：

- `docs/rl_strategy_league_scaling_report.md`
- `data/reports/league_training/final_comparison.json`
- `data/reports/league_training/stage_3_completion.json`

## 每个实现切片的固定验证

- [ ] 开始前运行 `git status --short --branch` 并保护无关用户改动。
- [ ] 先增加失败测试或保存基线，再做最小 coherent change。
- [ ] 运行相关聚焦测试。
- [ ] 运行完整测试和编译检查：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

- [ ] 对手选择、rollout、trajectory 或训练行为变化额外运行：

```powershell
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

- [ ] 运行固定 seed 的 opponent-selection、checkpoint/resume、log-probability、
  value、hidden-state、trajectory 和 PPO-generation 等价测试。
- [ ] 运行 `git diff --check`、检查 staged diff 和显式文件列表。
- [ ] 只暂存本切片文件，创建可独立回退的 Git checkpoint；不 amend、rebase、
  reset、checkout、push 或 force-push，除非用户另有明确授权。
