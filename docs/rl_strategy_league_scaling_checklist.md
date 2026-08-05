# PPO 策略能力、League 自博弈与人类数据下一阶段 Checklist

最后更新：2026-08-05

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

- [x] 运行 `git status --short --branch`，记录当前 commit、分支和全部用户改动。
- [x] 冻结 SQLite、RuleBook、Catalog、训练池、八套固定牌组、7x7 实际牌组
  调度、Observation、action schema、policy architecture 和 seed derivation
  的版本/hash。
- [x] 冻结六个新 1M checkpoint 和三个旧 3M 锚点的 SHA-256、训练步数、规则
  版本、Extra PP 语义和来源报告。
- [x] 建立训练 seed、调参 seed、最终评估 seed 三个不重叠集合；最终评估 seed
  不得用于 PFSP 权重更新。
- [x] 把以下项目预注册为主指标：
  - [x] 对冻结锚点池的配对平均胜率及 95% CI；
  - [x] payoff matrix 中 meta-strategy 的最差期望收益/可利用性代理；
  - [x] 7x7 职业格子的最差项和第 10 百分位；
  - [x] 战术回放集的 preferred-action top-1、top-k 和 preference margin；
  - [x] 零非法动作、零 mask mismatch 和截断率。
- [x] 将训练吞吐、GPU/CPU/RAM、episode 长度、entropy、value loss、policy loss、
  clip fraction、KL、explained variance、grad norm 和 opponent selection 分布
  注册为诊断指标，不把其中任何一个单独当作策略变强。
- [x] 保存字节稳定的冻结 manifest，并增加字段完整性和 hash 失效测试。

产物：

- `data/reports/league_training/baseline_manifest.json`
- `data/reports/league_training/evaluation_protocol.json`
- `data/reports/league_training/checkpoint_registry.json`

完成标准：

- 同一工作区重复生成 manifest 字节一致。
- 任一规则、牌库、Observation、action、checkpoint 字节变化都会使旧实验合同
  明确失效，而不是静默继续。

完成记录（2026-08-04）：基线 commit 为 `884249b`，League 分支为
`feature/league-core`；`scripts/report_ppo_league_baseline.py` 生成并校验上述三份
manifest，`tests/test_ppo_league_baseline.py` 覆盖字段完整性、seed 隔离、字节稳定
以及五类关键 hash 的显式失效。用户修改的 `docs/roadmap.md` 保持在工作区且未
纳入本阶段提交。

## 3.1 完成并审计当前 seed payoff matrix

- [x] 等待现有 33-pair 队列自然完成；不得重启、覆盖或混合修改运行中产物。
- [x] 验证 33/33 组、6468/6468 局均存在，且无缺失、重复或错误 checkpoint。
- [x] 验证全部对局非法动作和 action-mask mismatch 为 0；逐组报告截断。
- [x] 记录每组胜率、Wilson 95% CI、双方位置胜率、相对 Elo、平均局长和 7x7
  职业格子。
- [x] 明确把旧 3M 对新 1M 标为跨规则历史锚点比较，不能解释成纯 steps ablation。
- [x] 对 95% CI 覆盖 50% 的接近模型不强行排序。
- [x] 对可能影响后续池组成的接近对局补跑到至少 980 局（本轮池组成保留全部
  九个模型，没有接近对局触发补跑；未来移除模型时必须先补足）。
- [x] 生成一个机器可读的 antisymmetric payoff matrix；验证交换行列时符号相反。
- [x] 检测三策略循环：若 A>B、B>C、C>A 的每条边均超过预注册阈值（首轮
  使用 55%），保存循环和各边 CI，不用单一 Elo 抹平该关系。

产物：

- `data/reports/league_training/seed_payoff_matrix.json`
- `data/reports/league_training/seed_payoff_matrix.md`
- `data/reports/league_training/close_pair_confirmations/`

决策门 3.1：

- 矩阵未完整、存在非法动作/mask mismatch 或 checkpoint 来源无法确认时，
  不进入正式 League 训练。
- 未检测到显著循环不代表可以丢弃策略池；仍继续 3.2 检查覆盖和遗忘。

完成记录（2026-08-04）：`scripts/report_ppo_league_seed_matrix.py` 对原始 33 份
report 和 checkpoint registry 逐项交叉校验，确认 6468 局全部正常终止且非法动作、
mask mismatch、截断均为 0；产物保留完整逐组 7x7 格子并生成反对称矩阵。29 组
95% CI 覆盖 50%，均不强行排序。Generation 0 明确保留六个候选和三个锚点，
所以没有任何接近对局会影响池组成，本轮无需额外 980 局；若未来要据此移除模型，
`close_pair_confirmations/plan.json` 要求先补足至少 980 局。点估计有两个循环，
但没有三条边均超过 55% 的预注册强循环。

## 3.2 建立 population 评估与 meta-strategy

- [x] 从 payoff matrix 计算并保存：
  - [x] uniform mixture 的期望收益；
  - [x] 零和 meta-game 的 Nash/线性规划 mixture；
  - [x] 每个 checkpoint 在 mixture 下的期望收益；
  - [x] effective population size 和每个策略的 mixture 权重；
  - [x] 相对当前 population 的 best-response/exploitability proxy；
  - [x] 全局与逐职业的循环克制图。
- [x] 对求解器设置确定性排序和固定容差；重复运行结果字节一致。
- [x] 对奇异、重复、全平和非严格反对称矩阵提供显式诊断，不静默返回权重。
- [x] 使用 paired bootstrap 对 class cell 和 match seed 重采样，报告各 population
  指标的 95% CI。
- [x] 增加性能 profile 和 IQM 汇总，避免均值被一个 seed 或一个职业支配。
- [x] 保存“候选对手为何进入池”的证据：Nash 权重、独特克制边、历史锚点、
  风格/职业覆盖或固定基准；不得只因为它的 Elo 高。

产物：

- `scripts/report_ppo_league_meta_game.py`
- `data/reports/league_training/meta_game.json`
- `data/reports/league_training/meta_game.md`
- `tests/test_ppo_league_meta_game.py`

完成标准：

- 合成石头剪刀布、完全传递、重复策略和不完整矩阵测试全部通过。
- 真实矩阵可重生成，所有输入 report 和 checkpoint hash 可追溯。

完成记录（2026-08-04）：`scripts/report_ppo_league_meta_game.py` 使用固定
`1e-10` 容差的确定性零和 LP/KKT support 与顶点枚举求解六模型完整子矩阵，并用
2000 次固定 seed 的 paired bootstrap 同时重采样 49 个 class cell 和每格两个
deck/match seed。点估计 Nash mixture 为 `20260903=8/19`、`20260906=8/19`、
`20260907=3/19`，其余为 0，effective population size 为 2.635；uniform mixture
的内部可利用性代理为 0.05442。该代理只衡量现有 population 内的 best response，
不是对未知策略的真实 exploitability 上界。九个模型均保留进入 Generation 0 的
可审计理由，旧 3M 只作 anchor，不进入缺少 anchor-vs-anchor 边的 Nash 求解。

## 3.3 实现共同冻结对手池，先只做均匀抽样

本节只验证“跨 seed 共同池”本身，不同时加入 PFSP。

- [x] 扩展 `OpponentEntry`，支持带来源 hash、策略 seed、训练步数、generation、
  role 和规则版本的外部冻结 checkpoint；不得伪装成自己的 historical snapshot。
- [x] 对手池从只读 manifest 加载外部 checkpoint，并在启动时一次性完成 schema、
  policy architecture、Observation/action、Catalog/RuleBook 和文件 hash 校验。
- [x] 一个 generation 内 manifest 不可变；串行训练的后启动 seed 不得看到先完成
  seed 的新权重。
- [x] 外部对手始终 `eval()`、禁用梯度且不进入 optimizer；只收集 learner 一侧
  transition。
- [x] 保持每个 episode 独立 recurrent hidden state、policy RNG、先后手分配、
  action 归属和 PPO policy generation 边界。
- [x] 对手 checkpoint 缓存有确定性 key 和显式内存上限；淘汰缓存不能改变 RNG
  或 opponent selection 序列。
- [x] checkpoint/resume 保存共同池 manifest hash、选择计数和下一次选择所需
  状态；中断续训后的选择序列与不中断运行一致。
- [x] 共同池先使用 uniform 权重，构造 Generation 0：
  - [x] 六个新规则下的 1M final checkpoint；
  - [x] 各新 seed 具有代表性的 250k/500k/750k 自身历史，按 3.2 结果去重；
  - [x] 三个旧 3M checkpoint 仅作为带旧规则标签的冻结锚点；
  - [x] random/fixed 仅作评估锚点，正式 multiprocess PPO 默认权重为 0。
- [x] 对池内每个模型至少命中一次的固定 seed 测试覆盖双方位置和 7x7 调度。
- [x] 运行 10k 和 100k agent-step smoke，记录吞吐、显存、缓存命中、模型切换
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

完成记录（2026-08-04）：共同冻结池以只读 manifest 独立于 learner 自身历史，
启动时校验 manifest/schema、checkpoint SHA-256、模型结构、Observation/action、
Catalog/RuleBook 和完整 versions 合同。外部模型使用确定性 LRU 缓存，始终为
`eval()` 且参数无梯度；optimizer 和 rollout transition 均只属于 learner。
`episode_seed_clustered` 调度保持原始逐 episode 的固定 seed 抽样、先后手与 RNG
语义，只重排实际执行波次以减少模型切换；其 pending wave、选择计数、manifest
identity、policy RNG、recurrent hidden 和缓存统计均进入 checkpoint，显式
`replace_opponent_pool` 才能在 resume 时更换池。

Generation 0 共 27 个不可变条目：六个新规则 1M final、18 个对应的
250k/500k/750k 历史进入训练，三个旧规则 3M 仅作零权重 anchor。固定
`master_seed=20261001` 的 4096 episode 审计命中全部 24 个训练对手、双方位置和
全部 98 个 7x7 职业/位置格，未抽到 anchor。10k smoke 完成 11,120 steps，
125.05 steps/s、147 局、0 截断；100k smoke 完成 100,132 steps，139.63
steps/s、1267 局、平均 79.03 steps/局、0 截断、0 非法动作、0 mask mismatch，
显存峰值 14,116 MiB，缓存 1121 hit/146 miss/139 eviction、180 次模型切换。

吞吐决策门使用同 worker/thread/batch-wait/PPO 语义的 100k 自有历史池配对基线：
共同池 139.63 对基线 89.61 steps/s，比值 1.558，超过 0.90 门槛。六个完整 1M
旧运行的 156.86 steps/s 中位数继续作为保守长期参照（共同池比值 0.890），但
因运行长度和 RTX 4080 power-state 窗口不同，不替代预注册的同运行配置门槛。
汇总、输入文件/checkpoint hash、配置差异、缓存、系统监控和所有 gate 固化在
`data/reports/league_training/uniform_pool_smoke.json`；逐项合同由 generation、
opponent、PPO、checkpoint、vector rollout 和训练 CLI 测试覆盖。

## 3.4 实现 payoff-aware PFSP

阶段状态：**已通过（2026-08-05）**。100k 实现筛查选择 `hard` 作为后续动态
六 lineage League 的默认 sampler；`uniform` 保留为对照，`variance` 本轮淘汰。

- [x] PFSP 只读取训练专用 evaluator 产生的 payoff 快照，不读取最终评估 seed。
- [x] 第一版预注册三种 sampler，按顺序单变量比较：
  - [x] `uniform`：共同池均匀抽取，作为 3.3 基线；
  - [x] `variance`：权重与 `p * (1 - p)` 成正比，优先约 50% 胜率对手；
  - [x] `hard`：权重与 `(1 - p)^alpha` 成正比，`alpha` 首轮固定为 1。
- [x] 所有 sampler 使用 epsilon floor，避免永久遗忘低权重对手；首轮固定
  `epsilon=0.02`，不得在看到最终结果后修改。
- [x] 单个对手的采样概率默认封顶 35%；只有候选不足时才能超过，并在报告中
  显式记录。
- [x] 增加 forgotten-opponent 标记：先前胜率不低于 70%，新快照降到 40% 以下
  时，进入独立的遗忘优先队列。
- [x] payoff 更新只发生在 generation 边界；同一 PPO rollout 和同一 generation
  内不得动态改变分布。
- [x] 保存每次分布计算的输入矩阵、公式、参数、归一化前后权重和选择计数。
- [x] 合成矩阵测试覆盖全胜、全败、全平、缺失 CI、重复模型、只有一个候选、
  epsilon floor、概率上限和固定 seed 重现。

产物：

- `data/reports/league_training/pfsp_sampler_scan.json`
- `data/reports/league_training/sampler_screen_20260804/manifests/`
- `data/reports/league_training/sampler_screen_20260804/sampler_screen_result.json`
- `tests/test_pfsp_sampling.py`
- `tests/test_ppo_league_sampler_screen_results.py`

决策门 3.4：

- 先用 3 个训练 seed、每配置 100k steps 做实现筛查；吞吐与语义不通过的配置
  立即拒绝。
- 100k 筛查最多保留 `uniform` 和一个稳定胜出的 payoff-aware 配置，避免无归因
  能力的超参数笛卡尔积。
- 原方案中的“单 focal 静态池继续 500k，再做 1M 确认”已在本次筛查启动前由
  2026-08-04 修订的 3.5–3.10 六 lineage 动态方案取代：`hard` 的长期确认放到
  Generation 0→1 及后续共同进化中完成，不再把 `20260903` 单独对固定旧池训练。
- 100k 筛查只决定下一阶段工程默认值，不证明 `hard` 长期必胜，也不允许只挑
  本轮最强的 `hard/20261102` checkpoint 代替六模型 population 结论。

实现冻结记录（2026-08-04）：`swb/rl/pfsp.py`
实现三个预注册公式、最终概率 `epsilon=0.02` 下界、35% 上界及候选不足的显式
例外；有对局但缺 CI 的行不会影响 payoff-aware 原始权重，只保留 floor，并在
全部原始权重为零时安全退回均匀分布。训练 payoff loader 强制读取
`pfsp_tuning` 分区、拒绝最终评估 seed、绑定上一代 manifest hash，且只允许在
`generation_end` 生成下一代权重。Generation 1 可以保留出生于 Generation 0
的冻结模型，避免把旧模型错误标记成新代模型。遗忘队列按先前 `>=70%`、当前
`<40%` 确定性排序。

首个 evaluator 已在
`data/reports/league_training/generation_000_payoff_evaluation_plan.json` 预注册：
以 `seed_20260903_1m` 为共同 focal（六候选内部总分最高且有非零 Nash 权重），
只用 tuning master seed `20261001`，对 24 个训练对手各跑 196 局完整 7x7/双方
位置，共 4,704 局；期望 0 非法动作、0 mask mismatch、0 截断。可恢复串行队列
为 `scripts/evaluate_ppo_league_generation_000_payoffs.ps1`，结束后才生成真实
payoff snapshot、sampler scan 和三份冻结 sampler manifest。

完成记录（2026-08-05）：冻结队列完成 9 个训练任务（`uniform`、`variance`、
`hard` 各 3 个独立训练 seed，每个从同一 `20260903` 1M parent 新增 100k）和
54 组候选评估；每个 sampler 共 3,528 局，全部 10,584 局为 0 截断、0 非法动作、
0 action-mask mismatch。综合胜率为 `hard=56.207%`、`uniform=53.245%`、
`variance=53.090%`；`hard` 相对 `uniform` 提高 2.962 个百分点，在三个配对训练
seed 中均排名第一，并对六个 active 1M 对手中的五个取得更高胜率。逐局配对中
`hard` 得分更高 606 局、`uniform` 得分更高 502 局、2,420 局结果相同（全体
10,584 局中有 1 局平局，按 0.5 分计入得分率）。`variance` 的24个
候选权重仅分布在 4.056%–4.212%，实际接近 uniform，未显示收益。可重复聚合、
全部输入 hash、逐 seed/对手/职业结果和选择分布保存在
`sampler_screen_20260804/sampler_screen_result.json`；因此 3.4 正式通过并选择
`hard`，下一未完成节点为 3.5。

## 3.5 冻结六活跃 Main 的迭代式 PFSP League 合同

3.3 的共同冻结池和 3.4 的单 focal PFSP 只用于建立 Generation 0 与筛查
sampler，不得把同一批 250k–1M 对手固定不变地直接训练到 3M。长期 scaling
改为六条活跃 lineage 分代共同进步；历史 checkpoint 继续冻结，用于保持策略
多样性和检测遗忘。

- [x] 将新规则下六个 1M checkpoint `20260903`–`20260908` 冻结为
  Generation 0 的六个 active parent；所有六条 lineage 都进入后续结果，禁止
  删除较差 lineage 后只汇报最佳模型。
- [x] 明确定义并分别记录：
  - [x] `active_latest`：六条 lineage 在上一代的最新冻结 checkpoint；
  - [x] `historical_archive`：250k/500k/750k、旧 active parent 和以后被替换的
    历史版本，只读且不继续训练；
  - [x] `evaluation_anchor`：旧规则 3M、random/fixed 和 validation policies，
    默认训练权重为 0。
- [x] 每条 active learner 使用自己的 payoff 行和 PFSP 权重；不得把
  `20260903` 的难度分布直接套给另外五条 lineage。
- [x] 单卡串行执行逻辑同步 generation：Generation `g+1` 的六个 learner 全部
  只读取不可变的 Generation `g` manifest；后启动 lineage 不得看到同代先完成
  checkpoint。
- [x] 初始 cadence 固定为：每 100k 保存可恢复 checkpoint 和安全指标、每新增
  250k 完成一个 generation、每新增 500k 执行一次历史档案遗忘审计。100k
  checkpoint 默认不触发 payoff 重算或对手池刷新。
- [x] generation 内保持 opponent distribution、规则、Observation/action、PPO、
  7x7 调度和 sparse terminal reward 不变；任何更新只发生在 generation barrier。
- [x] 一个 generation 只有在六条 lineage 全部成功完成并通过安全门后才能发布；
  失败任务必须原 checkpoint/resume 重试，不得以剩余五条模型生成下一代。
- [x] 预注册从总计 1M 到最多 3M 的 generation 编号、父 checkpoint hash、每条
  lineage 的训练 seed、步数、调参/最终评估 seed 和预计计算预算。

产物：

- `data/reports/league_training/evolving_league_contract.json`
- `data/reports/league_training/generation_schedule.json`

决策门 3.5：

- 3.4 的 100k sampler 筛查可以继续使用静态 Generation 0，但任何超过该筛查
  的正式 scaling 必须服从本节的 active-parent 刷新合同。
- exact resume、lineage identity、manifest immutability 和 generation barrier
  必须有固定 seed 测试；串行顺序变化不得改变任一 learner 可见的对手集合。
- 未建立六 lineage barrier 和每 learner payoff 合同前，禁止启动到 3M 的长训。

完成记录（2026-08-05）：`evolving_league_contract.json` 和
`generation_schedule.json` 已冻结六条 lineage、G0→G8 的 250k cadence、100k
checkpoint、500k archive 审计、训练/调参/最终评估 seed 分区及全部父 checkpoint
hash。generation runner 只在 barrier 后原子发布下一代 manifest。

## 3.6 建立 Generation 0 六模型 payoff 与双层对手分布

每代 PFSP 的主要比较对象是六个最新 active checkpoint，而不是每次重新评估
全部历史 checkpoint。历史档案通过独立的固定质量通道进入训练，避免评估成本
随 generation 数量无界增长。

- [x] 使用训练专用 tuning seed 建立六个 Generation 0 最终 1M 模型的完整反对称
  payoff matrix；六模型只需 15 个唯一非自对阵，自对阵按 `p=0.5` 记录。
- [x] 复用本次 `20260903` 对另外五个 1M 模型的报告前，逐项校验 checkpoint、
  rules/database/Observation/action hash、7x7、双方位置、196 局和 tuning partition；
  只补跑剩余十个唯一模型对。
- [x] 对每个唯一 A-vs-B 报告同时生成 `p(A,B)` 和 `p(B,A)=1-p(A,B)`；反向
  95% CI 为 `[1-upper, 1-lower]`，并用测试覆盖平局计分和双方位置。
- [x] 在 3.4 胜出 sampler 确定后，为六条 learner 分别生成自己的
  `active_latest` PFSP 权重；缺边、错误 hash 或最终评估 seed 污染必须硬失败。
- [x] 对手选择改为两阶段且首轮固定：
  - [x] 70% episode 从六个 `active_latest` 中按 learner-specific PFSP 抽取；
  - [x] 30% episode 从 `historical_archive` 中按确定性均匀分布抽取；
  - [x] 70/30 只是首轮预注册工程值，不声称最优；只有在 500k 审计边界才能作为
    独立单变量实验调整。
- [x] `epsilon=0.02` 和 35% cap 应用于 `active_latest` 条件分布；archive 不使用
  未刷新的 learner-specific payoff 冒充当前难度。
- [x] 一个训练 manifest 最多包含 32 个有正训练权重的对手：六个最新 active
  必须保留，archive 最多 26 个；其余 checkpoint 只进入可追溯的零权重归档。
- [x] archive 超限时按预注册确定性顺序保留：上一代 active、forgotten 标记、
  历史 hard/variance 高权重、meta-game support 和 lineage/step 多样性；内容相同
  或行为重复的 checkpoint 只保留代表项。
- [x] 保存每条 learner 的 active/archive 质量、条件概率、最终概率、预期命中数、
  实际选择计数和所有输入 report/checkpoint hash。

产物：

- `data/reports/league_training/generation_000_active_payoff_matrix.json`
- `data/reports/league_training/generation_000_lineage_manifests/`
- `data/reports/league_training/archive_selection_report.json`

决策门 3.6：

- 15 个唯一 active pair 全部为 196 局、完整 7x7/双方位置，且非法动作、mask
  mismatch 和截断均为 0。
- 六个 learner manifest 的 active 条件概率与总概率分别精确归一；固定 seed
  选择测试覆盖六个 active、archive、双方位置和全部职业格。
- 两阶段抽样不得改变 learner transition ownership、recurrent hidden、policy
  RNG、log probability、value 或 PPO generation 边界。

完成记录（2026-08-05）：15 个唯一 active pair 共 2,940 局和 108 个
active-vs-history 基线共 21,168 局全部通过 hash、7x7、双方位置和安全校验；六份
Hard learner manifest 均为 70% active + 30% archive、`epsilon=0.02`、35% cap，
固定 4,096 次选择审计覆盖全部 24 个正权重对手。

## 3.7 实现单卡六 lineage 的可恢复 generation runner

- [x] 实现一个串行、可恢复的 generation queue；输入包含 source generation、
  六个 parent checkpoint、六份 learner manifest、每条目标 steps 和输出目录。
- [x] 每条 lineage 从自己的 parent checkpoint 精确续训，保留模型、optimizer、
  scheduler 和 PPO 状态；只通过显式 generation 边界替换对手 manifest。
- [x] 每条 lineage 的 100k checkpoint、generation final、stdout/stderr、状态、
  吞吐和安全指标写入独立目录，禁止文件名或 resume 状态交叉。
- [x] queue 状态明确区分 pending/running/completed/failed；重启只能重跑未完成
  lineage，已完成产物必须先做 hash 和合同校验才能复用。
- [x] 六条 lineage 即使串行运行，也必须绑定同一个 source manifest hash；不得
  因训练顺序、缓存或先完成模型的存在改变 opponent selection。
- [x] generation publish 使用原子 manifest：六条全部验证成功后一次生成下一代；
  发布前任何部分 checkpoint 都不能成为同代其他 learner 的对手。
- [x] 先对六条 lineage 各运行 10k smoke，覆盖不同串行顺序、一次中断/resume、
  cache eviction 和从中间 lineage 重启。
- [x] 记录共享池加载、模型切换、缓存命中、GPU/CPU/RAM、agent steps/s 和 wall
  time；与 3.3 同配置吞吐相比不得低于 90%，否则先优化 runner。

建议修改文件：

- `swb/rl/opponents.py`
- `swb/rl/ppo.py`
- `swb/rl/vector_rollout.py`
- `scripts/train_ppo.py`
- `scripts/train_ppo_league_generation.ps1`
- `tests/test_ppo_league_generation.py`

产物：

- `data/reports/league_training/generation_runner_smoke_forward/generation_runner_smoke.json`
- `data/reports/league_training/generation_queue_schema.json`

完成标准：

- 两种串行顺序产生相同的可见对手合同；中断续训与不中断的选择序列、trajectory、
  log probability、value、hidden state 和 PPO generation 等价。
- 六条 10k smoke 均为零非法动作、零 mask mismatch、零异常截断和零残留 worker。

完成记录（2026-08-05）：六条真实 10k smoke 全部通过，`20260903` 额外执行半程
checkpoint→resume；吞吐 77.9–90.2、平均约 86.8 agent steps/s，高于 3.3 同为
10k 的 optimized 基线 82.4（约 105.3%）。六条非法动作、mask mismatch、异常
截断均为 0，退出后无残留 worker；正序/逆序逐 lineage 均解析到同一不可变 G0
manifest，100k periodic checkpoint 也会按真实 steps 自动选择最新文件续训。

## 3.8 运行首个共同进化 Generation 0 → 1

自动化状态（2026-08-05）：3.4、3.6、3.7 启动前门均已通过；正式 runner 已配置
从 G0 串行执行六条 250k 续训、15 pair payoff、六条 G0 parent validation、综合
population 报告与原子发布。以下运行项只有后台队列实际产出并过门后才能勾选。

- [ ] 只有 3.4 选出 sampler、3.6 六模型 payoff 完整且 3.7 runner 通过后才启动。
- [ ] 六条 lineage 分别从约 1M 增加 250k agent steps，目标成为六个约 1.25M
  Generation 1 checkpoint；总新增训练量约 1.5M agent steps。
- [ ] 每 100k 保存 checkpoint 和低成本安全/冻结 anchor 指标，但不得据此改变
  当代训练分布或提前选出“最佳 seed”。
- [ ] 保存逐 lineage 的 episode return、entropy、KL、clip fraction、explained
  variance、value loss、policy loss、grad norm、回合长度、截断和吞吐学习曲线。
- [ ] 六条全部完成后冻结 Generation 1，并对六个最新模型运行 15 个唯一 pair、
  每 pair 196 局的 tuning payoff 评估，共 2,940 局；不得再次无条件跑完整历史池。
- [ ] 使用 Generation 1 active matrix 生成六份下一代 learner-specific PFSP 权重；
  同时保留 30% archive 通道和全部来源 hash。
- [ ] 使用固定 validation anchors 比较 Generation 1 与各自 Generation 0 parent，
  报告六条 lineage、population worst-case、uniform/Nash mixture 和逐职业变化。

产物：

- `data/reports/league_training/generations/generation_001/population_manifest.json`
- `data/reports/league_training/generations/generation_001/active_payoff.json`
- `data/reports/league_training/generations/generation_001/training_report.json`

决策门 3.8：

- 六条训练和 2,940 局 active 评估均须零非法动作、零 mask mismatch、零异常截断、
  无 hidden 串局、无 checkpoint 泄漏和无残留 worker。
- 不要求每条 lineage 单代都提高，但至少 4/6 对固定 validation anchors 不退化，
  population worst-case 不得下降超过 2 个百分点；否则暂停并分析 sampler、
  archive 比例或 PPO 学习稳定性，不继续自动 scaling。
- 本节只证明第一次动态刷新可用；不得以其中单个最强 lineage 宣称 PFSP 胜出。

## 3.9 以 250k generation 将六模型有门禁地扩展到最多 3M

自动化状态（2026-08-05）：G0=约 1M 到 G8=约 3M 的八代谱系、100k periodic
checkpoint、250k barrier、每代安全/退化/平台门和 500k archive 审计已由
`scripts/run_ppo_league_generations.py` 实现。审计固定先跑 98 局，只有历史曾
`>=70%` 且筛查 `<=45%` 才补到 196 局，确认 `<40%` 后在下一 barrier 提高该
checkpoint 的 archive 优先级。每代报告自动保留六条 lineage，并计算 median、
IQM、paired bootstrap CI、performance profile、population worst-case、Nash/
uniform mixture、职业最差格和 payoff-profile diversity。以下运行项不会因后台
队列“已启动”而提前勾选；只按实际完成 generation 更新。

- [ ] 按固定谱系推进：G0=1.0M、G1=1.25M、G2=1.5M、G3=1.75M、G4=2.0M、
  G5=2.25M、G6=2.5M、G7=2.75M、G8=3.0M；实际 checkpoint 步数以文件记录
  为准，不用标签覆盖真实偏差。
- [ ] 每代严格执行“六条对上一代训练 → barrier → 六个新模型冻结 → 15 pair
  tuning 评估 → 六份新 PFSP 分布 → 发布下一代”，不得跳过 barrier 或复用
  过期 active payoff。
- [ ] 每 100k 只保存 checkpoint、学习曲线和低成本 safety/anchor screen；只有
  250k generation boundary 才刷新 active 对手。
- [ ] 在 1.5M、2.0M、2.5M 和 3.0M 执行历史 archive 审计：先用预注册低成本
  局数筛查全部保留历史对手，接近遗忘阈值的 pair 再补到 196 局确认。
- [ ] archive 审计检测到“此前得分率不低于 70%、当前低于 40%”时，将对应
  checkpoint 标为 forgotten，并只在下一 generation barrier 调整 archive 组成。
- [ ] 每代报告所有六条 lineage，不删除失败、退化或中断 run；汇总中位数、IQM、
  paired bootstrap CI、performance profile、population worst-case、meta-game
  mixture、职业最差格和 checkpoint 间行为多样性。
- [ ] 训练预算单独报告：六条 lineage 从 1M 到 3M 总新增 12M agent steps；
  评估预算区分每代 2,940 局 active matrix、500k archive 审计和最终确认。
- [ ] active/archive 70/30 比例只有在至少完成到 1.5M 且有两次 generation 数据
  后才能作为单变量消融；不得同时修改 sampler、PPO、reward、网络或 generation
  长度。

每代继续门：

- 任一非法动作、mask mismatch、generation 泄漏、异常截断、NaN/Inf、resume
  不等价或 worker 残留立即阻止下一代发布。
- 连续两个 generation 的 population 主指标提升均低于预注册最小效应（冻结
  anchor 平均 3 个百分点或 exploitability proxy 10%），停止纯步数 scaling，
  保存无收益证据，不机械跑满 3M。
- 如果平均指标上升但某 lineage/职业持续两代退化超过 5 个百分点，先执行遗忘
  审计和 archive 修正；修正后仍失败则停止，不以其他 lineage 的提升掩盖。
- 只有整个六模型 population 通过门禁才能进入下一代；不从同代六个模型中挑
  一个最好者替代 population 结论。

产物：

- `data/reports/league_training/generations/generation_00N/population_manifest.json`
- `data/reports/league_training/generations/generation_00N/active_payoff.json`
- `data/reports/league_training/generations/generation_00N/training_report.json`
- `data/reports/league_training/generations/generation_00N/archive_audit.json`（每 500k）

## 3.10 六模型 League 最终验收与后续路线边界

- [ ] 最终停止点无论早于或达到 3M，都冻结六个最新 active checkpoint、完整
  lineage、全部 generation manifest/payoff/sampler/archive 变化和 checkpoint hash。
- [ ] 使用从未参与训练和调参的最终 seed，对六模型 population、Generation 0、
  旧规则 3M anchors 和未参与训练的 validation policies 做统一评估。
- [ ] 六个最终 active 模型完成 15 个唯一 pair 的至少 980 局/对确认，并报告
  uniform/Nash mixture、worst-case、effective population size 和循环克制。
- [ ] 完成完整 7x7、双方位置、逐职业最差格、先后手差异和 95% CI；报告逐
  lineage、中位数、IQM、paired bootstrap CI、performance profile 和全部失败 run。
- [ ] 验证零非法动作、零 mask mismatch、无异常截断、无 hidden-state 串局、
  无 checkpoint/最终 seed 泄漏、无 OOM/死锁和无残留 worker。
- [ ] UI 如需单一部署模型，必须在最终测试前用 validation population 指标预先
  选择 canonical main；最终测试不得从六个模型中事后挑最高胜率者。
- [ ] 明确结论是“静态共同池有效”“动态六模型 League 有效”“继续 scaling
  已平台化”或“方案退化”；单个 seed Elo 不能替代 population 结论。
- [ ] 生成阶段 3 综合报告，并只根据本阶段证据选择下一条主路线。

本轮 3.5–3.10 明确不实现 Main Exploiter、League Exploiter、PBT、人类 BC、
公开 UI 数据合同、50 案例扩建或 ReBeL/SoG 搜索。这些路线不删除，但移动到六模型
动态 PFSP League 通过、平台化或出现稳定漏洞之后的独立 checklist，避免与本轮
共同进化 scaling 混合归因。

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
