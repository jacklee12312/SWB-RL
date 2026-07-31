# Observation v4.1 训练速度瓶颈报告

日期：2026-07-31

Checklist：2.3
诊断候选：`A-PROFILE-002`

## 结论

Observation v4.1 当前最值得处理的共同根因不是 card embedding 查表、GRU、
输入 H2D 或合法动作数量，而是结构化 token 热路径中的大量小算子、重复静态
构造、host-visible 校验和 kernel launch gap。其次是中央推理合批不足：真实
2.2 运行的平均 batch 只有 1.651，58.719% 容量槽为空，而固定输入从 batch 1
增加到 64 时单次 v4.1 前向延迟基本保持在 21--22 ms。

本报告只确定后续优化顺序，不声称已经取得端到端训练加速。任何采用的优化仍
须按照 2.9 使用同配置三次正式训练测量，并满足相应 A/B/C 类语义门禁。

## 范围与方法

- v4.1 是唯一优化目标。v3.6 只在相同固定 seed、相同 batch 和相同计时
  口径下提供纯前向参照；没有重做其 Worker、IPC、中央调度、Learner 或
  对局生命周期剖析。
- 两个模型均加载冻结 checkpoint，测试前后 SHA-256、大小和 mtime 不变：
  - v4.1：
    `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd`
  - v3.6：
    `5ab6466d59f1f762e84e973dafde4130f374a97aa0e3b2e3825ccdb576844c59`
- 固定合成输入按模型合同生成 shape/dtype 有效的 observation、card index、
  hidden state 和 action mask。最大 batch 只生成一次，小 batch 使用相同
  前缀；fixture seed 为 `20260801`。
- 每个 batch 预热 8 次，正式段每次执行 20 个 forward，独立重复 3 次。
  CUDA event 和 host wall time同时保存；表格使用 device median/P95。
- 合成输入用于隔离 shape、batch 和模型执行成本，不代表真实游戏状态分布、
  学习质量或端到端训练收益。
- 组件 hooks 不替换模型执行图，但事件 hooks 本身增加调度成本。因此组件
  数据用于阶段占比和排序，纯前向绝对延迟以无 hooks 的 batch 扫描为准。

硬件和软件：

- GPU：NVIDIA GeForce RTX 4080
- PyTorch：2.10.0+cu130
- CUDA runtime：13.0
- 平台：Windows 11

## 固定输入纯前向

| Batch | v4.1 median ms | v4.1 P95 ms | v4.1 samples/s | v3.6 median ms | v3.6 samples/s | v4.1/v3.6 延迟 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.270 | 21.310 | 47.0 | 5.861 | 170.6 | 3.63x |
| 4 | 22.013 | 22.050 | 181.7 | 5.794 | 690.4 | 3.80x |
| 8 | 21.865 | 21.950 | 365.9 | 5.733 | 1,395.4 | 3.81x |
| 16 | 21.825 | 22.070 | 733.1 | 5.823 | 2,747.8 | 3.75x |
| 32 | 22.060 | 22.210 | 1,450.6 | 5.852 | 5,468.4 | 3.77x |
| 64 | 21.613 | 22.207 | 2,961.2 | 5.829 | 10,979.5 | 3.71x |

v4.1 的单 batch 延迟没有随 batch 线性增加，batch 4 的样本吞吐是 batch 1
的 3.86 倍。结合真实运行 58.719% 空槽，后续 batch wait/worker 扫描有明确
价值。不过固定输入不能证明真实请求一定能填满 batch，也不能替代 2.4 的
端到端调度实验。

在所有 batch 上，v4.1 固定输入前向延迟约为 v3.6 的 3.63--3.81 倍。
这确认额外成本位于 v4.1 模型路径，但 v3.6 不是后续优化对象。

## v4.1 组件分解

以下是 batch 4 的三重复 median。带 hooks 的完整 forward 为 31.062 ms，
比无 hooks 的 22.013 ms 慢，因此只解释组件结构。

| 组件 | Median ms | 说明 |
|---|---:|---|
| 93-token 结构化输入构造总段 | 20.550 | 完整前向的 66.16% |
| card embedding lookup | 0.090 | 不是主要成本 |
| card projection | 0.089 | 不是主要成本 |
| 非卡 Linear projection 合计 | 约 3.0 | 多个小 projection |
| 其他输入 embedding 合计 | 约 3.7 | 类别、语义 byte/kind 等 |
| token tensor op/launch residual | 约 14.8 | clone、round、clamp、gather、reduce、算术、launch gap 等 |
| Transformer | 4.165 | 明显小于 token 构造 |
| Transformer → GRU | 约 0.05 | 可忽略 |
| GRU | 0.143 | 不是主要成本 |
| action/value 完整阶段 | 6.214 | 包含 source/target gather 和 action feature |
| policy head 容器 | 0.199 | action/value 阶段的小部分 |
| value head 容器 | 0.294 | action/value 阶段的小部分 |

该占比与 2.2 真实 PPO 中“Transformer 前输入编码占中央前向 68.4%”相互
印证。真正值得拆解的是 token 构造剩余的 tensor op、静态张量创建、小
embedding/projection 和同步/launch 成本，而不是只优化单次 card embedding。

## CPU 打包、复制和 H2D

每个 v4.1 请求为 73,460 bytes。中央路径每 batch 分别 `np.stack`
observation、card indices 和 action mask，共三次分配/复制。

| Batch | 输入 bytes | NumPy stack ms | CPU tensor ms | H2D ms |
|---:|---:|---:|---:|---:|
| 1 | 73,460 | 0.020 | 0.009 | 0.105 |
| 64 | 4,701,440 | 1.054 | 0.013 | 0.429 |

batch 1 的打包加 H2D 远小于约 21.27 ms 模型前向，不是当前首要瓶颈。
batch 64 时 NumPy stack 上升到约 1.05 ms，说明扩大合批后应同步评估
预分配/复用缓冲区，避免把模型收益转移成 CPU copy 成本。

## GRU episode 长度

固定 batch 4、总计 512 recurrent steps，并在每个合成 episode 边界重置
hidden state：

| Episode 长度 | Device ms/recurrent step |
|---:|---:|
| 1 | 0.1324 |
| 16 | 0.1077 |
| 64 | 0.1032 |
| 256 | 0.1036 |

长度 16--256 的每步成本稳定；长度 1 略高来自频繁 hidden reset/小调用。
真实对局 P50/P95 远大于 1，因此 recurrent state 管理不是当前主要瓶颈。

## 合法动作候选数量

模型先固定计算全部 112 个 action logits，action mask 不进入
`forward_step()`。合法动作数只影响之后的 masked logits、softmax 和
log-softmax：

| 合法动作数 | Distribution device ms |
|---:|---:|
| 1 | 0.2100 |
| 8 | 0.1919 |
| 32 | 0.2195 |
| 64 | 0.2183 |
| 112 | 0.2129 |

变化处于三重复波动范围内。因此当前 dense action scoring 成本不随合法候选
数量变化，不能通过“局面合法动作较少”自动节省前向计算。

## PyTorch Profiler

batch 4 的 3 次 v4.1 forward 捕获：

- 1,938 个 CUDA kernel，即每次 forward 646 个；
- 1,938 个 kernel launch；
- 11 个同步事件，其中 9 个 `cudaStreamSynchronize`；
- kernel gap median 44.13 µs、P95 130.33 µs；
- 最大观测 gap 受 profiler 和同步边界影响，不作为稳定性能结论。

主要设备算子不是单个压倒性的 GEMM，而是大量小 GEMM、copy、add、gather、
round、clamp、reduce、layer norm 和 elementwise kernel。代码审计同时发现
`forward_step()` 对 GPU card-index tensor 执行 Python `bool(any())`，
`masked_logits()` 对 mask 执行 Python `bool(all())`，均要求 host-visible
标量结果。这些校验必须保留语义，但可以在 2.6 评估是否能移出可信热路径或
合并为不会逐 forward 同步的等价门禁。

完整 operator、kernel、gap 和同步事件保存于机器可读报告；压缩 Chrome
trace 可用 PyTorch/Chrome trace viewer 打开。

## 重复工作审计

确认存在：

1. 中央策略 Worker 调用 `env.step()` 时已经构造下一 Observation，但下一
   决策又调用 `env.observation()`，没有复用 `result.observation`。该项留给
   2.5，以轨迹、mask、perspective 和终止边界等价测试为前提。
2. 中央每个 batch 对 observation/card-index/action-mask 执行三次
   `np.stack`，之后再次 H2D。该项留给 2.4 的预分配和缓冲区复用。
3. `_v41_semantic_context()` 每次调用都重新创建 `torch.arange(4)`，适合
   在 2.6 作为注册静态 buffer 的小型 A 类候选验证。
4. device tensor 的 Python `bool` 校验产生 host 同步候选，留给 2.6。

没有发现：

- 同一次 v4.1 forward 内重复执行 card embedding/card projection；
  `card_tokens` 只构造一次并在所有字段间复用。

## 诊断根因排序

1. **v4.1 token 构造及 launch/sync 热路径（2.6）**

   它同时影响中央推理和 Learner forward，是共同根因。先逐项验证静态张量
   buffer、host sync 移出热路径、小 projection/布局等 A 类候选。
2. **中央合批与输入缓冲区（2.4）**

   真实 batch 空槽 58.719%，固定输入显示 batch 4 几乎能取得 4 倍样本
   吞吐。必须通过 batch wait/worker/thread 扫描验证，而不是直接增大等待。
3. **Learner forward/backward（2.7）**

   占 update 约 97.1%，但与第一项共享同一模型根因，不能把潜在收益重复
   相加。优先做不改变 PPO epoch/sequence/minibatch 语义的 A 类工作。
4. **重复下一 Observation 构造（2.5）**

   是明确的 A 类候选，但在当前 pipeline 中预计小于模型/合批机会；复用前
   必须证明 perspective、action mask 和 trajectory 完全一致。

该排序回答“当前成本最可能来自哪里”，不覆盖 checklist 的执行顺序。尤其
不能因为 2.6 排名第一，就跳过 2.4/2.5 已经要求的低风险扫描和已确认重复
工作。

## Checklist 实际执行顺序

1. **2.4A 配置扫描与决策门**

   先用单变量扫描和少量胜出组合确认 batch wait、worker 和线程配置是否有
   超过三次运行波动、且至少 5% 的端到端收益。通过才立即实施 2.4B；未通过
   则保存无明确收益数据并以有证据的不适用/延期关闭 2.4B。
2. **2.5 A-OBS-001 与决策门**

   先消除 `env.step()` 后重复构造下一 Observation 的已确认工作，并验证
   Observation、mask、trajectory、log probability、value、hidden state
   和 generation 边界。只有收益明确或 Observation 仍占 pipeline wall
   time 至少 5%，才扩展预计算、连续写入和缓冲区候选。
3. **2.6 token/launch/sync 共同根因**

   按 host sync、静态 buffer、重复 token 小算子和 kernel launch 的顺序做
   A 类切片；attention、卡牌缓存和 CUDA Graph 各自服从 profiler 证据门。
   `torch.compile` 因 646 次 launch 值得较早测量，但仍按 B 类完成稳定性、
   数值和小规模学习验收。
4. **2.7 Learner 专属工作**

   先继承 2.6 的共同网络收益并重建基线，再区分 padding 准备成本、padded
   compute 和 optimizer/gradient 专属成本，避免把同一模型收益计算两次。
5. **2.8 流水线重叠**

   只在更新后剖析仍显示至少 5% 可调度 wall time 时实施同步重叠；异步
   actor/learner 继续作为 C 类，并要求三 seed 学习与固定对阵证据。

以上执行门槛用于避免为了完成 checkbox 而实现已被数据判定为低价值的优化。
被关闭的候选仍须保存机器可读数据、结论和复现实验配置。

## 阶段 2.4A 单变量配置扫描

冻结 v4.1 checkpoint 上完成 11 个去重配置、每个三次的真实 PPO
端到端比较。每次排除两个 warm-up update，并至少统计三个 steady update
和 6,144 agent steps；诊断 profiling 开关保持关闭。冻结的三次 100k
基线 median 为 44.7054 steps/s，relative range 为 0.3601%。

| 单变量配置 | 三次 median steps/s | 相对冻结基线 | Mean batch | P95 batch |
|---|---:|---:|---:|---:|
| wait 0 ms | 31.1002 | -30.43% | 1.00 | 1 |
| wait 0.1 ms | 43.6795 | -2.29% | 1.73 | 2 |
| wait 0.25 ms | 43.9441 | -1.70% | 1.73 | 3 |
| wait 0.5 ms / 4 workers / 2 threads | 44.1885 | -1.16% | 1.73 | 3 |
| wait 1.0 ms | 59.3714 | +32.80% | 2.48 | 4 |
| 2 workers | 31.6352 | -29.24% | 1.02 | 1 |
| 3 workers | 38.9961 | -12.77% | 1.38 | 2 |
| 5 workers | 48.8632 | +9.30% | 2.09 | 4 |
| 6 workers | 51.7712 | +15.80% | 2.45 | 5 |
| 1 thread/worker | 44.0137 | -1.55% | 1.73 | 3 |
| 4 threads/worker | 44.1724 | -1.19% | 1.74 | 3 |

0 ms 与 2-worker 配置都把 batch 压回约 1，并把吞吐压回约
31 steps/s；1.0 ms 和更多 workers 都能提高 mean/P95 batch。数据因此把
根因收窄为“等待窗口与请求到达率共同限制合批”，而不是无效等待或
worker CPU 线程不足。1.0 ms、6 workers 和 5 workers 都通过至少 5% 且
超过三次波动的 gate；下一步只组合 wait=1.0 ms 与 worker 维度稳定最佳值，
不做全笛卡尔积。

机器可读汇总：
`data/reports/training_speed/stage_2_4_scan.json`；逐 run 证据：
`data/reports/training_speed/stage_2_4_runs/`；扫描器：
`scripts/scan_training_speed_stage_2_4.py`。

### 稳定 winner 交互

只组合跨维度稳定 winner，没有做全笛卡尔积：

| 组合 | 三次 steps/s | Median | 相对冻结基线 | 相对最强构成项 |
|---|---|---:|---:|---:|
| 5 workers + 1.0 ms | 63.8337 / 63.7021 / 64.3763 | 63.8337 | +42.79% | +7.52% |
| 6 workers + 1.0 ms | 63.6191 / 64.8932 / 64.4729 | 64.4729 | +44.22% | +8.59% |

采用后续候选运行时配置为 6 workers、每 worker 2 threads、1.0 ms wait。
它的 mean/P95 batch 为 3.10/6；三次均有完整系统监控且无异常退出。
机器可读汇总为
`data/reports/training_speed/stage_2_4_b_interactions.json`，逐 run 证据位于
`data/reports/training_speed/stage_2_4_b_interaction_runs/`。

### 2.4B 低上限候选关闭

当前一个决策已经只有一条聚合 request 和一条 action response；request
queue put 为 0.0104 ms/request，而 response wait 为 42.5090 ms/request。
再减消息必须预取尚未由上一动作产生的状态，或让每 worker 同时持有多个
环境，都会改变当前决策/拓扑边界，因此不作为 A 类实现。

采用 batch 桶附近的实测上限如下：

| 候选段 | Batch 4 median ms | 相对 22.0133 ms device forward |
|---|---:|---:|
| 三次 NumPy stack + CPU Tensor 构造 | 0.0431 | 0.196% |
| H2D | 0.1222 | 0.555% |
| 打包 + H2D 理想全部消除 | 0.1652 | 0.751% |

三个值均远低于 5% materiality gate。batch buffer 复用与 pinned/non-blocking
H2D 因此分别以“低于门槛”关闭，避免为小于测量波动的理论收益加入缓冲区
生命周期、锁页内存和异步同步复杂度。机器可读 acceptance 与来源哈希为
`data/reports/training_speed/stage_2_4_acceptance.json`。

阶段 2.4 最终采用 6 workers、2 threads/worker、1.0 ms wait，三次短跑
median 64.4729 steps/s，相对冻结 100k 基线提升 44.22%。完整测试
2,875 项通过（1 skip），compile check 通过，确定性 mixed match 通过。

## 产物与限制

- 机器可读报告：
  `data/reports/training_speed/v4_1_inference_breakdown.json`
- 压缩 PyTorch Profiler trace：
  `data/reports/training_speed/v4_1_profiler_trace.json.gz`
- 生成脚本：
  `scripts/profile_v4_1_inference_breakdown.py`

已知限制：

- 合成 fixture 不覆盖真实卡牌/语义稀疏度分布。
- 组件 hooks 有诊断开销，不能拿 hooked 绝对值宣称吞吐。
- Profiler 只运行 batch 4 的 3 次 forward；它定位 kernel 和同步，不替代
  长时间端到端测量。
- v3.6 仅用于纯前向参照，没有也不需要形成完整组件报告。
- 本阶段没有修改模型、Observation、PPO、规则或训练参数，因而没有产生
  性能优化收益；后续候选仍需独立实现和验收。
