# SWB v4.1 第一轮训练速度优化综合报告

## 结论摘要

本轮在冻结规则提交 `fae33c2`、冻结 v4.1 checkpoint 和同一硬件上完成。
可信基线的三次吞吐为 `44.7054 / 44.5781 / 44.7390 agent steps/s`，
median 为 `44.7054`。最终正式采用：

- `A-BATCH-WAIT-001`：central inference batch wait 从 `0.5 ms` 调为
  `1.0 ms`；
- `A-WORKERS-001`：rollout workers 从 `4` 调为 `6`，每 worker 保持
  `2` 个 PyTorch threads；
- `B-BATCHED-LEARNER-001`：一次性编码整个有效 v4.1 minibatch，再按
  recurrent chunk 执行学习前向。

仅 A 类组合的三次 median 为 `64.4729 steps/s`，相对冻结基线提升
`44.2171%`，超过本轮 `25%` 目标。最终采用栈的三次吞吐为
`136.6923 / 132.6622 / 130.8879 steps/s`，median 为 `132.6622`；
相对冻结基线提升 `196.7474%`，相对采用 A-OBS-001 后的同运行配置
对照提升 `106.3092%`。

这些采用结论均来自基线和候选各三次无诊断插桩的端到端运行，不使用
单次短 benchmark、组件 microbenchmark 或 CPU/GPU 利用率曲线代替。

## 冻结实验契约

| 维度 | 冻结值 |
| --- | --- |
| 规则提交 | `fae33c2` |
| checkpoint | `data/checkpoints/training_speed/frozen_v4_1_seed_20260801_500k.pt` |
| checkpoint SHA-256 | `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd` |
| Observation | `v4.1` |
| policy architecture | `entity_action_v1` |
| rollout / sequence | `2048 / 32` |
| minibatch sequences / epochs | `8 / 2` |
| 训练卡组 | `official_qr_evolve_haven_20260727` |
| 对手池 | 7 套冻结 international QR 卡组 |
| 硬件 | Intel Core i7-13700KF、RTX 4080 16 GB、32 GB RAM、Windows 11 |
| 软件 | Python 3.13.5、PyTorch 2.13.0+cu130 |

Worker 数和 batch wait 的 A 类实验只允许改变被测运行配置维度。网络、
Observation、规则、卡组、checkpoint 和训练超参数保持冻结。B 类 learner
优化因严格参数漂移不满足 A 类逐参数等价，按 B 类完成有限数值、NaN/Inf、
最长局、三 seed 小规模学习和固定对阵门禁。

## 三次端到端结果

| 配置 | 三次 agent steps/s | median | 相对冻结基线 |
| --- | --- | ---: | ---: |
| 冻结 v4.1 基线，4 workers / 2 threads / 0.5 ms | 44.7054 / 44.5781 / 44.7390 | 44.7054 | — |
| A 类组合，6 workers / 2 threads / 1.0 ms | 63.6191 / 64.8932 / 64.4729 | 64.4729 | +44.2171% |
| 最终采用栈，加 batched learner | 136.6923 / 132.6622 / 130.8879 | 132.6622 | +196.7474% |

基线 collect/update P95 分别为 `32.2342 / 22.9009 s`。A 类组合将
collect P95 median 降至 `13.4502 s`，但 update 仍为 `22.7265 s`，
因此 learner 成为下一主瓶颈。B-BATCHED-LEARNER-001 采用后，最终三次
collect/update P95 median 为 `14.6308 / 3.5392 s`；它相对同为
6 workers / 2 threads / 1.0 ms 的三次对照 median `64.3026 steps/s`
提升 `106.3092%`，超过对照的 `1.4606%` 三次波动范围。

## Wall time 瓶颈归因

最终采用源码上的同步分段剖析覆盖 `52.1329 s` steady-state pipeline：

| 分段 | wall time | pipeline 占比 |
| --- | ---: | ---: |
| rollout | 41.5701 s | 79.7387% |
| learner update | 10.5628 s | 20.2613% |
| rollout central forward | 31.1179 s | 59.6896% |
| worker-message + batch-formation holes | 8.1004 s | 15.5380% |
| rollout CPU prepare + H2D | 1.1276 s | 2.1630% |
| learner next-minibatch prepare + H2D | 0.2878 s | 0.5520% |

因此最终主瓶颈是同一 PPO generation 内的 rollout central forward，而
不是由低 CPU/GPU 利用率曲线猜测出的抽象“GPU 未吃满”。剩余
`15.5380%` queue holes 没有可独立调度的同 generation CUDA work：
worker 必须先收到当前动作、推进环境并构造下一 Observation 才能发出下一
请求；multiprocessing queue 已经缓存其他 ready requests。将其称为可重叠
收益会违反实际因果边界。CPU prepare/H2D 的上界也低于 materiality gate，
所以 A-OVERLAP-001 正确关闭，保留同步默认路径。

## 候选处置

机器可读候选矩阵覆盖全部 24 项，逐项保存 disposition、证据层级、路径和
SHA-256。

正式采用：

- `A-BATCH-WAIT-001`
- `A-WORKERS-001`
- `B-BATCHED-LEARNER-001`

已实测但无明确收益、数值门失败或被拒绝：

- `A-OBS-001`、`A-NET-001/002/003`：各有三次端到端数据，提升未超过
  对应运行波动；
- `A-FORWARD-001`：native SDPA 数值和 micro gate 失败；
- `A-PADDED-COMPUTE-001`：出现严格参数漂移，转为 B 类路线；
- `B-LEARNER-AMP-001`：float16/bfloat16 增益低于当前运行波动且有参数
  漂移；
- `B-PRECISION-001`：micro 或有限数值门失败。

在实现前因 materiality、布局或重复工作证据不足而关闭：

- `A-IPC-001`、`A-OBS-002`、`A-NET-004`、`A-STATIC-ENC-001`、
  `A-LEARNER-001`、`A-OPTIMIZER-001`、`A-OVERLAP-001`。

延期或被环境阻塞：

- `A-CUDA-GRAPH-001`：当前动态输入/分配前置条件不满足；
- `B-COMPILE-001`：当前 Windows 环境缺少 Triton；
- `C-ASYNC-001`：会改变 PPO on-policy generation 边界，只能作为独立
  算法实验；
- `C-HYPERPARAM-001`、`C-MODEL-001`：只允许在至少三 seed 学习有效性
  与固定对阵门完成后重开。

`A-PROFILE-001` 是冻结基线与分段诊断的前置证据，不作为速度采用项。

## 102,400+ steps 稳定性

最终采用配置在无 central/IPC/learner 诊断插桩下完成 `102,511 agent
steps`、`1,326` 局和 `45` 个连续 PPO updates，耗时 `786.738 s`，
本次长跑吞吐为 `130.299 steps/s`。全部 update 都有正 agent steps，
policy loss、value loss、entropy 和 grad norm 等更新指标均为有限数，
没有 OOM、worker timeout、死锁、零进度 update 或异常退出。

系统监控每 `0.5 s` 采样，共保存 `1,421` 个样本，覆盖 `786.425 s`
（完整长跑的 `99.960%`）。CPU total median 为 `10.5292%`，单核峰值
`96.4%`；系统 RAM 使用峰值 `25,543,933,952 bytes`。RTX 4080 显存峰值
为 `13,779 / 16,376 MiB`，仍有 `2,597 MiB` 余量。

Windows 报告的 pagefile committed usage 从 `13,034,553,344` 增到
`33,310,076,928 bytes`；这个值不能被误写成“没有使用 pagefile”。
关键的实际 paging I/O 计数在全程 `sin=0`、`sout=0`，即没有 page-in
或 page-out。报告同时保留 used/peak 与 I/O 两组原始字段，避免混淆
commit charge 和磁盘分页。

长跑中 `14 / 1,326` 局触及配置的 `256 agent steps` 环境上限，总截断率
`1.0558%`。前半程为 `0.7788%`，后半程为 `1.3158%`，增加 `0.5370`
个百分点，低于预先写入验收器的 `1` 个百分点趋势门，且总率低于 `2%`
绝对门。它们是显式环境上限截断，不是 worker 崩溃或 PPO update 中断；
没有异常截断增长。

长跑仅在内存中更新 trainer，从不写回冻结 checkpoint；运行前后文件大小、
mtime 和 SHA-256 完全相同。正式配置写入机器报告和下面的复现命令，而不是
污染基线 checkpoint。

## 正确性与兼容性

- v4.1 学习输入、`entity_action_v1` 网络输出、hidden state、action mask、
  log probability、value 和 PPO generation 边界保持不变。
- 从最终 learner 采用提交 `a75af5c` 到本报告，`swb/rl` 和
  `swb/engine` 没有后续差异；2.8–2.10 只增加诊断、证据、测试和文档。
- 冻结 checkpoint 可以继续加载和 resume，长跑前后哈希不变。
- A 类固定 seed 轨迹/精确输出门和 B 类数值/学习门均通过。
- 最终采用栈的 100 局 invariant self-play 为 `56:44`，draw、truncation、
  illegal action 和 mask mismatch 均为 `0`；mixed RL match 正常终局。
- 完整回归通过 `2,925` tests（`1` skip），耗时 `464.160s`；
  PettingZoo API test 通过。完整输出保存在
  `data/reports/training_speed/stage_2_10_complete_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests` 通过。

## 复现正式长跑

```powershell
E:\anaconda\python.exe -m scripts.profile_ppo_training `
  --checkpoint data/checkpoints/training_speed/frozen_v4_1_seed_20260801_500k.pt `
  --additional-agent-steps 102400 `
  --exclude-warmup-updates 2 `
  --device cuda `
  --rollout-workers 6 `
  --rollout-worker-threads 2 `
  --central-inference-batch-wait-ms 1.0 `
  --monitor-system `
  --monitor-interval-seconds 0.5 `
  --output data/reports/training_speed/stage_2_10_stability_100k.json
```

该命令记录每个 update、全程系统样本、page-in/page-out、CPU/RAM/GPU、
截断趋势、checkpoint 完整性和实际完成 steps。正式三次吞吐比较仍应使用
无 `--profile-*` 诊断开关的冻结 benchmark 流程，不能用这次稳定性长跑
替代三次对比。

## 仍有的硬瓶颈与下一路线

最终同步 pipeline 的约 `79.74%` wall time 在 rollout，且 central forward
约占完整 pipeline `59.6896%`。现有 A-NET-001/002/003、native SDPA、
CUDA graph 和 overlap 路线均已被三次波动、数值、前置条件或 materiality
证据关闭。下一轮若继续追求大幅提升，应按以下顺序：

1. 在当前最终 shape 上重新确认可用编译后端；只有 Triton/compile 环境
   可用且通过数值门时才重开 `B-COMPILE-001`。
2. 若改变 model shape 或架构，按 `C-MODEL-001` 执行三 seed 学习曲线、
   固定对阵、checkpoint 迁移和 Observation/输出契约评审，不能只报告
   steps/s。
3. 若尝试异步 rollout，必须显式保存 trajectory policy generation、
   最大 update lag 和行为策略 log probability，重新论证 ratio/clip，
   再完成三 seed 和固定对阵；当前同步 PPO generation 边界仍是默认。

本轮没有发现需要隐藏的未支持卡牌语义，也没有以速度优化改变规则引擎行为。
规则/卡牌覆盖限制继续由现有 coverage 和 roadmap 机制显式报告；本报告只对
训练 pipeline 的已测配置负责。

## 证据索引

- 最终机器对比：`data/reports/training_speed/final_comparison.json`
- 冻结三次基线：`data/reports/training_speed/baseline_summary.json`
- A 类组合：`data/reports/training_speed/stage_2_4_b_interactions.json`
- B 类 learner 三次端到端：
  `data/reports/training_speed/stage_2_7_b_batched_learner_001_end_to_end.json`
- B 类三 seed 学习门：
  `data/reports/training_speed/stage_2_7_b_batched_learner_001_learning.json`
- 最终 wall-time 剖析：
  `data/reports/training_speed/stage_2_8_overlap_gate.json`
- 24 项候选统一验收：
  `data/reports/training_speed/stage_2_9_acceptance.json`
- 102,400+ steps 原始稳定性：
  `data/reports/training_speed/stage_2_10_stability_100k.json`
