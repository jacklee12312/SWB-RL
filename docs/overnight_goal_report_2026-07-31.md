# 2026-07-31 夜间目标执行报告

截止时间：2026-07-31 07:00（Asia/Shanghai）

## 结论

阶段 1 已全部完成：八套训练卡组门禁和 735 张可收集卡 / 91 张衍生卡
完整卡池门禁均通过，Bug 台账中的 6 个 P0、2 个 P1 全部修复，未关闭
P0/P1/P2/P3 均为 0。735 张卡全部保留审计能力；新采样训练 Catalog 暂为
734 张，因为《帕梅拉的舞蹈》（10233310）涉及一个没有直接官方裁定的低频
SET_STATS/临时修正到期交互，已显式排除而不是假装确定。

阶段 2 已完成 2.0、2.1，以及 2.2 worker 侧前三项计时：

- command/resolution；
- legal command/action mask；
- Observation v4.1 构造。

目标尚未全部完成。第一个未完成项是 2.2 的 IPC 请求序列化、发送和等待时间；
2.3–2.11 尚未开始。未在截止前仓促实现 IPC 计时，因为
`multiprocessing.Queue.put()` 使用后台 feeder，单独测量 `put()` 返回时间
不能代表真实序列化和发送成本。

## SWB-CARD-0008 复盘

### 原错误

固定 seed 120445 的随机自博弈发现：较早的 `+2/-2` 修正错误残留在后续
“生命设为 1”的内部基值下，超进化得到攻击 8、当前生命 4、生命上限 2，
触发状态不变量异常。第一次修复只消除了异常，却错误地把预期写成 8/2；
这是依据内部模型推断规则、没有先查官方 Q&A 造成的裁定错误。

### 官方证据与正确结论

- Cygames《雪人觉醒》官方卡牌 Q&A：
  <https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10132320>
  明确说明，被其变为 4/1 的随从普通进化后为 6/3，超进化后为 7/4。
- Cygames 官方帮助/用语集：
  <https://shadowverse-wb.com/ja/help/?tab=tab0>
  说明普通进化为 +2/+2，超进化为 +3/+3，生命增减同时影响当前生命和
  生命上限。
- 《涸绝的显现·吉尔内莉莎》官方卡牌页：
  <https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10304120>
  支持真实卡牌链的前置 `+2/-2` 效果。

因此真实链为：美杜莎 3/7 → 吉尔内莉莎 5/5 → 雪人觉醒 5/1 →
超进化 8/4，当前生命和生命上限均为 4。

### 实现与验证

提交 `b6f1d95` 在通用 `Unit.set_stats()` 层实现按维度覆盖旧修正的语义，
没有添加 card ID 特判。原始 107 步失败复现、缩减后的 86 步可移植复现、
合成最小 fixture、修复后真实回放和相关普通/超进化回归均已保存。

关键证据：

- `data/audits/card_ruling_reviews.json`
- `data/reports/card_bug_audit/repros/SWB-CARD-0008.json`
- `data/reports/card_bug_audit/repros/SWB-CARD-0008-synthetic.json`
- `data/reports/card_bug_audit/reproductions/SWB-CARD-0008-random-self-play.json`
- `data/reports/card_bug_audit/reproductions/SWB-CARD-0008-postfix-official.json`

最终规则下 1,000 局共享引擎门禁胜场 `[521, 479]`，平局、截断、非法动作和
mask mismatch 均为 0；完整测试在该修复点为 2,812 项通过、1 项条件跳过。

## 裁定问题与官方来源优先流程

本次共固化两个裁定条目：

| 裁定 | 状态 | 官方资料与结论 |
|---|---|---|
| `SWB-RULING-0008-A` | `confirmed` | 《雪人觉醒》有直接官方 Q&A；真实三卡链由该 Q&A、官方进化词条和卡牌结算顺序组合，结论为 8/4。 |
| `SWB-RULING-SET-STATS-TEMP-001` | `ruling_uncertain` | 检查《雪人觉醒》《妃花》《帕梅拉的舞蹈》卡牌页、官方帮助及相关 Q&A 后，未找到“较早临时修正→设为特定值→旧修正到期”的直接或足够类似裁定。 |

第二项采用的暂定解释是：设为特定数值覆盖此前同一维度修正，旧临时修正到期
不再反向改变该维度；设定后新增的修正仍正常生效和到期。其他可能解释、采用
理由、影响、查询词、访问日期、官方 URL 和 follow-up 均记录在
`data/audits/card_ruling_reviews.json`。在获得版本化客户端复现或官方答复前，
它不会被描述为官方结论，也不会仅因测试通过而关闭；10233310 继续从新采样
训练 Catalog 排除。

checklist 现已强制要求按以下顺序查证：官方卡牌页/Q&A → 官方帮助和规则 →
官方公告/勘误 → 官方其他语言页面 → 可重复客户端证据。没有直接证据时必须
记录为 `ruling_uncertain`，保存替代解释和影响，不能根据引擎行为反推规则。

## 卡牌 Bug 与完整门禁

Bug 台账最终为 8/8 fixed：

| Bug | 严重度 | 修复内容 |
|---|---:|---|
| `SWB-CARD-0001` | P0 | 结晶本体可支付时不再同时暴露结晶模式。 |
| `SWB-CARD-0002` | P0 | 达到爆能强化费用时不再允许绕过强制增强。 |
| `SWB-CARD-0003` | P0 | 必杀在 0 战斗伤害/屏障防伤后仍按官方语义触发。 |
| `SWB-CARD-0004` | P0 | 在线 history API 不再泄漏 AI 手牌和私密日志。 |
| `SWB-CARD-0005` | P1 | 已结构化谢幕曲不再误报 placeholder。 |
| `SWB-CARD-0006` | P1 | 结构化关键词来源不再产生低频假阳性。 |
| `SWB-CARD-0007` | P0 | 谢幕曲致死在恢复父效果选择前完成状态检查。 |
| `SWB-CARD-0008` | P0 | SET_STATS、旧修正与进化生命增量按官方 Q&A 对齐。 |

最终完整卡池门禁 9/9 通过：

- 735/735 可收集卡逐 clause 审计；
- 91/91 衍生卡入口、生产者和行为审计；
- 10,000/10,000 局、98 个分层、909,158 次 mask 检查；
- 735 张可收集卡进入分层牌组，824 张卡实际出现；
- 98 次固定 seed 重放全部一致；
- 异常、非法动作、placeholder、mask mismatch、截断和重放失败均为 0；
- P0/P1 未关闭数均为 0。

最终报告：

- `docs/card_bug_audit_report.md`
- `data/reports/card_bug_audit/final_gate.json`
- `data/reports/card_bug_audit/full_pool_sampling_10000.json`

## 训练速度进度

### 冻结基线

规则冻结提交为 `fae33c2`，固定 checkpoint、Observation、网络、八套卡组、
seed、4 worker、每 worker 2 线程、rollout 2048、sequence length 32、
2 epoch 和硬件环境。两种 Observation 各完成三次 100,000+ steady-state
agent steps：

| 输入 | 三次 steps/s | median | 范围 | collect P95 | update P95 |
|---|---|---:|---:|---:|---:|
| v4.1 | 44.705 / 44.578 / 44.739 | 44.705 | 0.161 | 32.234s | 22.901s |
| v3.6 | 130.276 / 129.801 / 129.404 | 129.801 | 0.871 | 9.961s | 8.765s |

v3.6 与 v4.1 只分别作为自身优化前后的基线，不把不同 Observation/模型之间的
差异误称为实现回归。六次测量前后 checkpoint SHA、大小和 mtime 均未变化。

基线证据：

- `data/reports/training_speed/baseline_configuration.json`
- `data/reports/training_speed/baseline_summary.json`
- `data/reports/training_speed/baseline_run_v4_1_1.json` 至 `_3.json`
- `data/reports/training_speed/baseline_run_v3_6_1.json` 至 `_3.json`

### 2.2 已完成的真实测量

三次独立 2,048-step smoke 都实际完成 2,282 steps，使用冻结 v4.1
checkpoint，且 checkpoint 未变化：

- command decode 约 0.014 秒，resolution 约 1.295 秒；
- action mask/legal command 0.137650 秒；
- Observation v4.1 构造 3.955569 秒：
  - 决策前 0.154314 秒；
  - `env.step()` 返回值 3.799565 秒；
  - bootstrap 0.001690 秒。

这表明 step-return 的首次 Observation 构造占该类总成本约 96%，是后续
A 类“消除同一 state version 重复转换”的明确候选；目前只测量，没有改变
轨迹、log probability、value、hidden state 或 PPO generation 边界。

机器可读证据：

- `data/reports/training_speed/stage_2_2_command_resolution_smoke.json`
- `data/reports/training_speed/stage_2_2_action_mask_smoke.json`
- `data/reports/training_speed/stage_2_2_observation_v4_1_smoke.json`

当前代码最终验证为：

- `python -m unittest discover -s tests -v`：运行 2,848 项，1 项条件跳过、
  其余通过，438.899 秒，API test 通过；
- `python -m compileall -q swb scripts tests`：通过；
- `python -m scripts.random_self_play --games 100`：`wins=[56, 44]`，
  draw/truncation/mask mismatch 均为 0；
- `python -m scripts.rl_mixed_match --output data/rl_mixed_match.log`：
  player 2 获胜，最终生命 0:18。

## Checkpoint commits

卡牌审计阶段：

- `0a3063b` 1.0；`71426f0` 1.1；`387f0d6` 1.2；`e08d120` 1.3；
- `f895051` / `0b8b397` 1.4；`6ecdae0` 1.5；`d109abf` 1.6；
- `40d6907` 1.7；`fbd263a` 1.8；`60d1c2f` / `d83751e` 1.9；
- `c6ef052` / `82bd251` / `8ef3500` 1.10；
- `bb5635b` / `1bbab63` 1.12；
- `b6f1d95` 官方裁定修正；`063c527` 1.13；`4bf72c2` 1.14；
- `9699ab9` 不确定裁定 Catalog 隔离；`fae33c2` 1.15。

性能阶段：

- `db23a12` 2.0 候选 A/B/C 边界；
- `a638943` 2.1 冻结基线；
- `ef35f02` command/resolution 计时；
- `d32e2bb` action mask/legal command 计时；
- `41e0e8d` Observation v4.1 构造计时。

未执行 push。用户原有 `docs/roadmap.md` 改动始终保留在工作区，未进入上述
checkpoint。

## 未完成项与下一切片

第一个未完成 checkbox 是：

> 2.2 Worker 侧：单独统计 IPC 请求序列化、发送和等待时间。

下一切片应先建立可验证的 IPC 计时边界。不能用 `Queue.put()` 调用耗时冒充
真实发送成本；需要区分 payload 构造/序列化、进入 pipe、中央端取出、响应
发送和 worker 阻塞等待，并为字段求和、空样本以及 request
episode/player/step/generation 身份不变增加测试。完成聚焦测试、完整 2,848+
测试、compileall 和规定 smoke 后，再独立提交 checkpoint。

已知限制：

- `SWB-RULING-SET-STATS-TEMP-001` 仍待官方答复或客户端复现；
- 新采样训练 Catalog 因此为 734/735；
- 2.2 尚未达到互斥阶段解释 90% wall time 的总体验收；
- 尚未实施 A 类速度优化，也尚无优化后三次同配置端到端对比；
- B/C 类方案尚未开始，不能据当前短 profile 推断学习有效性或最终收益。
