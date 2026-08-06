[English](README.md) | [简体中文](README.zh-CN.md)

# SWB RL

> **非官方开源研究项目。** 本项目自行编写的规则引擎、训练代码、测试、
> UI 和文档采用 MIT License。Shadowverse: Worlds Beyond（影之诗：超凡世界）
> 的卡牌名称、文本及其他游戏内容不属于 MIT 授权范围，详见
> [第三方声明](THIRD_PARTY_NOTICES.md)。

SWB RL 是一个面向 Shadowverse: Worlds Beyond（下文简称 SWB）的确定性规则
引擎、PettingZoo/Gym 兼容强化学习环境，以及使用收益感知 PFSP 对手采样的
多随机种子 PPO League 训练平台。

这个公开仓库不是一个只有示例代码的空壳。它包含完整的社区汇总卡牌目录、
SQLite 数据库、结构化规则、测试、对战 UI、League 元数据和继续训练所需的
代码。约 3.2 GiB 的 PyTorch checkpoint 因体积原因放在 GitHub Release 中，
可以用一条命令安装。卡图和音频不随仓库发布。

## 快速开始

推荐使用 Python 3.11+ 和 Node.js 22+。在 PowerShell 中执行：

```powershell
git clone https://github.com/jacklee12312/SWB-RL.git
cd SWB-RL
python -m scripts.bootstrap --install --with-ui
```

如果需要恢复当前公开的 League/PFSP 研究进度，再下载最新续训快照：

```powershell
python -m scripts.bootstrap --release latest
```

Release 中不是只能推理的精简权重，而是完整 schema-v2 checkpoint，包含：

- 六条活跃训练谱系的模型参数和优化器状态；
- 已完成的 agent steps、训练配置和循环状态；
- Python、NumPy、PyTorch 和 CUDA 随机数状态；
- 对手池、谱系 manifest、payoff 数据和历史对手；
- 继续已发布 generation 所需的全部大文件及 SHA-256 校验值。

安装后可以先执行：

```powershell
# 不下载 Release 大文件时的轻量验证
python -m scripts.run_ci_tests --without-release
python -m compileall -q swb scripts tests

# 安装 Release 后执行完整测试
python -m unittest discover -s tests -v

# 运行 100 局确定性随机自博弈
python -m scripts.random_self_play --games 100

# 启动本地人类对 PPO 对战 UI
python -m scripts.run_match_simulator
```

## Clone 下来以后有什么

| 内容 | Git 仓库 | GitHub Release |
| --- | --- | --- |
| Python 规则引擎、RL 环境、训练与评估代码 | 有 | 不需要 |
| React 本地对战 UI | 有 | 不需要 |
| `shadowverse_cards.json` 完整卡牌目录 | 有 | 不需要 |
| `data/cards.sqlite3` 规范化数据库 | 有 | 不需要 |
| 结构化规则、审计报告、测试与 League 元数据 | 有 | 不需要 |
| Generation 1 六个活跃 checkpoint | 无 | 有 |
| 历史对手、初始化模型和续训大文件 | 无 | 有 |
| 卡图和音频 | 无 | 无，可由用户本地提供 |

因此，其他人 clone 后可以直接阅读、修改和测试规则引擎；执行
`scripts.bootstrap --release latest` 后，还可以从目前公开的研究进度继续训练，
无需从零重新跑六个 seed。

## 当前项目状态

### 卡牌与规则

- 数据库保存 826 张卡牌：735 张可收集卡和 91 张生成/不可收集卡。
- 91 张生成卡全部保留，但不会进入初始牌组采样。
- 牌组校验要求恰好 40 张、职业与中立合法，并排除不可收集卡。
- 735 张可收集卡均有结构化审计记录；当前随机训练池包含其中 734 张。
- `10233310`（`帕梅拉的舞蹈`）因一个旧临时属性修改的官方裁定尚未确认，
  暂不进入随机训练牌组，但仍可用于显式审计和历史回放。
- 结构化规则保存在 `data/rules/`，未知或不完整语义会明确暴露，
  不会静默伪装成已实现。

数据库完整不等于所有卡牌语义都已证明正确。完整的能力、条款、区域资源、
战斗、终局、随机性和规则覆盖审计可在 [英文 README](README.md) 及
`data/reports/`、`data/audits/` 中查看。

### RL 接口

当前公开训练合同为：

- 策略架构：`entity_action_v1`；
- observation：`observation-v4.1`；
- action：`action-112-v2`；
- 对局配置：官方七职业、七套牌分布；
- 默认奖励：稀疏终局奖励；
- 对手采样：共享跨 seed 对手池上的 Hard PFSP；
- 每个 checkpoint 都嵌入动作布局、观察结构、卡牌词表、规则和训练池 hash。

如果 checkpoint 与当前引擎合同不兼容，训练会直接拒绝加载，而不是在结构
已经变化时悄悄续训。

### 最新稳定 League generation

Generation 1 是仓库中第一代完整发布的演化 PFSP 群体，于 2026-08-05 通过
全部发布门槛。它包含 `20260903` 到 `20260908` 六条独立谱系：

| Seed | Agent steps | Checkpoint SHA-256 |
| --- | ---: | --- |
| 20260903 | 1,251,592 | `c3b7b689cd9d8ebece200edec37f666488e55dfe4dcaf6f58ca547f67c39a5ae` |
| 20260904 | 1,250,138 | `261dc0daaef9c12cd595aea37ec6f78563c2f802694897d9e964f65f190e3f12` |
| 20260905 | 1,252,604 | `b967adbb6dc1234002d089f8d14aec9bbbfd2a6f3b8ccdfe1940fa82e34792bd` |
| 20260906 | 1,251,765 | `8c3da19d22b8a3618c52963c17c27c7fcb8fa45a0be52ddd2aba7f7236911e4b` |
| 20260907 | 1,251,450 | `37d81b4984f1fddc1db4ca2a35c861059526bc4306db84449b50ffa4eba687c4` |
| 20260908 | 1,251,217 | `ed82c9230e4a62bb20ada59ce6c307f0a1f34575bee46b16fd3e30fbb2f528bc` |

该 generation 的活跃模型评估共 2,940 局，即 15 个不重复配对 × 每组 196 局。
uniform-population 最差分数为 `0.4600`。六条谱系对各自父代验证的平均分为
`0.5238`，最低谱系分为 `0.4949`，全部通过非退化门槛。训练记录中非法动作、
action mask 不一致、NaN/Inf 和截断局均为 0。

机器可读记录位于：
`data/reports/league_training/generations/generation_001/`。
详细研究交接说明见
[当前研究状态](docs/CURRENT_RESEARCH_STATE.md)。

## PFSP / League 是怎么工作的

当前活跃群体有六条独立 seed 谱系。每一代会把六个模型依次增加约 250,000
agent steps。某个模型训练时，不只和自己的旧 checkpoint 对局，也会从共享的
跨 seed 对手池中按历史胜率采样；Hard PFSP 会提高“有学习价值、尚未稳定击败”
的对手权重，同时保留受限的历史档案来减少遗忘。

一个 generation 只有在以下条件全部完成后才算发布：

1. 六条谱系都完成训练；
2. 六个活跃模型完成 15 组两两对局，每组 196 局；
3. 六条谱系分别完成父代验证；
4. 非法动作、mask、NaN/Inf 和截断安全检查通过；
5. 到期的历史遗忘审计通过；
6. 注册的 generation stopping gate 通过。

仅仅生成了六个新 `.pt` 文件不代表新一代已经发布。

安装 Release 后，可以继续基线队列：

```powershell
python -m scripts.run_ppo_league_generations --max-target-generation 8
```

runner 会读取已有的不可变 generation 报告，跳过已经验证的工作，并在存在
partial checkpoint 时恢复。实验 NFSP、DeepNash、奖励预测或 Oracle Guiding
等新算法时，应另开分支并使用独立的报告和 checkpoint 根目录，不要覆盖此基线。

## 本地人机对战 UI

本地 UI 支持：

- 人类对 PPO 对战、换牌和所有当前合法命令/选择动作；
- 双方场面和公共资源展示，只在实时对局中显示人类手牌；
- 固定牌组二维码配置和可选择的 PPO checkpoint；
- 对局动作、完整状态、合法动作和解析时间线的原子化保存；
- 已完成对局中的 AI 手牌、logits、归一化概率、所选动作和 value 估计回看；
- 攻击、伤害、法术、护符、召唤、进化、破坏、治疗和胜负的轻量动画；
- 本地卡图目录，不把图片复制进前端构建产物。

首次安装前端依赖后启动：

```powershell
cd simulator-ui
npm install
cd ..
python -m scripts.run_match_simulator
```

服务默认递归搜索 `data/checkpoints/` 下的 `.pt` 文件。训练历史、调参、初始化
和 preflight 产物不会出现在 UI 模型列表中；API 只能加载启动时建立的模型目录，
不能用外部请求传入任意磁盘路径。

每一步对局会保存到被 Git 忽略的 `data/match_history/`。开始新对局时，未结束的
旧记录会标记为 abandoned，而不会被覆盖。这里可能包含双方隐藏信息，公开或上传
前请先做隐私审查。

启动参数示例：

```powershell
python -m scripts.run_match_simulator `
  --checkpoint path/to/model.pt `
  --checkpoint-directory data/checkpoints `
  --device cuda `
  --frontend-port 3002 `
  --port 8000
```

当前 UI 是本地研究工具。不要在缺少认证、限流、输入校验和生产部署审查的情况下，
把它直接暴露到公网。

## 战术回放评估集

在 UI 中发现的策略问题可以固化为 `data/tactical_scenarios/` 下的确定性案例。
每个案例保存固定牌组、match seed、动作前缀、目标决策状态 hash，以及语义化的
偏好/反偏好动作。标签按卡牌和动作类型描述，不依赖临时 `entity_id`。

对循环网络 checkpoint 评分时，评估器会用该玩家此前真实采取的动作进行
teacher forcing，先恢复目标时刻的隐藏状态，再比较动作概率。例如：

```powershell
python -m scripts.evaluate_tactical_suite `
  --case data/tactical_scenarios/TACT-SE-0001-empty-board-storm.json `
  --checkpoint path/to/first.pt `
  --checkpoint path/to/second.pt `
  --device cuda `
  --output data/reports/tactical_suite/evaluation.json
```

这些标签用于回归检查“某个场面下应该更偏好什么”，不声称某一步必然赢得整局。
格式和提取流程见
[战术案例说明](data/tactical_scenarios/README.md)。

## 确定性与架构边界

规则引擎和 RL 动作编号是分离的：

```text
玩家意图 commands
        ↓
确定性规则解析 resolution
        ↓
状态变化 state + 事实 events
        ↓
RL environment 编码 observation / action mask / reward
```

核心原则包括：

- 引擎自有的 seeded RNG 是游戏随机性的唯一来源；
- 相同 seed、牌组和命令序列必须复现相同结果；
- 非法命令和非法 RL action 不得修改状态；
- 实体使用稳定 `entity_id`，场上位置只是展示槽位；
- `action_mask()` 必须与真实可执行命令一致；
- 对手隐藏手牌内容和牌库顺序不得进入 observation；
- 卡牌特有行为优先写成 `trigger → condition → target → operation` 的结构化规则，
  而不是在共享引擎中堆叠卡牌 ID 判断。

主要代码边界：

| 路径 | 作用 |
| --- | --- |
| `swb/db/` | 数据库 schema、导入与卡牌查询 |
| `swb/engine/state.py` | 对局可变状态、区域与实体 |
| `swb/engine/commands.py` | 规则核心接受的玩家意图 |
| `swb/engine/events.py` | 解析过程中发生的事实 |
| `swb/engine/effects.py` | 通用效果与目标原语 |
| `swb/engine/card_rules.py` | 结构化卡牌规则加载 |
| `swb/engine/abilities.py` | 关键词注册与通用行为 |
| `swb/engine/resolution.py` | 命令校验和确定性解析 |
| `swb/engine/environment.py` | RL observation、action、mask 和 reward |
| `data/rules/` | 可审计的卡牌特有规则 |
| `scripts/` | 数据库、对局、训练、评估和报告入口 |
| `tests/` | 行为合同与回归测试 |

## 数据库

仓库已经提交 `data/cards.sqlite3`，普通使用者不需要先重建数据库。需要从当前
JSON 重建时执行：

```powershell
python -m scripts.build_database
```

需要从上游 SVA 数据刷新时，请先阅读英文 README 中的刷新说明和审计要求。
数据更新可能改变卡牌文本、规则 hash、训练池和 checkpoint 兼容性，不能把它
当成无影响的普通资源更新。

## 开发与验证

完整 Python 测试和编译检查：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

如果修改了合法动作、解析、卡牌、目标、战斗、回合或 observation，还应执行：

```powershell
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

对共享引擎进行大范围修改时，应把随机自博弈提高到 1,000 局。不要声称未实际
执行的命令已经通过。

## 已知局限

- 目前模型已经能完成基本合法出牌和进攻，但尚未证明具备稳定的高水平战术能力；
  长期资源规划、进化规划和跨回合决策仍然偏弱。
- 战术评估集规模还小，需要继续从人工审查的真实回放和确定性合成场面扩充。
- 六谱系群体是可复现研究基线，不是 PFSP 已优于 NFSP、DeepNash、
  Global Reward Prediction 或 Oracle Guiding 的证据。
- 覆盖报告代表现有规则和测试所证明的范围，不等于对未验证边界作保证。
- 卡图和音频不发布；UI 可以读取用户放入 `data/card_images/` 的本地图片。
- 本地对战 UI 尚未按照生产级公网服务进行加固。

## 开源许可与第三方内容

本项目作者自行编写的源代码与文档使用 [MIT License](LICENSE)。MIT 许可并不
自动覆盖第三方材料：

- Shadowverse: Worlds Beyond、商标、卡牌名称、卡牌文本、美术和音频归各自
  权利人所有；
- `shadowverse_cards.json` 与 `data/cards.sqlite3` 是用于互操作、规则引擎和
  回归测试的社区汇总数据；发布它们不代表作者拥有或能够重新许可底层游戏内容；
- Python 与 JavaScript 依赖继续遵循各自许可证；
- 仓库不发布卡图和音频。

如需修正来源、署名或移除内容，请提交 issue。完整声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 参与贡献

欢迎提交规则、测试、UI、训练和文档改进。项目优先保证准确性、确定性和可复现性，
不追求把未验证语义快速标记成“已支持”。提交 PR 前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)，并说明：

- 改变了什么行为；
- 实际运行了哪些测试；
- 是否影响确定性或 checkpoint 兼容性；
- 还保留哪些未支持语义。

安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要在公开 issue 中附带
凭据、可利用部署细节、私人对局记录或其他敏感数据。引用信息见
[CITATION.cff](CITATION.cff)。

## 进一步阅读

- [英文完整 README](README.md)：逐项实现、审计、训练和工具说明；
- [当前研究状态](docs/CURRENT_RESEARCH_STATE.md)：Generation 1、实验合同和续训交接；
- [贡献指南](CONTRIBUTING.md)：开发约定和提交要求；
- [第三方声明](THIRD_PARTY_NOTICES.md)：游戏内容和依赖的授权边界；
- [安全策略](SECURITY.md)：漏洞与敏感信息报告方式。
