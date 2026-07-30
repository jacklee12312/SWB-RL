# 卡牌 Bug 排查与训练速度优化 Checklist

最后更新：2026-07-29

## 目标与执行顺序

本阶段只包含两个大任务，并严格按顺序推进：

1. 完成可复现、可审计的卡牌 Bug 排查，先关闭当前八套训练卡组的
   规则风险，再扩展到完整卡池。
2. 在规则基线冻结后优化 PPO 训练速度，优先做不改变 PPO 数据含义和
   on-policy 边界的优化。

在任务 1 的 P0/P1 门禁通过以前，不启动新的正式长时间训练。规则 Bug
修复可能改变轨迹分布，因此任务 2 的性能基线必须在最终规则提交上重新测量。

当前事实基线：

- 当前数据库有 735 张可收集卡和 91 张非收集/衍生卡。
- 当前八套固定卡组覆盖七个职业，合计约 111 张不同的可收集卡。
- 当前覆盖报告将 735 张可收集卡列为 `covered_exact`，但该状态不等于
  所有运行时边界和机制组合均已验证。
- 当前完整测试为 2664 项通过、1 项跳过。
- 最近一次 v4.1 三 seed 实验约为 99.84 agent steps/s；该数据来自既有
  实验，不作为优化后的公平基线。

## Checklist 使用规则

- 只有同时具备代码/规则、自动化测试和保存的报告证据时才能勾选任务。
- 每个发现的 Bug 必须先保存最小复现，再修复，再增加永久回归测试。
- “随机对局没有报错”不能代替能力分支覆盖。
- “卡牌有一项测试”不能代替逐条卡牌文字和边界条件审计。
- 不确定规则必须保持显式待确认，不能根据现有引擎行为反推官方规则。
- 性能对比必须使用相同提交、卡库、规则、Observation、网络、checkpoint、
  seed、对阵表和训练步数。

---

# 阶段 1：完整卡牌 Bug 排查

## 1.0 定义严重程度和处理规则

- [x] 固化 P0 定义：非法动作、费用/模式合法性、攻击权限、隐藏信息泄漏、
  胜负、伤害、奖励、动作掩码与命令不一致。
- [x] 固化 P1 定义：训练卡组中常见卡牌的效果、目标、时机、区域移动或
  职业资源结算错误。
- [x] 固化 P2 定义：低频卡牌或罕见组合的错误，当前训练轨迹中出现概率低，
  但仍需登记和回归。
- [x] 固化 P3 定义：只影响 UI、动画建议或文字日志，不改变引擎状态和策略输入。
- [x] 建立 Bug 台账字段：编号、等级、卡牌、机制、发现提交、最小 seed、
  复现文件、预期、实际、受影响卡组、修复提交、回归测试。
- [x] 明确处理规则：P0 立即暂停正式训练；P1 在长训练前清零；P2 可以与
  小规模试验并行；P3 不阻塞训练。

产物：

- `data/reports/card_bug_audit/bug_ledger.json`
- `data/reports/card_bug_audit/bug_ledger.md`

证据（2026-07-29）：

- `scripts/report_card_bug_audit.py` 固化 P0–P3、状态、训练阻塞规则及台账字段，
  并校验最小复现路径、唯一编号、关闭理由、修复提交和永久回归测试。
- `tests/test_card_bug_audit_ledger.py` 的 7 项聚焦测试通过；保存的 JSON/Markdown
  与确定性生成结果逐字节一致。初始台账为空并不代表门禁通过；当前 checkpoint
  已登记后续扫描发现的 2 个 open P0，正式训练保持阻断，修复随 1.4 提交。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2671 项通过，1 项条件跳过。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。

## 1.1 冻结可审计基线

- [x] 记录审计起点 Git commit、分支和工作区状态。
- [x] 记录 SQLite 文件 SHA-256、卡牌数量和 SVA 数据源快照信息。
- [x] 记录 RuleBook、Clause Audit、Token Audit、Ability Audit 和 Catalog 哈希。
- [x] 记录八套固定卡组的名称、职业、40 张实体列表和 deck SHA-256。
- [x] 生成八套卡组的可收集卡并集，确认不同卡牌数量。
- [x] 从卡牌引用、召唤、加入手牌、变身、融合和纹章/信仰定义递归计算
  衍生卡闭包。
- [x] 检查闭包中的每个卡牌 ID 都能在数据库和 RuleBook 中解析。
- [x] 将固定基线写成机器可读文件，并为生成逻辑增加确定性测试。

产物：

- `data/reports/card_bug_audit/baseline.json`
- `data/reports/card_bug_audit/training_deck_card_closure.json`

完成标准：

- 相同提交重复生成得到字节一致的结果。
- 八套卡组与全部递归衍生卡均有唯一、稳定的审计条目。

证据（2026-07-29）：

- `scripts/report_card_bug_audit_baseline.py` 记录审计起点
  `845b3c99257b28e77e52db42d98e2e9de9f61f98`、`main`、上游 ahead 5
  和路径级工作区状态。生成报告自身从状态列表中显式排除，避免输出
  自引用破坏同一工作区的逐字节重生成。
- SQLite SHA-256 为
  `df069e713a97493c885266b72f303874035beea571147ba14b77c57c9e631376`；
  数据库为 826 张（735 可收集、91 非收集/衍生），并保存 SVA URL、
  抓取时间、源 payload SHA-256 和本地 `shadowverse_cards.json` SHA-256。
- `baseline.json` 保存 RuleBook、Clause Audit、Token Audit、Ability Audit、
  Coverage Report、Catalog、card vocabulary 和 training pool 哈希，以及
  八套卡组的名称、职业、40 张 ID、数量表、来源 deck hash 和 deck SHA-256。
- 八套卡组并集为 111 张可收集卡；数据库引用与结构化规则中的加入手牌/
  牌库、召唤、变身、换牌库、融合、纹章和信仰产物递归闭包为 147 张，
  其中 116 张可收集、31 张衍生。每张卡有唯一 `audit_id`、稳定发现路径、
  引用证据、数据库解析结果、RuleBook 查找结果及 collectible/token 审计状态。
- `tests/test_card_bug_audit_baseline.py` 的 8 项聚焦测试通过，包括全部产物
  类型的合成解析测试、闭包完整性、稳定排序和两次生成逐字节一致。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2679 项通过，1 项条件跳过；耗时 396.591 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片未修改引擎、规则、动作、目标、战斗、回合或 Observation，
  因此未触发额外随机自博弈和 `rl_mixed_match` 门禁。

## 1.2 建立逐卡、逐条款审计表

- [x] 为每张卡生成基础字段：ID、名称、职业、类型、所属卡组、数量和引用关系。
- [x] 为每条主卡文字和替代模式文字生成独立 clause 行，不用一个
  `covered_exact` 布尔值代表整张卡。
- [x] 每条 clause 记录对应的结构化触发、条件、目标和操作。
- [x] 每条 clause 记录直接测试、机制测试、官方来源和最后验证提交。
- [x] 每张卡分别记录以下状态：
  - [x] 原文映射
  - [x] 正常路径
  - [x] 费用/阈值边界
  - [x] 替代模式
  - [x] 进入方式
  - [x] 目标与选择
  - [x] 时机与优先级
  - [x] 区域与容量
  - [x] 随机性与确定性
  - [x] command/action mask 一致性
  - [x] 运行时实际覆盖
  - [x] 官方裁定或客户端复现
- [x] 报告必须把“未适用”“已通过”“未测试”“规则不确定”和“已失败”
  分开表示。
- [x] 为报告的稳定排序、字段完整性、断链测试引用和过期 source hash
  增加自动化测试。

产物：

- `data/reports/card_bug_audit/card_clause_matrix.json`
- `data/reports/card_bug_audit/card_clause_matrix.md`

证据（2026-07-29）：

- `scripts/report_card_clause_matrix.py` 从冻结的 147 卡闭包生成 147 个稳定
  卡牌条目和 161 个独立 clause 条目（143 条主文字、18 条替代模式），记录
  卡组数量、递归引用、结构化候选 trigger/condition/target/operation、直接/
  机制测试、官方证据槽位和最后验证提交。
- 145 张卡沿用 Clause Audit 的 source hash 并重新计算验证；`90071150`
  的“守护”文字首次由本矩阵独立冻结；无印刷文字且旧 Token Audit 已确认
  完整的 `90021110` 显式记为 hash 不适用。69 个被引用测试文件全部存在。
- 十二个运行时维度分别使用 `not_applicable`、`passed`、`not_tested`、
  `ruling_uncertain`、`failed`，没有从 `covered_exact` 推导运行时通过。
  当前仅 147/147 原文映射为 `passed`，仍有 1224 个适用运行时行待后续扫描；
  `training_runtime_gate_ready=false`。
- `tests/test_card_clause_matrix.py` 的 8 项聚焦测试覆盖稳定排序、字段完整性、
  主文字/替代模式独立行、状态分离、断链测试引用、过期 source hash、保存
  报告和两次生成逐字节一致。
- 第一次完整门禁发现把 Token hash 写回旧 Clause/Coverage 会迁移 Catalog/
  coverage 哈希并使冻结 checkpoint 不兼容；该方案已撤回，完整数据由矩阵
  自身保存。第二次门禁发现换行字节哈希与保存 baseline 不同步；重生成顺序
  固化为 baseline/closure 后再生成 matrix。两项门禁都按预期阻止了误勾选。
- 最终 `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2687 项通过，1 项条件跳过；耗时 401.121 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片未修改引擎、`data/rules/`、动作、目标、战斗、回合或 Observation，
  因此未触发额外随机自博弈和 `rl_mixed_match` 门禁。

## 1.3 校对规则来源

- [x] 将数据库中的中文、英文、日文主文字及替代模式文本与当前规则条款对齐。
- [x] 检查费用、攻击、生命、类型、职业、特性和基础关键词。
- [x] 检查卡牌引用是否指向正确实体，而不是同名或错误版本。
- [x] 检查每个“若”“每当”“直到”“本回合”“自己的回合”等条件是否有
  明确的取值时刻和持续时间。
- [x] 检查“随机”“全部”“一个”“其他”“不同名称”等数量和去重语义。
- [x] 将规则有歧义的卡加入待确认队列，不按现有实现自行判定。
- [x] 对歧义项按以下证据顺序确认：官方 Q&A/规则说明、客户端可重复复现、
  多个可靠录像或测试、单个攻略来源、纯文字推断。
- [x] 为每个外部裁定保存 URL、访问日期、结论和不超过必要长度的摘要。
- [x] 官方文字或数据库刷新后，确保旧 source hash 自动失效。

产物：

- `data/reports/card_bug_audit/ruling_queue.json`
- `data/reports/card_bug_audit/source_alignment.json`
- `data/reports/card_bug_audit/source_alignment.md`
- `data/audits/` 下相应的裁定和 clause 更新

证据（2026-07-29）：

- `scripts/report_card_source_alignment.py` 对八套卡组的 147 卡递归闭包逐卡
  比较 SQLite 规范化字段与每张卡保留的原始导入记录；147/147 张卡的费用、
  攻击、生命、类型、职业、特性、三语名称和基础关键词来源均通过，161 条
  中文/英文/日文主文字或替代模式文字逐字一致。
- 76 个数据库引用全部同时通过目标 `card_id` 存在性、引用名与目标三语名称
  匹配、以及原始导入 references 一致性检查，没有同名或错误版本引用。
- 对中文条款检出的 17 个条件、2 个持续时间、15 个己方回合、27 个随机、
  41 个全部、109 个单一数量、14 个“其他”和 2 个不同种类/名称语义，
  均在完整结构化规则条目中找到显式标记；“每当”和“本回合”在本闭包
  原文中没有适用项，显式记为 `not_applicable`，未据引擎行为补写裁定。
- `ruling_queue.json` 固化五级证据顺序和外部裁定字段契约。当前源文字到
  结构化规则没有未解决歧义，队列为 0；这只通过 1.3 来源门禁，不替代
  后续官方/客户端复现或 1224 个运行时维度。
- 当前没有外部裁定条目，因此没有可写入 `data/audits/` 的 URL/访问日期；
  生成器会拒绝任何缺少 URL、访问日期、结论或必要摘要却被关闭的裁定项。
- 生成器把数据库、规则目录、baseline、closure 和 matrix 哈希写入两份
  机器可读报告；数据库字节变化会立即拒绝旧矩阵。来源报告 SHA-256 为
  `d0de6a3b203075d837bdbc1302b75da31d5999bbd5b3e46a641f29deadf03b95`，
  裁定队列 SHA-256 为
  `fd157beecc9fe41ab2fd9a3c9ec98f5d2ba8b24d5a4bf6f692788e1c50dd772b`。
- `tests/test_card_source_alignment.py` 的 8 项聚焦测试覆盖原始字段/三语条款、
  基础关键词、版本引用、条件/时机/数量/去重、证据策略、外部裁定必填字段、
  数据库刷新失效和保存报告逐字节重生成。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2695 项通过，1 项条件跳过；耗时 406.809 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片未修改引擎、`data/rules/`、动作、目标、战斗、回合或 Observation，
  因此未触发额外随机自博弈和 `rl_mixed_match` 门禁。

## 1.4 替代模式和费用边界扫描

适用范围：Accelerate/激奏、Crystallize/结晶、Enhance/增强、Mode、
Invocation/瞬念召唤及其他替代出牌方式。

- [x] 自动枚举每张适用卡的剩余 PP：0、替代费用前一档、恰好替代费用、
  本体费用前一档、恰好本体费用和高于本体费用。
- [x] 区分印刷费用、当前费用、剩余 PP 和模式要求，禁止混用。
- [x] 测试手牌临时/永久加费和减费后，模式可用性是否仍按官方规则判断。
- [x] 测试可以打本体时是否错误暴露激奏或结晶动作。
- [x] 测试模式互斥；一次出牌不能同时执行本体和替代模式能力。
- [x] 测试替代模式是否使用正确卡牌类型、墓场去向和职业资源计数。
- [x] 测试无合法目标、场面满和手牌满时的“禁止出牌”或“跳过效果”语义。
- [x] 对每个费用点同时比较 `legal_commands()`、`action_mask()` 和实际执行结果。
- [x] 对非法命令验证状态 fingerprint、RNG 和所有区域均不改变。
- [x] 为此前发现的高 PP 仍能激奏问题保留真实卡牌回归测试。

完成标准：

- 八套训练卡组闭包中所有适用卡通过扫描。
- 全卡池所有适用卡通过相同生成器扫描。
- 零 command/action mask 分歧。

2026-07-30 完成证据：

- 修复提交 `f895051` 统一执行官方替代模式互斥语义：达到增强阈值时强制
  使用最高可支付增强；仅在本体当前费用不可支付时暴露激奏/结晶。修复前
  最小复现保存在
  `data/reports/card_bug_audit/reproductions/SWB-CARD-0001.json` 和
  `SWB-CARD-0002.json`，外部裁定 URL、访问日期、预期与实际结果均随包保存。
- `scripts/report_play_mode_boundary_audit.py` 从当前数据库和规则目录动态枚举
  54 张适用卡、55 个 PP 替代模式，其中训练闭包 17 张；对印刷费用以及
  临时/永久加减费五种场景执行 1,546 个费用边界案例和 55 个满场案例。
  Mode/Invocation 不属于 PP 替代模式，继续由既有专门测试覆盖，报告明确记录
  该范围界线。机器可读与 Markdown 结果分别为
  `data/reports/card_bug_audit/play_mode_boundary_audit.json` 和 `.md`：
  0 command/action-mask 分歧、0 非法操作原子性失败、0 执行失败。
- `tests/test_play_mode_boundary_audit.py` 的 7 项报告契约测试通过；
  受增强语义影响的 518 项真实卡与替代模式聚焦测试通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2,710 项通过，1 项条件跳过；耗时 410.514 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100
  --output data/reports/card_bug_audit/smoke/stage_1_4_random_self_play_100.json
  --validate-invariants`：100 局，0 平局、0 截断、0 mask 分歧，acceptance=pass。
- 同配置 1,000 局广泛共享引擎门禁保存在
  `data/reports/card_bug_audit/smoke/stage_1_4_random_self_play_1000.json`：
  507/493，0 平局、0 截断、0 mask 分歧，acceptance=pass。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：通过。Bug 台账的两个 P0 均已由永久回归测试关闭；
  当前 P0/P1 为 0。

## 1.5 关键词来源和进入方式扫描

- [x] 建立进入方式矩阵：正常打出、替代模式、直接召唤、从牌库召唤、
  复活、复制、变身、普通进化、超进化和效果进化。
- [x] 对疾驰、突进、守护、潜行、屏障、必杀、攻击次数、不可攻击、
  不可指定等关键词逐项记录来源。
- [x] 区分卡牌固有关键词、条件关键词、进化后关键词和临时获得关键词。
- [x] 验证未满足条件时不提前授予关键词。
- [x] 验证通过不同进入方式时是否应触发入场曲，以及是否继承实体修正。
- [x] 验证返回手牌、变身、沉默和离场后应清除或保留的动态能力。
- [x] 验证普通进化、超进化、“进化时”和“本随从进化时”的触发范围。
- [x] 验证疾驰/突进随从的当回合主战者和随从攻击目标。
- [x] 为佐伊未爆能获得疾驰问题保留真实卡牌回归测试。
- [x] 将关键词矩阵同时接入 RL 攻击动作掩码检查。

完成标准：

- 八套训练卡组闭包的所有关键词来源都有明确测试。
- 全卡池所有声明关键词和运行时授予关键词均通过适用矩阵。

2026-07-30 完成证据：

- `scripts/report_keyword_entry_audit.py` 动态扫描当前 826 张数据库卡、
  `RuleBook` 的全部规则根、嵌套操作、纹章和被动定义；识别 321 张存在
  关键词、攻击次数或攻击/指定限制来源的卡，其中 274 张可收集卡、47 张
  衍生卡，八套训练卡组递归闭包中 59 张，另有 6 个纹章级全局来源。
  每张来源卡均关联 `covered_exact` 或衍生卡专用审计状态及永久测试证据。
- 机器可读报告
  `data/reports/card_bug_audit/keyword_entry_audit.json` 和 Markdown 报告
  `.md` 覆盖 9 种运行时随从关键词、8 类来源和 12 种进入/进化路径：
  正常/增强打出、直接/牌库召唤、复活、定义复制、完全复制、变身、普通/
  超进化及两种效果进化。0 来源问题、0 契约失败、0 矩阵失败。
- 对每种运行时关键词实际执行正常/增强打出、牌库召唤、复活、定义复制、
  完全复制和变身；确认只有打出触发入场曲，完全复制继承动态关键词，
  返回手牌、变身和沉默清除对应动态状态。4 个进化契约区分“进化时/
  超进化时”关键词能力与“本随从进化/超进化时”状态触发。
- 14 个 command/action-mask 契约覆盖疾驰、突进、守护、潜行、威慑、
  两次攻击、不可攻击和不可指定；全部 command、RL mask 与预期一致。
- 佐伊真实卡回归继续由
  `tests/test_real_listener_context_leader_runtime_nineteenth_batch.py`
  固化：普通 5 PP 出牌不获得疾驰，增强 10 获得疾驰。官方卡牌页
  `https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10444120`
  于 2026-07-30 复核，疾驰只出现在增强 10 条款。
- `E:\anaconda\python.exe -m unittest tests.test_keyword_entry_audit -v`：
  12 项通过；关键词/区域/进化/真实卡聚焦回归集 299 项通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2,722 项通过，1 项条件跳过；耗时 412.117 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
  本切片未修改引擎、规则、动作编码或 Observation，因此未重复触发随机
  自博弈和 `rl_mixed_match`；1.4 保存的 1,000 局共享引擎门禁仍是当前
  规则提交的有效结果。未发现新 Bug，台账开放 P0/P1 维持 0。

## 1.6 目标、选择和待决状态扫描

- [x] 为每个选择操作测试零个、一个、多个合法候选。
- [x] 区分己方、敌方、双方、随从、护符、主战者、手牌和墓场目标。
- [x] 测试“其他”是否排除来源自身。
- [x] 测试不可指定、潜行、无视守护等限制只影响正确的目标种类。
- [x] 测试多目标是否允许重复、是否保持选择顺序、候选不足时是否截断。
- [x] 测试选择过程中目标死亡、离场、变身、改变控制者或不再符合筛选条件。
- [x] 测试来源在选择过程中离场时，已绑定目标和依赖来源的后续操作。
- [x] 测试一个效果中选择目标和随机/全部目标混合的顺序。
- [x] 测试无候选时属于禁止出牌、跳过操作还是执行 else 分支。
- [x] 验证 pending choice 可以 snapshot、restore，并在相同 seed 下继续得到
  相同结果。
- [x] 验证候选顺序、112 位动作编码和 UI 展示顺序一致。

2026-07-30 完成证据：

- `scripts/report_target_choice_audit.py` 动态扫描当前 826 张数据库卡和
  `RuleBook` 全部规则根、嵌套操作及纹章；识别 477 张正式目标/选择来源卡，
  其中 441 张可收集卡、36 张衍生卡、八套训练卡组递归闭包内 90 张，
  另有 21 个纹章级全局来源。8 个只存在于明确命名 `*_demo.json` 的合成
  规则来源单独列出，不计入 826 张正式卡池；正式来源卡全部关联 exact/token
  覆盖状态和存在的永久测试证据。
- 机器可读报告
  `data/reports/card_bug_audit/target_choice_audit.json` 和 Markdown 报告
  `.md` 盘点 245 个手选、128 个随机、241 个全部目标、103 个隐式/已绑定
  目标及 74 个决策操作。14 种手选目标域逐项验证空场和满候选的精确候选
  顺序，并用零/一/多个合法候选验证出牌门禁和零候选原子性；同一工作区
  连续生成的 JSON 与 Markdown SHA-256 均保持不变。
- 通用契约区分己方/敌方/双方的随从、护符、混合场面、主战者、手牌和
  墓场；“其他”排除来源但保留兼容目标。不可指定与潜行只阻止敌方手选
  能力，随机/全部目标不受影响；守护和无视守护只改变战斗目标，不改变
  能力选择。
- 多目标契约验证禁止重复时非法选择不改变指纹、允许重复时按次数结算、
  选择顺序保留且候选不足时截断。待决目标死亡、离场、变成不兼容护符、
  改变控制者或失去筛选资格时均重新验证并继续后续操作。
- 来源在选择期间离场后，已绑定且仍合法的目标继续结算；依赖来源攻击力
  的操作安全跳过且后续抽牌继续。手选破坏后再随机、再全部目标的实际
  结算结果证明三类目标按规则声明顺序读取更新后的场面。
- 零候选契约分别固定 `requires_target` 禁止出牌、可选手选操作跳过、
  随机/全部目标安全空操作和 `target_exists` 执行 else 分支。pending
  choice 的 snapshot/restore 在同一选择和 RNG 后产生相同事件及确定性
  指纹。
- 112 位 RL 契约验证 `pending_choice.options` 的 UI 顺序、`Choose`
  command 顺序、action mask 连续槽位和 action decode 逐项一致。
- `E:\anaconda\python.exe -m unittest tests.test_target_choice_audit -v`：
  13 项通过；目标/选择/区域/融合/真实卡聚焦回归集 434 项通过。
  本切片未发现新 Bug，台账开放 P0/P1 维持 0。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2735 项通过，1 项条件跳过；耗时 414.153 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片只新增审计器、报告、测试和文档，未修改引擎、规则、动作、目标、
  战斗、回合或 Observation，因此未触发额外随机自博弈和
  `rl_mixed_match` 门禁。

## 1.7 触发时机、优先级和批次扫描

- [x] 建立回合开始、回合结束、攻击时、交战时、受伤时、入场、进化、
  超进化、谢幕曲、护符倒数、纹章和信仰触发矩阵。
- [x] 验证回合边界各来源的官方优先级。
- [x] 保留马文纹章优先于场上卡片结算的真实回归测试。
- [x] 验证同时死亡先收集完整死亡批次，再开始谢幕曲。
- [x] 验证当前触发批次中生成的新纹章、新随从或新获得的回合结束能力
  不加入当前批次。
- [x] 验证触发条件应在批次开始还是实际结算时取快照。
- [x] 验证触发来源中途离场后，已排队能力是否继续结算。
- [x] 验证触发中产生选择时，队列暂停和恢复顺序不改变。
- [x] 验证致死结果确定后，未结算伤害、治疗和谢幕曲不能改变胜负。
- [x] 验证双方主战者同时受到伤害时的胜负归属。
- [x] 为递归触发加入有诊断信息的最大步数限制和最小循环复现。

2026-07-30 完成证据：

- `scripts/report_trigger_timing_audit.py` 动态读取当前 826 张数据库卡、
  `RuleBook`、规则覆盖报告和八套训练卡组递归闭包；共识别 770 张正式
  触发来源卡（711 张可收集卡、59 张衍生卡、训练闭包内 131 张），以及
  只存在于明确命名 `*_demo.json` 的 26 个合成来源。正式来源全部具有
  exact/token 覆盖状态和存在的永久测试证据，合成来源与正式卡池隔离。
- 机器可读报告
  `data/reports/card_bug_audit/trigger_timing_audit.json` 和 Markdown 报告
  `.md` 盘点 1,066 个卡牌触发定义、98 个事件监听器定义、67 个纹章定义
  及 55 个纹章触发、5 个信仰定义及 5 个信仰触发。回合开始、回合结束、
  攻击、交战、受伤后存活、入场、进化、超进化、谢幕曲、倒数、纹章和
  信仰 12 类矩阵均有正式来源和实际执行测试，报告零库存、证据或行为失败。
- `data/audits/timing_priority_evidence.json` 保存官方时序证据：Sandalphon
  官方 Q&A 固定纹章、倒数破坏、瞬念召唤、谢幕曲、瞬念后续能力和通常
  抽牌的回合开始顺序；Balt 官方 Q&A 固定双方主战者均为 1 时由对手获胜。
  马文官网卡牌文字确认其纹章是回合结束来源；马文纹章先于场上卡能力的
  跨来源顺序由现有真实卡回归保留，并明确记录当前官网未检出该组合的
  专门 Q&A，未把回归测试伪装成官方裁定。
- 11 个行为契约验证完整死亡批次先于谢幕曲、同批新增来源延后、批次条件
  与操作分支快照、已排队来源离场后继续、攻击/交战/回合/纹章/信仰/
  谢幕曲选择队列暂停恢复，以及致死后停止后续伤害、治疗和谢幕曲。新增
  直接契约证明较早来源破坏两张随从后，已经排队的第二张回合结束来源仍
  从快照对敌方主战者造成 3 点伤害。
- 现有 `MAX_RESOLUTION_STEPS=20,000` 门禁已经满足本项，不重复改写引擎；
  最小递归死亡/纹章循环会抛出 `ResolutionLoopError`，诊断包含事件、
  效果、死亡、纹章、监听器、暂停栈和日志，且相同 seed 的 JSON 诊断
  完全一致。`death_batch_start` 纹章触发仍是显式不支持边界；当前正式
  卡池来源数为 0，规则 schema 拒绝测试保持通过。
- `E:\anaconda\python.exe -m unittest tests.test_trigger_timing_audit -v`：
  10 项通过；1.7 触发/回合边界/死亡批次/纹章/信仰/监听器/真实卡聚焦
  回归集 288 项通过，耗时 22.733 秒。本切片未发现新 Bug，台账开放
  P0/P1 维持 0。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2745 项通过，1 项条件跳过；耗时 415.595 秒。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片只新增审计器、证据、报告、测试和文档，未修改引擎、规则、动作、
  目标、战斗、回合或 Observation，因此未触发额外随机自博弈和
  `rl_mixed_match` 门禁。

## 1.8 区域、容量和职业资源扫描

- [x] 手牌分别测试 0、8、9 张以及过量抽牌。
- [x] 场面分别测试 0、4、5 个实体以及随从死亡后腾出位置。
- [x] 测试抽牌、加入手牌、弃牌、返回手牌、消失、变身和回牌库的实体归属。
- [x] 验证一张实体卡不能同时存在于两个区域。
- [x] 测试空牌库、抽空时胜负和特殊空牌库规则。
- [x] 测试护符倒数、破坏、消失和激活的区别。
- [x] 测试纹章与信仰共享五格、同名纹章不叠加、区域满时的处理。
- [x] 分别验证连击、协作、墓影、觉醒、土之印、法术增幅、融合和
  奥义/解放奥义的增减时机。
- [x] 测试过量抽牌不算成功抽牌，不能触发相应抽牌监听器。
- [x] 验证墓场/消失区统计和 Observation 公共直方图与真实区域一致。

2026-07-30 完成证据：

- `scripts/report_zone_resource_audit.py` 动态读取当前 826 张数据库卡、
  `RuleBook`、规则覆盖报告和八套训练卡组递归闭包；识别 611 张正式
  区域/容量/资源来源卡（579 张可收集卡、32 张衍生卡、训练闭包内
  107 张）及 21 个只存在于明确命名 `*_demo.json` 的合成来源。正式
  来源全部具有 exact/token 覆盖状态和存在的永久测试证据，报告零库存、
  矩阵、契约或官方证据失败。
- 机器可读报告
  `data/reports/card_bug_audit/zone_resource_audit.json` 和 Markdown 报告
  `.md` 覆盖抽牌、加入手牌、弃牌、回手、消失、变身、回牌库、召唤、
  破坏、倒数/启动、领袖区、空牌库，以及连击、协作、墓影/唤灵、觉醒、
  土之印、魔力增幅、融合、奥义和解放奥义 21 类来源。9 组行为契约均
  有直接可执行测试。
- `data/audits/zone_resource_evidence.json` 保存 Cygames 官方帮助页、
  对战说明和祥和圣堂 Q&A：固定手牌九张、场面五张、纹章与信仰共享
  五格、空牌库通常败北、第十张抽牌进入墓场，以及各职业资源基础语义。
  报告同时保留“信仰不计入卡牌效果所说的纹章数量”，未把共享容量误写
  成效果计数。
- 新增 7 个横向行为测试：直接覆盖 `0→1`、`8→9`、`9→过抽`，场面
  `0→1`、`4→5`、满场失败和死亡腾位；逐项验证抽牌、生成入手、弃牌、
  回手、消失、变身和回牌库后的唯一实体归属；区分护符倒数破坏、效果
  破坏、消失和启动；验证四个信仰与一个纹章占满共享五格后，同名纹章
  和第六个不同名纹章均不进入区域。
- 资源时机沿用并实际运行现有专门测试：连击计入当前使用卡且回合末后
  清零，协作只在成功入场时增长，墓影按破坏增长并由唤灵一次支付，
  觉醒在 7 PP 边界切换，土之秘术先支付土之印，魔力增幅只增幅使用后
  留在手牌中的其他卡，融合素材不进入墓场/消失区且每张融合卡每回合
  一次，奥义/解放奥义按“当前回合数 + 在手期间己方进化次数”在
  10/15 阈值触发。
- 过量抽牌路径不发出 `CARD_DRAWN`；绝望神社鼠、坚固雾卷花纹章和
  锈蚀巨偶三条现有监听器回归均只响应成功抽牌。新增 Observation 契约
  逐卡比较双方墓场/消失区真实数量和公共直方图，并验证双方视角只交换
  顺序、不改变统计。
- `E:\anaconda\python.exe -m unittest tests.test_zone_resource_audit -v`：
  7 项通过；1.8 区域/容量/九类资源/监听器/真实卡聚焦回归集 538 项
  通过，耗时 26.792 秒。本切片未发现新 Bug，台账开放 P0/P1 维持 0。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2752 项通过，1 项条件跳过；耗时 419.317 秒，API test 通过。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 本切片只新增审计器、官方证据、报告、测试和文档，未修改引擎、规则、
  动作、目标、战斗、回合或 Observation，因此不触发额外随机自博弈和
  `rl_mixed_match` 门禁。

## 1.9 战斗、伤害、终局和随机性扫描

- [x] 测试随从攻击随从、攻击主战者、守护强制目标和无视守护。
- [x] 测试攻击前、交战前、交战中、战斗伤害后和攻击后触发的顺序。
- [x] 测试超进化随从保护、击败随从后对主战者的额外伤害和保护失效时点。
- [x] 测试效果伤害、战斗伤害、反伤、治疗、伤害替代和伤害上限。
- [x] 测试必杀、效果破坏、消失、变身和生命降到零的离场差异。
- [x] 测试单方致死、双方致死、牌库败北和特殊胜利。
- [x] 验证 game over 后队列停止，不能再产生可见状态变化。
- [x] 所有随机操作使用引擎 RNG，并在事件中保留足够的选择证据。
- [x] 相同 seed、卡组和命令序列必须产生相同 fingerprint、事件和胜负。
- [x] 无候选、跳过或非法分支不得意外消耗 RNG。

2026-07-30 完成证据：

- `scripts/report_combat_endgame_random_audit.py` 动态读取当前 826 张数据库
  卡（735 张可收集卡、91 张衍生卡）、`RuleBook`、规则覆盖报告和八套
  训练卡组递归闭包；识别 643 张战斗/伤害/终局/随机性正式来源卡，其中
  122 张位于训练闭包。七类来源矩阵和十项 checklist 行为契约均有存在的
  永久测试证据，报告零库存、契约或官方证据失败。
- 机器可读报告
  `data/reports/card_bug_audit/combat_endgame_random_audit.json` 和 Markdown
  报告 `.md` 覆盖攻击目标与守护、攻击/交战/伤害顺序、超进化保护及穿透
  伤害、伤害/治疗/替代/上限、五类离场、普通与特殊终局、game over 队列
  停止，以及确定性随机选择。AST 扫描覆盖 `swb/engine/*.py` 的 30 个
  随机调用点，违规 0：调用点只使用引擎自有 `self.random`，或由引擎向
  目标解析器显式传入的 `rng`。
- `data/audits/combat_endgame_random_evidence.json` 保存 Cygames 官方帮助
  词条、对战说明、卡塔莉娜卡牌文字、巴尔特 Q&A 和胜利卡词条，分别
  固化攻击/守护/必杀/屏障/离场/终局、超进化保护及额外伤害、单次伤害
  上限、双方致死停止顺序，以及特殊空牌库胜利的来源结论。
- 扫描发现并按“先复现、再失败测试、后修复”处理 P0
  `SWB-CARD-0003`：旧引擎把必杀错误绑定到 `actual damage > 0`，导致
  0 攻必杀和被屏障把伤害变为 0 的必杀不发动。修复前证据保存于
  `data/reports/card_bug_audit/reproductions/SWB-CARD-0003.json`；
  `60d1c2f` 改为按战斗接触触发必杀，并复用通用效果破坏路径，使效果
  破坏免疫和超进化保护继续有效。四项永久回归测试已登记在 Bug 台账，
  台账当前开放 P0/P1 均为 0。
- `E:\anaconda\python.exe -m unittest tests.test_keywords
  tests.test_combat_endgame_random_audit tests.test_card_bug_audit_ledger
  tests.test_play_mode_boundary_audit -v`：50 项通过；同时验证本次引擎源
  哈希变化后的玩法/模式边界报告已确定性重生成。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100`：
  100 局完成，胜场 `[50, 50]`、平局 0、截断 0、动作掩码不一致 0，
  acceptance `pass`。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 1000
  --mulligan-policy curve --validate-invariants --assert-official-acceptance
  --output
  data/reports/card_bug_audit/combat_endgame_random_self_play_1000.json`：
  1,000 局完成，胜场 `[468, 532]`、平局 0、截断 0、动作掩码不一致
  0，acceptance `pass`；机器可读结果已保存。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：完成，玩家 2 获胜，日志已保存。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2764 项通过，1 项条件跳过；耗时 422.004 秒，API test 通过。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。

## 1.10 RL 接口、Observation 和隐藏信息扫描

- [x] 对每个可执行命令验证至少有一个且只有预期的动作编码。
- [x] 对动作掩码中每个真位执行命令，确认合法且不会路由到错误目标/模式。
- [x] 对动作掩码中代表非法场景的位置抽样执行，确认状态原子不变。
- [x] 验证选项翻页不会遗漏候选、重复候选或形成无限 episode。
- [x] 改变对手隐藏手牌身份，确认策略 Observation 不变。
- [x] 改变对手未知牌库内容和顺序，确认策略 Observation 不变。
- [x] 验证公开打牌、攻击、目标和区域变化会正确进入 Observation 历史。
- [x] 验证完整私密状态只保存在持久化复盘日志，不通过在线 UI 或策略输入泄漏。
- [x] 验证 v3.6 和 v4.1 在相同引擎状态下各自满足正式 shape、dtype 和版本契约。
- [x] 规则修复导致状态字段变化时，明确判断是否需要 Observation 版本迁移。

2026-07-30 完成证据：

- `scripts/report_rl_interface_privacy_audit.py` 固化 `action-112-v2` 的
  112 个连续且无重叠动作槽、全部 9 种 `CommandType`、十项 1.10
  checklist 契约，以及 `observation-v3.6`/`observation-v4.1` 的正式
  schema manifest。机器可读报告
  `data/reports/card_bug_audit/rl_interface_privacy_audit.json` 和 Markdown
  报告 `.md` 当前结论为 PASS，失败 0。
- 新增十项横向回归：在普通行动、墓场选择、融合选择和调度阶段逐一验证
  合法 command 的唯一编码，并执行所有 mask 真位；按动作布局每段抽样
  mask 假位，比较完整引擎 fingerprint、RNG、分页、pending choice、
  state/transition version 和 agent step，确认非法动作原子不变。既有
  完整卡池出牌模式报告的 1,546 个费用/模式边界案例同时纳入门禁，
  command/mask mismatch 和非法原子性失败均为 0。
- 墓场 41 个候选的三页 fixture 逐页验证所有候选恰好出现一次、前后翻页
  有界、翻页不改变规则核心且计入 episode step；沿用墓场分页专门测试
  验证同 continuation 的新请求会回到第一页。
- v3.6 与 v4.1 都通过隐藏手牌身份、未知牌库内容和牌序的双环境逐字段
  相等测试。公开历史测试验证打牌、选择目标、区域离场、攻击及攻击目标
  分别进入正式 Observation；两个 live Gym space 均接受其 Observation。
  v3.6 为 18 个字段，manifest SHA-256 为
  `380bba38c548b392ab4e993e574bc9357d36bf38c6c4415332f68d133c1afcef`；
  v4.1 为 61 个字段，manifest SHA-256 为
  `bba4f4b923de6de1e5144b2725efe97907379d88b157a4b232bac3bf203b54b6`。
- 扫描发现并按“先复现、再失败测试、后修复”处理 P0
  `SWB-CARD-0004`：在线 `/api/history/<match_id>` 原样返回本地持久化
  JSON，泄漏 AI 手牌、私密日志/事件及完整策略分布。修复前证据保存于
  `data/reports/card_bug_audit/reproductions/SWB-CARD-0004.json`；
  `82bd251` 使在线 history 使用记录自身的 `human_player` 脱敏快照、
  日志、事件、动画、隐藏选择和完整策略分布，同时本地 schema-v2
  JSON 继续保留完整离线复盘。永久回归已登记，Bug 台账开放 P0/P1
  恢复为 0。
- `data/audits/rl_interface_privacy_evidence.json` 明确记录迁移判断：
  `60d1c2f` 只修正既有必杀结算语义，`82bd251` 只修正 simulator
  在线序列化边界；两者均未增删或重解释策略输入字段，也未改变动作布局，
  因而不迁移 Observation 版本。旧 checkpoint 仍由规则库哈希与新规则
  运行分离。
- `E:\anaconda\python.exe -m unittest tests.test_rl_interface_privacy_audit
  -v`：10 项通过；动作、Observation、缓存、分页、版本、在线复盘、
  出牌模式、融合、启动和官方开局相邻回归集 182 项通过，耗时
  38.441 秒。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2775 项通过，1 项条件跳过；耗时 436.361 秒，API test 通过。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100`：
  100 局完成，胜场 `[50, 50]`、平局 0、截断 0、动作掩码不一致 0，
  acceptance `pass`。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：完成，玩家 2 获胜，日志已保存。

## 1.11 建立运行时能力覆盖

- [x] 在测试/审计模式记录卡牌被抽到、打出、进化、超进化、攻击和离场。
- [x] 记录每种替代模式是否实际执行。
- [x] 记录每条结构化 clause 是否进入条件判断、真分支、假分支和操作执行。
- [x] 记录每种目标种类、无目标分支、容量不足分支和随机候选规模。
- [x] 记录 placeholder、unsupported、resolution step limit、非法动作和
  action mask mismatch。
- [x] 覆盖数据只使用稳定 card/clause ID，不依赖中文日志解析。
- [x] 将覆盖数据按卡牌、机制、卡组和对阵汇总为 JSON/Markdown。
- [x] 报告明确区分“能力没有触发”和“能力触发并通过”。

产物：

- `data/reports/card_bug_audit/runtime_coverage.json`
- `data/reports/card_bug_audit/runtime_coverage.md`

实现与证据（2026-07-30）：

- `swb/engine/runtime_coverage.py` 提供默认关闭、排除在 snapshot 和确定性
  fingerprint 之外的结构化采集器；稳定 ID 来自 card/trigger/mode/listener/
  emblem/operation 规则树路径，不读取中文日志。
- `tests/test_runtime_coverage.py` 的 9 项直接合同覆盖生命周期、六类替代
  入口、条件真/假、执行/未执行、无目标、容量不足、随机候选、六类异常
  计数、稳定 ID、snapshot operation round-trip、多 session clause 状态合并、
  聚合维度及审计开关不改变确定性状态。
- `E:\anaconda\python.exe -m scripts.report_runtime_coverage`：通过；固定卡组
  smoke 使用 seed 111，44 个 agent step 正常终局、未截断，登记训练闭包
  458 条结构化 operation clause，其中 15 条 `triggered_passed`、3 条
  `triggered_not_executed`、440 条 `not_triggered`；六类异常均为 0。
- 本切片发现并修复运行时诊断假阳性：已有结构化倒计时纹章仍会重复记录
  通用 `COUNTDOWN` placeholder。修复只校正能力覆盖判定，不改变对局状态、
  Observation 或动作布局；无结构化入口的同关键词仍由回归测试保证可见。
- 覆盖工具回归还锁定两项边界：同一 clause 跨 session 以
  `triggered_passed` > `triggered_not_executed` > `not_triggered` 合并，避免
  重复累计未触发；snapshot/clone 反序列化后的 operation 仍映射回原规则树
  clause ID，不退化成 dynamic ID。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：2,784 项通过，
  1 项跳过；`E:\anaconda\python.exe -m compileall -q swb scripts tests`：
  通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100
  --validate-invariants --assert-official-acceptance`：100 局完成，胜场
  `[50, 50]`、平局 0、截断 0、动作掩码不一致 0，acceptance `pass`。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 1000
  --validate-invariants --assert-official-acceptance --output
  data/reports/card_bug_audit/runtime_coverage_self_play_1000.json`：1,000 局
  完成，胜场 `[488, 512]`、平局 0、截断 0、动作掩码不一致 0，
  acceptance `pass`；机器可读结果已保存。该运行是 1.11 共享引擎改动
  门禁，不替代 1.12 的八套卡组对阵矩阵和 runtime coverage 定向采样。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log --validate-invariants`：完成，玩家 2 获胜，日志已保存。
- 440 条未触发项明确保留给 1.12 的强制场景、八套卡组矩阵和覆盖引导采样，
  不能作为卡牌能力已通过的证据。

## 1.12 强制场景生成和随机对局

- [ ] 为费用、目标、场面容量和资源阈值生成最小 GameState fixture。
- [ ] 为普通进化、超进化、回合开始/结束和同时死亡生成机制 fixture。
- [ ] 场景生成器只能通过公开 state/command/effect 接口准备局面，避免引入
  无法在真实对局出现的状态；必须直接改状态时要随后执行 invariants。
- [ ] 先对八套卡组闭包运行全部适用强制场景。
- [ ] 在八套卡组对阵矩阵上运行至少 1,000 局确定性 smoke。
- [ ] 同时使用 random legal 和当前 policy 采样，避免只覆盖一种动作分布。
- [ ] 根据 runtime coverage 对未触发 clause 继续生成定向局面，直到没有
  未解释的空白。
- [ ] 将工具扩展到 735 张可收集卡和 91 张衍生卡。
- [ ] 对完整卡池运行按机制分层的强制场景和至少 10,000 局采样。
- [ ] 为长局、截断和 Myuu 对阵单独保存分布及复现。

## 1.13 最小复现与自动回归闭环

- [ ] 定义可序列化复现包：数据库/规则哈希、卡组、seed、出错前 snapshot、
  命令、合法动作、掩码、事件、预期和实际。
- [ ] 复现包不得依赖当时 UI 或进程内对象。
- [ ] 增加命令序列缩减工具，优先删除与失败无关的早期回合和动作。
- [ ] 若无法缩减为合法自然对局，则输出最小 synthetic fixture。
- [ ] 每个已确认 Bug 先增加会失败的测试，再修改规则或引擎。
- [ ] 修复后运行同机制所有真实卡牌测试，防止只修某个 card ID。
- [ ] 卡牌行为优先写入 `data/rules/` 和通用 primitive，不在
  `resolution.py` 增加大段卡牌 ID 分支。
- [ ] 报告修复影响的旧 checkpoint；保留旧模型用于历史比较，但不得与
  新规则模型混作公平强度结论。

产物：

- `data/reports/card_bug_audit/repros/`
- 对应 `tests/test_*.py` 永久回归测试

## 1.14 八套训练卡组门禁

- [ ] 111 张左右的可收集卡并集及全部衍生闭包均有完整审计行。
- [ ] 所有适用替代模式和费用边界通过。
- [ ] 所有适用关键词来源和进入方式通过。
- [ ] 所有目标、时机、容量和职业资源条款至少有直接或生成测试。
- [ ] Runtime coverage 没有未解释的未触发 clause。
- [ ] 零 P0、零 P1 未关闭 Bug。
- [ ] 零 unsupported/placeholder、非法状态变更和 mask mismatch。
- [ ] 1,000 局八套卡组 smoke 无引擎异常，并能按 seed 完整复现。
- [ ] 完整单元测试、compileall 和规定 smoke 命令全部通过。

通过本门禁后，允许进行短期性能实验和小规模训练，但完整卡池门禁仍需继续。

## 1.15 完整卡池门禁

- [ ] 735 张可收集卡全部具备逐 clause 审计状态。
- [ ] 91 张衍生卡全部具备入口、行为和生产者审计。
- [ ] 所有适用机制矩阵对完整卡池运行完毕。
- [ ] 所有规则不确定项已有裁定，或被明确排除在训练 Catalog 外。
- [ ] Runtime coverage 没有把“未触发”误报为“通过”。
- [ ] 10,000 局分层采样无状态不变量、掩码、确定性和未支持能力错误。
- [ ] 零 P0、零 P1；剩余 P2/P3 有明确影响说明和复现。
- [ ] 同一复现集合在最终规则提交上全部通过。
- [ ] 生成最终总结并冻结 Git、数据库、规则、Catalog、Observation 和
  测试哈希。

产物：

- `docs/card_bug_audit_report.md`
- `data/reports/card_bug_audit/final_gate.json`

阶段 1 完成定义：

- 八套训练卡组门禁和完整卡池门禁均通过。
- 所有验证命令有真实保存结果。
- 规则基线提交后工作区干净，之后的性能优化不混入规则语义修改。

---

# 阶段 2：训练速度优化

## 2.0 性能优化边界

将候选改动分为三类，报告和 Git 提交不能混合：

- A 类：不改变 PPO 语义，例如减少拷贝、批量推理、缓存、预分配、
  合理线程数和等价 CUDA kernel。
- B 类：可能引入数值变化，但不改变数据流程，例如 FP16/BF16、TF32、
  `torch.compile` 和 fused optimizer。
- C 类：改变采样或优化语义，例如 actor/learner 异步、策略滞后、rollout
  长度、epoch、minibatch、网络宽度和 Observation 删减。

- [ ] 每个候选优化在实施前标记 A/B/C 类。
- [ ] 优先完成 A 类；B 类必须补数值和学习有效性实验。
- [ ] C 类不作为单纯“速度优化”合并，必须另做算法实验和三 seed 强度对比。
- [ ] 性能提交不得同时修改卡牌规则、Observation 含义或奖励函数。

## 2.1 建立公平性能基线

- [ ] 在阶段 1 的冻结规则提交上选择固定 v4.1 checkpoint。
- [ ] 记录 CPU、GPU、内存、PyTorch、CUDA、驱动、Python 和 Windows 版本。
- [ ] 记录电源模式、后台训练进程和 GPU 显存占用。
- [ ] 固定八套卡组对阵调度、master seed、4 worker、线程数、rollout、
  sequence length、epoch 和 minibatch。
- [ ] 先运行预热更新，预热数据不计入统计。
- [ ] 每次测量至少 100,000 agent steps。
- [ ] 基线独立运行三次，保存 median、P95 和波动范围。
- [ ] 同时测量 v3.6 和 v4.1，但只比较相同 Observation/模型内部的优化前后；
  不把不同输入宽度直接当作实现回归。
- [ ] 验证 checkpoint 在 profile 前后大小和 mtime 不变。
- [ ] 保存系统监控采样：CPU 总负载/每核负载、GPU utilization、显存、
  功耗和 RAM/页面文件。

产物：

- `data/reports/training_speed/baseline_run_*.json`
- `data/reports/training_speed/baseline_summary.json`

## 2.2 补全分阶段耗时统计

Worker 侧：

- [ ] 单独统计引擎 command/resolution 时间。
- [ ] 单独统计 legal command/action mask 时间。
- [ ] 单独统计 Observation v4.1 构造时间。
- [ ] 单独统计 IPC 请求序列化、发送和等待时间。
- [ ] 单独统计新对局 reset、换牌和卡组构造时间。
- [ ] 统计 worker 空闲比例、每局步数、长局和截断。

中央推理侧：

- [ ] 统计请求进入队列到组 batch 的等待时间。
- [ ] 记录每个推理 batch 的实际大小分布、P50/P95 和空槽。
- [ ] 统计 CPU 输入拼装和重复拷贝。
- [ ] 统计 host-to-device 传输。
- [ ] 统计 Transformer/GRU/action head 前向时间。
- [ ] 统计 masked distribution、采样和值函数处理。
- [ ] 统计 device-to-host 和结果分发。
- [ ] 统计 GPU 忙碌与等待 worker 的比例。

Learner 侧：

- [ ] 统计 trajectory 整理、padding 和张量构造。
- [ ] 统计 host-to-device 传输。
- [ ] 统计前向、loss、反向、梯度裁剪、optimizer step。
- [ ] 统计每 epoch/minibatch 的有效 token 数和 padding 比例。
- [ ] 统计 CUDA synchronize 对测量结果的影响，避免异步计时失真。

总体验收：

- [ ] 各互斥阶段耗时之和能够解释至少 90% 的 wall time。
- [ ] 报告同时给出总耗时、每步毫秒、占比、median 和 P95。
- [ ] 计时开关关闭时不明显降低正式训练吞吐。
- [ ] 为汇总计算、空样本和阶段求和增加测试。

## 2.3 定位 v4.1 相对 v3.6 的额外成本

- [ ] 用固定合成输入分别测量 batch 1/4/8/16/32/64 的纯前向吞吐。
- [ ] 分离非卡数值投影、卡牌 embedding、93 个语义 token 构造、
  Transformer、GRU 和 action head。
- [ ] 测量 token 打包在 CPU 和 GPU 上的时间及拷贝量。
- [ ] 测量不同长度 episode 的 GRU recurrent state 管理成本。
- [ ] 测量合法动作候选数量对 action-conditioned scoring 的影响。
- [ ] 使用 PyTorch Profiler 确定主要 CUDA kernel、launch gap 和同步点。
- [ ] 检查是否存在重复 Observation 转换、重复 embedding 或重复 mask 拷贝。
- [ ] 形成按预计收益排序的瓶颈清单，不凭 GPU/CPU 方波猜测原因。

产物：

- `data/reports/training_speed/v4_1_inference_breakdown.json`
- `docs/training_speed_bottleneck_report.md`

## 2.4 优化中央 GPU 推理合批

- [ ] 扫描 batch wait：0、0.1、0.25、0.5、1.0 ms。
- [ ] 扫描稳定 worker 数；先测 2/3/4/5/6，避免直接重试已发生分页压力的
  8 worker 配置。
- [ ] 扫描每 worker PyTorch 线程数 1/2/4。
- [ ] 每组记录 batch 大小分布、worker 等待、GPU 空闲、吞吐和回合长度。
- [ ] 保持每个 episode 独立 recurrent state 和独立 policy RNG。
- [ ] 保持一个 rollout generation 内权重固定，不引入策略滞后。
- [ ] 验证 batch 内请求排序变化不会把 hidden state 或动作发给错误 episode。
- [ ] 减少单步消息数量，在不改变决策边界的情况下批量传输固定字段。
- [ ] 使用预分配/复用缓冲区，减少 Python 对象和 NumPy/Tensor 重建。
- [ ] 评估 pinned memory 和 non-blocking H2D 是否有实际收益。
- [ ] 对每项候选分别提交和对比，不能一次合并多项后无法归因。

## 2.5 优化 Observation v4.1 热路径

- [ ] 确认同一 state version 的 Observation 和 action mask 只构造一次。
- [ ] 检查卡牌静态字段能否按 vocabulary 预计算，只查询运行时变化字段。
- [ ] 检查固定 93-token 布局能否直接写入连续数组，减少字典和临时列表。
- [ ] 预分配非卡数值、卡牌索引和 token feature 缓冲区。
- [ ] 避免在 worker 与中央推理之间传输未被 policy 使用的调试字段。
- [ ] 检查 dtype，避免 int64/float64 的不必要带宽和隐式转换。
- [ ] 保留隐藏信息和顺序不变性测试，优化不能改变 v4.1 语义。
- [ ] 对 cold/cached observation、完整环境 step 和端到端 PPO 分别测量；
  只优化 microbenchmark 不算完成。

## 2.6 优化 v4.1 网络前向

A 类候选：

- [ ] 将固定输入的张量布局调整为连续内存，减少 `permute/contiguous`。
- [ ] 合并可等价合并的小 projection，减少 CUDA kernel launch。
- [ ] 使用 PyTorch 原生 scaled-dot-product attention 的最快等价后端。
- [ ] 缓存真正静态且不参与梯度的推理侧卡牌编码。
- [ ] 对固定 batch bucket 评估 CUDA Graph；动态尾 batch 保留普通路径。
- [ ] 对网络每项改动验证相同输入的 logits/value/hidden state 在规定容差内一致。

B 类候选：

- [ ] 分别测试 TF32、FP16 autocast 和 BF16 autocast。
- [ ] 测试 `torch.compile` 的首次编译成本、稳态收益和 Windows 稳定性。
- [ ] 检查 masked logits、softmax、log probability 和 value 是否出现 NaN/Inf。
- [ ] 检查不同精度下动作概率误差、argmax 翻转率和 recurrent state 漂移。
- [ ] B 类只有在三 seed 小规模学习实验不退化后才能成为默认值。

## 2.7 优化 learner 更新

- [ ] 复核现有 backward/gradient clipping 和 forward/loss 占比。
- [ ] 复用 rollout tensor 缓冲区，减少每次 update 的分配和 Python 拼装。
- [ ] 将可提前完成的 padding/mask 计算移出 minibatch 内循环。
- [ ] 检查梯度清零方式、fused optimizer 和 foreach gradient clipping。
- [ ] 测试 AMP + GradScaler 的稳定性和真实端到端收益。
- [ ] 测试 minibatch 在显存允许范围内增大是否提高 GPU 利用率；该项属于
  C 类时必须另做学习有效性实验。
- [ ] 统计更新阶段每个 minibatch 的实际 token 数，避免 padding 吞掉收益。
- [ ] 任何改变 epoch、sequence length、rollout 或 minibatch 的方案都单独
  保存超参数实验，不伪装成纯实现优化。

## 2.8 评估流水线重叠和策略滞后

- [ ] 先评估同一 rollout 内 CPU 准备、H2D 和 CUDA 前向的安全重叠。
- [ ] 评估 learner update 中下一 minibatch 准备与当前 CUDA 计算的重叠。
- [ ] 不允许 worker 使用正在更新的权重。
- [ ] 若考虑 actor/learner 异步，明确记录 policy generation、最大 lag 和
  每条 trajectory 的行为策略 log probability。
- [ ] 异步方案列为 C 类；必须重新证明 PPO ratio、clip 和 on-policy 边界合理。
- [ ] 异步方案需要与同步方案做三 seed 学习曲线和固定评估，不只比较 steps/s。
- [ ] 在同步 A/B 类收益耗尽前，不优先实现分布式 learner。

## 2.9 每项性能候选的统一验收

- [ ] 使用同一基线配置至少运行三次。
- [ ] 报告 median steps/s、P95 stage time、GPU/CPU/RAM 和 batch 分布。
- [ ] 提升小于运行波动范围的候选判定为无明确收益。
- [ ] A 类候选必须通过固定 seed 轨迹、log probability、value、hidden state
  和 checkpoint resume 等价测试。
- [ ] B 类候选必须通过数值容差、NaN/Inf、长局和小规模学习测试。
- [ ] C 类候选必须通过至少三 seed 的固定对阵强度实验。
- [ ] 所有候选必须保持零非法动作、零 mask mismatch、零 worker 残留。
- [ ] 所有候选必须通过完整单元测试、compileall 和规定 smoke。
- [ ] 记录失败或无收益的候选，避免以后重复试验。

## 2.10 第一轮速度目标

- [ ] 在最终规则基线上得到可信的 v4.1 三次基线。
- [ ] 找到能解释主要 wall time 的瓶颈，不再只依赖 CPU/GPU 利用率曲线。
- [ ] 至少完成一项 A 类端到端优化。
- [ ] A 类组合相对新基线 median agent steps/s 提升至少 25%，或在未达到时
  给出测量证据说明硬瓶颈和下一路线。
- [ ] 100,000+ agent steps 稳定运行，无 OOM、分页、死锁和异常截断增加。
- [ ] v4.1 学习输入、网络输出契约、PPO generation 边界和 checkpoint
  兼容性保持不变。
- [ ] 生成优化前后综合报告，包括采用、拒绝和待研究方案。

产物：

- `docs/training_speed_optimization_report.md`
- `data/reports/training_speed/final_comparison.json`

阶段 2 完成定义：

- 端到端吞吐提升经过同配置三次测量，而非来自单次短 benchmark。
- 所有采用的优化有独立测试、独立性能证据和明确语义分类。
- 正式长训练配置被写入 checkpoint/report，其他电脑可按同一配置复现。

---

# 每个实现切片的固定流程

- [ ] 开始前运行 `git status --short --branch`，保留无关用户改动。
- [ ] 明确本切片修改的文件、行为和严重等级。
- [ ] 先增加失败测试或保存性能基线。
- [ ] 做最小 coherent change，不混入另一阶段的修改。
- [ ] 运行聚焦测试。
- [ ] 运行完整测试：

```powershell
python -m unittest discover -s tests -v
python -m compileall -q swb scripts tests
```

- [ ] 规则、动作、Observation 或战斗变化额外运行：

```powershell
python -m scripts.random_self_play --games 100
python -m scripts.rl_mixed_match --output data/rl_mixed_match.log
```

- [ ] 广泛共享引擎改动将随机自博弈提高到 1,000 局。
- [ ] 更新 checklist、报告和 roadmap 中受影响的事实。
- [ ] 保存本地 Git 版本仅在用户明确要求时进行。
- [ ] 推送远端仅在用户明确要求时进行。
