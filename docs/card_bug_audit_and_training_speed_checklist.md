# 卡牌 Bug 排查与训练速度优化 Checklist

最后更新：2026-07-31

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
- 当前完整测试为 2812 项通过、1 项跳过。
- 最近一次 v4.1 三 seed 实验约为 99.84 agent steps/s；该数据来自既有
  实验，不作为优化后的公平基线。

## Checklist 使用规则

- 只有同时具备代码/规则、自动化测试和保存的报告证据时才能勾选任务。
- 每个发现的 Bug 必须先保存最小复现，再修复，再增加永久回归测试。
- “随机对局没有报错”不能代替能力分支覆盖。
- “卡牌有一项测试”不能代替逐条卡牌文字和边界条件审计。
- 不确定规则必须保持显式待确认，不能根据现有引擎行为反推官方规则。
- 具体交互裁定必须先依次搜索 Cygames 官方卡牌页/Q&A、官方帮助与规则、
  官方公告/勘误、官方其他语言页面，再检查可重复客户端证据；查询词、日期、
  URL、适用范围和结论保存到 `data/audits/card_ruling_reviews.json`。
- 未找到直接或足够类似的官方裁定时只能采用标记为 `ruling_uncertain` 的
  暂定解释，并记录其他解释、选择理由、影响与客户端复现/官方答复 follow-up；
  测试和不变量通过本身不能关闭裁定问题。
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

- [x] 为费用、目标、场面容量和资源阈值生成最小 GameState fixture。
- [x] 为普通进化、超进化、回合开始/结束和同时死亡生成机制 fixture。
- [x] 场景生成器只能通过公开 state/command/effect 接口准备局面，避免引入
  无法在真实对局出现的状态；必须直接改状态时要随后执行 invariants。
- [x] 先对八套卡组闭包运行全部适用强制场景。
- [x] 在八套卡组对阵矩阵上运行至少 1,000 局确定性 smoke。
- [x] 同时使用 random legal 和当前 policy 采样，避免只覆盖一种动作分布。
- [x] 根据 runtime coverage 对未触发 clause 继续生成定向局面，直到没有
  未解释的空白。
- [x] 将工具扩展到 735 张可收集卡和 91 张衍生卡。
- [x] 对完整卡池运行按机制分层的强制场景和至少 10,000 局采样。
- [x] 为长局、截断和 Myuu 对阵单独保存分布及复现。

完成证据（2026-07-31；发现基线
`bb5635b58709e0c3e6cf5486f6708530f47be3f2`，SWB-CARD-0008 官方裁定
修正工作树 `af133ea272ccc77eb964577c364360dab3ef5526`）：

- `swb/engine/forced_scenarios.py` 提供费用、目标、容量、资源、普通/超进化、
  回合开始/结束和同时死亡共 9 个通用最小 fixture；17 次必要直接状态准备
  每次后均运行 invariants，共记录 35 次不变量检查。
- `data/reports/card_bug_audit/forced_scenario_audit.json`：
  9/9 fixture 通过；八套训练卡组 147 张闭包和完整 735+91 卡池共生成
  2,793 个机制分层场景分配，0 个未解释 runtime clause、0 个缺失测试文件。
  直接测试证据与运行时已触发证据保持分栏，未把“有测试”冒充“随机局已触发”。
- `data/reports/card_bug_audit/training_matrix_1000.json`：
  1,024/1,024 局终局，960 局 random legal、64 局冻结 checkpoint policy，
  128 个有序卡组/策略分层；1,024 次相同 seed 重放全部一致；95,230 次
  mask 检查，placeholder、mask 分歧、非法动作、异常和截断均为 0。
- `data/reports/card_bug_audit/full_pool_sampling_10000.json`：
  10,000/10,000 局终局，9,804 局 random legal、196 局冻结 checkpoint
  policy，98 个职业/策略分层；735/735 可收集卡进入卡组、实际遇见 824 张卡，
  98 次分层重放全部一致；909,158 次 mask 检查，placeholder、mask 分歧、
  非法动作、异常和截断均为 0，acceptance `pass`。
- `data/reports/card_bug_audit/full_pool_sampling_10000_failed_20260730.json`
  保留首次长跑失败数据；`SWB-CARD-0005`/`0006` 记录结构化规则已实现却误报
  placeholder 的 P1 诊断缺陷，`SWB-CARD-0007` 记录谢幕曲致死后父效果过早
  请求选择、留下 0 生命单位的 P0 解析缺陷。最终 1,000 局共享引擎门禁又在
  game 433、seed 120445 发现 `SWB-CARD-0008`：更早的 `+2/-2` 修正错误残留
  在后续“生命设为 1”的内部基值下，超进化结算为攻击 8、当前生命 4、生命
  上限 2，并触发不变量异常。三张真实卡的 107 步动作复现保存于
  `data/reports/card_bug_audit/reproductions/SWB-CARD-0008-random-self-play.json`，
  最小回归与真实 post-fix 8/4（生命上限 4）精确回放结论保存于
  `SWB-CARD-0008.json`，机器可读回放为 `SWB-CARD-0008-postfix-official.json`。
  旧 2/2 结论仅证明不变量不再失败，因与 Cygames《雪人觉醒》官方 Q&A
  冲突而被重开并纠正。最终 `Unit.set_stats()` 以通用维度覆盖语义修复，
  不含 card ID 特判；Bug 台账 P0/P1 未关闭数均为 0。
- `data/audits/card_ruling_reviews.json` 固化官方来源优先流程并保存本次查询。
  `SWB-RULING-0008-A` 有《雪人觉醒》直接官方 Q&A；更早临时修正到期是否
  反向影响后续设定值未找到直接官方资料，保留为
  `SWB-RULING-SET-STATS-TEMP-001` / `ruling_uncertain`，采用“设定覆盖此前
  同维度修正”的暂定解释并等待客户端复现。
- `data/reports/card_bug_audit/long_truncation_myuu_distribution.json`：
  汇总 11,024 局，保存 95 个长局、0 个截断和 240 个 Myuu 对局的完整复现
  manifest；Myuu 截断 0，回合 p99/max 为 37/46，agent steps p99/max 为
  175/360。人类可读摘要见同名 `.md`。
- 最终规则下的共享引擎门禁保存在
  `data/reports/card_bug_audit/stage_1_12_0008_official_random_self_play_100.json`
  和 `stage_1_12_0008_official_random_self_play_1000.json`：固定 seed
  120012 的 100/1,000
  局均无平局、截断、非法动作和 mask 分歧，官方开局验收均为 `pass`；
  1,000 局胜场 `[521, 479]`，Extra PP 使用 1,895 次。自博弈脚本现在会在
  异常时保存失败局号、局 seed、双方卡组、完整动作序列、mask、场面和状态
  指纹，避免长跑失败只留下 traceback。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：最终规则下
  2,812 项通过，1 项条件跳过，耗时 435.269 秒，API test 通过；完整控制台
  记录保存在本地
  `data/reports/card_bug_audit/stage_1_12_0008_official_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log --validate-invariants`：完成，玩家 2 获胜，日志已保存。

## 1.13 最小复现与自动回归闭环

- [x] 定义可序列化复现包：数据库/规则哈希、卡组、seed、出错前 snapshot、
  命令、合法动作、掩码、事件、预期和实际。
- [x] 复现包不得依赖当时 UI 或进程内对象。
- [x] 增加命令序列缩减工具，优先删除与失败无关的早期回合和动作。
- [x] 若无法缩减为合法自然对局，则输出最小 synthetic fixture。本次条件
  未触发：真实轨迹已从 107 步合法缩至 86 步；仍额外保存 synthetic
  primitive fixture 作为更小的永久回归。
- [x] 每个已确认 Bug 先增加会失败的测试，再修改规则或引擎。
- [x] 修复后运行同机制所有真实卡牌测试，防止只修某个 card ID。
- [x] 卡牌行为优先写入 `data/rules/` 和通用 primitive，不在
  `resolution.py` 增加大段卡牌 ID 分支。
- [x] 报告修复影响的旧 checkpoint；保留旧模型用于历史比较，但不得与
  新规则模型混作公平强度结论。

1.13 证据（2026-07-31）：

- `scripts.card_bug_repro_package` 对 `SWB-CARD-0008` 的原始 107 步轨迹
  执行 755 次候选回放，得到 86 步合法自然对局；最终 command 仍为
  `SuperEvolve(player_index=0, unit_id=69)`，前态 5/1/1，后态 8/4/4。
  可移植包保存在
  `data/reports/card_bug_audit/repros/SWB-CARD-0008.json`，仅含 JSON
  原生值，并记录数据库/规则哈希、精确卡组/seed、完整前态、112 位
  action mask、合法 command、转移事件、官方预期与修复前实际。
- 附加 synthetic fixture 保存在
  `data/reports/card_bug_audit/repros/SWB-CARD-0008-synthetic.json`；
  自然缩减已经成功，因此它不是替代真实对局的回退证据。
- Bug ledger 的 8/8 个 confirmed/fixed 项均有修复前实际、发现版本、
  reproduction 文件和永久回归引用。SET_STATS 同机制的《雪人觉醒》
  普通/超进化、姬华、圣骑士团员和帕斯卡尔 5 项真实卡测试均通过；
  修复只进入 `Unit.set_stats` 与通用 SET_STATS 结算，新增 card-ID
  分支为 0。
- 只读扫描本地 52 个 checkpoint，52/52 可读且均早于 `b6f1d95`。
  文件全部保留，但只能用于历史复现，不得与修复后模型混作公平强度
  结论。由于本次是 Python 引擎语义修改，旧模型即使 rulebook hash
  相同也不代表轨迹兼容。机器可读清单见
  `data/reports/card_bug_audit/repros/checkpoint_impact.json`。
- `E:\anaconda\python.exe -m unittest
  tests.test_card_bug_repro_package tests.test_card_bug_checkpoint_impact
  tests.test_card_audit_reproduction -v`：12 项通过。
- 5 项 SET_STATS 同机制真实卡定向回归：5 项通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：2,823 项
  通过，1 项条件跳过，耗时 434.462 秒，API test 通过；完整控制台
  记录保存在本地
  `data/reports/card_bug_audit/stage_1_13_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 1.13 未修改引擎、规则、command、mask、Observation 或对局语义；
  使用上一 checkpoint `b6f1d95` 已通过的 100/1,000 局、八套 1,024
  局和全池 10,000 局门禁作为冻结规则证据，本工具/报告 slice 不重复
  执行对局 smoke。
- 综合机器可读结论：
  `data/reports/card_bug_audit/stage_1_13_repro_closure.json`。

产物：

- `data/reports/card_bug_audit/repros/`
- 对应 `tests/test_*.py` 永久回归测试

## 1.14 八套训练卡组门禁

- [x] 111 张左右的可收集卡并集及全部衍生闭包均有完整审计行。
- [x] 所有适用替代模式和费用边界通过。
- [x] 所有适用关键词来源和进入方式通过。
- [x] 所有目标、时机、容量和职业资源条款至少有直接或生成测试。
- [x] Runtime coverage 没有未解释的未触发 clause。
- [x] 零 P0、零 P1 未关闭 Bug。
- [x] 零 unsupported/placeholder、非法状态变更和 mask mismatch。
- [x] 1,000 局八套卡组 smoke 无引擎异常，并能按 seed 完整复现。
- [x] 完整单元测试、compileall 和规定 smoke 命令全部通过。

1.14 证据（2026-07-31）：

- 八套固定卡组直接可收集卡并集为 111 张；递归引用新增 36 张，最终
  闭包 147 张，其中 116 张可收集、31 张衍生。1.14 汇总门禁将
  1.5 的结构化矩阵与后续实际执行证据合并，147/147 最终审计行均有
  source alignment、适用 forced scenario、直接测试及 runtime
  解释，0 行失败。
- 替代模式门禁覆盖训练闭包中的 17 张模式卡、55 个模式、1,546 个
  费用边界和 55 个满场边界，command/mask mismatch、非法操作
  原子性失败和执行失败均为 0。
- 关键词门禁覆盖训练闭包 59 个关键词来源、9 个运行时关键词、12 种
  进入方式；inventory、contract 和 matrix failure 均为 0。
- 目标、时机、容量与职业资源共获得 2,793 个 forced-scenario
  assignment；9/9 最小 public-interface fixture 通过，17 次必要
  直接状态设置后执行了 35 次 invariant check。八套卡组的所有适用
  scenario 均通过。
- 训练闭包共有 458 个 structured runtime clause：15 个在 smoke
  中实际触发并通过；原始 440 个 `not_triggered` 和 3 个
  `triggered_not_executed` 保持原标签，另由重新执行的直接测试逐条
  解释，未解释 clause 为 0，未把“未触发”误报成“runtime passed”。
- Bug ledger 为 6 个 P0、2 个 P1，8/8 fixed，未关闭 P0/P1 均为 0。
  八套矩阵累计 95,230 次 mask check，unsupported/placeholder、
  illegal action、mask mismatch、exception 和 truncation 均为 0。
- 八套矩阵实际完成 1,024 局（960 random-legal、64 frozen-policy），
  128/128 sampling strata，1,024/1,024 按 seed 完整重放，0 replay
  failure，全部正常终局。
- `E:\anaconda\python.exe -m unittest tests.test_training_deck_gate -v`：
  6 项通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：2,829 项通过，
  1 项条件跳过，耗时 436.925 秒；随后 API test 通过。完整输出保存于
  `data/reports/card_bug_audit/stage_1_14_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- 规则冻结提交 `b6f1d95` 后的规定 smoke 证据沿用 1.12 的实际重跑：
  `stage_1_12_0008_official_random_self_play_100.json` 和
  `stage_1_12_0008_official_random_self_play_1000.json` 均通过，
  `data/rl_mixed_match.log --validate-invariants` 完成且玩家 2 获胜；
  1.14 未修改引擎、规则、动作、目标、战斗、回合或 Observation。
- 机器可读报告：
  `data/reports/card_bug_audit/training_deck_gate.json`；可读摘要：
  `data/reports/card_bug_audit/training_deck_gate.md`。

通过本门禁后，允许进行短期性能实验和小规模训练，但完整卡池门禁仍需继续。

## 1.15 完整卡池门禁

- [x] 735 张可收集卡全部具备逐 clause 审计状态。
- [x] 91 张衍生卡全部具备入口、行为和生产者审计。
- [x] 所有适用机制矩阵对完整卡池运行完毕。
- [x] 所有规则不确定项已有裁定，或被明确排除在训练 Catalog 外。
- [x] Runtime coverage 没有把“未触发”误报为“通过”。
- [x] 10,000 局分层采样无状态不变量、掩码、确定性和未支持能力错误。
- [x] 零 P0、零 P1；剩余 P2/P3 有明确影响说明和复现。
- [x] 同一复现集合在最终规则提交上全部通过。
- [x] 生成最终总结并冻结 Git、数据库、规则、Catalog、Observation 和
  测试哈希。

产物：

- `docs/card_bug_audit_report.md`
- `data/reports/card_bug_audit/final_gate.json`

1.15 已验证证据（2026-07-31，完成）：

- `data/reports/rule_coverage.json` 重新经完整测试的确定性生成检查：
  826 张数据库定义中，735/735 张可收集卡均为 `covered_exact` 且
  clause audit 为 `mapped_exact`；`unverified_exact`、partial、
  missing rule/schema/primitive/targeting、timing/text uncertain 和
  external blocker 均为 0。
- `data/reports/token_audit.json` 覆盖 91/91 张非收集/衍生卡，
  `entry_behavior_complete=91`；partial、无入口、文本不清楚和外部
  blocker 均为 0。
- 完整卡池机制报告均在 2,829 项完整测试中重新生成比对并通过：
  play-mode 54 张卡/55 个模式/1,546 个费用边界/55 个满场边界，
  keyword-entry 0 failure，target/choice 477 个正式来源，
  trigger/timing 770 个正式来源，zone/resource 611 个正式来源，
  combat/endgame/random 643 个正式来源；各报告 failure 均为 0。
- 唯一待确认项 `SWB-RULING-SET-STATS-TEMP-001` 于 2026-07-31
  再次按官方卡牌 Q&A → 帮助/用语集 → 公告/勘误 → 其他语言的顺序检索。
  官方仍只直接确认《雪人觉醒》设定生命后的进化增量，没有回答较早临时
  修正到期。查询、官方页面、可选解释、影响和后续均保存于
  `data/audits/card_ruling_reviews.json`，状态继续保持
  `ruling_uncertain`，没有用当前引擎或测试反推官方规则。
- 本地规则与数据库路径核对证明该边界可经跨职业复制进入实战，不能以职业
  限制判为不可达。`data/audits/training_catalog_exclusions.json`
  因此仅将 `10233310`《帕梅拉的舞蹈》排除在新采样初始牌组外；
  735 张仍全部审计和可解析，训练池为 734 张。被排除卡仍可用于显式 fixture
  和历史回放；获得直接官方 Q&A 或版本化客户端复现后才能重新纳入。
- Catalog/裁定/冻结 baseline/固定卡组/版本合同聚焦验证：
  `E:\anaconda\python.exe -m unittest tests.test_training_catalog
  tests.test_card_ruling_reviews tests.test_card_bug_audit_baseline
  tests.test_fixed_decks tests.test_rl_versioning -v`，35 项通过。
- 首次完整运行暴露旧 checkpoint 的 Catalog/训练池哈希不匹配，保存于
  `stage_1_15_catalog_exclusion_unittest.log`。修复后训练恢复继续严格拒绝
  不兼容 checkpoint；固定卡组纯推理仅在卡牌词表、Observation 和动作合同
  兼容时允许加载，并明确警告不能恢复训练、复现原分布或公平比较。模拟器、
  checkpoint 和 Catalog 聚焦回归 38 项通过。
- 第二次完整运行的唯一失败是 RL 接口审计输入文件哈希随新增测试过期，
  保存于 `stage_1_15_catalog_exclusion_unittest_rerun.log`；重新生成
  `rl_interface_privacy_audit` 后 1,546 个 case、0 failure，其 10 项测试
  与联动的 1.14 汇总 6 项测试均通过。
- Catalog 隔离切片完成时的完整
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2,831 项通过，1 项条件跳过，耗时 437.101 秒，API test 通过；完整输出
  保存于 `stage_1_15_catalog_exclusion_unittest_final.log`。
- 加入最终门禁生成器与 8 项门禁回归后的最终完整套件：
  `E:\anaconda\python.exe -m unittest discover -s tests -v`，
  2,839 项通过，1 项条件跳过，耗时 437.775 秒，API test 通过；完整输出
  保存于 `data/reports/card_bug_audit/stage_1_15_final_gate_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `data/reports/card_bug_audit/forced_scenario_audit.json` 保留 2,151 个
  runtime clause 的原始分类：1,693 个 `not_sampled_full_pool`、440 个
  `not_triggered`、3 个 `triggered_not_executed`、15 个
  `triggered_passed`；前 2,136 个没有被改标为通过，未解释条款为 0。
- 最终 `full_pool_sampling_10000.json` 完成 10,000/10,000 局、
  98 个职业/策略分层、735/735 张可收集卡入组、824 张卡实际遇见、
  909,158 次 mask check 和 98 次固定 seed 重放；异常、截断、非法动作、
  placeholder、mask mismatch 和重放失败均为 0。更早失败报告
  `full_pool_sampling_10000_failed_20260730.json` 保留 2 次异常和
  194 个 placeholder，没有被成功报告覆盖。
- Bug ledger 的 8 个已确认问题全部 fixed（6 个 P0、2 个 P1），未关闭
  P0/P1/P2/P3 均为 0。`card_bug_repro_package --validate-only` 在冻结规则
  提交重放 86 个合法动作并得到官方预期 8/4/4；同一复现/回归集合与最终
  门禁合计 20 项聚焦测试通过。
- `scripts/report_full_pool_gate.py` 汇总上述独立证据，9/9 门禁通过；
  聚焦 `tests.test_full_pool_gate` 8 项通过。机器可读报告为
  `data/reports/card_bug_audit/final_gate.json`，可读最终报告为
  `docs/card_bug_audit_report.md`。
- 最终冻结标识包括规则引擎提交 `b6f1d95`、Catalog 策略提交
  `9699ab9`、数据库/规则/Catalog/训练池 SHA-256、v3.6/v4.1
  Observation manifest、112-action layout、测试和脚本目录哈希。

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

- [x] 每个候选优化在实施前标记 A/B/C 类。
- [x] 优先完成 A 类；B 类必须补数值和学习有效性实验。
- [x] C 类不作为单纯“速度优化”合并，必须另做算法实验和三 seed 强度对比。
- [x] 性能提交不得同时修改卡牌规则、Observation 含义或奖励函数。

2.0 已验证证据（2026-07-31）：

- `data/reports/training_speed/candidate_registry.json` 在实施前登记 15 项候选：
  10 项 A 类、2 项 B 类、3 项 C 类，并为每项记录语义边界和当前处置。
- 注册表固定执行顺序为 A → B → C；B 类要求数值、长局和三 seed
  小规模学习证据，C 类只允许作为独立算法实验并要求至少三 seed
  固定对阵强度比较。
- 性能提交的禁止混合范围明确包含 `data/rules/`、`swb/engine/`、
  Observation 语义和奖励函数。
- `E:\anaconda\python.exe -m unittest
  tests.test_training_speed_candidate_registry -v`：4 项通过。
- 首次完整套件正确暴露 1.15 报告将冻结测试目录错误地与未来 HEAD
  比较；失败输出保存在
  `data/reports/training_speed/stage_2_0_unittest.log`。冻结回归现改为验证
  `fae33c2` 中已提交的报告 blob，未来新增性能测试只归一化动态目录哈希，
  不重写历史门禁。
- 修正后的 `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2,843 项通过，1 项条件跳过，耗时 438.762 秒，API test 通过；完整输出
  保存于 `data/reports/training_speed/stage_2_0_unittest_final.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。

## 2.1 建立公平性能基线

- [x] 在阶段 1 的冻结规则提交上选择固定 v4.1 checkpoint。
- [x] 记录 CPU、GPU、内存、PyTorch、CUDA、驱动、Python 和 Windows 版本。
- [x] 记录电源模式、后台训练进程和 GPU 显存占用。
- [x] 固定八套卡组对阵调度、master seed、4 worker、线程数、rollout、
  sequence length、epoch 和 minibatch。
- [x] 先运行预热更新，预热数据不计入统计。
- [x] 每次测量至少 100,000 agent steps。
- [x] 基线独立运行三次，保存 median、P95 和波动范围。
- [x] 同时测量 v3.6 和 v4.1，但只比较相同 Observation/模型内部的优化前后；
  不把不同输入宽度直接当作实现回归。
- [x] 验证 checkpoint 在 profile 前后大小和 mtime 不变。
- [x] 保存系统监控采样：CPU 总负载/每核负载、GPU utilization、显存、
  功耗和 RAM/页面文件。

产物：

- `data/reports/training_speed/baseline_run_*.json`
- `data/reports/training_speed/baseline_summary.json`

2.1 已验证证据（2026-07-31）：

- 固定规则基线为 `fae33c2`。源 500k checkpoint 早于 Catalog 隔离策略，
  且历史环境快照引用旧 Catalog pickle 签名；工具没有放宽正式
  `load_checkpoint`。它验证规则、卡牌词表、Observation、动作布局等合同
  完全相同且隔离卡 `10233310` 不在八套固定牌后，用当前冻结环境/RNG
  容器严格载入原模型与 optimizer 权重，生成专用只读测速副本。源文件
  SHA-256、大小和 mtime 保持不变，迁移仅限 `catalog_sha256` 与
  `training_pool_sha256`；完整来源记录在
  `data/reports/training_speed/baseline_configuration.json`。
- 固定配置为 master seed `20260801`、官方主教训练牌组及 7 套对手牌组、
  4 worker、每 worker 2 个 PyTorch 线程、0.5ms 合批等待、rollout 2048、
  sequence length 32、2 epoch、8 minibatch sequences；v4.1 学习率
  `1e-4`，v3.6 保留其自身 checkpoint 的 `3e-4`，跨 Observation 不做
  直接实现回归比较。
- 每次请求 104,096 steps，先排除 2 个完整预热 update，再验证正式统计段
  至少 100,000 steps。v4.1 三次正式段分别为 101,258 / 101,437 /
  101,132 steps，吞吐 44.705 / 44.578 / 44.739 steps/s；median
  44.705，范围 0.161 steps/s，rollout/update P95 为
  32.234 / 22.901 秒。
- v3.6 三次正式段分别为 101,141 / 101,666 / 101,666 steps，吞吐
  130.276 / 129.801 / 129.404 steps/s；median 129.801，范围
  0.871 steps/s，rollout/update P95 为 9.961 / 8.765 秒。该差异仅说明
  两个冻结输入/模型的各自成本，不作为优化前后比较。
- 三次 v4.1 共保存 3,470 个、三次 v3.6 共保存 1,204 个 2 秒系统样本，
  包含 CPU 总量/每核、RAM、页面文件、GPU 利用率、显存和功耗；机器为
  i7-13700KF、RTX 4080 16GiB、32GiB RAM、Windows 11 build 26200、
  PyTorch 2.13.0+cu130、驱动 591.86、卓越性能电源方案。每份报告还保存
  Python/CUDA/cuDNN、后台训练进程和 profile 前后系统快照。
- 六次均验证测速 checkpoint 的 SHA-256、大小和 mtime 不变；逐次报告为
  `data/reports/training_speed/baseline_run_*.json`，汇总为
  `data/reports/training_speed/baseline_summary.json`。多进程请求到达
  时序使 run 1 与 run 2/3 在后段出现少量 episode 长度分叉，因此后续
  A 类轨迹等价验证必须另用受控请求序列，不能把端到端同 seed 误称为
  字节级轨迹确定。
- `E:\anaconda\python.exe -m unittest tests.test_training_speed_baseline -v`：
  与既有 profiling 汇总回归合计 7 项通过；最终
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：
  2,847 项通过，1 项条件跳过，耗时 440.987 秒，API test 通过，完整输出
  保存于 `data/reports/training_speed/stage_2_1_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。

## 2.2 补全分阶段耗时统计

Worker 侧：

- [x] 单独统计引擎 command/resolution 时间。
- [x] 单独统计 legal command/action mask 时间。
- [x] 单独统计 Observation v4.1 构造时间。
- [x] 单独统计 IPC 请求序列化、发送和等待时间。
- [x] 单独统计新对局 reset、换牌和卡组构造时间。
- [x] 统计 worker 空闲比例、每局步数、长局和截断。

中央推理侧：

- [x] 统计请求进入队列到组 batch 的等待时间。
- [x] 记录每个推理 batch 的实际大小分布、P50/P95 和空槽。
- [x] 统计 CPU 输入拼装和重复拷贝。
- [x] 统计 host-to-device 传输。
- [x] 统计 Transformer/GRU/action head 前向时间。
- [x] 统计 masked distribution、采样和值函数处理。
- [x] 统计 device-to-host 和结果分发。
- [x] 统计 GPU 忙碌与等待 worker 的比例。

Learner 侧：

- [x] 统计 trajectory 整理、padding 和张量构造。
- [x] 统计 host-to-device 传输。
- [x] 统计前向、loss、反向、梯度裁剪、optimizer step。
- [x] 统计每 epoch/minibatch 的有效 token 数和 padding 比例。
- [x] 统计 CUDA synchronize 对测量结果的影响，避免异步计时失真。

2.2 worker command/resolution 切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。`ShadowverseEnv.step()` 新增默认关闭的可选
  timing sink；未传入时不调用性能时钟，现有正式执行路径与 API 保持兼容。
- 中央策略 worker 在原有 `worker_engine_step_seconds` 总量之外，分别汇总
  `worker_command_decode_seconds` 与 `worker_resolution_seconds`；page-only
  动作明确记录 resolution 为 0，不伪造引擎执行时间。
- 等价回归从同一环境 snapshot 分别执行普通/计时 step，验证
  `StepResult` 与最终 snapshot 完全相同；PPO 多 worker 回归验证两个新字段
  均为正且其和不超过原 engine-step 总量。
- 冻结 v4.1 checkpoint 的 2,048-step 实测实际完成 2,282 steps：
  worker engine-step 合计 5.381254 秒，其中 command decode 0.014082 秒、
  resolution 1.295004 秒；checkpoint 哈希保持不变。机器可读结果保存于
  `data/reports/training_speed/stage_2_2_command_resolution_smoke.json`。
- `E:\anaconda\python.exe -m unittest
  tests.test_environment.EnvironmentTests.test_optional_step_timing_preserves_result_and_state
  tests.test_ppo.PPOTrainerTests.test_entity_action_policy_collects_and_updates_fixed_deck
  -v`：2 项通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：运行 2,848 项，
  1 项条件跳过、其余通过，耗时 438.608 秒，API test 通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_command_timing_unittest_final.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100`：100 局
  全部完成，`wins=[56, 44]`、draw/truncation/mask mismatch 均为 0；
  输出保存于
  `data/reports/training_speed/stage_2_2_command_timing_self_play_100.log`。
- `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：通过，player 2 获胜，最终生命 0:18；
  控制台输出保存于
  `data/reports/training_speed/stage_2_2_command_timing_rl_mixed.log`。

2.2 worker legal command/action mask 切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。`ShadowverseEnv.step()` 的可选 timing sink
  精确累加执行前合法性校验及状态变化后下一决策两次 `action_mask()`；
  缓存未命中时该调用包含 `_cached_legal_commands()` 的实际生成成本。
- worker 将每局结果汇总为 `worker_action_mask_seconds`；该字段和 command
  decode、resolution 的总和不得超过既有 `worker_engine_step_seconds`。
- 冻结 v4.1 checkpoint 的 2,048-step 实测实际完成 2,282 steps：
  action mask/legal command 0.137650 秒、command decode 0.013936 秒、
  resolution 1.296719 秒，三者均包含在 engine-step 5.413001 秒内；
  checkpoint 哈希保持不变。机器可读结果保存于
  `data/reports/training_speed/stage_2_2_action_mask_smoke.json`。
- 两项聚焦等价/汇总回归通过；最终
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：运行 2,848 项，
  1 项条件跳过、其余通过，耗时 438.597 秒，API test 通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_action_mask_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100`：100 局
  全部完成，`wins=[56, 44]`、draw/truncation/mask mismatch 均为 0；
  `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：通过，player 2 获胜，最终生命 0:18。

2.2 worker Observation v4.1 构造切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。worker 分别统计决策前、`env.step()` 返回值
  及截断 bootstrap 的 Observation 构造时间，并汇总为
  `worker_observation_construction_seconds`；三项分量必须精确等于汇总值。
- 原有 `worker_observation_seconds` 继续表示决策 Observation 构造加
  flatten/card-index/mask NumPy 打包总量，从而保持既有 profiling schema
  兼容，并允许后续独立拆分打包成本。
- 冻结 v4.1 checkpoint 的 2,048-step 实测实际完成 2,282 steps：
  Observation 构造合计 3.955569 秒，其中决策前 0.154314 秒、step 返回值
  3.799565 秒、bootstrap 0.001690 秒；checkpoint 哈希保持不变。结果表明
  首次 step-return 构造是后续 A 类重复转换优化的明确候选，但本切片未改变
  执行路径。机器可读结果保存于
  `data/reports/training_speed/stage_2_2_observation_v4_1_smoke.json`。
- 两项聚焦等价/汇总回归通过；最终
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：运行 2,848 项，
  1 项条件跳过、其余通过，耗时 438.899 秒，API test 通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_observation_v4_1_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
- `E:\anaconda\python.exe -m scripts.random_self_play --games 100`：100 局
  全部完成，`wins=[56, 44]`、draw/truncation/mask mismatch 均为 0；
  `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log`：通过，player 2 获胜，最终生命 0:18。

2.2 worker IPC 请求切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。新增默认关闭的 `profile_ipc_timing`；
  只有 `scripts.profile_ppo_training --profile-ipc-timing` 会把原始策略请求
  使用 `ForkingPickler` 显式序列化成诊断 envelope，正常训练的 Queue
  消息路径不变。
- 序列化时间定义为精确策略请求 tuple 的 `ForkingPickler` 用时；发送时间
  定义为诊断 envelope 提交队列到中央进程取出该 envelope；等待时间定义为
  中央进程取出请求到 worker 收到对应动作。避免把仅写入本地 feeder buffer
  的 `Queue.put()` 返回时间误报为完成发送。
- 固定 seed 回归分别在诊断开关关闭和开启时运行，动作轨迹、log probability、
  value、hidden state、bootstrap 和 PPO generation 边界完全一致；另一个
  真实 v4.1 收集/更新回归验证三段计时均为正、请求数与 records 相等且计时
  和不超过既有 round-trip。
- 冻结 v4.1 checkpoint 的 2,048-step 实测实际完成 2,282 steps 和
  2,282 个请求：序列化 0.156790 秒（0.068707 ms/请求）、发送
  41.995354 秒（18.402872 ms/请求）、等待 51.167325 秒
  （22.422140 ms/请求），worker 推理往返为 93.330320 秒
  （40.898475 ms/请求）；三段解释 99.9884% 的 worker 推理往返。
  平均序列化请求为 73,787 bytes。
- 上述秒数是四个并行 worker 的逐请求累计，不能直接与 collect wall time
  相加比较；诊断运行的 40.240 steps/s 含显式 envelope 成本，不作为优化
  前后吞吐结论。机器可读方法、原始 iteration 和派生指标保存于
  `data/reports/training_speed/stage_2_2_ipc_timing_smoke.json`。
- 测速 checkpoint 的 SHA-256 仍为
  `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd`，
  与冻结基线一致，大小和 mtime 也未改变。
- 两项聚焦等价/汇总回归通过；最终
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：运行
  2,849 项，1 项条件跳过、其余通过，耗时 438.102 秒，API test 通过；
  完整输出保存于
  `data/reports/training_speed/stage_2_2_ipc_timing_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。

2.2 worker 对局生命周期切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。每局分别统计固定/采样卡组构造、
  `ShadowverseEnv` 构造、`reset()` 和 mulligan 阶段的完整 `env.step()`；
  换牌步数必须与实际进入 mulligan 的动作数对应，不从对局总 setup
  时间反推。
- worker 空闲拆成等待下一局分配及等待中央策略动作返回；空闲比例以
  `episode total + assignment wait` 为观察窗口，避免把四个并行 worker
  的累计秒数误当 wall time。每局步数保存为直方图并派生 mean、P50、
  P95、最小/最大值；长局定义为达到该局 agent-step 上限的 75%，另保存
  terminated/truncated 计数。
- 冻结 v4.1 checkpoint 的两次 update 实测完成 4,434 steps、56 局：
  卡组构造累计 0.004258 秒、reset 0.082499 秒、112 个 mulligan 动作
  0.183858 秒。两次 update 的局长 P50 为 69/74、P95 为 114/102，
  最大 256；55 局正常结束、1 局截断，1 局达到 192-step 长局阈值。
- 四 worker 累计等待 episode 分配 121.658235 秒、等待中央动作返回
  185.977275 秒；两个 update 的 worker 空闲比例为 0.9515/0.9720，
  表明 reset、换牌和建牌堆不是当前主瓶颈，主要空闲来自中央推理等待及
  同步 learner update。机器可读原始 iteration、直方图和汇总保存在
  `data/reports/training_speed/stage_2_2_worker_lifecycle_smoke.json`。
- 测速 checkpoint 大小和 mtime 保持不变；两项聚焦等价/汇总回归通过。
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：2,849 项
  通过、1 项条件跳过，耗时 439.799 秒，API test 通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_worker_lifecycle_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
  本切片只增加 PPO profiling 字段，不改变引擎、规则、动作、目标、战斗、
  回合或 Observation，因此未触发额外 self-play/`rl_mixed_match`。

2.2 中央推理切片证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。新增默认关闭的
  `--profile-central-timing`；关闭时不启用逐组件 CUDA event 和同步，
  开启时将 request 在中央出队的时间戳附加到本地 metadata，不改变
  worker 传输的策略输入。固定 seed 回归覆盖关闭、IPC-only、
  central-only，并另以 v4.1 entity-action 模型验证动作、log probability、
  value、hidden state、bootstrap 和 PPO generation 边界完全一致。
- 请求进入中央调度器到 batch 关闭累计等待 0.236416 秒，即
  0.103600 ms/请求；等待下一条 worker 消息 1.049764 秒，配置的合批窗口
  累计等待 0.151195 秒。该口径从中央实际 dequeue 开始，不重复计算
  worker→Queue 的 IPC 发送时间。
- 2,282 个请求组成 1,382 个推理 batch，batch 1/2/3/4 的数量分别为
  844/178/358/2，平均 1.651、P50 1、P95 3；5,528 个容量槽中
  3,246 个为空，空槽比例 58.719%。该分布来自诊断运行的全部 batch，
  不是用平均值反推。
- CPU 对 observation/card-index/mask 执行三次 `np.stack`：累计
  0.076359 秒、复制 167,635,720 bytes（73,460 bytes/请求）；
  CPU tensor view/dtype 构造 0.026750 秒，hidden-state 拼装
  0.089399 秒。CPU 拼装和结构化复制本身不是当前主耗时。
- H2D wall/CUDA-event 时间分别为 0.470165/0.450637 秒。模型完整前向
  CUDA 时间 28.395150 秒，其中进入 Transformer 前的 v4.1
  token/embedding/projection 构造 19.420124 秒（68.4%），Transformer
  3.009437 秒、Transformer→GRU 0.058437 秒、GRU 0.208032 秒、
  action/value 阶段 5.699119 秒；其中 policy/value head 模块自身分别
  0.322723/0.244109 秒。
- masked logits + softmax/log-softmax 的 wall/CUDA 时间为
  0.505175/0.436018 秒；D2H wall/CUDA 为 0.268035/0.207218 秒，
  CPU multinomial 采样 0.185320 秒。record packaging 与 response
  Queue 分发合计 0.165865 秒。
- CUDA-event 忙碌阶段累计 29.489024 秒；中央阻塞等待 worker 加合批
  等待为 1.200959 秒，在两者之和中的诊断比例为 96.087%/3.913%。
  该比例不等同于系统 GPU utilization，也不把 CPU packaging 或 Python
  调度伪装成 GPU busy；结果说明本次运行的首要中央成本是模型输入编码，
  同时 58.7% batch 空槽仍是后续合批优化机会。
- 冻结 v4.1 checkpoint 的 2,048-step 请求实际完成 2,282 steps；
  前后 SHA-256 均为
  `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd`。
  诊断运行吞吐 39.519 steps/s，包含逐组件同步开销，不能作为优化前后
  吞吐结论。机器可读方法、原始 iteration、batch 直方图及全部分段保存于
  `data/reports/training_speed/stage_2_2_central_inference_smoke.json`。
- v4.1 等价、开关组合和保存报告不变量共 5 项聚焦测试通过；最终
  `E:\anaconda\python.exe -m unittest discover -s tests -v`：2,853 项
  通过、1 项条件跳过，耗时 448.799 秒，API test 通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_central_inference_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
  本切片只增加默认关闭的 PPO profiling，不修改规则、动作、Observation
  含义或奖励，因此未触发额外 self-play/`rl_mixed_match`。

2.2 Learner 与总体验收证据（2026-07-31）：

- 候选分类为 `A-PROFILE-001`。新增默认关闭的
  `--profile-learner-timing`；开启时用 CUDA event 统计异步 GPU 阶段，
  只在现有 update 边界统一同步，不在组件之间额外同步。固定 checkpoint、
  rollout 和随机状态回归证明开关关闭/开启时动作、log probability、value、
  hidden state、bootstrap、PPO generation 边界、指标和更新后参数完全一致。
- 冻结 v4.1 checkpoint 的 6,144-step 请求实际完成 6,497 steps、3 个
  update，单次 update 为 19.742304 / 19.710845 / 19.544825 秒；
  checkpoint 前后 SHA-256 均为
  `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd`。
- 3 个 update 的有效 token/容量槽分别为 1,952/2,624、
  2,054/3,072、1,906/2,688，padding 比例为 25.610% / 33.138% /
  29.092%；同时保存 minibatch 有效 token 的 P50/P95/min/max 和每 epoch
  均值，不用 batch 数或平均序列长度反推。
- Learner 三次合计：padding/NumPy 0.148415 秒、CPU tensor 构造
  0.001799 秒、H2D 0.118101 秒、前向 CUDA 29.369683 秒、loss
  0.338888 秒、反向 CUDA 27.900011 秒、梯度裁剪 0.097853 秒、
  optimizer 0.168695 秒。前向与反向是本次诊断的主成本；这些诊断数据
  不单独构成后续优化收益结论。
- CUDA event 与 host launch 分开保存；已知会同步的 loss/grad/参数校验和
  显式 optimizer 同步共计约 0.77 秒，单次 update 为 0.25--0.26 秒，
  避免把异步 launch 时间误当成 GPU 执行时间。机器可读原始 iteration、
  token、同步点和方法保存于
  `data/reports/training_speed/stage_2_2_learner_timing_smoke.json`。
- 补齐中央 rollout 启动、collection setup/finalize、episode completion
  和 model restore 后，互斥阶段对 wall time 的解释率为：完整 pipeline
  99.883%、collect 99.855%、update 99.926%，均超过 90%。每个规范化阶段
  均保存总耗时、每 agent-step 毫秒、wall time 占比、median 和 P95。
- 关闭全部 profiling 的独立 10,000-step 守卫运行排除前 2 个预热 update，
  正式统计 3 个 update 的吞吐为 43.364 / 44.460 / 44.439 steps/s，
  median 44.439；相对 2.1 同配置基线 median 44.705 低 0.595%，处于预设
  2% 容差内。该短实验只证明默认关闭路径没有明显 profiling 开销，不作为
  性能优化前后结论。原始数据保存于
  `data/reports/training_speed/stage_2_2_profiling_disabled_smoke.json`。
- 汇总、空样本、互斥阶段求和、字段完整性、三稳态样本和 2% 守卫均有
  自动测试；`E:\anaconda\python.exe -m unittest tests.test_rl_profiling
  tests.test_training_speed_stage_2_2
  tests.test_ppo.PPOTrainerTests.test_learner_profile_preserves_update_results
  -v`：11 项通过。
- 独立验收器和来源 SHA-256 结果保存于
  `data/reports/training_speed/stage_2_2_acceptance.json`，总结果
  `passed=true`。最终 `E:\anaconda\python.exe -m unittest discover -s
  tests -v`：2,861 项通过、1 项条件跳过，耗时 452.726 秒，API test
  通过；完整输出保存于
  `data/reports/training_speed/stage_2_2_complete_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
  本切片只增加默认关闭的 PPO profiling、统计和报告，不修改引擎、规则、
  动作、目标、战斗、回合、Observation 含义或奖励，因此未触发额外
  self-play/`rl_mixed_match`。

总体验收：

- [x] 各互斥阶段耗时之和能够解释至少 90% 的 wall time。
- [x] 报告同时给出总耗时、每步毫秒、占比、median 和 P95。
- [x] 计时开关关闭时不明显降低正式训练吞吐。
- [x] 为汇总计算、空样本和阶段求和增加测试。

## 2.3 确定 v4.1 的可操作瓶颈（v3.6 仅作轻量参照）

本节服务于 v4.1 优化路线选择，不为 v3.6 重做完整的 Worker、IPC、中央调度、
Learner 或对局生命周期剖析，也不优化 v3.6。v3.6 只在相同固定合成输入下
提供纯前向参照，用于区分 v4.1 特有成本与两代模型共有成本；所有候选排序、
Profiler 分析和后续优化均以 v4.1 为主。

- [x] 用固定合成输入完整测量 v4.1 batch 1/4/8/16/32/64 的纯前向
  吞吐，并用相同 batch 和测量口径生成 v3.6 轻量纯前向参照。
- [x] 分离非卡数值投影、卡牌 embedding、93 个语义 token 构造、
  Transformer、GRU 和 action head。
- [x] 测量 token 打包在 CPU 和 GPU 上的时间及拷贝量。
- [x] 测量不同长度 episode 的 GRU recurrent state 管理成本。
- [x] 测量合法动作候选数量对 action-conditioned scoring 的影响。
- [x] 使用 PyTorch Profiler 确定主要 CUDA kernel、launch gap 和同步点。
- [x] 检查是否存在重复 Observation 转换、重复 embedding 或重复 mask 拷贝。
- [x] 形成按预计收益排序的瓶颈清单，不凭 GPU/CPU 方波猜测原因。

产物：

- `data/reports/training_speed/v4_1_inference_breakdown.json`
- `docs/training_speed_bottleneck_report.md`

2.3 精简版诊断证据（2026-07-31）：

- 候选分类为 `A-PROFILE-002`。v4.1 是唯一优化目标；v3.6 只运行相同
  fixture seed、batch、预热、迭代和 CUDA-event 口径的纯前向参照，没有
  重做 Worker、IPC、中央调度、Learner 或生命周期剖析。
- 实际运行 `E:\anaconda\python.exe -m
  scripts.profile_v4_1_inference_breakdown`。固定合成输入按 checkpoint
  shape/dtype 生成一次并按 batch 前缀切片；每个 batch 预热 8 次，正式段
  每个 repeat 运行 20 个 forward，独立重复 3 次。
- v4.1 batch 1/4/8/16/32/64 的纯前向 median 为
  21.270/22.013/21.865/21.825/22.060/21.613 ms；v3.6 为
  5.861/5.794/5.733/5.823/5.852/5.829 ms。v4.1 延迟为 v3.6 的
  3.63--3.81 倍，但几乎不随 batch 增长；batch 4 样本吞吐为 batch 1
  的 3.86 倍，与 2.2 的 58.719% 中央空槽共同支持后续合批扫描。
- 组件 hooks 会使 batch 4 完整 forward 从无 hooks 的 22.013 ms 增至
  31.062 ms，因此只解释结构、不替代纯前向绝对值。hooked median 中
  93-token 构造 20.550 ms（66.16%）、Transformer 4.165 ms、GRU
  0.143 ms、action/value 6.214 ms；policy/value head 容器自身仅
  0.199/0.294 ms。card embedding lookup/projection 各约 0.09 ms；
  token tensor op、其他 embedding、重复静态构造与 launch gap 是更大的
  可操作范围。
- 每请求输入 73,460 bytes。batch 1 的 NumPy stack/CPU tensor/H2D
  median 为 0.020/0.009/0.105 ms；batch 64 为
  1.054/0.013/0.429 ms。小 batch 输入复制不是首要成本，扩大合批后必须
  同时验证预分配和缓冲区复用。
- 固定 batch 4、总计 512 recurrent steps 时，episode 长度
  1/16/64/256 的 GRU device median 为
  0.1324/0.1077/0.1032/0.1036 ms/step；除极短 episode reset 外无明显
  长度放大。合法动作 1/8/32/64/112 的 masked distribution 为
  0.2100/0.1919/0.2195/0.2183/0.2129 ms，处于波动范围；模型始终先密集
  计算 112 个 logits，合法候选数不改变 action scoring。
- PyTorch Profiler 在 batch 4 的 3 个 forward 捕获 1,938 个 kernel 和
  1,938 次 launch，即每 forward 646 个；kernel gap median/P95 为
  44.13/130.33 µs，并记录 11 个同步事件。trace 表明成本分散在许多小
  GEMM、copy、add、gather、round、clamp、reduce 和 elementwise kernel，
  不存在一个足以单独解释 wall time 的大 kernel。
- 静态与运行时证据确认：中央 Worker 没有复用 `env.step()` 已构造的下一
  Observation；每 batch 重建三份 `np.stack` 并重新 H2D；语义 context
  重建 `torch.arange(4)`；device tensor 的 Python `bool` 校验形成 host
  sync 候选。同一 forward 内 card embedding/projection 只执行一次并复用，
  未发现重复 card embedding。
- 按可独立获得收益和依赖关系排序：v4.1 token/launch/sync 共同根因（2.6）、
  中央合批与缓冲区（2.4）、不与共同根因重复计数的 Learner 专属工作（2.7）、
  下一 Observation 重复构造（2.5）。完整口径、三重复原始样本、checkpoint
  SHA-256、operator、kernel、gap、同步和审计位置保存于
  `data/reports/training_speed/v4_1_inference_breakdown.json`；压缩 trace
  保存于 `data/reports/training_speed/v4_1_profiler_trace.json.gz`，
  结论和限制保存于 `docs/training_speed_bottleneck_report.md`。
- `E:\anaconda\python.exe -m unittest
  tests.test_training_speed_stage_2_3 -v`：5 项通过。
- `E:\anaconda\python.exe -m unittest discover -s tests -v`：共运行
  2,866 项，1 项条件跳过、其余通过，耗时 454.194 秒，API test 通过；
  完整输出保存于
  `data/reports/training_speed/stage_2_3_complete_unittest.log`。
- `E:\anaconda\python.exe -m compileall -q swb scripts tests`：通过。
  本阶段只新增只读诊断脚本、报告和测试，不修改模型、PPO、引擎、规则、
  动作、目标、战斗、回合或 Observation 语义，因此未触发额外
  self-play/`rl_mixed_match`。

阶段 2.3 的“根因优先级”和后续 checklist 的“实际执行顺序”分开记录：
2.6 的 token/launch/sync 是当前最重的共同根因，但仍按 2.4 → 2.5 → 2.6
推进。2.4 先完成配置扫描并过决策门；2.5 先完成一个已确认重复工作的最小
切片并过决策门；只有门槛内有真实收益或仍有足够占比，才扩展同节的后续
实现。未过门的候选要保存结果并以“有证据的不适用/延期”关闭，不为勾选而
实现。纯前向或组件 microbenchmark 只用于定位，不能代替端到端验收。

## 2.4 扫描中央 GPU 推理合批并设置实施决策门

### 2.4A 配置扫描（必须先完成）

- [x] 固定 checkpoint、Observation、网络、卡组、seed、训练参数和硬件，
  保存未调参基线；所有进入比较的配置至少端到端重复三次。
- [x] 单变量扫描 batch wait：0、0.1、0.25、0.5、1.0 ms。
- [x] 单变量扫描稳定 worker 数：2/3/4/5/6；避免直接重试已发生分页压力的
  8 worker 配置。
- [x] 单变量扫描每 worker PyTorch 线程数：1/2/4。
- [x] 第一轮保持其他参数为基线；只对单变量胜出且稳定的值补做交互组合，
  避免没有归因能力的全笛卡尔积。
- [x] 每组记录 batch 大小 mean/P50/P95、空槽率、worker 等待、GPU 空闲、
  median steps/s、P95 stage time、回合长度、CPU/GPU/RAM 和异常退出。
- [x] 区分“等待窗口太短”和“请求到达率不足”；不能只看到平均 batch 偏小
  就直接增加等待时间或 worker。
- [x] 保持每个 episode 独立 recurrent state 和独立 policy RNG。
- [x] 保持一个 rollout generation 内权重固定，不引入策略滞后。
- [x] 验证 batch 内请求排序变化不会把 hidden state 或动作发给错误 episode。
- [x] 完成 2.4 决策门：只有最佳稳定配置相对基线的 median 端到端吞吐提升
  至少 5%，且提升超过三次运行波动范围，才继续 2.4B；否则保存“无明确
  收益”报告，将 2.4B 以有证据的不适用/延期关闭并继续 2.5。

2.4A 单变量扫描证据（2026-07-31）：

- 使用冻结 v4.1 checkpoint
  `4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd`；
  11 个去重配置各端到端运行三次，每次排除 2 个 warm-up update，并保留
  至少 3 个 steady update/6,144 agent steps。33/33 份报告的 checkpoint、
  runtime 配置、监控样本和异常退出校验全部通过。
- 冻结 100k 基线 median 为 `44.7054 steps/s`，三次相对 range 为
  `0.3601%`。单变量最佳值：`wait=1.0ms` median `59.3714`
  （`+32.80%`）、`workers=6` median `51.7712`（`+15.80%`）、
  `workers=5` median `48.8632`（`+9.30%`）；三者均同时超过 5% 门槛和
  三次基线波动。`threads=1/4` median 为 `44.0137/44.1724`，无明确收益。
- `wait=0ms` 将 mean/P50/P95 batch 降为 `1/1/1`，median 吞吐
  `31.1002`；`wait=1.0ms` 将 mean/P95 batch 提高到 `2.48/4`、空槽率
  降至 `37.99%`。worker 扫描从 2 到 6 的 median 吞吐依次为
  `31.6352/38.9961/44.1885/48.8632/51.7712`，证明当前同时受等待窗口
  和请求到达/worker 上限约束，而非单纯无效等待。
- episode 独立 hidden state/RNG、rollout generation 权重冻结和请求
  episode/order 对应关系沿用中央策略 rollout 的既有契约，并由
  `tests.test_ppo`、`tests.test_vector_rollout` 与新扫描配置完整性测试继续
  覆盖；本候选没有改变该执行路径。
- 机器可读汇总为
  `data/reports/training_speed/stage_2_4_scan.json`，33 份逐 run 证据及日志
  位于 `data/reports/training_speed/stage_2_4_runs/`；可恢复扫描器为
  `scripts/scan_training_speed_stage_2_4.py`。2.4 决策门通过，按要求进入
  仅包含稳定 winner 的交互验证。

### 2.4B 实现候选（仅在 2.4 决策门通过后）

- [x] A 类：减少单步消息数量，在不改变决策边界的情况下批量传输固定字段。
- [x] A 类：复用 observation/card-index/action-mask 的 batch 缓冲区，减少
  已确认的三次 `np.stack`、Python 对象和 Tensor 重建。
- [x] A 类：仅在 H2D 占比和端到端实验支持时采用 pinned memory 与
  non-blocking H2D；当前 batch 1 的约 0.105 ms H2D 不能单独构成立项理由。
- [x] 每项候选独立实现、独立提交并至少做三次同配置端到端对比，不能合并
  多项后再倒推收益。

2.4B 稳定 winner 交互证据（2026-07-31）：

- 只把 worker 维度的稳定 winner `5/6` 分别与 wait 维度 winner `1.0ms`
  组合；线程维度没有稳定 winner，未做无归因能力的全笛卡尔积。两个组合
  各运行三次，继续使用冻结 checkpoint、2 warm-up update 与至少
  6,144 steady steps。
- `workers=5, wait=1.0ms` 三次为
  `63.8337/63.7021/64.3763 steps/s`，median `63.8337`；
  `workers=6, wait=1.0ms` 三次为
  `63.6191/64.8932/64.4729`，median `64.4729`。后者相对冻结基线
  `+44.22%`，相对最强单变量 `wait=1.0ms` 再提升 `+8.59%`，因此采用
  `6 workers / 2 threads / 1.0ms wait` 作为后续性能候选的运行时配置。
- 采用组合的 mean/P95 batch 为 `3.10/6`，median CPU 总利用率约
  `9.59%`；三次均无异常退出、监控缺失、checkpoint 变化或 pagefile
  压力异常。机器可读报告为
  `data/reports/training_speed/stage_2_4_b_interactions.json`，逐 run 证据
  位于 `data/reports/training_speed/stage_2_4_b_interaction_runs/`。

2.4B 实现候选适用性结论（2026-07-31）：

- 当前每个决策已经把 episode/generation/step/player、Observation、card
  index 与 action mask 合在一条 request 中，并只返回一条 action response；
  request queue put 为 `0.0104 ms/request`，response wait 为
  `42.5090 ms/request`。在不预取未来状态、不增加每 worker 多环境且不改变
  决策边界的前提下，消息数不能再低于一来一回，故该实现候选以
  `closed_already_minimal_without_semantic_change` 关闭。
- adopted batch bucket 附近的 batch 4 实测：三次 `np.stack` 为
  `0.03233 ms/batch`，CPU Tensor 构造 `0.01073 ms`，H2D
  `0.12218 ms`，设备 forward `22.01325 ms`。完全消除打包的理论上限仅为
  forward 的 `0.196%`；连同完全消除 H2D 的理想上限也仅 `0.751%`，
  低于 5% materiality gate 和端到端波动。因此复用 batch buffer 与
  pinned/non-blocking H2D 分别以 `closed_below_materiality_gate` 关闭，
  未引入复杂生命周期/同步实现，也不声称其已优化。
- “每项独立实现、独立三次比较”只约束进入实现的适用候选；本轮三个候选
  分别通过独立 gate 后均不适用。真正采用的 batch wait 与 worker 组合已
  独立完成三次比较并分别建立 checkpoint，没有把低上限候选混入后倒推
  收益。机器可读 gate 与来源 SHA-256 保存于
  `data/reports/training_speed/stage_2_4_acceptance.json`。
- 阶段验收：
  `E:\anaconda\python.exe -m unittest discover -s tests -v` 通过
  `2875` tests（`1` skip）；`E:\anaconda\python.exe -m compileall -q swb
  scripts tests` 通过；`E:\anaconda\python.exe -m scripts.rl_mixed_match
  --output data/rl_mixed_match.log` 通过，player 2 获胜、最终生命
  `0:18`。本阶段未改变规则、合法动作、Observation 或引擎语义，故未额外
  触发 100-game self-play 要求。

## 2.5 先消除重复 Observation 构造，再决定是否扩展

- [x] A-OBS-001：先保存最小调用轨迹和等价测试，证明中央 Worker 在
  `env.step()` 已返回下一 Observation 后又调用 `env.observation()`。
- [x] A-OBS-001：复用 `env.step()` 返回的下一 Observation，终止、截断、
  reset 和 perspective 切换边界保持原行为；同一决策状态的 Observation
  和 action mask 只构造一次。
- [x] A-OBS-001：固定 seed 对比 Observation 全字段/字节、card index、
  action mask、动作、log probability、value、hidden state、trajectory 和
  PPO generation 边界。
- [x] 分别测量 cold/cached Observation、完整环境 step 和三次端到端 PPO；
  只优化 microbenchmark 不算完成。
- [x] 完成 2.5 决策门：若 Observation 构造仍占端到端 pipeline wall time
  至少 5%，或 A-OBS-001 收益明确超过运行波动且剖析显示还有同源成本，
  才继续本节其余候选；否则保存结果并将其余候选以有证据的不适用/延期关闭。
- [x] A 类：按 vocabulary 预计算卡牌静态字段，只查询运行时变化字段。
- [x] A 类：将固定 93-token 布局直接写入连续数组，减少字典和临时列表。
- [x] A 类：预分配非卡数值、卡牌索引和 token feature 缓冲区。
- [x] A 类：避免在 worker 与中央推理之间传输 policy 未使用的调试字段。
- [x] A 类：检查 dtype，避免 int64/float64 的不必要带宽和隐式转换。
- [x] 保留隐藏信息、perspective、顺序不变性和 mask 一致性测试，优化不能
  改变 Observation v4.1 语义。

2.5 决策门证据（2026-07-31）：

- `A-OBS-001` 的最小调用轨迹确认中央 worker 在 `env.reset()`/`env.step()`
  已返回当前决策需要的 Observation 后，又以相同 state version、
  perspective 和 action mask 调用 `env.observation()`；缓存命中仍会产生
  一次深拷贝。`swb/rl/vector_rollout.py` 现在直接保留 reset 返回值，并在
  每次 step 后把 `StepResult.observation` 传给下一决策。终止/截断 bootstrap
  路径未改；decision 侧重复构造计时由冻结前 6,316 steps 的
  `0.447265` 秒变为三次正式运行均精确 `0`。
- 固定 seed 中央 rollout 契约测试逐字段比较 observation、card indices、
  action mask、动作、log probability、value、hidden state、完整 trajectory
  和 PPO generation；另有 entity-action 路径与持久多进程训练回归。
  Observation v4.1 的隐藏信息、perspective、顺序和 mask 语义均未改变。
- 环境 microbenchmark（seed `20260801`）为 cached Observation
  `8,096.052/s`、cold Observation `545.532/s`、缓存加速 `14.841x`，
  完整环境 step `259.434/s`，通过既有阈值。结果保存于
  `data/reports/training_speed/stage_2_5_environment_benchmark.json`。
- 采用 2.4 的 `6 workers / 2 threads / 1.0 ms` 配置、相同冻结 checkpoint
  运行三次，每次排除两个 warm-up update 并统计至少三个 steady update：
  `63.463 / 64.402 / 64.303` agent steps/s，中位数 `64.303`。相对 2.4
  同配置中位数 `64.473` 为 `-0.264%`，未超过其三次相对波动范围
  `1.976%`；三次均为零异常退出、关闭可选 profiling、checkpoint 哈希
  不变。
- 冻结前 Observation worker 计时总和为 `11.591003` 秒；按 4 个同时运行
  的 worker 对 `143.178` 秒 pipeline wall 归一后占 `2.024%`，低于
  `5%` 门槛。未除并发度的串行 worker 计时和为 `8.095%`，只作诊断，
  不当作可串行消除的 pipeline wall；其中每步必需的 `env.step()`
  Observation 构造也不是重复工作。
- 因占比门槛和端到端波动门槛均未通过，本节其余五项候选按预先声明的
  决策门关闭为 `closed_below_materiality_and_variability_gate`，并非声称
  已实现：静态卡牌字段预计算、固定 token 连续写入、Observation 缓冲区
  预分配、删除 policy 未使用的 IPC 调试字段，以及 dtype 收窄/转换清理。
  原始三次报告、门槛、候选处置和所有来源 SHA-256 保存于
  `data/reports/training_speed/stage_2_5_a_obs_001.json`。
- 阶段验收：
  `E:\anaconda\python.exe -m unittest discover -s tests -v` 通过
  `2,878` tests（`1` skip），耗时 `452.451` 秒，API test 通过；
  `E:\anaconda\python.exe -m compileall -q swb scripts tests` 通过；
  `E:\anaconda\python.exe -m scripts.random_self_play --games 100` 通过，
  `wins=[56, 44]`、draw/truncation/mask mismatch 均为 `0`；
  `E:\anaconda\python.exe -m scripts.rl_mixed_match --output
  data/rl_mixed_match.log` 通过，player 2 获胜、最终生命 `0:18`。

## 2.6 优先优化 v4.1 token/launch/sync 热路径

A 类候选按以下顺序实施：

- [x] A-NET-001：在不削弱非法输入拒绝和原子性的前提下，将 device tensor
  的 Python `bool` 校验移出每次 forward 的 GPU 热路径，或合并为无需逐
  forward host sync 的等价门禁。
- [x] A-NET-002：将 `torch.arange(4)`、固定位置和其他不变量注册为静态
  buffer，避免每次 forward 重新创建。
- [ ] A-NET-003：在逐字段语义等价的前提下，合并重复的
  `round → long → clamp` 和 semantic-context 小算子。
- [ ] 调整固定输入的连续内存布局，并合并可等价合并的小 projection/
  elementwise 操作，减少 `permute/contiguous` 和 CUDA kernel launch。
- [ ] 每个候选分别记录 batch 1/2/4/8/16/32/64 纯前向、组件时间、kernel
  数、kernel gap、同步事件及三次端到端结果；以 batch 4 当前每 forward
  646 次 launch 和 11 个同步事件作为诊断参照。
- [ ] 在 token 热路径完成后再评估原生 scaled-dot-product attention；
  当前 hooked Transformer 约 4.17 ms，不把 attention 当作唯一主因。
- [ ] 仅在 profiler 证明超过噪声后评估推理侧静态卡牌编码缓存；缓存必须按
  policy generation 失效，且当前 card lookup/projection 各约 0.09 ms、
  同一 forward 无重复 card embedding 的证据必须写入决策。
- [ ] 只有 host sync 已处理且 batch bucket 稳定后才评估 CUDA Graph；
  动态尾 batch 保留普通路径。
- [ ] 每项 A 类改动验证相同输入的 logits/value/hidden state 精确或在既定
  浮点容差内一致，并通过固定 seed 轨迹、log probability、PPO generation
  和 checkpoint resume 等价测试。

B 类候选：

- [ ] 优先评估 `torch.compile` 的首次编译成本、稳态收益、graph break、
  checkpoint 兼容性和 Windows 稳定性；646 次 launch 使其值得较早测量，
  但在稳定性和数值证据完成前仍属于 B 类。
- [ ] 分别测试 TF32、FP16 autocast 和 BF16 autocast。
- [ ] 检查 masked logits、softmax、log probability 和 value 是否出现
  NaN/Inf。
- [ ] 检查不同精度/编译路径下动作概率误差、argmax 翻转率和 recurrent
  state 漂移。
- [ ] B 类只有在数值稳定、长局和三 seed 小规模学习实验不退化后才能成为
  默认值。

2.6 A-NET-001 负结果（2026-07-31）：

- 实验将 card-index 范围与每个 live action-mask row 的合法动作存在性移到
  CPU/H2D 边界验证；中央推理和 learner recurrent loop 只在通过该门禁后
  使用显式 prevalidated 路径。公开 `forward_step()`/`masked_logits()`
  默认仍执行原校验，负索引、越界索引、空 mask 与形状不匹配继续抛错。
- 固定输入的 logits、value 和 hidden state 逐 bit 相同。batch
  `1/2/4/8/16/32/64` 均保存三重复纯前向与组件计时；batch 4 device
  forward 从阶段 2.3 的 `22.013` ms 降至 `21.573` ms（`2.00%`），但
  其他共同 batch 的变化落在 `-0.59%` 到 `+1.75%`。三次 batch-4 forward
  的 profiler kernel 从 `1,938` 降至 `1,926`，同步事件从 `11` 降至
  `5`，说明技术机制生效但 micro 改善较小。
- 相同冻结 checkpoint、`6 workers / 2 threads / 1.0 ms`、排除两个
  warm-up update 后三次端到端为 `63.890 / 65.151 / 64.911` steps/s，
  中位数 `64.911`。相对紧邻 2.5 中位数 `64.303` 仅 `+0.946%`，低于
  2.5 三次相对 range `1.461%`，故按统一门槛判定为
  `rejected_gain_within_run_variability`，不保留为默认实现。
- 机器可读 micro、profiler trace、三次原始端到端报告与汇总分别保存于
  `data/reports/training_speed/stage_2_6_a_net_001_micro.json`、
  `data/reports/training_speed/stage_2_6_a_net_001_trace.json.gz`、
  `data/reports/training_speed/stage_2_6_a_net_001_runs/` 和
  `data/reports/training_speed/stage_2_6_a_net_001.json`。

2.6 A-NET-002 负结果（2026-07-31）：

- 实验将 semantic byte、player relation、leader slot、zone、history 和
  record 的 7 组固定位置注册为 `persistent=False` buffer；它们随模型迁移到
  `cuda:0`，但不进入 checkpoint state dict，旧冻结 checkpoint 可严格加载。
- 改动前 reference 与改动后固定输入的 logits、value、hidden state
  SHA-256 逐项相同。batch `1/4/8/16/32/64` 的纯前向相对阶段 2.3
  分别改善约 `2.4% / 2.6% / 1.2% / 1.9% / 0.9% / 2.7%`；batch 4
  从 `22.013` ms 降至 `21.433` ms。三次 batch-4 forward 的 profiler
  kernel 从 `1,938` 降至 `1,902`，同步事件为 `8`，说明固定 tensor
  分配确实从热路径消失。
- 相同冻结 checkpoint、`6 workers / 2 threads / 1.0 ms` 的三次端到端为
  `64.237 / 64.477 / 65.177` steps/s，中位数 `64.477`。相对紧邻 2.5
  中位数仅 `+0.271%`，低于既定 `1.461%` 波动门槛，故判定为
  `rejected_gain_within_run_variability`，不保留为默认实现。
- 改动前 reference、micro、profiler trace、三次原始报告与汇总分别保存于
  `data/reports/training_speed/stage_2_6_a_net_002_reference.json`、
  `data/reports/training_speed/stage_2_6_a_net_002_micro.json`、
  `data/reports/training_speed/stage_2_6_a_net_002_trace.json.gz`、
  `data/reports/training_speed/stage_2_6_a_net_002_runs/` 和
  `data/reports/training_speed/stage_2_6_a_net_002.json`。

## 2.7 在继承网络收益后优化 learner 更新

- [ ] 采用 2.6 候选后重新建立 learner 分段基线，区分共享模型
  forward/backward 收益与 learner 专属收益，禁止重复计数。
- [ ] 复核 backward、gradient clipping、forward/loss、padding 准备和实际
  padded compute 占比；当前约 97.1% 的 forward+backward 是入口，不证明
  NumPy/H2D 准备本身是主瓶颈。
- [ ] 统计每个 minibatch 的有效 token、padding token 和实际计算比例，
  区分“padding 准备耗时很小”和“padding 后无效 GPU 计算可能较大”。
- [ ] A 类：复用 rollout tensor 缓冲区，将可提前完成的 padding/mask
  计算移出 minibatch 内循环；分别验证分配减少和端到端收益。
- [ ] A 类：检查梯度清零方式、fused optimizer 和 foreach gradient
  clipping，逐项保存反向数值和端到端结果。
- [ ] B 类：测试 AMP + GradScaler 的数值稳定性和真实端到端收益。
- [ ] C 类：任何改变 minibatch、epoch、sequence length、rollout 长度、
  采样顺序或梯度累积语义的方案，都单独保存超参数与三 seed 学习有效性
  实验，不伪装成纯实现优化。

## 2.8 有条件地评估流水线重叠和策略滞后

- [ ] 在 2.4–2.7 已采用候选上重新剖析；只有可调度的 CPU 准备、H2D 或
  pipeline 空洞占 wall time 至少 5%，且理论上能与 CUDA 工作重叠，才进入
  实现；否则保存“无明确收益/延期”证据并关闭本节同步候选。
- [ ] A 类：评估同一 rollout 内 CPU 准备、H2D 和 CUDA 前向的安全重叠；
  当前 batch 1 H2D 约 0.105 ms，不能只凭利用率曲线实施异步拷贝。
- [ ] A 类：评估 learner update 中下一 minibatch 准备与当前 CUDA 计算的
  重叠。
- [ ] 同步重叠不得改变 worker 权重版本、请求顺序、RNG 消耗、hidden state
  所属 episode 或 PPO generation 边界。
- [ ] 若考虑 actor/learner 异步，明确记录 policy generation、最大 lag 和
  每条 trajectory 的行为策略 log probability。
- [ ] 异步方案列为 C 类；必须重新证明 PPO ratio、clip 和 on-policy 边界
  合理，并与同步方案做至少三 seed 学习曲线和固定评估，不能只比较 steps/s。
- [ ] 在同步 A/B 类收益耗尽前，不优先实现异步或分布式 learner。

## 2.9 每项性能候选的统一验收

- [ ] 使用同一冻结规则提交、checkpoint、Observation、网络、卡组、seed、
  worker、训练参数和硬件环境，基线与候选各至少端到端运行三次。
- [ ] 报告 median steps/s、P95 stage time、GPU/CPU/RAM、batch 分布和回合
  长度；按候选补充 Observation 时间、kernel 数/gap/同步或 padding 比例。
- [ ] 单次短 benchmark、组件 microbenchmark 或 CPU/GPU 利用率曲线只能
  定位，不能作为采用结论。
- [ ] 提升小于运行波动范围或对应决策门的候选判定为无明确收益。
- [ ] A 类候选必须通过固定 seed 轨迹、log probability、value、hidden
  state、PPO generation 和 checkpoint resume 等价测试。
- [ ] B 类候选必须通过数值容差、NaN/Inf、长局和小规模学习测试。
- [ ] C 类候选必须通过至少三 seed 的学习有效性与固定对阵强度实验。
- [ ] 所有候选必须保持零非法动作、零 mask mismatch、零 worker 残留。
- [ ] 所有候选必须通过完整单元测试、compileall 和规定 smoke。
- [ ] 保存采用、失败、无明显收益和因门槛延期候选的机器可读原始数据，
  避免以后重复试验。

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
