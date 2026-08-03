# 卡牌 Bug 审计最终报告

审计日期：2026-07-31

## 结论

完整卡池门禁 9/9 通过。735 张可收集卡和 91 张衍生卡均保留逐条审计证据；P0/P1 未关闭 Bug 为 0。
当前新采样训练池为 734 张可收集卡；《帕梅拉的舞蹈》因 `SWB-RULING-SET-STATS-TEMP-001` 尚无直接官方裁定而被显式排除，但仍可解析、审计和历史回放。

## 门禁结果

| 门禁 | 结果 | 关键指标 |
|---|---|---|
| 1.15.1 735 collectible clause audit | passed | covered_exact=735, mapped_exact=735, blocker_count=0 |
| 1.15.2 91 generated-card audit | passed | total=91, complete=91 |
| 1.15.3 full-pool mechanism matrices | passed | mechanism_report_count=7, passed_report_count=7, forced_assignments=2793 |
| 1.15.4 uncertain-ruling disposition | passed | uncertain_ruling_count=1, excluded_collectible_count=1, trainable_collectible_count=734 |
| 1.15.5 runtime-coverage honesty | passed | runtime_clause_count=2151, raw_nonpassed_count=2136, unexplained_count=0 |
| 1.15.6 10,000-game stratified sampling | passed | games=10000, strata=98, mask_checks=909158, encountered_cards=824, replays=98 |
| 1.15.7 bug severity closure | passed | total=8, fixed=8, open_p0=0, open_p1=0 |
| 1.15.8 portable reproduction collection | passed | confirmed_bugs=8, fixed_bugs=8, portable_package=data/reports/card_bug_audit/repros/SWB-CARD-0008.json, minimized_actions=86 |
| 1.15.9 frozen implementation manifests | passed | rules_engine_commit=22d8d76806df6ee67ad91761286c4884e3872e03, catalog_policy_commit=9699ab97b3c9865b047c29168df284f0a933a3ee, database_sha256=df069e713a97493c885266b72f303874035beea571147ba14b77c57c9e631376, rules_sha256=449230d69db016ff99fa27de963f89adc9ac9ac5b54e913991ffe3314e4f2017, catalog_sha256=57a9c1927de54ea5901a02d48d27438211ad9edc24f61ee6bc92d624d707f2f6, tests_sha256=0d0690436bce7d5593025e8fc732bfc25b9121621b4f8bd5cd2510ab2789cdb2 |

## 10,000 局分层采样

最终报告完成 10,000/10,000 局、98 个分层、909,158 次 mask 检查和 98 次固定 seed 重放；异常、截断、非法动作、placeholder、mask mismatch 和重放失败均为 0。更早失败报告未被覆盖，仍保留两次非正生命不变量异常和 194 个 placeholder，作为本轮修复来源。

## Runtime coverage 解释

原始未采样、未触发和已触发未执行条款没有改标为通过。它们只通过 forced-scenario 报告中独立重跑的直接测试获得解释，因此随机 smoke 覆盖与直接行为证据保持分离。

## 冻结标识

- 规则引擎提交：`22d8d76806df6ee67ad91761286c4884e3872e03`
- Catalog 策略提交：`9699ab97b3c9865b047c29168df284f0a933a3ee`
- 数据库 SHA-256：`df069e713a97493c885266b72f303874035beea571147ba14b77c57c9e631376`
- 规则 SHA-256：`449230d69db016ff99fa27de963f89adc9ac9ac5b54e913991ffe3314e4f2017`
- Catalog SHA-256：`57a9c1927de54ea5901a02d48d27438211ad9edc24f61ee6bc92d624d707f2f6`
- 训练池 SHA-256：`7bdf090f725fcf80bcef59fb42a94d9c2d0b85affd15cde8d838df46016e7146`
- 测试 SHA-256：`0d0690436bce7d5593025e8fc732bfc25b9121621b4f8bd5cd2510ab2789cdb2`

## 已知限制

- `SWB-RULING-SET-STATS-TEMP-001`：Older temporary stat-modifier expiry after a later specific-value assignment remains officially unconfirmed. 10233310 is excluded from newly sampled initial decks; the card remains auditable and resolvable.
- `RUNTIME-COVERAGE-SAMPLING-LIMIT`：2136 of 2151 runtime clauses were not sampled, not triggered or not executed in the smoke corpus and retain those labels. Each is separately attributed to re-executed direct tests in the forced-scenario report.

机器可读完整结果：`data/reports/card_bug_audit/final_gate.json`。
