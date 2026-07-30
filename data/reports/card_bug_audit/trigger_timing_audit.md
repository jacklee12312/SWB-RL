# Trigger timing, priority, and batch audit

- Result: **PASS**; 0 failures.
- Snapshot: 826 cards (735 collectible / 91 generated).
- Trigger sources: 770 cards; 131 in the training closure; 26 isolated demo sources.
- Resolution loop guard: 20000 steps.

## External timing evidence

| Evidence | Authority | Card | Accessed | Conclusion |
|---|---|---:|---|---|
| SWB-TIMING-CARD-003 | official_card_text_plus_retained_regression | 10364120 | 2026-07-30 | 官网卡牌文字确认玛温纹章的回合结束触发；跨来源优先级由既有真实卡回归保留，当前官网未检出专门回答该组合的 Q&A。 |
| SWB-TIMING-OFFICIAL-001 | official_qa | 10404110 | 2026-07-30 | 官方 Q&A 明确了纹章、倒数破坏、瞬念召唤、谢幕曲、瞬念后续能力和通常抽牌的回合开始顺序。 |
| SWB-TIMING-OFFICIAL-002 | official_qa | 10153140 | 2026-07-30 | 官方 Q&A 明确巴尔特纹章在双方主战者均为 1 时由对手获胜，证明致死立即确定并停止剩余伤害。 |

## Trigger matrix

| Category | Sources | Collectible | Generated | Training | Records | Demo | Result |
|---|---:|---:|---:|---:|---:|---:|:---:|
| turn_start | 9 | 9 | 0 | 2 | 11 | 4 | PASS |
| turn_end | 62 | 52 | 10 | 14 | 63 | 1 | PASS |
| attack | 16 | 14 | 2 | 2 | 17 | 0 | PASS |
| clash | 5 | 5 | 0 | 1 | 5 | 0 | PASS |
| damage_survived | 3 | 3 | 0 | 1 | 4 | 0 | PASS |
| entry | 631 | 591 | 40 | 110 | 682 | 25 | PASS |
| evolve | 169 | 169 | 0 | 31 | 169 | 0 | PASS |
| super_evolve | 94 | 94 | 0 | 20 | 94 | 0 | PASS |
| last_words | 78 | 68 | 10 | 13 | 78 | 1 | PASS |
| countdown | 60 | 57 | 3 | 13 | 72 | 2 | PASS |
| emblem | 56 | 56 | 0 | 8 | 97 | 8 | PASS |
| faith | 5 | 5 | 0 | 3 | 10 | 0 | PASS |
| other_event_listener | 27 | 27 | 0 | 7 | 28 | 0 | PASS |
| other_card_trigger | 75 | 70 | 5 | 16 | 88 | 3 | PASS |

## Behavioral contracts

| Contract | Evidence tests | External evidence | Result |
|---|---:|---:|:---:|
| trigger_matrix | 27 | 0 | PASS |
| official_turn_boundary_priority | 1 | 1 | PASS |
| marwynn_crest_before_board | 1 | 1 | PASS |
| simultaneous_death_batch_before_last_words | 2 | 0 | PASS |
| new_sources_wait_for_next_batch | 3 | 0 | PASS |
| conditions_snapshot_at_batch_start | 2 | 0 | PASS |
| queued_source_leaves_play | 2 | 0 | PASS |
| pending_choice_preserves_queue_order | 4 | 0 | PASS |
| terminal_result_stops_remaining_queue | 3 | 0 | PASS |
| all_leader_damage_winner | 1 | 1 | PASS |
| recursive_trigger_step_limit | 4 | 0 | PASS |

## Explicit unsupported boundaries

| Mechanism | Status | Production sources | Result |
|---|---|---:|:---:|
| death_batch_start_emblem_trigger | explicitly_unsupported_not_applicable | 0 | PASS |

## Source inventory

| Card | Categories | Records | Training | Result |
|---|---|---:|:---:|:---:|
| 999001 synthetic-demo-999001 | entry | 1 | no | PASS |
| 999002 synthetic-demo-999002 | entry | 1 | no | PASS |
| 999101 synthetic-demo-999101 | entry | 1 | no | PASS |
| 999102 synthetic-demo-999102 | entry | 1 | no | PASS |
| 999103 synthetic-demo-999103 | entry | 1 | no | PASS |
| 999801 synthetic-demo-999801 | entry | 1 | no | PASS |
| 999802 synthetic-demo-999802 | entry, other_card_trigger | 2 | no | PASS |
| 999803 synthetic-demo-999803 | entry, other_card_trigger | 2 | no | PASS |
| 999804 synthetic-demo-999804 | entry | 1 | no | PASS |
| 999805 synthetic-demo-999805 | entry | 1 | no | PASS |
| 999806 synthetic-demo-999806 | entry, other_card_trigger | 2 | no | PASS |
| 999901 synthetic-demo-999901 | turn_start, entry, emblem | 3 | no | PASS |
| 999902 synthetic-demo-999902 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 999903 synthetic-demo-999903 | turn_start, entry, emblem | 3 | no | PASS |
| 999910 synthetic-demo-999910 | turn_start, entry, emblem | 3 | no | PASS |
| 999911 synthetic-demo-999911 | turn_start, entry, emblem | 3 | no | PASS |
| 999912 synthetic-demo-999912 | entry, emblem | 3 | no | PASS |
| 999913 synthetic-demo-999913 | entry, emblem | 3 | no | PASS |
| 999914 synthetic-demo-999914 | entry, countdown, emblem | 5 | no | PASS |
| 999950 synthetic-demo-999950 | entry | 1 | no | PASS |
| 999951 synthetic-demo-999951 | entry | 1 | no | PASS |
| 999952 synthetic-demo-999952 | entry | 1 | no | PASS |
| 999953 synthetic-demo-999953 | entry | 1 | no | PASS |
| 999954 synthetic-demo-999954 | entry | 1 | no | PASS |
| 999955 synthetic-demo-999955 | entry | 1 | no | PASS |
| 999956 synthetic-demo-999956 | last_words | 1 | no | PASS |
| 10001110 不屈的剑斗士 | entry, other_card_trigger | 2 | no | PASS |
| 10001120 叮当天使·莉亚 | evolve, last_words | 2 | no | PASS |
| 10001210 侦探的放大镜 | other_card_trigger | 1 | no | PASS |
| 10002110 煌响使者·亨莉雅妲 | evolve, super_evolve | 2 | no | PASS |
| 10002210 冒险者公会 | entry, other_card_trigger | 2 | no | PASS |
| 10011110 妖精驯服者 | entry | 1 | no | PASS |
| 10011120 流浪兽人 | entry | 1 | no | PASS |
| 10011130 温厚的树精 | attack, entry | 2 | no | PASS |
| 10011210 缭乱之庭 | entry, countdown | 3 | no | PASS |
| 10012110 冒险精灵·小梅 | entry | 1 | no | PASS |
| 10012120 音速射手·塞尔文 | super_evolve | 1 | no | PASS |
| 10012310 昆虫的忠告 | entry | 1 | no | PASS |
| 10021120 战斗商贩 | last_words | 1 | no | PASS |
| 10021310 女仆的礼仪 | entry | 1 | no | PASS |
| 10022110 王室御用车夫 | last_words | 1 | no | PASS |
| 10022120 魔煌的诡谲者·拉斯提 | super_evolve | 1 | no | PASS |
| 10022210 昭示正统的王冠 | entry, countdown, other_card_trigger | 4 | no | PASS |
| 10031110 闪光魔法剑士 | entry | 1 | no | PASS |
| 10031210 魔女的炼金炉 | entry, other_card_trigger | 2 | no | PASS |
| 10031310 智慧光辉 | entry | 1 | yes | PASS |
| 10031320 召唤真理 | entry | 1 | no | PASS |
| 10032110 双面魔女·蕾米拉米 | entry, super_evolve | 2 | no | PASS |
| 10032310 魔爆 | entry | 1 | no | PASS |
| 10041110 烈焰火蜥蜴 | entry | 1 | no | PASS |
| 10041130 凶鲨战士 | entry | 1 | no | PASS |
| 10041310 龙人碎击 | entry | 1 | no | PASS |
| 10042110 猛攻的龙战士 | evolve, super_evolve | 2 | no | PASS |
| 10042120 咆哮的驭龙使 | entry, other_card_trigger | 2 | no | PASS |
| 10042310 龙之启示 | entry | 1 | yes | PASS |
| 10051120 黑夜鬼人 | entry | 1 | no | PASS |
| 10051130 恶毒的小木乃伊 | entry | 1 | no | PASS |
| 10051310 混沌诅咒 | entry | 1 | yes | PASS |
| 10052110 魅惑的魅魔·莉莉姆 | last_words | 1 | no | PASS |
| 10052120 多情的唤灵师 | evolve, super_evolve | 2 | no | PASS |
| 10052310 捕食灵魂 | entry | 1 | no | PASS |
| 10061110 治愈的修女 | entry | 1 | no | PASS |
| 10061130 圣翼战士 | entry, evolve | 2 | no | PASS |
| 10061210 投影鸟像 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10062110 铁拳神父 | evolve, super_evolve | 2 | no | PASS |
| 10062120 神圣狮鹫 | other_event_listener | 1 | no | PASS |
| 10062210 羽翼石像 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10071110 炮击猫兽人 | entry | 1 | no | PASS |
| 10071120 人偶长矛手 | entry | 1 | no | PASS |
| 10071310 来自异次元的枪击 | entry | 1 | no | PASS |
| 10072110 电鞭手 | entry | 1 | no | PASS |
| 10072120 魔钢骑兵 | evolve, super_evolve | 2 | no | PASS |
| 10072210 人偶剧场 | turn_end, entry, countdown | 3 | yes | PASS |
| 10101110 贪婪的智天使·露比 | entry | 1 | no | PASS |
| 10101120 观察的侦探 | last_words | 1 | no | PASS |
| 10101310 哥布林的偷袭 | entry | 1 | no | PASS |
| 10102110 迸发的光明·阿波罗 | entry, evolve | 2 | no | PASS |
| 10102310 炽天使的福音 | entry | 1 | no | PASS |
| 10103110 爽朗的天宫·菲尔德亚 | evolve | 1 | no | PASS |
| 10103310 神之雷霆 | entry | 1 | no | PASS |
| 10104110 勇武的堕天使·奥莉薇 | entry, super_evolve | 2 | no | PASS |
| 10104120 终极之罪·深渊之主 | entry | 1 | no | PASS |
| 10111110 舞动的妖精 | entry | 1 | no | PASS |
| 10111120 恋触妖精 | entry, evolve | 2 | no | PASS |
| 10111130 深奥的妖精守护圣兽 | entry | 1 | no | PASS |
| 10111140 勤劳的蚂蚱 | entry | 1 | no | PASS |
| 10111150 言传的杂草人长老 | entry | 1 | no | PASS |
| 10111310 妖精召集令 | entry | 1 | no | PASS |
| 10112110 霜寒冰晶·艾琳 | entry | 1 | no | PASS |
| 10112120 纯真的水之妖精 | last_words | 1 | no | PASS |
| 10112130 年幼宝石兽 | entry, super_evolve | 2 | no | PASS |
| 10112210 磷光辉岩 | entry, other_card_trigger | 2 | no | PASS |
| 10112310 薰交的思慕 | entry | 1 | no | PASS |
| 10113110 纯洁冰晶·莉莉 | entry, evolve | 2 | no | PASS |
| 10113120 薰交的天宫·巴克伍德 | entry, evolve | 2 | no | PASS |
| 10113130 煌击战士·贝鲁 | entry, other_event_listener | 2 | no | PASS |
| 10113140 屠戮破魔虫 | entry | 1 | no | PASS |
| 10113210 圣树法杖 | turn_end, entry, other_card_trigger | 4 | no | PASS |
| 10114110 自然妖精公主·阿丽雅 | entry, evolve, emblem | 4 | no | PASS |
| 10114120 丰丽的玫瑰皇后 | entry | 1 | no | PASS |
| 10114130 起源剑师·阿玛兹 | entry, evolve | 2 | no | PASS |
| 10121110 爱之骑士·尹安 | entry | 1 | no | PASS |
| 10121120 和平商人·艾尔涅丝塔 | entry, evolve | 2 | no | PASS |
| 10121130 救援的鲁米那斯治疗师·莉拉拉 | entry | 2 | no | PASS |
| 10121140 军犬 | entry, other_card_trigger | 2 | no | PASS |
| 10121150 异端武士 | entry | 1 | no | PASS |
| 10121310 剑士的斩击 | entry | 1 | no | PASS |
| 10122110 统率的鲁米那斯骑士 | entry, evolve | 2 | no | PASS |
| 10122120 卓越的鲁米那斯法师 | entry | 2 | no | PASS |
| 10122130 勇猛的鲁米那斯枪士 | entry | 2 | no | PASS |
| 10122140 忍者鼯鼠 | evolve | 1 | no | PASS |
| 10122310 王断的威光 | entry | 1 | no | PASS |
| 10123110 雷维翁之斧·杰诺 | attack | 1 | no | PASS |
| 10123120 沉默的狙击手·瓦路兹 | entry | 1 | no | PASS |
| 10123130 王断的天宫·斯塔奇乌姆 | evolve | 1 | no | PASS |
| 10123140 煌刃勇者·阿玛利亚 | entry | 2 | no | PASS |
| 10123310 触手撕咬 | entry | 1 | no | PASS |
| 10124110 雷维翁的迅雷·阿尔贝尔 | entry, other_card_trigger | 2 | no | PASS |
| 10124120 白银骑士团团长·艾蜜莉亚 | entry, super_evolve | 2 | no | PASS |
| 10124130 常在战场·景光 | super_evolve, last_words, countdown, emblem | 6 | no | PASS |
| 10131110 符文剑操控师 | entry, other_event_listener | 2 | no | PASS |
| 10131120 见习占星术师 | entry | 1 | no | PASS |
| 10131130 唤枭士 | entry, evolve | 2 | no | PASS |
| 10131140 追梦的企鹅魔法师 | entry, evolve | 2 | no | PASS |
| 10131310 彩虹奇迹 | entry | 1 | no | PASS |
| 10131320 暴风破 | entry | 1 | yes | PASS |
| 10132110 惹人怜爱的教师·米兰 | entry, evolve | 2 | no | PASS |
| 10132120 奇迹女巫·爱蜜儿 | evolve | 1 | no | PASS |
| 10132130 玛纳利亚的学生·威廉 | entry, evolve | 2 | no | PASS |
| 10132310 理光的证明 | entry | 1 | no | PASS |
| 10132320 雪人觉醒 | entry | 1 | no | PASS |
| 10133110 黎明炼金术师·诺诺 | turn_end, entry, evolve, countdown, emblem | 6 | no | PASS |
| 10133120 魔法药剂师·佩内洛普 | entry, super_evolve | 2 | no | PASS |
| 10133130 理光的天宫·艾德薇诗 | entry, evolve | 2 | no | PASS |
| 10133310 做作业啦！ | entry, other_event_listener | 2 | no | PASS |
| 10133320 唤鬼术 | entry | 1 | no | PASS |
| 10134110 五行之巅·久苑 | entry, super_evolve | 2 | no | PASS |
| 10134120 玛纳利亚密友·安&古蕾雅 | entry, evolve | 2 | no | PASS |
| 10134310 超越次元 | entry | 1 | no | PASS |
| 10141110 云海龙骑兵 | last_words | 1 | no | PASS |
| 10141120 海沟大剑龙 | entry | 1 | no | PASS |
| 10141130 初出茅庐的屠龙者 | entry | 1 | no | PASS |
| 10141140 育龙少女 | entry, evolve | 2 | no | PASS |
| 10141150 白鳞的使者 | entry | 1 | no | PASS |
| 10141310 灾祸吐息 | entry | 1 | no | PASS |
| 10142110 煌牙的义勇·基德 | other_card_trigger | 1 | no | PASS |
| 10142120 御风者·叶花 | entry | 1 | no | PASS |
| 10142130 读风者·杰鲁 | super_evolve | 1 | no | PASS |
| 10142140 艳丽龙人·玛利翁 | entry | 1 | no | PASS |
| 10142310 荣弦的奏乐 | entry | 1 | no | PASS |
| 10143120 荣弦的天宫·龙芙 | entry, evolve | 2 | no | PASS |
| 10143130 惊涛龙骑士·扎哈尔 | entry | 1 | no | PASS |
| 10143140 夜幕龙 | entry, super_evolve | 2 | no | PASS |
| 10143210 乙姬的宝扇 | other_card_trigger | 1 | no | PASS |
| 10144110 灼热的安纳提玛·班德奈特 | turn_start, entry, super_evolve, emblem, other_event_listener | 6 | no | PASS |
| 10144130 龙人演义·卧龙 | entry, super_evolve | 2 | no | PASS |
| 10151110 午夜眷属·埃拉尔 | entry | 2 | no | PASS |
| 10151120 无名恶魔 | evolve | 1 | no | PASS |
| 10151130 白骨少女 | last_words | 1 | no | PASS |
| 10151140 杂耍亡灵 | entry | 1 | no | PASS |
| 10151150 禁约恶魔 | entry, evolve | 2 | no | PASS |
| 10151310 死神挥刀 | entry | 1 | no | PASS |
| 10152110 干练的死神·蜜诺 | last_words | 1 | no | PASS |
| 10152120 梦魇·维莉 | entry, evolve | 2 | no | PASS |
| 10152130 怪奇探索者·尤娜 | last_words | 1 | no | PASS |
| 10152140 穿刺公·弗拉德 | entry | 1 | no | PASS |
| 10152210 瞑地的陵园 | entry, other_card_trigger | 2 | no | PASS |
| 10153110 蓝蔷薇千金·赛蕾丝 | turn_end | 1 | no | PASS |
| 10153120 燃烧魔剑·欧特鲁斯 | entry, evolve | 2 | no | PASS |
| 10153130 瞑地的天宫·穆甘 | entry, evolve | 3 | no | PASS |
| 10153140 地下赏金猎人·巴尔特 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10153310 蛇神之怒 | entry | 1 | no | PASS |
| 10154110 奔放的狱焰·凯尔贝洛斯 | entry, super_evolve | 2 | no | PASS |
| 10154120 剧毒公主·美杜莎 | attack | 1 | no | PASS |
| 10154130 无极猎人·阿拉加维 | entry, evolve | 2 | no | PASS |
| 10161110 圣心光棱牧师 | entry, evolve | 2 | no | PASS |
| 10161130 指明目标的光辉天使 | entry | 1 | no | PASS |
| 10161140 潜伏的曼纽 | other_event_listener | 1 | no | PASS |
| 10161210 祥和的教会 | entry, last_words, countdown, other_card_trigger | 6 | no | PASS |
| 10161310 翎雨 | entry | 1 | no | PASS |
| 10162110 煌枪魔兔·萨莉沙 | evolve, last_words | 2 | no | PASS |
| 10162120 煌翼羽人·莉诺 | clash, super_evolve | 2 | no | PASS |
| 10162130 大地守护神·米维 | last_words | 1 | no | PASS |
| 10162210 禁密的圣地 | other_card_trigger | 1 | no | PASS |
| 10162220 神圣注射 | other_card_trigger | 1 | no | PASS |
| 10163110 终焉的白骨圣堂之主 | entry | 1 | no | PASS |
| 10163120 禁密的天宫·罗纳维罗 | evolve | 1 | no | PASS |
| 10163130 伟大的炽天使·勒碧丝 | last_words, countdown, emblem | 5 | no | PASS |
| 10163210 野兽公主的誓约 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10163220 邪教法器 | other_card_trigger | 1 | no | PASS |
| 10164110 裁决的安纳提玛·罗德欧 | entry, super_evolve | 2 | no | PASS |
| 10164120 纯白圣女·贞德 | entry | 1 | no | PASS |
| 10164130 水之守护神·萨蕾法 | entry, evolve | 2 | no | PASS |
| 10171110 耀眼的发明家·伊莉斯 | last_words | 1 | no | PASS |
| 10171120 钢铁佣兵·迪尔克 | entry | 1 | no | PASS |
| 10171130 永不停火的枪手 | entry, evolve | 2 | no | PASS |
| 10171140 自动机械刺客 | entry | 2 | no | PASS |
| 10171310 人偶替身 | entry | 1 | no | PASS |
| 10171320 创造物充能 | entry | 1 | no | PASS |
| 10172110 抗争领袖·露琪娜 | entry, evolve | 2 | no | PASS |
| 10172120 依恋的人偶师 | entry, evolve | 2 | no | PASS |
| 10172130 杀意之丝·诺亚 | entry | 1 | no | PASS |
| 10172310 生命的奔流 | entry | 1 | no | PASS |
| 10172320 改境的重启 | entry | 1 | no | PASS |
| 10173110 决心之誓·米莉亚姆 | entry, evolve | 2 | no | PASS |
| 10173120 箱庭的断罪者·希尔薇娅 | entry, evolve, super_evolve | 3 | no | PASS |
| 10173130 疯狂的创造者·历亚姆 | entry, evolve | 2 | no | PASS |
| 10173140 改境的天宫·阿洛艾特 | entry, evolve | 2 | no | PASS |
| 10173210 遗产的炮击 | entry, other_event_listener | 2 | no | PASS |
| 10174110 崭新的少女·欧丝 | turn_end, entry, evolve, countdown, emblem | 6 | no | PASS |
| 10174120 迈进之心·奥契丝 | entry, super_evolve | 3 | no | PASS |
| 10174130 增幅加速·洛拉米亚 | entry, super_evolve | 2 | no | PASS |
| 10201110 双刀哥布林 | entry | 1 | no | PASS |
| 10201310 逆向变化 | entry | 1 | no | PASS |
| 10202110 女仆天使·切蕾塔 | entry | 1 | no | PASS |
| 10203110 联结的天使·蕾娜 | evolve | 1 | no | PASS |
| 10203120 雷火双神·福尼加尔&亚文哈尔 | last_words | 1 | no | PASS |
| 10204110 命运黄昏·奥丁 | entry | 1 | no | PASS |
| 10204120 飓风天业·格里姆尼尔 | turn_end, entry, emblem | 3 | no | PASS |
| 10211120 木锤矮人 | entry | 1 | no | PASS |
| 10211310 森林的游行 | entry | 1 | no | PASS |
| 10212110 热情的精灵·莱昂内尔 | entry | 1 | no | PASS |
| 10212120 妖精击剑士 | entry, super_evolve | 2 | no | PASS |
| 10212310 来自树上的偷袭 | entry | 1 | no | PASS |
| 10213110 森林骑士道·辛西亚 | entry, evolve | 2 | no | PASS |
| 10213310 花园的指引 | entry | 1 | no | PASS |
| 10214110 翅翼女王·提泰妮娅 | turn_start, entry, evolve, emblem | 4 | no | PASS |
| 10214120 缠绕密林·丽梅格 | entry, super_evolve | 2 | no | PASS |
| 10221110 扳机女仆·赛莉亚 | entry | 1 | no | PASS |
| 10221310 商谈成立 | entry | 1 | no | PASS |
| 10222110 无畏的副团长·格尔德 | turn_end | 1 | no | PASS |
| 10222120 平凡骑士·拉奇尔 | entry, evolve | 2 | no | PASS |
| 10222310 三将姬的乱击 | entry | 1 | no | PASS |
| 10223110 剑士公主·萝泽 | attack, entry, other_card_trigger | 3 | no | PASS |
| 10223120 假日中的王女·普莉姆 | entry, super_evolve | 2 | no | PASS |
| 10224110 静寂的安纳提玛·吉尔达利娅 | entry, evolve | 3 | no | PASS |
| 10224120 雷维翁超越者·尤里乌斯 | entry | 2 | no | PASS |
| 10231110 憧憬的魔女·梅薇 | entry | 1 | no | PASS |
| 10231120 魔导图书管理员 | entry | 1 | no | PASS |
| 10231310 冰锥穿击 | entry | 1 | no | PASS |
| 10232110 否定的咏唱·芭赛特 | turn_start, entry, evolve, emblem | 4 | no | PASS |
| 10232120 调香的魔法师 | entry, evolve | 2 | no | PASS |
| 10232310 混沌赤焰 | entry | 1 | no | PASS |
| 10233110 玛纳利亚剑士·欧文 | clash, evolve | 2 | no | PASS |
| 10233310 帕梅拉的舞蹈 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10234110 暴食的安纳提玛·拉拉安瑟姆 | super_evolve, last_words | 2 | no | PASS |
| 10234120 精金炼金术师·诺曼 | entry, evolve | 2 | no | PASS |
| 10241110 庇护的智龙 | entry, super_evolve | 2 | no | PASS |
| 10241120 飞跃的银白幼龙 | entry | 1 | no | PASS |
| 10241310 虎鲸的呼声 | entry | 1 | no | PASS |
| 10242110 宿愿的龙人公主 | entry | 1 | no | PASS |
| 10242120 身经百战的鱼人 | entry, evolve | 2 | no | PASS |
| 10242210 炎龙之剑 | other_card_trigger | 1 | no | PASS |
| 10243110 苍海的制裁·尼普顿 | entry, super_evolve, emblem | 4 | no | PASS |
| 10243310 龙骑突击 | entry | 1 | no | PASS |
| 10244110 银冰龙少女·菲琳 | entry, evolve | 2 | no | PASS |
| 10244120 绚丽凤凰·小凤 | entry | 1 | no | PASS |
| 10251110 银色子弹·雷文 | evolve | 1 | no | PASS |
| 10251120 怨恨的栽培者 | last_words | 1 | no | PASS |
| 10251310 诅咒派对 | entry | 1 | no | PASS |
| 10252110 爆破之翼·圮尤拉 | super_evolve | 1 | no | PASS |
| 10252120 尸兵 | entry | 1 | no | PASS |
| 10252310 使唤蝙蝠 | entry | 1 | no | PASS |
| 10253110 悲惨战争·萝拉 | entry, super_evolve | 2 | no | PASS |
| 10253120 夜曲将军·艾瑟拉 | entry | 1 | no | PASS |
| 10254110 双轮夜行·吟雪&夕月 | entry | 1 | no | PASS |
| 10254120 流动堕落的冥河·凯伦 | turn_start, entry, super_evolve, countdown, emblem | 7 | no | PASS |
| 10261110 粉碎的圣职者 | evolve | 1 | no | PASS |
| 10261120 恶意的神谕·达姆斯 | entry | 1 | no | PASS |
| 10261210 流光香炉 | other_card_trigger | 1 | no | PASS |
| 10262110 弹幕驱魔人·珂蕾特 | entry | 1 | no | PASS |
| 10262120 有洁癖的审判者 | entry | 1 | no | PASS |
| 10262310 神圣守护 | entry | 1 | no | PASS |
| 10263110 速断之刃·阿尼耶丝 | attack | 1 | no | PASS |
| 10263310 疯狂的恩宠 | entry, countdown, emblem | 5 | no | PASS |
| 10264110 天之守护神·埃忒耳 | entry, super_evolve | 2 | no | PASS |
| 10264120 呜咽的圣骑士·维尔伯特 | entry, evolve, last_words | 3 | no | PASS |
| 10271110 引擎剑士 | entry, evolve | 2 | no | PASS |
| 10271120 猫偶 | entry | 1 | no | PASS |
| 10271210 创造物弹射器 | entry, other_card_trigger | 2 | no | PASS |
| 10272110 心灵屠戮者·菲亚 | entry | 1 | no | PASS |
| 10272120 绝望之王·阿基姆 | evolve | 1 | no | PASS |
| 10272310 伊卡洛斯的飞翔 | entry | 1 | no | PASS |
| 10273110 暗狱的余晖·贾丝珀 | entry, evolve | 2 | no | PASS |
| 10273310 心有灵犀的共斗 | entry | 1 | no | PASS |
| 10274110 交响之心·枷薇 | entry, evolve | 3 | no | PASS |
| 10274120 精神武艺·迦尔拉 | entry, super_evolve | 2 | no | PASS |
| 10301110 涸绝的使徒 | entry | 1 | no | PASS |
| 10301310 至高的凌驾 | entry | 1 | no | PASS |
| 10302110 抗拒叹息之人 | super_evolve | 1 | no | PASS |
| 10303110 充满勇气之人 | super_evolve | 1 | no | PASS |
| 10303210 试炼的石板 | entry, other_card_trigger | 2 | no | PASS |
| 10304110 绝大的显现·麦哲佩恩 | turn_end, entry, evolve, emblem | 4 | no | PASS |
| 10304120 涸绝的显现·吉尔内莉莎 | entry, evolve | 2 | yes | PASS |
| 10311110 不弑的肯定者 | entry, evolve | 2 | no | PASS |
| 10311120 妖精剑的后继者 | entry | 1 | no | PASS |
| 10311310 野性的猛袭 | entry | 1 | no | PASS |
| 10312110 不弑的祈祷者 | entry, super_evolve | 2 | no | PASS |
| 10312120 树海的战士 | last_words | 1 | no | PASS |
| 10312210 不弑之乡 | entry, other_card_trigger | 2 | no | PASS |
| 10313110 不弑的团结者 | entry | 1 | no | PASS |
| 10313310 驱逐的死矢 | entry | 1 | no | PASS |
| 10314110 不弑的继承者·库露露 | entry, super_evolve, countdown, emblem, other_event_listener | 7 | yes | PASS |
| 10314120 绝命的显现·艾斯迪亚 | entry, evolve | 2 | no | PASS |
| 10321110 篡夺的肯定者 | entry, last_words | 2 | no | PASS |
| 10321120 剑圣的同胞 | last_words | 1 | no | PASS |
| 10321310 护盾强袭 | entry | 1 | no | PASS |
| 10322110 篡夺的祈祷者 | entry, last_words | 2 | no | PASS |
| 10322120 活泼的斥候 | entry, evolve | 2 | no | PASS |
| 10322210 篡夺的据点 | other_card_trigger | 1 | no | PASS |
| 10323110 篡夺的团结者 | entry, evolve, other_event_listener | 4 | no | PASS |
| 10323310 奉还的剑闪 | entry | 1 | no | PASS |
| 10324110 篡夺的继承者·辛瑟莱兹 | entry, super_evolve | 2 | no | PASS |
| 10324120 空绝的显现·奥克托丽丝 | entry, evolve, countdown, emblem, other_event_listener | 10 | no | PASS |
| 10331110 真理的肯定者 | entry | 1 | no | PASS |
| 10331120 五行修行者 | entry, evolve | 2 | no | PASS |
| 10331310 水晶的指引 | entry, countdown, emblem | 5 | no | PASS |
| 10332110 真理的祈祷者 | entry | 1 | no | PASS |
| 10332210 真理的研究设施 | entry, countdown, other_card_trigger | 5 | no | PASS |
| 10332310 双重创造 | entry | 1 | no | PASS |
| 10333110 真理的团结者 | entry, super_evolve | 2 | no | PASS |
| 10333310 虚假的术式 | entry | 1 | yes | PASS |
| 10334110 真理的继承者·蓓哈丽雅 | entry, evolve | 2 | no | PASS |
| 10334120 绝尽的显现·莱奥 | entry | 1 | no | PASS |
| 10341110 侮蔑的肯定者 | damage_survived | 1 | no | PASS |
| 10341120 风雪龙人 | entry | 1 | no | PASS |
| 10341310 雷霆之怒 | entry | 1 | no | PASS |
| 10342110 侮蔑的祈祷者 | turn_end | 1 | no | PASS |
| 10342120 海洋骑手 | entry | 2 | no | PASS |
| 10342210 侮蔑之国 | entry, countdown, other_card_trigger | 4 | no | PASS |
| 10343110 侮蔑的团结者 | turn_end, entry | 2 | no | PASS |
| 10343310 威猛炽焰 | entry | 1 | no | PASS |
| 10344110 侮蔑的继承者·安吉拉弗利特 | damage_survived, entry, super_evolve | 3 | no | PASS |
| 10344120 烈绝的显现·嘉尔缪 | damage_survived, entry | 3 | yes | PASS |
| 10351110 混融的肯定者 | entry, evolve | 2 | yes | PASS |
| 10351120 泡沫鬼姬 | entry | 1 | no | PASS |
| 10351310 前进的暴虐 | entry | 1 | no | PASS |
| 10352110 混融的祈祷者 | entry, evolve | 2 | yes | PASS |
| 10352120 新来的守墓人 | entry, other_card_trigger | 2 | no | PASS |
| 10352210 混融之城 | entry, other_card_trigger | 2 | yes | PASS |
| 10353110 混融的团结者 | entry, super_evolve | 2 | yes | PASS |
| 10353310 叫唤与憎恶 | entry | 1 | yes | PASS |
| 10354110 混融的继承者·莎木·纳克雅 | entry, super_evolve, faith, other_event_listener | 5 | yes | PASS |
| 10354120 绝叫与爱绝的显现·鲁鲁纳伊&巴娜蕾卡 | entry | 1 | yes | PASS |
| 10361110 安息的肯定者 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10361120 圣骑士团员 | other_event_listener | 1 | no | PASS |
| 10361310 圣辉闪烁 | entry | 1 | no | PASS |
| 10362110 安息的祈祷者 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10362210 安息的神殿 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10362220 羽翼狮子像 | entry, last_words, countdown, other_card_trigger | 5 | yes | PASS |
| 10363110 安息的团结者 | turn_end, entry, evolve, countdown, emblem | 6 | no | PASS |
| 10363210 闪耀的失意 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10364110 安息的继承者·妃花 | turn_end, entry, super_evolve, countdown, emblem | 6 | no | PASS |
| 10364120 绝望的显现·玛温 | turn_end, entry, evolve, emblem | 4 | no | PASS |
| 10371110 破坏的肯定者 | entry | 1 | no | PASS |
| 10371120 音速飞行兵 | entry, evolve | 2 | no | PASS |
| 10371310 丝线突袭 | entry | 1 | no | PASS |
| 10372110 破坏的祈祷者 | entry, evolve | 2 | no | PASS |
| 10372120 现场工程师 | entry | 1 | no | PASS |
| 10372210 破坏的荒野 | entry, last_words | 2 | yes | PASS |
| 10373110 破坏的团结者 | entry | 1 | yes | PASS |
| 10373310 歼灭的歌声 | entry | 1 | yes | PASS |
| 10374110 破坏的继承者·阿克西娅 | super_evolve | 1 | yes | PASS |
| 10374120 奏绝的显现·莉洁纳 | entry, evolve | 2 | yes | PASS |
| 10401120 亲爱的搭档·碧 | entry | 1 | no | PASS |
| 10402110 宙域使者·尤妮 | turn_end | 1 | no | PASS |
| 10403110 征服苍空的骑空士·古兰&姬塔 | entry | 1 | yes | PASS |
| 10403120 掌握天空命运的少女·露莉亚 | entry, other_card_trigger | 2 | yes | PASS |
| 10404110 天司长的继承者·圣德芬 | turn_start, turn_end, entry, countdown, emblem | 8 | yes | PASS |
| 10411110 爱恨舞者·晧&曜 | attack | 1 | no | PASS |
| 10411120 可爱如琬似花·玛娜玛尔 | turn_end, evolve | 2 | no | PASS |
| 10411310 彗星 | entry | 1 | no | PASS |
| 10412110 美妆少女·克洛伊 | entry, other_card_trigger | 2 | no | PASS |
| 10412120 绯焰舞姬·安苏莉娅 | entry | 1 | no | PASS |
| 10412310 绮罗星 | entry, countdown, emblem | 5 | no | PASS |
| 10413110 幻彩弓手·丘比丹 | evolve | 1 | yes | PASS |
| 10413310 亚尔夫海姆 | entry | 1 | no | PASS |
| 10414120 调和的舞者·尤艾尔&苏丝雅 | entry, super_evolve, countdown, emblem | 5 | yes | PASS |
| 10421110 信念腿法·兰德尔 | entry, other_card_trigger | 2 | no | PASS |
| 10421120 决心之辉龙·亚瑟 | evolve | 1 | no | PASS |
| 10421130 迷茫的狮子·莫德雷德 | evolve | 1 | no | PASS |
| 10422110 冰心霸王·艾格罗瓦尔 | entry | 1 | no | PASS |
| 10423110 真王之刃·黄金骑士 | entry | 1 | no | PASS |
| 10423310 骁勇骑士 | entry | 1 | no | PASS |
| 10424110 真红与群青·塞达&贝阿朵丽丝 | entry | 1 | yes | PASS |
| 10431110 不可思议的哲学家·菲拉索佩娅 | entry | 1 | no | PASS |
| 10431120 流浪的家庭教师·斯芙拉玛尔 | turn_end, evolve | 2 | no | PASS |
| 10431310 符文秘术 | entry | 1 | yes | PASS |
| 10432110 报仇的占卜师·艾塞克莱因 | entry | 1 | no | PASS |
| 10432120 荆棘旅途·米蕾耶&莉赛特 | entry | 1 | no | PASS |
| 10432310 能量外溢 | entry | 1 | no | PASS |
| 10433110 缅怀之火·埃尔默特 | turn_start, entry, super_evolve, emblem | 4 | no | PASS |
| 10433310 炼金炎爆 | entry | 1 | no | PASS |
| 10434110 水之法则·瓦姆杜斯 | entry, super_evolve, other_event_listener | 3 | no | PASS |
| 10434120 天才美少女炼金术士·卡莉奥丝特罗 | turn_start, entry, emblem | 3 | no | PASS |
| 10441120 梅格的挚友·玛丽亲 | turn_end, super_evolve | 2 | no | PASS |
| 10441310 破浪新月 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10442110 冰封的命运·伊什米尔 | entry, evolve | 2 | no | PASS |
| 10442120 淳朴的钢铁之躯·无限 | super_evolve | 1 | no | PASS |
| 10442310 至爱狂轰 | entry | 1 | no | PASS |
| 10443110 平平无奇的女孩·梅格 | entry | 1 | no | PASS |
| 10443310 星晶兽吸收之力 | entry | 1 | no | PASS |
| 10444110 炎之法则·威尔纳斯 | entry, evolve | 2 | yes | PASS |
| 10444120 世界的伙伴·佐伊 | entry | 1 | yes | PASS |
| 10451110 憧憬的铁锤·阿尔梅达 | entry, other_card_trigger | 2 | no | PASS |
| 10451120 不屈利刃·巴萨拉卡 | last_words | 1 | no | PASS |
| 10451310 妖异利刃 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10452110 霸空武神·哪吒 | turn_end | 1 | no | PASS |
| 10452120 爱的旅人·萨堤洛斯 | entry | 1 | no | PASS |
| 10452130 元素共鸣·巴尔 | entry | 1 | yes | PASS |
| 10453110 生与死之技·涅槃 | entry | 1 | no | PASS |
| 10453310 堕落 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10454110 暗之法则·菲迪埃尔 | turn_end, entry | 2 | no | PASS |
| 10454120 狡诈的堕天司·彼列 | entry, super_evolve, countdown, emblem | 4 | no | PASS |
| 10461110 英雄幻视·托路 | other_event_listener | 1 | yes | PASS |
| 10461120 克己复礼的修女·拉姆蕾达 | turn_end, evolve | 2 | no | PASS |
| 10461210 莉莉艾的鼓舞 | other_card_trigger | 1 | no | PASS |
| 10462110 沙神的巫女·莎拉 | entry, evolve, other_card_trigger | 3 | no | PASS |
| 10462120 赞恩教僧侣·索菲娅 | entry, super_evolve | 2 | no | PASS |
| 10462210 骑驰天空之艇 | other_event_listener, other_card_trigger | 2 | no | PASS |
| 10463110 魔杖傍身的外科医生·缇可 | evolve, other_event_listener | 2 | no | PASS |
| 10463210 蕾·菲耶的宝石 | other_card_trigger | 1 | yes | PASS |
| 10464110 土之法则·伽莱翁 | turn_end, entry | 2 | no | PASS |
| 10464120 威严的星晶骑士·薇拉 | entry | 1 | no | PASS |
| 10471110 夜王再起·翔 | entry | 1 | no | PASS |
| 10471120 爆燃老大·翼 | entry | 1 | no | PASS |
| 10471130 报恩工匠·艾萨克 | last_words | 1 | yes | PASS |
| 10472110 轰雷闪狼·尤斯提斯 | clash | 1 | no | PASS |
| 10472120 严厉的教官·伊尔莎 | entry | 1 | no | PASS |
| 10472310 身无长物唯有石 | entry | 1 | no | PASS |
| 10473110 向往天空的回归者·卡西乌斯 | entry, last_words | 2 | no | PASS |
| 10473310 混沌军势 | entry | 1 | no | PASS |
| 10474110 光之法则·龙敖 | attack, entry, countdown, emblem | 5 | no | PASS |
| 10474120 唯一王者·别西卜 | entry | 1 | yes | PASS |
| 10501110 挥毫的怪物 | entry | 1 | no | PASS |
| 10502110 星辉女神 | evolve | 1 | no | PASS |
| 10502120 手持军配团扇的伟丈夫 | evolve | 1 | yes | PASS |
| 10503210 大游戏世界 | entry, last_words, countdown, other_card_trigger | 5 | yes | PASS |
| 10503310 《世界》的呈现 | entry | 1 | yes | PASS |
| 10504110 八界花·下天央 | entry | 1 | no | PASS |
| 10511110 熟虑的狸猫 | evolve | 1 | yes | PASS |
| 10511120 森林羽子板工匠 | entry, evolve | 3 | no | PASS |
| 10511310 虫风花的飞翔 | entry | 1 | no | PASS |
| 10512110 新晋搭档 | entry | 1 | no | PASS |
| 10512310 寂静的助力 | entry | 1 | no | PASS |
| 10513110 引路船工 | entry, evolve, super_evolve | 3 | yes | PASS |
| 10513310 优雅的虫风花 | entry | 1 | no | PASS |
| 10514110 脚踩天穹的《倒吊人》·罗弗拉德 | entry, evolve | 2 | no | PASS |
| 10514120 虫风花·魅禄 | entry, evolve | 2 | yes | PASS |
| 10521110 好施的名人 | entry | 1 | no | PASS |
| 10521120 烟管美玉 | entry, evolve | 2 | no | PASS |
| 10521310 丽金花的挥霍 | entry | 1 | no | PASS |
| 10522110 迅猛的武术家 | entry, other_event_listener | 2 | no | PASS |
| 10522120 吉祥蛙 | turn_end, evolve | 2 | no | PASS |
| 10522310 温柔援军 | entry | 1 | no | PASS |
| 10523110 不动如山的将校 | turn_end, entry, super_evolve | 3 | no | PASS |
| 10523310 荣耀的丽金花 | entry | 1 | no | PASS |
| 10524110 威猛的《战车》·奥辂昂 | turn_end | 1 | no | PASS |
| 10524120 丽金花·云庆 | turn_end, entry, super_evolve, countdown, emblem | 6 | no | PASS |
| 10531110 创造魔法师 | entry, super_evolve | 2 | no | PASS |
| 10531120 流动控符师 | entry | 1 | no | PASS |
| 10531310 明越花的转变 | entry | 1 | yes | PASS |
| 10532110 失眠女巫 | entry, evolve, countdown, emblem | 4 | no | PASS |
| 10532120 余韵俳谐师 | entry, last_words | 2 | no | PASS |
| 10532310 魔猫戏法 | entry | 1 | no | PASS |
| 10533110 元素支配者 | entry | 2 | no | PASS |
| 10533310 壮美的明越花 | entry | 1 | no | PASS |
| 10534110 漫步的《愚者》·琳库露 | entry, super_evolve, emblem | 4 | no | PASS |
| 10534120 明越花·阿罗 | entry, evolve | 2 | no | PASS |
| 10541110 涌泉打水人 | evolve, super_evolve | 2 | no | PASS |
| 10541120 水滴打拍者 | entry | 2 | no | PASS |
| 10541310 波摇花的裁决 | entry | 1 | no | PASS |
| 10542110 铁锤龙骑士 | entry | 1 | no | PASS |
| 10542120 水母舞姬 | entry | 2 | no | PASS |
| 10542310 日珥咆哮 | entry | 1 | yes | PASS |
| 10543110 破灭屠戮者 | super_evolve | 1 | no | PASS |
| 10543310 懒惰的波摇花 | entry | 1 | yes | PASS |
| 10544110 约束的《正义》·伊兰翠 | turn_end, evolve | 2 | yes | PASS |
| 10544120 波摇花·夕夜 | attack, entry, evolve, emblem | 6 | no | PASS |
| 10551110 鼓舞之狼 | entry, other_card_trigger | 2 | no | PASS |
| 10551120 红符的魂魄道士 | entry, evolve | 2 | no | PASS |
| 10551310 奥夜花的开战 | entry | 1 | no | PASS |
| 10552110 制造麻烦的唤灵师 | entry, evolve | 2 | no | PASS |
| 10552120 牵线搭桥的青鬼 | entry, evolve | 2 | no | PASS |
| 10552310 残虐的炸裂 | entry | 1 | no | PASS |
| 10553110 致命掠夺者 | entry, evolve, last_words | 3 | no | PASS |
| 10553310 严酷的奥夜花 | turn_end, entry, countdown, emblem | 5 | no | PASS |
| 10554110 充实的《恋人与节制》·米路缇欧&卢泽 | entry, evolve, super_evolve | 3 | no | PASS |
| 10554120 奥夜花·释藤 | entry, super_evolve | 2 | no | PASS |
| 10561110 先见的神官 | entry, evolve | 2 | no | PASS |
| 10561120 连结的使徒 | entry, other_event_listener, other_card_trigger | 3 | no | PASS |
| 10561310 雾卷花的激愤 | entry | 1 | no | PASS |
| 10562110 毫不动摇的圣骑士 | entry | 1 | no | PASS |
| 10562120 穷途末路的巫女 | entry, evolve, other_event_listener | 3 | no | PASS |
| 10562210 穹顶护甲 | other_card_trigger | 1 | no | PASS |
| 10563110 至圣威仪 | entry, evolve, super_evolve, other_event_listener | 4 | no | PASS |
| 10563210 坚固的雾卷花 | entry, other_card_trigger | 2 | no | PASS |
| 10564110 思念的《力量》·索菲娜 | turn_end, entry | 2 | yes | PASS |
| 10564120 雾卷花·茎白 | entry, emblem, other_event_listener | 5 | no | PASS |
| 10571110 舞台缔造者 | entry | 1 | no | PASS |
| 10571120 繁花技师 | entry | 2 | yes | PASS |
| 10571310 尽小花的临照 | entry | 1 | no | PASS |
| 10572110 新时代地理学者 | entry, super_evolve | 2 | no | PASS |
| 10572120 清宵玉兔 | entry | 1 | no | PASS |
| 10572310 苏生调律 | entry | 1 | no | PASS |
| 10573110 神经遮蔽者 | entry, last_words | 2 | no | PASS |
| 10573310 诚心的尽小花 | entry | 1 | yes | PASS |
| 10574110 转动的《命运之轮》·斯洛士 | turn_start, turn_end, countdown, emblem | 6 | no | PASS |
| 10574120 尽小花·伊鞠 | entry, super_evolve | 3 | yes | PASS |
| 10601110 浑浊之民 | last_words | 1 | no | PASS |
| 10601120 匍匐的异类 | entry | 1 | no | PASS |
| 10602210 被侵略的世界 | other_card_trigger | 1 | no | PASS |
| 10603110 彷徨于黑暗之兽 | entry | 1 | no | PASS |
| 10603210 黑暗次元 | turn_end, countdown | 2 | yes | PASS |
| 10604110 恐惧的象征·欧米伽奥提普 | entry, super_evolve | 2 | no | PASS |
| 10611110 慈育的森民 | entry, evolve | 2 | no | PASS |
| 10611120 伊甸之猴 | entry | 1 | no | PASS |
| 10611310 天枪授予 | entry | 1 | no | PASS |
| 10612110 慈颜的拥趸 | entry | 1 | no | PASS |
| 10612310 向女王献花 | entry, evolve | 2 | no | PASS |
| 10613110 慈惠的心腹 | entry, evolve | 2 | no | PASS |
| 10613310 慈爱的天枪 | entry | 1 | no | PASS |
| 10614110 慈爱的凛华·奥尔提雅 | entry, super_evolve | 2 | yes | PASS |
| 10614120 古旧天枪·萨莎妮德 | entry, evolve, faith | 4 | no | PASS |
| 10621110 勇烈的士兵 | entry, other_card_trigger | 2 | no | PASS |
| 10621120 懒惰女仆 | entry | 1 | no | PASS |
| 10621310 天剑授予 | entry | 1 | no | PASS |
| 10622110 忠烈的近卫兵 | entry, evolve, other_card_trigger | 3 | no | PASS |
| 10622120 猫人水手 | entry | 1 | no | PASS |
| 10622310 威风的行军 | entry, countdown, emblem | 5 | no | PASS |
| 10623110 暴烈的参谋 | entry | 1 | no | PASS |
| 10623310 惨烈的天剑 | entry | 1 | no | PASS |
| 10624110 惨烈的剑王·罗德诺艾尔四世 | entry, super_evolve | 2 | no | PASS |
| 10624120 古旧天剑·伊德梅塔 | entry, evolve, faith, other_event_listener | 5 | yes | PASS |
| 10631120 空想的图书管理员 | entry, super_evolve | 2 | no | PASS |
| 10631310 天晶授予 | entry | 1 | yes | PASS |
| 10632110 魔境的学生 | entry | 2 | yes | PASS |
| 10632120 冒险魔导书 | entry, last_words, other_card_trigger | 3 | no | PASS |
| 10632310 正常的侵蚀 | entry | 1 | yes | PASS |
| 10633110 魔醉的教师 | entry, evolve | 2 | no | PASS |
| 10633310 魔恋的天晶 | entry | 1 | yes | PASS |
| 10634110 魔恋的爱慕·希姆 | attack, entry, super_evolve, emblem | 4 | yes | PASS |
| 10634120 古旧天晶·卡卢基典瑟拉 | entry, evolve, faith | 6 | yes | PASS |
| 10641110 决断的龙人 | entry, last_words | 2 | no | PASS |
| 10641120 熟透的海鱼 | entry, last_words, other_card_trigger | 3 | no | PASS |
| 10641310 天刀授予 | entry, other_card_trigger | 2 | no | PASS |
| 10642110 果断的剑圣 | entry | 1 | no | PASS |
| 10642120 尖刺龙 | turn_end, evolve | 2 | no | PASS |
| 10642310 赤流 | entry | 1 | yes | PASS |
| 10643110 隔断的龙斗士 | entry, evolve | 2 | no | PASS |
| 10643310 断头的天刀 | entry, other_card_trigger | 2 | no | PASS |
| 10644110 断头的斩姬·相枛津 | entry | 1 | yes | PASS |
| 10644120 古旧天刀·波菈莱 | evolve, super_evolve, other_card_trigger | 3 | yes | PASS |
| 10651110 渴望的恶魔 | last_words | 1 | no | PASS |
| 10651120 逃避幽灵者 | last_words | 1 | no | PASS |
| 10651310 天眼授予 | entry | 1 | no | PASS |
| 10652110 渴欲的唤灵师 | entry, other_card_trigger | 2 | no | PASS |
| 10652120 失恋恶魔 | entry, evolve, other_card_trigger | 3 | no | PASS |
| 10652310 “最强”的诱惑 | entry | 1 | no | PASS |
| 10653110 渴命的破坏者 | entry, evolve | 2 | no | PASS |
| 10653310 枯渴的天眼 | entry | 1 | yes | PASS |
| 10654110 枯渴的魔神·阿尔弭斯 | clash, super_evolve | 2 | no | PASS |
| 10654120 古旧天眼·比芭提 | entry, evolve | 2 | no | PASS |
| 10661110 崇奉的懦者 | entry, last_words, other_card_trigger | 4 | yes | PASS |
| 10661210 污浊的圣水 | entry, last_words, countdown, other_card_trigger | 5 | no | PASS |
| 10661310 天书授予 | entry | 1 | no | PASS |
| 10662110 崇敬的涂描者 | entry, last_words, other_card_trigger | 3 | no | PASS |
| 10662120 飞马骑手 | entry, evolve | 2 | no | PASS |
| 10662210 救赎的圣典 | entry, last_words, countdown, other_card_trigger | 4 | no | PASS |
| 10663110 崇拜的圣骑士 | entry, evolve, last_words, other_card_trigger | 5 | yes | PASS |
| 10663210 崇高的天书 | entry, last_words, countdown | 3 | no | PASS |
| 10664110 崇高的憎恶·康蒂玛 | entry, super_evolve | 2 | no | PASS |
| 10664120 古旧天书·莲妥丝 | turn_end, entry, last_words, faith | 5 | no | PASS |
| 10671110 低劣的玩具 | entry, other_card_trigger | 3 | yes | PASS |
| 10671120 聪明的创造者 | entry | 1 | no | PASS |
| 10671310 天斧授予 | entry | 1 | no | PASS |
| 10672110 拙劣的人偶 | entry | 1 | yes | PASS |
| 10672120 胆小鬼先锋 | entry | 1 | no | PASS |
| 10672310 平庸的制图 | entry | 1 | no | PASS |
| 10673110 愚劣的兵器 | turn_end, entry, evolve | 3 | yes | PASS |
| 10673310 恶劣的天斧 | entry | 2 | no | PASS |
| 10674110 恶劣的纯心·卡密希拉 | entry, super_evolve | 3 | yes | PASS |
| 10674120 古旧天斧·尤泽塔 | entry | 1 | no | PASS |
| 10701110 纯真孩童 | evolve | 1 | no | PASS |
| 10701310 颓废之泪 | entry | 1 | no | PASS |
| 10702110 神话记者 | super_evolve, last_words | 2 | yes | PASS |
| 10703110 享乐的上级市民 | entry | 1 | no | PASS |
| 10703210 巴别隆城 | turn_end, entry, countdown, other_card_trigger | 5 | yes | PASS |
| 10704110 特殊目标·海雷姆哈妮 | attack, last_words, countdown, emblem | 6 | no | PASS |
| 10704120 巴别隆市长·埃尔塔罗 | turn_end | 1 | no | PASS |
| 10711110 巨型熊 | entry | 1 | no | PASS |
| 10711120 精灵陷阱师 | entry | 1 | no | PASS |
| 10711310 人格切换 | entry | 1 | no | PASS |
| 10712110 绿风细剑师 | entry | 1 | no | PASS |
| 10712120 弓兵指挥者 | entry, evolve | 2 | no | PASS |
| 10712310 忧虑缩小 | entry, countdown, emblem | 5 | no | PASS |
| 10713110 冰箭射手 | turn_end, entry | 2 | no | PASS |
| 10713310 恶意扩大 | entry, countdown, emblem | 5 | no | PASS |
| 10714110 操量的安纳提玛·达斯特迪兹 | turn_end, entry, evolve, countdown, emblem | 6 | no | PASS |
| 10714120 冰界鹿王 | turn_end, entry, super_evolve, countdown, emblem | 7 | no | PASS |
| 10721110 曲行工兵 | entry, super_evolve | 2 | no | PASS |
| 10721120 传调联络兵 | entry | 1 | yes | PASS |
| 10721310 敌我的调律 | entry | 1 | no | PASS |
| 10722110 听略谍报兵 | evolve, last_words | 2 | yes | PASS |
| 10722120 斩奏医护兵 | entry, evolve | 2 | yes | PASS |
| 10722310 无音的包围 | entry | 1 | yes | PASS |
| 10723110 响爪分队长 | entry | 1 | yes | PASS |
| 10723310 带来静寂的拔刀 | entry | 1 | no | PASS |
| 10724110 统音的安纳提玛·吉尔达利娅 | entry, evolve, countdown, emblem | 7 | yes | PASS |
| 10724120 宽严的音帅·塞扎尔 | entry, super_evolve | 2 | yes | PASS |
| 10731110 小巧捕食者 | entry | 1 | no | PASS |
| 10731120 小型怪兽 | entry, evolve | 2 | no | PASS |
| 10731310 召唤仆从 | entry, other_event_listener | 2 | no | PASS |
| 10732110 迷人怪兽 | super_evolve, last_words | 2 | no | PASS |
| 10732120 甜蜜猎食者 | entry | 1 | no | PASS |
| 10732310 暴食的零嘴 | entry | 1 | no | PASS |
| 10733110 甜美存在 | entry, evolve | 2 | no | PASS |
| 10733310 饕餮魔咒 | entry, other_event_listener | 2 | no | PASS |
| 10734110 万食的安纳提玛·拉拉安瑟姆 | turn_end, entry, evolve, countdown, emblem | 6 | no | PASS |
| 10734120 可爱杰作 | entry, super_evolve, last_words | 3 | no | PASS |
| 10741110 宣扬的龙人 | entry, other_card_trigger | 2 | yes | PASS |
| 10741120 载运飞龙 | entry, evolve | 2 | no | PASS |
| 10741310 百无聊赖的睥睨 | entry | 1 | no | PASS |
| 10742110 豪龙守门人 | entry | 1 | no | PASS |
| 10742120 龙族侍者 | turn_end, entry | 2 | no | PASS |
| 10742310 焦龙的午睡 | entry | 1 | no | PASS |
| 10743110 龙人先驱者 | entry | 1 | no | PASS |
| 10743310 黑炎的奔流 | entry | 1 | no | PASS |
| 10744110 焦灰的安纳提玛·班德奈特 | turn_start, entry, super_evolve, emblem, other_event_listener | 6 | yes | PASS |
| 10744120 龙峪的古龙 | turn_end, entry, super_evolve, countdown, emblem | 6 | no | PASS |
| 10751110 暗夜键盘手·露露米 | last_words | 1 | no | PASS |
| 10751120 恶魔鼓手·拉兹 | evolve, last_words | 2 | yes | PASS |
| 10751310 灵魂调律 | entry | 1 | no | PASS |
| 10752110 猫咪走绳师 | entry, evolve | 2 | no | PASS |
| 10752120 乌鸦杂耍师 | entry | 1 | no | PASS |
| 10752310 讴歌青春 | entry | 1 | no | PASS |
| 10753110 骸骨驯兽师 | entry, super_evolve | 3 | no | PASS |
| 10753310 夜之歌的演唱会 | entry | 1 | no | PASS |
| 10754110 傍死的安纳提玛·徒姬 | entry, super_evolve | 3 | no | PASS |
| 10754120 死亡主持人·马克米朗 | entry | 2 | no | PASS |
| 10761110 营利支援者 | entry, last_words | 2 | no | PASS |
| 10761120 广域传教士 | entry, evolve | 2 | no | PASS |
| 10761210 阳光耳饰 | entry, other_card_trigger | 2 | no | PASS |
| 10762110 神圣策划人 | entry | 1 | no | PASS |
| 10762120 传言圣鸟 | entry | 1 | no | PASS |
| 10762210 完美的时钟 | entry, other_card_trigger | 3 | no | PASS |
| 10763110 审理的守卫 | entry, last_words | 2 | no | PASS |
| 10763210 海蚀三叉戟 | entry, other_card_trigger | 2 | no | PASS |
| 10764110 裁神的安纳提玛·罗德欧 | entry, evolve | 2 | no | PASS |
| 10764120 崇拜经理人·伊尼西雅 | entry, super_evolve | 2 | no | PASS |
| 10771110 个性店主 | entry, evolve | 2 | no | PASS |
| 10771120 炫酷舞者 | entry | 1 | no | PASS |
| 10771310 跑酷 | entry | 1 | yes | PASS |
| 10772110 悠然的滑手 | entry, evolve | 2 | yes | PASS |
| 10772120 大胆的涂鸦师 | entry | 1 | no | PASS |
| 10772310 闪光一瞬 | entry | 1 | no | PASS |
| 10773110 狂野播报员 | entry | 2 | yes | PASS |
| 10773310 瞬移斩击 | entry | 1 | no | PASS |
| 10774110 虚刻的安纳提玛·斯卡雷特 | entry | 1 | yes | PASS |
| 10774120 奋厉追赶·米乌 | entry, evolve, super_evolve | 3 | yes | PASS |
| 10801110 自律的圣鸟·汉萨 | evolve | 1 | no | PASS |
| 10801120 无尽旅途·蕾娜 | entry | 1 | no | PASS |
| 10802110 激动的欢喜·阿尔菲德 | entry, evolve | 2 | no | PASS |
| 10802310 救世的英姿 | entry | 1 | no | PASS |
| 10803110 遗忘的纯真·爱卡 | entry, evolve | 2 | no | PASS |
| 10803310 传承的意志 | entry | 1 | yes | PASS |
| 10804110 阿尔比昂巴哈姆特 | entry | 1 | yes | PASS |
| 10804120 高洁的黑翼·奥莉薇 | entry | 1 | yes | PASS |
| 10811110 昔日的天秤·马龙 | entry | 1 | no | PASS |
| 10811120 异端隐士·西特拉斯 | entry, evolve | 2 | yes | PASS |
| 10811130 忧郁少女·莫埃尔 | entry | 1 | no | PASS |
| 10812110 太古的妖精·露芙蕾 | last_words, other_event_listener | 2 | yes | PASS |
| 10812120 情念的毒荆·莉柯瑞丝 | entry | 1 | yes | PASS |
| 10812310 宁静的孤独 | entry | 1 | no | PASS |
| 10813110 温柔读心者·米榭儿 | entry | 1 | yes | PASS |
| 10813310 水镜的信赖 | entry | 1 | yes | PASS |
| 10814110 离合有终·赛德斯&梅希亚 | entry | 1 | yes | PASS |
| 10814120 永恒冰晶·蒂亚 | entry, other_event_listener, other_card_trigger | 3 | yes | PASS |
| 10821110 武力与治安·娜哈特·娜哈特&宾森特 | entry, super_evolve | 2 | yes | PASS |
| 10821120 寡言的刺客·夏伊莉 | evolve | 1 | no | PASS |
| 10821130 悠久的骑士·莎夏 | entry, evolve | 2 | no | PASS |
| 10822110 越狱者·卡婕 | entry, evolve | 2 | no | PASS |
| 10822120 织田信长 | entry | 1 | no | PASS |
| 10822310 重历新生 | entry | 1 | no | PASS |
| 10823110 冲田总司 | attack, entry | 2 | no | PASS |
| 10823310 相伴相随的日常 | entry | 1 | no | PASS |
| 10824110 天命的子弹·巴妮&巴隆 | entry, evolve | 2 | yes | PASS |
| 10824120 焦灼炎将·玛尔斯 | entry, super_evolve | 3 | yes | PASS |
| 10831110 森绿的恩惠·喵鲁&圆滚滚2号&吉娜 | entry, evolve | 2 | no | PASS |
| 10831120 玛纳利亚书记官·波比 | entry | 1 | no | PASS |
| 10831310 暴风破 | entry | 1 | no | PASS |
| 10832110 快乐绽花·萨米&玛莉 | entry | 1 | no | PASS |
| 10832310 其乐融融的团聚 | entry | 1 | no | PASS |
| 10832320 伏地雷击 | entry | 1 | no | PASS |
| 10833110 玛纳利亚文书官·琪可 | entry, evolve, super_evolve, emblem | 5 | no | PASS |
| 10833310 钢铁的小憩 | entry | 1 | yes | PASS |
| 10834110 恩爱的大地·坦忒拉&拉缇卡 | entry | 1 | no | PASS |
| 10834120 灾难言灵·洋荷 | entry, evolve | 3 | no | PASS |
| 10841110 狼人族首领·契特 | entry | 1 | no | PASS |
| 10841120 沙尘守宝龙 | last_words | 1 | no | PASS |
| 10841130 沧海之精 | entry, evolve, emblem | 4 | no | PASS |
| 10842110 闪耀旋律·莉芙&萝萝 | turn_end, entry | 2 | no | PASS |
| 10842120 满面笑容的烹饪·琪米卡 | entry, evolve | 2 | yes | PASS |
| 10842310 末世死化妆 | entry | 1 | no | PASS |
| 10843110 穹顶的战火·杰亚达 | attack | 1 | no | PASS |
| 10843310 狐火蜃景 | entry | 1 | no | PASS |
| 10844110 反照的赤红·德莱克&亚瑞札特 | entry, last_words, countdown, emblem | 6 | no | PASS |
| 10844120 金银绚烂·璐米欧儿&雅尔贞特 | entry, super_evolve | 2 | yes | PASS |
| 10851120 可爱恶魔·莉莉姆 | attack, last_words | 2 | no | PASS |
| 10851130 兔耳恶魔·莉蜜儿 | entry, evolve | 2 | no | PASS |
| 10852110 母爱恶魔·菲欧蕾 | entry | 2 | no | PASS |
| 10852120 黑暗骑士·玛莎 | entry, evolve | 2 | no | PASS |
| 10852310 启程的退场 | entry | 1 | no | PASS |
| 10853110 诚实的诅咒师·丝姬 | entry, evolve | 2 | no | PASS |
| 10853310 改变的流向 | entry | 1 | no | PASS |
| 10854110 出发的憧憬·苇剑&武津御 | entry, evolve | 2 | yes | PASS |
| 10854120 日月的蔷薇·赛蕾丝 | turn_end, clash, entry | 3 | yes | PASS |
| 10861110 图书室的魔女·莉莉尤姆 | evolve, last_words | 2 | yes | PASS |
| 10861120 勤劳的女祭司·泰瑞莎 | entry | 1 | no | PASS |
| 10861130 亡灵猎人·格兰特 | entry, evolve | 2 | no | PASS |
| 10862110 天阳的使徒·艾迪特 | super_evolve, last_words | 2 | no | PASS |
| 10862120 深渊探究者·维切 | super_evolve | 1 | yes | PASS |
| 10862310 威胁的残渣 | entry | 1 | no | PASS |
| 10863110 圣洁驱魔人·珂蕾特 | entry, evolve | 2 | yes | PASS |
| 10863210 同窗好友 | turn_end, entry, countdown, other_card_trigger | 4 | yes | PASS |
| 10864110 飞跃的姐妹·贝尔迪俪亚&卡诗黛儿 | attack, entry, super_evolve, emblem | 4 | yes | PASS |
| 10864120 希望的光彩·莉迪耶尔 | entry, evolve, countdown, emblem | 6 | yes | PASS |
| 10871110 满溢的幸福·库伦特司 | last_words | 1 | no | PASS |
| 10871120 过度守护者·莉欧娜 | super_evolve | 1 | no | PASS |
| 10871130 器械操纵者·吉尔克 | entry | 1 | no | PASS |
| 10872110 人造的馈赠·蕾拉 | last_words | 1 | no | PASS |
| 10872120 门扉接续者·拉姿莉 | entry | 1 | yes | PASS |
| 10872310 纯净无垢的日常 | entry | 1 | no | PASS |
| 10873110 知恩图报·米莉亚姆 | entry | 1 | no | PASS |
| 10873310 开辟未来 | entry | 1 | no | PASS |
| 10874110 决断的交错·亚修雷&莉缇雅 | entry, evolve | 2 | yes | PASS |
| 10874120 你的前辈·欧丝 | entry, evolve | 2 | yes | PASS |
| 90004130 边狱的邪祟 | entry | 1 | no | PASS |
| 90004310 阿斯塔罗特的宣判 | entry | 1 | no | PASS |
| 90004320 绝大的证明 | entry | 1 | no | PASS |
| 90004330 涸绝的甘露 | entry | 1 | yes | PASS |
| 90011120 新绿的妖精 | turn_end | 1 | yes | PASS |
| 90011310 森林的奥秘 | entry | 1 | no | PASS |
| 90014310 蔷薇之闪击 | entry | 1 | no | PASS |
| 90014320 绝命的痛击 | turn_end, entry | 2 | no | PASS |
| 90014330 天枪深渊 | entry | 1 | no | PASS |
| 90021310 黄金短剑 | entry | 1 | no | PASS |
| 90021320 黄金之杯 | entry | 1 | no | PASS |
| 90021330 黄金之靴 | entry | 1 | no | PASS |
| 90021340 黄金项链 | entry | 1 | no | PASS |
| 90021350 闪耀的金币 | entry | 1 | no | PASS |
| 90023110 安静的女仆·诺嘉 | entry | 1 | no | PASS |
| 90024310 空绝的残光 | entry | 1 | no | PASS |
| 90024320 天剑深渊 | entry | 1 | yes | PASS |
| 90024330 亡命者的枪击 | entry | 1 | yes | PASS |
| 90031130 式神·小纸人 | last_words | 1 | no | PASS |
| 90031140 式神·暴鬼 | last_words | 1 | no | PASS |
| 90031210 大地之魔片 | other_card_trigger | 1 | no | PASS |
| 90031310 玛纳利亚魔弹 | entry | 1 | no | PASS |
| 90032110 洋葱军团兵 | attack | 1 | no | PASS |
| 90033310 有所成长了！ | entry | 1 | no | PASS |
| 90034110 式神·天后 | last_words | 1 | no | PASS |
| 90034120 式神·贵人 | entry | 1 | no | PASS |
| 90034130 安的巨大英灵 | turn_end | 1 | no | PASS |
| 90034310 绝尽的伪证 | entry | 1 | no | PASS |
| 90034320 伟大之术 | entry | 1 | no | PASS |
| 90034330 天晶深渊 | entry | 1 | yes | PASS |
| 90034340 苍奏之四 | entry | 1 | no | PASS |
| 90034350 宏大的回归 | entry | 1 | no | PASS |
| 90044310 银冰吐息 | entry | 1 | no | PASS |
| 90044320 烈绝的灭牙 | entry | 1 | yes | PASS |
| 90044330 天刀深渊 | entry, other_card_trigger | 2 | yes | PASS |
| 90051130 怨灵 | turn_end | 1 | no | PASS |
| 90051140 腐臭的僵尸 | last_words | 1 | no | PASS |
| 90054110 守卫犬的右腕·米米 | last_words | 1 | no | PASS |
| 90054120 守卫犬的左腕·可可 | last_words | 1 | no | PASS |
| 90054130 一尾狐 | last_words | 1 | no | PASS |
| 90054310 绝叫的扩散 | entry | 1 | yes | PASS |
| 90054320 爱绝的飞翔 | entry | 1 | yes | PASS |
| 90054330 天眼深渊 | entry | 1 | no | PASS |
| 90064210 月影指环 | turn_end, entry, countdown, other_card_trigger | 4 | no | PASS |
| 90064310 绝望的奔流 | entry | 1 | no | PASS |
| 90064320 天书深渊 | entry | 1 | no | PASS |
| 90071110 悬丝傀儡 | turn_end | 1 | yes | PASS |
| 90071120 改良型·悬丝傀儡 | turn_end | 1 | no | PASS |
| 90071130 解析的创造物 | entry | 1 | yes | PASS |
| 90072130 屠戮人偶 | last_words | 1 | no | PASS |
| 90073110 毁灭创造物α | turn_end | 1 | yes | PASS |
| 90073120 毁灭创造物β | turn_end | 1 | yes | PASS |
| 90073130 毁灭创造物γ | turn_end | 1 | yes | PASS |
| 90074110 卓越创造物Ω | entry | 1 | yes | PASS |
| 90074130 维多利亚 | attack | 1 | no | PASS |
| 90074210 新约·白之章 | entry, last_words, countdown, other_card_trigger | 4 | yes | PASS |
| 90074220 新约·黑之章 | entry, last_words, countdown, other_card_trigger | 4 | yes | PASS |
| 90074310 奏绝的独唱 | entry | 1 | yes | PASS |
| 90074320 天斧深渊 | entry | 1 | no | PASS |
