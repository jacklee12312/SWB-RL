# Target, choice, and pending-state audit

- Result: **PASS**; 0 failures.
- Snapshot: 826 cards (735 collectible / 91 generated).
- Target/choice sources: 477 cards; 90 in the training closure; 21 global sources.
- Manual target kinds: 14; contract failures: 0.

## Operation categories

| Category | Operations |
|---|---:|
| manual | 245 |
| random | 128 |
| all | 241 |
| implicit_or_bound | 103 |
| decision | 74 |

## Manual target-domain matrix

| Target | Empty | Populated | Result |
|---|---:|---:|:---:|
| own_unit | 0 | 2 | PASS |
| enemy_unit | 0 | 2 | PASS |
| own_unit_or_leader | 1 | 3 | PASS |
| enemy_unit_or_leader | 1 | 3 | PASS |
| any_unit_or_leader | 2 | 6 | PASS |
| own_board | 0 | 4 | PASS |
| enemy_board | 0 | 4 | PASS |
| any_unit | 0 | 4 | PASS |
| own_amulet | 0 | 2 | PASS |
| enemy_amulet | 0 | 2 | PASS |
| any_amulet | 0 | 4 | PASS |
| any_board | 0 | 8 | PASS |
| own_hand | 0 | 2 | PASS |
| own_graveyard_card | 0 | 2 | PASS |

## Behavioral contracts

| Group | Case | Result |
|---|---|:---:|
| candidate_cardinalities | 0 | PASS |
| candidate_cardinalities | 1 | PASS |
| candidate_cardinalities | 2 | PASS |
| source_exclusion | 0 | PASS |
| restrictions | cannot_be_targeted_affects_manual_enemy_effect_only | PASS |
| restrictions | ambush_affects_manual_enemy_effect_only | PASS |
| restrictions | ward_and_ignore_ward_affect_combat_not_effect_choice | PASS |
| multi_target | distinct_targets_preserve_selection_order | PASS |
| multi_target | duplicate_targets_apply_in_order_when_allowed | PASS |
| multi_target | candidate_shortage_truncates_without_duplicates | PASS |
| stale_targets | target_died | PASS |
| stale_targets | target_left_play | PASS |
| stale_targets | target_transformed | PASS |
| stale_targets | target_changed_controller | PASS |
| stale_targets | target_failed_filter | PASS |
| source_leaving | bound_target_survives_source_leaving | PASS |
| source_leaving | source_dependent_operation_skips_but_queue_continues | PASS |
| mixed_target_order | 0 | PASS |
| no_candidate_policies | required_selected_target_prohibits_play | PASS |
| no_candidate_policies | unavailable_selected_operation_skips | PASS |
| no_candidate_policies | random_and_all_no_candidates_are_safe_noops | PASS |
| no_candidate_policies | target_exists_executes_else_branch | PASS |
| snapshot_restore | 0 | PASS |
| action_order | 0 | PASS |

## Source inventory

| Card | Categories | Operations | Training | Result |
|---|---|---:|:---:|:---:|
| 999001 synthetic-demo-999001 | manual | 1 | no | PASS |
| 999101 synthetic-demo-999101 | manual | 1 | no | PASS |
| 999102 synthetic-demo-999102 | random | 1 | no | PASS |
| 999103 synthetic-demo-999103 | all | 1 | no | PASS |
| 999952 synthetic-demo-999952 | decision | 1 | no | PASS |
| 999953 synthetic-demo-999953 | decision | 1 | no | PASS |
| 999955 synthetic-demo-999955 | decision, manual | 2 | no | PASS |
| 999956 synthetic-demo-999956 | decision | 1 | no | PASS |
| 10001210 侦探的放大镜 | manual | 1 | no | PASS |
| 10002210 冒险者公会 | manual | 1 | no | PASS |
| 10011210 缭乱之庭 | random | 1 | no | PASS |
| 10012110 冒险精灵·小梅 | manual | 1 | no | PASS |
| 10012120 音速射手·塞尔文 | manual | 1 | no | PASS |
| 10012310 昆虫的忠告 | decision, manual, random | 3 | no | PASS |
| 10021310 女仆的礼仪 | manual | 1 | no | PASS |
| 10022120 魔煌的诡谲者·拉斯提 | implicit_or_bound | 1 | no | PASS |
| 10031110 闪光魔法剑士 | all, decision | 2 | no | PASS |
| 10032110 双面魔女·蕾米拉米 | implicit_or_bound, manual | 2 | no | PASS |
| 10032310 魔爆 | all | 1 | no | PASS |
| 10041110 烈焰火蜥蜴 | manual | 1 | no | PASS |
| 10041310 龙人碎击 | manual | 2 | no | PASS |
| 10042110 猛攻的龙战士 | all, manual | 2 | no | PASS |
| 10051310 混沌诅咒 | decision | 1 | yes | PASS |
| 10052120 多情的唤灵师 | implicit_or_bound | 2 | no | PASS |
| 10052310 捕食灵魂 | manual | 1 | no | PASS |
| 10061130 圣翼战士 | manual | 2 | no | PASS |
| 10062110 铁拳神父 | all, manual | 2 | no | PASS |
| 10071310 来自异次元的枪击 | manual | 1 | no | PASS |
| 10101110 贪婪的智天使·露比 | manual | 1 | no | PASS |
| 10102110 迸发的光明·阿波罗 | all | 2 | no | PASS |
| 10103110 爽朗的天宫·菲尔德亚 | manual | 1 | no | PASS |
| 10103310 神之雷霆 | all, random | 2 | no | PASS |
| 10104110 勇武的堕天使·奥莉薇 | manual | 1 | no | PASS |
| 10111110 舞动的妖精 | all | 1 | no | PASS |
| 10111150 言传的杂草人长老 | random | 2 | no | PASS |
| 10112110 霜寒冰晶·艾琳 | manual | 1 | no | PASS |
| 10112130 年幼宝石兽 | manual | 1 | no | PASS |
| 10112210 磷光辉岩 | manual | 1 | no | PASS |
| 10113110 纯洁冰晶·莉莉 | manual | 2 | no | PASS |
| 10113120 薰交的天宫·巴克伍德 | all | 1 | no | PASS |
| 10113130 煌击战士·贝鲁 | manual | 1 | no | PASS |
| 10113210 圣树法杖 | manual | 1 | no | PASS |
| 10114120 丰丽的玫瑰皇后 | all | 1 | no | PASS |
| 10114130 起源剑师·阿玛兹 | random | 1 | no | PASS |
| 10121110 爱之骑士·尹安 | manual | 1 | no | PASS |
| 10121120 和平商人·艾尔涅丝塔 | all | 2 | no | PASS |
| 10121310 剑士的斩击 | manual | 1 | no | PASS |
| 10122310 王断的威光 | all, decision | 2 | no | PASS |
| 10123120 沉默的狙击手·瓦路兹 | manual | 1 | no | PASS |
| 10123130 王断的天宫·斯塔奇乌姆 | all | 1 | no | PASS |
| 10123310 触手撕咬 | manual | 1 | no | PASS |
| 10124110 雷维翁的迅雷·阿尔贝尔 | all | 1 | no | PASS |
| 10124120 白银骑士团团长·艾蜜莉亚 | all | 1 | no | PASS |
| 10131110 符文剑操控师 | manual | 1 | no | PASS |
| 10131120 见习占星术师 | manual | 1 | no | PASS |
| 10131130 唤枭士 | manual | 1 | no | PASS |
| 10131140 追梦的企鹅魔法师 | all | 1 | no | PASS |
| 10131310 彩虹奇迹 | manual | 1 | no | PASS |
| 10131320 暴风破 | manual | 1 | yes | PASS |
| 10132110 惹人怜爱的教师·米兰 | all, manual | 3 | no | PASS |
| 10132120 奇迹女巫·爱蜜儿 | all | 1 | no | PASS |
| 10132130 玛纳利亚的学生·威廉 | all | 2 | no | PASS |
| 10132310 理光的证明 | all, decision | 2 | no | PASS |
| 10132320 雪人觉醒 | implicit_or_bound, manual | 2 | no | PASS |
| 10133110 黎明炼金术师·诺诺 | manual | 1 | no | PASS |
| 10133130 理光的天宫·艾德薇诗 | random | 1 | no | PASS |
| 10134110 五行之巅·久苑 | all, manual | 2 | no | PASS |
| 10134120 玛纳利亚密友·安&古蕾雅 | all, manual | 2 | no | PASS |
| 10134310 超越次元 | all | 2 | no | PASS |
| 10141130 初出茅庐的屠龙者 | manual | 1 | no | PASS |
| 10141310 灾祸吐息 | all | 1 | no | PASS |
| 10142110 煌牙的义勇·基德 | random | 1 | no | PASS |
| 10142130 读风者·杰鲁 | manual | 1 | no | PASS |
| 10142140 艳丽龙人·玛利翁 | manual | 2 | no | PASS |
| 10143140 夜幕龙 | all | 1 | no | PASS |
| 10143210 乙姬的宝扇 | manual | 1 | no | PASS |
| 10144110 灼热的安纳提玛·班德奈特 | all, manual | 2 | no | PASS |
| 10144130 龙人演义·卧龙 | all | 2 | no | PASS |
| 10151150 禁约恶魔 | manual | 1 | no | PASS |
| 10151310 死神挥刀 | manual | 2 | no | PASS |
| 10152140 穿刺公·弗拉德 | manual | 1 | no | PASS |
| 10153120 燃烧魔剑·欧特鲁斯 | random | 1 | no | PASS |
| 10153310 蛇神之怒 | decision, manual | 2 | no | PASS |
| 10154110 奔放的狱焰·凯尔贝洛斯 | all | 1 | no | PASS |
| 10154130 无极猎人·阿拉加维 | all | 2 | no | PASS |
| 10161110 圣心光棱牧师 | manual | 1 | no | PASS |
| 10161310 翎雨 | all | 1 | no | PASS |
| 10162210 禁密的圣地 | manual | 1 | no | PASS |
| 10162220 神圣注射 | manual | 1 | no | PASS |
| 10163110 终焉的白骨圣堂之主 | all | 2 | no | PASS |
| 10163120 禁密的天宫·罗纳维罗 | manual | 1 | no | PASS |
| 10163220 邪教法器 | all | 1 | no | PASS |
| 10164110 裁决的安纳提玛·罗德欧 | all, manual, random | 3 | no | PASS |
| 10164120 纯白圣女·贞德 | all | 2 | no | PASS |
| 10164130 水之守护神·萨蕾法 | all | 1 | no | PASS |
| 10171130 永不停火的枪手 | manual | 1 | no | PASS |
| 10172130 杀意之丝·诺亚 | all | 1 | no | PASS |
| 10172310 生命的奔流 | manual | 1 | no | PASS |
| 10172320 改境的重启 | implicit_or_bound, manual | 2 | no | PASS |
| 10173120 箱庭的断罪者·希尔薇娅 | decision, manual | 3 | no | PASS |
| 10173130 疯狂的创造者·历亚姆 | all | 2 | no | PASS |
| 10173140 改境的天宫·阿洛艾特 | manual | 1 | no | PASS |
| 10173210 遗产的炮击 | random | 1 | no | PASS |
| 10174130 增幅加速·洛拉米亚 | all, manual | 2 | no | PASS |
| 10201110 双刀哥布林 | manual | 1 | no | PASS |
| 10201310 逆向变化 | manual | 1 | no | PASS |
| 10203110 联结的天使·蕾娜 | all | 1 | no | PASS |
| 10203120 雷火双神·福尼加尔&亚文哈尔 | random | 1 | no | PASS |
| 10204110 命运黄昏·奥丁 | manual | 1 | no | PASS |
| 10211120 木锤矮人 | all, manual | 2 | no | PASS |
| 10212310 来自树上的偷袭 | random | 3 | no | PASS |
| 10213110 森林骑士道·辛西亚 | all | 1 | no | PASS |
| 10214110 翅翼女王·提泰妮娅 | manual | 1 | no | PASS |
| 10214120 缠绕密林·丽梅格 | decision, implicit_or_bound | 4 | no | PASS |
| 10221110 扳机女仆·赛莉亚 | random | 1 | no | PASS |
| 10222110 无畏的副团长·格尔德 | all | 1 | no | PASS |
| 10222310 三将姬的乱击 | random | 2 | no | PASS |
| 10223110 剑士公主·萝泽 | implicit_or_bound | 1 | no | PASS |
| 10223120 假日中的王女·普莉姆 | all | 1 | no | PASS |
| 10224110 静寂的安纳提玛·吉尔达利娅 | all, implicit_or_bound | 3 | no | PASS |
| 10231120 魔导图书管理员 | manual | 1 | no | PASS |
| 10231310 冰锥穿击 | manual | 1 | no | PASS |
| 10232120 调香的魔法师 | manual | 2 | no | PASS |
| 10232310 混沌赤焰 | all | 1 | no | PASS |
| 10233110 玛纳利亚剑士·欧文 | all | 1 | no | PASS |
| 10234110 暴食的安纳提玛·拉拉安瑟姆 | manual | 1 | no | PASS |
| 10234120 精金炼金术师·诺曼 | decision, implicit_or_bound | 4 | no | PASS |
| 10242210 炎龙之剑 | implicit_or_bound, manual | 2 | no | PASS |
| 10243310 龙骑突击 | all, manual | 2 | no | PASS |
| 10244110 银冰龙少女·菲琳 | all | 1 | no | PASS |
| 10251110 银色子弹·雷文 | manual | 1 | no | PASS |
| 10253110 悲惨战争·萝拉 | manual | 2 | no | PASS |
| 10254110 双轮夜行·吟雪&夕月 | decision, random | 2 | no | PASS |
| 10261110 粉碎的圣职者 | manual | 1 | no | PASS |
| 10261120 恶意的神谕·达姆斯 | manual | 1 | no | PASS |
| 10262110 弹幕驱魔人·珂蕾特 | random | 2 | no | PASS |
| 10262120 有洁癖的审判者 | manual | 1 | no | PASS |
| 10262310 神圣守护 | manual, random | 2 | no | PASS |
| 10263110 速断之刃·阿尼耶丝 | random | 1 | no | PASS |
| 10264110 天之守护神·埃忒耳 | all | 2 | no | PASS |
| 10271120 猫偶 | implicit_or_bound | 1 | no | PASS |
| 10271210 创造物弹射器 | implicit_or_bound, manual | 2 | no | PASS |
| 10272110 心灵屠戮者·菲亚 | manual | 1 | no | PASS |
| 10272120 绝望之王·阿基姆 | implicit_or_bound, manual | 2 | no | PASS |
| 10272310 伊卡洛斯的飞翔 | decision, implicit_or_bound | 3 | no | PASS |
| 10273110 暗狱的余晖·贾丝珀 | decision, implicit_or_bound | 3 | no | PASS |
| 10274120 精神武艺·迦尔拉 | manual | 1 | no | PASS |
| 10301110 涸绝的使徒 | manual | 1 | no | PASS |
| 10304120 涸绝的显现·吉尔内莉莎 | manual | 2 | yes | PASS |
| 10311110 不弑的肯定者 | manual | 2 | no | PASS |
| 10311310 野性的猛袭 | manual | 1 | no | PASS |
| 10312110 不弑的祈祷者 | all | 2 | no | PASS |
| 10312210 不弑之乡 | manual | 2 | no | PASS |
| 10313310 驱逐的死矢 | random | 1 | no | PASS |
| 10314110 不弑的继承者·库露露 | all | 1 | yes | PASS |
| 10314120 绝命的显现·艾斯迪亚 | manual | 1 | no | PASS |
| 10321120 剑圣的同胞 | implicit_or_bound | 1 | no | PASS |
| 10321310 护盾强袭 | manual, random | 2 | no | PASS |
| 10322120 活泼的斥候 | manual | 1 | no | PASS |
| 10322210 篡夺的据点 | decision | 1 | no | PASS |
| 10323110 篡夺的团结者 | random | 2 | no | PASS |
| 10323310 奉还的剑闪 | random | 1 | no | PASS |
| 10324110 篡夺的继承者·辛瑟莱兹 | all | 2 | no | PASS |
| 10332110 真理的祈祷者 | all | 1 | no | PASS |
| 10332210 真理的研究设施 | decision, implicit_or_bound | 3 | no | PASS |
| 10333310 虚假的术式 | manual, random | 2 | yes | PASS |
| 10334110 真理的继承者·蓓哈丽雅 | decision, implicit_or_bound | 3 | no | PASS |
| 10334120 绝尽的显现·莱奥 | random | 1 | no | PASS |
| 10341120 风雪龙人 | all | 1 | no | PASS |
| 10341310 雷霆之怒 | all | 2 | no | PASS |
| 10342210 侮蔑之国 | all | 1 | no | PASS |
| 10343110 侮蔑的团结者 | all | 2 | no | PASS |
| 10343310 威猛炽焰 | manual, random | 2 | no | PASS |
| 10344110 侮蔑的继承者·安吉拉弗利特 | all | 2 | no | PASS |
| 10344120 烈绝的显现·嘉尔缪 | random | 1 | yes | PASS |
| 10351110 混融的肯定者 | all, decision | 4 | yes | PASS |
| 10351120 泡沫鬼姬 | manual | 1 | no | PASS |
| 10351310 前进的暴虐 | random | 1 | no | PASS |
| 10352110 混融的祈祷者 | decision, random | 4 | yes | PASS |
| 10352210 混融之城 | decision, random | 4 | yes | PASS |
| 10353110 混融的团结者 | all, decision, implicit_or_bound | 8 | yes | PASS |
| 10353310 叫唤与憎恶 | decision, random | 2 | yes | PASS |
| 10354110 混融的继承者·莎木·纳克雅 | decision, implicit_or_bound | 3 | yes | PASS |
| 10354120 绝叫与爱绝的显现·鲁鲁纳伊&巴娜蕾卡 | decision | 1 | yes | PASS |
| 10361310 圣辉闪烁 | all | 1 | no | PASS |
| 10363110 安息的团结者 | manual | 1 | no | PASS |
| 10363210 闪耀的失意 | all | 1 | no | PASS |
| 10364110 安息的继承者·妃花 | all | 1 | no | PASS |
| 10371110 破坏的肯定者 | all | 1 | no | PASS |
| 10371120 音速飞行兵 | manual | 1 | no | PASS |
| 10371310 丝线突袭 | manual | 1 | no | PASS |
| 10372110 破坏的祈祷者 | manual, random | 4 | no | PASS |
| 10372120 现场工程师 | manual | 1 | no | PASS |
| 10372210 破坏的荒野 | manual | 1 | yes | PASS |
| 10373110 破坏的团结者 | all, random | 2 | yes | PASS |
| 10373310 歼灭的歌声 | manual | 1 | yes | PASS |
| 10374110 破坏的继承者·阿克西娅 | all | 1 | yes | PASS |
| 10401110 驰骋天空的守护者·卡塔莉娜 | random | 1 | no | PASS |
| 10402110 宙域使者·尤妮 | all | 1 | no | PASS |
| 10403110 征服苍空的骑空士·古兰&姬塔 | decision, random | 2 | yes | PASS |
| 10404110 天司长的继承者·圣德芬 | random | 5 | yes | PASS |
| 10411110 爱恨舞者·晧&曜 | all | 1 | no | PASS |
| 10411120 可爱如琬似花·玛娜玛尔 | all | 1 | no | PASS |
| 10411310 彗星 | manual | 1 | no | PASS |
| 10412110 美妆少女·克洛伊 | manual | 1 | no | PASS |
| 10412120 绯焰舞姬·安苏莉娅 | all | 1 | no | PASS |
| 10412310 绮罗星 | random | 1 | no | PASS |
| 10413110 幻彩弓手·丘比丹 | random | 1 | yes | PASS |
| 10413310 亚尔夫海姆 | all, decision | 9 | no | PASS |
| 10414120 调和的舞者·尤艾尔&苏丝雅 | random | 1 | yes | PASS |
| 10422110 冰心霸王·艾格罗瓦尔 | all | 1 | no | PASS |
| 10423110 真王之刃·黄金骑士 | all, decision | 3 | no | PASS |
| 10423310 骁勇骑士 | all, decision | 4 | no | PASS |
| 10424110 真红与群青·塞达&贝阿朵丽丝 | implicit_or_bound | 1 | yes | PASS |
| 10424120 十天众统领·希耶提 | all | 2 | yes | PASS |
| 10431120 流浪的家庭教师·斯芙拉玛尔 | all | 1 | no | PASS |
| 10431310 符文秘术 | all | 1 | yes | PASS |
| 10432110 报仇的占卜师·艾塞克莱因 | manual | 1 | no | PASS |
| 10432120 荆棘旅途·米蕾耶&莉赛特 | implicit_or_bound | 1 | no | PASS |
| 10432310 能量外溢 | decision, random | 3 | no | PASS |
| 10433110 缅怀之火·埃尔默特 | implicit_or_bound, manual | 2 | no | PASS |
| 10433310 炼金炎爆 | manual | 1 | no | PASS |
| 10434110 水之法则·瓦姆杜斯 | all, decision | 4 | no | PASS |
| 10441120 梅格的挚友·玛丽亲 | random | 1 | no | PASS |
| 10442110 冰封的命运·伊什米尔 | all | 1 | no | PASS |
| 10442120 淳朴的钢铁之躯·无限 | manual | 1 | no | PASS |
| 10442310 至爱狂轰 | implicit_or_bound, manual | 2 | no | PASS |
| 10443310 星晶兽吸收之力 | implicit_or_bound, manual | 2 | no | PASS |
| 10444110 炎之法则·威尔纳斯 | manual | 2 | yes | PASS |
| 10452110 霸空武神·哪吒 | random | 2 | no | PASS |
| 10452130 元素共鸣·巴尔 | decision, random | 3 | yes | PASS |
| 10453110 生与死之技·涅槃 | all | 1 | no | PASS |
| 10453310 堕落 | all | 1 | no | PASS |
| 10454110 暗之法则·菲迪埃尔 | all, implicit_or_bound | 3 | no | PASS |
| 10454120 狡诈的堕天司·彼列 | all | 2 | no | PASS |
| 10461120 克己复礼的修女·拉姆蕾达 | all | 1 | no | PASS |
| 10461210 莉莉艾的鼓舞 | manual | 1 | no | PASS |
| 10462110 沙神的巫女·莎拉 | manual | 1 | no | PASS |
| 10462120 赞恩教僧侣·索菲娅 | all | 1 | no | PASS |
| 10462210 骑驰天空之艇 | manual | 1 | no | PASS |
| 10463110 魔杖傍身的外科医生·缇可 | manual | 1 | no | PASS |
| 10463210 蕾·菲耶的宝石 | decision, random | 2 | yes | PASS |
| 10464110 土之法则·伽莱翁 | random | 1 | no | PASS |
| 10464120 威严的星晶骑士·薇拉 | manual | 1 | no | PASS |
| 10471120 爆燃老大·翼 | all | 1 | no | PASS |
| 10472110 轰雷闪狼·尤斯提斯 | decision, manual | 2 | no | PASS |
| 10472120 严厉的教官·伊尔莎 | decision, random | 4 | no | PASS |
| 10472310 身无长物唯有石 | random | 6 | no | PASS |
| 10473110 向往天空的回归者·卡西乌斯 | all, decision | 2 | no | PASS |
| 10473310 混沌军势 | all | 2 | no | PASS |
| 10474110 光之法则·龙敖 | all, random | 2 | no | PASS |
| 10474120 唯一王者·别西卜 | decision, implicit_or_bound | 3 | yes | PASS |
| 10501110 挥毫的怪物 | decision | 1 | no | PASS |
| 10502110 星辉女神 | manual | 1 | no | PASS |
| 10502120 手持军配团扇的伟丈夫 | all | 1 | yes | PASS |
| 10503310 《世界》的呈现 | all, random | 2 | yes | PASS |
| 10504110 八界花·下天央 | all, decision, implicit_or_bound | 3 | no | PASS |
| 10511310 虫风花的飞翔 | all | 1 | no | PASS |
| 10512110 新晋搭档 | implicit_or_bound | 1 | no | PASS |
| 10512310 寂静的助力 | all | 2 | no | PASS |
| 10513110 引路船工 | all | 1 | yes | PASS |
| 10514110 脚踩天穹的《倒吊人》·罗弗拉德 | all | 1 | no | PASS |
| 10514120 虫风花·魅禄 | all, decision | 4 | yes | PASS |
| 10521110 好施的名人 | manual | 1 | no | PASS |
| 10521310 丽金花的挥霍 | manual, random | 3 | no | PASS |
| 10522120 吉祥蛙 | all | 1 | no | PASS |
| 10523110 不动如山的将校 | all | 1 | no | PASS |
| 10524110 威猛的《战车》·奥辂昂 | all, random | 2 | no | PASS |
| 10524120 丽金花·云庆 | manual | 1 | no | PASS |
| 10531120 流动控符师 | all, random | 2 | no | PASS |
| 10531310 明越花的转变 | manual | 1 | yes | PASS |
| 10532120 余韵俳谐师 | all | 1 | no | PASS |
| 10533310 壮美的明越花 | all | 1 | no | PASS |
| 10534120 明越花·阿罗 | manual | 2 | no | PASS |
| 10541110 涌泉打水人 | manual | 2 | no | PASS |
| 10541310 波摇花的裁决 | random | 1 | no | PASS |
| 10542110 铁锤龙骑士 | manual | 1 | no | PASS |
| 10542310 日珥咆哮 | all | 1 | yes | PASS |
| 10543110 破灭屠戮者 | all | 1 | no | PASS |
| 10543310 懒惰的波摇花 | random | 2 | yes | PASS |
| 10544110 约束的《正义》·伊兰翠 | random | 1 | yes | PASS |
| 10544120 波摇花·夕夜 | manual | 1 | no | PASS |
| 10551310 奥夜花的开战 | manual | 1 | no | PASS |
| 10552120 牵线搭桥的青鬼 | manual | 2 | no | PASS |
| 10552310 残虐的炸裂 | all | 1 | no | PASS |
| 10553110 致命掠夺者 | all | 2 | no | PASS |
| 10554110 充实的《恋人与节制》·米路缇欧&卢泽 | random | 1 | no | PASS |
| 10554120 奥夜花·释藤 | all | 2 | no | PASS |
| 10561110 先见的神官 | manual | 2 | no | PASS |
| 10561310 雾卷花的激愤 | random | 1 | no | PASS |
| 10562120 穷途末路的巫女 | all | 1 | no | PASS |
| 10563210 坚固的雾卷花 | manual, random | 3 | no | PASS |
| 10564110 思念的《力量》·索菲娜 | all, decision, implicit_or_bound, random | 4 | yes | PASS |
| 10564120 雾卷花·茎白 | random | 2 | no | PASS |
| 10571110 舞台缔造者 | implicit_or_bound | 2 | no | PASS |
| 10571120 繁花技师 | all | 1 | yes | PASS |
| 10572110 新时代地理学者 | manual | 1 | no | PASS |
| 10572310 苏生调律 | manual | 1 | no | PASS |
| 10573310 诚心的尽小花 | manual | 1 | yes | PASS |
| 10574110 转动的《命运之轮》·斯洛士 | all | 2 | no | PASS |
| 10574120 尽小花·伊鞠 | manual | 1 | yes | PASS |
| 10601120 匍匐的异类 | manual | 1 | no | PASS |
| 10602210 被侵略的世界 | manual | 1 | no | PASS |
| 10603210 黑暗次元 | all | 1 | yes | PASS |
| 10604110 恐惧的象征·欧米伽奥提普 | random | 1 | no | PASS |
| 10611310 天枪授予 | all | 1 | no | PASS |
| 10613310 慈爱的天枪 | decision, random | 2 | no | PASS |
| 10614110 慈爱的凛华·奥尔提雅 | manual | 1 | yes | PASS |
| 10621120 懒惰女仆 | manual | 1 | no | PASS |
| 10621310 天剑授予 | implicit_or_bound | 3 | no | PASS |
| 10622110 忠烈的近卫兵 | manual | 1 | no | PASS |
| 10622120 猫人水手 | all | 1 | no | PASS |
| 10622310 威风的行军 | all | 1 | no | PASS |
| 10623110 暴烈的参谋 | manual | 1 | no | PASS |
| 10623310 惨烈的天剑 | decision, random | 3 | no | PASS |
| 10624110 惨烈的剑王·罗德诺艾尔四世 | all, implicit_or_bound | 5 | no | PASS |
| 10624120 古旧天剑·伊德梅塔 | all | 1 | yes | PASS |
| 10631120 空想的图书管理员 | all | 1 | no | PASS |
| 10632120 冒险魔导书 | all | 1 | no | PASS |
| 10632310 正常的侵蚀 | manual | 1 | yes | PASS |
| 10633110 魔醉的教师 | all | 1 | no | PASS |
| 10633310 魔恋的天晶 | decision, implicit_or_bound | 7 | yes | PASS |
| 10634120 古旧天晶·卡卢基典瑟拉 | implicit_or_bound | 2 | yes | PASS |
| 10641110 决断的龙人 | manual | 1 | no | PASS |
| 10641310 天刀授予 | manual | 1 | no | PASS |
| 10642110 果断的剑圣 | manual | 1 | no | PASS |
| 10642120 尖刺龙 | all | 1 | no | PASS |
| 10642310 赤流 | manual | 2 | yes | PASS |
| 10643110 隔断的龙斗士 | manual, random | 4 | no | PASS |
| 10643310 断头的天刀 | all | 1 | no | PASS |
| 10644110 断头的斩姬·相枛津 | manual | 1 | yes | PASS |
| 10652120 失恋恶魔 | manual | 1 | no | PASS |
| 10652310 “最强”的诱惑 | implicit_or_bound, manual | 2 | no | PASS |
| 10653110 渴命的破坏者 | implicit_or_bound, manual | 2 | no | PASS |
| 10653310 枯渴的天眼 | all, decision | 2 | yes | PASS |
| 10661210 污浊的圣水 | random | 1 | no | PASS |
| 10662210 救赎的圣典 | all | 1 | no | PASS |
| 10663110 崇拜的圣骑士 | manual | 1 | yes | PASS |
| 10663210 崇高的天书 | decision, implicit_or_bound | 2 | no | PASS |
| 10664110 崇高的憎恶·康蒂玛 | all, decision, implicit_or_bound | 3 | no | PASS |
| 10664120 古旧天书·莲妥丝 | manual | 1 | no | PASS |
| 10671120 聪明的创造者 | implicit_or_bound | 2 | no | PASS |
| 10671310 天斧授予 | manual | 1 | no | PASS |
| 10672110 拙劣的人偶 | implicit_or_bound | 1 | yes | PASS |
| 10672120 胆小鬼先锋 | manual | 1 | no | PASS |
| 10672310 平庸的制图 | implicit_or_bound | 3 | no | PASS |
| 10673110 愚劣的兵器 | all | 2 | yes | PASS |
| 10673310 恶劣的天斧 | random | 2 | no | PASS |
| 10701310 颓废之泪 | manual | 1 | no | PASS |
| 10703110 享乐的上级市民 | all, manual | 2 | no | PASS |
| 10703210 巴别隆城 | decision, implicit_or_bound, random | 3 | yes | PASS |
| 10711120 精灵陷阱师 | manual | 1 | no | PASS |
| 10711310 人格切换 | manual | 2 | no | PASS |
| 10712120 弓兵指挥者 | manual | 2 | no | PASS |
| 10713110 冰箭射手 | manual | 1 | no | PASS |
| 10713310 恶意扩大 | random | 1 | no | PASS |
| 10714110 操量的安纳提玛·达斯特迪兹 | decision, implicit_or_bound | 2 | no | PASS |
| 10714120 冰界鹿王 | all | 1 | no | PASS |
| 10721110 曲行工兵 | manual | 1 | no | PASS |
| 10721310 敌我的调律 | implicit_or_bound, manual | 2 | no | PASS |
| 10722110 听略谍报兵 | manual | 1 | yes | PASS |
| 10722120 斩奏医护兵 | manual | 1 | yes | PASS |
| 10722310 无音的包围 | implicit_or_bound | 2 | yes | PASS |
| 10723110 响爪分队长 | all, decision | 5 | yes | PASS |
| 10723310 带来静寂的拔刀 | all, manual | 2 | no | PASS |
| 10724120 宽严的音帅·塞扎尔 | all, manual | 3 | yes | PASS |
| 10731120 小型怪兽 | manual | 2 | no | PASS |
| 10732310 暴食的零嘴 | all, decision | 2 | no | PASS |
| 10733110 甜美存在 | all, decision | 4 | no | PASS |
| 10733310 饕餮魔咒 | manual | 1 | no | PASS |
| 10734110 万食的安纳提玛·拉拉安瑟姆 | manual | 1 | no | PASS |
| 10734120 可爱杰作 | all | 1 | no | PASS |
| 10741120 载运飞龙 | manual | 2 | no | PASS |
| 10741310 百无聊赖的睥睨 | all | 1 | no | PASS |
| 10742110 豪龙守门人 | manual | 1 | no | PASS |
| 10743110 龙人先驱者 | decision, random | 3 | no | PASS |
| 10743310 黑炎的奔流 | all | 1 | no | PASS |
| 10744110 焦灰的安纳提玛·班德奈特 | all | 1 | yes | PASS |
| 10744120 龙峪的古龙 | all | 1 | no | PASS |
| 10751120 恶魔鼓手·拉兹 | manual | 1 | yes | PASS |
| 10751310 灵魂调律 | manual | 1 | no | PASS |
| 10752110 猫咪走绳师 | manual | 2 | no | PASS |
| 10752120 乌鸦杂耍师 | manual | 1 | no | PASS |
| 10752310 讴歌青春 | implicit_or_bound | 3 | no | PASS |
| 10753110 骸骨驯兽师 | manual, random | 2 | no | PASS |
| 10753310 夜之歌的演唱会 | all | 1 | no | PASS |
| 10754110 傍死的安纳提玛·徒姬 | all | 1 | no | PASS |
| 10761120 广域传教士 | all | 1 | no | PASS |
| 10761210 阳光耳饰 | manual | 2 | no | PASS |
| 10762110 神圣策划人 | manual | 1 | no | PASS |
| 10762120 传言圣鸟 | manual | 1 | no | PASS |
| 10762210 完美的时钟 | all | 2 | no | PASS |
| 10763210 海蚀三叉戟 | manual | 2 | no | PASS |
| 10764110 裁神的安纳提玛·罗德欧 | random | 1 | no | PASS |
| 10764120 崇拜经理人·伊尼西雅 | manual | 2 | no | PASS |
| 10771310 跑酷 | decision | 1 | yes | PASS |
| 10772120 大胆的涂鸦师 | manual | 1 | no | PASS |
| 10772310 闪光一瞬 | all | 2 | no | PASS |
| 10773310 瞬移斩击 | all | 1 | no | PASS |
| 10774110 虚刻的安纳提玛·斯卡雷特 | all | 1 | yes | PASS |
| 10774120 奋厉追赶·米乌 | random | 1 | yes | PASS |
| 10802110 激动的欢喜·阿尔菲德 | manual | 1 | no | PASS |
| 10802310 救世的英姿 | random | 1 | no | PASS |
| 10803310 传承的意志 | all, decision, random | 3 | yes | PASS |
| 10804110 阿尔比昂巴哈姆特 | all, decision | 4 | yes | PASS |
| 10811110 昔日的天秤·马龙 | random | 1 | no | PASS |
| 10811130 忧郁少女·莫埃尔 | manual | 1 | no | PASS |
| 10812310 宁静的孤独 | manual | 1 | no | PASS |
| 10813110 温柔读心者·米榭儿 | all | 1 | yes | PASS |
| 10813310 水镜的信赖 | all | 1 | yes | PASS |
| 10814110 离合有终·赛德斯&梅希亚 | all, manual | 2 | yes | PASS |
| 10814120 永恒冰晶·蒂亚 | all | 1 | yes | PASS |
| 10821110 武力与治安·娜哈特·娜哈特&宾森特 | all | 2 | yes | PASS |
| 10821120 寡言的刺客·夏伊莉 | manual | 1 | no | PASS |
| 10821130 悠久的骑士·莎夏 | implicit_or_bound | 2 | no | PASS |
| 10822110 越狱者·卡婕 | random | 1 | no | PASS |
| 10822120 织田信长 | all | 1 | no | PASS |
| 10822310 重历新生 | random | 1 | no | PASS |
| 10823310 相伴相随的日常 | decision, random | 3 | no | PASS |
| 10831110 森绿的恩惠·喵鲁&圆滚滚2号&吉娜 | all, manual | 4 | no | PASS |
| 10831310 暴风破 | manual | 1 | no | PASS |
| 10832310 其乐融融的团聚 | all | 1 | no | PASS |
| 10832320 伏地雷击 | random | 1 | no | PASS |
| 10833110 玛纳利亚文书官·琪可 | all | 1 | no | PASS |
| 10834120 灾难言灵·洋荷 | all | 1 | no | PASS |
| 10842120 满面笑容的烹饪·琪米卡 | manual | 2 | yes | PASS |
| 10842310 末世死化妆 | random | 1 | no | PASS |
| 10843310 狐火蜃景 | manual | 1 | no | PASS |
| 10844120 金银绚烂·璐米欧儿&雅尔贞特 | all, manual | 2 | yes | PASS |
| 10852120 黑暗骑士·玛莎 | all | 4 | no | PASS |
| 10852310 启程的退场 | decision, random | 2 | no | PASS |
| 10853110 诚实的诅咒师·丝姬 | manual | 2 | no | PASS |
| 10853310 改变的流向 | implicit_or_bound, manual | 2 | no | PASS |
| 10854110 出发的憧憬·苇剑&武津御 | all, decision | 3 | yes | PASS |
| 10854120 日月的蔷薇·赛蕾丝 | all | 1 | yes | PASS |
| 10861110 图书室的魔女·莉莉尤姆 | implicit_or_bound, manual | 2 | yes | PASS |
| 10861120 勤劳的女祭司·泰瑞莎 | implicit_or_bound | 2 | no | PASS |
| 10861130 亡灵猎人·格兰特 | manual | 2 | no | PASS |
| 10862110 天阳的使徒·艾迪特 | implicit_or_bound, manual | 2 | no | PASS |
| 10862310 威胁的残渣 | all, manual | 2 | no | PASS |
| 10863110 圣洁驱魔人·珂蕾特 | random | 2 | yes | PASS |
| 10864110 飞跃的姐妹·贝尔迪俪亚&卡诗黛儿 | implicit_or_bound | 1 | yes | PASS |
| 10864120 希望的光彩·莉迪耶尔 | all, decision | 2 | yes | PASS |
| 10871110 满溢的幸福·库伦特司 | implicit_or_bound | 1 | no | PASS |
| 10871120 过度守护者·莉欧娜 | manual | 1 | no | PASS |
| 10873110 知恩图报·米莉亚姆 | manual | 1 | no | PASS |
| 10873310 开辟未来 | manual | 1 | no | PASS |
| 10874110 决断的交错·亚修雷&莉缇雅 | manual, random | 2 | yes | PASS |
| 10874120 你的前辈·欧丝 | manual | 1 | yes | PASS |
| 90004130 边狱的邪祟 | manual | 1 | no | PASS |
| 90004320 绝大的证明 | manual | 1 | no | PASS |
| 90004330 涸绝的甘露 | random | 1 | yes | PASS |
| 90014310 蔷薇之闪击 | manual | 1 | no | PASS |
| 90014330 天枪深渊 | manual | 1 | no | PASS |
| 90021310 黄金短剑 | manual | 1 | no | PASS |
| 90021330 黄金之靴 | implicit_or_bound, manual | 2 | no | PASS |
| 90021340 黄金项链 | implicit_or_bound, manual | 2 | no | PASS |
| 90021350 闪耀的金币 | decision, random | 2 | no | PASS |
| 90024310 空绝的残光 | all | 1 | no | PASS |
| 90024320 天剑深渊 | manual | 2 | yes | PASS |
| 90024330 亡命者的枪击 | random | 2 | yes | PASS |
| 90031130 式神·小纸人 | all | 1 | no | PASS |
| 90031140 式神·暴鬼 | all | 1 | no | PASS |
| 90031310 玛纳利亚魔弹 | random | 1 | no | PASS |
| 90032110 洋葱军团兵 | all | 1 | no | PASS |
| 90033310 有所成长了！ | random | 1 | no | PASS |
| 90034110 式神·天后 | all | 1 | no | PASS |
| 90034310 绝尽的伪证 | all | 2 | no | PASS |
| 90034320 伟大之术 | manual | 1 | no | PASS |
| 90034330 天晶深渊 | implicit_or_bound | 1 | yes | PASS |
| 90034340 苍奏之四 | all | 1 | no | PASS |
| 90034350 宏大的回归 | manual | 1 | no | PASS |
| 90044310 银冰吐息 | all, decision | 3 | no | PASS |
| 90044320 烈绝的灭牙 | all | 1 | yes | PASS |
| 90051140 腐臭的僵尸 | implicit_or_bound | 1 | no | PASS |
| 90054130 一尾狐 | random | 1 | no | PASS |
| 90054310 绝叫的扩散 | implicit_or_bound | 2 | yes | PASS |
| 90054320 爱绝的飞翔 | manual | 1 | yes | PASS |
| 90064210 月影指环 | all | 1 | no | PASS |
| 90064310 绝望的奔流 | all, random | 2 | no | PASS |
| 90064320 天书深渊 | decision, implicit_or_bound | 2 | no | PASS |
| 90073130 毁灭创造物γ | all | 1 | yes | PASS |
| 90074110 卓越创造物Ω | all | 1 | yes | PASS |
| 90074310 奏绝的独唱 | manual, random | 2 | yes | PASS |
| 90074320 天斧深渊 | manual | 1 | no | PASS |
