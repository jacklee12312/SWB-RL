# Zone, Capacity, and Resource Audit

This report inventories structured zone/resource sources and preserves the direct executable contracts used by checklist section 1.8.

## Summary

- Cards: 826 (735 collectible, 91 generated)
- Training closure: 147
- Production source cards: 611
- Synthetic demo sources: 21
- Behavioral contracts: 9
- Failures: 0
- Result: PASS

## Official evidence

| Evidence | Authority | Conclusion |
|---|---|---|
| [SWB-ZONE-OFFICIAL-001](https://shadowverse-wb.com/ja/help/?tab=tab0) | official_help | 官方帮助页统一固定手牌九张、场面五张、纹章与信仰共享五格、空牌库抽牌败北、过量抽牌/加入手牌增加墓场、破坏/消失/舍弃/变身、倒数/启动，以及连击、协作、觉醒、唤灵、土之印、魔力增幅、融合、奥义和解放奥义的基础规则。 |
| [SWB-ZONE-OFFICIAL-002](https://shadowverse-wb.com/ja/system/cardbattle/battle/) | official_system_guide | 官网对战说明明确手牌最多九张，第十张抽到的卡会进入墓场。 |
| [SWB-ZONE-OFFICIAL-003](https://shadowverse-wb.com/ja/deck/cardslist/card/?card_id=10362210) | official_card_qa | 祥和圣堂官方 Q&A 明确信仰不计入“纹章数量”；报告据此区分共享区域容量与卡牌效果中的纹章计数。 |

## Category matrix

| Category | Sources | Collectible | Generated | Training | Demo | Result |
|---|---:|---:|---:|---:|---:|:---:|
| draw | 137 | 133 | 4 | 26 | 12 | PASS |
| add_to_hand | 114 | 112 | 2 | 21 | 1 | PASS |
| discard | 25 | 25 | 0 | 7 | 0 | PASS |
| return_to_hand | 6 | 6 | 0 | 1 | 1 | PASS |
| banish | 20 | 18 | 2 | 1 | 1 | PASS |
| transform | 12 | 12 | 0 | 1 | 0 | PASS |
| return_to_deck | 19 | 19 | 0 | 0 | 0 | PASS |
| summon | 210 | 204 | 6 | 47 | 1 | PASS |
| destroy | 98 | 90 | 8 | 20 | 0 | PASS |
| countdown_or_activate | 55 | 50 | 5 | 10 | 0 | PASS |
| leader_area | 65 | 65 | 0 | 13 | 8 | PASS |
| empty_deck | 2 | 2 | 0 | 0 | 0 | PASS |
| combo | 27 | 27 | 0 | 0 | 0 | PASS |
| cooperation | 6 | 6 | 0 | 3 | 0 | PASS |
| shadows_necromancy | 21 | 21 | 0 | 3 | 0 | PASS |
| overflow | 16 | 16 | 0 | 1 | 0 | PASS |
| earth_sigils | 35 | 34 | 1 | 0 | 0 | PASS |
| spellboost | 32 | 28 | 4 | 2 | 0 | PASS |
| fusion | 8 | 3 | 5 | 2 | 0 | PASS |
| union_burst | 11 | 11 | 0 | 3 | 0 | PASS |
| super_skybound_art | 11 | 11 | 0 | 3 | 0 | PASS |

## Behavioral contracts

| Contract | Evidence tests | Result |
|---|---:|:---:|
| hand_capacity_0_8_9_overdraw | 1 | PASS |
| board_capacity_0_4_5_death_slot | 1 | PASS |
| zone_ownership_and_uniqueness | 3 | PASS |
| empty_deck_default_and_victory_card | 2 | PASS |
| amulet_exit_and_activation_modes | 1 | PASS |
| shared_leader_area_capacity | 1 | PASS |
| resource_increment_and_consumption_timing | 18 | PASS |
| overdraw_is_not_successful_draw | 3 | PASS |
| public_zone_histograms_match_state | 1 | PASS |

## Source inventory

| Card | Categories | Records | Training | Result |
|---|---|---:|:---:|:---:|
| 999002 synthetic-demo-999002 | draw | 1 | no | PASS |
| 999101 synthetic-demo-999101 | add_to_hand, return_to_hand | 2 | no | PASS |
| 999102 synthetic-demo-999102 | summon | 1 | no | PASS |
| 999103 synthetic-demo-999103 | banish | 1 | no | PASS |
| 999803 synthetic-demo-999803 | draw | 1 | no | PASS |
| 999805 synthetic-demo-999805 | draw | 1 | no | PASS |
| 999901 synthetic-demo-999901 | leader_area | 2 | no | PASS |
| 999902 synthetic-demo-999902 | leader_area | 2 | no | PASS |
| 999903 synthetic-demo-999903 | draw, leader_area | 3 | no | PASS |
| 999910 synthetic-demo-999910 | leader_area | 2 | no | PASS |
| 999911 synthetic-demo-999911 | leader_area | 2 | no | PASS |
| 999912 synthetic-demo-999912 | draw, leader_area | 3 | no | PASS |
| 999913 synthetic-demo-999913 | leader_area | 2 | no | PASS |
| 999914 synthetic-demo-999914 | leader_area | 2 | no | PASS |
| 999950 synthetic-demo-999950 | draw | 1 | no | PASS |
| 999951 synthetic-demo-999951 | draw | 1 | no | PASS |
| 999952 synthetic-demo-999952 | draw | 1 | no | PASS |
| 999953 synthetic-demo-999953 | draw | 1 | no | PASS |
| 999954 synthetic-demo-999954 | draw | 1 | no | PASS |
| 999955 synthetic-demo-999955 | draw | 1 | no | PASS |
| 999956 synthetic-demo-999956 | draw | 1 | no | PASS |
| 10001120 叮当天使·莉亚 | draw | 2 | no | PASS |
| 10001210 侦探的放大镜 | destroy, countdown_or_activate | 3 | no | PASS |
| 10002210 冒险者公会 | draw, destroy, countdown_or_activate | 4 | no | PASS |
| 10011110 妖精驯服者 | add_to_hand | 1 | no | PASS |
| 10011120 流浪兽人 | combo | 1 | no | PASS |
| 10011130 温厚的树精 | combo | 1 | no | PASS |
| 10011210 缭乱之庭 | add_to_hand, countdown_or_activate | 2 | no | PASS |
| 10012110 冒险精灵·小梅 | combo | 1 | no | PASS |
| 10012120 音速射手·塞尔文 | return_to_hand | 1 | no | PASS |
| 10012310 昆虫的忠告 | return_to_hand | 1 | no | PASS |
| 10021120 战斗商贩 | draw | 1 | no | PASS |
| 10021310 女仆的礼仪 | draw, return_to_deck | 2 | no | PASS |
| 10022110 王室御用车夫 | summon | 1 | no | PASS |
| 10022120 魔煌的诡谲者·拉斯提 | draw | 1 | no | PASS |
| 10022210 昭示正统的王冠 | countdown_or_activate | 1 | no | PASS |
| 10031110 闪光魔法剑士 | earth_sigils, spellboost | 2 | no | PASS |
| 10031210 魔女的炼金炉 | draw, countdown_or_activate, earth_sigils | 4 | no | PASS |
| 10031310 智慧光辉 | draw | 1 | yes | PASS |
| 10031320 召唤真理 | summon | 1 | no | PASS |
| 10032110 双面魔女·蕾米拉米 | summon, earth_sigils | 2 | no | PASS |
| 10032120 魔焰毁灭者 | spellboost | 1 | no | PASS |
| 10032310 魔爆 | draw, earth_sigils | 2 | no | PASS |
| 10041310 龙人碎击 | overflow | 1 | no | PASS |
| 10042120 咆哮的驭龙使 | summon | 1 | no | PASS |
| 10042310 龙之启示 | draw | 1 | yes | PASS |
| 10051130 恶毒的小木乃伊 | shadows_necromancy | 1 | no | PASS |
| 10051310 混沌诅咒 | draw, summon, shadows_necromancy | 3 | yes | PASS |
| 10052110 魅惑的魅魔·莉莉姆 | add_to_hand | 1 | no | PASS |
| 10052120 多情的唤灵师 | summon | 2 | no | PASS |
| 10052310 捕食灵魂 | draw, destroy | 2 | no | PASS |
| 10061210 投影鸟像 | summon, countdown_or_activate | 4 | no | PASS |
| 10062110 铁拳神父 | banish | 2 | no | PASS |
| 10062210 羽翼石像 | summon, countdown_or_activate | 4 | no | PASS |
| 10071110 炮击猫兽人 | add_to_hand | 1 | no | PASS |
| 10071120 人偶长矛手 | add_to_hand | 1 | no | PASS |
| 10071310 来自异次元的枪击 | add_to_hand, destroy | 2 | no | PASS |
| 10072110 电鞭手 | add_to_hand | 1 | no | PASS |
| 10072120 魔钢骑兵 | summon | 2 | no | PASS |
| 10072210 人偶剧场 | add_to_hand, countdown_or_activate | 3 | yes | PASS |
| 10101110 贪婪的智天使·露比 | draw, return_to_deck | 2 | no | PASS |
| 10101120 观察的侦探 | add_to_hand | 1 | no | PASS |
| 10101310 哥布林的偷袭 | summon | 1 | no | PASS |
| 10102310 炽天使的福音 | draw | 1 | no | PASS |
| 10103110 爽朗的天宫·菲尔德亚 | destroy | 1 | no | PASS |
| 10103310 神之雷霆 | destroy | 1 | no | PASS |
| 10104110 勇武的堕天使·奥莉薇 | draw | 1 | no | PASS |
| 10104120 终极之罪·深渊之主 | return_to_deck, empty_deck | 2 | no | PASS |
| 10111110 舞动的妖精 | combo | 1 | no | PASS |
| 10111120 恋触妖精 | add_to_hand, summon | 4 | no | PASS |
| 10111130 深奥的妖精守护圣兽 | draw | 1 | no | PASS |
| 10111140 勤劳的蚂蚱 | draw, combo | 2 | no | PASS |
| 10111150 言传的杂草人长老 | combo | 1 | no | PASS |
| 10111310 妖精召集令 | add_to_hand | 1 | no | PASS |
| 10112110 霜寒冰晶·艾琳 | destroy | 1 | no | PASS |
| 10112120 纯真的水之妖精 | add_to_hand | 1 | no | PASS |
| 10112130 年幼宝石兽 | return_to_hand | 1 | no | PASS |
| 10112210 磷光辉岩 | add_to_hand, destroy, countdown_or_activate, combo | 5 | no | PASS |
| 10112310 薰交的思慕 | draw, add_to_hand | 2 | no | PASS |
| 10113110 纯洁冰晶·莉莉 | draw, combo | 2 | no | PASS |
| 10113120 薰交的天宫·巴克伍德 | draw | 1 | no | PASS |
| 10113140 屠戮破魔虫 | combo | 1 | no | PASS |
| 10113210 圣树法杖 | draw, return_to_hand, destroy, countdown_or_activate, combo | 6 | no | PASS |
| 10114110 自然妖精公主·阿丽雅 | summon, leader_area | 3 | no | PASS |
| 10114120 丰丽的玫瑰皇后 | transform | 1 | no | PASS |
| 10121130 救援的鲁米那斯治疗师·莉拉拉 | summon | 1 | no | PASS |
| 10121140 军犬 | summon | 1 | no | PASS |
| 10121310 剑士的斩击 | summon, destroy | 2 | no | PASS |
| 10122110 统率的鲁米那斯骑士 | summon | 1 | no | PASS |
| 10122120 卓越的鲁米那斯法师 | summon | 1 | no | PASS |
| 10122130 勇猛的鲁米那斯枪士 | summon | 1 | no | PASS |
| 10122140 忍者鼯鼠 | summon | 1 | no | PASS |
| 10122310 王断的威光 | summon | 1 | no | PASS |
| 10123110 雷维翁之斧·杰诺 | summon | 1 | no | PASS |
| 10123130 王断的天宫·斯塔奇乌姆 | summon | 1 | no | PASS |
| 10123140 煌刃勇者·阿玛利亚 | summon | 1 | no | PASS |
| 10124120 白银骑士团团长·艾蜜莉亚 | draw | 1 | no | PASS |
| 10124130 常在战场·景光 | summon, leader_area | 3 | no | PASS |
| 10131120 见习占星术师 | draw, return_to_deck, earth_sigils | 3 | no | PASS |
| 10131130 唤枭士 | earth_sigils | 1 | no | PASS |
| 10131140 追梦的企鹅魔法师 | draw, spellboost | 2 | no | PASS |
| 10131310 彩虹奇迹 | draw, spellboost | 2 | no | PASS |
| 10131320 暴风破 | spellboost | 1 | yes | PASS |
| 10132110 惹人怜爱的教师·米兰 | spellboost | 2 | no | PASS |
| 10132120 奇迹女巫·爱蜜儿 | summon, spellboost | 2 | no | PASS |
| 10132130 玛纳利亚的学生·威廉 | spellboost | 2 | no | PASS |
| 10132310 理光的证明 | earth_sigils | 1 | no | PASS |
| 10132320 雪人觉醒 | spellboost | 1 | no | PASS |
| 10133110 黎明炼金术师·诺诺 | summon, leader_area, earth_sigils | 5 | no | PASS |
| 10133120 魔法药剂师·佩内洛普 | draw, earth_sigils | 3 | no | PASS |
| 10133130 理光的天宫·艾德薇诗 | earth_sigils | 1 | no | PASS |
| 10133310 做作业啦！ | draw, transform, spellboost | 3 | no | PASS |
| 10133320 唤鬼术 | summon, spellboost | 2 | no | PASS |
| 10134110 五行之巅·久苑 | summon, destroy | 3 | no | PASS |
| 10134120 玛纳利亚密友·安&古蕾雅 | summon, spellboost | 2 | no | PASS |
| 10134310 超越次元 | draw, return_to_deck, spellboost | 4 | no | PASS |
| 10141110 云海龙骑兵 | summon | 1 | no | PASS |
| 10141120 海沟大剑龙 | overflow | 1 | no | PASS |
| 10141130 初出茅庐的屠龙者 | destroy | 1 | no | PASS |
| 10141140 育龙少女 | summon | 2 | no | PASS |
| 10141150 白鳞的使者 | overflow | 1 | no | PASS |
| 10142120 御风者·叶花 | overflow | 1 | no | PASS |
| 10142140 艳丽龙人·玛利翁 | overflow | 1 | no | PASS |
| 10142310 荣弦的奏乐 | draw, overflow | 2 | no | PASS |
| 10143120 荣弦的天宫·龙芙 | overflow | 1 | no | PASS |
| 10143130 惊涛龙骑士·扎哈尔 | summon | 1 | no | PASS |
| 10143140 夜幕龙 | draw | 1 | no | PASS |
| 10143210 乙姬的宝扇 | discard, summon, countdown_or_activate | 4 | no | PASS |
| 10144110 灼热的安纳提玛·班德奈特 | discard, leader_area | 3 | no | PASS |
| 10144130 龙人演义·卧龙 | summon | 1 | no | PASS |
| 10151110 午夜眷属·埃拉尔 | summon | 1 | no | PASS |
| 10151120 无名恶魔 | summon | 1 | no | PASS |
| 10151130 白骨少女 | summon | 1 | no | PASS |
| 10151140 杂耍亡灵 | summon, shadows_necromancy | 2 | no | PASS |
| 10151150 禁约恶魔 | draw | 1 | no | PASS |
| 10151310 死神挥刀 | destroy | 1 | no | PASS |
| 10152110 干练的死神·蜜诺 | add_to_hand | 1 | no | PASS |
| 10152130 怪奇探索者·尤娜 | add_to_hand | 1 | no | PASS |
| 10152210 瞑地的陵园 | summon, destroy, countdown_or_activate, shadows_necromancy | 5 | no | PASS |
| 10153120 燃烧魔剑·欧特鲁斯 | shadows_necromancy | 2 | no | PASS |
| 10153130 瞑地的天宫·穆甘 | summon, shadows_necromancy | 2 | no | PASS |
| 10153140 地下赏金猎人·巴尔特 | leader_area | 2 | no | PASS |
| 10154110 奔放的狱焰·凯尔贝洛斯 | summon, shadows_necromancy | 4 | no | PASS |
| 10154120 剧毒公主·美杜莎 | destroy | 1 | no | PASS |
| 10161110 圣心光棱牧师 | draw | 1 | no | PASS |
| 10161130 指明目标的光辉天使 | draw | 1 | no | PASS |
| 10161210 祥和的教会 | draw, countdown_or_activate | 4 | no | PASS |
| 10161310 翎雨 | summon | 1 | no | PASS |
| 10162130 大地守护神·米维 | summon | 1 | no | PASS |
| 10162210 禁密的圣地 | countdown_or_activate | 2 | no | PASS |
| 10162220 神圣注射 | destroy, countdown_or_activate | 3 | no | PASS |
| 10163110 终焉的白骨圣堂之主 | destroy | 1 | no | PASS |
| 10163120 禁密的天宫·罗纳维罗 | destroy | 1 | no | PASS |
| 10163130 伟大的炽天使·勒碧丝 | summon, leader_area | 3 | no | PASS |
| 10163210 野兽公主的誓约 | summon, countdown_or_activate | 4 | no | PASS |
| 10163220 邪教法器 | destroy, countdown_or_activate | 3 | no | PASS |
| 10164110 裁决的安纳提玛·罗德欧 | discard, summon, destroy | 3 | no | PASS |
| 10171110 耀眼的发明家·伊莉斯 | add_to_hand | 1 | no | PASS |
| 10171120 钢铁佣兵·迪尔克 | summon | 1 | no | PASS |
| 10171130 永不停火的枪手 | add_to_hand | 1 | no | PASS |
| 10171140 自动机械刺客 | add_to_hand | 1 | no | PASS |
| 10171310 人偶替身 | summon | 1 | no | PASS |
| 10171320 创造物充能 | add_to_hand | 1 | no | PASS |
| 10172110 抗争领袖·露琪娜 | add_to_hand, summon | 2 | no | PASS |
| 10172120 依恋的人偶师 | add_to_hand | 2 | no | PASS |
| 10172130 杀意之丝·诺亚 | add_to_hand | 1 | no | PASS |
| 10172310 生命的奔流 | add_to_hand | 1 | no | PASS |
| 10172320 改境的重启 | summon | 1 | no | PASS |
| 10173110 决心之誓·米莉亚姆 | add_to_hand | 2 | no | PASS |
| 10173120 箱庭的断罪者·希尔薇娅 | draw, destroy | 3 | no | PASS |
| 10173130 疯狂的创造者·历亚姆 | summon | 1 | no | PASS |
| 10173140 改境的天宫·阿洛艾特 | add_to_hand, summon | 2 | no | PASS |
| 10173210 遗产的炮击 | add_to_hand | 1 | no | PASS |
| 10174110 崭新的少女·欧丝 | draw, leader_area | 4 | no | PASS |
| 10174120 迈进之心·奥契丝 | summon | 2 | no | PASS |
| 10174130 增幅加速·洛拉米亚 | summon | 1 | no | PASS |
| 10204110 命运黄昏·奥丁 | banish | 1 | no | PASS |
| 10204120 飓风天业·格里姆尼尔 | leader_area | 2 | no | PASS |
| 10211120 木锤矮人 | combo | 1 | no | PASS |
| 10211310 森林的游行 | summon | 1 | no | PASS |
| 10212110 热情的精灵·莱昂内尔 | summon | 1 | no | PASS |
| 10212120 妖精击剑士 | add_to_hand | 1 | no | PASS |
| 10212310 来自树上的偷袭 | combo | 1 | no | PASS |
| 10213110 森林骑士道·辛西亚 | summon | 1 | no | PASS |
| 10213310 花园的指引 | draw, fusion | 3 | no | PASS |
| 10214110 翅翼女王·提泰妮娅 | add_to_hand, transform, summon, leader_area | 5 | no | PASS |
| 10221310 商谈成立 | draw | 1 | no | PASS |
| 10222120 平凡骑士·拉奇尔 | draw | 1 | no | PASS |
| 10223110 剑士公主·萝泽 | draw, destroy | 2 | no | PASS |
| 10223120 假日中的王女·普莉姆 | add_to_hand | 1 | no | PASS |
| 10224110 静寂的安纳提玛·吉尔达利娅 | summon, cooperation | 2 | no | PASS |
| 10224120 雷维翁超越者·尤里乌斯 | summon | 1 | no | PASS |
| 10231110 憧憬的魔女·梅薇 | add_to_hand, earth_sigils | 2 | no | PASS |
| 10231120 魔导图书管理员 | draw, return_to_deck | 2 | no | PASS |
| 10231310 冰锥穿击 | destroy, earth_sigils | 2 | no | PASS |
| 10232110 否定的咏唱·芭赛特 | summon, leader_area | 4 | no | PASS |
| 10232120 调香的魔法师 | earth_sigils | 2 | no | PASS |
| 10232310 混沌赤焰 | spellboost | 1 | no | PASS |
| 10233110 玛纳利亚剑士·欧文 | draw, spellboost | 2 | no | PASS |
| 10233310 帕梅拉的舞蹈 | draw, leader_area, earth_sigils | 5 | no | PASS |
| 10234110 暴食的安纳提玛·拉拉安瑟姆 | summon, destroy, earth_sigils | 3 | no | PASS |
| 10234120 精金炼金术师·诺曼 | draw, summon, earth_sigils | 6 | no | PASS |
| 10241110 庇护的智龙 | summon | 1 | no | PASS |
| 10241120 飞跃的银白幼龙 | draw, overflow | 2 | no | PASS |
| 10241310 虎鲸的呼声 | draw, summon, overflow | 3 | no | PASS |
| 10242110 宿愿的龙人公主 | draw | 1 | no | PASS |
| 10242120 身经百战的鱼人 | summon | 2 | no | PASS |
| 10242210 炎龙之剑 | summon, destroy, countdown_or_activate | 4 | no | PASS |
| 10243110 苍海的制裁·尼普顿 | summon, leader_area | 4 | no | PASS |
| 10244110 银冰龙少女·菲琳 | add_to_hand, overflow | 2 | no | PASS |
| 10251110 银色子弹·雷文 | destroy | 1 | no | PASS |
| 10251120 怨恨的栽培者 | add_to_hand | 1 | no | PASS |
| 10251310 诅咒派对 | add_to_hand | 1 | no | PASS |
| 10252120 尸兵 | summon | 1 | no | PASS |
| 10252310 使唤蝙蝠 | summon | 1 | no | PASS |
| 10254110 双轮夜行·吟雪&夕月 | summon, destroy | 2 | no | PASS |
| 10254120 流动堕落的冥河·凯伦 | summon, leader_area, shadows_necromancy | 6 | no | PASS |
| 10261110 粉碎的圣职者 | destroy | 1 | no | PASS |
| 10261210 流光香炉 | destroy, countdown_or_activate | 3 | no | PASS |
| 10262120 有洁癖的审判者 | banish | 1 | no | PASS |
| 10263110 速断之刃·阿尼耶丝 | destroy | 1 | no | PASS |
| 10263310 疯狂的恩宠 | leader_area | 2 | no | PASS |
| 10264110 天之守护神·埃忒耳 | summon | 1 | no | PASS |
| 10264120 呜咽的圣骑士·维尔伯特 | summon, leader_area | 3 | no | PASS |
| 10271110 引擎剑士 | add_to_hand, summon | 4 | no | PASS |
| 10271120 猫偶 | add_to_hand | 1 | no | PASS |
| 10271210 创造物弹射器 | add_to_hand, summon, destroy, countdown_or_activate | 5 | no | PASS |
| 10272110 心灵屠戮者·菲亚 | transform | 1 | no | PASS |
| 10272120 绝望之王·阿基姆 | banish, summon | 2 | no | PASS |
| 10272310 伊卡洛斯的飞翔 | draw | 1 | no | PASS |
| 10273110 暗狱的余晖·贾丝珀 | add_to_hand | 1 | no | PASS |
| 10273310 心有灵犀的共斗 | summon | 1 | no | PASS |
| 10274110 交响之心·枷薇 | summon | 2 | no | PASS |
| 10274120 精神武艺·迦尔拉 | summon | 1 | no | PASS |
| 10301310 至高的凌驾 | draw | 1 | no | PASS |
| 10303210 试炼的石板 | draw, banish, countdown_or_activate | 4 | no | PASS |
| 10304110 绝大的显现·麦哲佩恩 | draw, add_to_hand, discard, return_to_deck, leader_area, empty_deck | 7 | no | PASS |
| 10304120 涸绝的显现·吉尔内莉莎 | add_to_hand | 1 | yes | PASS |
| 10311310 野性的猛袭 | draw, combo | 2 | no | PASS |
| 10312120 树海的战士 | add_to_hand | 1 | no | PASS |
| 10312210 不弑之乡 | draw, discard, destroy, countdown_or_activate | 5 | no | PASS |
| 10313110 不弑的团结者 | summon | 1 | no | PASS |
| 10313310 驱逐的死矢 | combo | 1 | no | PASS |
| 10314110 不弑的继承者·库露露 | leader_area | 2 | yes | PASS |
| 10314120 绝命的显现·艾斯迪亚 | add_to_hand | 1 | no | PASS |
| 10321110 篡夺的肯定者 | add_to_hand | 2 | no | PASS |
| 10321120 剑圣的同胞 | summon | 1 | no | PASS |
| 10322110 篡夺的祈祷者 | add_to_hand | 2 | no | PASS |
| 10322120 活泼的斥候 | summon | 1 | no | PASS |
| 10322210 篡夺的据点 | add_to_hand, destroy, countdown_or_activate | 4 | no | PASS |
| 10323110 篡夺的团结者 | add_to_hand | 1 | no | PASS |
| 10323310 奉还的剑闪 | draw, add_to_hand, fusion | 4 | no | PASS |
| 10324110 篡夺的继承者·辛瑟莱兹 | fusion | 3 | no | PASS |
| 10324120 空绝的显现·奥克托丽丝 | add_to_hand, countdown_or_activate, leader_area | 5 | no | PASS |
| 10331120 五行修行者 | summon | 2 | no | PASS |
| 10331310 水晶的指引 | draw, leader_area | 3 | no | PASS |
| 10332110 真理的祈祷者 | draw | 1 | no | PASS |
| 10332210 真理的研究设施 | draw, countdown_or_activate | 5 | no | PASS |
| 10332310 双重创造 | summon | 1 | no | PASS |
| 10333110 真理的团结者 | summon | 2 | no | PASS |
| 10333310 虚假的术式 | destroy | 1 | yes | PASS |
| 10334110 真理的继承者·蓓哈丽雅 | draw, banish | 2 | no | PASS |
| 10334120 绝尽的显现·莱奥 | transform | 1 | no | PASS |
| 10341110 侮蔑的肯定者 | draw | 1 | no | PASS |
| 10341120 风雪龙人 | destroy | 1 | no | PASS |
| 10341310 雷霆之怒 | overflow | 1 | no | PASS |
| 10342120 海洋骑手 | summon, overflow | 2 | no | PASS |
| 10342210 侮蔑之国 | countdown_or_activate | 3 | no | PASS |
| 10343310 威猛炽焰 | draw, overflow | 2 | no | PASS |
| 10344120 烈绝的显现·嘉尔缪 | add_to_hand, leader_area | 3 | yes | PASS |
| 10351120 泡沫鬼姬 | destroy | 1 | no | PASS |
| 10352120 新来的守墓人 | summon, shadows_necromancy | 2 | no | PASS |
| 10352210 混融之城 | draw, destroy, countdown_or_activate | 5 | yes | PASS |
| 10353110 混融的团结者 | summon | 2 | yes | PASS |
| 10353310 叫唤与憎恶 | draw | 1 | yes | PASS |
| 10354110 混融的继承者·莎木·纳克雅 | add_to_hand, destroy, leader_area | 4 | yes | PASS |
| 10354120 绝叫与爱绝的显现·鲁鲁纳伊&巴娜蕾卡 | add_to_hand | 1 | yes | PASS |
| 10361110 安息的肯定者 | leader_area | 2 | no | PASS |
| 10361310 圣辉闪烁 | draw | 1 | no | PASS |
| 10362110 安息的祈祷者 | leader_area | 2 | no | PASS |
| 10362210 安息的神殿 | countdown_or_activate | 3 | no | PASS |
| 10362220 羽翼狮子像 | summon, countdown_or_activate | 4 | yes | PASS |
| 10363110 安息的团结者 | draw, destroy, leader_area | 4 | no | PASS |
| 10363210 闪耀的失意 | countdown_or_activate | 3 | no | PASS |
| 10364110 安息的继承者·妃花 | leader_area | 2 | no | PASS |
| 10364120 绝望的显现·玛温 | add_to_hand, leader_area | 3 | no | PASS |
| 10371110 破坏的肯定者 | destroy | 1 | no | PASS |
| 10371120 音速飞行兵 | summon | 1 | no | PASS |
| 10371310 丝线突袭 | add_to_hand, destroy | 2 | no | PASS |
| 10372110 破坏的祈祷者 | destroy | 2 | no | PASS |
| 10372120 现场工程师 | draw, discard | 2 | no | PASS |
| 10372210 破坏的荒野 | draw, destroy | 3 | yes | PASS |
| 10373110 破坏的团结者 | destroy | 1 | yes | PASS |
| 10373310 歼灭的歌声 | summon, destroy | 2 | yes | PASS |
| 10374110 破坏的继承者·阿克西娅 | destroy | 1 | yes | PASS |
| 10374120 奏绝的显现·莉洁纳 | add_to_hand, summon | 2 | yes | PASS |
| 10401110 驰骋天空的守护者·卡塔莉娜 | union_burst | 1 | no | PASS |
| 10403110 征服苍空的骑空士·古兰&姬塔 | draw, union_burst | 2 | yes | PASS |
| 10403120 掌握天空命运的少女·露莉亚 | draw | 1 | yes | PASS |
| 10404110 天司长的继承者·圣德芬 | return_to_hand, leader_area, super_skybound_art | 4 | yes | PASS |
| 10411310 彗星 | draw | 1 | no | PASS |
| 10412110 美妆少女·克洛伊 | return_to_hand, summon | 2 | no | PASS |
| 10412310 绮罗星 | add_to_hand, leader_area, combo | 4 | no | PASS |
| 10413110 幻彩弓手·丘比丹 | union_burst, super_skybound_art | 2 | yes | PASS |
| 10413310 亚尔夫海姆 | draw, super_skybound_art | 3 | no | PASS |
| 10414110 风之法则·艾云尼亚 | union_burst | 1 | no | PASS |
| 10414120 调和的舞者·尤艾尔&苏丝雅 | leader_area | 2 | yes | PASS |
| 10421120 决心之辉龙·亚瑟 | summon | 1 | no | PASS |
| 10421130 迷茫的狮子·莫德雷德 | summon | 1 | no | PASS |
| 10424110 真红与群青·塞达&贝阿朵丽丝 | summon | 1 | yes | PASS |
| 10424120 十天众统领·希耶提 | union_burst, super_skybound_art | 2 | yes | PASS |
| 10431110 不可思议的哲学家·菲拉索佩娅 | draw | 1 | no | PASS |
| 10431120 流浪的家庭教师·斯芙拉玛尔 | spellboost | 1 | no | PASS |
| 10432110 报仇的占卜师·艾塞克莱因 | earth_sigils | 1 | no | PASS |
| 10432120 荆棘旅途·米蕾耶&莉赛特 | summon, earth_sigils | 2 | no | PASS |
| 10432310 能量外溢 | draw | 1 | no | PASS |
| 10433110 缅怀之火·埃尔默特 | leader_area | 2 | no | PASS |
| 10433310 炼金炎爆 | earth_sigils, union_burst | 2 | no | PASS |
| 10434110 水之法则·瓦姆杜斯 | spellboost | 1 | no | PASS |
| 10434120 天才美少女炼金术士·卡莉奥丝特罗 | add_to_hand, leader_area, earth_sigils, union_burst, super_skybound_art | 8 | no | PASS |
| 10441310 破浪新月 | leader_area | 2 | no | PASS |
| 10442120 淳朴的钢铁之躯·无限 | destroy, super_skybound_art | 2 | no | PASS |
| 10443110 平平无奇的女孩·梅格 | union_burst | 1 | no | PASS |
| 10443310 星晶兽吸收之力 | add_to_hand, banish | 2 | no | PASS |
| 10451120 不屈利刃·巴萨拉卡 | summon | 1 | no | PASS |
| 10451310 妖异利刃 | leader_area | 2 | no | PASS |
| 10453310 堕落 | leader_area, super_skybound_art | 4 | no | PASS |
| 10454110 暗之法则·菲迪埃尔 | summon, shadows_necromancy | 2 | no | PASS |
| 10454120 狡诈的堕天司·彼列 | countdown_or_activate, leader_area, super_skybound_art | 4 | no | PASS |
| 10461210 莉莉艾的鼓舞 | draw, transform, destroy, countdown_or_activate | 5 | no | PASS |
| 10462110 沙神的巫女·莎拉 | destroy | 1 | no | PASS |
| 10462120 赞恩教僧侣·索菲娅 | summon | 1 | no | PASS |
| 10462210 骑驰天空之艇 | destroy, countdown_or_activate | 3 | no | PASS |
| 10463210 蕾·菲耶的宝石 | draw, destroy, countdown_or_activate | 4 | yes | PASS |
| 10464120 威严的星晶骑士·薇拉 | banish, super_skybound_art | 2 | no | PASS |
| 10471120 爆燃老大·翼 | union_burst, super_skybound_art | 2 | no | PASS |
| 10471130 报恩工匠·艾萨克 | add_to_hand | 1 | yes | PASS |
| 10472110 轰雷闪狼·尤斯提斯 | union_burst | 1 | no | PASS |
| 10473110 向往天空的回归者·卡西乌斯 | add_to_hand | 1 | no | PASS |
| 10473310 混沌军势 | super_skybound_art | 1 | no | PASS |
| 10474110 光之法则·龙敖 | leader_area, union_burst | 3 | no | PASS |
| 10502110 星辉女神 | add_to_hand, discard | 2 | no | PASS |
| 10502120 手持军配团扇的伟丈夫 | destroy | 1 | yes | PASS |
| 10503210 大游戏世界 | draw, countdown_or_activate | 3 | yes | PASS |
| 10503310 《世界》的呈现 | draw, destroy | 2 | yes | PASS |
| 10504110 八界花·下天央 | draw, discard | 2 | no | PASS |
| 10511110 熟虑的狸猫 | draw | 1 | yes | PASS |
| 10511120 森林羽子板工匠 | summon | 2 | no | PASS |
| 10511310 虫风花的飞翔 | add_to_hand | 1 | no | PASS |
| 10512110 新晋搭档 | summon, combo | 2 | no | PASS |
| 10512310 寂静的助力 | combo | 1 | no | PASS |
| 10513110 引路船工 | summon | 2 | yes | PASS |
| 10513310 优雅的虫风花 | draw, combo | 2 | no | PASS |
| 10514110 脚踩天穹的《倒吊人》·罗弗拉德 | add_to_hand, discard, combo | 3 | no | PASS |
| 10514120 虫风花·魅禄 | add_to_hand | 2 | yes | PASS |
| 10521110 好施的名人 | discard | 1 | no | PASS |
| 10521120 烟管美玉 | add_to_hand | 1 | no | PASS |
| 10521310 丽金花的挥霍 | discard | 1 | no | PASS |
| 10522110 迅猛的武术家 | draw | 1 | no | PASS |
| 10522120 吉祥蛙 | add_to_hand | 1 | no | PASS |
| 10522310 温柔援军 | summon | 2 | no | PASS |
| 10523110 不动如山的将校 | summon | 1 | no | PASS |
| 10523310 荣耀的丽金花 | add_to_hand | 2 | no | PASS |
| 10524120 丽金花·云庆 | add_to_hand, banish, leader_area | 5 | no | PASS |
| 10531110 创造魔法师 | summon, earth_sigils | 2 | no | PASS |
| 10531120 流动控符师 | spellboost | 1 | no | PASS |
| 10531310 明越花的转变 | draw, discard | 2 | yes | PASS |
| 10532110 失眠女巫 | leader_area | 3 | no | PASS |
| 10532120 余韵俳谐师 | draw, spellboost | 3 | no | PASS |
| 10532310 魔猫戏法 | summon, earth_sigils | 2 | no | PASS |
| 10533110 元素支配者 | summon, earth_sigils | 2 | no | PASS |
| 10533310 壮美的明越花 | transform | 1 | no | PASS |
| 10534110 漫步的《愚者》·琳库露 | return_to_deck, leader_area | 3 | no | PASS |
| 10534120 明越花·阿罗 | transform, spellboost | 2 | no | PASS |
| 10541120 水滴打拍者 | summon | 1 | no | PASS |
| 10541310 波摇花的裁决 | draw | 1 | no | PASS |
| 10542110 铁锤龙骑士 | summon | 1 | no | PASS |
| 10542120 水母舞姬 | add_to_hand | 1 | no | PASS |
| 10543110 破灭屠戮者 | banish | 1 | no | PASS |
| 10543310 懒惰的波摇花 | overflow | 1 | yes | PASS |
| 10544120 波摇花·夕夜 | add_to_hand, discard, summon, leader_area | 5 | no | PASS |
| 10551120 红符的魂魄道士 | summon, shadows_necromancy | 4 | no | PASS |
| 10551310 奥夜花的开战 | return_to_deck | 1 | no | PASS |
| 10552110 制造麻烦的唤灵师 | summon | 2 | no | PASS |
| 10552120 牵线搭桥的青鬼 | draw | 2 | no | PASS |
| 10553110 致命掠夺者 | transform | 1 | no | PASS |
| 10553310 严酷的奥夜花 | draw, summon, leader_area | 4 | no | PASS |
| 10554110 充实的《恋人与节制》·米路缇欧&卢泽 | summon, destroy, leader_area, shadows_necromancy | 5 | no | PASS |
| 10554120 奥夜花·释藤 | draw, discard | 4 | no | PASS |
| 10561120 连结的使徒 | draw | 1 | no | PASS |
| 10561310 雾卷花的激愤 | draw, return_to_deck | 2 | no | PASS |
| 10562110 毫不动摇的圣骑士 | summon | 1 | no | PASS |
| 10562120 穷途末路的巫女 | draw | 2 | no | PASS |
| 10562210 穹顶护甲 | draw, destroy, countdown_or_activate | 4 | no | PASS |
| 10563110 至圣威仪 | summon | 1 | no | PASS |
| 10563210 坚固的雾卷花 | draw, return_to_deck, destroy, countdown_or_activate | 5 | no | PASS |
| 10564120 雾卷花·茎白 | draw, return_to_deck, summon, leader_area | 5 | no | PASS |
| 10571110 舞台缔造者 | summon | 2 | no | PASS |
| 10571120 繁花技师 | draw | 1 | yes | PASS |
| 10571310 尽小花的临照 | draw | 1 | no | PASS |
| 10572110 新时代地理学者 | add_to_hand, summon | 2 | no | PASS |
| 10572310 苏生调律 | add_to_hand, discard | 2 | no | PASS |
| 10573110 神经遮蔽者 | draw | 1 | no | PASS |
| 10573310 诚心的尽小花 | transform | 1 | yes | PASS |
| 10574110 转动的《命运之轮》·斯洛士 | banish, leader_area | 3 | no | PASS |
| 10574120 尽小花·伊鞠 | draw, discard, summon | 4 | yes | PASS |
| 10601120 匍匐的异类 | destroy | 1 | no | PASS |
| 10602210 被侵略的世界 | transform, countdown_or_activate | 3 | no | PASS |
| 10603210 黑暗次元 | countdown_or_activate | 1 | yes | PASS |
| 10604110 恐惧的象征·欧米伽奥提普 | destroy | 1 | no | PASS |
| 10611110 慈育的森民 | add_to_hand, summon | 2 | no | PASS |
| 10611120 伊甸之猴 | combo | 1 | no | PASS |
| 10611310 天枪授予 | summon | 1 | no | PASS |
| 10612110 慈颜的拥趸 | add_to_hand | 1 | no | PASS |
| 10612310 向女王献花 | draw | 1 | no | PASS |
| 10613110 慈惠的心腹 | summon | 1 | no | PASS |
| 10613310 慈爱的天枪 | summon, destroy | 2 | no | PASS |
| 10614110 慈爱的凛华·奥尔提雅 | summon, destroy | 2 | yes | PASS |
| 10614120 古旧天枪·萨莎妮德 | add_to_hand, leader_area | 3 | no | PASS |
| 10621120 懒惰女仆 | draw, return_to_deck | 2 | no | PASS |
| 10621310 天剑授予 | summon | 1 | no | PASS |
| 10622120 猫人水手 | destroy | 1 | no | PASS |
| 10622310 威风的行军 | summon, countdown_or_activate, leader_area | 4 | no | PASS |
| 10623110 暴烈的参谋 | add_to_hand, destroy | 2 | no | PASS |
| 10623310 惨烈的天剑 | draw | 2 | no | PASS |
| 10624110 惨烈的剑王·罗德诺艾尔四世 | summon | 2 | no | PASS |
| 10624120 古旧天剑·伊德梅塔 | add_to_hand, leader_area | 3 | yes | PASS |
| 10631120 空想的图书管理员 | summon | 1 | no | PASS |
| 10631310 天晶授予 | summon | 1 | yes | PASS |
| 10632110 魔境的学生 | summon | 1 | yes | PASS |
| 10632120 冒险魔导书 | summon, spellboost | 2 | no | PASS |
| 10632310 正常的侵蚀 | draw, destroy | 2 | yes | PASS |
| 10633110 魔醉的教师 | summon | 1 | no | PASS |
| 10633310 魔恋的天晶 | summon | 2 | yes | PASS |
| 10634110 魔恋的爱慕·希姆 | summon, leader_area | 3 | yes | PASS |
| 10634120 古旧天晶·卡卢基典瑟拉 | add_to_hand, summon, leader_area | 3 | yes | PASS |
| 10641110 决断的龙人 | draw, discard | 2 | no | PASS |
| 10641120 熟透的海鱼 | summon | 1 | no | PASS |
| 10641310 天刀授予 | add_to_hand | 1 | no | PASS |
| 10642110 果断的剑圣 | discard | 1 | no | PASS |
| 10642310 赤流 | discard, destroy | 2 | yes | PASS |
| 10643110 隔断的龙斗士 | discard | 2 | no | PASS |
| 10643310 断头的天刀 | add_to_hand | 1 | no | PASS |
| 10644110 断头的斩姬·相枛津 | add_to_hand, discard | 2 | yes | PASS |
| 10644120 古旧天刀·波菈莱 | add_to_hand, summon | 3 | yes | PASS |
| 10651110 渴望的恶魔 | draw | 1 | no | PASS |
| 10651120 逃避幽灵者 | add_to_hand | 1 | no | PASS |
| 10651310 天眼授予 | draw, shadows_necromancy | 2 | no | PASS |
| 10652110 渴欲的唤灵师 | summon, shadows_necromancy | 2 | no | PASS |
| 10652310 “最强”的诱惑 | banish, summon | 2 | no | PASS |
| 10653110 渴命的破坏者 | summon, destroy | 2 | no | PASS |
| 10654110 枯渴的魔神·阿尔弭斯 | destroy | 1 | no | PASS |
| 10654120 古旧天眼·比芭提 | add_to_hand, shadows_necromancy | 2 | no | PASS |
| 10661110 崇奉的懦者 | summon | 1 | yes | PASS |
| 10661210 污浊的圣水 | draw, destroy, countdown_or_activate | 5 | no | PASS |
| 10661310 天书授予 | draw | 1 | no | PASS |
| 10662110 崇敬的涂描者 | summon | 1 | no | PASS |
| 10662120 飞马骑手 | summon | 2 | no | PASS |
| 10662210 救赎的圣典 | draw, countdown_or_activate | 2 | no | PASS |
| 10663110 崇拜的圣骑士 | summon, destroy | 3 | yes | PASS |
| 10663210 崇高的天书 | summon, destroy, countdown_or_activate | 3 | no | PASS |
| 10664110 崇高的憎恶·康蒂玛 | summon, destroy | 2 | no | PASS |
| 10664120 古旧天书·莲妥丝 | add_to_hand, destroy, leader_area | 4 | no | PASS |
| 10671110 低劣的玩具 | draw, summon | 2 | yes | PASS |
| 10671120 聪明的创造者 | summon | 1 | no | PASS |
| 10671310 天斧授予 | draw | 1 | no | PASS |
| 10672110 拙劣的人偶 | summon | 2 | yes | PASS |
| 10672120 胆小鬼先锋 | banish | 1 | no | PASS |
| 10672310 平庸的制图 | summon | 1 | no | PASS |
| 10673110 愚劣的兵器 | summon | 2 | yes | PASS |
| 10674110 恶劣的纯心·卡密希拉 | summon | 1 | yes | PASS |
| 10674120 古旧天斧·尤泽塔 | add_to_hand | 1 | no | PASS |
| 10701110 纯真孩童 | draw | 1 | no | PASS |
| 10701310 颓废之泪 | banish | 1 | no | PASS |
| 10702110 神话记者 | draw, summon | 2 | yes | PASS |
| 10703110 享乐的上级市民 | discard | 1 | no | PASS |
| 10703210 巴别隆城 | discard, destroy, countdown_or_activate | 5 | yes | PASS |
| 10704110 特殊目标·海雷姆哈妮 | summon, leader_area | 3 | no | PASS |
| 10704120 巴别隆市长·埃尔塔罗 | draw | 1 | no | PASS |
| 10711110 巨型熊 | summon | 1 | no | PASS |
| 10711120 精灵陷阱师 | add_to_hand, return_to_deck | 2 | no | PASS |
| 10711310 人格切换 | draw, return_to_deck | 2 | no | PASS |
| 10712110 绿风细剑师 | combo | 1 | no | PASS |
| 10712120 弓兵指挥者 | combo | 2 | no | PASS |
| 10712310 忧虑缩小 | add_to_hand, leader_area, combo | 4 | no | PASS |
| 10713110 冰箭射手 | draw, combo | 2 | no | PASS |
| 10713310 恶意扩大 | add_to_hand, leader_area, combo | 4 | no | PASS |
| 10714110 操量的安纳提玛·达斯特迪兹 | leader_area, combo | 4 | no | PASS |
| 10714120 冰界鹿王 | add_to_hand, leader_area, combo | 5 | no | PASS |
| 10721120 传调联络兵 | add_to_hand, summon | 2 | yes | PASS |
| 10721310 敌我的调律 | cooperation | 1 | no | PASS |
| 10722110 听略谍报兵 | summon | 1 | yes | PASS |
| 10722120 斩奏医护兵 | draw, summon | 2 | yes | PASS |
| 10722310 无音的包围 | add_to_hand, cooperation | 2 | yes | PASS |
| 10723110 响爪分队长 | summon | 1 | yes | PASS |
| 10723310 带来静寂的拔刀 | cooperation | 1 | no | PASS |
| 10724110 统音的安纳提玛·吉尔达利娅 | summon, leader_area, cooperation | 4 | yes | PASS |
| 10724120 宽严的音帅·塞扎尔 | summon, destroy | 2 | yes | PASS |
| 10731110 小巧捕食者 | earth_sigils | 1 | no | PASS |
| 10731120 小型怪兽 | earth_sigils | 2 | no | PASS |
| 10731310 召唤仆从 | draw, earth_sigils | 2 | no | PASS |
| 10732110 迷人怪兽 | summon, earth_sigils | 3 | no | PASS |
| 10732120 甜蜜猎食者 | earth_sigils | 1 | no | PASS |
| 10732310 暴食的零嘴 | earth_sigils | 1 | no | PASS |
| 10733110 甜美存在 | draw, earth_sigils | 4 | no | PASS |
| 10733310 饕餮魔咒 | destroy, earth_sigils | 2 | no | PASS |
| 10734110 万食的安纳提玛·拉拉安瑟姆 | summon, destroy, leader_area, earth_sigils | 6 | no | PASS |
| 10734120 可爱杰作 | summon, earth_sigils | 2 | no | PASS |
| 10741110 宣扬的龙人 | summon | 1 | yes | PASS |
| 10741310 百无聊赖的睥睨 | transform | 1 | no | PASS |
| 10742110 豪龙守门人 | destroy | 1 | no | PASS |
| 10742310 焦龙的午睡 | draw, overflow | 2 | no | PASS |
| 10743110 龙人先驱者 | destroy | 1 | no | PASS |
| 10744110 焦灰的安纳提玛·班德奈特 | leader_area | 2 | yes | PASS |
| 10744120 龙峪的古龙 | summon, countdown_or_activate, leader_area | 5 | no | PASS |
| 10751110 暗夜键盘手·露露米 | add_to_hand | 1 | no | PASS |
| 10751120 恶魔鼓手·拉兹 | summon | 1 | yes | PASS |
| 10751310 灵魂调律 | draw | 1 | no | PASS |
| 10752110 猫咪走绳师 | summon | 2 | no | PASS |
| 10752120 乌鸦杂耍师 | summon, destroy, shadows_necromancy | 3 | no | PASS |
| 10752310 讴歌青春 | summon | 1 | no | PASS |
| 10753110 骸骨驯兽师 | summon, destroy | 2 | no | PASS |
| 10753310 夜之歌的演唱会 | shadows_necromancy | 1 | no | PASS |
| 10754110 傍死的安纳提玛·徒姬 | summon | 1 | no | PASS |
| 10754120 死亡主持人·马克米朗 | summon, shadows_necromancy | 2 | no | PASS |
| 10761110 营利支援者 | draw, summon | 2 | no | PASS |
| 10761120 广域传教士 | draw | 1 | no | PASS |
| 10761210 阳光耳饰 | draw, return_to_deck, destroy, countdown_or_activate | 7 | no | PASS |
| 10762120 传言圣鸟 | banish | 1 | no | PASS |
| 10762210 完美的时钟 | destroy, countdown_or_activate | 3 | no | PASS |
| 10763210 海蚀三叉戟 | destroy, countdown_or_activate | 3 | no | PASS |
| 10764110 裁神的安纳提玛·罗德欧 | summon, countdown_or_activate | 2 | no | PASS |
| 10764120 崇拜经理人·伊尼西雅 | banish | 2 | no | PASS |
| 10771110 个性店主 | summon | 1 | no | PASS |
| 10771120 炫酷舞者 | summon | 1 | no | PASS |
| 10771310 跑酷 | add_to_hand | 1 | yes | PASS |
| 10772110 悠然的滑手 | add_to_hand | 2 | yes | PASS |
| 10772120 大胆的涂鸦师 | summon, destroy | 2 | no | PASS |
| 10773110 狂野播报员 | summon | 2 | yes | PASS |
| 10774120 奋厉追赶·米乌 | summon | 1 | yes | PASS |
| 10802310 救世的英姿 | draw, add_to_hand | 2 | no | PASS |
| 10803110 遗忘的纯真·爱卡 | add_to_hand | 2 | no | PASS |
| 10803310 传承的意志 | summon, shadows_necromancy | 2 | yes | PASS |
| 10804110 阿尔比昂巴哈姆特 | banish, leader_area | 2 | yes | PASS |
| 10811110 昔日的天秤·马龙 | destroy | 1 | no | PASS |
| 10811120 异端隐士·西特拉斯 | summon | 2 | yes | PASS |
| 10811130 忧郁少女·莫埃尔 | draw, return_to_deck | 2 | no | PASS |
| 10812110 太古的妖精·露芙蕾 | add_to_hand, summon | 2 | yes | PASS |
| 10812120 情念的毒荆·莉柯瑞丝 | summon | 1 | yes | PASS |
| 10812310 宁静的孤独 | destroy | 1 | no | PASS |
| 10813110 温柔读心者·米榭儿 | summon | 1 | yes | PASS |
| 10813310 水镜的信赖 | summon | 1 | yes | PASS |
| 10814110 离合有终·赛德斯&梅希亚 | destroy | 1 | yes | PASS |
| 10814120 永恒冰晶·蒂亚 | add_to_hand | 1 | yes | PASS |
| 10821110 武力与治安·娜哈特·娜哈特&宾森特 | summon | 2 | yes | PASS |
| 10821130 悠久的骑士·莎夏 | summon | 2 | no | PASS |
| 10822110 越狱者·卡婕 | add_to_hand | 1 | no | PASS |
| 10822310 重历新生 | summon | 1 | no | PASS |
| 10824110 天命的子弹·巴妮&巴隆 | add_to_hand, summon, cooperation | 3 | yes | PASS |
| 10824120 焦灼炎将·玛尔斯 | summon | 2 | yes | PASS |
| 10831110 森绿的恩惠·喵鲁&圆滚滚2号&吉娜 | spellboost | 2 | no | PASS |
| 10831120 玛纳利亚书记官·波比 | add_to_hand | 1 | no | PASS |
| 10831310 暴风破 | spellboost | 1 | no | PASS |
| 10832110 快乐绽花·萨米&玛莉 | draw, spellboost | 2 | no | PASS |
| 10832310 其乐融融的团聚 | spellboost | 1 | no | PASS |
| 10832320 伏地雷击 | add_to_hand, earth_sigils | 2 | no | PASS |
| 10833110 玛纳利亚文书官·琪可 | add_to_hand, leader_area | 3 | no | PASS |
| 10833310 钢铁的小憩 | draw, spellboost | 2 | yes | PASS |
| 10834110 恩爱的大地·坦忒拉&拉缇卡 | add_to_hand, spellboost | 2 | no | PASS |
| 10834120 灾难言灵·洋荷 | summon, spellboost | 3 | no | PASS |
| 10841120 沙尘守宝龙 | draw | 1 | no | PASS |
| 10841130 沧海之精 | add_to_hand, leader_area | 3 | no | PASS |
| 10842110 闪耀旋律·莉芙&萝萝 | add_to_hand, summon | 2 | no | PASS |
| 10842120 满面笑容的烹饪·琪米卡 | draw, discard | 4 | yes | PASS |
| 10843310 狐火蜃景 | draw, return_to_deck, overflow | 3 | no | PASS |
| 10844110 反照的赤红·德莱克&亚瑞札特 | add_to_hand, leader_area | 3 | no | PASS |
| 10844120 金银绚烂·璐米欧儿&雅尔贞特 | draw, discard | 2 | yes | PASS |
| 10851120 可爱恶魔·莉莉姆 | add_to_hand | 1 | no | PASS |
| 10851130 兔耳恶魔·莉蜜儿 | summon | 2 | no | PASS |
| 10852110 母爱恶魔·菲欧蕾 | summon | 1 | no | PASS |
| 10852310 启程的退场 | shadows_necromancy | 1 | no | PASS |
| 10853310 改变的流向 | draw, return_to_deck | 2 | no | PASS |
| 10854110 出发的憧憬·苇剑&武津御 | draw | 1 | yes | PASS |
| 10854120 日月的蔷薇·赛蕾丝 | shadows_necromancy | 1 | yes | PASS |
| 10861110 图书室的魔女·莉莉尤姆 | draw | 1 | yes | PASS |
| 10861120 勤劳的女祭司·泰瑞莎 | summon | 1 | no | PASS |
| 10861130 亡灵猎人·格兰特 | destroy | 2 | no | PASS |
| 10862110 天阳的使徒·艾迪特 | summon, destroy | 2 | no | PASS |
| 10862310 威胁的残渣 | banish | 2 | no | PASS |
| 10863210 同窗好友 | draw, countdown_or_activate | 2 | yes | PASS |
| 10864110 飞跃的姐妹·贝尔迪俪亚&卡诗黛儿 | summon, leader_area | 3 | yes | PASS |
| 10864120 希望的光彩·莉迪耶尔 | summon, leader_area | 3 | yes | PASS |
| 10871110 满溢的幸福·库伦特司 | summon | 1 | no | PASS |
| 10871130 器械操纵者·吉尔克 | add_to_hand | 1 | no | PASS |
| 10872110 人造的馈赠·蕾拉 | add_to_hand | 1 | no | PASS |
| 10872120 门扉接续者·拉姿莉 | add_to_hand | 1 | yes | PASS |
| 10872310 纯净无垢的日常 | draw | 1 | no | PASS |
| 10873110 知恩图报·米莉亚姆 | summon, destroy | 2 | no | PASS |
| 10874110 决断的交错·亚修雷&莉缇雅 | destroy | 1 | yes | PASS |
| 10874120 你的前辈·欧丝 | add_to_hand | 1 | yes | PASS |
| 90004320 绝大的证明 | destroy | 1 | no | PASS |
| 90021350 闪耀的金币 | draw | 1 | no | PASS |
| 90031130 式神·小纸人 | spellboost | 1 | no | PASS |
| 90031140 式神·暴鬼 | spellboost | 1 | no | PASS |
| 90031210 大地之魔片 | countdown_or_activate, earth_sigils | 3 | no | PASS |
| 90032110 洋葱军团兵 | spellboost | 1 | no | PASS |
| 90033310 有所成长了！ | draw | 1 | no | PASS |
| 90034110 式神·天后 | spellboost | 1 | no | PASS |
| 90034130 安的巨大英灵 | destroy | 1 | no | PASS |
| 90034310 绝尽的伪证 | destroy | 1 | no | PASS |
| 90034330 天晶深渊 | summon | 1 | yes | PASS |
| 90044310 银冰吐息 | destroy | 1 | no | PASS |
| 90051130 怨灵 | banish | 1 | no | PASS |
| 90051140 腐臭的僵尸 | summon | 1 | no | PASS |
| 90054310 绝叫的扩散 | summon | 1 | yes | PASS |
| 90054330 天眼深渊 | draw | 1 | no | PASS |
| 90064210 月影指环 | countdown_or_activate | 1 | no | PASS |
| 90064310 绝望的奔流 | banish, countdown_or_activate | 2 | no | PASS |
| 90064320 天书深渊 | add_to_hand, destroy | 2 | no | PASS |
| 90071110 悬丝傀儡 | destroy | 1 | yes | PASS |
| 90071120 改良型·悬丝傀儡 | destroy | 1 | no | PASS |
| 90071130 解析的创造物 | draw | 1 | yes | PASS |
| 90071210 未来核心 | fusion | 1 | no | PASS |
| 90071220 过往核心 | fusion | 1 | no | PASS |
| 90072110 攻击创造物 | fusion | 1 | yes | PASS |
| 90072120 城堡创造物 | fusion | 1 | no | PASS |
| 90072130 屠戮人偶 | summon | 1 | no | PASS |
| 90073110 毁灭创造物α | fusion | 1 | yes | PASS |
| 90074210 新约·白之章 | summon, countdown_or_activate | 2 | yes | PASS |
| 90074220 新约·黑之章 | summon, countdown_or_activate | 2 | yes | PASS |
| 90074310 奏绝的独唱 | destroy | 1 | yes | PASS |
| 90074320 天斧深渊 | add_to_hand | 1 | no | PASS |
