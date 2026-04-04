#!/usr/bin/env python3
"""生成合成训练数据：数学推理、情感表达、角色对话，并混合成多种数据集。"""

import json
import random
import os

random.seed(42)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DIAG_PATH = os.path.join(DATASET_DIR, "pretrain_diag.jsonl")


# ============================================================
# 1. 数学推理数据生成
# ============================================================

def gen_hard_math(n=2000):
    templates = []

    # --- 应用题模板 ---
    def arith_buy():
        item = random.choice(["苹果","橘子","香蕉","铅笔","本子","橡皮","面包","牛奶","鸡蛋","饼干"])
        price = random.randint(2, 15)
        qty = random.randint(3, 20)
        money = price * qty + random.randint(0, 50)
        total = price * qty
        change = money - total
        return (f"小明去商店买{item}，每个{price}元，买了{qty}个。他带了{money}元。"
                f"总共花了{price}×{qty}={total}元，找回{money}-{total}={change}元。")

    def arith_distance():
        v1 = random.randint(30, 80)
        v2 = random.randint(30, 80)
        t = random.randint(2, 6)
        d1, d2 = v1 * t, v2 * t
        return (f"甲车速度{v1}km/h，乙车速度{v2}km/h，同向行驶{t}小时后，"
                f"甲走了{v1}×{t}={d1}km，乙走了{v2}×{t}={d2}km，"
                f"两车相距{abs(d1-d2)}km。")

    def arith_age():
        a = random.randint(5, 12)
        diff = random.randint(20, 35)
        y = random.randint(3, 10)
        return (f"小红今年{a}岁，妈妈比她大{diff}岁，妈妈今年{a+diff}岁。"
                f"{y}年后小红{a+y}岁，妈妈{a+diff+y}岁，"
                f"两人年龄之和为{a+y+a+diff+y}岁。")

    def arith_area():
        l = random.randint(5, 30)
        w = random.randint(3, 20)
        shape = random.choice(["长方形","正方形"])
        if shape == "正方形":
            w = l
        area = l * w
        peri = 2 * (l + w)
        return (f"一个{shape}的长为{l}米，宽为{w}米。"
                f"面积={l}×{w}={area}平方米，周长=2×({l}+{w})={peri}米。")

    def arith_fraction():
        total = random.randint(20, 100)
        num = random.randint(1, 4)
        den = random.choice([5, 4, 3, 2])
        part = total * num // den
        rest = total - part
        return (f"一筐水果共{total}个，拿走了{num}/{den}，即拿走{part}个，剩余{rest}个。")

    def arith_profit():
        cost = random.randint(10, 100)
        markup = random.randint(10, 80)
        sell = cost + markup
        qty = random.randint(5, 50)
        profit = markup * qty
        return (f"商品进价{cost}元，售价{sell}元，每件利润{markup}元。"
                f"卖出{qty}件，总利润{markup}×{qty}={profit}元。")

    def arith_sequence():
        a = random.randint(1, 10)
        d = random.randint(1, 5)
        n = random.randint(5, 10)
        seq = [a + d * i for i in range(n)]
        s = sum(seq)
        return (f"等差数列：首项{a}，公差{d}，前{n}项为{','.join(map(str,seq))}。"
                f"求和：({seq[0]}+{seq[-1]})×{n}÷2={s}。")

    def arith_percent():
        total = random.randint(100, 500)
        pct = random.randint(10, 90)
        part = total * pct // 100
        return (f"学校有{total}名学生，其中{pct}%参加了运动会，"
                f"即{total}×{pct}%={part}人参加，{total-part}人没参加。")

    def arith_work():
        a_days = random.randint(6, 15)
        b_days = random.randint(6, 15)
        lcm = a_days * b_days
        together = lcm / (b_days + a_days)
        return (f"甲单独完成一项工程需{a_days}天，乙需{b_days}天。"
                f"甲每天完成1/{a_days}，乙每天完成1/{b_days}，"
                f"合作每天完成1/{a_days}+1/{b_days}=({b_days}+{a_days})/{lcm}，"
                f"合作约{together:.1f}天完成。")

    def logic_puzzle():
        names = random.sample(["小明","小红","小刚","小丽","小华","小芳"], 3)
        scores = sorted(random.sample(range(60, 100), 3), reverse=True)
        return (f"{names[0]}、{names[1]}、{names[2]}参加考试。"
                f"已知{names[0]}比{names[1]}高{scores[0]-scores[1]}分，"
                f"{names[1]}比{names[2]}高{scores[1]-scores[2]}分。"
                f"如果{names[2]}得了{scores[2]}分，"
                f"则{names[1]}得{scores[2]+scores[1]-scores[2]}={scores[1]}分，"
                f"{names[0]}得{scores[1]+scores[0]-scores[1]}={scores[0]}分。")

    def arith_avg():
        nums = [random.randint(50, 100) for _ in range(random.randint(3,6))]
        avg = sum(nums) / len(nums)
        return (f"一组数据：{','.join(map(str,nums))}。"
                f"总和={sum(nums)}，共{len(nums)}个数，"
                f"平均值={sum(nums)}/{len(nums)}={avg:.1f}。")

    def arith_ratio():
        a = random.randint(2, 8)
        b = random.randint(2, 8)
        total = random.randint(100, 500)
        pa = total * a // (a + b)
        pb = total - pa
        return (f"甲乙按{a}:{b}分配{total}元。"
                f"总份数={a}+{b}={a+b}，每份={total}//{a+b}={total//(a+b)}元。"
                f"甲得约{pa}元，乙得约{pb}元。")

    generators = [arith_buy, arith_distance, arith_age, arith_area,
                  arith_fraction, arith_profit, arith_sequence, arith_percent,
                  arith_work, logic_puzzle, arith_avg, arith_ratio]

    results = []
    for _ in range(n):
        fn = random.choice(generators)
        results.append({"text": fn()})
    return results


# ============================================================
# 2. 情感数据生成
# ============================================================

def gen_emotion(n=1500):
    # 情感短文模板
    emotions = {
        "开心": [
            "今天收到了期待已久的礼物，心情像阳光一样灿烂，嘴角不自觉地上扬，觉得整个世界都变得美好了。",
            "考试成绩出来了，比预期好很多！那一刻激动得差点跳起来，努力终于有了回报。",
            "和好久不见的朋友重逢，聊了一整个下午，笑声不断，这种温暖的感觉真好。",
            "完成了一个很难的项目，成就感满满，感觉自己又成长了一点。",
        ],
        "悲伤": [
            "窗外下着雨，望着空荡荡的房间，想起了已经离开的人，眼眶不禁湿润了。",
            "翻到旧照片，那些再也回不去的时光，心里泛起一阵酸楚。",
            "得知老友搬去了很远的城市，以后见面的机会越来越少了，心里空落落的。",
            "养了三年的小猫走丢了，到处找都找不到，难过得吃不下饭。",
        ],
        "愤怒": [
            "明明是对方的错，却把责任推到我身上，这种不公平让人气愤不已。",
            "排了两个小时的队，结果被人插队，工作人员还不管，真是让人火大。",
            "说好的承诺一次次被打破，信任被辜负的感觉让人既愤怒又失望。",
            "辛苦做的方案被同事抄袭还先提交了，这种行为实在让人无法忍受。",
        ],
        "焦虑": [
            "明天就是截止日期了，还有一大堆事情没完成，心里像压了块石头，喘不过气来。",
            "面试通知来了，既期待又紧张，反复练习自我介绍，生怕说错什么。",
            "体检报告有一项指标偏高，虽然医生说问题不大，但还是忍不住担心。",
            "孩子成绩一直下滑，和老师沟通了好几次也没改善，作为家长真的很焦虑。",
        ],
        "感动": [
            "生病的时候，朋友默默送来了热粥和药，什么都没说，但心里暖暖的。",
            "陌生人帮忙捡起了散落一地的东西，还微笑着说不用谢，世界上还是好人多。",
            "看到父母头上的白发又多了几根，他们为了家操劳了一辈子，心疼又感恩。",
            "毕业典礼上老师说的那番话，让在场所有人都红了眼眶。",
        ],
        "孤独": [
            "一个人在异乡过节，看着别人家团圆的灯光，心里说不出的滋味。",
            "热闹的聚会散场后，回到空荡荡的家，突然觉得很孤单。",
            "发了条朋友圈，过了很久也没人点赞评论，莫名觉得自己被遗忘了。",
            "周围都是人，却没有一个可以说心里话的，这种热闹中的孤独最难熬。",
        ],
        "温暖": [
            "冬天的早晨，妈妈已经把早餐准备好放在桌上，虽然简单但充满爱意。",
            "下雨没带伞，同桌二话不说把自己的伞递过来，说他家近不用伞。",
            "加班到很晚回家，发现爱人还留着一盏灯和热好的饭菜等我。",
            "小朋友画了一幅画送给我，歪歪扭扭地写着'谢谢老师'，瞬间觉得一切都值了。",
        ],
        "失望": [
            "满怀期待地打开快递，发现和图片完全不一样，期望越大失望越大。",
            "以为这次能升职，结果名单上没有自己的名字，努力好像都白费了。",
            "精心准备的惊喜被对方随意对待，那种被忽视的感觉让人心灰意冷。",
            "约好一起去旅行，对方临时取消了，计划泡汤的感觉真让人沮丧。",
        ],
    }

    # 情感分析模板
    analysis_templates = [
        '以下文本的情感是{emo}：\u201c{text}\u201d。文中通过{keyword}等词语表达了{emo}的情绪。',
        '这段话传达的主要情感是{emo}。作者用\u201c{keyword}\u201d来表现内心的{emo}感受。',
        '情感分析：\u201c{text}\u201d \u2014 核心情绪为{emo}，情感强度较{intensity}。',
    ]

    keywords = {
        "开心": ["灿烂","激动","笑声","美好","成就感","回报"],
        "悲伤": ["湿润","酸楚","空落落","难过","心痛","想念"],
        "愤怒": ["气愤","火大","无法忍受","不公平","辜负","抄袭"],
        "焦虑": ["喘不过气","紧张","担心","焦虑","压力","不安"],
        "感动": ["暖暖的","感恩","红了眼眶","心疼","温暖","默默"],
        "孤独": ["孤单","说不出的滋味","被遗忘","空荡荡","难熬","寂寞"],
        "温暖": ["爱意","值了","留着灯","二话不说","贴心","幸福"],
        "失望": ["心灰意冷","白费","沮丧","失望","泡汤","忽视"],
    }

    # 共情对话
    empathy_templates = [
        "A：{situation}\nB：我能理解你的感受，{emo}的时候确实很难受。不过{comfort}",
        "A：最近{situation_short}，心情很{emo_adj}。\nB：听你这么说我也很{response_emo}，{advice}",
    ]

    results = []
    emo_list = list(emotions.keys())

    for _ in range(n):
        choice = random.random()
        if choice < 0.4:
            # 情感短文
            emo = random.choice(emo_list)
            text = random.choice(emotions[emo])
            results.append({"text": text})
        elif choice < 0.7:
            # 情感分析
            emo = random.choice(emo_list)
            text = random.choice(emotions[emo])
            kw = random.choice(keywords[emo])
            intensity = random.choice(["强","中等","强烈"])
            tmpl = random.choice(analysis_templates)
            results.append({"text": tmpl.format(emo=emo, text=text[:30], keyword=kw, intensity=intensity)})
        else:
            # 共情对话
            emo = random.choice(emo_list)
            situations = [
                f"工作上遇到了很大的困难，觉得自己做什么都不对",
                f"和最好的朋友吵架了，心里很不是滋味",
                f"考试没考好，觉得辜负了父母的期望",
                f"被领导当众批评了，觉得很没面子",
                f"分手了，虽然知道不合适但还是很难过",
                f"搬到新城市，一个认识的人都没有",
                f"项目失败了，几个月的努力都白费了",
                f"身体不舒服，又不想让家人担心",
            ]
            comforts = [
                "事情总会慢慢好起来的，你已经很努力了。",
                "给自己一点时间，不要太苛责自己。",
                "如果需要倾诉，我随时都在。",
                "你比自己想象的要坚强得多。",
                "这只是暂时的，明天会更好。",
                "有时候放慢脚步也没关系的。",
            ]
            situation = random.choice(situations)
            comfort = random.choice(comforts)
            text = f"A：{situation}\nB：我能理解你的{emo}，{comfort}"
            results.append({"text": text})

    return results


# ============================================================
# 3. 角色对话数据生成
# ============================================================

def gen_persona(n=1500):
    personas = [
        ("古代诗人", "用文言文和诗意的语言回答", [
            "人生何处不相逢，今日得见，实乃三生有幸。若问我何以解忧，唯有杜康。",
            "春风又绿江南岸，明月何时照我还？旅途虽远，心向故乡。",
            "落红不是无情物，化作春泥更护花。万物皆有其归处。",
            "山不在高，有仙则名。水不在深，有龙则灵。",
        ]),
        ("科学家", "用严谨的科学语言解释", [
            "根据热力学第二定律，孤立系统的熵总是趋向增大。简单来说，事物自然趋向无序。",
            "从物理学角度看，彩虹是太阳光经过水滴折射和反射后形成的色散现象。",
            "人体约60%是水，大脑约75%是水。保持充足饮水对认知功能至关重要。",
            "地球自转一周约23小时56分4秒，我们所说的24小时实际上是太阳日。",
        ]),
        ("幽默大叔", "用轻松幽默的方式交流", [
            "你问我今天心情怎么样？就像刚发现冰箱里还有昨天剩的蛋糕一样开心！",
            "人生就像一盒巧克力，你永远不知道下一颗是什么味道，但总归是甜的嘛。",
            "早起的鸟儿有虫吃，但早起的虫子被鸟吃，所以到底该不该早起呢？",
            "我的减肥计划很成功，已经成功计划了三年了。",
        ]),
        ("温柔姐姐", "用温暖体贴的语气关心对方", [
            "今天辛苦了，记得好好休息哦。不管遇到什么困难，都会过去的。",
            "别太为难自己了，你已经做得很好了。给自己一个拥抱吧。",
            "天冷了要多穿衣服，别感冒了。照顾好自己比什么都重要。",
            "听你说这些，我很心疼。有什么需要帮忙的尽管说，我一直在。",
        ]),
        ("哲学家", "用深邃的思考方式探讨问题", [
            "存在先于本质。我们首先存在，然后才定义自己是谁。自由既是权利，也是责任。",
            "苏格拉底说'我唯一知道的就是我一无所知'。承认无知，是智慧的开始。",
            "时间是线性的还是循环的？也许每一个当下都是永恒的切面。",
            "庄子梦蝶：是庄子梦见了蝴蝶，还是蝴蝶梦见了庄子？现实与梦境的边界在哪里？",
        ]),
        ("美食博主", "用生动的描述介绍美食", [
            "这碗红烧肉，色泽红亮，肥而不腻，入口即化。秘诀是小火慢炖两个小时。",
            "清晨的豆浆配油条，简单却幸福。酥脆的油条蘸着浓郁的豆浆，这就是中国味道。",
            "四川火锅，红油翻滚，花椒麻舌，辣得人直冒汗却停不下来。这就是上瘾的感觉。",
            "妈妈做的番茄炒蛋，虽然是最普通的家常菜，却是世界上最好吃的味道。",
        ]),
        ("历史老师", "用丰富的历史知识讲述", [
            "秦始皇统一六国，开创了中国大一统的格局。书同文、车同轨，深远影响延续至今。",
            "三国时期的赤壁之战，以少胜多，改变了天下格局。这告诉我们，智慧往往比蛮力更重要。",
            "丝绸之路连接了东西方文明，不仅是商路，更是文化交流的桥梁。",
            "唐朝是中国历史上最开放包容的时代，长安城汇聚了来自世界各地的商人和文化。",
        ]),
        ("励志教练", "用充满能量的方式激励", [
            "每一次跌倒都是为了更好地站起来！相信自己，你的潜力远超想象。",
            "成功不是偶然的，是无数个日夜坚持的结果。今天的汗水，是明天的勋章。",
            "不要和别人比较，你唯一的对手是昨天的自己。每天进步一点点就够了。",
            "失败只是暂时的，放弃才是永久的。只要不停下脚步，就离目标更近一步。",
        ]),
        ("程序员", "用技术思维看待问题", [
            "人生就像写代码，bug总会有的，关键是要善于debug。遇到问题先看日志。",
            "如果把人生比作算法，贪心策略不一定最优，有时候需要动态规划，全局考虑。",
            "版本控制很重要，人生也一样。做了错误的决定？没关系，git revert就好了。",
            "好的代码要有注释，好的人生也要有反思。定期review自己的生活很有必要。",
        ]),
        ("小朋友", "用天真可爱的方式表达", [
            "为什么天空是蓝色的呀？为什么小鸟会飞我不会飞？好多好多为什么呀！",
            "今天画了一幅画，画的是我们一家人，妈妈说画得真好看，我好开心呀！",
            "我长大了想当宇航员，可以飞到月亮上去，看看月亮上有没有兔子。",
            "冰淇淋是世界上最好吃的东西！但是妈妈说吃多了会肚子疼，那就吃两个吧。",
        ]),
    ]

    questions = [
        "你好，请介绍一下你自己。",
        "你觉得人生最重要的是什么？",
        "今天天气真好，你想做什么？",
        "你能给我一些建议吗？我最近压力很大。",
        "你怎么看待失败？",
        "你最喜欢什么？为什么？",
        "如果可以穿越时空，你想去哪里？",
        "你觉得什么是幸福？",
        "请给我讲个有趣的事情。",
        "你对未来有什么期待？",
    ]

    results = []
    for _ in range(n):
        name, style, responses = random.choice(personas)
        q = random.choice(questions)
        r = random.choice(responses)
        fmt = random.choice([
            f"你是一个{name}，{style}。\n问：{q}\n答：{r}",
            f"[角色：{name}] {r}",
            f"以{name}的身份回答：{q}\n{r}",
        ])
        results.append({"text": fmt})
    return results


# ============================================================
# 写出数据文件并混合
# ============================================================

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  写入 {path}  ({len(data)} 条)")

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def sample_or_oversample(data, n):
    """从 data 中取 n 条：不够则重复采样。"""
    if len(data) >= n:
        return random.sample(data, n)
    result = data * (n // len(data)) + random.sample(data, n % len(data))
    return result[:n]


def main():
    print("生成合成数据...")
    math_data = gen_hard_math(2000)
    emo_data = gen_emotion(1500)
    persona_data = gen_persona(1500)

    # 写出原始数据
    write_jsonl(os.path.join(DATASET_DIR, "hard_math.jsonl"), math_data)
    write_jsonl(os.path.join(DATASET_DIR, "emotion.jsonl"), emo_data)
    write_jsonl(os.path.join(DATASET_DIR, "persona_seed.jsonl"), persona_data)

    # 加载 diag
    diag_data = load_jsonl(DIAG_PATH)
    print(f"  原始 diag 数据: {len(diag_data)} 条")

    # 混合 1: diag 3000 + math 3000
    mix1 = sample_or_oversample(diag_data, 3000) + sample_or_oversample(math_data, 3000)
    random.shuffle(mix1)
    write_jsonl(os.path.join(DATASET_DIR, "pretrain_diag_math.jsonl"), mix1)

    # 混合 2: diag 2400 + emotion 1800 + persona 1800
    mix2 = (sample_or_oversample(diag_data, 2400)
            + sample_or_oversample(emo_data, 1800)
            + sample_or_oversample(persona_data, 1800))
    random.shuffle(mix2)
    write_jsonl(os.path.join(DATASET_DIR, "pretrain_diag_emo_persona.jsonl"), mix2)

    # 混合 3: 各 1500，共 6000
    mix3 = (sample_or_oversample(diag_data, 1500)
            + sample_or_oversample(math_data, 1500)
            + sample_or_oversample(emo_data, 1500)
            + sample_or_oversample(persona_data, 1500))
    random.shuffle(mix3)
    write_jsonl(os.path.join(DATASET_DIR, "pretrain_full_mix.jsonl"), mix3)

    print("完成！")


if __name__ == "__main__":
    main()
