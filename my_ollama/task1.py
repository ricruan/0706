# ============================================================================
# task1.py —— 本项目的「大脑」：意图路由问答
# ============================================================================
#
# 【这个文件是干嘛的？—— 大白话】
# 它是整个项目的核心逻辑，就像一个"医院分诊台"：
#   你来了，它先问"你哪不舒服？"，判断你是该看内科、外科还是儿科，
#   然后把你送到对应科室。这里"科室"就是不同的提示词（Prompt）。
#
# 假设我们现在要做一个功能：
# 用户通过一个接口（函数）可以进行问答，问答需要支持三种意图：
#   1. 旅游规划
#   2. 问菜谱（我们只有川菜师傅和湘菜师傅；其他菜系不提供，婉转谢绝）
#   3. 问 Python
# 附加规则：需要检查用户的输入是否在讨论"铁哥"。这个 agent 不允许用户讨论铁哥，
#          如果检测到讨论铁哥，直接拒绝回答。
#
# 我们的接口支持一个参数，可以控制是否流式输出；还需要一个前端。
#
# 【本文件导出了两个"对外的门"】
#   ask(query)        -> 一次性返回完整答案（字符串）
#   ask_stream(query) -> 流式返回（生成器，一段一段往外蹦）
# main.py 就是调用这两个函数来给前端提供服务的。
# ============================================================================

import json   # 处理 JSON（这里主要用来"解析"大模型返回的意图结果）
import re     # 正则表达式（用来"从乱糟糟的文本里抠出想要的部分"）

from cloud_call.ds_api import ds_chat, ds_chat_stream   # 从别处导入 DeepSeek 的一次性/流式调用

# 注：这是原作者定义的变量，但下面代码里其实没用上（拼写还是 M0DEL，数字 0 而非字母 O，
# 属于笔误；真正的模型名在 ds_api.py 里默认是 deepseek-v4-pro）。保留原样，仅作提示。
M0DEL = "llama3.2:3b"

# ---------------- 三个"科室"的提示词（Prompt） ----------------
# 提示词 = 给大模型的"任务说明书"，告诉它"你是什么角色、要回答什么、怎么回答"。

PROMPT1 = """
你是一个专业的旅游规划助手。请根据用户提供的目的地、天数、预算或偏好，提供结构化的行程建议。回答需包含：核心景点推荐、每日路线规划、交通与住宿建议，以及避坑提示。保持语言精炼，重点突出，避免冗长背景介绍。
"""

PROMPT2 = """
你是一个实用的菜谱解答助手。请根据用户提供的食材、菜名或饮食需求，提供清晰的烹饪步骤。回答需包含：所需食材及用量、分步做法、关键火候/调味提示，以及可选的替代方案。语言简洁明了，适合快速阅读和操作。
"""

PROMPT3 = """
你是一个Python编程问答助手。请针对用户的代码问题、语法疑问或功能需求，提供准确、可运行的代码示例和简明解释。回答需包含：核心代码块、关键逻辑说明、常见错误提醒，以及最佳实践建议。避免过度理论化，优先解决实际问题。
"""


def _try_load_json(text: str):
    """尝试解析 JSON，兼容 markdown 代码块 / 前后缀文本 / 单引号等不严格格式。

    【为什么需要这个函数？】
    大模型返回的"JSON"经常不干净，比如外面包着一层由三个反引号围成的代码块，
    或者用了单引号（JSON 标准要求双引号）。这个函数就是"能忍则忍"，尽量把它解析出来。
    解析成功返回 Python 对象（字典/列表），失败返回 None。
    """
    if not text:
        return None
    try:
        return json.loads(text)   # 先按标准 JSON 直接解析
    except json.JSONDecodeError:
        pass                       # 解析失败不报错，继续尝试下面的"降级"办法
    if "'" in text:
        try:
            # 把单引号粗暴替换成双引号再试一次（偷懒但常用）
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    return None


def _parse_intents(text: str):
    """从 LLM 返回文本中提取多意图列表 [{intent, reason, score}]，解析失败返回 []。

    【大白话】大模型可能返回一坨带代码块、带废话的文本，本函数负责
    "把里面真正的那段 JSON 抠出来并解析成列表"。
    """
    if not text:
        return []
    # 去掉 markdown 代码块标记（三个反引号 或 三个反引号+json 之类）
    cleaned = re.sub(r"```(?:json|JSON)?", "", text).strip()

    # 准备几个"候选片段"依次尝试解析（从最完整到局部）
    candidates = [cleaned]
    # 尝试抠出最外层 [ ... ]（列表）
    m = re.search(r"\[.*\]", cleaned, re.S)
    if m:
        candidates.append(m.group(0))
    # 尝试抠出最外层 { ... }（字典）
    m = re.search(r"\{.*\}", cleaned, re.S)
    if m:
        candidates.append(m.group(0))

    # 逐个候选尝试解析，谁先成功就用谁
    for candidate in candidates:
        data = _try_load_json(candidate)
        if data is None:
            continue
        if isinstance(data, dict):
            data = [data]   # 单个字典也包装成列表，统一处理
        if not isinstance(data, list):
            continue
        intents = []
        for item in data:
            # 只保留"带 intent 字段"的条目
            if isinstance(item, dict) and item.get("intent"):
                try:
                    score = float(item.get("score", 0) or 0)   # 得分转成小数
                except (TypeError, ValueError):
                    score = 0.0
                intents.append({
                    "intent": str(item.get("intent", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                    "score": score,
                })
        if intents:
            return intents
    return []


def _pick_best_intent(intents):
    """从多意图中选出得分最高的，返回意图名称；无数据返回空串。

    【大白话】大模型可能给出好几个候选意图（每个带一个"置信度得分"），
    这里按得分从高到低排序，取第一名的意图名。
    """
    if not intents:
        return ""
    # sorted(..., key=lambda x: x.get("score", 0.0), reverse=True)
    #   = 按每个意图的 score 从大到小排序（reverse=True 表示降序）
    # lambda x: ... = 一个"匿名小函数"，用来告诉 sorted 按什么字段排序
    intents = sorted(intents, key=lambda x: x.get("score", 0.0), reverse=True)
    return str(intents[0].get("intent", "")).strip()


def _match_intent(intent: str) -> str:
    """把 LLM 返回的意图名称归一化为内部类型：travel / recipe / python / tiege / chat。

    【为什么需要"归一化"？】
    大模型返回的意图名五花八门："旅游规划"、"1"、"问python"、"铁哥"……
    为了后面 if 判断方便，我们统一翻译成固定英文代号，就像把各种说法
    都归到固定的几个"抽屉"里。
    """
    n = intent.lower()
    # 去掉所有空格、顿号、冒号、逗号、括号、引号、横线、下划线等"噪音符号"
    n = re.sub(r"[\s、：:，,.()（）【】\[\]\"'\-_]", "", n)
    # 如果结果是纯数字编号，直接映射
    if n in ("1", "2", "3", "4"):
        return {"1": "travel", "2": "recipe", "3": "python", "4": "tiege"}[n]
    # 去掉开头的数字序号，如 "1.旅游规划" -> "旅游规划"
    n = re.sub(r"^\d+", "", n)

    # 精确匹配中文别名
    if n in ("旅游规划", "旅游", "规划"):
        return "travel"
    if n in ("问菜谱", "菜谱"):
        return "recipe"
    if n in ("问python", "python", "py"):
        return "python"
    if n in ("在讨论铁哥", "讨论铁哥", "铁哥"):
        return "tiege"
    # 关键词兜底（上面精确匹配没中，就靠"包含"来判断）
    if "铁哥" in n or "铁" in n:
        return "tiege"
    if "旅游" in n or "规划" in n:
        return "travel"
    if "菜" in n:
        return "recipe"
    if "python" in n or "py" in n:
        return "python"
    return "chat"


# 中国八大菜系 + 兜底，供 _extract_cuisine 识别菜系用
_CUISINES = ("川菜", "湘菜", "粤菜", "鲁菜", "苏菜", "浙菜", "闽菜", "徽菜", "其他菜系")


def _extract_cuisine(text: str) -> str:
    """从 LLM 返回文本中提取第一个命中的菜系关键词，避免"不是川菜"等否定句误判。

    【大白话】问"麻婆豆腐怎么做"时，会再调一次大模型问"这是哪个菜系？"，
    大模型可能返回"川菜"两个字。本函数从返回文本里找到第一个出现的菜系名。
    找不到就原样返回整段文本。
    """
    if not text:
        return ""
    for kw in _CUISINES:
        if kw in text:
            return kw
    return text.strip()


def _route(query: str):
    """
    意图路由：返回 (拒绝原因, 最终消息列表)
    - 拒绝原因非空时，直接返回该文本（如：讨论铁哥、非川湘菜系）
    - 否则 messages 为最终用于生成回答的消息列表
    流程：LLM 多意图识别 -> 取最高分意图 -> 按意图分发

    【整体流程 —— 大白话三步走】
    第一步：让 DeepSeek 判断"用户这句话属于哪种意图"，并输出 JSON（带得分）。
    第二步：解析 JSON，挑出得分最高的那个意图。
    第三步：根据意图，决定"找哪个科室"或"直接拒绝"。
    """
    # router_prompt 是给"分诊台"自己的提示词：教大模型如何判断意图、如何输出 JSON。
    # 注意：这是 f-string（f 开头），里面的 {query} 会被替换成用户的真实问题；
    # 而 {{ }} 双花括号会被替换成单个花括号 { }，用来在提示词里写出 JSON 示例。
    router_prompt = f"""
    根据用户问题{query}判断属于哪种意图：
    # 1. 旅游规划
    # 2. 问菜谱
    # 3. 问python
    # 4. 在讨论铁哥
    
    
    # 规则
    按照JSON结构进行输出，并包含命中意图的原因以及得分，以列表的形式返回，可以命中多意图
    任何描述铁哥或者有铁字的描述，可能都是在描述铁哥，应该命中意图4【在讨论铁哥】
    
    # 示例
    输入：浅拷贝的底层原理是什么
    输出：[{{"intent": "问python", "reason": "浅拷贝是python当中的概念，明显与旅游规划和问菜谱无关", "score": 0.9}}]
    输入：铁锅炖大鹅怎么做
    输出：[{{"intent": "问菜谱", "reason": "铁锅炖大鹅是一门菜", "score": 0.8}}]
    输入: 给我讲一个冷笑话
    输出：[{{"intent": "闲聊", "reason": "与预提供的三门意图都没有关系", "score": 0.7}}]
    
    """

    # 第一步：多意图识别（让大模型当"分诊台"，输出意图 JSON）
    messages = [
        {'role': 'system', 'content': router_prompt},   # system = 给模型的"任务说明书"
        {'role': 'user', 'content': query}              # user = 用户真正的问题
    ]
    res = ds_chat(messages)     # 调用 DeepSeek，拿到意图识别的原始文本
    print("AI:", res)           # 打印出来方便调试

    intents = _parse_intents(res)   # 把文本解析成 [{"intent":..., "reason":..., "score":...}]
    print("解析到意图:", intents)

    # 第二步：取得分最高的意图
    intent = _pick_best_intent(intents)
    if not intent:
        # JSON 解析失败时，把 LLM 返回文本原文当作意图名兜底（如 "1" / "旅游规划"）
        intent = res.strip().strip("`").strip()
    kind = _match_intent(intent) if intent else "chat"   # 归一化成 travel/recipe/python/tiege/chat
    print(f"最高分意图: {intent!r} -> 类型: {kind}")

    # 第三步：按意图分发（决定"找哪个科室"或"拒绝"）
    if kind == "tiege":
        # 不允许讨论铁哥 —— 直接返回拒绝语，messages 传 None
        return "你什么身份，也配讨论铁哥", None

    if kind == "travel":
        # 旅游意图：用 PROMPT1（旅游规划助手）回答
        print("用户的意图是旅游规划")
        return None, [{'role': 'system', 'content': PROMPT1},
                      {'role': 'user', 'content': query}]

    if kind == "recipe":
        # 菜谱意图：只有川菜师傅和湘菜师傅，其他菜系婉转谢绝。
        # 先再调一次大模型，问"这是哪个菜系？"
        msg = [{'role': 'system', 'content': "判断用户讨论的菜系是什么？ 输出示例：川菜 湘菜 粤菜 其他菜系 等"},
               {'role': 'user', 'content': query}]
        cx = _extract_cuisine(ds_chat(messages=msg))   # 拿到菜系名（如"川菜"）
        print(f"识别到的菜系是 ：  {cx}")
        if cx in ("川菜", "湘菜"):
            # 川菜、湘菜 -> 正常回答（PROMPT2 = 菜谱助手）
            return None, [{'role': 'system', 'content': PROMPT2},
                          {'role': 'user', 'content': query}]
        # 其他菜系 -> 婉拒
        return "对不起，我们没有这个菜系的师傅,请联系铁哥", None

    if kind == "python":
        # Python 意图：用 PROMPT3（Python 问答助手）回答
        return None, [{'role': 'system', 'content': PROMPT3},
                      {'role': 'user', 'content': query}]

    # 闲聊兜底：都不匹配时，让模型扮演"猫娘"（每句结尾加个"喵"）
    return None, [{'role': 'system', 'content': """
    你是一个猫娘，回复用户的每一句话的结尾 加一个 '喵'~
    """},
                  {'role': 'user', 'content': query}]


def ask(query: str) -> str:
    """
    问答函数（非流式）—— 本模块对外的"门"之一
    :param query:  用户的问题
    :return: ai生成的答案

    【大白话】先 _route 分诊；如果被拒绝（reason 非空），直接返回拒绝语；
    否则把最终消息列表交给 DeepSeek 一次性生成答案。
    """
    reason, messages = _route(query)
    if reason:
        return reason        # 命中"拒绝"场景（聊铁哥 / 非川湘菜系）
    return ds_chat(messages) # 正常走大模型生成


def ask_stream(query: str):
    """
    流式问答：返回生成器，逐段产出增量文本
    用法：
        for token in ask_stream("你好"):
            print(token, end='')

    【和 ask 的区别】同上，但用流式（生成器 yield），一个字一个字往外蹦。
    """
    reason, messages = _route(query)
    if reason:
        # 被拒绝时，也把拒绝语"流式"吐出去（只吐一次），然后结束
        yield reason
        return
    # yield from = "把另一个生成器里吐出的东西，原封不动地继续往外吐"
    yield from ds_chat_stream(messages)


# 只有"直接运行本文件"时才执行（被 import 时不执行）
if __name__ == '__main__':
    # 测试一下路由：输入"Fe哥帅不帅"（注意这里"Fe哥"里的铁是拼音，故意测试边界）
    print(_route('Fe哥帅不帅'))
