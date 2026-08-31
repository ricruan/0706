"""
ds_api.py —— 调用 DeepSeek（深度求索）大模型的封装

【这个文件是干嘛的？—— 大白话】
把它想象成"给 DeepSeek 公司打电话的专用座机"：
- 其他文件想用 DeepSeek，只需要导入这里的 ds_chat() / ds_chat_stream() 两个函数。
- 这个文件内部已经帮你配好了"电话号码"（base_url）和"门禁卡"（API Key），
  别人不用管这些细节，直接喊一句"帮我问个问题"就行。

【关键术语】
- OpenAI 兼容协议：DeepSeek、通义千问、Ollama 等很多模型，都模仿了 OpenAI 的"接口格式"，
  所以用同一个 openai 库、换个 base_url 就能调用不同的模型。
- base_url：模型的"服务器地址"（就像快递的仓库地址）。
- API Key：你的"门禁卡"，证明你有权使用、用来计费。
"""

import os                       # 操作系统相关，这里用来读环境变量

from dotenv import load_dotenv  # 把 .env 文件里的内容加载成"环境变量"
from openai import OpenAI       # OpenAI 官方 SDK（也兼容 DeepSeek 等）

# 加载项目根目录的 .env 文件（里面存着 DEEPSEEK_API_KEY 等密钥）
load_dotenv()

# 创建一个"客户端"：相当于拨通 DeepSeek 的电话，之后都用这个 client 来对话
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),  # 从环境变量读密钥（密钥存在 .env，不写死在代码里）
    base_url="https://api.deepseek.com")    # DeepSeek 的服务器地址


def ds_chat(messages: list[str],
            model: str = "deepseek-v4-pro",
            reasoning_effort: str = 'low',
            extra_body=None):
    """一次性问答：把整段回答完整地拿回来。

    参数说明（新手友好）：
    - messages：对话记录列表，形如
        [{"role": "system", "content": "你是助手"},
         {"role": "user", "content": "你好"}]
      role='system' 是给模型的"总设定"，role='user' 是用户说的话。
      （小提示：这里类型注解写的是 list[str]，实际传的是 list[dict]，属于原作者的小笔误，
        不影响运行，因为 Python 不强制检查类型。）
    - model：用哪个模型，默认 deepseek-v4-pro
    - reasoning_effort：模型"思考用力程度"，low = 少想一点、快一点
    - extra_body：额外参数，默认开启"思考模式"

    返回：模型回复的纯文本（字符串）。
    """
    if extra_body is None:
        # 开启 DeepSeek 的"思考模式"（让模型先想再答）
        extra_body = {"thinking": {"type": "enabled"}}
    # 真正发起请求，等待模型把完整答案生成完
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,               # False = 一次性返回（等全部生成完）
        reasoning_effort=reasoning_effort,
        extra_body=extra_body
    )
    # 从返回结果里"抠"出真正的文字内容
    # response.choices[0] 是第一个（也是唯一的）候选回答，.message.content 是回答文本
    return response.choices[0].message.content


def ds_chat_stream(messages: list[str],
                   model: str = "deepseek-v4-pro",
                   reasoning_effort: str = 'low',
                   extra_body=None):
    """
    流式对话：返回生成器，逐段产出增量文本
    用法：
        for token in ds_chat_stream(messages):
            print(token, end='')

    【流式 vs 一次性 —— 大白话】
    - 一次性：像点外卖，等整份餐做好，一次性送到你面前（要等，但一次拿全）。
    - 流式：像火锅，菜一盘一盘上，边吃边等（不用干等，体验更好）。

    生成器（yield）就像"水龙头"：每次 yield 就"流出一小段文字"。
    """
    if extra_body is None:
        extra_body = {"thinking": {"type": "enabled"}}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,                # True = 流式：边生成边返回
        reasoning_effort=reasoning_effort,
        extra_body=extra_body
    )
    # 逐块读取模型"挤出来"的文字
    for chunk in response:
        # chunk.choices 可能为空（某些控制块），所以先判断一下，避免报错
        delta = chunk.choices[0].delta if chunk.choices else None
        # 只有真正有内容时，才 yield 出去（增量文本 delta.content）
        if delta and delta.content:
            yield delta.content


# 测试用的对话记录（供下面的 if __name__ == '__main__' 使用）
test_messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "今天天气怎么样"},
]


# 只有"直接运行本文件"时才执行（被 import 时不执行）
if __name__ == '__main__':
    res = ds_chat(test_messages)

    print(res)

    # 再问一个"依赖上下文"的问题，看看模型记不记得上一句
    # （注意：这里每次都是新起 messages，模型其实不记得上一轮，这是演示用）
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "我上一个问题问的什么"},
    ]

    res = ds_chat(test_messages)

    print(res)
