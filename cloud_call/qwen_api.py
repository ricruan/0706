"""
qwen_api.py —— 通义千问（DashScope）大模型调用封装

设计目标：
1. 易用性：函数式一行调用 qwen_chat("你好")；类式支持多轮会话 / 流式输出
2. 可扩展性：模型、API Key、超时、重试均可配置；底层基于 OpenAI 兼容协议，
   换用 DeepSeek / Ollama 等只需改 base_url 与模型名，代码结构可直接复用

依赖：pip install openai python-dotenv
配置：在项目根目录 .env 中写入 DASHSCOPE_API_KEY=sk-xxxx

【这个文件是干嘛的？—— 大白话】
这是"通义千问的专用电话本"，比 ds_api.py 更高级、更完整：
- 它用"类（class）"把一堆功能打包在一起，像一部功能齐全的智能手机。
- 最厉害的一点：同一个 QwenClient 实例会"自动记住聊过的内容"，实现多轮对话。
  （就像你和一个真人聊天，他记得你上一句说了什么，能接得上话。）

【先搞懂三个概念】
1. 类（class）：一个"模板"，描述某类东西有什么"属性"（数据）和"方法"（功能）。
   QwenClient 就是一个类；用 QwenClient(...) 造出来的 bot 叫"实例"。
   类比：类 = 手机设计图纸；实例 = 你手上真正能用的那部手机。
2. 实例方法：属于实例的函数，调用时写 实例.方法名()，如 bot.ask("你好")。
3. self：类里面每个方法的第一个参数，指"当前这个实例自己"。
   就像说"我自己"——bot.ask() 里的 self 就代表 bot 这部手机。
"""

import os                                   # 操作系统相关，读环境变量
from typing import Callable, Iterator, List, Optional   # 类型标注工具（只起提示作用，不影响运行）

from dotenv import load_dotenv              # 加载 .env 文件
from openai import OpenAI                   # OpenAI 兼容 SDK

load_dotenv()                                # 真正加载 .env（这样后面才能读到密钥）

DEFAULT_MODEL = "qwen-max"  # 原示例中的 qwen3.8-max 为笔误，默认改为官方旗舰版；可选 qwen-plus / qwen-turbo / qwen3-max
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问的服务器地址


class QwenClient:
    """通义千问客户端封装。

    同一个实例的 ask() / ask_stream() 会自动维护多轮会话上下文；
    多轮对话直接复用实例即可，无需手动拼接历史。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
        **client_kwargs,
    ):
        """构造方法：造一部"通义千问手机"时自动执行，负责初始化各种设置。

        类比：买手机开机时，自动帮你插好 SIM 卡、设置语言、连上 WiFi。

        参数解释：
        - api_key：门禁卡；不传就从环境变量 DASHSCOPE_API_KEY 读
        - model：默认用哪个模型
        - base_url：服务器地址
        - timeout：超时时间（秒），超过就放弃
        - max_retries：失败自动重试几次
        - system_prompt：可选，给模型的"总设定"
        - **client_kwargs：把多余的关键字参数"打包成字典"，原样透传给底层 SDK
          （** 的作用 = "把剩下的零散参数都塞进一个麻袋里传下去"）
        """
        # 优先用传入的 api_key，没传就找环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            # 没找到密钥就立刻报错，提醒用户去配置 .env
            raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 配置")
        self.model = model
        # 创建真正的底层客户端（真正"拨通电话"）
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **client_kwargs,
        )
        # 会话历史：供 ask() / ask_stream() 自动维护多轮上下文
        # 就像一个"聊天记录本"，每轮对话都记在上面，模型才能"记得"聊过什么
        self._messages: List[dict] = []
        if system_prompt:
            self.set_system(system_prompt)

    # ---------- 会话管理 ----------

    def set_system(self, system_prompt: str) -> None:
        """设置系统提示词（会清空已有历史，重新开始新会话）。

        system_prompt 是给模型的"总设定"，比如"你是一个耐心的老师"。
        调用它会"清空聊天记录，重新开始"，所以多轮对话中途慎用。
        """
        self._messages = [{"role": "system", "content": system_prompt}]

    def clear(self) -> None:
        """清空会话历史（把聊天记录本擦干净）"""
        self._messages = []

    @property
    def history(self) -> List[dict]:
        """当前会话历史（返回副本，避免外部误改）。

        @property 装饰器的作用：让 history 可以像"属性"一样访问（bot.history），
        而不是像方法一样调用（bot.history()）。
        """
        # list(self._messages) 会复制一份，别人改副本不影响原件
        return list(self._messages)

    # ---------- 核心调用 ----------

    def chat(
        self,
        messages: Optional[List[dict]] = None,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """一次性对话（不维护历史）。messages 省略时使用内部会话历史。

        :param messages: 完整消息列表，如 [{"role": "user", "content": "你好"}]
        :param model:    临时指定模型，覆盖实例默认模型
        :param temperature: 采样温度，0~2，越大越随机
        :param kwargs:   透传给 OpenAI SDK 的其他参数，如 max_tokens、top_p
        :return: 完整回复文本

        （* 单独成参数：表示它后面的 model、temperature 必须用"关键字"方式传，
          比如 chat(messages, temperature=0.5)，不能写成 chat(messages, 0.5)。）
        """
        # 没传 messages 就用内部记录的历史；若还是没有，就报错
        messages = messages or self._messages
        if not messages:
            raise ValueError("messages 为空，请传入消息或先调用 ask()")
        model = model or self.model   # 没指定就用默认模型
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            # 把底层报错包装成更友好的提示（可能因为密钥错、没钱、模型名错）
            raise RuntimeError(
                f"调用 {model} 失败：{e}（请检查 API Key、账户余额与模型名）"
            ) from e
        return resp.choices[0].message.content

    def ask(
        self,
        question: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """单轮问答（自动把问答追加到历史，支持多轮上下文）。

        :param question: 用户问题
        :param system:   传入时重置会话并设置系统提示词；不传则沿用当前历史
        :param kwargs:   透传给 chat()
        """
        if system is not None:
            self.set_system(system)   # 传了 system 就"重新开始 + 设定角色"
        # 把用户问题追加到聊天记录
        self._messages.append({"role": "user", "content": question})
        try:
            answer = self.chat(self._messages, **kwargs)
        except Exception:
            # 调用失败就"回滚"：把刚追加的用户问题删掉，避免记录错乱
            self._messages.pop()
            raise
        # 把 AI 的回答也追加到聊天记录，这样下一轮模型就知道自己刚说过什么
        self._messages.append({"role": "assistant", "content": answer})
        return answer

    def ask_stream(
        self,
        question: str,
        system: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """流式问答：逐段产出增量文本（yield），生成完毕后自动写入历史。

        :param question: 用户问题
        :param system:   同上 ask()
        :param on_token: 可选回调，收到每个增量片段时调用，如 on_token=print
        :param kwargs:   透传给 OpenAI SDK
        :yield: 增量文本片段

        【on_token 回调 —— 大白话】
        "回调"就是"回头叫你一声"。你传一个函数进去，每收到一小段文字，
        它就用这个函数"通知你一下"。比如 on_token=print 表示每收到一段就打印一段。
        """
        if system is not None:
            self.set_system(system)
        self._messages.append({"role": "user", "content": question})
        collected: List[str] = []   # 用个"口袋"把零散片段收集起来，最后拼成完整回答
        try:
            stream = self._client.chat.completions.create(
                model=kwargs.pop("model", self.model),   # pop 取出 model（若无则用默认），同时从 kwargs 里移除避免重复传
                messages=self._messages,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    collected.append(delta)   # 收集到口袋里
                    if on_token:
                        on_token(delta)        # 回调通知外部
                    yield delta                # 把这段文字"流"给调用方
        except Exception as e:
            self._messages.pop()   # 调用失败回滚
            raise RuntimeError(
                f"调用 {self.model} 失败：{e}（请检查 API Key、账户余额与模型名）"
            ) from e
        else:
            # try 里没出错才会走到这里（else 分支）：把完整回答写入历史
            self._messages.append(
                {"role": "assistant", "content": "".join(collected)}
            )

    def __call__(self, question: str, **kwargs) -> str:
        """client("你好") 等价于 client.ask("你好")。

        __call__ 魔法方法的作用：让"实例"本身可以像函数一样被调用。
        即 bot("你好") 会自动转成 bot.ask("你好")，写起来更顺。
        """
        return self.ask(question, **kwargs)


# ---------- 模块级便捷函数 ----------

_default_client: Optional[QwenClient] = None   # 全局共享的客户端实例（懒加载）


def get_client(**kwargs) -> QwenClient:
    """获取全局共享的客户端实例（首次调用时创建，可通过 kwargs 覆盖默认配置）。

    好处：不用每次都用 QwenClient(...) 造新手机，全局复用同一部，省资源。
    """
    global _default_client   # 声明要修改的是"全局变量" _default_client
    if _default_client is None:
        _default_client = QwenClient(**kwargs)   # 第一次调用时才创建
    return _default_client


def qwen_chat(
    question: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """最简用法：qwen_chat("你好")；临时换模型：qwen_chat("x", model="qwen-turbo")"""
    if model:
        kwargs["model"] = model   # 把 model 塞进 kwargs，透传给底层
    return get_client().ask(question, system=system, **kwargs)


def qwen_stream(
    question: str,
    on_token: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> Iterator[str]:
    """流式用法：for t in qwen_stream("你好", on_token=print): ..."""
    return get_client().ask_stream(question, on_token=on_token, **kwargs)


# 只有"直接运行本文件"时才执行（被 import 时不执行）
if __name__ == "__main__":
    # 1. 一行调用
    # print("== 简单问答 ==")
    # print(qwen_chat("用一句话介绍你自己"))

    # 2. 多轮对话（自动记住上下文）
    print("\n== 多轮对话 ==")
    bot = QwenClient(system_prompt="你是一个助手")
    print("AI:", bot("今天天气如何"))
    print("AI:", bot("铁哥帅不帅"))
    print("AI:", bot("我上一个问题是什么"))

    s = 1
    # 3. 流式输出
    # print("\n== 流式输出 ==")
    # for _ in qwen_stream(
    #     "用三句话介绍量子计算", on_token=lambda t: print(t, end="", flush=True)
    # ):
    #     pass
    # print()
