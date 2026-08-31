"""
完整示例：使用 ollama Python 包调用本地模型

【这个文件是干嘛的？—— 大白话】
上一个文件（ollama_by_http.py）是"手写 HTTP 请求"调用 Ollama；
这个文件演示更省事的方式——用官方提供的 ollama 包（SDK），
不用自己拼 URL、不用管 HTTP 细节，直接调用现成函数即可。

类比：HTTP 方式 = 自己发微信打字点单；
      SDK 方式   = 下载官方 App，点几下按钮就下单，更省心。

安装依赖：pip install ollama
运行前确保：
  1. Ollama 已安装并启动
  2. 已下载模型：ollama pull qwen2.5:0.5b
运行方式：python 02_ollama_chat.py
"""
import ollama   # Ollama 官方 Python 包

MODEL = "qwen2.5:0.5b"

# ========== 方式一：简单问答（单次调用） ==========
print("=== 简单问答 ===")
# ollama.generate：一问一答，问完拿到完整答案
response = ollama.generate(
    model=MODEL,
    prompt="用一句话解释什么是 Token"
)
print(response["response"])   # response 是字典，取 "response" 键得到回答文字

# ========== 方式二：多轮对话（带角色设定） ==========
print("\n=== 多轮对话 ===")

# 第一轮
# messages 是"对话记录"列表，每条是一轮发言：
#   role='system'  → 给模型的"总设定"（它是什么角色、怎么说话）
#   role='user'    → 用户说的话
#   role='assistant' → AI 之前的回答（用于让模型"记得上下文"）
messages = [
    {"role": "system", "content": "你是一个耐心的编程老师，用简单的话解释概念，每个回答不超过 3 句话"},
    {"role": "user", "content": "什么是 Token？"},
]

# ollama.chat：多轮对话，把整段历史一起发给模型
response = ollama.chat(model=MODEL, messages=messages)
answer = response["message"]["content"]   # chat 的返回结构里，回答在 message.content
print("AI:", answer)

# 第二轮（带上之前的对话历史，AI 才知道"上下文"）
# 关键点：把上一轮 AI 的回答也塞进历史，再追加新问题，
# 这样模型才知道"你接着上一句在问什么"。
messages.append({"role": "assistant", "content": answer})  # 把 AI 的回答加入历史
messages.append({"role": "user", "content": "那上下文窗口又是什么？"})

response = ollama.chat(model=MODEL, messages=messages)
print("AI:", response["message"]["content"])

# 💡 注意：每次调用都是"无状态"的——对话历史需要你自己维护
# 这就是术语手册中 Context Engineering（上下文工程）要解决的问题
# 【大白话】"无状态"= 模型本身不记事，每问一次都像第一次见面；
# 想让它"记得"，就得像上面这样，每次都把之前的对话内容重新发给它。
