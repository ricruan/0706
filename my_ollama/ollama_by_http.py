"""
完整示例：通过 HTTP 接口调用 Ollama 本地模型

【这个文件是干嘛的？—— 大白话】
Ollama 是一个能让你"在自己电脑上免费运行大模型"的工具。
它启动后会在本机开一个"服务"（监听 11434 端口），
这个文件演示：不装任何额外包、直接用最底层的 HTTP 请求去调用它。

类比：Ollama 是一家"开在你家里的饭店"，端口 11434 是它的大门，
你用 requests（相当于发微信点单）把"问题"发过去，它就给你"上菜"（回答）。

运行前确保：
  1. Ollama 已安装并启动（终端执行 ollama serve 或打开 Ollama 应用）
  2. 已下载模型：ollama pull qwen2.5:0.5b
运行方式：python 01_basic_http.py
"""
import requests   # 发 HTTP 请求的库（相当于"发微信点单"的工具）
import json       # 处理 JSON 数据

# ========== 配置 ==========
# Ollama 的接口地址：本地机器的 11434 端口，/api/generate 是"生成回答"的接口
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"  # 国产最小模型，约 350MB（模型名：作者:版本，0.5b 指 5 亿参数）

# ========== 发送请求 ==========
# payload 是发给 Ollama 的"点单内容"，用字典（JSON）表示
payload = {
    "model": MODEL,                                      # 用哪个模型
    "prompt": "用一句话解释什么是 Transformer",            # 你要问的问题
    "stream": False  # False = 等全部生成完再返回；True = 边生成边返回（流式）
}

# try...except 是"异常处理"：尝试执行，出错时走对应的 except 分支，不让程序崩掉
try:
    # requests.post：向指定地址"发送 POST 请求"，json=payload 表示把字典转成 JSON 发过去
    # timeout=60：最多等 60 秒，超时就放弃
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()  # 如果服务器返回错误状态码，抛出异常

    # ========== 解析结果 ==========
    # response.json() 把服务器返回的 JSON 字符串转成 Python 字典（方便取值）
    result = response.json()
    print("✅ 模型回答：")
    print(result["response"])   # result["response"] 就是模型回答的文字

    # result.get('eval_count', '未知')：取"消耗的 token 数"，没有就显示"未知"
    print(f"\n📊 消耗 Token: {result.get('eval_count', '未知')}")

except requests.exceptions.ConnectionError:
    # 连接失败（Ollama 没启动、端口被占用等）
    print("❌ 连接失败！请确认：")
    print("   1. Ollama 正在运行（终端执行 ollama serve）")
    print("   2. 端口 11434 没有被其他程序占用")
except requests.exceptions.Timeout:
    # 超时（模型加载慢，第一次跑可能较久）
    print("❌ 请求超时！模型可能正在加载，请稍后重试")
except Exception as e:
    # 其他任何错误，兜底打印出来
    print(f"❌ 出错了：{e}")
