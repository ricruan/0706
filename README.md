# 意图路由问答助手（AI 问答服务）

> 一个「能自动识别你问的是什么，再找对应大模型来回答」的网页聊天机器人。

---

## 一、这个项目是干嘛的？（一句话版）

你在网页里打字提问，后台先判断你的问题属于哪一类（**旅游规划 / 问菜谱 / 问 Python / 闲聊**），
然后交给合适的大模型（DeepSeek、通义千问）生成回答，最后显示在网页上。

---

## 二、用大白话理解整个项目（比喻）

把这个项目想象成一家「**有多个专家的智能客服公司**」：

| 文件/文件夹 | 比喻 | 它到底做什么 |
| :--- | :--- | :--- |
| html/index.html | 客服窗口（前台） | 你在这里打字、看回答，是漂亮的前端聊天页面 |
| main.py | 前台接待员 | 接收你的消息，决定用「一次性回答」还是「一个字一个字蹦出来（流式）」，再把专家的回答送回去 |
| my_ollama/task1.py | 分诊台（路由） | 判断你的问题该找哪位专家；还负责拦截「聊铁哥」这种话题 |
| cloud_call/ds_api.py | 专家 A：DeepSeek | 真正调用 DeepSeek 大模型，拿到回答 |
| cloud_call/qwen_api.py | 专家 B：通义千问 | 真正调用通义千问大模型，支持多轮对话、流式输出 |
| my_ollama/* | 自学教材 | 教你怎么用「自己电脑上免费的本地模型 Ollama」 |
| task/task2.py | 娱乐小实验 | 让两个 AI（贴吧老哥 vs 小学生）互相辩论「先有鸡还是先有蛋」 |
| .env | 保险柜 | 存放各家大模型的 API Key（密钥），代码从这里读，绝不写死在代码里 |

---

## 三、项目结构

```
0706/
├── main.py                      # 后端入口：FastAPI 服务，把问答功能变成 HTTP 接口
├── .env                         # 密钥配置（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）
├── README.md                    # 本文件
├── html/
│   └── index.html               # 前端聊天页面（界面 + 调后端接口的 JS 逻辑）
├── cloud_call/                  # 调用「云端大模型」的封装
│   ├── ds_api.py                # DeepSeek 调用（一次性 + 流式）
│   └── qwen_api.py              # 通义千问调用（类封装 + 多轮 + 流式）
├── my_ollama/                   # 本地模型 Ollama 的学习示例
│   ├── ollama_by_http.py        # 方式1：用 HTTP 接口调用本地模型
│   ├── ollama_by_sdk.py         # 方式2：用 Python 官方包调用本地模型
│   └── task1.py                 # 核心：意图路由问答（本项目的大脑）
└── task/
    └── task2.py                 # 娱乐：两个 AI 互相对话
```

---

## 四、每个文件的作用详解

### 1. main.py —— 后端入口（把功能变成网站接口）

- 用 FastAPI 起了一个网站服务，跑在 http://127.0.0.1:8000 。
- 提供两个接口：
  - GET / ：返回前端页面 html/index.html 。
  - POST /ask ：接收你的问题 {query, stream}，返回回答。
- stream=false：一次性把完整回答打包成 JSON 给你（像快递整箱送到）。
- stream=true：用 SSE 流式，让回答像打字机一样一个字一个字出来（像看直播逐句刷新）。

**大白话例子**：你点「发送」，前端就把问题 POST 到 /ask，main.py 调 ds_chat 或 ask_stream 拿到回答，再送回前端显示。

### 2. cloud_call/ds_api.py —— 调用 DeepSeek 大模型

- 从 .env 读 DEEPSEEK_API_KEY，用 OpenAI 兼容协议连接 https://api.deepseek.com 。
- ds_chat(messages)：一次性问答，返回完整回答文本。
- ds_chat_stream(messages)：流式问答，返回生成器，逐段产出文字（省得用户干等）。

### 3. cloud_call/qwen_api.py —— 调用通义千问大模型

- 封装了一个 QwenClient 类：同一个实例会自动记住上下文，实现多轮对话。
- 提供模块级快捷函数：
  - qwen_chat("你好")：一行搞定问答。
  - qwen_stream("你好")：流式问答。
- 底层也是 OpenAI 兼容协议，所以换 DeepSeek / Ollama 只需改 base_url 和模型名。

### 4. html/index.html —— 前端聊天页面

- 一个漂亮的深色渐变聊天界面：消息气泡、输入框、发送按钮、「流式输出」开关、「清空对话」按钮。
- 内置 JS 逻辑：把问题发给后端 /ask，收到回答后逐字显示在气泡里。

### 5. my_ollama/task1.py —— 本项目「大脑」（意图路由问答）

这是核心逻辑，做四件事：

1. 识别意图：让 DeepSeek 判断你的问题属于「旅游规划 / 菜谱 / Python / 讨论铁哥」中的哪一类（输出 JSON，还带得分）。
2. 选最高分意图：从多个候选里挑得分最高的。
3. 按意图分发：
   - 旅游 → 用旅游规划的提示词
   - 菜谱 → 先判断菜系，只有川菜、湘菜才回答，其他菜系婉拒
   - Python → 用 Python 问答提示词
   - 讨论铁哥 → 直接拒绝（内置规则：不许聊铁哥）
   - 其他 → 猫娘闲聊兜底
4. 提供 ask()（一次性）和 ask_stream()（流式）两个对外函数。

### 6. my_ollama/ollama_by_http.py —— 本地模型教程（HTTP 方式）

- 用 requests 直接向本机 Ollama 的 http://localhost:11434/api/generate 发请求。
- 教你「不需要联网、不需要花钱」用自己电脑上的 qwen2.5:0.5b 模型。
- 演示了如何优雅地处理「连不上」「超时」等错误。

### 7. my_ollama/ollama_by_sdk.py —— 本地模型教程（官方 SDK 方式）

- 用 ollama 官方 Python 包调用本地模型，比手写 HTTP 更省事。
- 演示了「简单问答」和「多轮对话（手动维护历史）」两种用法。
- 重点提醒：每次调用都是「无状态」的，对话历史要自己维护。

### 8. task/task2.py —— 娱乐实验：两个 AI 辩论

- 创建一个「贴吧老哥」AI（基于 DeepSeek，观点：先有蛋）和一个「小学生」AI（基于通义千问，观点：先有鸡）。
- 让它们来回对话 20 轮，互相抬杠，纯属娱乐。

### 9. .env —— 密钥文件

- 存放 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY。
- ⚠️ 这个文件是私密的，绝不能提交到 Git 或发给别人（详见下方安全提醒）。

---

## 五、如何运行

### 前提：安装依赖

```bash
pip install fastapi uvicorn pydantic openai python-dotenv requests ollama
```

### 1. 启动后端

```bash
python main.py
# 或者
uvicorn main:app --reload
```

启动后：
- 网页：http://127.0.0.1:8000/
- 接口文档（Swagger，可在线测试）：http://127.0.0.1:8000/docs

### 2. 打开前端页面

浏览器访问 http://127.0.0.1:8000/ ，在输入框里提问即可。

**可以试的例子**：
- 帮我规划一个 3 天 2 夜的成都之旅
- 麻婆豆腐怎么做
- Python 浅拷贝的底层原理
- 给我讲个冷笑话

### 3.（可选）运行本地 Ollama 教程

```bash
# 先启动 Ollama 并下载模型
ollama pull qwen2.5:0.5b

# 再运行示例
python my_ollama/ollama_by_http.py
python my_ollama/ollama_by_sdk.py
```

---

## 六、新手术语表（几分钟看懂关键词）

| 术语 | 大白话解释 |
| :--- | :--- |
| API | 别人写好的「服务窗口」，你按规矩发请求，它按规矩给你结果。比如 /ask 就是一个 API |
| API Key（密钥） | 访问大模型的「门禁卡 / 密码」，证明你有权使用、用来计费 |
| LLM / 大模型 | 像 ChatGPT 这种能对话、写代码、做规划的大型 AI 模型 |
| 意图识别 / 路由 | 先判断「用户到底想问啥」，再决定交给哪个专家处理 |
| 提示词（Prompt） | 你给 AI 的「任务说明书」，告诉它该怎么回答（比如「你是一个旅游规划助手」） |
| 流式输出（stream） | 回答不是一次性蹦出来，而是一个字一个字地实时出现，体验更好 |
| SSE | 一种「服务器持续往客户端推数据」的技术，用来实现流式输出 |
| 多轮对话 / 上下文 | AI 记住你们之前聊了什么，才能接得上你的下一句（否则每次都像失忆） |
| JSON | 一种「结构化数据」的写法，像填表格一样 {"名字": "小明", "分数": 90} |
| 生成器（yield） | Python 里一种「边算边给」的写法，算一点返回一点，不用等全部算完 |

---

## 七、安全提醒 ⚠️

- .env 里的 API Key 是花钱买服务的凭证，泄露了别人会盗刷你的额度。
- 建议：立刻去 DeepSeek / 阿里云百炼后台把这两个 Key 重置（重新生成），因为本项目可能已被提交或分享过。
- 在项目根目录加一个 .gitignore，里面写一行 .env，防止以后误提交。

---

## 八、文件的调用关系（谁用了谁）

```
html/index.html  ──POST /ask──▶  main.py
                                  │
                ┌─────────────────┼──────────────────┐
                ▼                 ▼                  ▼
      task1.ask / ask_stream   ds_api.ds_chat    qwen_api.qwen_stream
                │                 │                  │
                └─────▶ ds_api.ds_chat（识别意图）    │
                                                     ▼
                                            QwenClient 类（多轮对话）

task/task2.py ──▶ ds_api.ds_chat + qwen_api.QwenClient（两个 AI 辩论）
```

> 一句话总结：**前端问 → main.py 收 → task1 分诊 → 调 DeepSeek/通义千问 → 答案送回前端显示**。
