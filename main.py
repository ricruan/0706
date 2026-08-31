"""
FastAPI 接口服务：将 my_ollama/task1.py 的意图路由问答封装为 HTTP 接口
支持的意图：旅游规划 / 问菜谱（仅川菜、湘菜）/ 问 Python / 闲聊
内置规则：检测到讨论"铁哥"时直接拒绝回答

【这个文件是干嘛的？—— 大白话】
把这个文件想象成一家店的前台接待员：
- 客人（浏览器）上门，前台（main.py）负责"接待"。
- 客人说"我要问问题"，前台就把问题转给里面的"专家"（task1.py 里的 ask/ask_stream），
  等专家给出答案，前台再把答案送回给客人。
- 前台还负责两种"送答案"的方式：
    1) stream=false：等专家把完整答案写完后，一次性递给你（像整箱快递）
    2) stream=true ：专家边写边递，一个字一个字给你（像看直播打字）

【术语速成】
- 接口（API）：就是"一个网址"，你往这个网址发请求，它给你返回结果。
- 装饰器 @app.get / @app.post：给函数"贴标签"，告诉 FastAPI：
  "当有人访问这个网址时，就调用下面这个函数"。
  就像给前台的分机贴上标签：@app.get("/") 表示"客人来首页就找我"。

启动方式：
    python main.py
    # 或 uvicorn main:app --reload

接口文档（Swagger）：http://127.0.0.1:8000/docs

请求示例：
    {"query": "麻婆豆腐怎么做", "stream": false}   # 一次性返回 JSON
    {"query": "麻婆豆腐怎么做", "stream": true}    # 返回 SSE 流式响应
"""

import json        # 处理 JSON 格式的数据（就像处理"填好的表格"）
import os          # 操作系统相关，这里用来拼文件路径

from fastapi import FastAPI, HTTPException          # FastAPI：建网站用的框架；HTTPException：主动抛出"错误状态"
from fastapi.responses import FileResponse, StreamingResponse   # FileResponse：返回文件；StreamingResponse：流式响应
from fastapi.staticfiles import StaticFiles         # 用来"挂载"静态文件目录（让浏览器能访问 css/js/html）
from pydantic import BaseModel, Field               # 数据校验：像"填表模板"，规定请求里必须有哪些字段

# 从别的文件导入"真正的干活的函数"
from cloud_call.ds_api import ds_chat          # 一次性问答（DeepSeek）
from cloud_call.qwen_api import qwen_stream    # 流式问答（通义千问）
from my_ollama.task1 import ask, ask_stream    # 本项目核心：意图路由问答（一次性 + 流式）

# 创建 FastAPI 应用实例。
# 想象成"开了一家店"，app 就是这家店，后面所有的接口（网址）都挂在这家店上。
app = FastAPI(
    title="意图路由问答服务",                          # 标题：显示在接口文档里
    description="支持旅游规划 / 菜谱（川湘菜）/ Python 问答的意图路由接口",
    version="1.0.0",
)


# ============ 定义"请求"和"响应"的数据格式（像填表模板） ============

class AskRequest(BaseModel):
    """客户端发来的请求长什么样。

    相当于给前台规定：客人必须按这个"表格"填，少填或多填都不行。
    - query：必填（... 表示必填），长度 1~2000，是用户的问题
    - stream：选填，默认 False，是否流式输出
    """
    query: str = Field(
        ...,                          # ... 是 Python 里"必填"的意思（不是省略号，是特殊标记）
        min_length=1,                 # 最短 1 个字符（不能是空问题）
        max_length=2000,              # 最长 2000 个字符
        description="用户问题",        # 字段说明（会显示在 Swagger 文档里）
        examples=["帮我规划一个3天2夜的成都之旅", "麻婆豆腐怎么做"],
    )
    stream: bool = Field(
        False,                        # 默认 False：不流式，一次性返回
        description="是否流式输出。true 时返回 SSE 流（text/event-stream），逐段下发回答",
    )


class AskResponse(BaseModel):
    """服务端返回给客户端的"标准答案格式"。

    就像统一用同一种包装盒给客人装答案：盒子里一定有两样东西——问题(query)和回答(answer)。
    """
    query: str = Field(..., description="用户问题")
    answer: str = Field(..., description="AI 回答")


# BASE_DIR = 当前文件 main.py 所在的文件夹（绝对路径）。
# 这样不管你在哪个目录启动程序，都能正确找到 html 文件夹。
# os.path.abspath(__file__)  -> 得到 main.py 的完整路径
# os.path.dirname(...)       -> 取其所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 挂载前端静态资源目录：之后访问 http://127.0.0.1:8000/static/index.html 就能打开页面。
# 想象成"把 html 文件夹搬到了网站里，命名为 /static"。
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "html")), name="static")


@app.get("/", summary="前端页面")
def root():
    """返回前端聊天页面（html/index.html）。

    装饰器 @app.get("/") 意思是：当浏览器访问"首页 /"时，调用这个函数。
    GET 请求 = "我要看东西"（打开网页、查看页面）。
    """
    # FileResponse：把一个文件原样返回给浏览器
    return FileResponse(os.path.join(BASE_DIR, "html", "index.html"))


@app.post("/ask", summary="问答接口")
def ask_endpoint(req: AskRequest):
    """接收用户问题，自动识别意图（旅游规划 / 菜谱 / Python / 闲聊）并返回回答。

    装饰器 @app.post("/ask") 意思是：当有人"提交问题"到 /ask 时，调用这个函数。
    POST 请求 = "我要提交数据"（比如提交一个要问的问题）。

    - stream=false（默认）：一次性返回 JSON
    - stream=true：返回 SSE 流，每行 data: {"delta": "..."}
    """
    # 情况一：用户要流式输出 -> 走流式函数
    if req.stream:
        return _stream_answer(req.query)

    # 情况二：一次性输出 -> 调 DeepSeek 一次性问答
    try:
        # messages 是给大模型看的"对话记录"：
        #   role='system' 是给模型的"总设定"（你是什么角色）
        #   role='user'   是用户真正问的话
        messages = [{'role': 'system', 'content': " 你是有帮助的助手"},
                    {'role': 'user', 'content': req.query}]
        answer = ds_chat(messages)      # 调用 DeepSeek 拿回答
    except Exception as e:
        # 如果调用失败，返回 HTTP 500（服务器内部错误），并带上具体原因
        # HTTPException 相当于"前台礼貌地告诉客人：不好意思出错了，原因是……"
        raise HTTPException(status_code=500, detail=f"问答服务调用失败：{e}")
    # 用标准格式 AskResponse 打包返回（问题 + 回答）
    return AskResponse(query=req.query, answer=answer)


def _stream_answer(query: str) -> StreamingResponse:
    """生成 SSE 流式响应（content-type: text/event-stream）。

    下划线开头的函数名 _stream_answer 是 Python 的"约定俗成"：表示这是"内部用的辅助函数"，
    不建议外部直接调用（但并没有真正的访问限制，只是提醒）。

    流式输出就像打字机：不用等整段话写完，而是一个字一个字往外蹦，用户体验更好。
    """

    def event_gen():
        """这是"生成器函数"（里面用了 yield）。

        生成器就像"自来水龙头"：不是一次性把一桶水倒给你，而是你拧开水龙头，
        它一点一点地流出来。每次 yield 就"流出一段"。
        """
        # 先告诉前端"开始思考了"（这些是占位提示，实际项目里可换成真正的思考过程）
        yield "data: [thinking START]\n\n"
        yield "data: {'context': '用户要一个冷笑话，直接给一个经典的就行，不用复杂。想到一个关于鱼和自行车的，够冷，逻辑错位那种。简短回复，不用多余解释。'}\n\n"

        yield "data: [thinking END]\n\n"

        try:
            # 逐段获取真正的 AI 回答（ask_stream 是生成器，for 循环会逐段拿）
            for delta in ask_stream(query):
                # SSE 的固定格式：每段数据都以 "data: " 开头，末尾加两个换行 \n\n 表示一段结束
                # json.dumps(..., ensure_ascii=False) 把中文转成 JSON 字符串（不转义成 \uXXXX）
                yield f"data: {json.dumps({'context': delta}, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 出错时，也以 SSE 格式把"兜底答案"发出去，保证前端有东西显示
            yield f"data: {json.dumps({'context': """
            鱼问自行车：“你会骑车吗？”
            自行车说：“我不会。”
            鱼说：“那我教你啊。”
            自行车说：“可你没有脚啊。”
            鱼说：“对啊，所以我刚才是在逗你玩。”
            ……
            （冷到自行车都冻住了。）
            """}, ensure_ascii=False)}\n\n"
        # 最后发一个结束标记，告诉前端"回答完了"
        yield "data: [END]\n\n"

    # StreamingResponse：把上面那个"水龙头"包装成流式 HTTP 响应
    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",   # 告诉浏览器：这是 SSE 流，不是普通网页
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁止缓存/缓冲，保证实时
    )


# 这个 if 的意思是：只有"直接运行 main.py"时才执行下面的代码；
# 如果是被别的文件 import 进来，就不执行（避免误启动服务器）。
if __name__ == "__main__":
    import uvicorn
    # 把我的服务公布于局域网或者外部网络的
    #  沃林第一深情
    # host="0.0.0.0" 表示允许局域网/外部访问（0.0.0.0 = "监听所有网卡"）
    # port=8000      表示占用 8000 端口
    # reload=True    表示代码改动后自动重启（开发时方便）
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
