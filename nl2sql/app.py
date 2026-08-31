"""
nl2sql/app.py —— 智能问数（自然语言转 SQL）FastAPI 接口服务

流程：自然语言问题 -> 大模型生成 SQL -> 执行 SQL -> 大模型润色结果
返回：{sql, columns, rows, affected_rows, answer}
- sql: 生成的 SQL 语句
- columns: 查询结果的列名（写语句为空列表）
- rows: 查询结果数据（写语句为空列表）
- affected_rows: 写语句影响的行数（查询语句为 None）
- answer: 大模型润色的自然语言回答文本

启动方式：
    python nl2sql/app.py
    # 或 uvicorn nl2sql.app:app --reload

接口文档（Swagger）：http://127.0.0.1:8001/docs
前端页面：http://127.0.0.1:8001/
"""

import os
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cloud_call.ds_api import ds_chat
from nl2sql.db import execute_sql

# 目标数据库名（执行 SQL 时连接的默认库，可按实际修改）
DATABASE = "school_db"

# 表结构信息（用于提示大模型生成 SQL，参考 nl2sql_01.py）
TABLE_INFO = {
    'student': """
    CREATE TABLE `students` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `class` varchar(50) NOT NULL,
  `age` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    """,
    'score': """
    -- school_db.scores 定义

CREATE TABLE `scores` (
  `id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `subject` varchar(50) NOT NULL,
  `score` decimal(5,2) NOT NULL,
  `exam_date` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    """,
}

# 当前文件所在目录（不管在哪启动都能正确找到 html 文件夹）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(os.path.dirname(BASE_DIR), "html")

app = FastAPI(
    title="智能问数服务",
    description="自然语言转 SQL 并返回查询结果与文本润色结果的接口",
    version="1.0.0",
)

# 允许跨域访问，方便前端页面单独部署时调用本接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态资源目录（html 文件夹）
app.mount("/static", StaticFiles(directory=HTML_DIR), name="static")


class NL2SQLRequest(BaseModel):
    """请求体：用户提出的自然语言问题"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户自然语言问题",
        examples=["查询所有学生的姓名和年龄", "删除名叫铁哥的学生"],
    )


def clean_sql(sql: str) -> str:
    """清洗大模型生成的 SQL：去掉 ```sql ``` 代码块标记和多余空白"""
    sql = sql.strip()
    # 去掉 Markdown 代码块包裹（如 ```sql ... ```）
    sql = re.sub(r"^```[a-zA-Z]*\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";")


def format_result(result) -> tuple:
    """
    把 execute_sql 的返回值统一成 (columns, rows, affected_rows)。
    - 查询类：返回 list[dict]，转为列名 + 行数据
    - 写类：返回 int 影响行数
    """
    if isinstance(result, list):
        columns = list(result[0].keys()) if result else []
        return columns, result, None
    return [], [], result


def nl2sql(query: str) -> dict:
    """智能问数核心逻辑：生成 SQL -> 执行 -> 润色，返回结构化结果"""
    prompt = f"""
你是一个sql大师，精通mysql，可以根据自然语言查询生成对应的sql语句。
请根据自然语言查询生成对应的sql语句。
以下是我的schema信息：
{TABLE_INFO}

自然语言查询:
{query}

# 规则
禁止生成SQL之外的任何解释性文本
禁止生成Markdown格式的 ```sql ``` 包裹sql代码
"""

    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': query},
    ]

    # 第一步：生成 SQL
    sql = clean_sql(ds_chat(messages=messages))

    # 第二步：执行 SQL
    result = execute_sql(sql, database=DATABASE)
    columns, rows, affected_rows = format_result(result)

    # 第三步：让 AI 对执行结果进行润色
    if affected_rows is not None:
        # 写语句：告诉模型影响了多少行
        polish_content = (
            f"用户问题：{query}\n"
            f"执行的 SQL：{sql}\n"
            f"受影响行数：{affected_rows}"
        )
    elif rows:
        polish_content = (
            f"用户问题：{query}\n"
            f"执行的 SQL：{sql}\n"
            f"查询结果：{rows}"
        )
    else:
        polish_content = (
            f"用户问题：{query}\n"
            f"执行的 SQL：{sql}\n"
            f"查询结果为空"
        )

    answer = ds_chat(messages=[
        {'role': 'system', 'content': '请用自然语言对用户提供的SQL执行结果进行总结和分析，语言简洁友好。'},
        {'role': 'user', 'content': polish_content},
    ])

    return {
        "query": query,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "affected_rows": affected_rows,
        "answer": answer,
    }


@app.get("/", summary="智能问数前端页面")
def index():
    """返回前端页面 html/nl2sql.html"""
    return FileResponse(os.path.join(HTML_DIR, "nl2sql.html"))


@app.post("/api/nl2sql", summary="智能问数接口")
def nl2sql_endpoint(req: NL2SQLRequest):
    """接收自然语言问题，返回 {sql, columns, rows, affected_rows, answer}"""
    try:
        return nl2sql(req.query)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智能问数服务调用失败：{e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nl2sql.app:app", host="0.0.0.0", port=8001, reload=True)
