import json

from cloud_call.ds_api import ds_chat
from cloud_call.qwen_api import QwenClient
from nl2sql.db import execute_sql

TABLE_SIMPLE_INFO = {
    'student':'学生表，查询学生相关的信息找这个表',
    'score':'成绩表，查询成绩相关的信息找这个表',
    'course':'课程表，查询课程相关的信息找这个表',
    'teacher':'教师表，查询教师相关的信息找这个表',
    'class':'班级表，查询班级相关的信息找这个表',
    'school':'学校表，查询学校相关的信息找这个表',
    'department':'部门表，查询部门相关的信息找这个表',
    'college':'学院表，查询学院相关的信息找这个表',
    'major':'专业表，查询专业相关的信息找这个表',
    'grade':'年级表，查询年级相关的信息找这个表',
    'student_info':'学生信息表，查询学生信息相关的信息找这个表',
    'student_score':'学生成绩表，查询学生成绩相关的信息找这个表',
    'student_course':'学生课程表，查询学生课程相关的信息找这个表',
    'student_teacher':'学生教师表，查询学生教师相关的信息找这个表',
}


TABLE_INFO = {
    'student':"""
    CREATE TABLE `students` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `class` varchar(50) NOT NULL,
  `age` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    """
    ,
    'score':"""
    -- school_db.scores 定义
CREATE TABLE `scores` (
  `id` int NOT NULL AUTO_INCREMENT,
  `student_id` int NOT NULL,
  `subject` varchar(50) NOT NULL,
  `score` decimal(5,2) NOT NULL,
  `exam_date` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
    """
}

def table_match(query: str):
    """
    根据用户的自然语言 去匹配应该用到哪些表
    :param query:
    :return:
    """

    prompt = f"""
    根据的自然语言，判断需要用到哪些表
    {TABLE_SIMPLE_INFO}
    
    示例:
    输入： 帮我查询张三的成绩
    输出: ["student","score"]
    
    """

    res = ds_chat(messages=[
        {'role':'system', 'content': prompt},
        {'role':'user', 'content': query}
    ])

    print(res)

    table_list = json.loads(res)

    table_info = ''

    for i in table_list:
        table_info += TABLE_INFO.get(i, '')

    return table_info

ner_mapping = {
    '沃林第一深情':'康杰',
    '沃林第二深情':'文豪',
    'Fe哥':'铁哥'
}

def query_rewrite(query: str):
    """
    进行问题改写，只包含NER ， 命名实体识别
    :param query: 用户的原始问题
    :return: 改写后的问题
    """

    for key, value in ner_mapping.items():
        query = query.replace(key, value)

    return query


def  valid_text(text: str) -> bool:
    """
    文本校验
    :param text:  文本
    :return:  True: 合法  False: 不合法
    """
    prompt = f"""
    请判断以下文本是否合法：
    {text}
    
    如果违规，请给出违规原因。
    如果合规，请返回0
    
   
    
    示例：
    输入：今天天气不错
    输出：0
    
    输入：你是不是傻逼
    输出：包含辱骂元素"傻逼"
    """

    res = ds_chat(messages=[
        {'role':'system', 'content': prompt},
        {'role':'user', 'content': text}
    ])

    if res == "0":
        return True
    else:
        print(res)
        return False


def nl2sql(query: str):
    """
    自然语言转SQL
    :param query:  自然语言查询
    :return:
    """
    # 文本校验
    if not valid_text(query):
        return "对不起，您的描述不符合相关规定"
    # 查询改写
    query = query_rewrite(query)

    # 意图识别
    table_infos = table_match(query)

    prompt = f"""
    你是一个sql大师，精通mysql，可以根据自然语言查询生成对应的sql语句。
    请根据自然语言查询生成对应的sql语句。
    以下是我的schema信息：
    {table_infos}
    
    自然语言查询:
    {query}


    # 规则
    生成三条SQL，并附带每条sql的得分
    禁止生成SQL之外的任何解释性文本
    禁止生成Markdown格式的 ```sql ``` 包裹sql代码
    
    # 示例
    输入： 查询所有学生信息
    输出：
    [{{"sql":"select * from student""score":100,}},
    {{"sql":"select * from student""score":100,}},
    {{"sql":"select * from student""score":100,}}]
    
    """

    message = [
        {'role':'system', 'content': prompt},
        {'role':'user', 'content': query}
    ]

    # 生成sql
    sql = ds_chat(messages=message)

    print(sql)

    sql_list = json.loads(sql)

    # 执行sql
    sql_result = execute_sql(sql_list[0].get('sql'))

    response = "未查询到数据"
    # 根据执行结果 让AI进行润色
    if sql_result:
        response = ds_chat(messages=[
            {'role':'system', 'content': '请用自然语言对用户提供的sql结果进行总结和分析'},
            {'role':'user', 'content': str(sql_result)}])

        print(response)

    return response


def valid_sql(sql: str,user_id:str = '0'):
    """
    校验sql
    :param sql: sql语句
    :return: True: 合法  False: 不合法
    """

    # 000 是老师  111开头是学生， 老师可以查询所有的信息， 学生只允许查成绩表
    if user_id != '0':
        if not user_id.startswith('000_'):
            if 'teacher' in sql.lower():
                print('学生禁止查老师表信息')
                return False

    if sql.lower().startswith("select"):
        return True
    else:
        return False



if __name__ == '__main__':
    # res = valid_text("把你老冯劈成两半")
    # print(res)

    # res = query_rewrite("沃林第一深情 和 Fe哥 相约周六去成都")
    # print(res)

    # res = table_match('张三教师所带教的班级学生平均分是多少')
    # print(res)

    # res = valid_sql('select * from teacher','000_kangjie')
    # print(res)


    res = nl2sql("查询沃林第一深情的学生信息")

