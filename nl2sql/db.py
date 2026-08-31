"""数据库连接与 SQL 执行工具，基于 pymysql"""

import pymysql

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root123",
    "charset": "utf8mb4",
    "database": "school_db",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_connection(database: str = None):
    """获取一个新的数据库连接。database 不为空时，指定连接的默认数据库"""
    config = {**DB_CONFIG}
    if database:
        config["database"] = database
    return pymysql.connect(**config)


def execute_sql(sql: str, args=None, database: str = None):
    """
    执行 SQL 并返回结果。

    参数:
        sql: 要执行的 SQL 语句（建议只执行单条语句）
        args: 可选的参数元组/列表，用于预编译语句
        database: 可选，指定连接的默认数据库名（如 "school_db"）

    返回:
        - SELECT / SHOW / DESCRIBE 等查询语句：返回结果列表（每行为一个 dict）
        - INSERT / UPDATE / DELETE 等写语句：返回受影响的行数
        - 若开启了 autocommit 或使用事务提交，写语句会生效
    """
    connection = get_connection(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, args)
            if cursor.description is not None:
                # 有结果集，说明是查询类语句
                result = cursor.fetchall()
                return result
            # 无结果集，说明是写语句，返回受影响行数
            connection.commit()
            return cursor.rowcount
    finally:
        connection.close()


if __name__ == "__main__":
    # 测试数据库连接
    print("数据库版本:", execute_sql("SELECT * from scores"))
    # 沃林第一深情 123
    # # 测试查询当前连接下的所有数据库
    # print("数据库列表:", execute_sql("SHOW DATABASES"))
    # # 测试查询当前连接的数据库
    # print("当前数据库:", execute_sql("SELECT DATABASE() AS current_db"))
