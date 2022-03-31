import sqlite3 as sql
import os
class SqlClient():

    def __init__(self):
        db_path = os.environ.get("DB_PATH",  '../Dataset/basketball.sqlite')
        self.conn = sql.connect(db_path)


    def custom_sql_call(self, sql):
        return self.conn.execute(sql)
