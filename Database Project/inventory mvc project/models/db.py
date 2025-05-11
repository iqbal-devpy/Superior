import os
import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv

load_dotenv(dotenv_path='config/.env')

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

class Database:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cur = self.conn.cursor(dictionary=True)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise RuntimeError('Authentication error')
            else:
                raise

    def execute(self, query, params=None, fetch=False):
        try:
            self.cur.execute(query, params or [])
            if fetch:
                return self.cur.fetchall()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def call_proc(self, proc_name, params):
        self.cur.callproc(proc_name, params)
        results = []
        for result in self.cur.stored_results():
            results.extend(result.fetchall())
        self.conn.commit()
        return results

    def close(self):
        self.cur.close()
        self.conn.close()