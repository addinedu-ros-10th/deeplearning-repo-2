# MySQL 데이터베이스 연결 및 쿼리 실행 파일
# server/db_connector.py

import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG

def connect_db():
    """MySQL 데이터베이스에 연결하고 커넥션 객체를 반환합니다."""
    try:
        # 설정 정보를 사용하여 DB에 연결
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        # 연결 중 에러 발생 시 출력
        print(f"DB 연결 중 에러 발생: {e}")
        return None

# --- 이 파일이 직접 실행될 때만 아래 코드가 동작 (연결 테스트용) ---
if __name__ == '__main__':
    conn = connect_db()
    if conn and conn.is_connected():
        print("DB 연결 성공!")
        conn.close() # 테스트 후에는 반드시 연결을 닫아줍니다.
        print("DB 연결이 닫혔습니다.")
    else:
        print("DB 연결 실패.")