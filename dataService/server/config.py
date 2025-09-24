 # DB 접속 정보, 미디어 폴더 경로 등 설정 파일
# server/config.py

# --- 데이터베이스 설정 ---
DB_CONFIG = {
    'host': 'localhost',      # DB 서버 주소 
    'user': 'root',  # DB 사용자 이름
    'password': '0000',# DB 비밀번호
    'database': 'patroldb'   # 사용할 데이터베이스 이름
}

# --- 미디어 스토리지 설정 ---
# server 폴더 기준 상대 경로 또는 절대 경로를 지정
MEDIA_STORAGE_ROOT = '../Event_Video/'