# server_config.py
import pymysql

# DB 접속 정보
DB_CONFIG = {
    'host': 'localhost',      
    'user': 'root',  
    'password': '0000',
    'database': 'patroldb',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
}

# 서버 포트 설정
PATROL_CAR_SERVER_PORT = 2301  # situationDetector (순찰차) -> 서버
GUI_SERVER_PORT = 3401         # monitoringGUI (관제 GUI) -> 서버

# 미디어 저장소 루트 경로
# DB의 rel_path가 시작되는 지점의 상위 폴더입니다.
# 예: rel_path가 './Event_Video/...' 라면, 이 경로는 '.'에 해당합니다.
MEDIA_STORAGE_ROOT = '/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/'