
import socket
import json
import mysql.connector
from mysql.connector import Error
import os
import base64
from datetime import datetime

# --- 1. DB 연결 정보 설정 ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '0000',
    'database': 'patroldb'
}

# --- 2. 이벤트 이름 매핑 ---
# class_name 과 실제 저장될 디렉터리명을 매핑
EVENT_DIR_MAP = {
    "Smoke_Event": "Smoke_Event",
    "Fire_Event": "Fire_Event",
    "Crime_Event": "Crime_Event",
    "Trash_Event": "Trash_Event",
    "Faint_Event": "Faint_Event",
    "Missing_Event": "Missing_Event"
}

BASE_VIDEO_DIR = "Event_Video"  # 최상위 디렉토리


def save_video_frame(data_dict):
    """
    수신한 video_frame 데이터를 로컬 파일로 저장합니다.
    디렉토리 구조: Event_Video/Patrol_Car_X/EventType/YYYY-MM-DD_HH_MM_SS.mp4
    """
    try:
        patrol_car = data_dict.get("patrol_car", "Unknown_Car")
        class_name = data_dict.get("class_name", "Unknown_Event")
        video_frame_b64 = data_dict.get("video_frame")

        if not video_frame_b64:
            print("[영상 저장] video_frame 데이터 없음. 저장 스킵.")
            return None

        # 이벤트 디렉토리명 매핑
        event_dir = EVENT_DIR_MAP.get(class_name, class_name)

        # 디렉토리 경로 생성
        save_dir = os.path.join(BASE_VIDEO_DIR, patrol_car, event_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 파일명 (타임스탬프 기반)
        timestamp = data_dict.get("timestamp")
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_name = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d_%H_%M_%S") + ".mp4"
        file_path = os.path.join(save_dir, file_name)

        # base64 디코딩 후 파일로 저장
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(video_frame_b64))

        print(f"[영상 저장] 동영상 프레임 저장 완료: {file_path}")
        return file_path

    except Exception as e:
        print(f"[영상 저장] 오류 발생: {e}")
        return None


def process_and_store_data(data_dict):
    """
    수신한 데이터를 DB에 저장 (situation + media).
    """
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        connection.start_transaction()

        # 1. situation 테이블 저장
        situation_query = """
            INSERT INTO situation (timestamp, patrol_car, class_name, raw_result)
            VALUES (%s, %s, %s, %s)
        """
        raw_result_str = json.dumps(data_dict.get('raw_result'))
        situation_args = (
            data_dict.get('timestamp'),
            data_dict.get('patrol_car'),
            data_dict.get('class_name'),
            raw_result_str
        )
        cursor.execute(situation_query, situation_args)
        class_id = cursor.lastrowid
        print(f"[DB] situation 저장 완료. class_id: {class_id}")

        # 2. media 테이블 저장
        media_info = data_dict.get('media_info', {})
        
        # video_frame 이 있으면 실제 저장 후 경로 반영
        saved_video_path = save_video_frame(data_dict)
        if saved_video_path:
            media_info["rel_path"] = saved_video_path

        media_query = """
            INSERT INTO media (class_id, rel_path, validation, mime_type)
            VALUES (%s, %s, %s, %s)
        """
        media_args = (
            class_id,
            media_info.get('rel_path'),
            media_info.get('validation'),
            media_info.get('mime_type')
        )
        cursor.execute(media_query, media_args)
        print("[DB] media 저장 완료.")

        connection.commit()
        return True

    except Error as e:
        print(f"[DB] 오류 발생: {e}")
        if connection:
            connection.rollback()
            print("[DB] 트랜잭션 롤백됨.")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def start_server(host='0.0.0.0', port=9999):
    """TCP 서버 실행"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[서버] {host}:{port} 에서 연결 대기 중...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[서버] 클라이언트 연결됨: {addr}")

                full_data = b''
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    full_data += chunk

                print("\n[TCP] === Received JSON full_data ===")
                print("Raw:", full_data.decode())
                # print("Type:", type(message))
                
                if not full_data:
                    print("[서버] 데이터 없음.")
                    continue

                try:
                    message = json.loads(full_data.decode('utf-8'))
                    if process_and_store_data(message):
                        response = {"status": "success", "message": "데이터 저장 완료"}
                    else:
                        response = {"status": "error", "message": "DB 저장 실패"}

                except json.JSONDecodeError:
                    print("[서버] JSON 오류")
                    response = {"status": "error", "message": "Invalid JSON"}
                except Exception as e:
                    print(f"[서버] 처리 오류: {e}")
                    response = {"status": "error", "message": f"Unexpected error: {e}"}

                conn.sendall(json.dumps(response).encode('utf-8'))


if __name__ == "__main__":
    start_server()

