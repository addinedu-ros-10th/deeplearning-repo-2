
import socket
import json
import pymysql
import os
import datetime
import base64                      # base64 모듈 추가
import time

# DB 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "0000",
    "database": "patroldb",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

# 기본 저장 디렉토리
MEDIA_BASE = "./Event_Video"
os.makedirs(MEDIA_BASE, exist_ok=True)


def save_to_db(detection_data, video_path=None, mime_type=None):
    """
    situation 테이블과 media 테이블에 데이터 저장
    """
    conn = pymysql.connect(**DB_CONFIG)

    # "detection" 딕셔너리가 비어있는지 확인 (원본 코드의 안정성 로직 유지)
    if not detection_data.get("event_video_meta"):
        print("경고: 'detection' 데이터가 비어있어 DB 저장을 건너뜁니다.")
        return

    try:
        with conn.cursor() as cursor:
            # situation 테이블에 저장
            sql_situation = """
                INSERT INTO situation (timestamp, patrol_number, class_name, raw_result)
                VALUES (%s, %s, %s, %s)
            """
            ts = detection_data.get("timestamp")
            patrol_number_db = f"Patrol_number_{detection_data.get('patrol_number', 0)}"

            # class_name: detection에서 첫 번째 항목 이름
            detection_key = next(iter(detection_data["event_video_meta"]))
            first_class_name = detection_data["event_video_meta"][detection_key][0].get("class_name", "Unknown")

            cursor.execute(sql_situation, (ts, patrol_number_db, first_class_name, json.dumps(detection_data)))
            class_id = cursor.lastrowid

            # media 테이블에 저장 (영상이 있는 경우)
            if video_path:
                sql_media = """
                    INSERT INTO media (class_id, rel_path, validation, mime_type)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql_media, (class_id, video_path, True, mime_type))

        conn.commit()
    finally:
        conn.close()


def handle_client(conn, addr):
    print(f"[INFO] Connected by {addr}")
    video_file = None
    detection_data = None
    video_path = None
    event_dir = None
    last_log_time = time.time()

    # 초기에 기본 디렉토리 설정
    base_event_dir = os.path.join(MEDIA_BASE, "Patrol_Car_Unknown", "Unknown_Event")
    os.makedirs(base_event_dir, exist_ok=True)
    event_dir = base_event_dir

    buffer = b""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            
            buffer += data
            
            while True:
                try:
                    json_str = buffer.decode('utf-8')
                    json_obj, idx = json.JSONDecoder().raw_decode(json_str)
                    buffer = buffer[idx:]

                    # 1. 메타데이터 처리
                    if "event_video_meta" in json_obj:
                        detection_data = json_obj
                        patrol_number = detection_data.get('patrol_number', 'Unknown')
                        detection_dict = detection_data.get("event_video_meta", {})
                        
                        if detection_dict:
                            print("[INFO] Detection JSON received:", detection_dict)
                            event_key = next(iter(detection_dict), "Unknown")
                            event_name = f"{event_key}_Event"
                            patrol_car_dir = f"Patrol_Car_{patrol_number}"
                            event_dir = os.path.join(MEDIA_BASE, patrol_car_dir, event_name)
                            os.makedirs(event_dir, exist_ok=True)
                        else:
                            if time.time() - last_log_time >= 10:
                                print("[INFO] detection_dict is empty.")
                                last_log_time = time.time()
                    
                    # 2. 비디오 데이터 처리
                    elif "video_size" in json_obj:
                        video_content = json_obj.get("video")

                        # 2-1. 영상 수신 완료 ('DONE')
                        if video_content == 'DONE':
                            print("[INFO] Video reception complete.")
                            if video_file:
                                video_file.close()  # 파일 닫기
                                print(f"[INFO] Video saved successfully at {video_path}")
                                
                                # DB에 메타데이터와 영상 경로 저장
                                if detection_data:
                                    save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
                                else:
                                    print("[WARN] Video received but no detection metadata.")
                                
                                # 초기화
                                video_file = None
                                detection_data = None

                            # 'DONE' 신호를 받으면 루프 종료 준비
                            data = b'' # 루프를 빠져나가기 위한 조건
                            break

                        # 2-2. 영상 데이터 청크 수신
                        else:
                            # 첫 영상 청크 수신 시, 파일 생성
                            if not video_file:
                                timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                                video_filename = f"{timestamp_str}.mp4"
                                video_path = os.path.join(event_dir, video_filename)
                                video_file = open(video_path, "wb")
                                print(f"[INFO] Receiving video file. Path: {video_path}")
                            
                            # Base64 디코딩하여 파일에 쓰기
                            decoded_chunk = base64.b64decode(video_content.encode('utf-8'))
                            video_file.write(decoded_chunk)

                except (json.JSONDecodeError, IndexError):
                    # 버퍼에 완전한 JSON이 없으면 다음 데이터 수신을 위해 대기
                    break
            
            if not data:
                break

    finally:
        # 예외 발생 시 파일이 열려있으면 닫기
        if video_file and not video_file.closed:
            video_file.close()
            print(f"[INFO] Video file closed due to connection termination.")

        # 영상 없이 메타데이터만 있는 경우 DB에 저장
        if detection_data and not video_path:
            save_to_db(detection_data)

        print(f"[INFO] Closing connection for {addr}")
        conn.close()


def start_server(host="0.0.0.0", port=2301):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"[INFO] Server started at {host}:{port}")

        while True:
            conn, addr = server_socket.accept()
            handle_client(conn, addr)


if __name__ == "__main__":
    start_server()
