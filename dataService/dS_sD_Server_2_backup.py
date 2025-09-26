import socket
import json
import pymysql
import os
import datetime
import base64 # base64 모듈 추가

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
    if not detection_data.get("detection"):
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
            detection_key = next(iter(detection_data["detection"]))
            first_class_name = detection_data["detection"][detection_key][0].get("class_name", "Unknown")

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
    
    # TCP 스트림에서 JSON 객체를 분리하기 위한 버퍼
    buffer = b""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            
            buffer += data
            
            # 버퍼에서 완전한 JSON 객체를 찾아 분리하고 처리
            while True:
                try:
                    # 버퍼에서 JSON 객체를 찾고, 그 객체와 나머지 부분을 분리
                    # 이 방식은 간단하지만, 복잡한 데이터에서는 불안정할 수 있음
                    # 더 안정적인 방법은 메시지 앞에 길이를 붙이는 것
                    json_obj, idx = json.JSONDecoder().raw_decode(buffer.decode('utf-8'))
                    buffer = buffer[idx:]
                    
                    if "detection" in json_obj:
                        detection_data = json_obj

                        # print("[INFO] Detection JSON received")
                        # (기존 코드와 동일) 경로 설정 로직...
                        patrol_number = detection_data.get('patrol_number', 'Unknown')
                        
                        # event_key = next(iter(detection_data.get("detection", {"Unknown": []})))
                        
                        detection_dict = detection_data.get("detection", {})

                        if len(detection_dict) != 0:
                            print(detection_dict)
                        event_key = next(iter(detection_dict), "Unknown")


                        event_name = f"{event_key}_Event"
                        patrol_car_dir = f"Patrol_Car_{patrol_number}"
                        event_dir = os.path.join(MEDIA_BASE, patrol_car_dir, event_name)
                        os.makedirs(event_dir, exist_ok=True)

                    elif "videosize" in json_obj:

                        # 영상 데이터 처리
                        if not video_file: # 첫 영상 패킷인 경우
                            if not event_dir:
                                print("[ERROR] Video data received before detection data.")
                                event_dir = os.path.join(MEDIA_BASE, "unclassified")
                                os.makedirs(event_dir, exist_ok=True)
                            
                            timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                            video_filename = f"{timestamp_str}.mp4"
                            video_path = os.path.join(event_dir, video_filename)
                            video_file = open(video_path, "wb")
                            print(f"[INFO] Receiving video file: {video_path}")
                        
                        video_content = json_obj.get("video")

                        if video_content == 'DONE':
                            print("[INFO] Video transmission finished.")
                            
                            # 'DONE' 신호를 받으면 루프 종료 준비
                            data = b'' # 외부 루프 탈출을 위해
                            break
                        
                        # Base64 디코딩하여 파일에 쓰기
                        decoded_chunk = base64.b64decode(video_content.encode('ascii'))
                        video_file.write(decoded_chunk)

                except (json.JSONDecodeError, IndexError):
                    # 버퍼에 완전한 JSON이 없으면 대기
                    break
            if not data:
                break

    finally:
        if video_file:
            video_file.close()
            print(f"[INFO] Video saved successfully at {video_path}")
            if detection_data:
                save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
        
        elif detection_data:
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