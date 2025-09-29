import socket
import json
import pymysql
import os
import datetime

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
    """
    클라이언트 연결 및 데이터 처리를 핸들링하는 함수 (수정된 핵심 로직)
    """
    print(f"[INFO] Connected by {addr}")
    buffer = b""
    video_file = None
    detection_data = None
    video_path = None
    # --- [수정] 이벤트 유형에 따른 동적 경로를 저장할 변수 ---
    event_dir = None


    # try:

    while True:
        data = conn.recv(1024)
        if not data:
            break

        if data == b"DONE":
            print("[INFO] DATA DONE")
            break

        try:
                # JSON 데이터인지 확인
                json_obj = json.loads(data.decode("utf-8"))

                if "detection" in json_obj:
                    detection_data = json_obj
                    print("[INFO] Detection JSON received")

                    # --- [핵심 로직 1] 수신된 JSON 데이터로 저장 경로 생성 ---
                    patrol_number = detection_data.get('patrol_number', 'Unknown')
                    
                    # 'detection' 데이터가 비어있을 경우에 대한 방어 코드
                    detection_content = detection_data.get("detection", {})
                    if not detection_content:
                        event_name = "Unknown_Event"
                        # print("[WARN] 'detection' 필드가 비어있어 이벤트 타입을 'Unknown_Event'으로 설정합니다.")
                    else:
                        # 첫 번째 탐지된 객체의 이름을 이벤트 타입으로 사용
                        event_key = next(iter(detection_content))
                        event_name = f"{event_key}_Event"

                    # 최종 저장 디렉터리 경로 조합: Event_Video/Patrol_Car_(N)/Smoke_Event/
                    patrol_car_dir = f"Patrol_Car_{patrol_number}"
                    event_dir = os.path.join(MEDIA_BASE, patrol_car_dir, event_name)

                    # os.makedirs로 하위 디렉토리까지 한번에 생성 (이미 있어도 에러 없음)
                    os.makedirs(event_dir, exist_ok=True)
                    print(f"[INFO] Video save directory set to: {event_dir}")
                    # --- [핵심 로직 1 끝] ---

                elif "videosize" in json_obj:
                    # --- [핵심 로직 2] 결정된 경로에 파일 생성 준비 ---
                    # if not event_dir:
                    #     # detection JSON이 먼저 오지 않은 경우, 예외 처리
                    #     print("[ERROR] Video metadata received before detection data. Saving to default 'unclassified' directory.")
                    #     event_dir = os.path.join(MEDIA_BASE, "unclassified")
                    #     os.makedirs(event_dir, exist_ok=True)

                    # 파일명 생성: YYYY-MM-DD_HH_MM_SS.mp4
                    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                    video_filename = f"{timestamp_str}.mp4"
                    
                    # 최종 파일 전체 경로
                    video_path = os.path.join(event_dir, video_filename)
                    
                    # 파일 열기
                    video_file = open(video_path, "wb")
                    print(f"[INFO] Preparing to receive video file: {video_path}")
                    # --- [핵심 로직 2 끝] ---

        except (json.JSONDecodeError, UnicodeDecodeError):
            # JSON 파싱 실패 시, 영상 데이터로 간주하고 파일에 쓰기
            if video_file:
                video_file.write(data)

    # except Exception as e:
    #     # 모든 종류의 예외를 잡아서 로그로 남김
    #     print(f"[ERROR] An error occurred with client {addr}: {e}")
    #     # 필요하다면 traceback 전체를 출력해 디버깅
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     # 에러가 발생하든, 정상 종료되든 항상 연결을 닫아줌
    #     print(f"[INFO] Closing connection for {addr}")
    #     conn.close()


    # 모든 데이터 수신 완료 후 처리
    if video_file:
        video_file.close()
        print(f"[INFO] Video saved successfully at {video_path}")
        
        if detection_data:
            # DB에는 전체 경로를 저장
            save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
    elif detection_data:
        save_to_db(detection_data) # 영상 없이 탐지 정보만 저장

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