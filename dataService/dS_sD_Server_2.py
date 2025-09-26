
import socket
import json
import pymysql
import os
import datetime
import base64
from typing import Optional

# -------------------------
# 설정
# -------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "0000",
    "database": "patroldb",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

MEDIA_BASE = "./Event_Video"
os.makedirs(MEDIA_BASE, exist_ok=True)

# -------------------------
# DB 저장 함수
# -------------------------
def save_to_db(detection_data: dict, video_path: Optional[str] = None, mime_type: Optional[str] = None):
    """
    situation 테이블과 media 테이블에 데이터 저장.
    - detection_data: 수신된 JSON 전체
    - video_path: 실제 저장된 파일의 절대경로 (또는 None)
    """
    # detection 키가 없거나 비어있으면 경고 후 저장 시도 가능 (요구대로 DB에 저장할지 결정)
    detection_dict = detection_data.get("detection", {})
    if not detection_dict:
        print("[WARN] 'detection' 데이터가 비어있어 situation 테이블에 저장하지 않습니다.")
        return

    # 안전하게 class_name 추출
    first_key = next(iter(detection_dict), None)
    if first_key:
        first_list = detection_dict.get(first_key, [])
        if isinstance(first_list, list) and len(first_list) > 0:
            first_class_name = first_list[0].get("class_name", "Unknown")
        else:
            first_class_name = "Unknown"
    else:
        first_class_name = "Unknown"

    # patrol_car 칼럼은 "Patrol_Car_N" 형태로 저장
    patrol_number = detection_data.get("patrol_number", "Unknown")
    patrol_car = f"Patrol_Car_{patrol_number}"

    ts = detection_data.get("timestamp")
    raw_result_str = json.dumps(detection_data, ensure_ascii=False)

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            sql_situation = """
                INSERT INTO situation (timestamp, patrol_car, class_name, raw_result)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql_situation, (ts, patrol_car, first_class_name, raw_result_str))
            class_id = cursor.lastrowid

            if video_path:
                rel_path = os.path.relpath(video_path, MEDIA_BASE).replace("\\", "/")
                sql_media = """
                    INSERT INTO media (class_id, rel_path, validation, mime_type)
                    VALUES (%s, %s, %s, %s)
                """
                # validation은 현재 True로 표시 (추후 체크 로직 필요시 변경)
                cursor.execute(sql_media, (class_id, rel_path, True, mime_type or "video/mp4"))

        conn.commit()
        print(f"[INFO] DB 저장 완료: situation.class_id={class_id}")
    except Exception as e:
        print(f"[ERROR] DB 저장 실패: {e}")
    finally:
        conn.close()

# -------------------------
# 클라이언트 처리
# -------------------------
def handle_client(conn: socket.socket, addr):
    print(f"[INFO] Connected by {addr}")

    buffer_str = ""  # 수신 버퍼(문자열) — bytes -> decode 후 누적
    detection_data = None

    video_file = None        # 열린 파일 객체
    video_path = None
    event_dir = None
    expected_video_size = None
    bytes_received = 0

    decoder = json.JSONDecoder()

    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                # 클라이언트가 종료했음
                break

            try:
                chunk_decoded = chunk.decode("utf-8")
            except UnicodeDecodeError:
                # 수신 중간에 binary 조각이 섞여 들어오는 설계라면 여기는 처리하지 않음.
                # 그러나 본 설계는 모든 전송이 JSON 문자열(Base64 포함)이라 가정.
                print("[WARN] 수신 데이터 디코딩 실패 (utf-8); 해당 청크를 무시합니다.")
                continue

            buffer_str += chunk_decoded

            # buffer_str 안에 여러 개의 JSON 객체가 연속 있을 수 있으니 반복 파싱
            while buffer_str:
                buffer_str = buffer_str.lstrip()
                try:
                    obj, idx = decoder.raw_decode(buffer_str)
                except json.JSONDecodeError:
                    # 현재 buffer_str로는 완전한 JSON이 형성되지 않음 -> 추가 recv 대기
                    break

                # 객체 파싱 성공: obj, idx 위치 반환
                buffer_str = buffer_str[idx:]  # 남은 문자열로 갱신

                # -----------------------------
                # 수신한 JSON 오브젝트 처리
                # -----------------------------
                if "detection" in obj:
                    # detection(메타) 수신
                    detection_data = obj
                    print("[INFO] Detection JSON received:", detection_data)

                    # 디렉터리 구조 준비
                    patrol_number = detection_data.get("patrol_number", "Unknown")
                    detection_dict = detection_data.get("detection", {})
                    event_key = next(iter(detection_dict), "Unknown")
                    # event name mapping: e.g., feat_detect_smoke -> Smoke_Event
                    # 기본은 event_key 뒤에 "_Event" 추가
                    # 필요하면 매핑 테이블을 만들어 더 친절히 변환 가능
                    event_name = f"{event_key}_Event"

                    patrol_car_dir = f"Patrol_Car_{patrol_number}"
                    event_dir = os.path.join(MEDIA_BASE, patrol_car_dir, event_name)
                    os.makedirs(event_dir, exist_ok=True)
                    print(f"[INFO] Event directory ready: {event_dir}")

                elif "videosize" in obj and "video" in obj:
                    # 비디오 청크 혹은 종료 신호
                    video_size = obj.get("videosize")
                    video_chunk = obj.get("video")

                    # 첫 비디오 청크 수신 시 파일 오픈(디렉터리/파일명 생성)
                    if video_file is None:
                        if event_dir is None:
                            # detection이 먼저 안왔을 때는 unclassified 폴더로 저장
                            event_dir = os.path.join(MEDIA_BASE, "unclassified")
                            os.makedirs(event_dir, exist_ok=True)
                            print("[WARN] detection 데이터 없이 비디오 수신 - unclassified 사용")

                        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                        video_filename = f"{timestamp_str}.mp4"
                        video_path = os.path.join(event_dir, video_filename)
                        try:
                            video_file = open(video_path, "wb")
                            expected_video_size = int(video_size) if video_size is not None else None
                            bytes_received = 0
                            print(f"[INFO] Started receiving video => {video_path} (expected {expected_video_size})")
                        except Exception as e:
                            print(f"[ERROR] 영상 파일 열기 실패: {e}")
                            video_file = None

                    # 종료 신호 처리
                    if isinstance(video_chunk, str) and video_chunk == "DONE":
                        print("[INFO] Received video DONE signal.")
                        # 파일 닫기 및 DB 저장(있다면)
                        if video_file:
                            video_file.close()
                            print(f"[INFO] Video saved successfully at {video_path} (received {bytes_received} bytes)")
                            # DB 저장 (detection_data는 있을 수도, 없을 수도 있음)
                            if detection_data:
                                save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
                            else:
                                # detection이 없는 경우에는 빈 detection으로 DB 저장하지 않음(정책)
                                print("[INFO] No detection metadata available; DB insert skipped for situation.")
                            video_file = None
                            video_path = None
                            event_dir = None
                            expected_video_size = None
                            bytes_received = 0
                        continue

                    # 일반 데이터 청크: base64 디코딩 후 파일 쓰기
                    if isinstance(video_chunk, str):
                        try:
                            decoded = base64.b64decode(video_chunk)
                        except Exception as e:
                            print(f"[ERROR] base64 디코딩 실패: {e}")
                            decoded = b""

                        if video_file:
                            try:
                                video_file.write(decoded)
                                bytes_received += len(decoded)
                            except Exception as e:
                                print(f"[ERROR] video_file.write 실패: {e}")
                        else:
                            print("[WARN] video_file 이 준비되지 않은 상태에서 비디오 조각을 받았습니다. 해당 조각 무시됨.")

                else:
                    # 알 수 없는/다른 JSON이 올 수 있음 — 로그만 남김
                    print("[DEBUG] Unknown JSON object received:", obj)

        # end while recv
    except ConnectionResetError:
        print(f"[WARN] ConnectionReset by peer: {addr}")
    except Exception as e:
        print(f"[ERROR] handle_client 예외: {e}")
    finally:
        # 연결 종료 시 열린 파일 정리 및 DB 저장(비정상 종료 방어)
        try:
            if video_file:
                video_file.close()
                print(f"[INFO] Video saved (on finally) at {video_path}")
                if detection_data:
                    save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
        except Exception as e:
            print(f"[ERROR] finally cleanup 실패: {e}")

        # detection만 있고 비디오가 없는 경우 DB에 저장
        if detection_data and video_file is None:
            # 이미 비디오 포함하여 저장된 경우를 제외하고 상황정보만 DB에 저장
            save_to_db(detection_data, video_path=None, mime_type=None)

        print(f"[INFO] Closing connection for {addr}")
        try:
            conn.close()
        except Exception:
            pass

# -------------------------
# 서버 시작
# -------------------------
def start_server(host="0.0.0.0", port=2301):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"[INFO] Server started at {host}:{port}")

        while True:
            conn, addr = server_socket.accept()
            # 단일 스레드 모델: 각 연결을 순차 처리
            # 운영 환경에서는 threading.Thread 또는 concurrent.futures.ThreadPoolExecutor 권장
            handle_client(conn, addr)

if __name__ == "__main__":
    start_server()

