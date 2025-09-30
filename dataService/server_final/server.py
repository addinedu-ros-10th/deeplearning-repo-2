# server.py (최종 수정본)

import socket
import json
import pymysql
import os
import datetime
import base64
import time
import threading
import pickle
import struct

# 설정 파일 임포트
import server_config as config

# --- 네트워크 프로토콜 함수 (GUI <-> 서버) ---
def send_msg(sock, data):
    try:
        pickled_data = pickle.dumps(data)
        msg = struct.pack('L', len(pickled_data)) + pickled_data
        sock.sendall(msg)
        return True
    except Exception as e:
        print(f"[GUI Server] 메시지 전송 실패: {e}")
        return False

def recv_msg(sock):
    try:
        packed_len = sock.recv(struct.calcsize('L'))
        if not packed_len: return None
        msg_len = struct.unpack('L', packed_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet: return None
            data += packet
        return pickle.loads(data)
    except Exception as e:
        print(f"[GUI Server] 메시지 수신 실패: {e}")
        return None

# --- DB 관련 함수 ---
def get_db_connection():
    """DB 커넥션을 생성하고 반환합니다."""
    return pymysql.connect(**config.DB_CONFIG)

def save_to_db(detection_data, video_path=None, mime_type=None):
    """situation 및 media 테이블에 데이터를 저장합니다."""
    conn = get_db_connection()
    if not detection_data.get("event_video_meta"):
        print("[Patrol Server] 경고: 'event_video_meta' 데이터가 비어있어 DB 저장을 건너뜁니다.")
        conn.close()
        return

    try:
        with conn.cursor() as cursor:
            # 1. situation 테이블 저장
            sql_situation = "INSERT INTO situation (timestamp, patrol_number, class_name, raw_result) VALUES (%s, %s, %s, %s)"
            ts = detection_data.get("timestamp")
            patrol_number = f"Patrol_number_{detection_data.get('patrol_number', 0)}"
            detection_key = next(iter(detection_data["event_video_meta"]))
            raw_result_str = json.dumps(detection_data)
            cursor.execute(sql_situation, (ts, patrol_number, detection_key, raw_result_str))
            class_id = cursor.lastrowid

            # 2. media 테이블 저장 (영상이 있는 경우)
            if video_path:
                sql_media = "INSERT INTO media (class_id, rel_path, validation, mime_type) VALUES (%s, %s, %s, %s)"
                rel_path = os.path.relpath(video_path, config.MEDIA_STORAGE_ROOT)
                cursor.execute(sql_media, (class_id, rel_path, True, mime_type))
        conn.commit()
        print(f"[Patrol Server] DB 저장 성공. Class ID: {class_id}")
    except Exception as e:
        print(f"[Patrol Server] DB 저장 실패: {e}")
        conn.rollback()
    finally:
        conn.close()

# ==============================================================================
# SECTION 1: 순찰차(Patrol Car) 데이터 수신 서버
# ==============================================================================
def handle_patrol_client(conn, addr):
    print(f"[Patrol Server] 순찰차 연결됨: {addr}")
    video_file = None
    detection_data = None
    video_path = None
    buffer = b""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break # 클라이언트가 연결을 끊으면 루프 종료
            
            buffer += data
            
            while True:
                try:
                    json_str = buffer.decode('utf-8')
                    json_obj, idx = json.JSONDecoder().raw_decode(json_str)
                    buffer = buffer[idx:]

                    if "event_video_meta" in json_obj:
                        detection_data = json_obj
                        patrol_num = detection_data.get('patrol_number', 'Unknown')
                        event_key = next(iter(detection_data.get("event_video_meta", {})), "Unknown")
                        
                        event_dir = os.path.join(config.MEDIA_STORAGE_ROOT, f"Event_Video/Patrol_Car_{patrol_num}/{event_key}_Event")
                        os.makedirs(event_dir, exist_ok=True)
                        
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                        video_filename = f"{timestamp}.mp4"
                        video_path = os.path.join(event_dir, video_filename)
                        
                        print(f"[Patrol Server] 메타데이터 수신. 영상 저장 경로 설정: {video_path}")
                    
                    elif "video_size" in json_obj:
                        video_content = json_obj.get("video")
                        if video_content == 'DONE':
                            if video_file:
                                video_file.close()
                                print(f"[Patrol Server] 영상 저장 완료: {video_path}")
                                if detection_data:
                                    save_to_db(detection_data, video_path=video_path, mime_type="video/mp4")
                                video_file, detection_data, video_path = None, None, None # 상태 초기화
                            
                            # [핵심 수정] return 대신 continue를 사용하여 연결을 유지하고 다음 데이터를 기다림
                            continue
                        else:
                            if video_path and not video_file:
                                video_file = open(video_path, "wb")
                            if video_file:
                                decoded_chunk = base64.b64decode(video_content.encode('utf-8'))
                                video_file.write(decoded_chunk)

                except (json.JSONDecodeError, IndexError):
                    break

    finally:
        if video_file and not video_file.closed:
            video_file.close()
        # 연결이 완전히 종료되기 전에 처리 못한 데이터가 있으면 저장
        if detection_data:
            save_to_db(detection_data, video_path=video_path, mime_type="video/mp4" if video_path else None)
        print(f"[Patrol Server] 순찰차 연결 종료: {addr}")
        conn.close()

def start_patrol_server():
    host, port = "0.0.0.0", config.PATROL_CAR_SERVER_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"[INFO] 순찰차 데이터 수신 서버 시작 (Port: {port})")
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_patrol_client, args=(conn, addr))
            client_thread.start()

# ==============================================================================
# SECTION 2: 관제 GUI 요청 처리 서버
# ==============================================================================
def handle_gui_client(conn, addr):
    print(f"[GUI Server] GUI 연결됨: {addr}")
    try:
        request = recv_msg(conn)
        if not request: return

        command = request.get('command')
        params = request.get('params')
        
        if command == 'get_logs':
            print(f"[GUI Server] 로그 요청 수신: {params}")
            db_conn = get_db_connection()
            try:
                with db_conn.cursor() as cursor:
                    query = "SELECT class_id, timestamp, class_name, raw_result FROM situation WHERE timestamp BETWEEN %s AND %s"
                    args = [params['start_date'], params['end_date'] + ' 23:59:59']
                    
                    if params.get('event_types'):
                        query += " AND class_name IN %s"
                        args.append(tuple(params['event_types']))

                    query += " ORDER BY timestamp " + ("DESC" if params.get('orderby_latest', True) else "ASC")
                    
                    cursor.execute(query, args)
                    result = cursor.fetchall()
                    send_msg(conn, result)
            finally:
                db_conn.close()

        elif command == 'get_video_path':
            print(f"[GUI Server] 영상 요청 수신: {params}")
            class_id = params.get('class_id')
            db_conn = get_db_connection()
            try:
                with db_conn.cursor() as cursor:
                    cursor.execute("SELECT rel_path FROM media WHERE class_id = %s", (class_id,))
                    result = cursor.fetchone()
                
                if result and result.get('rel_path'):
                    rel_path = result['rel_path'].lstrip('./')
                    abs_path = os.path.join(config.MEDIA_STORAGE_ROOT, rel_path)
                    
                    if os.path.exists(abs_path):
                        send_msg(conn, {'status': 'streaming_start'})
                        with open(abs_path, 'rb') as f:
                            while True:
                                chunk = f.read(4096)
                                if not chunk: break
                                conn.sendall(chunk)
                        conn.sendall(b'DONE')
                        print(f"[GUI Server] 영상 전송 완료: {abs_path}")
                    else:
                        send_msg(conn, {'error': f'파일을 찾을 수 없습니다: {abs_path}'})
                else:
                    send_msg(conn, {'error': '해당 class_id의 영상을 찾을 수 없습니다.'})
            finally:
                db_conn.close()

    except Exception as e:
        print(f"[GUI Server] GUI 클라이언트 처리 중 오류 발생: {e}")
    finally:
        print(f"[GUI Server] GUI 연결 종료: {addr}")
        conn.close()

def start_gui_server():
    host, port = "0.0.0.0", config.GUI_SERVER_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"[INFO] GUI 요청 처리 서버 시작 (Port: {port})")
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_gui_client, args=(conn, addr))
            client_thread.start()

# ==============================================================================
# MAIN: 두 서버를 별도의 스레드에서 실행
# ==============================================================================
if __name__ == "__main__":
    patrol_thread = threading.Thread(target=start_patrol_server)
    gui_thread = threading.Thread(target=start_gui_server)

    patrol_thread.start()
    gui_thread.start()

    patrol_thread.join()
    gui_thread.join()