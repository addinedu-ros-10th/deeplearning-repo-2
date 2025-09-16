import queue
import threading
import cv2
import socket
import struct
import collections
import numpy as np

# deviceManager의 udp_ai_sender와 동일한 설정
NUM_CHUNKS = 20
UDP_PORT = 7701  # AI 서버 포트
DB_PORT = 7702   # DB 서버 포트 (다른 포트로 설정)

def receive_processed_video(gui_video_queue: queue.Queue, gui_db_queue: queue.Queue, shutdown_event: threading.Event):
    """AI 서버와 DB 서버로부터 데이터를 수신하고 처리"""
    HOST = '0.0.0.0'

    # AI 서버 소켓 설정
    ai_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ai_sock.bind((HOST, UDP_PORT))
    print(f"🖼️  Monitoring GUI (UDP) : AI 서버 영상 수신 대기 중... (Port: {UDP_PORT})")

    # DB 서버 소켓 설정
    db_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    db_sock.bind((HOST, DB_PORT))
    print(f"📊  Monitoring GUI (UDP) : DB 서버 데이터 수신 대기 중... (Port: {DB_PORT})")

    # AI 프레임 버퍼
    ai_frame_buffer = collections.defaultdict(dict)
    last_displayed_frame = -1

    # DB 데이터 버퍼 (간단한 예시로 가정)
    db_data_buffer = collections.deque(maxlen=100)  # 최근 100개 데이터 유지

    while not shutdown_event.is_set():
        try:
            # AI 서버 데이터 수신
            ai_readable, _, _ = select.select([ai_sock], [], [], 0.1)
            if ai_readable:
                packet, _ = ai_sock.recvfrom(65535)
                header = packet[:5]
                chunk_data = packet[5:]
                frame_id, chunk_id = struct.unpack('!LB', header)

                ai_frame_buffer[frame_id][chunk_id] = chunk_data

                if len(ai_frame_buffer[frame_id]) == NUM_CHUNKS:
                    if frame_id > last_displayed_frame:
                        last_displayed_frame = frame_id
                        sorted_chunks = [ai_frame_buffer[frame_id][i] for i in range(NUM_CHUNKS)]
                        data_bytes = b''.join(sorted_chunks)
                        frame = cv2.imdecode(np.frombuffer(data_bytes, dtype=np.uint8), 1)
                        if frame is not None:
                            gui_video_queue.put(frame)
                    del ai_frame_buffer[frame_id]

            # DB 서버 데이터 수신
            db_readable, _, _ = select.select([db_sock], [], [], 0.1)
            if db_readable:
                db_packet, _ = db_sock.recvfrom(65535)
                # 간단한 예시: DB 데이터는 JSON 문자열 또는 원시 데이터로 가정
                db_data = db_packet.decode('utf-8')  # 예: JSON 또는 텍스트 데이터
                db_data_buffer.append(db_data)
                gui_db_queue.put(db_data)  # GUI에 전달

            # 버퍼 관리
            if len(ai_frame_buffer) > 10:
                oldest_frame = min(ai_frame_buffer.keys())
                del ai_frame_buffer[oldest_frame]

        except Exception as e:
            print(f"🖼️  Monitoring GUI (UDP) : 오류 발생 - {e}")

    ai_sock.close()
    db_sock.close()
    print("🖼️  Monitoring GUI (UDP) : 모든 수신 스레드를 종료합니다.")

# select 모듈 import 추가
import select

if __name__ == "__main__":
    # 테스트용
    gui_video_queue = queue.Queue()
    gui_db_queue = queue.Queue()
    shutdown_event = threading.Event()
    
    receiver_thread = threading.Thread(target=receive_processed_video, args=(gui_video_queue, gui_db_queue, shutdown_event))
    receiver_thread.daemon = True
    receiver_thread.start()
    
    # 테스트: 큐에서 데이터 확인
    try:
        while True:
            if not gui_video_queue.empty():
                frame = gui_video_queue.get()
                print("Received video frame:", frame.shape)
            if not gui_db_queue.empty():
                db_data = gui_db_queue.get()
                print("Received DB data:", db_data)
            shutdown_event.wait(1)
    except KeyboardInterrupt:
        shutdown_event.set()
        receiver_thread.join()