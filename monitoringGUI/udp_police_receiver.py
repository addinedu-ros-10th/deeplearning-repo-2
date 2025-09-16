import queue
import threading
import cv2
import socket
import struct
import collections
import numpy as np

# deviceManager의 udp_ai_sender와 동일한 설정
NUM_CHUNKS = 20
UDP_PORT = 7701  # GUI1.py에서 import할 수 있도록 명시적 정의

def receive_processed_video(gui_video_queue: queue.Queue, shutdown_event: threading.Event):
    """AI 서버로부터 처리된 영상(프레임 조각)을 수신하고 완전한 프레임으로 복원"""
    HOST = '0.0.0.0'
    PORT = UDP_PORT  # PORT를 UDP_PORT와 동일하게 사용

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((HOST, PORT))
    print(f"🖼️  Monitoring GUI (UDP) : 처리된 영상 수신 대기 중... (Port: {PORT})")

    frame_buffer = collections.defaultdict(dict)
    last_displayed_frame = -1

    while not shutdown_event.is_set():
        try:
            packet, _ = udp_sock.recvfrom(65535)
            
            # 헤더(frame_id, chunk_id)와 청크 데이터 분리
            header = packet[:5]
            chunk_data = packet[5:]
            frame_id, chunk_id = struct.unpack('!LB', header)

            # 프레임 버퍼에 청크 저장
            frame_buffer[frame_id][chunk_id] = chunk_data

            # 모든 청크가 도착했는지 확인
            if len(frame_buffer[frame_id]) == NUM_CHUNKS:
                # 현재 프레임보다 이전 프레임은 무시 (네트워크 지연 등 고려)
                if frame_id > last_displayed_frame:
                    last_displayed_frame = frame_id
                    
                    # 청크를 순서대로 조합
                    sorted_chunks = [frame_buffer[frame_id][i] for i in range(NUM_CHUNKS)]
                    data_bytes = b''.join(sorted_chunks)
                    
                    # 바이트를 이미지로 디코딩
                    frame = cv2.imdecode(np.frombuffer(data_bytes, dtype=np.uint8), 1)
                    if frame is not None:
                        # GUI에 표시하기 위해 큐에 프레임 삽입
                        gui_video_queue.put(frame)

                # 처리 완료된 프레임 버퍼 정리
                del frame_buffer[frame_id]

            # 버퍼가 너무 커지는 것을 방지
            if len(frame_buffer) > 10:
                oldest_frame = min(frame_buffer.keys())
                del frame_buffer[oldest_frame]

        except Exception as e:
            print(f"🖼️  Monitoring GUI (UDP) : 오류 발생 - {e}")

    udp_sock.close()
    print("🖼️  Monitoring GUI (UDP) : 수신 스레드를 종료합니다.")