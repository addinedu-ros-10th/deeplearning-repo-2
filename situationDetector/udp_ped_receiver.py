# udp_receiver.py
"""
PED로부터 영상 픽셀데이터를 수신하는 클라이언트
(PED) -> (AI_Server)
"""
import cv2
import socket
import struct
import threading
import numpy as np
import queue
import time

UDP_HOST = '0.0.0.0'  # 모든 인터페이스에서 데이터 수신
UDP_PORT = 6601
NUM_CHUNKS = 20
FRAME_TIMEOUT = 2.0  # 2초 이상된 미완성 프레임은 삭제

def receive_video_udp(frame_queue: queue.Queue, shutdown_event: threading.Event):
    """
    UDP 패킷을 수신하고 영상 프레임을 재조립하여 화면에 표시합니다.
    """
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_PORT))
    # recvfrom()이 블로킹되지 않도록 타임아웃 설정
    udp_sock.settimeout(1.0)
    print(f"AI Server (UDP) : {UDP_HOST}:{UDP_PORT}에서 데이터를 기다립니다.")

    frame_buffer = {}  # 프레임 조각들을 저장할 버퍼
    last_processed_frame = 0 # 마지막으로 처리된 프레임 ID

    while not shutdown_event.is_set():
        try:
            packet, _ = udp_sock.recvfrom(65536)  # 수신 버퍼 크기
            
            # 헤더(5바이트)와 청크 데이터 분리
            header = packet[:5]
            chunk_data = packet[5:]
            
            # 헤더 언패킹: Frame ID (unsigned long, 4바이트), Chunk Index (unsigned char, 1바이트)
            frame_id, chunk_index = struct.unpack('!LB', header)

            # 이미 처리된 프레임보다 오래된 패킷은 무시
            if frame_id <= last_processed_frame:
                continue

            # 새 프레임 ID인 경우 버퍼에 새 항목 추가
            if frame_id not in frame_buffer:
                frame_buffer[frame_id] = {
                    'chunks': [None] * NUM_CHUNKS,
                    'count': 0
                }
            
            # 청크 데이터 저장 및 카운트 증가
            if frame_buffer[frame_id]['chunks'][chunk_index] is None:
                frame_buffer[frame_id]['chunks'][chunk_index] = chunk_data
                frame_buffer[frame_id]['count'] += 1

            # 모든 조각이 도착했는지 확인
            if frame_buffer[frame_id]['count'] == NUM_CHUNKS:
                # 조각들을 하나로 합침
                jpeg_data = b''.join(frame_buffer[frame_id]['chunks'])
                
                # JPEG 데이터를 이미지 프레임으로 디코딩
                frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # cv2.imshow('AI Server - Video Stream', frame)
                    # 큐 처리 부분
                    try:
                        frame_queue.put(frame, block=False)
                    except queue.Full:
                        # 큐가 가득찬 상태이면 (YOLO 처리가 늦어지는 경우)
                        # 현재 프레임 버림
                        pass

                # 메모리 누수 방지를 위해 처리된 프레임 버퍼 삭제
                del frame_buffer[frame_id]
                last_processed_frame = frame_id

                # 오래된 프레임 버퍼 정리 (타임아웃 기반)
                current_time = time.time()
                # list(keys())로 키의 복사본을 만들어 순회 중 딕셔너리 변경 문제를 방지
                for fid in list(frame_buffer.keys()):
                    if current_time - frame_buffer[fid]['timestamp'] > FRAME_TIMEOUT:
                        print(f"AI Server (UDP) : Frame {fid} 타임아웃. 버퍼를 삭제합니다.")
                        del frame_buffer[fid]
                        
            # if cv2.waitKey(1) & 0xFF == 27: # ESC 키 입력 시 종료
            #     break
        
        except socket.timeout:
            # 타임아웃은 정상 동작이므로 계속 진행
            continue
        except Exception as e:
            print(f"AI Server (UDP) : 데이터 수신 오류: {e}")

    udp_sock.close()
    cv2.destroyAllWindows()
    print("AI Server (UDP) : 수신 스레드를 종료합니다.")