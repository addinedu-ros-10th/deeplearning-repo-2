# udp_dm_receiver.py
"""
deviceManager로부터 영상 픽셀데이터를 수신 / 디코딩하는 클라이언트
(deviceManager) -> (situationDetector)
"""
import cv2
import time
import queue
import socket
import struct
import threading
import numpy as np
from typing import List

# deviceManager에서 수신하는 30초 영상 데이터 메타데이터
# 영상 데이터의 메타데이터는 영상 수신시 처음 수신하는 메타데이터로 저장함
video_shared_metadata = {
                            "source": 0x01,
                            "destination" : 0x02,
                            "patrol_number" : 1,
                            "timestamp" : None, # unsigned int[6] [년, 월, 일, 시, 분, 초]
                            "devicestatus" : 0, # 방송 동작 여부, 초기상태 : 0 (비동작)
                            "videosize" : 0, # 데이터 바이트 크기
                            "video" : 0, # TCP 영상 송수신 4096바이트 데이터 (이벤트 15초 전후 영상 데이터)
                        }


UDP_HOST = '192.168.0.181'  # situationDetector IP 주소
UDP_PORT = 1200             # situationDetector UDP 수신 포트 주소
NUM_CHUNKS = 20
FRAME_TIMEOUT = 2.0  # 2초 이상된 미완성 프레임은 삭제

def receive_video_udp(analyzer_input_queue: List[queue.Queue],
                    send_frame_queue: queue.Queue,
                    shutdown_event: threading.Event):
    """
    deviceManager으로부터 픽셀 데이터를 수신하고 디코딩하여 analysis_frame_queue와 send_frame_queue에 저장
    """
    frame_buffer = {}  # 프레임 조각들을 저장할 버퍼
    last_processed_frame = 0 # 마지막으로 처리된 프레임 ID

    while not shutdown_event.is_set():
        try:
            # 1. 메인 루프 안에서 UDP 소켓 생성 및 바인딩 -> 연결 실패 시에도 프로그램 종료 전까지 재시도
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.bind((UDP_HOST, UDP_PORT))
            # recvfrom()이 블로킹되지 않도록 타임아웃 설정
            udp_sock.settimeout(1.0)
            print(f"situationDetector (UDP) : {UDP_HOST}:{UDP_PORT}에서 데이터를 기다립니다.")

            # 2. 데이터 수신 루프
            while not shutdown_event.is_set():
                try:
                    packet, _ = udp_sock.recvfrom(65536)  # 수신 버퍼 크기
            
                    # 헤더(5바이트)와 픽셀 데이터 분리
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
                            'count': 0,
                            'timestamp': time.time()
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
                                # 모든 큐가 빌 때까지 기다렸다가 생산자 큐에 프레임을 복사해서 넣어줌
                                for q in analyzer_input_queue:
                                    if q.full():
                                        q.get() # 큐가 꽉 차있으면 이전 프레임을 버리고 새 것으로 교체 작업
                                    q.put(frame.copy())
                                send_frame_queue.put(frame, block=False)
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
                                print(f"situationDetector (UDP) : Frame {fid} 타임아웃. 버퍼를 삭제합니다.")
                                del frame_buffer[fid]
                except socket.timeout:
                    # 타임아웃은 정상 동작이므로 계속 진행
                    continue
                except Exception as e:
                    print(f"situationDetector (UDP) : 데이터 수신 오류: {e}")
        # 소켓 생성 / 바인딩 오류 시 처리 부분
        except Exception as e:
            print(f"situationDetector (UDP) : 소켓 생성 / 바인딩 오류 발생 {e}")
        
        finally:
            if udp_sock:
                udp_sock.close()
            # 종료 전까지 5초마다 연결 재시도
            if not shutdown_event.is_set():
                print(f"situationDetector (UDP) : 5초후 수신을 재시도합니다.")
                time.sleep(5)

    cv2.destroyAllWindows()
    print("situationDetector (UDP) : 수신 스레드를 종료합니다.")