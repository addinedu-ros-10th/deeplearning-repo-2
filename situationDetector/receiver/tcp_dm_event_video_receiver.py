# tcp_receiver.py
"""
dM 으로부터 이벤트 영상 수신 코드
(deviceManager) -> (situationDetector)
"""
import time
import json
import queue
import socket
import struct
import threading

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


TCP_HOST = 'localhost'  # 로컬 테스트 환경
# TCP_HOST = '192.168.0.181'  # situationDetector IP주소
TCP_PORT = 1201             # situationDetector TCP 수신 포트 : 1201

def data_validation(video_shared_metadata):
    """
    TODO : 30초 이벤트 영상 메타데이터 검증 코드 작성
    """
    return True

# B , unsigned char
# HEADER_FORMAT = "BBBIIIIIIBI"
HEADER_FORMAT = "BBBIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def receive_event_video(event_video_queue : queue.Queue,
                    event_video_metadata : dict, 
                    metadata_lock : threading.Lock,
                    shutdown_event : threading.Event):
    """
    TCP 클라이언트의 연결을 수락하고 30초 이벤트 영상 메타데이터를 수신
    """
    while not shutdown_event.is_set():
        server_sock = None
        conn = None
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_REUSEADDR 옵션을 설정하여 주소 재사용 문제를 방지
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((TCP_HOST, TCP_PORT))
            server_sock.listen()
            # accept() 호출이 블로킹되지 않도록 타임아웃 설정
            server_sock.settimeout(1.0)
            print(f"situationDetector (TCP) : 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기")

            # 2. 클라이언트 연결 대기 및 데이터 수신
            while not shutdown_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    with conn:
                        print(f"situationDetector (TCP) : 클라이언트 연결됨: {addr}")
                        while not shutdown_event.is_set():
                            # 3. 고정 크기의 헤더 수신
                            header_data = conn.recv(HEADER_SIZE)
                            
                            if not header_data:
                                print(f"situationDetector (TCP) : 클라이언트 연결 끊어짐: {addr}")
                                break
                            
                            # 4. 헤더 언패킹 및 json 형태로 변환
                            unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
                            video_shared_metadata = { # B B B IIIIII B I
                                "source": unpacked_header[0],
                                "destination" : unpacked_header[1],
                                "patrol_number" : unpacked_header[2],
                                "timestamp" : unpacked_header[3:9], # unsigned int[6] [년, 월, 일, 시, 분, 초]
                                # "devicestatus" : unpacked_header[9], # 방송 동작 여부, 초기상태 : 0 (비동작)
                            }
                            video_size = unpacked_header[9] # 이번에 수신할 데이터 동영상 바이트 크기
                            
                            # 5. 헤더 데이터 검증
                            if not data_validation(video_shared_metadata):
                                print(f"situationDetector (TCP) : TCP 영상 헤더 데이터 오류: {e}")

                            video_buffer = b'' # 버퍼 생성 및 버퍼에 비디오파일 수신
                            # remain_size = video_size
                            
                            # # 7. 지정된 길이의 비디오 데이터를 모두 수신할 때까지 반본
                            # while remain_size > 0:
                            #     chunk = conn.recv(min(4096, remain_size))
                            #     if not chunk:
                            #         print(f"situationDetector (TCP) : 영상 데이터 수신 중 연결 끊김")
                            #         break
                            #     video_buffer += chunk
                            #     remain_size -= len(chunk)

                            # b"DONE" 신호를 수신할 때까지 데이터를 계속 읽음
                            while not shutdown_event.is_set():
                                chunk = conn.recv(1024)
                                if not chunk:
                                    print(f"situationDetector (TCP) : 영상 데이터 수신 중 연결 끊김")
                                    break
                                
                                if chunk.endswith(b"DONE"):
                                    video_buffer += chunk[:-4] # b"DONE" 제외
                                    break # 수신 완료
                                else:
                                    video_buffer += chunk
                            

                            # TEST : 받아온 영상 데이터 시각화
                            print(video_item)
                            
                            # 4. 이벤트 영상을 온전하게 받아왔으면, 큐에 저장
                            if video_buffer:
                                print(f"situationDetector (TCP) : 영상 수신 완료. 크기: {len(video_buffer)} 바이트")
                                video_item = (video_shared_metadata, video_buffer)
                                event_video_queue.put(video_item)
                            else:
                                print(f"situationDetector (TCP) : 수신된 영상 데이터가 없습니다.")


                except socket.timeout:
                    # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
                    continue
                except Exception as e:
                    print(f"situationDetector (TCP) : 서버 오류: {e}")
                    break
        # 소켓 생성 / 바인딩 오류 처리
        except Exception as e:
            print(f"situationDetector (TCP) : 서버 준비 중 오류 발생 {e}")

        # 소켓,연결 정리
        finally:
            if conn:
                conn.close()
            if server_sock:
                server_sock.close()
            # 오류 발생 시 대기 후 재시도
            if not shutdown_event.is_set():
                print("situationDetector (TCP) : 오류 발생, 5초 후 서버 재시작을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP) : 수신 스레드를 종료합니다.")
