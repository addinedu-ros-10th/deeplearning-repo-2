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
    pass
    # if (self.recv_data[0] != self.situationDetector_ID):
    #     print("Wrong Data Source Input")
    #     self.recv_data = ""
    #     return
    # if (self.recv_data[1] != self.deviceManager_ID):
    #     print("Wrong Data Destination Input")
    #     self.recv_data = ""
    #     return
    # if (self.recv_data[2] != self.PATROL_NUMBER):
    #     print("Wrong Patrol Number")
    #     self.recv_data = ""
    #     return
    # self.alarm = int(self.recv_data[3])
    
    # if (self.alarm != 0):
    #     self.last_alarm = self.alarm


# B , unsigned char
HEADER_FORMAT = "BBBIIIIIIBI"
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
                                "devicestatus" : unpacked_header[9], # 방송 동작 여부, 초기상태 : 0 (비동작)
                            }
                            video_size = unpacked_header[10] # 이번에 수신할 데이터 동영상 바이트 크기
                            
                            # 5. 헤더 데이터 검증
                            if not data_validation(video_shared_metadata):
                                print(f"situationDetector (TCP) : TCP 영상 헤더 데이터 오류: {e}")

                            # 6. 받아올 남은 데이터가 있으면 남은 길이만큼 추가 수신
                            if video_size > 0:
                                video_buffer = b'' # 버퍼 생성 및 버퍼에 비디오파일 수신
                                remain_size = video_size
                                
                                # 7. 지정된 길이의 비디오 데이터를 모두 수신할 때까지 반본
                                while remain_size > 0:
                                    chunk = conn.recv(min(4096, remain_size))
                                    if not chunk:
                                        print(f"situationDetector (TCP) : 영상 데이터 수신 중 연결 끊김")
                                        break
                                    video_buffer += chunk
                                    remain_size -= len(chunk)
                                
                                # 8. 이벤트 영상을 온전하게 받아왔으면, 동영상 저장 공유 큐에 저장
                                # 메타데이터와 영상 데이터를 튜플로 묶어 큐에 삽입 작업
                                if len(video_buffer) == video_size:
                                    video_item = (video_shared_metadata, video_buffer)
                                    event_video_queue.put(video_item)

                                # TEST : 받아온 영상 데이터 시각화


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

    # # 1. 프로그램 종료 전까지 연결 재시도
    # while not shutdown_event.is_set():
    #     server_sock = None
    #     conn = None
    #     try:
    #         server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         # SO_REUSEADDR 옵션을 설정하여 주소 재사용 문제를 방지
    #         server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #         server_sock.bind((TCP_HOST, TCP_PORT))
    #         server_sock.listen()
    #         # accept() 호출이 블로킹되지 않도록 타임아웃 설정
    #         server_sock.settimeout(1.0)
    #         print(f"situationDetector (TCP) : 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기")

    #         # 2. 클라이언트 연결 대기 및 데이터 수신
    #         while not shutdown_event.is_set():
    #             try:
    #                 conn, addr = server_sock.accept()
    #                 with conn:
    #                     print(f"situationDetector (TCP) : 클라이언트 연결됨: {addr}")
    #                     while not shutdown_event.is_set():
    #                         # 데이터 수신 (4096 바이트 버퍼)
    #                         data = conn.recv(4096)
    #                         if not data:
    #                             print(f"situationDetector (TCP) : 클라이언트 연결 끊어짐: {addr}")
    #                             break
                            
    #                         try:
    #                             recv_data = data.
                                
    #                             # metadata = json.loads(data.decode('utf-8'))
    #                             # Lock을 사용하여 공유 데이터 업데이트
    #                             with metadata_lock:

    #                                 # shared_metadata.update(metadata)
    #                         except (json.JSONDecodeError, UnicodeDecodeError) as e:
    #                             print(f"situationDetector (TCP) : 데이터 파싱 오류: {e}")

    #             except socket.timeout:
    #                 # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
    #                 continue
    #             except Exception as e:
    #                 print(f"situationDetector (TCP) : 서버 오류: {e}")
    #                 break
    #     # 소켓 생성 / 바인딩 오류 처리
    #     except Exception as e:
    #         print(f"situationDetector (TCP) : 서버 준비 중 오류 발생 {e}")

    #     # 소켓,연결 정리
    #     finally:
    #         if conn:
    #             conn.close()
    #         if server_sock:
    #             server_sock.close()
    #         # 오류 발생 시 대기 후 재시도
    #         if not shutdown_event.is_set():
    #             print("situationDetector (TCP) : 오류 발생, 5초 후 서버 재시작을 시도합니다.")
    #             time.sleep(5)

    # print("situationDetector (TCP) : 수신 스레드를 종료합니다.")








# class Tcp_service_manager():
#     def __init__(self):
#         self.shutdown_event = threading.Event
        
#         self.TCP_HOST = 'localhost'  # 로컬 테스트 환경
#         self.TCP_PORT = 1201
#         self.PATROL_NUMBER = 1
#         self.deviceManager_ID = 0x01
#         self.situationDetector_ID = 0x02
#         self.recv_data = ""
#         self.alarm = 0
#         self.last_alarm = 0
    
#     def socket_init(self):
#         self.service_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         connected = False
#         try:
#             # 테스트 환경 : SO_REUSEADDR 옵션을 설정하여 주소 재사용 문제를 방지
#             self.service_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             self.service_socket.bind((TCP_HOST, TCP_PORT))
#             self.service_socket.listen()
#             # accept() 호출이 블로킹되지 않도록 타임아웃 설정
#             self.service_socket.settimeout(1.0)
#             print(f"situationDetector (TCP) : 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기")

#             # 2. 클라이언트 연결 대기 및 데이터 수신
#             while not shutdown_event.is_set():