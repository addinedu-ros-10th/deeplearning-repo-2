# tcp_receiver.py
"""
GUI으로부터 이벤트 헤제 요청 수신 서버
(GUI) -> (situationDetector)
"""
import time
import queue
import socket
import struct
import threading

TCP_HOST = '192.168.0.86'  # 로컬 테스트 환경
# TCP_HOST = '192.168.0.181'  # situationDetector IP주소
TCP_PORT = 2401             # situationDetector TCP 수신 포트 : 1201


"""
(situationDetector) -> (deviceManager) Binary Interface
Source          : uint8_t : B
Destination     : uint8_t : B
call_command    : uint8_t : B
Alarm type      : uint8_t : B # 해당 타입의 알람 무시


GUI로부터 알람 해제 요청이 있는 경우, 30초동안 경고 방송을 해지함 (감지된 데이터 무시)
"""
HEADER_FORMAT = "BBBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def gui_server_run(event_clear_queue : queue.Queue,
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
            print(f"situationDetector (TCP, gui_event_clear) : 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기")

            # 2. 클라이언트 연결 대기 및 데이터 수신
            while not shutdown_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    with conn:
                        print(f"situationDetector (TCP, gui_event_clear) : 클라이언트 연결됨: {addr}")
                        while not shutdown_event.is_set():
                            # 3. 고정 크기의 헤더 수신
                            header_data = conn.recv(HEADER_SIZE)
                            
                            if not header_data:
                                print(f"situationDetector (TCP, gui_event_clear) : 클라이언트 연결 끊어짐: {addr}")
                                break
                            
                            # 4. 헤더 언패킹 및 json 형태로 변환
                            unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
                            video_shared_metadata = { # B B B B
                                "source": unpacked_header[0],
                                "destination" : unpacked_header[1],
                                "call_command" : unpacked_header[2],
                                "alarm_type" : unpacked_header[3],
                            }
                            
                            # # 5. 헤더 데이터 검증
                            # sig = True # 헤더 데이터 검증 시그널
                            # if unpacked_header[0] != 0x04:
                            #     sig = False
                            # elif unpacked_header[1] != 0x02:
                            #     sig = False
                            # elif unpacked_header[2] != 0x01:
                            #     sig = False
                            # elif unpacked_header[3] < 0x00 or unpacked_header[3] > 0x04:
                            #     sig = False                            
                            
                            # print(f"situationDetector (TCP, gui_event_clear) : 수신 데이터: {unpacked_header}") # 로깅 추가
                            
                            # 5. 헤더 데이터 검증 (필요시 활성화)
                            sig = True 
                            if unpacked_header[0] != 0x04: sig = False
                            elif unpacked_header[1] != 0x02: sig = False
                            elif unpacked_header[2] != 0x01: sig = False
                            elif unpacked_header[3] > 0x04: sig = False                            
                            
                            if not sig:
                                print(f"situationDetector (TCP, gui_event_clear) : GUI 이벤트 해제 요청 헤더 데이터 오류: {unpacked_header}")
                                continue # 잘못된 데이터는 무시하고 다음 수신 대기

                            
                            # print(unpacked_header)
                            print(video_shared_metadata)
                            print("데이터 검증 : ", sig)
                            
                            # if not sig:
                            #     print(f"situationDetector (TCP, gui_event_clear) : GUI 이벤트 해제 요청 헤더 데이터 오류: {e}")
                            # event_clear_queue.put(video_shared_metadata)
                            
                            
                except socket.timeout:
                    # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
                    continue
                except Exception as e1:
                    print(f"situationDetector (TCP, gui_event_clear) : 서버 오류: {e1}")
                    break
            # 소켓 생성 / 바인딩 오류 처리
        except Exception as e2:
            print(f"situationDetector (TCP, gui_event_clear) : 서버 준비 중 오류 발생 {e2}")

        # 소켓,연결 정리
        # finally:
        if conn:
            conn.close()
        if server_sock:
            server_sock.close()
        # 오류 발생 시 대기 후 재시도
        if not shutdown_event.is_set():
            print("situationDetector (TCP, gui_event_clear) : 오류 발생, 5초 후 서버 재시작을 시도합니다.")
            time.sleep(5)

    print("situationDetector (TCP, gui_event_clear) : 수신 스레드를 종료합니다.")