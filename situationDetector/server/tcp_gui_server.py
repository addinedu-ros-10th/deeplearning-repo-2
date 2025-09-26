# tcp_gui_server.py
"""
GUI으로부터 이벤트 헤제 요청 수신 서버
(GUI) -> (situationDetector)
"""
# import json
import time
import json
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

def _handle_send(conn: socket.socket, addr: tuple, final_output_queues: queue.Queue, shutdown_event: threading.Event):
    """
    하나의 dataServer 클라이언트에 데이터를 지속적으로 송신하는 스레드.
    송신 데이터 : 합산 데이터 (6가지 모델 분석 결과 + 해제 요청이 있으면 30초간 무시)
    """
    print(f"situationDetector (TCP GUI Communicator) : [{addr}] 송신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            try:
                ai_result = final_output_queues.get()
                # conn.send(ai_result)
                json_string = json.dumps(ai_result) 
                conn.send(json_string.encode('utf-8'))                
                
                # print(f"situationDetector (TCP GUI Communicator) : [{addr}] 테스트 데이터 전송 완료 ({len(ai_result)} bytes)")

            except queue.Empty:
                # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
                continue
            except (socket.error, BrokenPipeError) as e:
                print(f"situationDetector (TCP GUI Communicator) : [{addr}] 소켓 오류 발생: {e}. 송신 스레드를 종료합니다.")
                break # 소켓 오류 시 루프 탈출
            finally:
                # 테스트 코드 : 전송한 데이터 그대로 출력
                print(ai_result)
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP GUI Communicator) : [{addr}] 송신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP GUI Communicator) : [{addr}] 송신 스레드 종료.")

def _handle_receive(conn: socket.socket, 
                    addr: tuple, 
                    event_clear_queue: queue.Queue, 
                    shutdown_event: threading.Event):
    """
    연결된 클라이언트로부터 이벤트 해제 요청 데이터를 지속적으로 수신하고 큐에 추가합니다.

    Args:
        conn (socket.socket): 클라이언트와 연결된 소켓 객체
        addr (tuple): 클라이언트의 주소 (IP, port)
        event_clear_queue (queue.Queue): 수신된 데이터를 전달할 큐
        shutdown_event (threading.Event): 스레드 종료를 알리는 이벤트 객체
    """
    print(f"situationDetector (TCP, receiver) : [{addr}] 수신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            # 1. 고정 크기의 헤더 데이터 수신
            # conn.recv()는 블로킹 함수이지만, 데이터가 없거나 연결이 끊기면 반환됨
            header_data = conn.recv(HEADER_SIZE)

            # 2. 클라이언트 연결 종료 확인
            if not header_data:
                print(f"situationDetector (TCP, receiver) : 클라이언트 [{addr}] 연결 끊어짐")
                break
            
            # 수신된 데이터의 길이가 예상과 다른 경우 무시
            if len(header_data) < HEADER_SIZE:
                print(f"situationDetector (TCP, receiver) : [{addr}] 로부터 불완전한 헤더 수신 (수신: {len(header_data)}B, 필요: {HEADER_SIZE}B)")
                continue

            # 3. 헤더 언패킹
            unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
            
            # 4. 헤더 데이터 검증
            # 프로토콜: source=0x04, destination=0x02, call_command=0x01, alarm_type=0x00~0x03
            source, destination, call_command, alarm_type = unpacked_header
            
            is_valid = (source == 0x04 and
                        destination == 0x02 and
                        call_command == 0x01 and
                        0x00 <= alarm_type <= 0x03)

            if not is_valid:
                print(f"situationDetector (TCP, receiver) : [{addr}] 로부터 잘못된 형식의 헤더 수신: {unpacked_header}")
                continue

            # 5. 검증된 데이터를 딕셔너리로 변환하여 큐에 추가
            video_shared_metadata = {
                "source": source,
                "destination": destination,
                "call_command": call_command,
                "alarm_type": alarm_type,
            }
            event_clear_queue.put(video_shared_metadata)
            print(f"situationDetector (TCP, receiver) : [{addr}] 로부터 유효한 이벤트 해제 요청 수신: {video_shared_metadata}")

    except ConnectionResetError:
        print(f"situationDetector (TCP, receiver) : 클라이언트 [{addr}]에 의해 연결이 강제로 재설정되었습니다.")
    except Exception as e:
        # shutdown_event가 설정되지 않은 경우에만 오류를 출력하여 정상 종료와 구분
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP, receiver) : [{addr}] 데이터 수신 중 예외 발생: {e}")
    finally:
        # 이 스레드는 소켓을 닫지 않습니다.
        # 소켓의 생명주기는 메인 스레드인 gui_server_run에서 관리하는 것이 더 안전합니다.
        print(f"situationDetector (TCP, receiver) : [{addr}] 수신 스레드 종료")


def gui_server_run(final_output_queue : queue.Queue,
                    event_clear_queue : queue.Queue,
                    shutdown_event : threading.Event):
    """
    TCP 클라이언트의 연결을 수락하고 영상 해제 데이터를 수신
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

            conn, addr = None, None
            # 2. 클라이언트 연결 대기 및 데이터 수신
            while not shutdown_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    print(f"situationDetector (TCP gui_event_clear) : 클라이언트 연결됨: {addr}")

                    # 3. 연결된 클라이언트를 위한 수신/송신 스레드 생성 및 시작
                    sender = threading.Thread(target=_handle_send, args=(conn, addr, final_output_queue, shutdown_event))
                    receiver = threading.Thread(target=_handle_receive, args=(conn, addr, event_clear_queue, shutdown_event))
                    
                    sender.daemon = True
                    receiver.daemon = True

                    sender.start()
                    receiver.start()
                    
                    # 스레드가 모두 종료될 때까지 대기 (즉, 연결이 끊어질 때까지)
                    sender.join()
                    receiver.join()
                    print(f"situationDetector (TCP dS Communicator) : [{addr}] 클라이언트와의 세션이 종료되었습니다. 새 연결을 대기합니다.")


                    # with conn:
                    #     print(f"situationDetector (TCP, gui_event_clear) : 클라이언트 연결됨: {addr}")
                    #     while not shutdown_event.is_set():
                    #         # 3. 고정 크기의 헤더 수신
                    #         print("TEST1")
                    #         header_data = conn.recv(HEADER_SIZE)
                            
                    #         if not header_data:
                    #             print(f"situationDetector (TCP, gui_event_clear) : 클라이언트 연결 끊어짐: {addr}")
                    #             break
                            
                    #         # 4. 헤더 언패킹 및 json 형태로 변환
                    #         unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
                    #         video_shared_metadata = { # B B B B
                    #             "source": unpacked_header[0],
                    #             "destination" : unpacked_header[1],
                    #             "call_command" : unpacked_header[2],
                    #             "alarm_type" : unpacked_header[3],
                    #         }
                            
                    #         print(f"situationDetector (TCP, gui_event_clear) : 수신 데이터: {unpacked_header}") # 로깅 추가
                            
                    #         # 5. 헤더 데이터 검증
                    #         sig = True # 헤더 데이터 검증 시그널
                    #         if unpacked_header[0] != 0x04:
                    #             sig = False
                    #         elif unpacked_header[1] != 0x02:
                    #             sig = False
                    #         elif unpacked_header[2] != 0x01:
                    #             sig = False
                    #         elif not (0x00 <= unpacked_header[3] <= 0x03):
                    #             sig = False                            
                            
                    #         if not sig:
                    #             print(f"situationDetector (TCP, gui_event_clear) : GUI 이벤트 해제 요청 헤더 데이터 오류: {unpacked_header}")
                            
                    #         # 테스트 코드 : 출력
                    #         print(sig)
                    #         print(conn)
                    #         print(unpacked_header)
                    #         print(video_shared_metadata)
                            
                    #         event_clear_queue.put(video_shared_metadata)
                            
                            
                except socket.timeout:
                    # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
                    continue
                except Exception as e:
                    print(f"situationDetector (TCP, gui_event_clear) : 서버 오류: {e}")
                    break
            # 소켓 생성 / 바인딩 오류 처리
        except Exception as e:
            print(f"situationDetector (TCP, gui_event_clear) : 서버 준비 중 오류 발생 {e}")
        finally:
            if server_sock:
                server_sock.close()
            # 오류 발생 시 대기 후 재시도
            if not shutdown_event.is_set():
                print("situationDetector (TCP, gui_event_clear) : 오류 발생, 5초 후 서버 재시작을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP, gui_event_clear) : 수신 스레드를 종료합니다.")

def gui_server_run_archive(event_clear_queue : queue.Queue,
                    shutdown_event : threading.Event):
    """
    TCP 클라이언트의 연결을 수락하고 영상 해제 데이터를 수신
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
                            print("TEST1")
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
                            
                            print(f"situationDetector (TCP, gui_event_clear) : 수신 데이터: {unpacked_header}") # 로깅 추가
                            
                            # 5. 헤더 데이터 검증
                            sig = True # 헤더 데이터 검증 시그널
                            if unpacked_header[0] != 0x04:
                                sig = False
                            elif unpacked_header[1] != 0x02:
                                sig = False
                            elif unpacked_header[2] != 0x01:
                                sig = False
                            elif not (0x00 <= unpacked_header[3] <= 0x03):
                                sig = False                            
                            
                            if not sig:
                                print(f"situationDetector (TCP, gui_event_clear) : GUI 이벤트 해제 요청 헤더 데이터 오류: {unpacked_header}")
                            
                            # 테스트 코드 : 출력
                            print(sig)
                            print(conn)
                            print(unpacked_header)
                            print(video_shared_metadata)
                            
                            event_clear_queue.put(video_shared_metadata)
                            
                            
                except socket.timeout:
                    # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
                    continue
                except Exception as e:
                    print(f"situationDetector (TCP, gui_event_clear) : 서버 오류: {e}")
                    break
            # 소켓 생성 / 바인딩 오류 처리
        except Exception as e:
            print(f"situationDetector (TCP, gui_event_clear) : 서버 준비 중 오류 발생 {e}")

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