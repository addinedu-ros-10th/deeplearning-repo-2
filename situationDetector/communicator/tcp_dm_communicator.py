# situationDetector/communicator/tcp_dm_communicator.py
"""
situationDetector <-> deviceManager 양방향 TCP 통신 모듈
"""
import socket
import threading
import queue
import time
import struct

# 통신 설정
TCP_HOST = 'localhost'  # situationDetector 자신의 IP 주소
TCP_PORT = 1201         # deviceManager와 통신할 단일 포트

# 수신 데이터 헤더 정보
HEADER_FORMAT = "BBBIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# 송신 테스트 데이터 (기존 sender 모듈 참고)
TEST_DATA_FIRE = b'\x02\x01\x01\x01'

def _handle_receive(conn: socket.socket, addr: tuple, event_video_queue: queue.Queue, shutdown_event: threading.Event):
    """
    하나의 deviceManager 클라이언트로부터 데이터를 지속적으로 수신하는 스레드.
    주요 기능: 30초 이벤트 영상 수신
    """
    print(f"situationDetector (TCP dM Communicator) : [{addr}] 수신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            # 1. 고정 크기의 헤더 수신
            header_data = conn.recv(HEADER_SIZE)
            if not header_data:
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 클라이언트 연결 끊어짐.")
                break
            
            # 2. 헤더 언패킹
            unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
            video_metadata = {
                "source": unpacked_header[0],
                "destination" : unpacked_header[1],
                "patrol_number" : unpacked_header[2],
                "timestamp" : list(unpacked_header[3:9]), # 튜플을 리스트로 변환
            }
            video_size = unpacked_header[9] # 영상 데이터 크기
            
            # 3. 헤더에 명시된 크기만큼 영상 데이터 수신
            video_buffer = b''
            remain_size = video_size
            
            while remain_size > 0:
                # 수신할 데이터 크기를 4096과 남은 크기 중 작은 값으로 선택
                chunk = conn.recv(min(4096, remain_size))
                if not chunk:
                    print(f"situationDetector (TCP dM Communicator) : [{addr}] 영상 데이터 수신 중 연결 끊김.")
                    # 데이터가 불완전하므로 루프를 빠져나감
                    break
                video_buffer += chunk
                remain_size -= len(chunk)
            
            # 수신이 완료되지 않았다면, 현재 영상은 폐기
            if remain_size > 0:
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 영상 데이터가 불완전하게 수신됨.")
                continue

            # 4. 수신 완료된 영상을 큐에 추가
            if video_buffer:
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 영상 수신 완료. 크기: {len(video_buffer)} 바이트")
                video_item = (video_metadata, video_buffer)
                event_video_queue.put(video_item)
            else:
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 수신된 영상 데이터가 없습니다.")

    except ConnectionResetError:
        print(f"situationDetector (TCP dM Communicator) : [{addr}] 클라이언트 연결이 리셋되었습니다.")
    except Exception as e:
        # 종료 이벤트가 설정되지 않은 경우에만 오류 출력
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dM Communicator) : [{addr}] 수신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dM Communicator) : [{addr}] 수신 스레드 종료.")

def _handle_send(conn: socket.socket, addr: tuple, dm_event_queue: queue.Queue, shutdown_event: threading.Event):
    """
    하나의 deviceManager 클라이언트에 데이터를 지속적으로 송신하는 스레드.
    주요 기능: 분석에 따른 이벤트 발생 데이터(알람 등) 전송
    """
    print(f"situationDetector (TCP dM Communicator) : [{addr}] 송신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            try:
                # TODO: 실제 운영 시에는 아래 주석 처리된 코드를 사용하여 dm_event_queue에서 데이터를 가져와 전송해야 합니다.
                # event_data = dm_event_queue.get(timeout=1.0)
                # conn.sendall(event_data)
                # print(f"situationDetector (TCP dM Communicator) : [{addr}] 데이터 전송 완료 ({len(event_data)} bytes)")

                # 테스트용 코드: 15초마다 화재 경고 데이터 전송 (기존 tcp_dm_event_sender.py 로직)
                time.sleep(15) 
                conn.sendall(TEST_DATA_FIRE)
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 테스트 데이터 전송 완료 ({len(TEST_DATA_FIRE)} bytes)")

            except queue.Empty:
                # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
                continue
            except (socket.error, BrokenPipeError) as e:
                print(f"situationDetector (TCP dM Communicator) : [{addr}] 소켓 오류 발생: {e}. 송신 스레드를 종료합니다.")
                break # 소켓 오류 시 루프 탈출
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dM Communicator) : [{addr}] 송신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dM Communicator) : [{addr}] 송신 스레드 종료.")

def dm_communicator_server(event_video_queue: queue.Queue,
                          dm_event_queue: queue.Queue,
                          shutdown_event: threading.Event):
    """
    deviceManager 클라이언트의 연결을 수락하고,
    각 클라이언트에 대해 양방향 통신(수신/송신) 스레드를 생성 및 관리.
    """
    server_sock = None
    while not shutdown_event.is_set():
        try:
            # 1. TCP 서버 소켓 생성 및 바인딩
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((TCP_HOST, TCP_PORT))
            server_sock.listen()
            # accept()가 블로킹되지 않도록 타임아웃 설정하여 shutdown_event를 주기적으로 확인
            server_sock.settimeout(1.0)
            print(f"situationDetector (TCP dM Communicator) : dM 통신 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기 중")

            conn, addr = None, None
            while not shutdown_event.is_set():
                try:
                    # 2. 클라이언트 연결 대기
                    conn, addr = server_sock.accept()
                    print(f"situationDetector (TCP dM Communicator) : dM 클라이언트 연결됨: {addr}")

                    # 3. 연결된 클라이언트를 위한 수신/송신 스레드 생성 및 시작
                    receiver = threading.Thread(target=_handle_receive, args=(conn, addr, event_video_queue, shutdown_event))
                    sender = threading.Thread(target=_handle_send, args=(conn, addr, dm_event_queue, shutdown_event))
                    receiver.daemon = True
                    sender.daemon = True
                    receiver.start()
                    sender.start()
                    
                    # 두 스레드가 모두 종료될 때까지 대기 (즉, 연결이 끊어질 때까지)
                    receiver.join()
                    sender.join()
                    print(f"situationDetector (TCP dM Communicator) : [{addr}] 클라이언트와의 세션이 종료되었습니다. 새 연결을 대기합니다.")

                except socket.timeout:
                    # 타임아웃은 정상 동작이므로 루프 계속
                    continue
                finally:
                    if conn:
                        conn.close()

        except Exception as e:
            print(f"situationDetector (TCP dM Communicator) : 서버 메인 루프 오류: {e}")
        finally:
            if server_sock:
                server_sock.close()
            # 종료 신호가 없을 경우에만 재시작
            if not shutdown_event.is_set():
                print("situationDetector (TCP dM Communicator) : 5초 후 서버 재시작을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP dM Communicator) : 통신 서버를 종료합니다.")