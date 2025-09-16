import threading # <-- 추가
import queue     # <-- 추가
import socket
import json
import threading
import queue
import time

# --- 설정 변수 ---
TCP_HOST = '0.0.0.0'  # 모든 IP 주소로부터 연결 허용
TCP_PORT = 7700       # 모니터링 GUI가 수신 대기할 TCP 포트

def handle_client_connection(conn: socket.socket, addr, gui_queue: queue.Queue, shutdown_event: threading.Event):
    """
    개별 클라이언트 연결을 처리하고 데이터를 수신하는 함수.
    """
    print(f"🖥️  Monitoring GUI (TCP) : 클라이언트 {addr} 와 연결되었습니다.")
    buffer = ""
    try:
        while not shutdown_event.is_set():
            # 1. 데이터 수신
            data = conn.recv(4096).decode('utf-8', errors='ignore')
            if not data:
                break  # 클라이언트가 연결을 종료함

            buffer += data
            
            # 2. 메시지 프레이밍 처리 (JSON 메시지는 개행 문자로 구분)
            # 버퍼에 개행 문자가 있는 동안 계속해서 메시지를 처리
            while '\n' in buffer:
                # 첫 번째 개행 문자를 기준으로 메시지와 나머지 버퍼를 분리
                message, buffer = buffer.split('\n', 1)
                
                if not message:
                    continue

                try:
                    # 3. JSON 데이터 파싱 시도
                    parsed_data = json.loads(message)
                    # 성공하면 JSON 객체를 큐에 넣음
                    gui_queue.put({'type': 'json', 'data': parsed_data})
                    
                except json.JSONDecodeError:
                    # 4. JSON 파싱 실패 시 바이너리 데이터로 간주
                    # 이 예제에서는 텍스트 기반이므로 오류로 처리.
                    # 만약 이미지를 TCP로 받는다면, 별도의 로직이 필요.
                    # 예를 들어, 요청-응답 패턴에서는 응답의 종류를 미리 알고 있어야 함.
                    print(f"⚠️  JSON 디코딩 오류: {message}")
                    # gui_queue.put({'type': 'binary', 'data': message.encode('utf-8')})

    except ConnectionResetError:
        print(f"🔌  클라이언트 {addr} 와의 연결이 초기화되었습니다.")
    except Exception as e:
        print(f"💥  처리 중 오류 발생 ({addr}): {e}")
    finally:
        print(f"🔌  클라이언트 {addr} 와의 연결이 종료되었습니다.")
        conn.close()

def receive_tcp_data(gui_queue: queue.Queue, shutdown_event: threading.Event):
    """
    TCP 서버를 열고 클라이언트의 연결을 기다립니다.
    연결된 각 클라이언트에 대해 별도의 처리 로직을 실행합니다.
    (이 코드에서는 한 번에 하나의 연결만 관리합니다.)
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 주소 재사용 옵션 설정
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((TCP_HOST, TCP_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0) # accept()가 블록되는 것을 방지

    print(f"🖥️  Monitoring GUI (TCP) : 서버 시작, 연결 대기 중... (Port: {TCP_PORT})")

    while not shutdown_event.is_set():
        try:
            # 5. 클라이언트 연결 대기
            conn, addr = server_sock.accept()
            # 이 예제는 간단하게 마지막 연결만 처리함.
            # 여러 동시 연결을 처리하려면 이 부분을 스레드로 만들어야 함.
            handle_client_connection(conn, addr, gui_queue, shutdown_event)
            
        except socket.timeout:
            continue
        except Exception as e:
            print(f"💥  서버 소켓 오류: {e}")
            break

    server_sock.close()
    print("🖥️  Monitoring GUI (TCP) : 수신 스레드를 종료합니다.")


# --- 이 파일을 직접 실행하여 테스트하기 위한 코드 ---
if __name__ == '__main__':
    print("--- TCP 수신기 테스트 시작 ---")
    
    # 가상 GUI 큐와 종료 이벤트 생성
    test_data_queue = queue.Queue()
    test_shutdown_event = threading.Event()

    # TCP 수신 스레드 시작
    receiver_thread = threading.Thread(
        target=receive_tcp_data,
        args=(test_data_queue, test_shutdown_event),
        daemon=True
    )
    receiver_thread.start()

    # 가상 클라이언트(송신자) 시뮬레이션
    def mock_client(messages):
        import time
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect(('localhost', TCP_PORT))
                print("\n[Mock Client] 서버에 연결됨.")
                for msg in messages:
                    # 메시지를 JSON 문자열로 변환하고 개행문자 추가
                    full_message = json.dumps(msg) + '\n'
                    sock.sendall(full_message.encode('utf-8'))
                    print(f"[Mock Client] 전송: {msg}")
                    time.sleep(1)
                print("[Mock Client] 전송 완료. 연결 종료.")
            except ConnectionRefusedError:
                print("[Mock Client] 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            except Exception as e:
                print(f"[Mock Client] 오류: {e}")

    # 테스트 메시지
    # situationDetector와 dataService가 보내는 데이터 형식을 모방
    mock_messages = [
        {"source": "live_analysis", "patrol_car": "Patrol_Car_1", "event": "Smoke", "count": 2},
        {"source": "db_response", "command": "get_logs", "data": [{"event_id": "evt_123", "type": "Fire"}]},
        {"source": "live_analysis", "patrol_car": "Patrol_Car_2", "event": "None", "count": 0},
    ]
    
    # 2초 후 클라이언트 실행
    time.sleep(2)
    mock_client(mock_messages)

    # 큐에 들어온 데이터 확인
    print("\n--- 수신된 데이터 확인 ---")
    try:
        while True:
            # 5초 동안 데이터가 없으면 종료
            item = test_data_queue.get(timeout=5)
            print(f"[Main Thread] 큐에서 데이터 수신: {item}")
    except queue.Empty:
        print("테스트 큐가 비어있습니다. 테스트를 종료합니다.")
    finally:
        test_shutdown_event.set()
        receiver_thread.join(timeout=2)
        print("\n--- TCP 수신기 테스트 종료 ---")