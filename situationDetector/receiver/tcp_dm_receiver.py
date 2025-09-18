# tcp_receiver.py
"""
PED로부터 영상 메타데이터를 수신하는 클라이언트
(PED) -> (situationDetector)
"""
import socket
import json
import threading

TCP_HOST = '192.168.0.181'  # situationDetector IP주소
TCP_PORT = 1201             # situationDetector TCP 수신 포트 : 1201

def handle_tcp_client(shared_metadata:dict,
                    metadata_lock: threading.Lock, 
                    shutdown_event: threading.Event):
    """
    TCP 클라이언트의 연결을 수락하고 메타데이터를 수신
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR 옵션을 설정하여 주소 재사용 문제를 방지
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((TCP_HOST, TCP_PORT))
    server_sock.listen()
    # accept() 호출이 블로킹되지 않도록 타임아웃 설정
    server_sock.settimeout(1.0)
    print(f"situationDetector (TCP) : 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기")

    while not shutdown_event.is_set():
        try:
            conn, addr = server_sock.accept()
            with conn:
                print(f"situationDetector (TCP) : 클라이언트 연결됨: {addr}")
                while not shutdown_event.is_set():
                    # 데이터 수신 (4096 바이트 버퍼)
                    data = conn.recv(4096)
                    if not data:
                        print(f"situationDetector (TCP) : 클라이언트 연결 끊어짐: {addr}")
                        break
                    
                    try:
                        metadata = json.loads(data.decode('utf-8'))
                        # Lock을 사용하여 공유 데이터 업데이트
                        with metadata_lock:
                            shared_metadata.update(metadata)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"situationDetector (TCP) : 데이터 파싱 오류: {e}")

        except socket.timeout:
            # 타임아웃은 정상적인 상황이므로 루프를 계속 진행
            continue
        except Exception as e:
            print(f"situationDetector (TCP) : 서버 오류: {e}")
            break
    
    server_sock.close()
    print("situationDetector (TCP) : 수신 스레드를 종료합니다.")