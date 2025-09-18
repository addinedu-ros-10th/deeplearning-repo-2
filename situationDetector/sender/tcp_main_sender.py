# tcp_db_sender.py
import socket
import time
import queue
import threading

TCP_HOST = '192.168.0.182'  # dataService IP 주소 
TCP_PORT = 2301             # dataService TCP 발신 포트 주소

def send_tcp_data_to_db(db_manager_queue: queue.Queue, 
                        tcp_connected_event : threading.Event,
                        shutdown_event: threading.Event):
    """
    db_queue에서 분석 데이터를 가져와 DB 서버로 TCP 전송
    """
    while not shutdown_event.is_set():
        sock = None
        try:
            # DB 서버에 연결 시도
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"situationDetector (TCP DB Sender) : DB 서버({TCP_HOST}:{TCP_PORT})에 연결을 시도합니다.")
            sock.connect((TCP_HOST, TCP_PORT))
            print(f"situationDetector (TCP DB Sender) : DB 서버에 연결되었습니다.")

            # 연결 성공 시 Event를 set하여 UDP 스레드에 알림
            tcp_connected_event.set()

            # 연결이 유지되는 동안 큐에서 데이터 전송
            while not shutdown_event.is_set():
                try:
                    # 큐에서 데이터 가져오기 (타임아웃 설정)
                    data_to_send = db_manager_queue.get(timeout=1.0)
                    sock.sendall(data_to_send)
                    # print(f"AI Server (TCP DB Sender) : 데이터 전송 완료 ({len(data_to_send)} bytes)")
                    db_manager_queue.task_done() # 큐 작업 완료 표시
                except queue.Empty:
                    # 큐가 비어있으면 계속 대기
                    continue
                except socket.error as e:
                    # 소켓 오류 발생 시 연결 재시도
                    print(f"situationDetector (TCP DB Sender) : 소켓 오류 발생: {e}. 재연결을 시도합니다.")
                    break # 내부 루프를 빠져나가 외부 루프에서 재연결 시도
        
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"situationDetector (TCP DB Sender) : DB 서버 연결 실패: {e}")
        
        finally:
            if sock:
                sock.close()
            # 재연결 시도 전 잠시 대기
            if not shutdown_event.is_set():
                print("situationDetector (TCP DB Sender) : 5초 후 재연결을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP DB Sender) : DB 전송 스레드를 종료합니다.")