# tcp_sender.py
import socket
import json
import time
import geocoder
import threading

# 통신 변수
TCP_HOST = 'localhost'
TCP_PORT = 6600
PATROL_CAR_NAME = 'patrol_car_1'
SEND_INTERVAL = 1  # 1초

def manage_tcp_connection(tcp_connected_event: threading.Event, shutdown_event: threading.Event):
    """
    TCP 서버 연결을 관리하고 주기적으로 메타데이터를 전송
    연결 상태에 따라 tcp_connected_event를 set/clear 합니다.
    """
    while not shutdown_event.is_set():
        tcp_sock = None
        try:
            # 1. TCP 소켓 생성 및 연결
            tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("PED (TCP) : AI Server에 연결 시도 중...")
            tcp_sock.connect((TCP_HOST, TCP_PORT))
            print("PED (TCP) : AI Server에 연결 완료.")
            
            # 연결 성공 시 Event를 set하여 UDP 스레드에 알림
            tcp_connected_event.set()

            # 2. 연결된 동안 1초마다 메타데이터 전송
            while not shutdown_event.is_set():
                g = geocoder.ip('me')
                metadata_json = {
                    'timestamp': time.time(),
                    'location': g.latlng if g.latlng else [0, 0],
                    'patrol_car_name': PATROL_CAR_NAME,
                }
                metadata_bytes = json.dumps(metadata_json).encode('utf-8')
                tcp_sock.sendall(metadata_bytes)
                print(f"PED (TCP) : 메타데이터 전송: {metadata_json['timestamp']}")
                time.sleep(SEND_INTERVAL)

        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"PED (TCP) : 연결 오류: {e}")
        except Exception as e:
            print(f"PED (TCP) : 예기치 않은 오류: {e}")
            shutdown_event.set() # 심각한 오류 발생 시 전체 종료
        finally:
            # 연결이 끊겼으므로 Event를 clear하여 UDP 스레드에 알림
            tcp_connected_event.clear()
            print("PED (TCP) : 연결이 끊겼습니다. 5초 후 재시도합니다.")
            if tcp_sock:
                tcp_sock.close()
            
            # 재연결 시도 전 대기 (shutdown 이벤트 확인)
            time.sleep(5)