import socket
import json
from datetime import datetime
import struct
import time

class Tcp_client_manager():
    def __init__(self):
        self.TCP_HOST = "192.168.0.180"
        self.TCP_PORT = 1201
        self.PATROL_NUMBER = 1
        self.deviceManager_ID = 0x01
        self.situationDetector_ID = 0x02
        self.recv_data = ""
        self.alarm = 0
        self.last_alarm = 0
        
    def socket_init(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while (not connected):
            try:
                print("Socket Waiting for Connection from situationDetector ")
                self.client_socket.connect((self.TCP_HOST, self.TCP_PORT))
                connected = True
                print("situationDetector Connected")
            except Exception as e:
                print(f"Socket Initiate Error Occured : {e}")
                time.sleep(5)

    def send_data(self, data):
        self.client_socket.send(data)

    def send_data_all(self, data):
        self.client_socket.sendall(data)
        
    def receive_data(self):
        data_size = 4
        while True:
            try:
                self.client_socket.settimeout(2.0)
                self.recv_data = self.client_socket.recv(data_size)
                self.data_validation()
                time.sleep(0.1)
            except Exception:
                pass    
            except KeyboardInterrupt:
                print("Keyboard Interruption")
                break
    
    def data_validation(self):
        if (self.recv_data[0] != self.situationDetector_ID):
            print("Wrong Data Source Input")
            self.recv_data = ""
            return
        if (self.recv_data[1] != self.deviceManager_ID):
            print("Wrong Data Destination Input")
            self.recv_data = ""
            return
        if (self.recv_data[2] != self.PATROL_NUMBER):
            print("Wrong Patrol Number")
            self.recv_data = ""
            return
        self.alarm = int(self.recv_data[3])
        
        if (self.alarm != 0):
            self.last_alarm = self.alarm


    def socket_close(self):
        self.client_socket.close()

    def __exit__(self):
        print("tcp close")
        self.socket_close()
























# def manage_tcp_connection(tcp_connected_event: threading.Event, shutdown_event: threading.Event):
#     """
#     TCP 서버 연결을 관리하고 주기적으로 메타데이터를 전송
#     연결 상태에 따라 tcp_connected_event를 set/clear 합니다.
#     """
#     while not shutdown_event.is_set():
#         tcp_sock = None
#         try:
#             # 1. TCP 소켓 생성 및 연결
#             tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             print("PED (TCP) : AI Server에 연결 시도 중...")
#             tcp_sock.connect((TCP_HOST, TCP_PORT))
#             print("PED (TCP) : AI Server에 연결 완료.")
            
#             # 연결 성공 시 Event를 set하여 UDP 스레드에 알림
#             tcp_connected_event.set()

#             # 2. 연결된 동안 1초마다 메타데이터 전송
#             while not shutdown_event.is_set():
#                 g = geocoder.ip('me')
#                 metadata_json = {
#                     'timestamp': time.time(),
#                     'location': g.latlng if g.latlng else [0, 0],
#                     'patrol_car_name': PATROL_CAR_NAME,
#                 }
#                 metadata_bytes = json.dumps(metadata_json).encode('utf-8')
#                 tcp_sock.sendall(metadata_bytes)
#                 print(f"PED (TCP) : 메타데이터 전송: {metadata_json['timestamp']}")
#                 time.sleep(SEND_INTERVAL)

#         except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError) as e:
#             print(f"PED (TCP) : 연결 오류: {e}")
#         except Exception as e:
#             print(f"PED (TCP) : 예기치 않은 오류: {e}")
#             shutdown_event.set() # 심각한 오류 발생 시 전체 종료
#         finally:
#             # 연결이 끊겼으므로 Event를 clear하여 UDP 스레드에 알림
#             tcp_connected_event.clear()
#             print("PED (TCP) : 연결이 끊겼습니다. 5초 후 재시도합니다.")
#             if tcp_sock:
#                 tcp_sock.close()
            
#             # 재연결 시도 전 대기 (shutdown 이벤트 확인)
#             time.sleep(5)