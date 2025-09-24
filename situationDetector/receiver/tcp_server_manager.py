import time
import socket
import threading

class TcpClientManager():
  def __init__(self, 
                tcp_host = "localhost",
                tcp_port = 1201,
                shutdown_event = None,
                client_name = None):
    """
    shutdown_event 이벤트 객체 주입 (선택)
    """
    self.tcp_host = tcp_host,
    self.tcp_port = tcp_port,
    self.PATROL_NUMBER = 1
    self.deviceManager_ID = 0x01
    self.situationDetector_ID = 0x02
    self.recv_data = ""
    self.connection = None # 연결된 클라이언트 소켓 저장 변수
    self.client_name = client_name
  
    self.shutdown_event = shutdown_event or threading.Event()
    
  def socket_init(self):
    """
    소켓 초기화 및 리슨 상태로 대기
    """
    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    
    # 2. 연결 시도 루프에서 shutdown_event 확인
    while not connected and not self.shutdown_event.is_set():
      try:
        print(f"situationDetector (TCP {self.client_name} Sender) : dM으로부터 연결을 기다리는 중...")
        # 테스트 환경 : 소켓 SO_REUSEADDR 옵션 설정
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_socket.bind((self.tcp_host, self.tcp_port))
        self.client_socket.listen()
        return True
      except Exception as e:
        print(f"situationDetector (TCP {self.client_name} Sender) : Socket Init Error! : {e}")
    
  def main_loop(self):
    """
    sub class에서 실제 기능에 따라 run메소드를 오버라이딩하여 사용
    현재 기능 : 받은 데이터 그대로 터미널에 출력
    """
    # 통신 시도 루프
    while not self.shutdown_event.is_set():
      try:
        # 클라이언트가 연결될 때까지 대기 ( accept()는 블로킹 함수임 )
        self.connection, addr = self.client_socket.accept()
        print(f"situationDetector (TCP {self.client_name} Sender) : TCP 연결")
        
      except Exception as e:
        print(f"situationDetector (TCP {self.client_name} Sender) : TCP 연결 오류 발생")
  
  def receive_data(self, data):
    self.client_socket.send(data)

  def receive_data_all(self, data):
    self.client_socket.sendall(data)
    