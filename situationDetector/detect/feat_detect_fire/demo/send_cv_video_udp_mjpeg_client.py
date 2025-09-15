"""
TCP 영상전송 vs UDP 영상전송
소켓타입
- TCP : 스트림 기반, socket.SOCK_STREAM
- UDP : 데이터그램 기반, socket.DGRAM

연결 방식
- TCP : 연결 기반 통신
  1. 클라이언트 : connect()으로 서버에 연결을 요청함
  2. 서버 : listen()과 accept()으로 연결을 수락함

- UDP : 비연결 기반 통신
  1. 클라이언트 : sendto()를 이용해 지정된 주소로 데이터를 보내기만 함
  2. 서버 : recvfrom()을 통해 데이터를 수신함

코드 기능
- TCP : 메타데이터 전송 (시간, 위치, 순찰차 이름)
- UDP : 영상 픽셀데이터 전송 (바이트)
"""

"""
전송 TCP 메타데이터 예시
{
  "timestamp" : <감지 시간>, (e.g. time.time() 정보)
  "location" : <현재 위치 정보>, (e.g. [순찰차 위도, 순찰차 경도] , type : <geocoder.ip('me').latlng> )
  "patrol_car_name" : <순찰차 이름>, (e.g. "patrol_car1" )
}



"""

import cv2
import time
import json
import struct
import geocoder
import socket

# 1. 통신 변수 지정
TCP_HOST = 'localhost' # TCP통신 : 메타데이터 전송에 사용
TCP_PORT = 6600

UDP_HOST = 'localhost' # UDP통신 : 영상 픽셀데이터 전송에 사용
UDP_PORT = 6601

NUM_CHUNKS = 20 # 수신할 조각의 수

PATROL_CAR_NAME = 'patrol_car_1'

def stream_video(cap : cv2.VideoCapture,
                  udp_sock : socket.socket,
                  udp_server_addr):
  """
  카메라 프레임을 (jpeg) 바이트로 변환하고 20개 조각으로 나누어 UDP 전송
  이 함수는 TCP 연결이 완료된 후에 실행됨
  """
  frame_id = 0 # 프레임 ID 카운터
  try:
    while True:
      success, frame = cap.read()
      if not success:
        continue
      
      # 프레임을 JPEG 형식의 바이트로 인코딩
      encode_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
      if not encode_success:
        continue

      data_bytes = buffer.tobytes()
      
      # 데이터 분할 전송
      total_size = len(data_bytes)
      chunk_size = (total_size + NUM_CHUNKS - 1) // NUM_CHUNKS

      for i in range(NUM_CHUNKS):
        # 각 조각 전송
        start = i * chunk_size
        end = start + chunk_size
        chunk = data_bytes[start:end]
        
        # 헤더: Frame ID(unsigned long) + Chunk Index(unsigned char)
        header = struct.pack('!LB', frame_id, i)
        packet = header + chunk
        udp_sock.sendto(packet, udp_server_addr)
        
        # packet = i.to_bytes(1, 'big') + chunk # 빅 엔디안 사용
        # udp_sock.sendto(packet, udp_server_addr)

      print(f"PED (UDP): Frame {frame_id}를 {NUM_CHUNKS}개 조각으로 전송 완료 (총 {total_size} bytes)")
      frame_id += 1 # 다음 프레임을 위해 ID 증가
      
      if cv2.waitKey(1) == 27:
        break
  finally:
    print("PED (UDP): 스트리밍 루프 종료")

def main():
  tcp_sock = None
  udp_sock = None
  cap = None
  
  try:
    # --- 1. TCP / UDP 소켓 생성 ---
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # --- 2. TCP 연결 확인 ---
    print(f"PED (TCP) : situationDetector에 연결 시도 중")
    tcp_sock.connect((TCP_HOST, TCP_PORT))
    print(f"PED (TCP) : situationDetector에 연결 완료")
    
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #   raise RuntimeError("opencv 카메라 열기 오류!")
    
    # 메타데이터 준비
    g = geocoder.ip('me')
    metadata_json = {
      'timestamp' : time.time(),
      'location' : g.latlng,
      'patrol_car_name' : PATROL_CAR_NAME,
    }
    metadata_bytes = json.dumps(metadata_json).encode('utf-8')
    tcp_sock.sendall(metadata_bytes)
    print(f"PED (TCP) : 영상 기본정보 전달 완료")

    # 'ACK' : Acknowledgement, 수신 측이 보낸 데이터 패킷을 성공적으로 받았다는 것을 송신 측에 알리는 신호이다.
    # 서버로부터 'OK' 신호 수신 대기
    ack = tcp_sock.recv(1024)
    if ack != b'OK':
      raise ConnectionError("PED (TCP) : situationDetector로부터 'OK'수신 실패")
    print("PED (TCP) : situationDetector 준비 완료 ('OK'수신) UDP전송 시작")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      raise RuntimeError("PED : 카메라를 열 수 없습니다.")
    
    # --- 3. UDP 전송 시작 ---
    stream_video(cap, udp_sock, (UDP_HOST, UDP_PORT))
  
  except Exception as e:
    print(f"PED : 오류 발생 : {e}")
  finally:
    # --- 4. 종료 처리 ---
    print("PED : 프로그램을 종료합니다.")
    if cap:
      cap.release()
    
    # 서버에 종료 신호 전송
    if tcp_sock:
      try:
        tcp_sock.sendall(b'STOP')
        print("PED (TCP) : 'STOP' 신호를 situationDetector에 전송")
      except BrokenPipeError:
        print("PED (TCP) : situationDetector와의 연결 끊어짐")
      finally:
        tcp_sock.close()
    if udp_sock:
      udp_sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()