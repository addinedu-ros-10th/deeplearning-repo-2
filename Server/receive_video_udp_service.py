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

TCP : 메타데이터 전송
UDP : 영상 픽셀데이터 전송
"""
import cv2
import json
import time
import socket
import struct
import numpy as np

# 1. 통신 변수 지정
TCP_HOST = 'localhost'
TCP_PORT = 6600

UDP_HOST = 'localhost'
UDP_PORT = 6601

NUM_CHUNKS = 20 # 수신할 조각의 수

UDP_BUFFER_SIZE = 65536
HEADER_SIZE = struct.calcsize('!LB') # Frame ID(4 bytes) + Chunk Index(1 byte)


def receive_udp_stream(udp_sock: socket.socket, video_info : dict):
  """
  UDP 소켓을 통해 비디오 프레임 조각을 수신하고 재조립하여 화면에 표시
  """
  patrol_car_name = video_info.get('patrol_car_name', 'Unknown')
  timestamp = video_info.get('timestamp', 'Unknown')
  loc = video_info.get('location', 'Unknown')
  lat, lng = loc
  
  window_name = f"[{patrol_car_name}]UDP Video Stream (위도/경도) [{lat} / {lng}] (시간) [{timestamp}]"
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

  frame_buffer = {} # 1. key : 프레임 ID / value : 프레임 조각 딕셔너리 선언
  last_frame_processed_time = time.time()

  try:
    while True:
      # 2. UDP 패킷 수신
      packet, addr = udp_sock.recvfrom(UDP_BUFFER_SIZE)
      
      if len(packet) <= HEADER_SIZE:
        continue
      
      # 3. 헤더 언패킹: Frame ID와 인덱스 추출
      header = packet[:HEADER_SIZE]
      data_chunk = packet[HEADER_SIZE:]    
      frame_id, index = struct.unpack('!LB', header)
    
      # 4. 프레임 버퍼에 수신한 조각 저장
      if frame_id not in frame_buffer:
        frame_buffer[frame_id] = [None] * NUM_CHUNKS

      # 5. 중복 수신 방지
      if frame_buffer[frame_id][index] is None:
        frame_buffer[frame_id][index] = data_chunk
      
      # 6. 완성된 프레임 확인 및 처리
      sorted_frame_ids = sorted(frame_buffer.keys())
      
      for f_id in sorted_frame_ids:
        chunks = frame_buffer[f_id]
        if all(c is not None for c in chunks): # 모든 chunks가 유효하면 (== None이 아니면)
          try:
            # 20조각을 합쳐 JPEG 바이트 데이터 생성
            full_data = b''.join(chunks)          
            # (JPEG byte) -> (numpy 배열)
            np_arr = np.frombuffer(full_data, np.uint8)
            # Numpy배열을 이미지 프레임으로 디코딩
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
              cv2.imshow(window_name, frame)
            else:
              print(f"서버 (UDP): Frame {f_id} 디코딩 실패. 수신된 데이터 크기: {len(full_data)}")

          except Exception as e:
            print(f"서버 (UDP) : Frame {f_id} 처리 중 오류: {e}")
          finally:
            # 처리된 프레임은 버퍼에서 삭제
            del frame_buffer[f_id]
            last_frame_processed_time = time.time()

      # 타임아웃 처리: 오래된 미완성 프레임 삭제
      # 1초 이상 새로운 프레임이 처리되지 않았다면 버퍼 정리
      if time.time() - last_frame_processed_time > 1.0:
        if frame_buffer:
          # 가장 오래된(가장 낮은 ID의) 미완성 프레임을 삭제하여 블로킹 방지
          oldest_frame_id = min(frame_buffer.keys())
          print(f"서버 (UDP): 타임아웃. 미완성 Frame {oldest_frame_id} 삭제.")
          del frame_buffer[oldest_frame_id]
          last_frame_processed_time = time.time() # 타임아웃 시간 초기화



      # 'ESC' 키를 누르면 스트리밍 종료
      if cv2.waitKey(1) == 27:
        break
  finally:
    print("서버 (UDP): 스트리밍을 종료합니다.")
    cv2.destroyAllWindows()

def main():
  tcp_sock = None
  udp_sock = None

  try:
    # --- 1. TCP/UDP 소켓 생성 및 바인딩 ---
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.bind((TCP_HOST,TCP_PORT))
    tcp_sock.listen()

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_PORT))

    print(f"situationDetector (TCP) : PED의 연결을 기다리는중 ({UDP_HOST}:{UDP_PORT})")
    
    # 데이터 수신 루프
    while True:
      # --- 2. TCP 연결 확인(핸드세이크) ---
      conn, addr = tcp_sock.accept()
      print(f"situationDetector (TCP) : PED와 연결됨")
      
      # PED으로부터 메타데이터 수신
      metadata_bytes = conn.recv(1024)
      if not metadata_bytes:
        conn.close()
        continue
      
      video_info = json.loads(metadata_bytes.decode('utf-8'))
      print(f"situationDetector (TCP) : 영상 기본정보 수신 완료")
      
      # PED에 영상 수신 준비완료 'OK'신호 전송
      conn.sendall(b'OK')
      print("situationDetector (TCP) : PED에 'OK' 전송. UDP 영상 수신 시작")
      
      # --- 3. UDP 스트리밍 시작 ---
      receive_udp_stream(udp_sock, video_info)

      # 스트리밍 종료 후, 클라이언트로부터 'STOP' 메시지 대기
      stop_signal = conn.recv(1024)
      if stop_signal == b'STOP':
        print("서버 (TCP): 클라이언트로부터 종료 신호 수신.")
      
      print(f"\n서버 (TCP): 클라이언트({addr})와의 세션 종료. 다음 연결을 기다립니다.")

  except Exception as e:
    print(f"서버 오류 발생: {e}")
  finally:
    print("서버를 종료합니다.")
    if tcp_sock:
      tcp_sock.close()
    if udp_sock:
      udp_sock.close()
    cv2.destroyAllWindows()
      
if __name__ == "__main__":
  main()