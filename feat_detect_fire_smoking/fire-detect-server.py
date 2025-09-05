# fire-detect-server.py
import socket
import json
import cv2
import numpy as np
import base64

"""
기능
1. 데이터 길이 수신 (이미지 데이터의 온전한 수신을 위해 데이터 길이를 받아옴)
2. 실제 데이터 수신
3. 수신한 데이터 디코딩 및 python json으로 변환
4. 받아온 영상을 cv2로 출력

특징
1. 빅 엔디안 사용
"""

HOST = 'localhost' # 로컬호스트
PORT = 6600 # 6600번 포트 사용

def recv_all(sock : socket.socket , n):
  """
  정확히 n 바이트를 수신할 때까지 receive를 반복할 수 있도록 해주는 헬퍼 함수
  """
  data = bytearray()
  while len(data) < n:
    packet = sock.recv(n - len(data))
    if not packet:
      return None
    data.extend(packet)
  return data

def main():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # 1. s.connect((HOST, PORT)) # 소켓을 여는 측은 서버, 연결하는 측은 클라이언트 이어야 함
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    print(f"서버 ({HOST}:{PORT}) 연결 대기.")

    # 2. 클라이언트 연결 수락
    conn, addr = s.accept()
    with conn:
      print(f"서버 ({HOST}:{PORT}) 에 연결되었습니다.")
      try:
        while True:
          # 1. 바이트 데이터 수신
          msglen_bytes = recv_all(conn, 4) # 4바이트
          if not msglen_bytes:
            print("데이터 길이 수신 실패. 연결 종료")
            break
          msglen = int.from_bytes(msglen_bytes, "big")

          # 2. 실제 데이터 수신
          data_bytes = recv_all(conn, msglen)
          if not data_bytes:
            print("데이터 수신 실패. 연결 종료")
            break
        
          # 3. 수신 데이터 디코딩
          # 여러 JSON 객체가 붇어서 올 수 있으므로 버퍼 사용
          json_string = data_bytes.decode('utf-8')

          # 4. 문자열을 python json으로 변환
          json_received = json.loads(json_string)

          # 5. timestamp, detections 출력
          print(f"[수신] : [timestamp : {json_received["timestamp"]}] / [detections : {json_received["detections"]}]")
          
          # 6. 이미지 출력
          base64_img_str = json_received["orig_img"]
          img_bytes = base64.b64decode(base64_img_str) # 6-2. Base64 문자열을 byte 데이터로 디코딩
          np_arr = np.frombuffer(img_bytes, np.uint8) # 6-3. byte 데이터를 numpy 배열로 변환
          img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # 6-4. numpy 배열을 OpenCV 이미지로 디코딩
          if img_decoded is not None: # 6-5. 화면에 이미지 표시
            cv2.imshow("Received Image", img_decoded)

          # 6-6. 키 입력 대기 (GUI 창 업데이트 및 유지를 위해 필수)
          # 1ms 동안 키 입력을 기다리며, ESC(ASCII 27)를 누르면 루프 종료
          if cv2.waitKey(1) & 0xFF == 27:
            break
      except ConnectionRefusedError:
          print(f"서버({HOST}:{PORT}) 연결 불가")
      except KeyboardInterrupt:
          print("\n사용자입력 : 프로그램 종료")
      finally:
          print("프로그램을 종료합니다.")

if __name__ == '__main__':
  main()