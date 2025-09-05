# fire-detect-client.py
import cv2
import time
import json
import base64
import socket
from typing import Dict
from ultralytics import YOLO

# 1. YOLO 모델 설정
# 경로 : 프로젝트 경로 기준
model = YOLO("feat_detect_fire_smoking/best.pt")

# 2. 통신 설정
HOST = 'localhost' # 로컬호스트
PORT = 6600 # 6600번 포트 사용

def generateJsonDump(result):
  """
  json 데이터 
  -> 보낼 형식에 맞는 데이터로 변환 
  -> str 데이터로 dump해서 변환하는 기능의 함수
  """
  # 이미지를 JPEG 형식의 byte stream으로 인코딩
  success, encoded_image = cv2.imencode('.jpg', result[0].orig_img)
  # base64를 이용해 byte stream을 텍스트(ASCII)로 변환
  base64_image = base64.b64encode(encoded_image).decode('utf-8')
  
  transform = {
    "timestamp" : time.time(),
    "detections" : result[0].names,
    # "orig_img" : result[0].orig_img, # : 이미지 데이터는 numpy이므로 그대로 전송이 불가능하다.
    "orig_img" : base64_image, # 
  }
  str_data = json.dumps(transform, ensure_ascii=False)
  return str_data.encode('utf-8') # str 바이트 데이터로 변환하여 return

def runCv(cap : cv2.VideoCapture, sock: socket.socket):
  """
  연결 완료된 소켓, 카메라를 인자로 받아
  카메라의 각 프레임마다 소켓을 통해 결과 전송
  """
  if not cap.isOpened():
    raise RuntimeError("Unable to open camera!")

  try:
    while True:
      # Read a frame from the video
      success, frame = cap.read()
      
      if success:
        # 각 프레임(이미지)에 대해서 YOLO 동작
        result = model.track(frame, persist = True)
        
        json_dump = generateJsonDump(result)
        send(sock, json_dump)

        # 27 : ESC / 키 입력 부분이 없으면 실행 간의 지연 시간이 없기 때문에 오류 발생
        if cv2.waitKey(1) == 27:
          break

  finally:
    cap.release()
    cv2.destroyAllWindows()

def send(sock : socket.socket, data : bytes):
  """
  주어진 소켓을 통해 데이터를 전송하는 기능
  소켓 연결은 담당하지 않고 데이터 전송만 담당함
  
  온전한 데이터 전송을 위해서, 데이터 길이를 보낸 후에 실제 데이터를 전송함
  """
  try:
    # 1. 데이터의 길이를 4바이트 빅 엔디안 정수로 변환
    data_len = len(data).to_bytes(4, byteorder='big')
    
    # 2. 데이터 길이 전송
    sock.sendall(data_len)
    
    # 3. 실제 데이터 전송
    sock.sendall(data)
    
    print("\n전송 완료")

  except (BrokenPipeError, ConnectionResetError):
    print(f"연결 끊어짐")
  finally:
    print("서버 종료")

def main():
  cap = cv2.VideoCapture(0)
  sock = None

  try:
    # 소켓 생성 / 연결 시도 : 1번만 시도
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"시스템 : 연결 대기 - {HOST} : {PORT}")
    sock.connect((HOST, PORT))
    print(f"시스템 : 연결 완료 - {HOST} : {PORT}")

    runCv(cap, sock)

  except ConnectionRefusedError:
    print(f"연결 거부")
  except Exception as e:
    print(f"오류 발생: {e}")
  finally:
    # 자원 해제
    print("프로그램을 종료합니다.")
    if sock:
      sock.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()