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
model = YOLO("feat_detect_fire/best.pt")

# 2. 통신 설정
HOST = 'localhost' # 로컬호스트
PORT = 6600 # 6600번 포트 사용

"""
보내는 데이터 형식
transform = {
  "timestamp" : <감지 시간>,
  "boxes" : <감지한 객체 정보>,
  "video_path" : <로컬에 저장한 비디오 저장 경로>
}

"boxes" 정보
{
  "class_id" : <객체 ID> <int> # e.g. 0, 1, 2, 3
  "class_name" : <객체 이름> <str>
  "detect_count" : <감지한 객체 수> <int>
  "confidence" : <confidence 값> <float>
  "box_xyxy" : <Bounding Box 좌표정보> 
}
"""

def generateJsonDump(result):
  """
  json 데이터 
  -> 보낼 형식에 맞는 데이터로 변환 
  -> str 데이터로 dump해서 변환하는 기능의 함수
  """  
  detection_count = 0 # 감지한 객체수 정보
  detections_list = []
  
  # 1. 감지된 객체가 있는 경우에 감지 갯수(detection_count)에 추가
  if result[0].boxes is not None:
    detection_count = len(result[0].boxes)
    
    # 2. 객체 상세정보 추가
    for box in result[0].boxes:
      class_id = int(box.cls)
      detection_info = {
        "class_id" : class_id,                        # 객체 ID
        "class_name" : result[0].names[class_id],     # 객체 이름
        "detect_count" : detection_count,
        "confidence" : float(box.conf),               # confidence 값
        "box_xyxy" : box.xyxy.cpu().numpy().tolist(), # Bounding Box 좌표
      }
      detections_list.append(detection_info)
  
  transform = {
    "timestamp" : time.time(),
    "boxes" : detections_list,
    "video_path" : "/patrol_car/video", # 로컬에서 저장한 이미지 디렉터리 경로 정보 전송 (예시)
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
        
        # 테스트 환경 : 2초에 한번 전송
        time.sleep(2)

  finally:
    cap.release()
    cv2.destroyAllWindows()

def send(sock : socket.socket, data : bytes):
  """
  주어진 소켓을 통해 데이터를 전송하는 기능
  소켓 연결은 담당하지 않고 데이터 전송만 담당함
  """
  try:
    # 3. 실제 데이터 전송
    sock.sendall(data)
    
    print("\n클라이언트 : 전송 완료")

  except (BrokenPipeError, ConnectionResetError):
    print(f"클라이언트 : 연결 끊어짐")
  finally:
    print("클라이언트 종료")

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