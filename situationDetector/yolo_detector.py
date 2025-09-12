"""
공유 큐에서 프레임을 가져와 YOLO 객체를 탐지하고 결과를 화면에 표시하는 데모 함수 파일
"""
import cv2
import json
import time
import queue
import threading
from ultralytics import YOLO

def generateDetectJsonDump(result, time, patrol_car_name):
  """
  DB Server에 보낼 json 데이터 생성
  프레임 해석 정보와 기본 정보 (영상 메타데이터 : 시간, 위치, 순찰차 이름)을 인자로 받아 새로운 json 데이터 생성
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
    "timestamp" : time, # 영상 메타데이터 : 순찰차 영상 촬영 시각
    "patrol_car_name" : patrol_car_name, # 영상 메타데이터 : 순찰차 이름
    "boxes" : detections_list,
  }
  str_data = json.dumps(transform, ensure_ascii=False)
  return str_data.encode('utf-8') # str 바이트 데이터로 변환하여 return



def detect_objects(frame_queue: queue.Queue, 
                    db_manager_queue:queue.Queue, 
                    shared_metadata: dict, 
                    metadata_lock:threading.Lock,
                    shutdown_event: threading.Event):
  print("AI Server (YOLO) : YOLO 스레드를 시작합니다. 모델을 로드합니다.")
  
  model = YOLO("situationDetector/best.pt")
  
  # 프로그램이 종료되지 않은 동안
  while not shutdown_event.is_set():
    try:
      # 큐에서 프레임을 가져오고, timeout을 설정하여 blocking방지 및 종료신호 확인
      frame = frame_queue.get(timeout=1.0)
      
      # YOLO 실행
      results = model(frame)
      
      # 변경/추가된 부분: 메타데이터 읽기 및 JSON 생성 후 DB 큐에 추가
      current_timestamp = None
      current_patrol_car = None
      with metadata_lock:
          current_timestamp = shared_metadata.get("timestamp")
          current_patrol_car = shared_metadata.get("patrol_car_name")

      if current_timestamp and current_patrol_car:
          json_output = generateDetectJsonDump(results, current_timestamp, current_patrol_car)
          try:
              db_manager_queue.put(json_output, block=False)
          except queue.Full:
              print("AI Server (YOLO) : DB 큐가 가득 찼습니다. 분석 결과를 버립니다.")
      
      # 시각화 부분 (필요시 주석 해제)
      annotated_frame = results[0].plot()
      cv2.imshow("Detector test", annotated_frame)
      if cv2.waitKey(1) & 0xFF == 27:
        print("AI Server (YOLO) : 종료 키 입력 감지. 종료 신호를 보냅니다.")
        shutdown_event.set()
        break

    except queue.Empty:
      # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
      continue
    except Exception as e:
      print(f"AI Server (YOLO) : 처리 중 오류 발생: {e}")
      break
  
  cv2.destroyAllWindows()
  print("AI Server (YOLO) : YOLO 스레드를 종료합니다.")

