import cv2
import json
import time
import queue
import threading
from ultralytics import YOLO

FIRE_MODEL_PATH = "situationDetector/detect/feat_detect_fire/best.pt"

def run_fire_detect(analysis_frame_queue: queue.Queue, 
                    aggregation_queue :queue.Queue,     # 취합 큐
                    analyzer_name : str,                # 모델 이름
                    shutdown_event: threading.Event):
  """
  1. analysis_frame_queue에서 1 프레임 가져옴 (get)
    - (frame_count, frame_time, frame) : (프레임 카운트, 프레임 시간, 프레임 원본 데이터)
  2. YOLO 실행
    - results = model(frame, verbose=False)
  3. 결과 데이터 results에서 필요한 데이터 추출하여 6가지 기능 AI_JSON Schema, RawData_JSON Schema에 맞는 형식 생성
    - detection_data = {
        
      }
    - result_package = {
      
      }
  4. aggregation_queue에 데이터 추가 작업 put(result_package)
  """
  print("situationDetector (YOLO) : YOLO 스레드 시작, 모델 로드")
  
  model = YOLO(FIRE_MODEL_PATH)
  
  # 프로그램이 종료되지 않은 동안
  while not shutdown_event.is_set():
    try:
      # # 큐에서 프레임을 가져오고, timeout을 설정하여 blocking방지 및 종료신호 확인
      # frame_count, frame_time, frame = analysis_frame_queue.get(timeout=1.0)

      # (수정) 큐에서 딕셔너리 형태로 데이터 가져오기
      input_data = analysis_frame_queue.get(timeout=1.0)
      
      # 딕셔너리에서 각 데이터 추출
      frame = input_data.get("frame")
      frame_count = input_data.get("frame_count")
      frame_time = input_data.get("timestamp")
      patrol_number = input_data.get("patrol_number")

      # 현재 프레임 카운트 정보가 없으면 처리하지 않음
      if not frame_count:
        continue

      # YOLO 실행
      results = model(frame, verbose=False, conf=0.6)
      
      detection_list = []
      
      # 감지된 모든 객체 (results[0].boxes의 모든 객체를 detection에 저장)
      for box in results[0].boxes.data:
        # 텐서값 -> 파이썬 데이터 값
        x1, y1, x2, y2, conf, cls = box
        
        class_name = None
        if int(cls) == 0:
          class_name = "detect_fire"
        if int(cls) == 1:
          class_name = "detect_fire_danger_smoke"
        if int(cls) == 2:
          class_name = "detect_fire_general_smoke"

        if not class_name:
          print(f"situationDetector (YOLO) : 모델 클래스 오류 발생")
        
        detection_data = {
          "class_id": int(cls),                    ## 화재감지 이벤트 ID
          "class_name": class_name,      ## 화재 감지 이벤트
          "confidence": float(conf),                 ## detection 신뢰도
          "bbox": {                         ## Box 표기 위치
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
          }
        }
        detection_list.append(detection_data)
      
      result_package = {
        "detection" : detection_list, # 감지된 객체 정보 리스트 추가
        "timestamp" : frame_time,
        "analyzer_name" : analyzer_name,
        "detection_count" : len(detection_list), # 감지된 객체의 수
        "patrol_number" : patrol_number, # 순찰차 이름
      }
      
      # print(result_package)
      
      try:
        # aggregation_queue 추가 타입 : json
        aggregation_queue.put(result_package)
      except queue.Full:
        print("situationDetector (YOLO) : DB 큐가 가득 참 / 분석 결과 버립")
      
    except queue.Empty:
      # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
      continue
    except Exception as e:
      print(f"situationDetector (YOLO) : 처리 중 오류 발생: {e}")
      break
  
  cv2.destroyAllWindows()
  print("situationDetector (YOLO) : YOLO 스레드 종료")