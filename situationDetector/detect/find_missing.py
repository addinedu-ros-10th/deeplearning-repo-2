import cv2
import json
import time
import queue
import threading
# from src
from situationDetector.detect.live_final import infer_once

def run_find_missing(analysis_frame_queue: queue.Queue, 
                    aggregation_queue :queue.Queue,     # 취합 큐
                    analyzer_name : str,                # 모델 이름
                    shutdown_event: threading.Event):
  print("situationDetector (YOLO) : find_missing 스레드 시작, 모델 로드")
  
  
  # 프로그램이 종료되지 않은 동안 처리 반복
  while not shutdown_event.is_set():
    try:

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
      
      results = infer_once(frame)
      
      detection_list = []
      
      for missing in results:
        detection_data = {
          "class_name": "detect_missing_person",
          "confidence" : missing['confidence'],
          "bbox" : {
            "x1" : missing['bbox'][0],
            "y1" : missing['bbox'][1],
            "x2" : missing['bbox'][2],
            "y2" : missing['bbox'][3],
          },
          "person_info" : {
            "name" : missing['name'],
            "birthday" : missing['birthday'],
            "mask_type" : missing['mask_type']
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
      
      # print(results) # 감지결과 출력 테스트 

        # print(result_package)
      
      try:
        # aggregation_queue 추가 타입 : json
        aggregation_queue.put(result_package)
      except queue.Full:
        print("situationDetector (YOLO) : find_missing DB 큐가 가득 참 / 분석 결과 버림")
      
    except queue.Empty:
      # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
      continue
    except Exception as e:
      print(f"situationDetector (YOLO) : find_missing 처리 중 오류 발생: {e}")
      break
  
  # cv2.destroyAllWindows()
  print("situationDetector (YOLO) : find_missing YOLO 스레드 종료")

"""
현재 프레임 매칭 결과: 
[
  {
    'name': 'choi', 
    'birthday': '19980317', 
    'mask_type': 'full', 
    'confidence': 0.6351401805877686, 
    'bbox': [
      435, 
      220, 
      534, 
      347
    ]
  }
]
"""

def run_find_missing_test():
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      
      matched_results = infer_once(frame)
      
      if matched_results:
        print(f"현재 프레임 매칭 결과: {matched_results}")
      else:
        print("매칭 없음")

      cv2.imshow("Real-time Face Match (q to quit)", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  finally:
    # 4. 루프 종료 시 리소스 자동 해제
    cap.release()
    cv2.destroyAllWindows()


'''
{
  'detection': {
    'feat_find_missing': [
      {
        'class_name': 'detect_missing_person',
        'confidence': 0.855461835861206, 
        'bbox': {
          331,
          205,
          342,
          173
        }, 
        'person_info': 
        {
          'name': 'choi',
          'birthday': '19980317',
          'mask_type': 'full'
        }
      }, 
      {
        'detection_count': 1
      }
    ]
  }, 
  'timestamp': '2025-09-29T16:38:55.374214', 
  'patrol_number': 1
}
'''