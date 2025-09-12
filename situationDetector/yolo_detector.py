"""
공유 큐에서 프레임을 가져와 YOLO 객체를 탐지하고 결과를 화면에 표시하는 데모 함수 파일
"""
import cv2
import queue
import threading
from ultralytics import YOLO

def detect_objects(frame_queue: queue.Queue, shutdown_event: threading.Event):
  print("AI Server (YOLO) : YOLO 스레드를 시작합니다. 모델을 로드합니다.")
  
  model = YOLO("situationDetector/best.pt")
  
  # 프로그램이 종료되지 않은 동안
  while not shutdown_event.is_set():
    try:
      # 큐에서 프레임을 가져오고, timeout을 설정하여 blocking방지 및 종료신호 확인
      frame = frame_queue.get(timeout=1.0)
      
      # YOLO 실행
      results = model(frame)
      
      annotated_frame = results[0].plot()
      
      # 이미지 표시
      cv2.imshow("Detector test", annotated_frame)
      
      # 'ESC' 키를 누르면 종료 이벤트를 설정하여 모든 스레드가 종료되도록 함
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