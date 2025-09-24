import cv2
import queue
import time
import threading
from typing import List

def receive_cv_video(analyzer_input_queue : List[queue.Queue],
                    shutdown_event : threading.Event):
  # Create a VideoCapture object. 0 refers to the default webcam.
  # If you have multiple webcams, you might need to try 1, 2, etc.
  # cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")
  cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")
  # cap = cv2.VideoCapture("http://192.168.0.149:5000/stream?src=0")

  # Check if the webcam is opened successfully
  if not cap.isOpened():
    print("situationDetector (TCP Broadcast receiver) : 순찰차 카메라 초기화 실패")
    return

  print("situationDetector (TCP Broadcast receiver) : 순찰차 카메라 초기화 성공")

  frame_count = 0 # 현재 프레임

  while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
      print("situationDetector (TCP Broadcast receiver) : 프레임을 가져오는데 실패했습니다.")
      break

    frame_count += 1
    frame_time = time.time()

    # analyzer_input_queue에 프레임 저장
    for q in analyzer_input_queue:
      if q.full():
        q.get() # 큐가 꽉 차있으면 이전 프레임을 버리고 새 것으로 교체 작업
      data_packet = (frame_count, frame_time, frame)
      q.put(data_packet)

    # Display the captured frame
    # cv2.imshow('Webcam Feed', frame)

    # Wait for a key press. If 'q' is pressed, exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the VideoCapture object and destroy all OpenCV windows
  cap.release()
  cv2.destroyAllWindows()





# ----------------------------------------------------------------------------------


def test():
  # Create a VideoCapture object. 0 refers to the default webcam.
  # If you have multiple webcams, you might need to try 1, 2, etc.
  cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")
  # cap = cv2.VideoCapture(0)

  # Check if the webcam is opened successfully
  if not cap.isOpened():
    print("Error: Could not open webcam.")
    return

  print("Webcam opened successfully. Press 'q' to quit.")

  while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
      print("Error: Failed to grab frame.")
      break

    # # analyzer_input_queue에 프레임 저장
    # for q in analyzer_input_queue:
    #   if q.full():
    #     q.get() # 큐가 꽉 차있으면 이전 프레임을 버리고 새 것으로 교체 작업
    #   q.put(frame.copy())

    # Display the captured frame
    cv2.imshow('Webcam Feed', frame)

    # Wait for a key press. If 'q' is pressed, exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the VideoCapture object and destroy all OpenCV windows
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    test()