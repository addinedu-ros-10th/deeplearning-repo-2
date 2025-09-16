import cv2
import mediapipe as mp
import numpy as np
import datetime

# --- (MediaPipe 및 모델 설정은 위 답변과 동일) ---

cap = cv2.VideoCapture(0)

# 녹화 상태를 관리할 변수
is_recording = False
video_writer = None
recording_end_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- (MediaPipe 자세 추정 및 LSTM 예측 로직) ---
    # 예시: 'is_dumping_detected' 변수가 예측 결과라고 가정
    is_dumping_detected = False # 실제로는 모델 예측 결과(True/False)가 들어갑니다.
    
    # 1. 무단투기 감지 & 녹화가 진행 중이 아닐 때
    if is_dumping_detected and not is_recording:
        is_recording = True
        # 현재 시간으로부터 10초 후를 녹화 종료 시간으로 설정
        recording_end_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
        
        # 고유 파일 이름으로 VideoWriter 객체 생성
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"detection_{current_time}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(file_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        
        print(f"[{current_time}] 무단투기 감지! 녹화를 시작합니다: {file_name}")

    # 2. 녹화가 진행 중일 때
    if is_recording:
        # 프레임을 비디오 파일에 저장
        video_writer.write(frame)
        cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 설정된 종료 시간이 되면 녹화 종료
        if datetime.datetime.now() >= recording_end_time:
            is_recording = False
            video_writer.release()
            video_writer = None
            print("녹화를 종료합니다.")

    cv2.imshow('Dumping Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 프로그램 종료 시 녹화 중이었다면 마무리
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()