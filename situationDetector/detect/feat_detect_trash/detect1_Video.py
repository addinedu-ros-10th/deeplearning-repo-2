import os
import torch
import cv2
import numpy as np
from PIL import Image
from collections import deque
from train import ActionClassifier, transform
import mediapipe as mp
from tqdm import tqdm

# --- 1. 설정값 ---
MODEL_PATH = 'trash_dumping_classifier.pth'
INPUT_VIDEO_PATH = './data/174-6_cam02_dump02_place01_day_summer.mp4'
OUTPUT_VIDEO_PATH = 'detection_output_1080p.mp4' # 최종 결과 영상 파일 이름

# ★★★ 최적화 설정 ★★★
OUTPUT_RESOLUTION = (1920, 1080) # 출력 해상도를 Full HD로 지정
FRAME_INTERVAL = 4 # 4프레임마다 한 번씩만 AI 모델을 실행

CLIP_LENGTH = 16
CONFIDENCE_THRESHOLD = 0.8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. 모델 및 비디오 초기화 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print(f"'{MODEL_PATH}'에서 훈련된 모델을 불러옵니다...")
model = ActionClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("모델 로딩 완료.")

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: '{INPUT_VIDEO_PATH}' 영상을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ★★★ 출력 해상도에 맞춰 VideoWriter 초기화 ★★★
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, OUTPUT_RESOLUTION)

# --- 3. 비디오 처리 및 추론/저장 ---
frames_buffer = deque(maxlen=CLIP_LENGTH)
last_prob = 0.0
frame_count = 0

print(f"\n'{INPUT_VIDEO_PATH}' 영상을 분석하여 '{OUTPUT_VIDEO_PATH}' 파일로 저장합니다...")
for _ in tqdm(range(total_frames), desc="영상 처리 진행률"):
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # ★★★ 처리 효율을 위해 원본 프레임을 먼저 리사이즈 ★★★
    frame_resized = cv2.resize(frame, OUTPUT_RESOLUTION)

    # AI 분석은 지정된 간격(FRAME_INTERVAL)마다 실행
    if frame_count % FRAME_INTERVAL == 0:
        # 버퍼는 리사이즈된 프레임으로 채움 (AI 입력용)
        # 이 부분은 추론 시에만 필요하므로, 추론 시점의 최신 프레임들로 구성
        # 버퍼를 매 프레임 업데이트 하는 대신, 필요할 때만 생성
        current_pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
        temp_buffer = []
        # 현재 프레임 기준 이전 16개 프레임 읽기 (효율적이지는 않으나 개념 설명용)
        # 더 빠른 방법은 매 프레임 버퍼를 유지하는 것
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame_rgb)

        if len(frames_buffer) == CLIP_LENGTH:
            clip_pil_images = [Image.fromarray(f) for f in frames_buffer]
            clip_tensors = [transform(img) for img in clip_pil_images]
            clip_tensor = torch.stack(clip_tensors, dim=1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(clip_tensor)
                prob = torch.sigmoid(output.squeeze()).item()
                last_prob = prob

    # 최종 출력 프레임은 리사이즈된 프레임을 사용
    output_frame = frame_resized

    # 투기 확률이 높을 때 Bounding Box 그리기
    if last_prob > CONFIDENCE_THRESHOLD:
        # MediaPipe는 리사이즈된 프레임의 RGB 버전을 사용
        frame_rgb_for_pose = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb_for_pose)
        if results.pose_landmarks:
            h, w, _ = output_frame.shape
            landmarks = results.pose_landmarks.landmark
            x_min = min([lm.x for lm in landmarks]) * w
            y_min = min([lm.y for lm in landmarks]) * h
            x_max = max([lm.x for lm in landmarks]) * w
            y_max = max([lm.y for lm in landmarks]) * h
            cv2.rectangle(output_frame, (int(x_min) - 20, int(y_min) - 20), (int(x_max) + 20, int(y_max) + 20), (0, 0, 255), 3)

    # 확률 텍스트 표시
    text = f"Dumping Prob: {last_prob:.2f}"
    color = (0, 0, 255) if last_prob > CONFIDENCE_THRESHOLD else (0, 255, 0)
    cv2.putText(output_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # 처리된 프레임을 비디오 파일에 쓰기
    video_writer.write(output_frame)

# --- 4. 종료 및 리소스 해제 ---
print(f"\n--- 작업 완료! ---")
print(f"결과가 '{OUTPUT_VIDEO_PATH}' 파일에 저장되었습니다.")

cap.release()
video_writer.release()
pose.close()