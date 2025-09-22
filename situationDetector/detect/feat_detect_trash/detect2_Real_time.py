import torch
import cv2
from PIL import Image
from collections import deque
import mediapipe as mp
# 'train.py' 파일이 존재하고, ActionClassifier와 transform이 정의되어 있다고 가정합니다.
# 만약 해당 파일이 없다면, 실행 전 이 부분을 실제 모델 클래스와 변환 함수에 맞게 수정해야 합니다.
from train import ActionClassifier, transform

# --- 1. 설정값 ---
MODEL_PATH = 'trash_dumping_classifier.pth'
# ★★★ 실시간 감지를 위한 설정 수정 ★★★
# INPUT_VIDEO_PATH와 OUTPUT_VIDEO_PATH는 더 이상 필요하지 않습니다.

# ★★★ 최적화 및 디스플레이 설정 ★★★
# 웹캠 성능에 따라 해상도를 낮추면 더 부드럽게 동작할 수 있습니다. (예: (1280, 720))
DISPLAY_RESOLUTION = (1920, 1080)
FRAME_INTERVAL = 4  # 4프레임마다 한 번씩만 AI 모델을 실행

CLIP_LENGTH = 16
CONFIDENCE_THRESHOLD = 0.8
# ★★★ CUDA 오류 해결을 위해 장치를 CPU로 고정 ★★★
DEVICE = torch.device('cpu')

# --- 2. 모델 및 비디오 초기화 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print(f"'{MODEL_PATH}'에서 훈련된 모델을 불러옵니다...")
model = ActionClassifier()
# 모델 파일이 없을 경우 예외 처리
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    print("스크립트를 실행하기 전에 모델 파일을 현재 디렉토리에 위치시켜 주세요.")
    exit()
model.to(DEVICE)
model.eval()
print("모델 로딩 완료.")

# ★★★ 웹캠을 비디오 소스로 사용 ★★★
cap = cv2.VideoCapture(0)  # 0번 카메라(기본 웹캠)를 엽니다.
if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

print("\n실시간 무단투기 감시를 시작합니다. 종료하려면 'q' 키를 누르세요.")

# --- 3. 실시간 비디오 처리 및 추론 ---
frames_buffer = deque(maxlen=CLIP_LENGTH)
last_prob = 0.0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
        break

    frame_count += 1

    # ★★★ 실시간 디스플레이를 위해 원본 프레임을 리사이즈 ★★★
    frame_resized = cv2.resize(frame, DISPLAY_RESOLUTION)
    
    # AI 모델 입력용으로 사용할 RGB 프레임 생성 및 버퍼에 추가
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frames_buffer.append(frame_rgb)

    # 프레임 간격을 두고, 버퍼가 충분히 찼을 때만 AI 분석 실행
    if frame_count % FRAME_INTERVAL == 0 and len(frames_buffer) == CLIP_LENGTH:
        clip_pil_images = [Image.fromarray(f) for f in frames_buffer]
        clip_tensors = [transform(img) for img in clip_pil_images]
        clip_tensor = torch.stack(clip_tensors, dim=1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(clip_tensor)
            prob = torch.sigmoid(output.squeeze()).item()
            last_prob = prob

    # 최종 출력 프레임은 리사이즈된 프레임을 사용
    output_frame = frame_resized.copy() # 원본에 영향을 주지 않도록 복사본 사용

    # 투기 확률이 높을 때 Bounding Box 그리기
    if last_prob > CONFIDENCE_THRESHOLD:
        results = pose.process(frame_rgb) # AI 분석에 사용된 RGB 프레임을 재사용
        if results.pose_landmarks:
            h, w, _ = output_frame.shape
            landmarks = results.pose_landmarks.landmark
            x_min = min([lm.x for lm in landmarks]) * w
            y_min = min([lm.y for lm in landmarks]) * h
            x_max = max([lm.x for lm in landmarks]) * w
            y_max = max([lm.y for lm in landmarks]) * h
            # 바운딩 박스를 조금 더 크게 그려서 사람을 잘 포함하도록 함
            cv2.rectangle(output_frame, (int(x_min) - 20, int(y_min) - 20), (int(x_max) + 20, int(y_max) + 20), (0, 0, 255), 3)

    # 확률 텍스트 표시
    text = f"Dumping Prob: {last_prob:.2f}"
    color = (0, 0, 255) if last_prob > CONFIDENCE_THRESHOLD else (0, 255, 0)
    cv2.putText(output_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # ★★★ 처리된 프레임을 화면에 표시 ★★★
    cv2.imshow('Real-time Dumping Detection', output_frame)

    # 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 종료 및 리소스 해제 ---
print("\n--- 프로그램을 종료합니다 ---")
cap.release()
pose.close()
cv2.destroyAllWindows()

