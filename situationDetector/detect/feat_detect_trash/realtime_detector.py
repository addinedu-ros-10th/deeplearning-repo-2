import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import threading
import time

# --- 1. 설정 및 모델 구조 정의 ---
MODEL_PATH = '/home/momo/Desktop/video/action_model.pt'
FIXED_FRAMES = 64 
CLASSES = {0: 'Normal', 1: 'Dumping Detected'}

# 학습 때 사용했던 모델 클래스
class ActionClassifier(nn.Module):
    def __init__(self):
        super(ActionClassifier, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        final_feature_size = 16 * (FIXED_FRAMES // 2) * (224 // 2) * (224 // 2)
        self.fc1 = nn.Linear(final_feature_size, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv3d(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

# --- 2. 모델 로드 ---
device = torch.device('cpu')
model = ActionClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ 모델 로딩 완료.")

# --- 3. 스레드 간 공유할 변수 ---
frame_buffer = deque(maxlen=FIXED_FRAMES)
prediction_text = "Initializing..."
lock = threading.Lock() # 변수 접근을 동기화하기 위한 Lock
is_running = True

# --- 4. 추론 스레드 함수 ---
def run_inference():
    global prediction_text, is_running, frame_buffer
    
    while is_running:
        if len(frame_buffer) == FIXED_FRAMES:
            # 버퍼 복사하여 추론 중 버퍼가 변경되어도 문제 없도록 함
            buffer_copy = list(frame_buffer)
            
            clip = np.array(buffer_copy)
            clip_tensor = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(clip_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                
                # Lock을 사용하여 공유 변수를 안전하게 업데이트
                with lock:
                    prediction_text = CLASSES[predicted_idx.item()]
        
        # 추론 스레드가 너무 빠르게 반복되는 것을 방지
        time.sleep(0.1) 

# --- 5. 메인 스레드 (카메라 및 디스플레이) ---
print("🚀 실시간 탐지를 시작합니다...")
cap = cv2.VideoCapture(0)

# 추론 스레드 시작
inference_thread = threading.Thread(target=run_inference, daemon=True)
inference_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 카메라를 열 수 없습니다.")
        break

    # 프레임 전처리 및 버퍼에 추가 (이 작업은 빠름)
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    frame_buffer.append(normalized_frame)

    # Lock을 사용하여 추론 스레드가 업데이트한 결과를 안전하게 읽어오기
    with lock:
        display_text = prediction_text
    
    color = (0, 255, 0) # 초록색 (Normal)
    if display_text == 'Dumping Detected':
        color = (0, 0, 255) # 빨간색 (Dumping)

    cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow('Real-time Dumping Detection (Threaded)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. 종료 처리 ---
print("종료 중...")
is_running = False # 추론 스레드 종료 신호
inference_thread.join(timeout=1) # 스레드가 종료될 때까지 잠시 대기
cap.release()
cv2.destroyAllWindows()