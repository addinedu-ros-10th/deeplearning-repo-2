import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# --- 경로 설정 ---
PROCESSED_TRAIN_DIR = '/home/momo/Desktop/video/processed_data/train'
PROCESSED_TEST_DIR = '/home/momo/Desktop/video/processed_data/test'

# ▼▼▼▼▼ [변경] 모든 클립의 길이를 통일할 프레임 수 ▼▼▼▼▼
FIXED_FRAMES = 64
# ▲▲▲▲▲ [변경] 여기까지 ▲▲▲▲▲

# --- 1. 데이터 로더 만들기 ---
class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        video_data = np.load(file_path)
        label = int(self.file_list[idx].split('_label_')[1].split('.npy')[0])

        # ▼▼▼▼▼ [변경] 프레임 수를 고정 길이로 맞추는 로직 추가 ▼▼▼▼▼
        num_frames = video_data.shape[0]
        if num_frames > FIXED_FRAMES:
            # 프레임 수가 더 많으면 중간 부분을 잘라 사용
            start = (num_frames - FIXED_FRAMES) // 2
            video_data = video_data[start : start + FIXED_FRAMES]
        elif num_frames < FIXED_FRAMES:
            # 프레임 수가 더 적으면 마지막 프레임을 복사하여 채움
            padding_needed = FIXED_FRAMES - num_frames
            padding = np.tile(video_data[-1:], (padding_needed, 1, 1, 1))
            video_data = np.concatenate([video_data, padding], axis=0)
        # ▲▲▲▲▲ [변경] 여기까지 ▲▲▲▲▲

        video_tensor = torch.tensor(video_data, dtype=torch.float32).permute(3, 0, 1, 2)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return video_tensor, label_tensor

train_dataset = VideoDataset(PROCESSED_TRAIN_DIR)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# --- 2. 모델 아키텍처 정의 ---
class ActionClassifier(nn.Module):
    def __init__(self):
        super(ActionClassifier, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # ▼▼▼▼▼ [변경] 고정된 프레임 수에 맞춰 입력 크기 정확히 계산 ▼▼▼▼▼
        # conv와 pool을 거친 후의 데이터 크기 계산
        # 프레임: 64 -> 32
        # 높이/너비: 224 -> 112
        final_feature_size = 16 * (FIXED_FRAMES // 2) * (224 // 2) * (224 // 2)
        self.fc1 = nn.Linear(final_feature_size, 2) # 2는 클래스 수 (dump, non-dump)
        # ▲▲▲▲▲ [변경] 여기까지 ▲▲▲▲▲

    def forward(self, x):
        x = self.pool(self.relu(self.conv3d(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

# --- 3. 학습 루프 ---
model = ActionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print("🚀 모델 학습을 시작합니다...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for videos, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("✅ 모델 학습이 완료되었습니다.")

# --- 4. 모델 저장 ---
MODEL_SAVE_PATH = '/home/momo/Desktop/video/action_model.pt'
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"학습된 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")