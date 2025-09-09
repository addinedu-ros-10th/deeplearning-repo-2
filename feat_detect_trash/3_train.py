import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm # 학습 진행률을 보여주는 라이브러리

# --- Dataset, Transform, Model 클래스는 이전에 작성한 것과 동일 ---

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []
        for label, class_name in enumerate(['non_dumping', 'dumping']):
            class_dir = os.path.join(self.root_dir, class_name)
            for clip_name in os.listdir(class_dir):
                clip_path = os.path.join(class_dir, clip_name)
                if len(os.listdir(clip_path)) >= clip_length:
                    self.samples.append((clip_path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        frame_files = sorted(os.listdir(clip_path))
        frames = []
        for i in range(self.clip_length):
            frame_path = os.path.join(clip_path, frame_files[i])
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        clip_tensor = torch.stack(frames, dim=1)
        return clip_tensor, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ActionClassifier(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2):
        super(ActionClassifier, self).__init__()
        self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        batch_size, _, seq_len, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size * seq_len, *x.shape[2:])
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        out = self.classifier(last_output)
        return out

# --- ★★★★★ 여기가 바로 최종 실행 블록입니다 ★★★★★ ---
if __name__ == '__main__':
    # --- 1. 하이퍼파라미터 및 설정 ---
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20 # 전체 데이터셋을 20번 반복 학습

    # GPU 사용 설정 (가능하면 GPU 사용, 없으면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용할 디바이스: {device}")

    # --- 2. 데이터 준비 (이전과 동일) ---
    print("데이터셋을 로딩합니다...")
    full_dataset = VideoClipDataset(root_dir='./clips', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"총 클립 수: {len(full_dataset)} | 훈련용: {len(train_dataset)} | 검증용: {len(val_dataset)}")

    # --- 3. 모델, 손실 함수, 옵티마이저 정의 ---
    model = ActionClassifier().to(device)
    # 손실 함수: 이진 분류 문제이므로 BCEWithLogitsLoss 사용 (Sigmoid 함수 포함)
    criterion = nn.BCEWithLogitsLoss()
    # 옵티마이저: Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 훈련 루프 ---
    print("\n모델 훈련을 시작합니다...")
    for epoch in range(NUM_EPOCHS):
        # ** 훈련 모드 **
        model.train()
        train_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [훈련]"):
            # 데이터를 설정한 디바이스로 이동
            clips, labels = clips.to(device), labels.to(device)
            
            optimizer.zero_grad() # 그래디언트 초기화
            outputs = model(clips) # 모델 예측
            loss = criterion(outputs.squeeze(), labels) # 손실 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            
            train_loss += loss.item()

        # ** 검증 모드 **
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # 그래디언트 계산 비활성화
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # 정확도 계산
                predicted = torch.sigmoid(outputs.squeeze()) > 0.5
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 에포크 결과 출력
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {accuracy:.2f}%")

    print("\n--- 훈련 완료! ---")

    # --- 5. 훈련된 모델 저장 ---
    torch.save(model.state_dict(), 'trash_dumping_classifier.pth')
    print("훈련된 모델을 'trash_dumping_classifier.pth' 파일로 저장했습니다.")