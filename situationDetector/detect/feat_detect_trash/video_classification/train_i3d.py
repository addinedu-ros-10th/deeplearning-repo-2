# train_i3d.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from pynput import keyboard
from PIL import Image

# --- 1. 비디오 데이터셋 클래스 정의 ---
class HMDB51Dataset(Dataset):
    def __init__(self, root_dir, clip_len=8, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset()
        print(f"Found {len(self.samples)} videos in {root_dir}.")

    def _find_classes(self, dir):
        classes = sorted([d.name for d in os.scandir(dir) if d.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {dir}.")
        return classes, {cls_name: i for i, cls_name in enumerate(classes)}

    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for video_name in os.listdir(class_path):
                    if video_name.endswith('.avi'):
                        video_path = os.path.join(class_path, video_name)
                        item = (video_path, self.class_to_idx[class_name])
                        dataset.append(item)
        return dataset
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 비디오 길이가 짧아도 에러가 나지 않도록 시작/끝 인덱스 보정
        start_frame = 0
        end_frame = total_frames - 1
        if total_frames < self.clip_len:
            indices = np.arange(total_frames).tolist() + [total_frames - 1] * (self.clip_len - total_frames)
        else:
            indices = np.linspace(start_frame, end_frame, self.clip_len, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            elif len(frames) > 0:
                frames.append(frames[-1]) # 읽기 실패 시 마지막 프레임 복제
        cap.release()
        
        # 아주 드물게 비디오를 전혀 읽지 못한 경우
        while len(frames) < self.clip_len:
            frames.append(Image.new('RGB', (224, 224)))

        if self.transform:
            video_tensor = self.transform(frames)

        return video_tensor, label

# --- 2. 데이터 변환 함수 ---
def get_transforms(train=False):
    transform_list = [
        transforms.Resize((224, 224)),
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    
    transform_list.append(transforms.ToTensor())
    # I3D 모델은 아래 정규화 값을 권장합니다.
    transform_list.append(transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]))
    
    frame_transform = transforms.Compose(transform_list)

    return lambda frames: torch.stack([frame_transform(frame) for frame in frames]).permute(1, 0, 2, 3)

# --- 3. 키보드 인터럽트 설정 ---
STOP_REQUESTED = False
def on_press(key):
    global STOP_REQUESTED
    try:
        if key.char == 's':
            print("\n>>> 's' 키 입력 감지! 현재 에포크 완료 후 훈련을 중단합니다... <<<")
            STOP_REQUESTED = True
            return False
    except AttributeError: pass

# --- 4. 메인 훈련 스크립트 ---
def main():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # --- 환경 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    DATASET_ROOT = '/home/momo/Desktop/fiftyone/hmdb51/' # 사용자 환경에 맞게 수정
    TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
    VALID_DIR = os.path.join(DATASET_ROOT, 'test')
    
    NUM_CLASSES = 51
    BATCH_SIZE = 2 # RTX 2060 (6GB) 환경에 맞춘 보수적인 설정
    EPOCHS = 30
    LEARNING_RATE = 1e-4 # 0.0001
    CLIP_LEN = 8

    # --- 데이터셋 및 로더 ---
    print("Loading datasets...")
    train_dataset = HMDB51Dataset(root_dir=TRAIN_DIR, clip_len=CLIP_LEN, transform=get_transforms(train=True))
    valid_dataset = HMDB51Dataset(root_dir=VALID_DIR, clip_len=CLIP_LEN, transform=get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Datasets loaded.")

    # --- 3D CNN (I3D) 모델 로딩 ---
    print("Loading I3D model pre-trained on Kinetics-400...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    # --- !!! 모델 구조를 확인하기 위한 디버깅 코드 추가 !!! ---
    print("--- Loaded Model Architecture ---")
    print(model)
    print("---------------------------------")
    # --- 이 부분은 확인 후 삭제하거나 주석 처리하세요 ---

    # 마지막 분류층을 HMDB51의 51개 클래스에 맞게 교체 (이 부분을 수정해야 함)
    # model.head.proj = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES) # <-- 에러 발생 지점
    model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES)

    model = model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 체크포인트 로딩 ---
    start_epoch, best_accuracy = 0, 0.0
    CHECKPOINT_PATH = 'i3d_checkpoint.pth'
    if os.path.exists(CHECKPOINT_PATH):
        print(f"'{CHECKPOINT_PATH}' 파일을 찾았습니다. 훈련을 재개합니다.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print(f"에포크 {start_epoch} 부터 훈련을 다시 시작합니다. (최고 정확도: {best_accuracy:.2f}%)")

    # --- 훈련 및 검증 루프 ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Training Loss: {running_loss / len(train_loader):.4f}")
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for videos, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} Validation Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            print(f"Accuracy improved ({best_accuracy:.2f}% --> {accuracy:.2f}%). Saving best model...")
            torch.save(model.state_dict(), 'i3d_hmdb51_pretrained.pth')
            best_accuracy = accuracy

        print("Saving checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': best_accuracy,
        }, CHECKPOINT_PATH)
        
        if STOP_REQUESTED:
            print("훈련 중단 요청에 따라 루프를 종료합니다.")
            break

    print('Finished Training')
    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    listener.join()
    
if __name__ == '__main__':
    main()