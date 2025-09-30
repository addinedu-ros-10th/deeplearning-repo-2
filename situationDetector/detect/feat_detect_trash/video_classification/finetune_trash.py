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

# --- 1. 사용자 정의 데이터셋 클래스 (변경 없음) ---
class TrashDataset(Dataset):
    def __init__(self, root_dir, clip_len=8, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset()
        print(f"Found {len(self.samples)} videos in {root_dir}. Classes: {self.classes}")

    def _find_classes(self, dir):
        classes = sorted([d.name for d in os.scandir(dir) if d.is_dir()])
        if not classes: raise FileNotFoundError(f"Couldn't find any class folder in {dir}.")
        return classes, {cls_name: i for i, cls_name in enumerate(classes)}

    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for video_name in os.listdir(class_path):
                if video_name.endswith(('.mp4', '.avi')):
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
        
        if total_frames < self.clip_len:
            indices = np.arange(total_frames).tolist() + [total_frames - 1] * (self.clip_len - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            elif len(frames) > 0:
                frames.append(frames[-1])
        cap.release()
        
        while len(frames) < self.clip_len:
            frames.append(Image.new('RGB', (224, 224)))

        return self.transform(frames), label

# --- 2. 데이터 변환 함수 (변경 없음) ---
def get_transforms(train=False):
    transform_list = [transforms.Resize((224, 224))]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]))
    frame_transform = transforms.Compose(transform_list)
    return lambda frames: torch.stack([frame_transform(frame) for frame in frames]).permute(1, 0, 2, 3)

# --- 3. 키보드 인터럽트 (변경 없음) ---
STOP_REQUESTED = False
def on_press(key):
    global STOP_REQUESTED
    try:
        if key.char == ']':
            print("\n>>> ']' 키 입력 감지! 현재 에포크 완료 후 훈련을 중단합니다... <<<")
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
    
    DATASET_FINETUNE_ROOT = '/home/momo/Desktop/trash_dataset/dataset_finetune/'
    PRETRAINED_MODEL_PATH = 'i3d_hmdb51_pretrained.pth'
    
    NUM_CLASSES = 2
    BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-5
    CLIP_LEN = 8

    # --- 데이터셋 및 로더 ---
    print("Loading finetuning datasets...")
    train_dataset = TrashDataset(root_dir=os.path.join(DATASET_FINETUNE_ROOT, 'train'), clip_len=CLIP_LEN, transform=get_transforms(train=True))
    valid_dataset = TrashDataset(root_dir=os.path.join(DATASET_FINETUNE_ROOT, 'valid'), clip_len=CLIP_LEN, transform=get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Datasets loaded.")

    # --- 사전 훈련된 I3D 모델 로딩 및 수정 ---
    print("Loading I3D model architecture...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, trust_repo=True)
    model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=51)
    
    try:
        print(f"Loading pre-trained weights from {PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Pre-trained model '{PRETRAINED_MODEL_PATH}' not found. Please run train_i3d.py first.")
        return
        
    print("Modifying the final classifier for 2 classes (dumping, normal)...")
    model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 훈련 루프 ---
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
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
            # --- !!! 여기가 핵심 수정 사항 !!! ---
            # model.state_dict() 대신 model 전체를 저장합니다.
            torch.save(model, 'trash_detector_i3d_full_model.pt')
            best_accuracy = accuracy
        
        if STOP_REQUESTED:
            print("Training interrupted by user.")
            break

    print('Finished Finetuning')
    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    listener.join()
    
if __name__ == '__main__':
    main()