#!train.py
import torch
import torchvision
import os
import json
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# 1. 데이터셋 클래스 정의
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        
        # 모든 이미지 ID를 가져옵니다.
        all_ids = list(sorted(self.coco.imgs.keys()))
        
        # 어노테이션이 있는 이미지 ID만 필터링하여 저장할 리스트
        valid_ids = []
        for img_id in all_ids:
            # 해당 이미지의 어노테이션 ID를 가져옵니다.
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            # 어노테이션이 하나 이상 존재하는 경우에만 유효한 ID로 추가합니다.
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
        
        self.ids = valid_ids

    def __getitem__(self, index):
        # 이미지 ID 가져오기
        img_id = self.ids[index]
        
        # 이미지 파일 경로 가져오기
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")

        # 해당 이미지에 대한 어노테이션 정보 가져오기
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 바운딩 박스 정보 추출
        boxes = []
        for ann in anns:
            # bbox: [x, y, width, height] -> [x1, y1, x2, y2]
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        # 레이블(클래스 ID) 정보 추출
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # target 딕셔너리 생성
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        # 데이터 변환 (transform) 적용
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # return img, target
        # 이미지를 텐서로 변환
        img = F.to_tensor(img)

        # (선택) 데이터 증강이 있다면 여기서 적용
        # for t in self.transforms:
        #     img, target = t(img, target)

        return img, target


    def __len__(self):
        return len(self.ids)


def get_transform(train):
    transforms = []
    # transforms.append(ToTensor()) 1차 수정
    # 훈련 시에는 데이터 증강을 추가할 수 있습니다.
    # if train:
    #     transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    # return torchvision.transforms.Compose(transforms)
    return transforms
    

# 데이터셋 경로와 작업 경로 설정
DATASET_PATH = '/home/momo/Desktop/dataset/Illegal dumping detection.v2i.coco'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VALID_DIR = os.path.join(DATASET_PATH, 'valid')
TRAIN_ANNOTATIONS = os.path.join(TRAIN_DIR, '_annotations.coco.json')
VALID_ANNOTATIONS = os.path.join(VALID_DIR, '_annotations.coco.json')

# 3. 모델 불러오기 및 수정
def get_model(num_classes):
    # 사전 훈련된 Faster R-CNN 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # 분류기(classifier)의 입력 특성 수를 가져옴
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 사전 훈련된 모델의 헤더를 새로운 헤더로 교체
    # num_classes는 배경(0)을 포함해야 함
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# 4. 훈련 및 검증을 위한 유틸리티 함수
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # GPU 사용 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 클래스 개수 설정 (배경 포함)
    num_classes = 3 

    # 데이터셋 생성
    dataset_train = CustomDataset(root=TRAIN_DIR, annotation_file=TRAIN_ANNOTATIONS, transforms=get_transform(train=True))
    dataset_valid = CustomDataset(root=VALID_DIR, annotation_file=VALID_ANNOTATIONS, transforms=get_transform(train=False))

    # 데이터 로더 생성
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    # 모델 가져오기
    model = get_model(num_classes)
    model.to(device)

    # 옵티마이저 및 학습률 스케줄러 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # --- 조기 종료를 위한 변수 설정 ---
    # 가장 좋았던 검증 손실 값을 저장 (초기값은 무한대로 설정)
    best_valid_loss = float('inf') 
    # 검증 손실이 개선되지 않아도 몇 에포크까지 기다릴지 설정
    patience = 3 
    patience_counter = 0

    # 훈련 에포크 설정 (조기 종료되므로 충분히 길게 설정해도 좋습니다)
    num_epochs = 20 

    for epoch in range(num_epochs):
        # --- 훈련 단계 ---
        model.train()
        i = 0
        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {losses.item()}")
        
        # 학습률 업데이트
        lr_scheduler.step()

        print(f"Epoch {epoch} training finished. Starting validation...")

        # --- 검증 단계 ---
        model.eval() # 모델을 평가 모드로 전환
        total_valid_loss = 0.0
        
        with torch.no_grad(): # 그래디언트 계산 비활성화
            for images, targets in data_loader_valid:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # torchvision 모델은 eval() 모드에서 loss를 반환하지 않으므로,
                # loss 계산을 위해 잠시 train() 모드로 전환했다가 다시 eval()로 돌립니다.
                model.train()
                loss_dict = model(images, targets)
                model.eval()
                
                losses = sum(loss for loss in loss_dict.values())
                total_valid_loss += losses.item()
        
        avg_valid_loss = total_valid_loss / len(data_loader_valid)
        print(f"Epoch: {epoch}, Validation Loss: {avg_valid_loss}")

        # --- 조기 종료 및 최고 성능 모델 저장 로직 ---
        if avg_valid_loss < best_valid_loss:
            print(f"Validation loss decreased ({best_valid_loss:.4f} --> {avg_valid_loss:.4f}). Saving model...")
            best_valid_loss = avg_valid_loss
            # 가장 성능이 좋은 모델의 가중치를 저장합니다.
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0 # 참을성 초기화
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break # 훈련 루프를 완전히 탈출합니다.

    print("Training complete. Best model saved to best_model.pth")

if __name__ == "__main__":
    main()