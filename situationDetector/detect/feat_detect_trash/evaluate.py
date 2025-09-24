import os
import torch
import torchvision
from train import CustomDataset, get_transform, collate_fn, get_model # 기존 train.py에서 필요한 함수들을 가져옵니다.
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate():
    # --- 환경 설정 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 평가할 데이터셋 경로
    DATASET_PATH = '/home/momo/Desktop/dataset/Illegal dumping detection.v2i.coco'
    VALID_DIR = os.path.join(DATASET_PATH, 'valid')
    VALID_ANNOTATIONS = os.path.join(VALID_DIR, '_annotations.coco.json')

    num_classes = 3 # 훈련 시와 동일하게 설정

    # --- 데이터셋 및 모델 불러오기 ---
    dataset_valid = CustomDataset(root=VALID_DIR, annotation_file=VALID_ANNOTATIONS, transforms=None) # 평가는 원본 이미지로 진행
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    model = get_model(num_classes)
    # 저장된 최고 성능의 가중치를 불러옵니다.
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval() # 반드시 평가 모드로 설정

    # --- mAP 계산기 초기화 ---
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in data_loader_valid:
            # 이미지 전처리 및 device로 이동
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 모델 예측
            predictions = model(images)
            
            # mAP 계산기에 예측값과 실제 정답값 추가
            metric.update(predictions, targets)

    # --- 최종 mAP 결과 출력 ---
    results = metric.compute()
    print("--- mAP Results ---")
    print(results)

if __name__ == "__main__":
    evaluate()