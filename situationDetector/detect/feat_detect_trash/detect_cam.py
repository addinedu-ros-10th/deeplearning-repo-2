import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from pynput import keyboard
from PIL import Image
import argparse
import collections

# --- 1. 데이터 변환 함수 (변경 없음) ---
def get_transforms():
    transform_list = [transforms.Resize((224, 224))]
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]))
    frame_transform = transforms.Compose(transform_list)
    return lambda frames: torch.stack([frame_transform(frame) for frame in frames]).permute(1, 0, 2, 3)

# --- 2. 메인 실행 부분 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Trash Dumping Detection from Webcam")
    parser.add_argument('--cam', type=int, default=0, help='Camera index to use (e.g., 0 for default webcam).')
    parser.add_argument('--model_path', type=str, default='trash_detector_i3d_full_model.pt', help='Path to the trained full model file.')
    args = parser.parse_args()

    # --- 환경 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    CLASS_NAMES = ['dumping', 'normal']
    CLIP_LEN = 8
    
    # --- [핵심 수정] 모델 전체를 파일에서 직접 로드 ---
    print(f"Loading full model from '{args.model_path}'...")
    try:
        # [수정] torch.load()에 weights_only=False 인자를 추가하여 #######################
        # 모델 구조와 가중치를 모두 불러오도록 명시합니다.
        model = torch.load(args.model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    model = model.to(device)
    model.eval() # 추론 모드로 설정

    # --- 웹캠 초기화 (변경 없음) ---
    print("Starting webcam...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {args.cam}")
        exit()

    frames_buffer = collections.deque(maxlen=CLIP_LEN)
    transform = get_transforms()
    
    print("Press 'q' to quit.")

    # --- 실시간 추론 루프 (변경 없음) ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frames_buffer.append(pil_frame)

        prediction_text = "..."
        text_color = (255, 255, 255)

        if len(frames_buffer) == CLIP_LEN:
            video_tensor = transform(list(frames_buffer))
            video_tensor = video_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(video_tensor)
                
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            prediction_text = f"{predicted_class.upper()}: {confidence_score:.1f}%"
            
            if predicted_class == 'dumping' and confidence_score > 70:
                text_color = (0, 0, 255)
            else:
                text_color = (0, 255, 0)

        cv2.putText(display_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.imshow('Trash Dumping Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")





# 위 코드는 save()로 생성된 모델을 구동 시킴
# torch.save : 직렬화된 객체를 디스크에 저장합니다. 이 함수는 Python의 pickle 유틸리티를 사용하여 직렬화합니다. 이 함수를 사용하여 모든 종류의 객체의 모델, 텐서, 딕셔너리를 저장할 수 있습니다.
# torch.load : pickle 의 언피클링 기능을 사용하여 피클링된 객체 파일을 메모리로 역직렬화합니다. 또한 이 함수는 장치가 데이터를 로드하는 데 도움을 줍니다( 장치 간 모델 저장 및 로드 참조 ).
# torch.nn.Module.load_state_dict : 역직렬화된 state_dict를 사용하여 모델의 매개변수 사전을 로드합니다 . state_dict 에 대한 자세한 내용은 state_dict란 무엇인가요?를 참조하세요 .
# 아래 코드는 load_state_dict 로 생성된 모델을 구동 시킴
# .pt , .pth  는 구조에 영향은 주지 않으나 .pt 를 save 함수를 사용한 모델로 약속한 것으로 보임


# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import transforms
# import cv2
# import numpy as np
# from PIL import Image
# import argparse
# import collections

# # --- 1. 데이터 변환 함수 (훈련 때와 완벽히 동일해야 함) ---
# def get_transforms():
#     transform_list = [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
#     ]
#     frame_transform = transforms.Compose(transform_list)
#     return lambda frames: torch.stack([frame_transform(frame) for frame in frames]).permute(1, 0, 2, 3)

# # --- 2. 메인 실행 부분 ---
# if __name__ == '__main__':
#     # --- 터미널에서 인자를 받기 위한 설정 ---
#     parser = argparse.ArgumentParser(description="Real-time Trash Dumping Detection from Webcam")
#     parser.add_argument('--cam', type=int, default=0, help='Camera index to use (e.g., 0 for default webcam).')
#     parser.add_argument('--model_path', type=str, default='trash_detector_i3d_full_model.pt', help='Path to the trained model weights.')
#     args = parser.parse_args()

#     # --- 환경 설정 ---
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     NUM_CLASSES = 2
#     CLASS_NAMES = ['dumping', 'normal']
#     CLIP_LEN = 8 # 8 프레임을 모아서 한 번씩 예측
    
#     # --- 모델 생성 및 가중치 로드 ---
#     print("Loading model architecture...")
#     model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, trust_repo=True)
#     model.blocks[5].proj = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES)
    
#     print(f"Loading trained weights from '{args.model_path}'...")
#     try:
#         model.load_state_dict(torch.load(args.model_path, map_location=device))
#     except FileNotFoundError:
#         print(f"Error: Model file not found at '{args.model_path}'")
#         exit()

#     model = model.to(device)
#     model.eval() # 추론 모드로 설정

#     # --- 웹캠 초기화 ---
#     print("Starting webcam...")
#     cap = cv2.VideoCapture(args.cam)
#     if not cap.isOpened():
#         print(f"Error: Cannot open camera with index {args.cam}")
#         exit()

#     # 프레임을 저장할 버퍼 (큐) 생성
#     frames_buffer = collections.deque(maxlen=CLIP_LEN)
#     transform = get_transforms()
    
#     print("Press 'q' to quit.")

#     # --- 실시간 추론 루프 ---
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame. Exiting...")
#             break

#         # 화면에 보여줄 원본 프레임 복사
#         display_frame = frame.copy()

#         # 전처리를 위해 프레임 포맷 변경 (BGR -> RGB) 및 버퍼에 추가
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_frame = Image.fromarray(frame_rgb)
#         frames_buffer.append(pil_frame)

#         prediction_text = "..."
#         text_color = (255, 255, 255) # 흰색 (기본)

#         # 버퍼에 충분한 프레임이 쌓이면 추론 실행
#         if len(frames_buffer) == CLIP_LEN:
#             # --- 비디오 전처리 ---
#             video_tensor = transform(list(frames_buffer))
#             video_tensor = video_tensor.unsqueeze(0).to(device)
            
#             # --- 추론 실행 ---
#             with torch.no_grad():
#                 outputs = model(video_tensor)
                
#             # --- 결과 해석 ---
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, predicted_idx = torch.max(probabilities, 1)
            
#             predicted_class = CLASS_NAMES[predicted_idx.item()]
#             confidence_score = confidence.item() * 100
            
#             prediction_text = f"{predicted_class.upper()}: {confidence_score:.1f}%"
            
#             if predicted_class == 'dumping' and confidence_score > 58: # 70% 이상 확신할 때만 빨간색
#                 text_color = (0, 0, 255) # 빨간색
#             else:
#                 text_color = (0, 255, 0) # 초록색

#         # --- 화면에 결과 표시 ---
#         cv2.putText(display_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
#         cv2.imshow('Trash Dumping Detection', display_frame)

#         # 'q' 키를 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # --- 자원 해제 ---
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam stopped.")