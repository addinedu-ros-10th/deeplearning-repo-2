import time
import torch
import torchvision
import cv2
from PIL import Image
from train import get_model
import torchvision.transforms.functional as F

def run_webcam_inference():
    # --- 환경 설정 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3

    CLASSES = [
        'BACKGROUND',
        'human',
        'trash',
    ]
    
    # --- 모델 불러오기 ---
    print("Loading model...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 웹캠 초기화 ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed... Press 'q' to quit.")

    # --- 실시간 추론 루프 ---
    prev_time = 0
    frame_counter = 0
    skip_frames = 2.5
    last_prediction = None
    
    # !!! 경고 상태를 루프 밖에서 관리 !!!
    warning_active = False
    last_warning_time = 0
    warning_duration = 2  # 경고 메시지 유지 시간 (초)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        frame_counter += 1
        
        if frame_counter % skip_frames == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img_tensor)
                last_prediction = prediction
        
        if last_prediction is not None:
            human_detected = False
            trash_detected = False
            
            for box, label, score in zip(last_prediction[0]['boxes'], last_prediction[0]['labels'], last_prediction[0]['scores']):
                if score > 0.6:
                    box = box.cpu().numpy().astype(int)
                    label_index = label.cpu().item()
                    class_name = CLASSES[label_index]
                    
                    if class_name == 'human':
                        human_detected = True
                        color = (0, 255, 0)
                    elif class_name == 'trash':
                        trash_detected = True
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # !!! 경고 상태 업데이트 로직 !!!
            if human_detected and trash_detected:
                warning_active = True
                last_warning_time = current_time # 경고가 마지막으로 감지된 시간 기록

        # !!! 경고 메시지 표시 로직 !!!
        # 경고가 활성화 상태이고, 마지막 경고 시간으로부터 3초가 지나지 않았다면 메시지 표시
        if warning_active and (current_time - last_warning_time < warning_duration):
            warning_text = "Warning: Illegal Dumping Detected!"
            cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            warning_active = False # 경고 메시지 유지 시간 지나면 경고 비활성화

        # --- FPS 계산 및 표시 ---
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Illegal Dumping Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 자원 해제 ---
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    run_webcam_inference()