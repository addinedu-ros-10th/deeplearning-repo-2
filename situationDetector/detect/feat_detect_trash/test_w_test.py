import time
import torch
import torchvision
import cv2
from PIL import Image
from train import get_model
import torchvision.transforms.functional as F
import logging
import datetime

# --- 로깅 설정 ---
# 로그 포맷 지정 (시간, 로그 레벨, 메시지)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO) # 로그 레벨을 INFO로 설정

# 1. 터미널(콘솔) 핸들러 추가
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# 2. 파일 핸들러 추가 (로그를 파일로 저장)
# 현재 시간을 기반으로 로그 파일 이름 생성
log_filename = f"detection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


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
    logging.info("Loading model...")
    model = get_model(num_classes)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error("Model file 'best_model.pth' not found. Please check the file path.")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return


    # --- 웹캠 초기화 ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    logging.info("Starting webcam feed... Press 'q' to quit.")

    # --- 실시간 추론 루프 ---
    prev_time = time.time()
    frame_counter = 0
    skip_frames = 2.5
    last_prediction = None
    
    # 경고 상태를 관리하는 변수
    warning_active = False
    last_warning_time = 0
    warning_duration = 2  # 경고 메시지 유지 시간 (초)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame from webcam.")
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
            
            # 매 프레임 탐지 로그 초기화
            detection_logs = []

            for box, label, score in zip(last_prediction[0]['boxes'], last_prediction[0]['labels'], last_prediction[0]['scores']):
                if score > 0.6:
                    box_np = box.cpu().numpy().astype(int)
                    label_index = label.cpu().item()
                    class_name = CLASSES[label_index]
                    
                    # 로그 메시지 생성
                    detection_logs.append(
                        f"Detected: {class_name}, Score: {score:.2f}, Box: [{box_np[0]} {box_np[1]} {box_np[2]} {box_np[3]}]"
                    )

                    if class_name == 'human':
                        human_detected = True
                        color = (0, 255, 0)
                    elif class_name == 'trash':
                        trash_detected = True
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (box_np[0], box_np[1]), (box_np[2], box_np[3]), color, 2)
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (box_np[0], box_np[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 탐지된 객체가 있을 경우에만 로그 출력
            if detection_logs:
                logging.info("--- Frame Detections ---")
                for log_msg in detection_logs:
                    logging.info(log_msg)

            # 경고 상태 업데이트 로직
            if human_detected and trash_detected:
                if not warning_active: # 경고가 방금 활성화된 경우에만 로그 기록
                    logging.warning("Illegal dumping TRIGGERED! (human and trash detected)")
                warning_active = True
                last_warning_time = current_time 

        # 경고 메시지 표시 로직
        if warning_active and (current_time - last_warning_time < warning_duration):
            warning_text = "Warning: Illegal Dumping Detected!"
            cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if warning_active: # 경고가 방금 비활성화된 경우에만 로그 기록
                logging.info("Warning state deactivated.")
            warning_active = False

        # FPS 계산 및 로그 출력
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"
        logging.info(f"Current {fps_text}")
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Illegal Dumping Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam feed stopped and resources released.")

if __name__ == "__main__":
    run_webcam_inference()