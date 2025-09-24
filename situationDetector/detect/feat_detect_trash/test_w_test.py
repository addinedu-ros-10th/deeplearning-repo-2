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
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_filename = f"detection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- !!! 핵심 상태 변수 (전역 또는 클래스 속성으로 관리) !!! ---
# 이 변수를 외부 시스템에서 읽어가거나 콜백 함수를 통해 전달할 수 있습니다.
DUMPING_STATE = "NORMAL" # 초기 상태는 '정상'

def run_webcam_inference():
    global DUMPING_STATE # 전역 변수 사용 선언

    # --- 환경 설정 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3
    CLASSES = ['BACKGROUND', 'human', 'trash']
    
    # --- 모델 불러오기 ---
    logging.info("Loading model...")
    model = get_model(num_classes)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return

    # --- 웹캠 초기화 ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    logging.info("Starting webcam feed... Press 'q' to quit.")

    # --- 루프 변수 ---
    prev_time = time.time()
    frame_counter = 0
    skip_frames = 3
    last_prediction = None
    
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
                last_prediction = model(img_tensor)
        
        if last_prediction is not None:
            human_detected = False
            trash_detected = False
            
            for box, label, score in zip(last_prediction[0]['boxes'], last_prediction[0]['labels'], last_prediction[0]['scores']):
                if score > 0.6:
                    class_name = CLASSES[label.item()]
                    if class_name == 'human': human_detected = True
                    elif class_name == 'trash': trash_detected = True
                    
                    # (시각화 로직은 기존과 동일)
                    box_np = box.cpu().numpy().astype(int)
                    color = (0, 255, 0) if class_name == 'human' else (0, 0, 255)
                    cv2.rectangle(frame, (box_np[0], box_np[1]), (box_np[2], box_np[3]), color, 2)
                    text = f"{class_name}: {score:.2f}"
                    cv2.putText(frame, text, (box_np[0], box_np[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 상태 변수 업데이트 로직 ---
            previous_state = DUMPING_STATE
            if human_detected and trash_detected:
                DUMPING_STATE = "DETECTED" # 상태를 '탐지됨'으로 변경
            else:
                DUMPING_STATE = "NORMAL"   # 상태를 '정상'으로 변경

            # 상태가 '정상'에서 '탐지됨'으로 바뀌는 순간에만 로그를 남김
            if previous_state == "NORMAL" and DUMPING_STATE == "DETECTED":
                logging.warning("Illegal dumping state changed: NORMAL -> DETECTED")
                # 여기에 다른 시스템에 알림을 보내는 코드를 추가할 수 있습니다.
                # 예: send_alert_to_server(DUMPING_STATE)
            
            # 상태가 '탐지됨'에서 '정상'으로 바뀌는 순간에만 로그를 남김
            elif previous_state == "DETECTED" and DUMPING_STATE == "NORMAL":
                logging.info("Illegal dumping state changed: DETECTED -> NORMAL")


        # --- 시각화: 현재 상태를 화면에 표시 ---
        state_text = f"STATE: {DUMPING_STATE}"
        color = (0, 0, 255) if DUMPING_STATE == "DETECTED" else (0, 255, 0)
        cv2.putText(frame, state_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS 표시
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Illegal Dumping Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam feed stopped and resources released.")

if __name__ == "__main__":
    run_webcam_inference()