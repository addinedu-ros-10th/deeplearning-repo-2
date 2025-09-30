import cv2
from ultralytics import YOLO
import argparse

# --- 메인 실행 함수 ---
def main(cam_index, model_path):
    """
    웹캠을 사용하여 실시간으로 'littering' 객체를 탐지하고 화면에 표시합니다.
    """
    # 1. 훈련된 YOLOv8 모델 가중치 파일을 불러옵니다.
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print(f"'{model_path}' 경로에 훈련된 모델 가중치 파일이 있는지 확인하세요.")
        return

    # 2. 지정된 인덱스의 웹캠을 엽니다.
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"오류: 카메라 인덱스 {cam_index}을(를) 열 수 없습니다.")
        return

    print("실시간 탐지를 시작합니다. 종료하려면 'q' 키를 누르세요.")

    # --- 실시간 영상 처리 루프 ---
    while True:
        # 프레임 단위로 영상 읽기
        success, frame = cap.read()
        if not success:
            print("영상을 더 이상 읽을 수 없습니다.")
            break

        # 3. YOLOv8 모델을 사용하여 현재 프레임에서 객체 탐지 수행
        #    - stream=True 옵션은 실시간 영상 처리에 더 효율적입니다.
        #    - conf=0.5 옵션으로 신뢰도 50% 이상인 객체만 표시합니다.
        results = model(frame, stream=True, verbose=False)

        # 4. 탐지 결과(사각형, 클래스 이름 등)를 프레임에 그립니다.
        for r in results:
            annotated_frame = r.plot()

        # 5. 처리된 프레임을 화면에 보여줍니다.
        cv2.imshow("YOLOv8 Littering Detection", annotated_frame)

        # 'q'를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("\n프로그램을 종료합니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="실시간 무단 투기 객체 탐지 스크립트 (웹캠용)")
    parser.add_argument('--cam', type=int, default=0, help='사용할 카메라 인덱스 (기본값: 0)')
    parser.add_argument('--model', type=str, default='runs/detect/yolov8n_littering_detector/weights/best.pt', help='훈련된 모델 가중치 파일 경로')
    args = parser.parse_args()

    main(args.cam, args.model)

