import cv2
import numpy as np
import os  # 파일 존재 여부 확인을 위해 os 모듈 추가

# --- 설정 ---
BASE_FILENAME = "webcam_output"  # 저장될 파일의 기본 이름
EXTENSION = ".mp4"  # 파일 확장자
FPS = 20.0  # 녹화할 영상의 초당 프레임 수

def main():
    """
    웹캠 영상을 녹화하고 'q' 키를 누르면 종료합니다.
    동일한 이름의 파일이 있으면 자동으로 번호를 붙여 저장합니다.
    """
    try:
        # --- 주요 변경 사항 시작 ---
        # 1. 최종 저장될 파일 이름 결정
        output_filename = f"{BASE_FILENAME}{EXTENSION}"
        counter = 1
        # 2. 파일이 이미 존재하면, 이름 뒤에 번호를 붙여 새로운 파일 이름 생성
        while os.path.exists(output_filename):
            output_filename = f"{BASE_FILENAME}_{counter}{EXTENSION}"
            counter += 1
        # --- 주요 변경 사항 끝 ---

        # 웹캠 열기 (0은 시스템의 기본 카메라를 의미)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("오류: 카메라를 열 수 없습니다.")
            return

        # 웹캠의 원본 프레임 너비와 높이를 가져와 녹화 해상도로 설정
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (frame_width, frame_height)

        # 비디오 저장을 위한 VideoWriter 객체 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, FPS, resolution)

        # 터미널에 녹화 정보 출력
        print(f"녹화를 시작합니다. [해상도: {frame_width}x{frame_height}] -> 저장 파일: '{output_filename}'")

        # 미리보기 창 설정
        window_name = "Recording... (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
                break

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print(f"\n녹화가 종료되었습니다. '{output_filename}' 파일로 저장되었습니다.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()