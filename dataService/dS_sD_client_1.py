
import socket
import json
import os

SERVER_HOST = "127.0.0.1"   # 서버 IP (테스트 시 localhost)
SERVER_PORT = 2301          # 서버 포트
VIDEO_FILE = "/home/geonpc/dev-ws/deeplearning-repo-2/dataService/Event_Video/0/Fight/2025-09-25_14_52_57.mp4"   # 테스트용 영상 파일 경로


def send_json(sock, data):

    try:
        """ JSON 데이터를 서버로 전송 """
        json_str = json.dumps(data)
        sock.sendall(json_str.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)                                   # 이 코드로 데이터 전송완료 됐음을 서버에 전달   
        print(f"[클라이언트] 데이터 전송 완료 ({len(json_str)} bytes)")
    except BrokenPipeError:
        # 서버 연결이 끊겼을 때의 처리
        print("[ERROR] Connection lost. The server may have closed the connection.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during send_json: {e}")

def send_video(sock, video_path):

    try:
        """ 영상 파일을 4096바이트 단위로 전송 """
        filesize = os.path.getsize(video_path)
        header = {
            "videosize": filesize,
            "video": "event_video"
        }
        # 영상 시작 정보 전송
        send_json(sock, header)

        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                sock.sendall(chunk)

            sock.shutdown(socket.SHUT_WR)                                   # 이 코드로 데이터 전송완료 됐음을 서버에 전달   

        print(f"[INFO] Video file sent: {video_path}")
    except BrokenPipeError:
        print("[ERROR] Connection lost while sending video. Aborting.")
        return # 전송 중단


def main():
    # 서버 연결
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print(f"[INFO] Connected to server {SERVER_HOST}:{SERVER_PORT}")

        # 1. JSON detection 데이터 전송
        detection_data = {
            "detection": {
                "feat_detect_smoke": [
                    {
                        "class_id": 1,
                        "class_name": "smoking",
                        "confidence": 0.95,
                        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
                    }
                ],
                "feat_detect_fire": [
                    {
                        "class_id": 2,
                        "class_name": "fire",
                        "confidence": 0.88,
                        "bbox": {"x1": 150, "y1": 250, "x2": 350, "y2": 450}
                    }
                ]
            },
            "timestamp": "2025-09-26 14:00:00",
            "patrol_number": 1
        }

        send_json(client_socket, detection_data)
        print("[INFO] Detection JSON sent")

        # 2. 영상 파일 전송
        if os.path.exists(VIDEO_FILE):
            send_video(client_socket, VIDEO_FILE)
        else:
            print(f"[WARN] Video file not found: {VIDEO_FILE}")

        # 3. 전송 완료 알림
        client_socket.sendall(b"DONE")
        print("[INFO] Transmission DONE")


if __name__ == "__main__":
    main()
