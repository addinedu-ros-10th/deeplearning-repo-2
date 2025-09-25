
import socket
import json
from datetime import datetime
import base64

def send_data_to_server(host='127.0.0.1', port=9999):
    """서버에 테스트용 JSON 데이터를 전송합니다."""

    # --- 테스트용 샘플 데이터 생성 ---
    # 실제 클라이언트에서 생성될 데이터와 유사한 구조로 만듭니다.
    sample_data = {
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
      "patrol_car": "0",
      "class_name": "Fight",
      "raw_result": {
        "confidence": 0.85,
        "object_count": 2,
        "location": { "lat": 37.5665, "lon": 126.9780 }
      },
      "media_info": {
        "rel_path": "/events/video/2025/09/25/suspicious_alpha.mp4",
        "validation": True,
        "mime_type": "video/mp4"
      },
      # [요구사항 2] DB에 저장되지 않아야 할 동영상 프레임 데이터 (예: 1x1 픽셀 이미지)
      "video_frame": base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\xde\x00\x00\x00\x00IEND\xaeB`\x82').decode('utf-8')
    }

    try:
        # 서버에 보낼 데이터를 JSON 문자열로 변환 후, UTF-8 바이트로 인코딩
        message = json.dumps(sample_data).encode('utf-8')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"[클라이언트] 서버({host}:{port})에 연결 시도 중...")
            s.connect((host, port))
            print("[클라이언트] 연결 성공. 데이터 전송 시작...")
            s.sendall(message)
            s.shutdown(socket.SHUT_WR) # 이 코드로 데이터 전송완료 됐음을 서버에 전달   
            print(f"[클라이언트] 데이터 전송 완료 ({len(message)} bytes)")

            # 서버로부터의 응답 수신
            response = s.recv(1024).decode('utf-8')
            print(f"[클라이언트] 서버 응답: {response}")

    except ConnectionRefusedError:
        print("[클라이언트] 오류: 연결을 거부했습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"[클라이언트] 오류 발생: {e}")

if __name__ == "__main__":
    send_data_to_server()



### 실행 방법

# 1.  터미널(명령 프롬프트)을 열고 `main_server.py`를 먼저 실행하여 서버를 가동합니다.
#     ```bash
#     python main_server.py
#     ```
# 2.  **다른** 터미널을 열고 `test_client.py`를 실행하여 서버에 데이터를 전송합니다.
#     ```bash
#     python test_client.py
    

