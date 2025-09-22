import socket
import json
import time
from datetime import datetime

try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)
    client_socket.connect(('localhost', 2401))
    print(f"서버에 연결 성공: localhost:2401")

    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "detection": [
                {"class_id": 0, "class_name": "smoking", "confidence": 1.0, "bbox": {"x1": 170.0, "y1": 481.2, "x2": 123.4, "y2": 456.7}},
                {"class_id": 3, "class_name": "fire", "confidence": 0.5 + 0.1 * (time.time() % 10 / 10), "bbox": {"x1": 170.0, "y1": 481.2, "x2": 123.4, "y2": 456.7}}
            ],
            "detection_count": 2,
            "timestamp": current_time,
            "patrol_number": 0
        }
        try:
            message = json.dumps(data).encode('utf-8')
            client_socket.send(message)
            print(f"데이터 전송: {message.decode('utf-8')}")
            time.sleep(5)  # 5초 간격으로 전송
        except socket.error as e:
            print(f"데이터 전송 오류: {e}")
            break
except socket.error as e:
    print(f"연결 오류: {e}")
finally:
    client_socket.close()
    print("클라이언트 연결 종료")