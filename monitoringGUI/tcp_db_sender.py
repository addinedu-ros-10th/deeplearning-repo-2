import socket
import json

def request_from_dataservice(request_payload):
    """dataService에 JSON 요청을 보내고 응답을 받는 함수"""
    HOST = 'localhost'  # dataService 서버 주소
    PORT = 8800         # dataService 서버 TCP 포트
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((HOST, PORT))
            
            # JSON 요청 전송 (개행문자로 메시지 끝 표시)
            message = json.dumps(request_payload) + '\n'
            sock.sendall(message.encode('utf-8'))
            
            # 응답 수신 (이미지 데이터 등 큰 데이터 수신을 위해 루프 사용)
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            
            return response_data
        except Exception as e:
            print(f"dataService 요청 오류: {e}")
            return None

# --- 사용 예시 ---
# 1. 로그 요청
log_request = {"command": "get_logs", "patrol_car": "Patrol_Car_1"}
logs_response = request_from_dataservice(log_request)
if logs_response:
    logs = json.loads(logs_response.decode('utf-8'))
    # GUI에 로그 표시

# 2. 이미지 요청
image_request = {"command": "get_image", "event_id": "evt_12345"}
image_bytes = request_from_dataservice(image_request)
if image_bytes:
    # 받은 바이트를 GUI에 이미지로 표시
    pass

# 3. 비디오 스트리밍 요청
video_request = {"command": "stream_video", "event_id": "evt_12345"}
video_response_bytes = request_from_dataservice(video_request)
if video_response_bytes:
    # {"status": "streaming_starting", "stream_id": "stream_evt_12345"}
    video_response = json.loads(video_response_bytes.decode('utf-8'))
    # 이 stream_id를 사용하여 UDP 스트림을 식별