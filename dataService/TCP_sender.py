
import socket
import json
import time


# HOST = 'localhost'  
# PORT = 8801 

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

            #✅ 수정됨: 연결을 유지하면서 여러 요청 처리
            while True:
                request = sock.recv(1024).decode()
                if not request:  # 클라이언트가 연결을 끊었을 때
                    print("[TCP] dataServvice disconnected.")
                    break

                print("[TCP] Received request:", request)

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
        

# -----------------------------
# TCP 이미지 수신
# -----------------------------
# def receive_image_chunks_tcp():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.bind((HOST, PORT))
#     sock.listen(1)
#     print(f"[TCP-IMG-RECV] Listening on {HOST}:{PORT}")

#     conn, addr = sock.accept()
#     print(f"[TCP-IMG-RECV] Connected from {addr}")

#     buffer = b""
#     try:
#         while True:
#             data = conn.recv(4096)
#             if not data: break
#             buffer += data

#             # '\n' 단위로 패킷 분리
#             while b"\n" in buffer:
#                 line, buffer = buffer.split(b"\n", 1)
#                 payload = json.loads(line.decode("utf-8"))

#                 if payload["type"] == "image_chunk":
#                     print(f"[TCP-IMG-RECV] Chunk {payload['seq']+1}/{payload['total']}")
#                 elif payload["type"] == "image_end":
#                     print(f"[TCP-IMG-RECV] Image stream {payload['stream_id']} complete")
#                     return
#     finally:
#         conn.close()
#         sock.close()



        
# --- 사용 예시 ---
# 1. 로그 요청
log_request = {"command": "get_logs", "patrol_car": "Patrol_Car_1"}
logs_response = request_from_dataservice(log_request)
if logs_response:
    logs = json.loads(logs_response.decode('utf-8'))
    # GUI에 로그 표시

# 2. 이미지 요청
# image_request = {"command": "get_image", "event_id": "evt_12345"}
# image_bytes = request_from_dataservice(image_request)
# if image_bytes:
#     # 받은 바이트를 GUI에 이미지로 표시
#     pass

# 3. 비디오 스트리밍 요청
# video_request = {"command": "stream_video", "event_id": "evt_12345"}
Search_json = {
    "command": "stream_video",
    "time_start": "20250918",           ## 검색 시작 일자
    "time_end": "20250920",             ## 검색 종료 일자
    "time_orderby": "true",             ## 검색 결과 순서, true: 최신순, false: 오래된 순
    "detection_type": ["0", "3"]        ## 검색 이벤트
    }

video_response_bytes = request_from_dataservice(Search_json)
if video_response_bytes:
    # {"status": "streaming_starting", "stream_id": "stream_evt_12345"}
    video_response = json.loads(video_response_bytes.decode('utf-8'))
    # 이 stream_id를 사용하여 UDP 스트림을 식별

# video_response_bytes = request_from_dataservice(video_request)
# if video_response_bytes:
#     # {"status": "streaming_starting", "stream_id": "stream_evt_12345"}
#     video_response = json.loads(video_response_bytes.decode('utf-8'))
#     # 이 stream_id를 사용하여 UDP 스트림을 식별


# # -----------------------------
# # 실행 테스트
# # -----------------------------
# if __name__ == "__main__":
#     import threading

#     # 수신 서버 실행 (별도 스레드)
#     threading.Thread(target=receive_image_chunks_tcp, daemon=True).start()

#     time.sleep(0.1)  # 서버 준비 대기