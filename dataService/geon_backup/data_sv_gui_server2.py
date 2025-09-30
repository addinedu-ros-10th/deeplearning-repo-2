
# data_service.py
import socket
import json
import time
import random

TCP_HOST, TCP_PORT = "127.0.0.1", 8801
UDP_HOST, UDP_PORT = "127.0.0.1", 9901


# TCP 서버 (video 요청 수신)
def tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)
    print(f"[TCP] Listening on {TCP_HOST}:{TCP_PORT}")

    conn, addr = server.accept()
    print(f"[TCP] Connection from {addr}")

    # ✅ 수정됨: 연결을 유지하면서 여러 요청 처리
    while True:
        request = conn.recv(1024).decode()
        if not request:  # 클라이언트가 연결을 끊었을 때
            print("[TCP] Client disconnected.")
            break

        print("[TCP] Received request:", request)

    conn.close()
    server.close()

# ------------------------------
# UDP 서버 (command 기반 요청 처리)
# ------------------------------
def udp_server():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] Listening on {UDP_HOST}:{UDP_PORT}")

    while True:
        data, addr = udp_sock.recvfrom(4096)
        try:
            request = json.loads(data.decode("utf-8"))
            print(f"[UDP] Received request: {request}")

            command = request.get("command")
            response = None

            # 1. 로그 요청
            if command == "get_logs":
                patrol_car = request.get("patrol_car", "unknown_car")
                logs = [
                    {"time": "2025-09-16 10:00", "event": f"{patrol_car} started patrol"},
                    {"time": "2025-09-16 10:30", "event": f"{patrol_car} checked area A"},
                ]
                response = json.dumps({"status": "ok", "logs": logs}).encode("utf-8")

            # 2. 이미지 요청
            elif command == "get_image":
                event_id = request.get("event_id", "unknown_evt")
                # 실제 구현에서는 DB 또는 파일에서 이미지 로드
                fake_image_bytes = b"\x89PNG\r\n\x1a\n...FAKE_IMAGE_DATA..."
                response = fake_image_bytes

            # 3. 비디오 스트리밍 요청
            elif command == "stream_video":
                event_id = request.get("event_id", "unknown_evt")
                stream_id = f"stream_{event_id}_{random.randint(1000,9999)}"
                response_dict = {"status": "streaming_starting", "stream_id": stream_id}
                response = json.dumps(response_dict).encode("utf-8")

                # 별도 UDP 스트림 전송 (비동기적으로 실행하는 게 좋음)
                udp_stream(event_id, addr)

            else:
                response = json.dumps({"status": "error", "message": "Unknown command"}).encode("utf-8")

        except Exception as e:
            response = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")

        udp_sock.sendto(response, addr)

# ------------------------------
# UDP 클라이언트 (video stream 전송)
# ------------------------------
def udp_stream(event_id="example", addr=None):
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(5):  # 실제 구현에서는 frame 단위 전송
        fake_frame = f"{event_id}-frame-{i}".encode()
        udp_sock.sendto(fake_frame, addr)
        print(f"[UDP-STREAM] Sent {fake_frame} to {addr}")
        time.sleep(1)
    udp_sock.sendto(b"END", addr)
    udp_sock.close()

# ------------------------------
# 실행 시작
# ------------------------------
if __name__ == "__main__":
    # TCP는 연결 유지형 (video 요청 처리용)
    # UDP는 요청/응답 및 스트리밍 처리용
    udp_server()
