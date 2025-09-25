
# data_service.py
import socket
import json
import time

TCP_HOST, TCP_PORT = "127.0.0.1", 9000
UDP_HOST, UDP_PORT = "127.0.0.1", 9901

# TCP 서버 (video 요청 수신)
def tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)
    print(f"[TCP] Listening on {TCP_HOST}:{TCP_PORT}")

    conn, addr = server.accept()
    print(f"[TCP] Connection from {addr}")

    request = conn.recv(1024).decode()
    print("[TCP] Received request:", request)

    try:
        req_json = json.loads(request)
        video_name = req_json.get("video", "unknown.mp4")
        response = {"status": "ok", "message": f"Video {video_name} request received"}
    except json.JSONDecodeError:
        response = {"status": "error", "message": "Invalid JSON request"}

    conn.send(json.dumps(response).encode())
    conn.close()
    server.close()

# UDP 클라이언트 (video stream 전송)
def udp_stream(video_name="example.mp4"):
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(5):  # 실제 구현에서는 video frame 단위로 전송 가능
        fake_frame = f"{video_name}-frame-{i}".encode()
        udp_sock.sendto(fake_frame, (UDP_HOST, UDP_PORT))
        print(f"[UDP] Sent {fake_frame}")
        time.sleep(1)
    udp_sock.sendto(b"END", (UDP_HOST, UDP_PORT))
    udp_sock.close()

if __name__ == "__main__":
    tcp_server()
    udp_stream("example.mp4")
