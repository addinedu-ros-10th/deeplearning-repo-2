
# monitoring_gui.py
import socket
import json

TCP_HOST, TCP_PORT = "127.0.0.1", 9000
UDP_HOST, UDP_PORT = "127.0.0.1", 9901

# TCP 요청 (video 파일 요청)
def request_video(video_name="sample_video.mp4"):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((TCP_HOST, TCP_PORT))
    print(f"[TCP] Connected to dataService at {TCP_HOST}:{TCP_PORT}")

    request = {"action": "request_video", "video": video_name}
    client.send(json.dumps(request).encode())

    response = client.recv(1024)
    print("[TCP] Response from dataService:", response.decode())
    client.close()

# UDP 수신 (video stream)
def receive_udp_stream():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] Listening on {UDP_HOST}:{UDP_PORT}")

    while True:
        data, addr = udp_sock.recvfrom(4096)
        if data == b"END":
            print("[UDP] Stream ended")
            break
        print(f"[UDP] Received: {data.decode()} from {addr}")
    udp_sock.close()

if __name__ == "__main__":
    request_video("example.mp4")
    receive_udp_stream()
