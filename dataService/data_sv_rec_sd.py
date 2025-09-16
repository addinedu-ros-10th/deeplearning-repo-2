
import socket
import json

# --------------------------
# TCP 수신 (XML/JSON 데이터)
# --------------------------
def start_tcp_receiver(host="127.0.0.1", port=6602):
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server.bind((host, port))
    tcp_server.listen(1)
    print(f"[TCP] dataService listening on {host}:{port}")

    conn, addr = tcp_server.accept()
    print(f"[TCP] Connected by {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            break
        try:
            # JSON 디코딩 시도
            message = json.loads(data.decode())
            print("\n[TCP] === Received JSON Message ===")
            print("Raw:", data.decode())
            print("Type:", type(message))
            
            if isinstance(message, dict):
                for k, v in message.items():
                    print(f"  - {k}: {v}")
            elif isinstance(message, list):
                for i, item in enumerate(message):
                    print(f"  [{i}]: {item}")
            else:
                print("Parsed message:", message)
            print("=================================\n")

        except Exception as e:
            print("[TCP] Raw data (not JSON):", data.decode(), f"(Error: {e})")

    conn.close()
    tcp_server.close()

# --------------------------
# UDP 수신 (Image/Video 데이터)
# --------------------------
def start_udp_receiver(host="127.0.0.1", port=9001):
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind((host, port))
    print(f"[UDP] dataService listening on {host}:{port}")

    while True:
        data, addr = udp_server.recvfrom(4096)
        print(f"[UDP] Received {len(data)} bytes from {addr}")
        print("Sample Data (first 50 bytes):", data[:50], "\n")

# --------------------------
# 실행부
# --------------------------
if __name__ == "__main__":
    import threading

    # TCP 수신 스레드
    tcp_thread = threading.Thread(target=start_tcp_receiver, daemon=True)
    tcp_thread.start()

    # UDP 수신 (메인 스레드)
    start_udp_receiver()

