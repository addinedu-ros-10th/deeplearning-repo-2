
# data_service.py
import socket
import json
import time
import threading
import random
import base64
import math
import uuid

TCP_HOST, TCP_PORT = "127.0.0.1", 8800
UDP_HOST, UDP_PORT = "127.0.0.1", 9901

# ------------------------------
# UDP 보조 함수들
# ------------------------------
def udp_stream(event_id="example", dest_addr=("127.0.0.1", 9901), stream_id=None, frame_count=5, delay=1):
    """UDP로 '비디오 프레임'을 dest_addr로 전송 (프레임은 간단한 바이트 문자열)"""
    # ✅ 추가됨: 스트리밍을 별도 소켓/스레드에서 처리
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if stream_id is None:
        stream_id = f"stream_{event_id}_{random.randint(1000,9999)}"
    try:
        for i in range(frame_count):
            payload = json.dumps({
                "type": "video_frame",
                "stream_id": stream_id,
                "seq": i,
                "data": f"{event_id}-frame-{i}"
            }).encode('utf-8')
            sock.sendto(payload, dest_addr)
            print(f"[UDP-STREAM] Sent frame {i} to {dest_addr}")
            time.sleep(delay)
        # 끝 표시
        sock.sendto(json.dumps({"type": "video_end", "stream_id": stream_id}).encode('utf-8'), dest_addr)
        print(f"[UDP-STREAM] Sent END for {stream_id} to {dest_addr}")
    finally:
        sock.close()

def send_image_over_udp(image_bytes, dest_addr, event_id="evt", chunk_size=4000):
    """이미지를 base64 청크로 나눠 전송 (UDP). header 응답은 이미 보내졌다고 가정."""
    # ✅ 추가됨: 이미지 전송(청크)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream_id = f"img_{event_id}_{random.randint(1000,9999)}"
    total = math.ceil(len(image_bytes) / chunk_size)
    try:
        for i in range(total):
            chunk = image_bytes[i*chunk_size:(i+1)*chunk_size]
            b64 = base64.b64encode(chunk).decode('ascii')
            payload = json.dumps({
                "type": "image_chunk",
                "stream_id": stream_id,
                "seq": i,
                "total": total,
                "data": b64
            }).encode('utf-8')
            sock.sendto(payload, dest_addr)
            print(f"[UDP-IMG] Sent chunk {i+1}/{total} to {dest_addr}")
            time.sleep(0.01)  # 약간의 쉬는시간
        # 완료 표시
        sock.sendto(json.dumps({"type": "image_end", "stream_id": stream_id}).encode('utf-8'), dest_addr)
        print(f"[UDP-IMG] Sent image_end for {stream_id} to {dest_addr}")
    finally:
        sock.close()

# ------------------------------
# UDP 서버 (명령 기반 요청 처리)
# ------------------------------
def udp_server():
    """UDP로 오는 command 요청을 처리"""
    # ✅ 추가됨: UDP 서버 (명령 처리)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] Listening on {UDP_HOST}:{UDP_PORT}")

    while True:
        try:
            data, addr = udp_sock.recvfrom(65536)
            threading.Thread(target=_handle_udp_request, args=(data, addr), daemon=True).start()
        except Exception as e:
            print("[UDP] recv error:", e)

def _handle_udp_request(data, addr):
    """UDP 요청 하나를 처리 (스레드로 실행)"""
    try:
        request = json.loads(data.decode("utf-8"))
    except Exception as e:
        # 단순 메시면 에러 응답
        resp = {"status": "error", "message": "Invalid JSON"}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(resp).encode('utf-8'), addr)
        sock.close()
        return

    print(f"[UDP] Received request from {addr}: {request}")
    command = request.get("command")

    if command == "get_logs":
        patrol_car = request.get("patrol_car", "unknown_car")
        logs = [
            {"time": "2025-09-16 10:00", "event": f"{patrol_car} started patrol"},
            {"time": "2025-09-16 10:30", "event": f"{patrol_car} checked area A"},
        ]
        resp = {"status": "ok", "logs": logs}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(resp).encode('utf-8'), addr)
        sock.close()
        return

    elif command == "get_image":
        event_id = request.get("event_id", "unknown_evt")
        # 실제 구현: 파일에서 읽거나 DB에서 꺼내서 바이트로 반환
        # 여기서는 예제용 가짜 PNG 바이트. 실제론 open('file','rb').read()
        fake_png = b"\x89PNG\r\n\x1a\n" + b"FAKE_IMAGE_DATA" * 10  # 작게 만듦
        total = math.ceil(len(fake_png) / 4000)
        stream_id = f"img_{event_id}_{random.randint(1000,9999)}"
        # 초기 응답: 곧 image chunk를 전송할 것임을 알림
        header = {"status": "sending_image", "stream_id": stream_id, "total_chunks": total}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(header).encode('utf-8'), addr)
        sock.close()
        # 별도 스레드에서 청크 전송
        threading.Thread(target=send_image_over_udp, args=(fake_png, addr, event_id), daemon=True).start()
        return

    elif command == "stream_video":
        event_id = request.get("event_id", "unknown_evt")
        stream_id = f"stream_{event_id}_{random.randint(1000,9999)}"
        resp = {"status": "streaming_starting", "stream_id": stream_id}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(resp).encode('utf-8'), addr)
        sock.close()
        # 별도 스레드에서 UDP 스트림 전송 (frame 단위)
        threading.Thread(target=udp_stream, args=(event_id, addr, stream_id, 10, 0.5), daemon=True).start()
        return

    else:
        resp = {"status": "error", "message": "Unknown command"}
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(resp).encode('utf-8'), addr)
        sock.close()
        return

# ------------------------------
# TCP 서버 (연결 유지, 멀티클라이언트)
# ------------------------------
def handle_tcp_client(conn, addr):
    """개별 TCP 클라이언트 처리 (연결을 유지하면서 여러 요청 처리)"""
    # ✅ 수정됨: 연결 유지 및 멀티클라이언트 처리
    print(f"[TCP] Handling client {addr}")
    conn.settimeout(60)
    try:
        while True:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                print(f"[TCP] timeout from {addr}, closing connection.")
                break
            if not data:
                print(f"[TCP] Client {addr} disconnected.")
                break

            text = data.decode('utf-8').strip()
            print(f"[TCP] Received from {addr}: {text}")
            try:
                req = json.loads(text)
            except json.JSONDecodeError:
                resp = {"status": "error", "message": "Invalid JSON"}
                conn.send(json.dumps(resp).encode('utf-8'))
                continue

            # 예: 기존 코드에서 사용한 "video" 키를 지원
            if "video" in req or req.get("command") == "stream_video":
                # UDP 스트리밍을 클라이언트의 UDP 포트로 보냄
                video_name = req.get("video", req.get("event_id", "unknown.mp4"))
                stream_id = f"tcpstream_{video_name}_{random.randint(1000,9999)}"
                # 클라이언트가 자신의 UDP 포트를 알려주면 사용, 없으면 기본 UDP_PORT 사용
                client_udp_port = req.get("udp_port", UDP_PORT)
                client_ip = addr[0]
                dest = (client_ip, int(client_udp_port))
                # 응답: 스트리밍 시작 알림
                resp = {"status": "ok", "message": f"Video {video_name} request received", "stream_id": stream_id}
                conn.send(json.dumps(resp).encode('utf-8'))
                # 스트리밍은 별도 스레드에서 실행
                threading.Thread(target=udp_stream, args=(video_name, dest, stream_id, 10, 0.5), daemon=True).start()
                continue

            # 단순 로그 요청을 TCP로 처리하는 예
            if req.get("command") == "get_logs":
                patrol_car = req.get("patrol_car", "unknown_car")
                logs = [
                    {"time": "2025-09-16 10:00", "event": f"{patrol_car} started patrol"},
                    {"time": "2025-09-16 10:30", "event": f"{patrol_car} checked area A"},
                ]
                resp = {"status": "ok", "logs": logs}
                conn.send(json.dumps(resp).encode('utf-8'))
                continue

            # 알 수 없는 요청
            resp = {"status": "error", "message": "Unknown request on TCP"}
            conn.send(json.dumps(resp).encode('utf-8'))

    finally:
        conn.close()
        print(f"[TCP] Connection to {addr} closed.")

def tcp_server():
    """TCP 서버: 다중 클라이언트 지원 (각 클라이언트는 스레드로 처리)"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(5)
    print(f"[TCP] Listening on {TCP_HOST}:{TCP_PORT}")

    while True:
        conn, addr = server.accept()
        print(f"[TCP] Accepted connection from {addr}")
        threading.Thread(target=handle_tcp_client, args=(conn, addr), daemon=True).start()

# ------------------------------
# 테스트용: UDP 클라이언트 헬퍼 (GUI/테스트에서 사용)
# ------------------------------
def request_from_dataservice(request: dict, timeout=5):
    """UDP로 request를 보내고 응답을 처리.
    - get_logs: JSON 반환 (bytes)
    - get_image: header 수신 -> 이후 image chunk를 받아 조립하여 bytes 반환
    - stream_video: header JSON 반환 (stream_id) — 이후 별도 UDP 리스너로 프레임 수신 필요
    """
    # ✅ 추가됨: 테스트용 UDP 클라이언트(동기적)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 0))  # OS가 포트 할당 (서버가 이 포트로 스트리밍을 보냄)
    sock.settimeout(timeout)
    try:
        sock.sendto(json.dumps(request).encode('utf-8'), (UDP_HOST, UDP_PORT))
        data, addr = sock.recvfrom(65536)
    except socket.timeout:
        print("[CLIENT] timeout waiting for response")
        sock.close()
        return None

    # try parse JSON header
    try:
        header = json.loads(data.decode('utf-8'))
    except Exception:
        sock.close()
        return data  # raw bytes
    # 로그나 streaming_starting 같은 JSON은 그대로 반환 (bytes)
    if header.get("status") in ("ok", "error", "streaming_starting"):
        sock.close()
        return json.dumps(header).encode('utf-8')

    # 이미지 전송 시작 (header 'sending_image')
    if header.get("status") == "sending_image":
        total = int(header.get("total_chunks", 0))
        stream_id = header.get("stream_id")
        chunks = [None] * total
        received = 0
        # 수신 루프
        while received < total:
            try:
                pkt, addr = sock.recvfrom(131072)
            except socket.timeout:
                print("[CLIENT] timeout while receiving image chunks")
                break
            try:
                pkt_json = json.loads(pkt.decode('utf-8'))
                if pkt_json.get("type") == "image_chunk" and pkt_json.get("stream_id") == stream_id:
                    seq = int(pkt_json["seq"])
                    chunks[seq] = base64.b64decode(pkt_json["data"])
                    received += 1
                    # 진행 로그
                    print(f"[CLIENT] Received chunk {seq+1}/{total}")
            except Exception:
                continue
        sock.close()
        if any(c is None for c in chunks):
            print("[CLIENT] Some image chunks missing")
            return None
        return b"".join(chunks)

    # 그 외
    sock.close()
    return json.dumps(header).encode('utf-8')

# ------------------------------
# 메인: TCP/UDP 서버 동시 실행
# ------------------------------
if __name__ == "__main__":
    # ✅ 추가됨: TCP와 UDP 서버를 동시에 데몬 스레드로 실행
    t_udp = threading.Thread(target=udp_server, daemon=True)
    t_tcp = threading.Thread(target=tcp_server, daemon=True)
    t_udp.start()
    t_tcp.start()
    print("[MAIN] TCP and UDP servers started. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[MAIN] Shutting down (threads are daemon; exiting main).")
