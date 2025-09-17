
import socket
import json
import time
import threading
import random
import base64
import math
import uuid
from typing import Dict, Tuple

# ------------------------------
# 상수 정의
# ------------------------------
# TCP/UDP 통신을 위한 로컬호스트 및 포트 번호
TCP_HOST, TCP_PORT = "127.0.0.1", 8800
UDP_HOST, UDP_PORT = "127.0.0.1", 8801
# 통신 버퍼 크기
BUFFER_SIZE = 65536
# 이미지 데이터 전송 시 청크 크기 (4KB)
IMAGE_CHUNK_SIZE = 4000

# ------------------------------
# Helper 함수 (UDP 통신)
# ------------------------------
def send_udp_response(sock: socket.socket, message: Dict, addr: Tuple[str, int]):
    """JSON 메시지를 UDP로 전송하는 도우미 함수"""
    try:
        sock.sendto(json.dumps(message).encode('utf-8'), addr)
    except Exception as e:
        print(f"[UDP-HELPER] Failed to send response to {addr}: {e}")

def stream_video_frames(
    event_id: str, 
    dest_addr: Tuple[str, int], 
    stream_id: str, 
    frame_count: int = 5, 
    delay: float = 1
):
    """
    지정된 주소로 UDP를 통해 가상 비디오 프레임을 스트리밍합니다.

    Args:
        event_id (str): 스트리밍 이벤트의 고유 ID.
        dest_addr (Tuple[str, int]): 프레임을 전송할 목적지 주소 (IP, Port).
        stream_id (str): 스트림의 고유 ID.
        frame_count (int): 전송할 프레임의 총 개수.
        delay (float): 각 프레임 전송 사이의 지연 시간(초).
    """
    print("Video_dest_addr : ", dest_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for i in range(frame_count):
            payload = {
                "type": "video_frame",
                "stream_id": stream_id,
                "seq": i,
                "data": f"{event_id}-frame-{i}"
            }
            send_udp_response(sock, payload, dest_addr)
            print(f"[UDP-STREAM] Sent frame {i+1}/{frame_count} to {dest_addr}")
            time.sleep(delay)
        
        # 스트리밍 종료 알림 전송
        end_message = {"type": "video_end", "stream_id": stream_id}
        send_udp_response(sock, end_message, dest_addr)
        print(f"[UDP-STREAM] Sent END for stream {stream_id}")
    finally:
        sock.close()

def send_image_chunks(
    image_bytes: bytes, 
    dest_addr: Tuple[str, int], 
    event_id: str, 
    stream_id: str
):
    """
    이미지 바이트를 Base64 인코딩된 청크로 나누어 UDP를 통해 전송합니다.

    Args:
        image_bytes (bytes): 전송할 이미지 데이터.
        dest_addr (Tuple[str, int]): 청크를 보낼 목적지 주소 (IP, Port).
        event_id (str): 이미지 이벤트의 고유 ID.
        stream_id (str): 이미지 스트림의 고유 ID.

    """
    print("image_dest_addr : ", dest_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    total_chunks = math.ceil(len(image_bytes) / IMAGE_CHUNK_SIZE)
    
    try:
        for i in range(total_chunks):
            start_index = i * IMAGE_CHUNK_SIZE
            end_index = (i + 1) * IMAGE_CHUNK_SIZE
            chunk = image_bytes[start_index:end_index]
            b64_chunk = base64.b64encode(chunk).decode('ascii')
            
            payload = {
                "type": "image_chunk",
                "stream_id": stream_id,
                "seq": i,
                "total": total_chunks,
                "data": b64_chunk
            }
            send_udp_response(sock, payload, dest_addr)
            print(f"[UDP-IMG] Sent chunk {i+1}/{total_chunks} to {dest_addr}")
            time.sleep(0.01) # 짧은 지연시간
            
        # 이미지 전송 완료 알림 전송
        end_message = {"type": "image_end", "stream_id": stream_id}
        send_udp_response(sock, end_message, dest_addr)
        print(f"[UDP-IMG] Sent image_end for stream {stream_id}")
    finally:
        sock.close()

# ------------------------------
# 데이터 서비스 서버 클래스
# ------------------------------
class DataServiceServer:
    """
    TCP 및 UDP 프로토콜을 사용하여 클라이언트 요청을 처리하는 서버 클래스.
    """
    def __init__(self):
        # 서버 소켓 초기화
        self.tcp_server_sock = None
        self.udp_server_sock = None

    def start_servers(self):
        """TCP 및 UDP 서버를 별도의 스레드에서 시작합니다."""
        print("[MAIN] Starting TCP and UDP servers...")
        
        # 데몬 스레드로 서버 실행 (메인 스레드 종료 시 함께 종료)
        tcp_thread = threading.Thread(target=self._run_tcp_server, daemon=True)
        udp_thread = threading.Thread(target=self._run_udp_server, daemon=True)
        
        tcp_thread.start()
        udp_thread.start()
        
        print("[MAIN] All servers are running. Press Ctrl-C to stop.")
        
        # 메인 스레드는 서버가 종료될 때까지 대기
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[MAIN] Servers are shutting down.")

    def _run_tcp_server(self):
        """TCP 서버를 실행하여 클라이언트 연결을 수락합니다."""
        self.tcp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server_sock.bind((TCP_HOST, TCP_PORT))
        self.tcp_server_sock.listen(5)
        print(f"[TCP] Listening on {TCP_HOST}:{TCP_PORT}")

        while True:
            try:
                conn, addr = self.tcp_server_sock.accept()
                print(f"[TCP] Accepted connection from {addr}")
                # 각 클라이언트는 별도 스레드에서 처리
                threading.Thread(
                    target=self._handle_tcp_client, 
                    args=(conn, addr), 
                    daemon=True
                ).start()
            except Exception as e:
                print(f"[TCP] Server error: {e}")
                break

    def _run_udp_server(self):
        """UDP 서버를 실행하여 명령 요청을 처리합니다."""
        self.udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_server_sock.bind((UDP_HOST, UDP_PORT))
        print(f"[UDP] Listening on {UDP_HOST}:{UDP_PORT}")

        while True:
            try:
                data, addr = self.udp_server_sock.recvfrom(BUFFER_SIZE)
                # 각 요청은 별도 스레드에서 처리
                threading.Thread(
                    target=self._handle_udp_request, 
                    args=(data, addr), 
                    daemon=True
                ).start()
            except Exception as e:
                print(f"[UDP] Server error: {e}")
                break

    def _handle_tcp_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """
        개별 TCP 클라이언트의 요청을 처리합니다.
        
        Args:
            conn (socket.socket): 클라이언트와의 연결 소켓.
            addr (Tuple[str, int]): 클라이언트의 주소.
        """
        print(f"[TCP] Handling client {addr}")
        conn.settimeout(60)

        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    print(f"[TCP] Client {addr} disconnected.")
                    break

                try:
                    request = json.loads(data.decode('utf-8').strip())
                except json.JSONDecodeError:
                    resp = {"status": "error", "message": "Invalid JSON"}
                    conn.send(json.dumps(resp).encode('utf-8'))
                    continue

                self._process_tcp_request(conn, addr, request)

        except socket.timeout:
            print(f"[TCP] Connection timeout from {addr}.")
        except Exception as e:
            print(f"[TCP] Error handling client {addr}: {e}")
        finally:
            conn.close()
            print(f"[TCP] Connection to {addr} closed.")

    def _process_tcp_request(self, conn: socket.socket, addr: Tuple[str, int], request: Dict):
        """TCP 요청을 분석하고 적절한 응답을 보냅니다."""
        command = request.get("command")
        
        if command == "stream_video":
            # 비디오 스트리밍 요청 처리 (UDP로 전달)
            video_name = request.get("event_id", "unknown_video")
            stream_id = f"tcpstream_{video_name}_{random.randint(1000, 9999)}"
            client_udp_port = request.get("udp_port", UDP_PORT)
            dest_addr = (addr[0], int(client_udp_port))
            
            resp = {"status": "ok", "message": "Video streaming initiated via UDP", "stream_id": stream_id}
            conn.send(json.dumps(resp).encode('utf-8'))
            print(f"Sent stream_video response to {addr}: {resp}")
            
            threading.Thread(
                target=stream_video_frames,
                args=(video_name, dest_addr, stream_id, 10, 0.5),
                daemon=True
            ).start()
        
        elif command == "get_logs":
            # 로그 요청 처리 (TCP로 응답)
            patrol_car = request.get("patrol_car", "unknown_car")
            logs = [{"time": "2025-09-16 10:00", "event": f"{patrol_car} started patrol"}]
            resp = {"status": "ok", "logs": logs}
            conn.send(json.dumps(resp).encode('utf-8'))
            print(f"Sent get_logs response to {addr}: {resp}")
            
        elif command == "get_image":
            event_id = request.get("event_id", "unknown_evt")
            # 가상 이미지 데이터 생성
            fake_png = b"\x89PNG\r\n\x1a\n" + b"FAKE_IMAGE_DATA" * 1000
            total_chunks = math.ceil(len(fake_png) / IMAGE_CHUNK_SIZE)
            stream_id = f"img_{event_id}_{random.randint(1000, 9999)}"

            # 이미지 전송 시작 알림을 먼저 보냄
            header = {"status": "sending_image", "stream_id": stream_id, "total_chunks": total_chunks}
            send_udp_response(self.udp_server_sock, header, addr)

            # print(f"Sent get_image response to {addr}: {resp}")


            # 별도 스레드에서 이미지 청크 전송 시작
            threading.Thread(
                target=send_image_chunks, 
                args=(fake_png, addr, event_id, stream_id), 
                daemon=True
            ).start()
        else:
            # 알 수 없는 요청
            resp = {"status": "error", "message": "Unknown command"}
            conn.send(json.dumps(resp).encode('utf-8'))
            

    def _handle_udp_request(self, data: bytes, addr: Tuple[str, int]):
        """
        개별 UDP 요청을 처리합니다.
        
        Args:
            data (bytes): 수신된 데이터.
            addr (Tuple[str, int]): 요청을 보낸 클라이언트의 주소.
        """
        try:
            request = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"[UDP] Invalid JSON from {addr}")
            resp = {"status": "error", "message": "Invalid JSON"}
            send_udp_response(self.udp_server_sock, resp, addr)
            return

        print(f"[UDP] Received request from {addr}: {request}")
        
        command = request.get("command")

        if command == "get_logs":
            patrol_car = request.get("patrol_car", "unknown_car")
            logs = [{"time": "2025-09-16 10:00", "event": f"{patrol_car} started patrol"}]
            resp = {"status": "ok", "logs": logs}
            send_udp_response(self.udp_server_sock, resp, addr)

        elif command == "get_image":
            event_id = request.get("event_id", "unknown_evt")
            # 가상 이미지 데이터 생성
            fake_png = b"\x89PNG\r\n\x1a\n" + b"FAKE_IMAGE_DATA" * 1000
            total_chunks = math.ceil(len(fake_png) / IMAGE_CHUNK_SIZE)
            stream_id = f"img_{event_id}_{random.randint(1000, 9999)}"

            # 이미지 전송 시작 알림을 먼저 보냄
            header = {"status": "sending_image", "stream_id": stream_id, "total_chunks": total_chunks}
            send_udp_response(self.udp_server_sock, header, addr)

            # 별도 스레드에서 이미지 청크 전송 시작
            threading.Thread(
                target=send_image_chunks, 
                args=(fake_png, addr, event_id, stream_id), 
                daemon=True
            ).start()

        elif command == "stream_video":
            event_id = request.get("event_id", "unknown_evt")
            stream_id = f"stream_{event_id}_{random.randint(1000, 9999)}"
            resp = {"status": "streaming_starting", "stream_id": stream_id}
            send_udp_response(self.udp_server_sock, resp, addr)
            
            # 별도 스레드에서 비디오 프레임 전송 시작
            threading.Thread(
                target=stream_video_frames, 
                args=(event_id, addr, stream_id, 10, 0.5), 
                daemon=True
            ).start()

        else:
            resp = {"status": "error", "message": "Unknown command"}
            send_udp_response(self.udp_server_sock, resp, addr)

# ------------------------------
# 실행 코드
# ------------------------------
if __name__ == "__main__":
    server = DataServiceServer()
    server.start_servers()