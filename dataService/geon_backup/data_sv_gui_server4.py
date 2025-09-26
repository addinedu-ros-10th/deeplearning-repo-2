
import socket
import json
import time
import threading
import random
import base64
import math
import uuid
import struct
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

def stream_video_frames_UDP(
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
    # dest_addr = ("127.0.0.1", 7702)
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

# ------------------------------
# Helper 함수 (TCP 통신)
# ------------------------------

def stream_video_frames_tcp(
    event_id: str,
    dest_addr: Tuple[str, int],
    stream_id: str,
    frame_count: int = 5,
    delay: float = 1
):
    """
    지정된 주소로 TCP를 통해 가상 비디오 프레임을 스트리밍합니다.

    Args:
        event_id (str): 스트리밍 이벤트의 고유 ID.
        dest_addr (Tuple[str, int]): 프레임을 전송할 목적지 주소 (IP, Port).
        stream_id (str): 스트림의 고유 ID.
        frame_count (int): 전송할 프레임의 총 개수.
        delay (float): 각 프레임 전송 사이의 지연 시간(초).
    """
    dest_addr = ("127.0.0.1", 7702)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(dest_addr)

        for i in range(frame_count):
            payload = {
                "type": "video_frame",
                "stream_id": stream_id,
                "seq": i,
                "data": f"{event_id}-frame-{i}"
            }
            message = json.dumps(payload).encode("utf-8")

            # TCP 전송 시 길이(4바이트) + 데이터
            sock.sendall(struct.pack("!I", len(message)) + message)
            print(f"[TCP-STREAM] Sent frame {i+1}/{frame_count} to {dest_addr}")
            time.sleep(delay)

        # 스트리밍 종료 알림 전송
        end_message = {"type": "video_end", "stream_id": stream_id}
        end_bytes = json.dumps(end_message).encode("utf-8")
        sock.sendall(struct.pack("!I", len(end_bytes)) + end_bytes)
        print(f"[TCP-STREAM] Sent END for stream {stream_id}")

    except Exception as e:
        print(f"[ERROR] TCP streaming failed: {e}")
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
    
    # UDP 전용 sock 
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # TCP 전용 sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    total_chunks = math.ceil(len(image_bytes) / IMAGE_CHUNK_SIZE)

    dest_addr = ("127.0.0.1",8801)

    for _ in range(5):
        try:
            sock.connect(dest_addr)
            break
        except ConnectionRefusedError:
            print("서버 준비 안됨, 1초 후 재시도")
            time.sleep(1)

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
            
            # JSON 데이터를 문자열로 변환하고, 끝을 구분하기 위한 구분자를 추가
            # 이 예제에서는 '\n'를 구분자로 사용. 수신부도 동일하게 처리해야 함.
            json_payload = json.dumps(payload).encode('utf-8') + b'\n'
            
            # TCP는 전송 보장을 하므로 UDP와 달리 재전송 로직이 필요 없음
            sock.sendall(json_payload)
            print(f"[TCP-IMG] Sent chunk {i+1}/{total_chunks} to {dest_addr}")

            # UDP 경우
            # send_udp_response(sock, payload, dest_addr)
            # print(f"[UDP-IMG] Sent chunk {i+1}/{total_chunks} to {dest_addr}")

            time.sleep(0.01) # 짧은 지연시간
            
        
        # TCP 경우  > 이미지 전송 완료 알림
        end_message = {"type": "image_end", "stream_id": stream_id}
        sock.sendall(json.dumps(end_message).encode('utf-8') + b'\n')
        print(f"[TCP-IMG] Sent image_end for stream {stream_id}")
    
        
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
        # print(f"TCP 서버가 {TCP_HOST}:{TCP_PORT}에서 대기 중입니다...")
        print(f"[TCP] Listening on {TCP_HOST}:{TCP_PORT}")

        while True:
            try:
                conn, addr = self.tcp_server_sock.accept()
                print(f"{addr}에서 접속했습니다.")
                # print(f"[TCP] Accepted connection from {addr}")

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

        conn.settimeout(10)
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
        
        # if command == "stream_video":
        #     # 비디오 스트리밍 요청 처리 (UDP로 전달)
        #     video_name = request.get("event_id", "unknown_video")
        #     stream_id = f"tcpstream_{video_name}_{random.randint(1000, 9999)}"
        #     client_udp_port = request.get("udp_port", UDP_PORT)
        #     dest_addr = (addr[0], int(client_udp_port))
            
        #     resp = {"status": "ok", "message": "Video streaming initiated via UDP", "stream_id": stream_id}
        #     conn.send(json.dumps(resp).encode('utf-8'))
        #     print(f"Sent stream_video response to {addr}: {resp}")
            
        #     threading.Thread(
        #         target=stream_video_frames_UDP,
        #         args=(video_name, dest_addr, stream_id, 10, 0.5),
        #         daemon=True
        #     ).start()
        
        if command == "stream_video":
            # 비디오 스트리밍 요청 처리 (TCP로 전달)
            video_name = request.get("event_id", "unknown_video")
            stream_id = f"tcpstream_{video_name}_{random.randint(1000, 9999)}"

            # 응답 전송
            # resp = {"status": "ok", "message": "Video streaming initiated via TCP", "stream_id": stream_id}

           
            Result_json = {
                "data_list": 
            [                                            # 데이터 리스트
                    {
                    "timestamp": "2025-09-18 14:00:00",  # 순찰 이벤트 발생 시각
                    "class_id": 1,                       # 순찰 이벤트 id
                    "class_name": "fire",                # 순찰 이벤트 이름
                    "confidence": 0.6,                   # Detection 신뢰도
                    "bbox": {                            # Box 표기 위치
                        "x1": 1.1,
                        "y1": 1.2,
                        "x2": 1.3,
                        "y2": 1.4
                    }
                },
                {
                    "timestamp": "2025-09-18 14:01:00",
                    "class_id": 3,
                    "class_name": "smoke",
                    "confidence": 0.5,
                    "bbox": {
                        "x1": 1.1,
                        "y1": 1.2,
                        "x2": 1.3,
                        "y2": 1.4
                    }
                }
            ],
            "data_count": 2  # 검색 데이터 수
        }

            conn.send(json.dumps(Result_json).encode('utf-8'))
            print(f"Sent stream_video response to {addr}: {Result_json}")

            # TCP 스트리밍 스레드 시작
            threading.Thread(
                target=stream_video_frames_tcp,
                args=(video_name, addr, stream_id, 10, 0.5),
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
            # resp = {"status": "ok", "logs": logs}
            resp = {"status": "ok", "header": header}

            # send_udp_response(self.udp_server_sock, header, addr)
            conn.send(json.dumps(resp).encode('utf-8'))

            print(f"Sent get_image response to {addr}: {resp}")

            # print(f"Sent get_image response to {addr}")


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
                target=stream_video_frames_UDP, 
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