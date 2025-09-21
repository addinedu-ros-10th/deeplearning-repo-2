

import socket
import struct
import json

HOST = "127.0.0.1"   # localhost
PORT = 7702          # 수신 포트

def receive_video_frames_and_json():
    """TCP 기반 동영상 스트리밍 + JSON 데이터 수신"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"🎥 TCP Receiver: {HOST}:{PORT} 에서 연결 대기 중...")
    conn, addr = sock.accept()
    print(f"✅ 연결됨: {addr}")

    data_buffer = b""
    payload_size = struct.calcsize("!I")  # 4바이트 (프레임 길이)

    try:
        while True:
            # 1) 프레임 길이 먼저 수신
            while len(data_buffer) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print("🚫 연결 종료")
                    return
                data_buffer += packet

            packed_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            frame_size = struct.unpack("!I", packed_size)[0]

            # 2) 프레임 데이터 수신
            while len(data_buffer) < frame_size:
                packet = conn.recv(4096)
                if not packet:
                    print("🚫 연결 종료")
                    return
                data_buffer += packet

            frame_data = data_buffer[:frame_size]
            data_buffer = data_buffer[frame_size:]

            print(f"📦 수신한 프레임 크기: {len(frame_data)} bytes")

            # 3) JSON 데이터 수신 (개행 문자 `\n` 기준으로 구분)
            if b"\n" in data_buffer:
                json_raw, data_buffer = data_buffer.split(b"\n", 1)
                try:
                    json_data = json.loads(json_raw.decode("utf-8"))
                    print(f"📝 수신한 JSON: {json_data}")
                except json.JSONDecodeError:
                    print("⚠️ JSON 디코딩 실패")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()
        sock.close()
        print("🔴 스트리밍 종료")

if __name__ == "__main__":
    receive_video_frames_and_json()
