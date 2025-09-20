
import socket
import struct
import cv2
import numpy as np

HOST = "127.0.0.1"   # localhost
PORT = 7702          # 수신 포트

def receive_video():
    """TCP 기반 동영상 스트리밍 수신"""
    # TCP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"🎥 TCP Video Receiver: {HOST}:{PORT} 에서 연결 대기 중...")
    conn, addr = sock.accept()
    print(f"✅ 연결됨: {addr}")

    data_buffer = b""
    payload_size = struct.calcsize("!I")  # 4바이트 정수 (프레임 길이)

    try:
        while True:
            # 프레임 길이 수신
            while len(data_buffer) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print("🚫 연결 종료")
                    return
                data_buffer += packet

            packed_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            frame_size = struct.unpack("!I", packed_size)[0]

            # 프레임 데이터 수신
            while len(data_buffer) < frame_size:
                packet = conn.recv(4096)
                if not packet:
                    print("🚫 연결 종료")
                    return
                data_buffer += packet

            frame_data = data_buffer[:frame_size]
            data_buffer = data_buffer[frame_size:]

            # OpenCV로 디코딩
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("TCP Video Stream", frame)

            # ESC 키 입력 시 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        conn.close()
        sock.close()
        cv2.destroyAllWindows()
        print("🔴 스트리밍 수신 종료")

if __name__ == "__main__":
    receive_video()

