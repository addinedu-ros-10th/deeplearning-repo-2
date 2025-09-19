import cv2
import socket
import numpy as np
import struct

"""
dataService 영상 데이터 수신을 모방하는 테스트용 코드
"""

UDP_HOST = 'localhost' # 로컬테스트 IP 주소
# UDP_HOST = '192.168.0.182'         # dataService IP 주소
UDP_PORT = 2300       # situationDetector으로부터 UDP 영상 수신 포트

NUM_CHUNKS = 20       # 1프레임을 쪼개는 조각 수

def main():
    """
    UDP 패킷을 수신하여 영상 프레임으로 재조립하고 화면에 표시합니다.
    """
    # UDP 소켓 생성 및 바인딩
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.bind((UDP_HOST, UDP_PORT))
        print(f"UDP 서버가 {UDP_HOST}:{UDP_PORT}에서 수신 대기 중입니다...")
    except OSError as e:
        print(f"소켓 바인딩에 실패했습니다: {e}")
        print("포트가 이미 사용 중일 수 있습니다. 다른 프로그램을 확인하거나 잠시 후 다시 시도하세요.")
        return

    # 프레임 재조립을 위한 변수
    chunks = {}
    current_frame_id = -1

    try:
        while True:
            # UDP 패킷 수신 (버퍼 크기는 헤더와 청크 데이터를 포함할 수 있도록 충분히 크게 설정)
            packet, addr = udp_sock.recvfrom(65536)

            # 패킷 헤더 분석
            header_size = struct.calcsize('!LB')
            if len(packet) < header_size:
                continue  # 유효하지 않은 패킷은 무시

            header = packet[:header_size]
            chunk_data = packet[header_size:]

            try:
                frame_id, chunk_id = struct.unpack('!LB', header)
            except struct.error:
                print("패킷 헤더를 언패킹하는 데 실패했습니다.")
                continue

            # 새로운 프레임이 시작되면 청크 저장소 초기화
            if frame_id > current_frame_id:
                current_frame_id = frame_id
                chunks.clear()

            # 현재 프레임에 해당하는 청크만 저장
            if frame_id == current_frame_id:
                chunks[chunk_id] = chunk_data

            # 모든 청크가 수신되었는지 확인
            if len(chunks) == NUM_CHUNKS:
                # 청크 ID 순서대로 정렬하여 데이터 재구성
                sorted_chunks = [chunks[i] for i in range(NUM_CHUNKS)]
                frame_data = b"".join(sorted_chunks)

                # 바이트 데이터를 이미지로 디코딩
                np_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                # 디코딩된 프레임을 화면에 표시
                if frame is not None:
                    print(f"Frame {current_frame_id} 표시")
                    cv2.imshow('UDP Video Receiver', frame)
                else:
                    print(f"Frame {current_frame_id} 디코딩 실패.")
                
                # 다음 프레임을 위해 청크 저장소 초기화
                chunks.clear()

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료 신호를 감지했습니다.")
                break

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 리소스 정리
        print("프로그램을 종료합니다.")
        udp_sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()