# udp_sender.py
import cv2
import queue
import socket
import struct
import threading
import time

# 통신 변수
UDP_HOST = 'localhost'
UDP_PORT = 6605 # 
NUM_CHUNKS = 20 # 1프레임을 20조각으로 분할하여 전송함

def send_udp_frame_to_gui(send_frame_queue: queue.Queue,
                        tcp_connected_event: threading.Event,
                        shutdown_event: threading.Event):
    """
    1. db와 TCP 연결이 있는동안 UDP 영상 전송
    = frame_queue에 프레임이 남아있으면 db에 그대로 전송
    """
    udp_sock = None
    cap = None
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_server_addr = (UDP_HOST, UDP_PORT)
        
        frame_id = 0
        while not shutdown_event.is_set():
            # 1. TCP 연결을 기다림 (최대 1초 대기 후 다시 확인)
            # wait() 메소드는 Event가 set될 때까지 블록되며, set되면 True를 반환
            is_connected = tcp_connected_event.wait(timeout=1.0)

            # 2. TCP 연결 확인
            if not is_connected:
                print("situationDetector (UDP) : gui TCP 연결 대기중...")
                continue
            
            # 3. 보낼 프레임이 있는지 확인
            if send_frame_queue.empty():
                continue # 보낼 프레임이 없으면 대기
            frame = send_frame_queue.get()

            # 4. 이미지 인코딩
            encode_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not encode_success:
                continue
            
            # 5. 데이터 분할 및 분할한 데이터 조각 전송
            data_bytes = buffer.tobytes()
            total_size = len(data_bytes)
            chunk_size = (total_size + NUM_CHUNKS - 1) // NUM_CHUNKS

            for i in range(NUM_CHUNKS):
                start = i * chunk_size
                end = start + chunk_size
                chunk = data_bytes[start:end]
                header = struct.pack('!LB', frame_id, i)
                packet = header + chunk
                udp_sock.sendto(packet, udp_server_addr)

            print(f"situationDetector (UDP) : Frame {frame_id} 전송 완료 ({total_size} bytes)")
            frame_id += 1
            
            # CPU 사용률을 낮추기 위해 짧은 대기 시간 추가
            time.sleep(0.01)

    except Exception as e:
        print(f"situationDetector (UDP) : 오류 발생: {e}")
        shutdown_event.set() # 오류 시 전체 종료
    finally:
        print("situationDetector (UDP) : 스트리밍 스레드를 종료합니다.")
        if cap:
            cap.release()
        if udp_sock:
            udp_sock.close()
        cv2.destroyAllWindows()



def stream_video_udp(tcp_connected_event: threading.Event, shutdown_event: threading.Event):
    """
    TCP 연결이 있는 경우에만 UDP로 영상을 스트리밍
    """
    udp_sock = None
    cap = None
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_server_addr = (UDP_HOST, UDP_PORT)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("PED (UDP) : 카메라를 열 수 없습니다.")

        frame_id = 0
        while not shutdown_event.is_set():
            # TCP 연결을 기다림 (최대 1초 대기 후 다시 확인)
            # wait() 메소드는 Event가 set될 때까지 블록되며, set되면 True를 반환
            is_connected = tcp_connected_event.wait(timeout=1.0)

            if not is_connected:
                print("PED (UDP) : TCP 연결 대기 중...")
                continue
            
            # TCP가 연결된 경우에만 아래 코드 실행
            success, frame = cap.read()
            if not success:
                continue

            encode_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not encode_success:
                continue

            data_bytes = buffer.tobytes()
            total_size = len(data_bytes)
            chunk_size = (total_size + NUM_CHUNKS - 1) // NUM_CHUNKS

            for i in range(NUM_CHUNKS):
                start = i * chunk_size
                end = start + chunk_size
                chunk = data_bytes[start:end]
                header = struct.pack('!LB', frame_id, i)
                packet = header + chunk
                udp_sock.sendto(packet, udp_server_addr)

            print(f"PED (UDP) : Frame {frame_id} 전송 완료 ({total_size} bytes)")
            frame_id += 1
            
            # CPU 사용률을 낮추기 위해 짧은 대기 시간 추가
            time.sleep(0.01)

    except Exception as e:
        print(f"PED (UDP) : 오류 발생: {e}")
        shutdown_event.set() # 오류 시 전체 종료
    finally:
        print("PED (UDP) : 스트리밍 스레드를 종료합니다.")
        if cap:
            cap.release()
        if udp_sock:
            udp_sock.close()
        cv2.destroyAllWindows()