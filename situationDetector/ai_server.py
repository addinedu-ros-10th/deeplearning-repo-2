# ai_server.py
import threading
import time
import queue
from situationDetector.tcp_ped_receiver import handle_tcp_client
from situationDetector.udp_ped_receiver import receive_video_udp
from situationDetector.yolo_detector import detect_objects

def main():
    """
    TCP, UDP, YOLO 처리 스레드 생성 및 관리
    """
    # 모든 스레드를 동시에 종료하기 위한 Event
    shutdown_event = threading.Event()

    # 스레드 간 프레임 공유를 위해 큐 생성
    frame_queue = queue.Queue(maxsize=10)

    # 1. TCP 수신 스레드 생성
    tcp_thread = threading.Thread(
        target=handle_tcp_client,
        args=(shutdown_event,),
        daemon=True
    )

    # 2. UDP 수신 스레드 생성 (인자로 situationDetector 전달)
    udp_thread = threading.Thread(
        target=receive_video_udp,
        args=(frame_queue, shutdown_event,),
        daemon=True
    )

    # 2. YOLO 처리 스레드 생성 (인자로 situationDetector 전달)
    yolo_thread = threading.Thread(
        target=detect_objects,
        args=(frame_queue, shutdown_event,),
        daemon=True
    )

    print("AI Server Main : AI 서버의 수신 스레드를 시작합니다.")
    tcp_thread.start()
    udp_thread.start()
    yolo_thread.start()

    try:
        # KeyboardInterrupt를 받기 위해 메인 스레드가 대기
        while tcp_thread.is_alive() and udp_thread.is_alive() and yolo_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nAI Server Main : 종료 신호를 감지. 모든 스레드 정리")
        # 모든 스레드에 종료 신호 전송
        shutdown_event.set()
    finally:
        # 스레드가 완전히 종료될 때까지 대기
        tcp_thread.join()
        udp_thread.join()
        yolo_thread.join()
        print("AI Server Main : 모든 스레드 종료. 프로그램 종료")

if __name__ == "__main__":
    main()