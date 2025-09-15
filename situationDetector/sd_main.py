# ai_server.py
import threading
import time
import queue
from situationDetector.receiver.tcp_dm_receiver import handle_tcp_client
from situationDetector.receiver.udp_dm_receiver import receive_video_udp
from situationDetector.detect.feat.feat_detect_fire.yolo_detector import detect_objects
from situationDetector.sender.tcp_ds_sender import send_data_to_db

"""
스레드 함수 전달 인자

1. 공유 데이터 (프레임 큐 / 분석 결과 큐)
2. 스레드 락 (다른 스레드가 처리 완료될 때까지 기다려야 하는 경우 락을 인자로 전달)
3. 
"""

def main():
    """
    TCP, UDP, YOLO 처리 스레드 생성 및 관리
    """
    # 모든 스레드를 동시에 종료하기 위한 Event
    shutdown_event = threading.Event()

    # 스레드 간 프레임 공유를 위해 큐 생성
    frame_queue = queue.Queue(maxsize=10) # 프레임 큐
    db_manager_queue = queue.Queue(maxsize=20) # 분석 결과 큐

    # TCP 메타데이터 공유를 위한 딕셔너리 + Lock
    shared_metadata = {
        "timestamp" : None,
        "patrol_car_name" : "Unknown"
    }
    metadata_lock = threading.Lock()

    # 1. TCP 메타데이터 수신 스레드 생성
    tcp_thread = threading.Thread(
        target=handle_tcp_client,
        args=(shared_metadata, metadata_lock, shutdown_event,),
        daemon=True
    )

    # 2. UDP 수신 스레드 생성 (인자로 frame_queue 전달)
    udp_thread = threading.Thread(
        target=receive_video_udp,
        args=(frame_queue, shutdown_event,),
        daemon=True
    )

    # 3. YOLO 처리 스레드 생성 (인자로 frame_queue, detect_queue 전달)
    yolo_thread = threading.Thread(
        target=detect_objects,
        args=(frame_queue, db_manager_queue, shared_metadata, metadata_lock, shutdown_event,),
        daemon=True
    )

    # 4. DB 전송 스레드 생성
    db_sender_thread = threading.Thread(
        target=send_data_to_db,
        args = (db_manager_queue, shutdown_event,),
        daemon=True
    )

    print("AI Server Main : AI 서버의 수신 스레드를 시작합니다.")
    tcp_thread.start()
    udp_thread.start()
    yolo_thread.start()
    db_sender_thread.start()

    try:
        # 모든 스레드가 살아있는 동안 메인 스레드 대기
        while all(t.is_alive() for t in [tcp_thread, udp_thread, yolo_thread, db_sender_thread]):
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
        db_sender_thread.join() # 변경/추가된 부분
        print("AI Server Main : 모든 스레드 종료. 프로그램 종료")

if __name__ == "__main__":
    main()