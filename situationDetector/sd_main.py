# ai_server.py
import threading
import time
import queue
from situationDetector.receiver.tcp_dm_receiver import handle_tcp_client
from situationDetector.receiver.udp_dm_receiver import receive_video_udp
from situationDetector.detect.feat_detect_fire.yolo_detector import detect_objects
from situationDetector.sender.tcp_ds_sender import send_tcp_data_to_db
from situationDetector.sender.udp_ds_sender import send_udp_frame_to_db

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
    tcp_connected_event = threading.Event()
    shutdown_event = threading.Event()

    # 스레드 간 프레임 공유를 위해 큐 생성
    analysis_frame_queue = queue.Queue(maxsize=10) # 데이터 분석을 위한 프레임 큐
    send_frame_queue = queue.Queue(maxsize=10) # db 전송을 위한 프레임 큐
    
    event_video_queue = queue.Queue(maxsize = 10) # 이벤트 비디오 큐
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
        # 수신한 프레임을 분석 프레임 큐에 추가
        # 수신한 프레임을 db서버 전송 프레임 큐에 추가
    udp_thread = threading.Thread(
        target=receive_video_udp,
        args=(analysis_frame_queue, send_frame_queue, shutdown_event,),
        daemon=True
    )

    # 3. YOLO 처리 스레드 생성 (인자로 analysis_frame_queue, db_manager_queue 전달)
        # 분석 프레임 큐에서 프레임을 꺼내 분석
        # 분석 결과(json)을 db_manager_queue 큐에 저장
    yolo_thread = threading.Thread(
        target=detect_objects,
        args=(analysis_frame_queue, db_manager_queue, shared_metadata, metadata_lock, shutdown_event,),
        daemon=True
    )

    # 4. DB (ds) TCP 전송 스레드 생성
        # 분석한 결과가 들어있는 db_manager_queue 큐의 데이터를 db서버에 전송
    db_tcp_sender_thread = threading.Thread(
        target=send_tcp_data_to_db,
        args = (db_manager_queue, tcp_connected_event, shutdown_event,),
        daemon=True
    )

    # 4. DB (ds) UDP 전송 스레드 생성 
        # 수신한 프레임 큐를 db서버에 그대로 전송
    db_udp_sender_thread = threading.Thread(
        target=send_udp_frame_to_db,
        args = (send_frame_queue, tcp_connected_event, shutdown_event,),
        daemon=True
    )

    print("AI Server Main : AI 서버의 수신 스레드를 시작합니다.")
    tcp_thread.start()
    udp_thread.start()
    yolo_thread.start()
    db_tcp_sender_thread.start()
    db_udp_sender_thread.start()

    try:
        # 모든 스레드가 살아있는 동안 메인 스레드 대기
        while all(t.is_alive() for t in [tcp_thread, udp_thread, yolo_thread, db_tcp_sender_thread, db_udp_sender_thread]):
        # while all(t.is_alive() for t in [tcp_thread, udp_thread, db_tcp_sender_thread, db_udp_sender_thread]):
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
        db_tcp_sender_thread.join()
        db_udp_sender_thread.join()
        print("AI Server Main : 모든 스레드 종료. 프로그램 종료")

if __name__ == "__main__":
    main()