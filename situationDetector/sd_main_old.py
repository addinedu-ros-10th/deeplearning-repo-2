# ai_server.py
import threading
import time
import queue

from situationDetector.receiver.tcp_dm_receiver import receive_event_video
from situationDetector.test_data.udp_dm_receiver import receive_video_udp

from situationDetector.sender.tcp_main_sender import send_tcp_data_to_main
from situationDetector.sender.udp_main_sender import send_udp_frame_to_main

# from situationDetector.detect.feat_detect_fall import run_fall_detect
from situationDetector.detect.feat_detect_fire import run_fire_detect
# from situationDetector.detect.feat_detect_smoke import run_smoke_detect
# from situationDetector.detect.feat_detect_trash import run_trash_detect
# from situationDetector.detect.feat_detect_violence import run_violence_detect
# from situationDetector.detect.feat_detect_weapon import run_weapon_detect

from situationDetector.test_data.test_constant import TEST_AI_JSON


"""
situationDetector main함수

이슈 : 단일 함수를 사용하지만 복잡한 구조 -> 단일 함수 + 스레드 구조에서 클래스 + 스레드 구조로 리펙토링?

주요 특징
1. main함수에서 통신/기능 스레드, 공유 데이터 관리
2. 세부 기능, 통신은 각 모듈의 1~3개의 간단한 함수로 구현
3. 영상 프레임 데이터를 GUI 전송용 + 6개 기능 구동용으로 7개로 분배해주는 구현에서
생산자 - 소비자 (Producer-Customer) 패턴 + 큐 조합으로 구현
"""

# 분석 모델 설정
ANALYZER_CONFIG = [
    # {"name" : "feat_detect_fall", "target" : run_fall_detect},
    {"name" : "feat_detect_fire", "target" : run_fire_detect},
    # {"name" : "feat_detect_smoke", "target" : run_smoke_detect},
    # {"name" : "feat_detect_trash", "target" : run_trash_detect},
    # {"name" : "feat_detect_violence", "target" : run_violence_detect},
    # {"name" : "feat_detect_weapon", "target" : run_weapon_detect},
]
NUM_ANALYZERS = len(ANALYZER_CONFIG)

# dM TCP 영상 수신 프레임 메타 데이터 (브로드캐스팅)
# 영상 수신시 첫번째 메타 데이터를 해당 영상의 메타 데이터로 활용
DM_MEDIA_TCP_META = {
    "source" : 0x01,
    "destination" : 0x02,
    "patrol_number" : 1,
    "timestamp" : None,
    "deviceStatus" : None,
    # "videoSize" : None,
    # "video" : None,
}


def main():
    """
    TCP, UDP, YOLO 처리 스레드 생성 및 관리
    """
    # 모든 스레드를 동시에 종료하기 위한 Event
    db_tcp_connected_event = threading.Event()
    shutdown_event = threading.Event()

    # --- 생산자 - 소비자 (Producer-Customer) 패턴을 이용한 영상 데이터 분베 ---
    # 1. 소비자 (영상 프레임을 소비하는 6가지 분석 기능 + GUI 픽셀 데이터 전송 UDP Sender)를 위한 입력 큐 리스트 정의
    analyzer_input_queue = [queue.Queue(maxsize=10) for _ in range(NUM_ANALYZERS)]
    db_send_frame_queue = queue.Queue(maxsize=10) # db 전송을 위한 프레임 큐 

    # 2. 분석 결과를 취합하는 결과 큐 생성
    final_output_queue = queue.Queue()

    # # 스레드 간 프레임 공유를 위해 큐 생성
    # frame_output_queue = queue.Queue(maxsize=PIXEL_THREAD * 10)
    # # 프레임을 7개 스레드에 나누어주어야 함 (6개 분석 스레드 + GUI 전송 영상)
    
    event_video_queue = queue.Queue(maxsize = 20) # 이벤트 비디오 큐 : TCP 프레임 수신 및 dataService로 즉시 전송
    db_manager_queue = queue.Queue(maxsize = 20) # 분석 결과 큐

    # thread의 start, join을 일괄로 하기 위해 각 스레드를 리스트로 저장
    threads = []

    # TCP 메타데이터 공유를 위한 딕셔너리 + Lock
    # 가장 최근 프레임의 메타데이터를 저장
    shared_metadata = {
        "timestamp" : None,
        "patrol_car_name" : "Unknown"
    }
    metadata_lock = threading.Lock()
    

    # 1. deviceManager TCP 수신 스레드 생성
        # json 영상 메타데이터 수신
    dm_tcp_receiver_thread = threading.Thread(
        target=receive_event_video,
        args=(shared_metadata, metadata_lock, shutdown_event,),
        daemon=True
    )
    threads.append(dm_tcp_receiver_thread)

    # 2. deviceManager UDP 수신 스레드 생성
        # 픽셀 데이터 20조각을 받아 조립하여 1개 프레임 완성 및 원본 영상 큐에 추가
        # 수신한 프레임을 분석 프레임 큐에 추가
        # 수신한 프레임을 db서버 전송 프레임 큐에 추가
    dm_udp_receiver_thread = threading.Thread(
        target=receive_video_udp,
        args=(analyzer_input_queue, db_send_frame_queue, shutdown_event,),
        daemon=True
    )
    threads.append(dm_udp_receiver_thread)

    # 3. 픽셀 데이터를 소비하는 6개 분석 소비자 스레드 생성
        # 특징 : for문과 ANALYZER_CONFIG의 인덱스를 통해 픽셀 데이터를 각 분석모델 스레드에 분배함
    for i, config in enumerate(ANALYZER_CONFIG):
        analyzer = threading.Thread(
            target = config["target"],
            args = (
                analyzer_input_queue[i],
                final_output_queue,
                config["name"],
                shutdown_event
            )
        )
        threads.append(analyzer)


    # # 3. YOLO 처리 스레드 생성
    #     # 원본 영상 큐에서 프레임을 꺼내 분석
    #     # 분석 결과(json)을 db_manager_queue 큐에 저장
    # yolo_thread = threading.Thread(
    #     target=detect_objects,
    #     args=(analysis_frame_queue, db_manager_queue, shared_metadata, metadata_lock, shutdown_event,),
    #     daemon=True
    # )

    # 4. DB TCP 전송 스레드 생성
        # 1. TODO 각 프레임에 대해서 6개 기능을 사용해 분석 완료한 json 데이터 전송
            # 현재구현 : firedetection 1개 기능에 대한 분석
        # 2. TODO 이벤트 전후 15초를 취합하여 생성한 영상 데이터의 메타데이터 전송
    main_tcp_sender_thread = threading.Thread(
        target=send_tcp_data_to_main,
        args = (db_manager_queue, db_tcp_connected_event, shutdown_event,),
        daemon=True
    )
    threads.append(main_tcp_sender_thread)

    # 5. DB UDP 전송 스레드 생성 
        # 2. TODO 이벤트 전후 15초를 취합하여 생성한 영상 데이터 전송
            # 현재 구현 : 받아온 픽셀 데이터를 그대로 dataService에 정송
    main_udp_sender_thread = threading.Thread(
        target=send_udp_frame_to_main,
        args = (db_send_frame_queue, db_tcp_connected_event, shutdown_event,),
        daemon=True
    )
    threads.append(main_udp_sender_thread)

    # --- 스레드 시작 및 종료 일괄 처리 ---
    try:
        print("situationDetector Main : 서비스의 수신 스레드를 시작합니다.")
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 모든 스레드가 살아있는 동안 메인 스레드 대기
        while all(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nsituationDetector Main : 종료 신호를 감지. 모든 스레드 정리")
        # 모든 스레드에 종료 신호 전송
        shutdown_event.set()
    finally:
        # 4. 스레드가 완전히 종료될 때까지 대기
        # 각 스레드가 정리 작업을 마칠 시간 보장함
        for t in threads:
            t.join()
        print("situationDetector Main : 모든 스레드 종료. 프로그램 종료")

if __name__ == "__main__":
    main()

"""
이슈

2. 픽셀 데이터 복사 문제 : 
    - main 스레드에서 6개 분석 모델 + GUI로 보낼 원본 영상 데이터를 배분해 주어야 하는 상황인데 7개 copy를 만드는 것은 속도 측면에서 좋지 않아보이는 상황인데, 
    다음과 같이 flag를 이용해서 처리 완료 표시를 하고 1개의 copy로만 문제를 해결하는것이 좋은 방법인지
    FRAME_PROCESS_THREAD = [ # 
    {"name" : "udp_thread", "process_flag" : 0}, # 1 : 처리완료, 0 : 처리 전
    {"name" : "fire_process_thread", "process_flag" : 0}, # 1 : 처리완료, 0 : 처리 전
]
"""