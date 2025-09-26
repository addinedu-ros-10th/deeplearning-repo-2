# situationDetector.py
import threading
import time
import queue

from situationDetector.receiver.broadcast_receiver import receive_cv_video

# 
from situationDetector.sender.tcp_main_sender import send_tcp_data_to_main
from situationDetector.sender.udp_main_sender import send_udp_frame_to_main

# 통합된 통신 모듈 import
from situationDetector.server.tcp_dm_server import dm_server_run
# from situationDetector.server.tcp_gui_server import gui_server_run_archive
from situationDetector.server.tcp_gui_server import gui_server_run
from situationDetector.server.tcp_ds_server import ds_server_run


from situationDetector.detect.feat_detect_fire import run_fire_detect

# from situationDetector.detect.feat_detect_fall import run_fall_detect
# from situationDetector.detect.feat_detect_smoke import run_smoke_detect
# from situationDetector.detect.feat_detect_trash import run_trash_detect
# from situationDetector.detect.feat_detect_violence import run_violence_detect
# from situationDetector.detect.feat_detect_weapon import run_weapon_detect

from situationDetector.test_data.test_constant import TEST_AI_JSON


"""
situationDetector 클래스

주요 기능

1. 세부 기능, 통신은 각 모듈의 1~3개의 간단한 함수로 구현
2. 영상 프레임 데이터를 GUI 전송용 + 6개 기능 구동용으로 7개로 분배해주는 구현에서
생산자 - 소비자 (Producer-Customer) 패턴 + 큐 조합으로 구현
3. 분석결과 취합 : 각 프레임에 대해서 분석 결과가 모두 모였는지 확인하여, 6가지 기능에 대한 분석 결과가 모두 모였으면 Main Server로 전송할 Json 데이터를 생성함
"""

class SituationDetector:
  """
  
  """
  def __init__(self):
    """
    SituationDetector의 공유 큐, 이벤트, 스레드 리스트 (StateMachine) 등 초기화
    """
    # 1. 분석 모델 설정 초기화
    self.ANALYZER_CONFIG = [
        # {"name" : "feat_detect_fall", "target" : run_fall_detect},
        {"name" : "feat_detect_fire", "target" : run_fire_detect},
        # {"name" : "feat_detect_smoke", "target" : run_smoke_detect},
        # {"name" : "feat_detect_trash", "target" : run_trash_detect},
        # {"name" : "feat_detect_violence", "target" : run_violence_detect},
        # {"name" : "feat_detect_weapon", "target" : run_weapon_detect},
    ]
    
    # 2. 최종 분석 결과를 받는 클라이언트 목록
    self.CLIENTS_CONFIG = [
      {"target" : gui_server_run},
      {"target" : ds_server_run},
    ]

    # 분석기 이름과 알람 타입 ID 매핑 (알람 30초 무시 요청 처리)
    self.ANALYZER_TO_ALARM_TYPE = {
        "feat_detect_fire": 0x00,
        "feat_detect_fall": 0x01,
        "feat_detect_smoke": 0x02,
        "feat_detect_trash": 0x03,
        "feat_detect_violence": 0x04,
        "feat_detect_weapon": 0x05,
    }
    
    self.NUM_ANALYZERS = len(self.ANALYZER_CONFIG)
    self.NUM_CLIENTS = len(self.CLIENTS_CONFIG)
    
    # 2. 스레드 공유 이벤트 변수 초기화
    self.shutdown_event = threading.Event()
    
    # 3. 공유 데이터 큐 초기화
      # analyzer_input_queues 데이터 형식
      # (<프레임 카운트>, <영상 시각>, <프레임 데이터>)
    self.analyzer_input_queues = [queue.Queue(maxsize=10) for _ in range(self.NUM_ANALYZERS)]
    self.aggregation_queue = queue.Queue() # 분석 결과를 모으는 큐
    self.final_output_queue = queue.Queue() # Main Server에 전송할 큐 (6가지 기능에 대한 분석이 완료될 시 저장)
    self.dm_event_queue = queue.Queue() # final_output_queue와 동일한 데이터를 dM으로 보내는 신호 생성용 큐

    # 4. 모델 분석 결과 데이터 초기화
    self.final_output_queues = [queue.Queue(maxsize=10) for _ in range(self.NUM_CLIENTS)]

    
    # GUI로부터 수신하는 경고 방송 해지신호 큐
      # dm_event_queue에서 감지된 데이터를 제거 처리함 (30초동안)
    self.event_clear_queue = queue.Queue() 
    
    # 4. 30초 이벤트 비디오 영상 큐 초기화
    self.event_video_queue = queue.Queue()

    # [기능] GUI 이벤트 해제 요청을 관리하기 위한 딕셔너리
    # 형식: { alarm_type: end_time }
    self.ignore_events = {}

    self.threads = []

  def aggregate_results(self):
    """
    1. 6가지 분석 스레드의 결과 스레드를 취합하여 하나의 Json으로 만듦
    2. 6가지 분석이 완전히 끝났을 때에만 Main Server로 전송하기 위한 별도의 스레드 함수
    (분석 결과 큐) -> (6개 분석결과 취합 큐)
    
    [추가 기능]
    - GUI로부터 특정 기능에 대한 알람 해제 요청이 오면 30초간 해당 기능의 감지 결과를 무시함.
    
    data_json = {
      detection : 
        "feat_detect_fire" : [
          {
          "class_id" : # 순찰 이벤트 id
          "class_name" : # 순찰 이벤트 이름
          "confidence" : # detection 신뢰도
          "bbox" : {
            "x1" :
            "y1" :
            "x2" :
            "y2" :
          }
          detection_count,
        ],
        "feat_detect_violance" : [
          {
            <이벤트 정보>
          },
          detection_count,
        ]
      }
      timestamp,
      patrol_number,
    }
    """
    print("situationDetector (Aggregator) : 취합 스레드 시작")
    results_buffer = {} # 1. 타임스탬프를 key로 하여 결과 저장
    
    while not self.shutdown_event.is_set():
      try:
        # [알람 해제 기능] 이벤트 해제 큐 확인
        try:
          clear_request = self.event_clear_queue.get_nowait()
          alarm_type_to_clear = clear_request.get("alarm_type")
          if alarm_type_to_clear is not None:
                print(f"situationDetector (Aggregator): {alarm_type_to_clear} 타입 알람 해제 요청 수신. 30초간 무시합니다.")
                # 현재 시간 + 30초로 무시 종료 시간 설정
                self.ignore_events[alarm_type_to_clear] = time.time() + 30
        except queue.Empty:
            pass # 처리할 해제 요청 없음

        # [알람 해제 기능] 만료된 무시 이벤트 정리
        current_time = time.time()
        expired_alarms = [alarm for alarm, expiry_time in self.ignore_events.items() if current_time > expiry_time]
        for alarm in expired_alarms:
            print(f"situationDetector (Aggregator): {alarm} 타입 알람 무시 기간 만료.")
            del self.ignore_events[alarm]

      
        # result_package 언패킹
        result_package = self.aggregation_queue.get(timeout=1.0)

        timestamp = result_package["timestamp"] # 시간
        analyzer_name = result_package["analyzer_name"] # 기능 이름
        
        # [알람 해제 기능] 현재 분석 결과가 무시 대상인지 확인
        alarm_type = self.ANALYZER_TO_ALARM_TYPE.get(analyzer_name)
        if alarm_type is not None and alarm_type in self.ignore_events:
            # 감지 결과를 0으로 만들어 무시 처리
            print(f"situationDetector (Aggregator): 활성화된 해제 요청에 따라 {analyzer_name}의 감지 결과를 무시합니다.")
            result_package["detection_count"] = 0
            result_package["detection"] = []
        
        # 2. 버퍼에 해당 타임스탬프가 없으면 새로 생성
        if timestamp not in results_buffer:
          results_buffer[timestamp] = {}
        
        # 3. 현재 분석 결과를 버퍼에 저장
        results_buffer[timestamp][analyzer_name] = result_package

        # 4. 모든 분석 결과 (6가지)가 모였는지 확인
        if len(results_buffer[timestamp]) == self.NUM_ANALYZERS:
          # print(f"situationDetector (Aggregator) : {timestamp} 에 대한 모든 결과 취합 완료.")

          final_detections = {}
          # 5. 순찰차 이름은 기능에 관계없이 동일하므로 하나만 참조
          patrol_number = result_package["patrol_number"]
          
          # 6. 감지 정보를 모든 버퍼에서 detection 리스트 확인 및 1개 json에 추가
          for name, result in results_buffer[timestamp].items():
            if result["detection_count"] > 0:
              
              # 6-1. 현재 기능의 detection 데이터 리스트 가져오기
              detection_list = result["detection"]
              
              # 6-2. AI_JSON Schema와 동일하게 기능별 감지 갯수 추가
              detection_list.append({
                "detection_count" : result["detection_count"], # 감지 수
              })

              # 6-3. 완성된 기능별 detect 데이터를 final_detections 딕셔너리에 저장
              final_detections[name] = detection_list

          # 5. 최종 json 데이터 생성
          agg_json_data = {
            # detection 데이터 : 기능 이름 () - 감지한 객체 리스트 저장
            "detection" : final_detections,
            "timestamp" : timestamp, # 시간
            "patrol_number" : patrol_number,
          }
          
          # # 테스트코드 : 최종 json 데이터 단순출력
          # print(agg_json_data)

          # 6. json 형태로 final_output_queue에 저장
          self.final_output_queue.put(agg_json_data)
          
              # analyzer_input_queue에 프레임 저장
          for q in self.final_output_queues:
            if q.full():
              q.get() # 큐가 꽉 차있으면 이전 프레임을 버리고 새 것으로 교체 작업
            q.put(agg_json_data)
          
          # TODO: dm_event_queue에 어떤 데이터를 넣을지 정책 수립 필요
          # self.dm_event_queue.put(b'\x02\x01\x01\x01') # 예시
          
          # 7. 처리 완료된 타임스탬프는 버퍼에서 제거
          del results_buffer[timestamp]
          
      except queue.Empty:
        continue
      except Exception as e:
        print(f"situationDetector (Aggregator) : 처리 중 오류 발생: {e}")
        break
    print("situationDetector (Aggregator) : 취합 스레드 종료")

  def setup_thread(self):
    """
    SituationDetector의 모든 스레드 초기화 및 리스트에 추가하는 초기 작업 수행
    """
    # 1. realtimeBroadcaster 실시간 영상 수신 스레드
    broadcaster_receiver_thread = threading.Thread(
      target=receive_cv_video,
      args=(self.analyzer_input_queues ,self.shutdown_event),
      daemon=True
    )
    self.threads.append(broadcaster_receiver_thread)

    # 2. DeviceManager 양방향 통신 스레드 (영상 수신 및 이벤트 송신)
    dm_communicator_thread = threading.Thread(
      target=dm_server_run,
      args=(self.event_video_queue, 
            self.final_output_queue,
            self.shutdown_event),
      daemon=True
    )
    self.threads.append(dm_communicator_thread)

    # 3. GUI 순찰 해제 통신 스레드
      # 1. 모델 분석 결과 송신
      # 2. 이벤트 수신
    ds_server_thread = threading.Thread(
      # target = self.CLIENTS_CONFIG[1]["target"],
      target = gui_server_run,
      args=(self.final_output_queues[0],
            self.event_clear_queue,
            self.shutdown_event),
      daemon=True
    )
    self.threads.append(ds_server_thread)

    # 4. dataService 모델 분석 결과 / 영상 데이터 통신 스레드
      # 1. 30초 이벤트 영상 송신
      # 2. 모델 분석 결과 송신
    main_tcp_sender_thread = threading.Thread(
      target=ds_server_run,
      args=(self.event_video_queue,
            self.final_output_queues[1], 
            self.shutdown_event),
      daemon=True
    )
    self.threads.append(main_tcp_sender_thread)

    # 5. N개 분석 스레드 - 소비자
    for i, config in enumerate(self.ANALYZER_CONFIG):
      analyzer_thread = threading.Thread(
        target=config["target"],
        args=(self.analyzer_input_queues[i], 
              self.aggregation_queue, 
              config["name"], 
              self.shutdown_event),
        daemon=True
      )
      self.threads.append(analyzer_thread)

    # 6. 결과 취합 스레드 (Aggregator)
    aggregator_thread = threading.Thread(
      target = self.aggregate_results,
      daemon=True
    )
    self.threads.append(aggregator_thread)



  def run(self):
    self.setup_thread()
    try:
      print("situationDetector Main : 서비스의 모든 스레드를 시작합니다.")
      for t in self.threads:
        t.start()
      
      # 메인 스레드를 종료하지 않고 대기하도록 설정
      while not self.shutdown_event.is_set():
        time.sleep(1)
    except KeyboardInterrupt:
      print("situationDetector Main : 종료 신호 감지. 모든 스레드를 정리합니다.")
    finally:
      self.stop()
    
  def stop(self):
    self.shutdown_event.set()
    print("\nsituationDetector Main : 모든 스레드의 종료를 기다립니다...")
    for t in self.threads:
      t.join(timeout = 5) # 5초간 스레드가 종료되지 않으면 넘어감
    print("situationDetector Main : 모든 스레드 종료. 프로그램을 종료합니다.")

if __name__ == "__main__":
  sd = SituationDetector()
  sd.run()