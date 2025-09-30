# situationDetector.py
import threading
import time
import queue

from situationDetector.receiver.broadcast_receiver import receive_cv_video

# 통합된 통신 모듈 import
from situationDetector.server.tcp_dm_server import dm_server_run
# from situationDetector.server.tcp_gui_server import gui_server_run_archive
from situationDetector.server.tcp_gui_server import gui_server_run
from situationDetector.server.tcp_ds_server import ds_server_run

from situationDetector.detect.detect_fire import run_fire_detect
# from situationDetector.detect.find_missing import run_find_missing
from situationDetector.detect.detect_fall import run_fall_detect
from situationDetector.detect.detect_smoke import run_smoke_detect

from situationDetector.videoManager import make_event_video

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
    # 1. 프레임을 소비하는 스레드 초기화
    self.ANALYZER_CONFIG = [
        # 분석 모델 스레드 (6가지)
        # {"name" : "feat_detect_missing_person", "target" : run_find_missing}, # (완료)
        
        # 
        # {"name" : "feat_detect_fall", "target" : run_fall_detect}, # (GUI 방송 기능만 되면 완료)
        # {"name" : "feat_detect_fire", "target" : run_fire_detect}, # (완료)

        # deviceManager 방송 기능만 되면 완료
        {"name" : "feat_detect_smoke", "target" : run_smoke_detect}, # (GUI 방송 기능만 되면 완료)
        
        
        # {"name" : "feat_detect_trash", "target" : run_trash_detect}, # ()
        # {"name" : "feat_detect_violence", "target" : run_violence_detect}, # (GUI)
    ]
    
    # 2. 최종 분석 결과를 받는 클라이언트 목록
    self.CLIENTS_CONFIG = [
      {"target" : gui_server_run},
      {"target" : ds_server_run},
    ]

    # 분석기 이름과 알람 타입 ID 매핑 (알람 30초 무시 요청 처리)
    self.ANALYZER_TO_ALARM_TYPE = {
        "feat_detect_fall": 0x00,
        "feat_detect_fire": 0x01,
        "feat_detect_violence": 0x02,
        "feat_detect_missing_person": 0x03,
        "feat_detect_smoke": 0x04,
        "feat_detect_trash": 0x05,
    }
    
    self.NUM_ANALYZERS = len(self.ANALYZER_CONFIG)
    self.NUM_CLIENTS = len(self.CLIENTS_CONFIG)
    
    # 2. 스레드 공유 이벤트 변수 초기화
    self.shutdown_event = threading.Event()
    
    # 3. 공유 데이터 큐 초기화
      # analyzer_input_queues 데이터 형식
      # (<프레임 카운트>, <영상 시각>, <프레임 데이터>)
    self.analyzer_input_queues = [queue.Queue(maxsize=10) for _ in range(self.NUM_ANALYZERS)]
    # 이벤트 데이터 생성 프레임 데이터 큐 초기화
      # (<프레임 카운트>, <영상 시각>, <프레임 데이터>)
    self.raw_frame_queue = queue.Queue(maxsize=10)    
    self.aggregation_queue = queue.Queue() # 분석 결과를 모으는 큐
    
    self.video_manager_event_queue = queue.Queue() # VideoManager 전용 이벤트 큐
    self.final_output_queue = queue.Queue() # # DeviceManager 전용 이벤트 큐 (6가지 기능에 대한 분석이 완료될 시 저장)
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
    
    # [2. 30초 재인식 방지 기능]
    # [기능] 한번 인식 / 전송한 이벤트는 30초간 재인식하지 않도록 관리하기 위한 딕셔너리
    # 형식: { alarm_type: end_time }
    self.received_events = {}

    self.threads = []

    # # dataService 데모 영상 전송 테스트
    # try:
    #     VIDEO_PATH = "situationDetector/test_data/test.mp4"
    #     with open(VIDEO_PATH, "rb") as f:
    #     # test_video = (video_metadata, video_size, video_file)
    #         video_buffer_test = f.read()
        
    #     video_size_test = len(video_buffer_test)
    #     # 테스트용 메타데이터 생성
    #     video_metadata_test = {
    #         "source": 0x01, # dM
    #         "destination" : 0x02, # sD
    #         "patrol_number" : 1, # Test patrol
    #         "timestamp" : time.time(), # Example timestamp
    #     }
        
    #     test_video_item = (video_metadata_test, video_size_test, video_buffer_test)
    #     self.event_video_queue.put(test_video_item)
    #     print(f"situationDetector (TCP dM Communicator) : 데모 영상 ({video_size_test} bytes)을 event_video_queue에 추가했습니다.")
    # except FileNotFoundError:
    #     print(f"situationDetector (TCP dM Communicator) : 데모 영상 파일을 찾을 수 없습니다: {VIDEO_PATH}")
    # except Exception as e:
    #     print(f"situationDetector (TCP dM Communicator) : 데모 영상 로딩 중 오류: {e}")




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
        # 이번 처리 주기에서 GUI 해제 이벤트가 수신되었는지 확인하기 위한 플래그
          # True일 경우에 감지된 객체가 0개인 경우에도 dM으로 보내는 이벤트 해제 신호를 생성하도록 함
        clear_event_received_this_cycle = False
        
        # [1. 알람 해제 기능] 이벤트 해제 큐 확인
        try:
          current_time = time.time()
          
          clear_request = self.event_clear_queue.get_nowait()
          alarm_type_to_clear = clear_request.get("alarm_type")
          if alarm_type_to_clear is not None:
            print(f"situationDetector (Aggregator): {alarm_type_to_clear} 타입 알람 해제 요청 수신. 30초간 무시합니다.")
            # 현재 시간 + 30초로 무시 종료 시간 설정
            self.ignore_events[alarm_type_to_clear] = current_time + 30.1
            # 해제 이벤트가 수신되었음을 플래그에 표시
            clear_event_received_this_cycle = True
        except queue.Empty:
            pass # 처리할 해제 요청 없음

        # [1. 알람 해제 기능] 만료된 무시 이벤트 정리
        current_time = time.time()
        expired_alarms = [alarm for alarm, expiry_time in self.ignore_events.items() if current_time > expiry_time]
        for alarm in expired_alarms:
          print(f"situationDetector (Aggregator): {alarm} 타입 알람 무시 기간 만료.")
          del self.ignore_events[alarm] # GUI 이벤트 해제 요청 삭제

        # [2. 30초 재인식 방지 기능] 만료된 재인식 방지 이벤트 정리
        expired_received = [alarm for alarm, expiry_time in self.received_events.items() if current_time > expiry_time]
        for alarm in expired_received:
          print(f"situationDetector (Aggregator): {alarm} 타입 이벤트 재인식 방지 기간 만료.")
          del self.received_events[alarm] # 재인식 방지 데이터 정보 삭제
      
        # result_package 언패킹
        result_package = self.aggregation_queue.get(timeout=1.0)

        timestamp = result_package["timestamp"] # 시간
        analyzer_name = result_package["analyzer_name"] # 기능 이름
        
        # [1. 알람 해제 기능] 현재 분석 결과가 무시 대상인지 확인
        alarm_type = self.ANALYZER_TO_ALARM_TYPE.get(analyzer_name)
        if alarm_type is not None and alarm_type in self.ignore_events:
          # 감지 결과를 0으로 만들어 무시 처리
          # print(f"situationDetector (Aggregator): 활성화된 해제 요청에 따라 {analyzer_name}의 감지 결과를 무시합니다.")
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
              
              # [2. 30초 재인식 방지 기능] 만료된 재인식 방지 이벤트 목록에 있으면 저장하지 않고 넘어감
              current_alarm_type = self.ANALYZER_TO_ALARM_TYPE.get(name)
              if current_alarm_type is not None and current_alarm_type in self.received_events:
                  # print(f"situationDetector (Aggregator): {name} 이벤트는 재인식 방지 기간이므로 전송하지 않습니다.")
                  continue # 재인식 방지 기간이므로 final_detections에 추가하지 않고 다음 결과로 넘어감
                
              # 6-1. 현재 기능의 detection 데이터 리스트 가져오기
              detection_list = result["detection"]
              
              # 6-2. AI_JSON Schema와 동일하게 기능별 감지 갯수 추가
              detection_list.append({
                "detection_count" : result["detection_count"], # 감지 수
              })

              # 6-3. 완성된 기능별 detect 데이터를 final_detections 딕셔너리에 저장
              final_detections[name] = detection_list

              # [2. 30초 재인식 방지 기능] 새로운 이벤트이므로 재인식 방지 목록에 추가
              if current_alarm_type is not None:
                print(f"situationDetector (Aggregator): {name} 이벤트 감지. 30초간 재인식 방지를 시작합니다.")
                self.received_events[current_alarm_type] = time.time() + 30.1

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
          self.video_manager_event_queue.put(agg_json_data)   # VideoManager 전용 이벤트 큐


          # self.final_output_queue.put(agg_json_data)          # DeviceManager 전용 이벤트 큐
          # 조건 1: 감지된 이벤트가 있을 때 (final_detections가 비어있지 않음)
          # 조건 2: 이번 주기에 GUI 해제 신호를 받았을 때 (clear_event_received_this_cycle이 True)
          if final_detections or clear_event_received_this_cycle: # final_detections 딕셔너리가 비어있지 않을 때만 True
            self.final_output_queue.put(agg_json_data)
            print(f"DEBUG [Aggregator]: Event detected! Sending to dM queue. Data: {agg_json_data}")

          
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
      args=(self.analyzer_input_queues, self.raw_frame_queue, self.shutdown_event),
      daemon=True
    )
    self.threads.append(broadcaster_receiver_thread)

    # 2. deviceManager 양방향 통신 스레드 (영상 수신 및 이벤트 송신)
    dm_communicator_thread = threading.Thread(
      target=dm_server_run,
      args=(self.event_video_queue, # 이벤트 영상 송신
            self.final_output_queue,
            self.ignore_events, # 알람 이벤트가 살아있는동안은 deviceManager에 255신호 전송
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

    # 7. 이벤트 발생 시 전후 30초 영상을 생성하는 스레드
    video_manager_thread = threading.Thread(
        target=make_event_video,
        args=(self.video_manager_event_queue,     # 이벤트 감지용 큐
              self.raw_frame_queue,        # 원본 영상 프레임 수신용 큐
              self.event_video_queue,      # 생성된 이벤트 영상을 보낼 큐
              self.shutdown_event),
        daemon=True
    )
    self.threads.append(video_manager_thread)

    # # 7. [추가] 이벤트 영상 큐 감시 및 파일 저장 스레드 (디버깅용)
    # video_saver_thread = threading.Thread(
    #   target=self._save_event_video_from_queue,
    #   daemon=True
    # )
    # self.threads.append(video_saver_thread)

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

# -------------------------------------------------------------------------------
# 디버깅 영상 저장

  def _save_event_video_from_queue(self):
    """
    self.event_video_queue를 지속적으로 감시하고,
    영상 데이터가 들어오면 지정된 경로에 파일로 저장하는 스레드 함수.
    (테스트 및 디버깅 목적)
    """
    print("situationDetector (Video Saver): 영상 저장 스레드 시작. 큐를 감시합니다.")
    while not self.shutdown_event.is_set():
      try:
        # 1. 큐에서 영상 아이템 가져오기 (데이터가 들어올 때까지 여기서 블로킹/대기)
        video_metadata, video_size, video_buffer = self.event_video_queue.get(timeout=1.0)
        
        # 2. 저장할 경로 지정
        SAVE_VIDEO_PATH = "situationDetector/test_data/test2.mp4"
        
        print(f"situationDetector (Video Saver): 큐에서 {video_size} 바이트 영상 수신. 저장을 시작합니다.")

        # 3. 지정된 경로에 바이너리 쓰기("wb") 모드로 파일 저장
        with open(SAVE_VIDEO_PATH, "wb") as f:
          f.write(video_buffer)
        
        print(f"situationDetector (Video Saver): 영상을 성공적으로 저장했습니다. -> {SAVE_VIDEO_PATH}")

      except queue.Empty:
        # timeout(1.0초) 동안 큐에 데이터가 없으면 예외 발생. 정상적인 상황임.
        continue
      except Exception as e:
        print(f"situationDetector (Video Saver): [에러] 영상 파일 저장 중 오류 발생: {e}")
        # 오류가 발생해도 스레드는 계속 실행
    
    print("situationDetector (Video Saver): 영상 저장 스레드 종료.")

# -------------------------------------------------------------------------------









if __name__ == "__main__":
  sd = SituationDetector()
  sd.run()



