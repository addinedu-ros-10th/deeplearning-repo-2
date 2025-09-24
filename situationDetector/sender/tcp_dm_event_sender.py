import socket
import time
import queue
import threading

"""
작업 진행 상황
1. deviceManager -> situationDetector 으로 이벤트 영상을 수신하는 코드는 마무리 단계
2. situationDetector -> deviceManager 으로 이벤트 영상 생성 요청 정보를 발신하는 통신 코드가 없어서 진행중
  - 단순 테스트 데이터 발신하는 코드 구현중
  - 모델 1개를 이용한 실제 데이터를 이용해서 이벤트를 발생시키는 코드 작성 및 향후 6개로 통합 예정
3. situationDetector -> dataService 으로 보낼 모델 해석 데이터를 만드는 껍데기 코드 완성 및 1개 기능에 대해서 테스트 완료
  - 단순 통신 (발신) 기능만 추가하면 완성 단계

situationDetector -> deviceManager 순찰 이벤트 설정 기능 구현

TODO
1. 통신기능 구현 마무리 / 포트번호 통일 등의 작업 마무리
2. 성경님 gui -> dM 바이의 기능 통합
"""


TCP_HOST = '192.168.0.86' # deviceManager IP 주소
TCP_PORT = 1201            # deviceManager TCP 발신 포트 주소

def send_event_data_to_dm(dm_event_queue : queue.Queue,
                        shutdown_event: threading.Event):
  """
  모델 분석 결과를 해석해서 이벤트 발생 데이터를 deviceManager에 발신하는 기능
  1. source
  2. Destination
  3. Patrol number
  4. Alarm type  
  """
  # 동영상 생성 테스트 : 단순 정보를 10초마다 발생
  # 예시 1: 알람 없음 (Alarm type: 0)
  # [Source][Destination][Patrol number][Alarm type]
  # [ 0x02 ][  0x01     ][    0x01     ][   0x00   ]
  TEST_DATA = b'\x02\x01\x01\x00'

  # 예시 2: 화재 경고 (Alarm type: 1)
  # [Source][Destination][Patrol number][Alarm type]
  # [ 0x02 ][  0x01     ][    0x01     ][   0x01   ]
  TEST_DATA_FIRE = b'\x02\x01\x01\x01'
  
  while not shutdown_event.is_set():
    sock = None
    try:
      # dM 서버에 연결 시도
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      print(f"situationDetector (TCP dM Sender) : dM 서버({TCP_HOST}:{TCP_PORT})에 연결을 시도합니다.")
      sock.connect((TCP_HOST, TCP_PORT))
      print(f"situationDetector (TCP dM Sender) : dM 서버에 연결되었습니다.")

      time.sleep(15) # 이벤트 영상 생성 대기

      # 연결이 유지되는 동안 큐에서 데이터 전송
      while not shutdown_event.is_set():
        try:
          sock.send(TEST_DATA_FIRE)
          print(f"situationDetector (TCP dM Sender) : 데이터 전송 완료 ({len(TEST_DATA_FIRE)} bytes)")
        except queue.Empty:
          # 큐가 비어있으면 계속 대기
          continue
        except socket.error as e:
          # 소켓 오류 발생 시 연결 재시도
          print(f"situationDetector (TCP dM Sender) : 소켓 오류 발생: {e}. 재연결을 시도합니다.")
          break # 내부 루프를 빠져나가 외부 루프에서 재연결 시도
    
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
      print(f"situationDetector (TCP dM Sender) : dM 서버 연결 실패: {e}")
    
    finally:
      if sock:
        sock.close()
      # 재연결 시도 전 잠시 대기
      if not shutdown_event.is_set():
        print("situationDetector (TCP dM Sender) : 5초 후 재연결을 시도합니다.")
        time.sleep(5)

  print("situationDetector (TCP dM Sender) : dM 전송 스레드를 종료합니다.")