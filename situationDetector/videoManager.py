import cv2
import queue
import threading
import os
from datetime import datetime

# final_output_queue SCHIMA (인출, dequeue): AI_JSON
"""
## AI_JSON 데이터 형식 예제
videometa = {
  detection: {
    'feat_detect_smoke' :
      [
        {
          "class_id": 1,               ## 순찰 이벤트 id
          "class_name": "smoking", ## 순찰 이벤트 이름
          "confidence": 1.,            ## detection 신뢰도
          "bbox": {                    ## Box 표기 위치
            "x1": 170.,
            "y1": 481.2,
            "x2": 123.4,
            "y2": 456.7,
            }
        },
        detection_count: 1, # 기능별 감지 갯수 (카운트)
      ],
    'feat_detect_fire' :
      [
        {
          객체 정보
        },
        detection_count: 2, # 기능별 감지 갯수 (카운트)
      ]
    ...
  timestamp: "2025-09-18 14:00:00",
  patrol_number: 1, # 순찰차 이름은 1로 고정
}
"""

# raw_frame_queue 형식 (인출, dequeue) :data_packet
"""
1. raw_frame_queue (예시)
data_packet = {
  "frame" : frame,
  "frame_count" : frame_count,
  "frame_time" : frame_time,
  "patrol_number" : 0x00,
}

"""

# event_video_queue 형식 (삽입, Enqueue) : (event_video_metadata, event_video)
"""
event_video_metadata : AI_JSON과 동일
event_video : (video_size, video_buffer)
  - video_size : 비디오 사이즈 : int
  - video_buffer : 비디오 원본 픽셀 데이터 : bytes
"""

def make_event_video(final_output_queue :queue.Queue,    # 취합 큐
                    raw_frame_queue : queue.Queue, # 원본 스트림 영상
                    event_video_queue: queue.Queue,    # 이벤트 비디오를 저장하는 공유 큐
                    shutdown_event: threading.Event):
  """
  기능 : 모델 해석 결과를 받아서 이벤트 비디오 생성
  1. final_output_queue는 스트림 데이터
  
  (final_output_queue , event_video_queue)
"""
  # --- 기존 deviceManager의 변수 및 설정값 ---
  before_video = list()
  after_video = list()
  capture_time = 15  # 초 단위
  capture_frame = 20 # 초당 프레임
  
  # 이벤트 발생 상태를 관리하는 변수
  alarm = 0
  # 현재 처리 중인 이벤트의 메타데이터를 저장하는 변수
  current_event_data = None
  
  # 생성된 비디오를 임시 저장할 경로 설정
  file_path = 'situationDetector/event_videos'
  os.makedirs(file_path, exist_ok=True)
  
  print("situationDetector (Video Manager): 이벤트 영상 생성 스레드가 시작되었습니다.")

  while not shutdown_event.is_set():
    # 1. 영상 프레임 가져오기 (기존 cap.read() 부분을 대체)
    try:
      # raw_frame_queue에서 data_packet 딕셔너리 가져옴
      data_packet = raw_frame_queue.get(timeout=1.0)
      # 실제 프레임 데이터 가져오기
      frame = data_packet.get("frame")
      if frame is None:
        continue
    except queue.Empty:
      # 1초 동안 프레임이 없으면 루프 계속
      continue

    # 2. alarm 상태에 관계없이
    # before_video 프레임이 완전하지 않으면 롤링 버퍼 채우기 동작
    # if len(before_video) < capture_frame * capture_time:
    #   before_video.append(frame)
    # elif len(before_video) >= capture_frame * capture_time:
    #   before_video.pop(0)
    #   before_video.append(frame)
    # alarm이 0일 때만 before_video 롤링 버퍼를 업데이트
    if alarm == 0:
      if len(before_video) >= capture_frame * capture_time:
        before_video.pop(0)
      before_video.append(frame)
      
    # # 2. [오류 방어 코드] : 시작 후 15초간 사전 영상 버퍼를 채운 후에 동작
    # if len(before_video) < capture_frame * capture_time:
    #   before_video.append(frame)
    #   continue
    # else:
    #   # 버퍼가 가득 찬 경우, 롤링 버퍼 유지
    #   # 알람이 없는 상태이면 실행
    #   if alarm == 0:
    #     before_video.pop(0)
    #     before_video.append(frame)

    # 3. 이벤트 발생 여부 확인 (기존 소켓 수신 부분을 대체)
    # 현재 알람이 없는 상태일 때만 새로운 이벤트를 확인
    if alarm == 0:
      try:
        # final_output_queue에서 취합된 분석 결과를 가져옴
        event_data = final_output_queue.get_nowait()
        # 'detection' 필드가 비어있지 않으면 이벤트가 발생한 것으로 간주
        if event_data and len(event_data.get('detection')):
          print("videoManager: 이벤트 감지. 영상 후-캡처를 시작합니다.")
          alarm = 1  # 알람 상태 활성화
          current_event_data = event_data  # 이벤트 메타데이터 저장
      except queue.Empty:
        # 큐가 비어있으면 이벤트가 없는 것이므로 통과
        pass

    # 4. 이벤트 발생 후 영상 캡처 및 비디오 생성 동작
    # 이벤트가 있는 경우에
    if alarm != 0:
      # if len(after_video) < capture_frame * capture_time + 200:
      if len(after_video) < capture_frame * capture_time:
        after_video.append(frame)
      # 필요한 만큼의 후-영상이 모두 저장되면 비디오 생성 절차 시작
      else:
        # 전/후 영상을 합쳐 최종 이벤트 비디오 리스트 생성
        event_video = before_video + after_video
        current_time = datetime.now()
        video_filename = f'{current_time.strftime("%Y%m%d%H%M%S")}.avi'
        video_full_path = os.path.join(file_path, video_filename)

        # 영상 프레임 정보를 바탕으로 VideoWriter 설정
        height, width, _ = frame.shape
        out = cv2.VideoWriter(video_full_path, cv2.VideoWriter_fourcc(*'DIVX'), capture_frame, (width, height))
        
        # 모든 프레임을 파일에 쓰기
        for i in range(len(event_video)):
          out.write(event_video[i])
        out.release()
        
        print(f"videoManager: 이벤트 영상 생성 완료. (이전: {len(before_video)} 프레임, 이후: {len(after_video)} 프레임)")
        
        # 생성된 비디오 파일을 바이너리 버퍼로 읽기 (전송을 위해)
        with open(video_full_path, "rb") as video:
          video_buffer = video.read()
        video_size = len(video_buffer)

        # 비디오 상황 데이터를 (메타데이터, 영상 크기, 영상 버퍼) 생성
        video_data = (current_event_data, video_size, video_buffer)
        
        # (비디오 상황 데이터, 비디오 데이터)를 이벤트 비디오 큐에 추가
        # (event_data, video_data)
        event_video_queue.put((event_data, video_data))
        print(f"videoManager: 생성된 이벤트 영상({video_size} bytes)을 큐에 추가했습니다.")

        # # 임시로 생성했던 비디오 파일 삭제
        # os.remove(video_full_path)

        # --- 상태 초기화 ---
        # 다음 이벤트를 위해, '이후' 영상의 뒷부분을 새로운 '이전' 영상 버퍼로 사용
        # before_video = after_video[400:]
        # before_video = after_video[450:]
        before_video = after_video[int(capture_frame * capture_time / 2):]
        after_video.clear()
        alarm = 0  # 알람 상태 비활성화
        current_event_data = None

  print("situationDetector (Video Manager): 이벤트 영상 생성 스레드가 종료되었습니다.")
