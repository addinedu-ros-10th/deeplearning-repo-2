# situationDetector/communicator/tcp_dm_communicator.py
"""
situationDetector <-> deviceManager 양방향 TCP 통신 모듈
"""
import socket
import threading
import queue
import time
import struct
from typing import Dict

# 통신 설정
TCP_HOST = '192.168.0.86'  # situationDetector 자신의 IP 주소
TCP_PORT = 1201         # deviceManager와 통신할 단일 포트

# 수신 데이터 헤더 정보
HEADER_FORMAT = "BBBIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# 송신 테스트 데이터 (기존 sender 모듈 참고)
TEST_DATA_FIRE = b'\x02\x01\x01\x01'

"""
situationDetector (TCP dM Receive) : 헤더 (1, 2, 1, 2025, 9, 26, 21, 12, 28, 28)
situationDetector (TCP dM Receive) : [('172.20.10.7', 48284)] 영상 수신 완료. 크기: 28 바이트
situationDetector (TCP dM Receive) : 헤더 (56, 0, 0, 33333, 2304000, 0, 2320, 0, 0, 1)
situationDetector (TCP dM Receive) : [('172.20.10.7', 48284)] 영상 수신 완료. 크기: 1 바이트
situationDetector (TCP dM Receive) : 헤더 (0, 16, 0, 3758096386, 1, 0, 0, 0, 1275068416, 2488554313)
situationDetector (Video Saver): 큐에서 28 바이트 영상 수신. 저장을 시작합니다.
{'detection': {}, 'timestamp': None, 'patrol_number': 0}
situationDetector (Video Saver): 영상을 성공적으로 저장했습니다. -> situationDetector/test_data/test2.mp4
situationDetector (Video Saver): 큐에서 1 바이트 영상 수신. 저장을 시작합니다.
situationDetector (Video Saver): 영상을 성공적으로 저장했습니다. -> situationDetector/test_data/test2.mp4
situationDetector (TCP dS Communicator) : dS 서버 ('172.20.10.10', 2301)에 연결 시도
{'detection': {}, 'timestamp': None, 'patrol_number': 0}
situationDetector (TCP dS Communicator) : 클라이언트 메인 루프 오류: [Errno 113] No route to host
{'detection': {}, 'timestamp': None, 'patrol_number': 0}
situationDetector (TCP dS Communicator) : dS 서버 ('172.20.10.10', 2301)에 연결 시도

"""
def _handle_receive(conn: socket.socket, addr: tuple, event_video_queue: queue.Queue, shutdown_event: threading.Event):
    """
    deviceManager 클라이언트로부터 영상 메타데이터 / 30초 이벤트 영상 수신
    주요 기능: 30초 이벤트 영상 수신
    """
    print(f"situationDetector (TCP dM Receive) : [{addr}] 수신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            # 1. 고정 크기의 헤더 수신
            header_data = conn.recv(HEADER_SIZE)
            if not header_data:
                print(f"situationDetector (TCP dM Receive) : [{addr}] 클라이언트 연결 끊어짐.")
                break
            
            # print(header_data)
            
            # 2. 헤더 언패킹
            unpacked_header = struct.unpack(HEADER_FORMAT, header_data)
            video_metadata = {
                "source": unpacked_header[0],
                "destination" : unpacked_header[1],
                "patrol_number" : unpacked_header[2],
                "timestamp" : list(unpacked_header[3:9]), # 튜플을 리스트로 변환
            }
            video_size = unpacked_header[9] # 영상 데이터 크기
            
            # [테스트] 헤더 수신 로그
            print(f"situationDetector (TCP dM Receive) : 헤더 {unpacked_header}")

            
            # 3. 헤더에 명시된 크기만큼 영상 데이터 수신
            video_buffer = b''
            remain_size = video_size
            
            while remain_size > 0:
                # 수신할 데이터 크기를 1024과 남은 크기 중 작은 값으로 선택
                chunk = conn.recv(min(1024, remain_size))
                if not chunk:
                    print(f"situationDetector (TCP dM Receive) : [{addr}] 영상 데이터 수신 중 연결 끊김.")
                    # 데이터가 불완전하므로 루프를 빠져나감
                    break
                video_buffer += chunk
                remain_size -= len(chunk)
            
            # 수신이 완료되지 않았다면, 현재 영상은 폐기
            if remain_size > 0:
                print(f"situationDetector (TCP dM Receive) : [{addr}] 영상 데이터가 불완전하게 수신됨.")
                continue

            # 4. 수신 완료된 영상을 큐에 추가
            if video_buffer:
                print(f"situationDetector (TCP dM Receive) : [{addr}] 영상 수신 완료. 크기: {len(video_buffer)} 바이트")
                video_item = (video_metadata, video_size, video_buffer)
                event_video_queue.put(video_item)


            else:
                print(f"situationDetector (TCP dM Receive) : [{addr}] 수신된 영상 데이터가 없습니다.")

    except ConnectionResetError:
        print(f"situationDetector (TCP dM Receive) : [{addr}] 클라이언트 연결이 리셋되었습니다.")
    except Exception as e:
        # 종료 이벤트가 설정되지 않은 경우에만 오류 출력
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dM Receive) : [{addr}] 수신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dM Receive) : [{addr}] 수신 스레드 종료.")

def _handle_send(conn: socket.socket, 
                addr: tuple, 
                final_output_queue: queue.Queue, 
                ignore_events : Dict, 
                shutdown_event: threading.Event):
    """
    하나의 deviceManager 클라이언트에 데이터를 지속적으로 송신하는 스레드.
    주요 기능: 분석에 따른 이벤트 발생 데이터(알람 등) 전송
    """
    print(f"situationDetector (TCP dM Send) : [{addr}] 송신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            try:
                # 결과 데이터 해석
                # 결과 데이터 형식 (예시)
                '''
                {'detection': 
                    {'feat_detect_fire': 
                        [
                            {
                                'class_id': 2, 
                                'class_name': 'detect_fire_general_smoke', 
                                'confidence': 0.25313425064086914, 
                                'bbox': {
                                    'x1': 74.0595703125, 
                                    'y1': 0.23963546752929688, 
                                    'x2': 605.1390380859375, 
                                    'y2': 35.71662902832031
                                }
                            }, 
                            {
                                'detection_count': 1
                            }
                        ]
                    }, 
                    'timestamp': None, 
                    'patrol_number': 0
                }
                '''
                # (situationDetector) -> (deviceManager)
                event = final_output_queue.get(timeout=1.0)
                # 1. 고정값 정의
                SOURCE = 0x02
                DESTINATION = 0x01
                
                # patrol_number = int(event['patrol_number'])
                patrol_number = 1
                alarm_type = 0 # 이벤트 없음
                
                # 이벤트 우선 순위 : fire -> smoke -> ....

                if len(ignore_events) != 0: # 알람해제 이벤트가 진행이면
                    alarm_type = 255 # 알람 해제중 신호로 설정
                                
                # 알람해제 이벤트가 없으면 감지된 이벤트 값으로 alarm_type 할당
                elif event.get('detection'):
                # elif event.get('detection'):
                    detection_data = event['detection']
                    if 'feat_detect_fire' in detection_data:
                        alarm_type = 1 # 화재 경고
                    elif 'feat_detect_assault' in detection_data:
                        alarm_type = 2 # 폭행 경고
                    elif 'feat_detect_trash' in detection_data:
                        alarm_type = 3 # 무단투기 경고
                    elif 'feat_detect_smoke' in detection_data:
                        alarm_type = 4 # 흡연자 경고

                # if len(ignore_events) != 0: # 알람해제 이벤트가 진행이면
                #     alarm_type = 255 # 알람 해제중 신호로 설정

                # print(alarm_type)
                
                data_packet = struct.pack('>BBBB', SOURCE, DESTINATION, patrol_number, alarm_type)
                
                # print(data_packet)
                # print(event)

                conn.send(data_packet)
                # print(f"situationDetector (TCP dM Communicator) : [{addr}] 테스트 데이터 전송 완료 ({len(TEST_DATA_FIRE)} bytes)")

            except queue.Empty:
                # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
                continue
            except (socket.error, BrokenPipeError) as e:
                print(f"situationDetector (TCP dM Send) : [{addr}] 소켓 오류 발생: {e}. 송신 스레드를 종료합니다.")
                break # 소켓 오류 시 루프 탈출
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dM Send) : [{addr}] 송신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dM Send) : [{addr}] 송신 스레드 종료.")

def dm_server_run(event_video_queue: queue.Queue,
                        final_output_queue: queue.Queue,
                        ignore_events: Dict,
                        shutdown_event: threading.Event):
    """
    deviceManager 클라이언트의 연결을 수락하고,
    각 클라이언트에 대해 양방향 통신(수신/송신) 스레드를 생성 및 관리.
    """
    # time.sleep(10)
    # event = final_output_queue.get(timeout=1.0)
    # print(event)
    
    # ------
    # time.sleep(10)
    # try:
    #         # (situationDetector) -> (deviceManager)
    #     event = final_output_queue.get(timeout=1.0)
    #     # 1. 고정값 정의
    #     SOURCE = 0x02
    #     DESTINATION = 0x01
        
    #     patrol_number = int(event['patrol_number'])
    #     alarm_type = 0 # 이벤트 없음
        
    #     # 이벤트 우선 순위 : fire -> smoke -> ....
        
    #     # event 딕셔너리에 'detection' 내용이 있는지 확인
    #     if event.get('detection'):
    #         detection_data = event['detection']
    #         if 'feat_detect_fire' in detection_data:
    #             alarm_type = 1 # 화재 경고
    #         elif 'feat_detect_assault' in detection_data: # 폭행 감지 키(가정)
    #             alarm_type = 2 # 폭행 경고
    #         elif 'feat_detect_littering' in detection_data: # 무단투기 감지 키(가정)
    #             alarm_type = 3 # 무단투기 경고
        
    #     data_packet = struct.pack('>BBBB', SOURCE, DESTINATION, patrol_number, alarm_type)
        
    #     print(data_packet)
    # except queue.Empty:
    #     # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
    #     print("EMPTY!")
    
    # -----
    
    server_sock = None
    while not shutdown_event.is_set():
        try:
            # 1. TCP 서버 소켓 생성 및 바인딩
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((TCP_HOST, TCP_PORT))
            server_sock.listen()
            # accept()가 블로킹되지 않도록 타임아웃 설정하여 shutdown_event를 주기적으로 확인
            server_sock.settimeout(1.0)
            print(f"situationDetector (TCP dM Communicator) : dM 통신 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기 중")

            conn, addr = None, None
            while not shutdown_event.is_set():
                try:
                    # 2. 클라이언트 연결 대기
                    conn, addr = server_sock.accept()
                    print(f"situationDetector (TCP dM Communicator) : dM 클라이언트 연결됨: {addr}")

                    # 3. 연결된 클라이언트를 위한 수신/송신 스레드 생성 및 시작
                    receiver = threading.Thread(target=_handle_receive, args=(conn, addr, event_video_queue, shutdown_event))
                    sender = threading.Thread(target=_handle_send, args=(conn, addr, final_output_queue, ignore_events, shutdown_event))
                    receiver.daemon = True
                    sender.daemon = True
                    receiver.start()
                    sender.start()
                    
                    # 두 스레드가 모두 종료될 때까지 대기 (즉, 연결이 끊어질 때까지)
                    receiver.join()
                    sender.join()
                    print(f"situationDetector (TCP dM Communicator) : [{addr}] 클라이언트와의 세션이 종료되었습니다. 새 연결을 대기합니다.")

                except socket.timeout:
                    # 타임아웃은 정상 동작이므로 루프 계속
                    continue
                finally:
                    if conn:
                        conn.close()

        except Exception as e:
            print(f"situationDetector (TCP dM Communicator) : 서버 메인 루프 오류: {e}")
        finally:
            if server_sock:
                server_sock.close()
            # 종료 신호가 없을 경우에만 재시작
            if not shutdown_event.is_set():
                print("situationDetector (TCP dM Communicator) : 5초 후 서버 재시작을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP dM Communicator) : 통신 서버를 종료합니다.")