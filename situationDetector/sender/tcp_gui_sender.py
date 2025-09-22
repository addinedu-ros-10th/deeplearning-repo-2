# tcp_gui_sender.py
import socket
import time
import json
import queue
import threading

TCP_HOST = '192.168.0.183'          # monitoringGUI IP 주소
TCP_PORT = 2401                     # monitoringGUI 발신 포트 주소

def generateGUIJsonDump(timestamp, patrol_car_name):
    """
    GUI에 보낼 json 데이터 생성
    프레임 해석 정보와 기본 정보 (영상 메타데이터 : 시간, 위치, 순찰차 이름)을 인자로 받아 새로운 json 데이터 생성
    """  
    transform = {
        "timestamp" : timestamp, # 영상 메타데이터 : 순찰차 영상 촬영 시각
        "patrol_car_name" : patrol_car_name, # 영상 메타데이터 : 순찰차 이름
    }
    str_data = json.dumps(transform, ensure_ascii=False)
    return str_data.encode('utf-8') # str 바이트 데이터로 변환하여 return

def send_tcp_data_to_gui(shared_metadata: dict, 
                        metadata_lock : threading.Lock,
                        gui_tcp_connected_event : threading.Event,
                        shutdown_event: threading.Event):
    """
    db_queue에서 분석 데이터를 가져와 DB 서버로 TCP 전송
    """
    while not shutdown_event.is_set():
        sock = None
        try:
            # DB 서버에 연결 시도
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"situationDetector (TCP GUI Sender) : GUI 서버({TCP_HOST}:{TCP_PORT})에 연결을 시도합니다.")
            sock.connect((TCP_HOST, TCP_PORT))
            print(f"situationDetector (TCP GUI Sender) : GUI 서버에 연결되었습니다.")

            # 연결 성공 시 Event를 set하여 UDP 스레드에 알림
            gui_tcp_connected_event.set()

            # 연결이 유지되는 동안 큐에서 데이터 전송
            while not shutdown_event.is_set():
                try:
                    # metadata 큐에서 데이터 가져오기
                    with metadata_lock:
                        current_timestamp = shared_metadata.get("timestamp",None)
                        current_patrol_car = shared_metadata.get("patrol_car_name", "Unknown")
                    
                    json_dump = generateGUIJsonDump(current_timestamp, current_patrol_car)
                    sock.sendall(json_dump)
                    
                    # 1초에 한번 메타데이터 전송
                    time.sleep(1)

                    # print(f"AI Server (TCP DB Sender) : 데이터 전송 완료 ({len(data_to_send)} bytes)")
                except socket.error as e:
                    # 소켓 오류 발생 시 연결 재시도
                    print(f"situationDetector (TCP GUI Sender) : 소켓 오류 발생: {e}. 재연결을 시도합니다.")
                    break # 내부 루프를 빠져나가 외부 루프에서 재연결 시도
        
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"situationDetector (TCP GUI Sender) : GUI 연결 실패: {e}")
        
        finally:
            if sock:
                sock.close()
            # 재연결 시도 전 잠시 대기
            if not shutdown_event.is_set():
                print("situationDetector (TCP GUI Sender) : 5초 후 재연결을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP GUI Sender) : DB 전송 스레드를 종료합니다.")