# situationDetector/server/dataService.py
"""
situationDetector <-> dataService 양방향 TCP 통신 모듈
"""
import json
import base64
import socket
import threading
import queue
import time
import struct

# 통신 설정
TCP_HOST = '172.20.10.10'  # situationDetector 자신의 IP 주소
TCP_PORT = 2301         # dataService와 통신할 단일 포트

def _handle_send(sock: socket.socket, 
                addr: tuple, 
                final_output_queue: queue.Queue, 
                event_video_queue : queue.Queue,
                shutdown_event: threading.Event):
    """
    하나의 dataServer 클라이언트에 데이터를 지속적으로 송신하는 스레드.
    송신 데이터 : 합산 데이터 (6가지 모델 분석 결과 + 해제 요청이 있으면 30초간 무시)

    - AI 분석 결과: final_output_queue에서 가져와 {'detection': ...} 형식으로 전송
    - 이벤트 영상: event_video_queue에서 가져와 {'video_event': ...} 형식으로 전송
    """

    # 새로운 영상 데이터 송신 시, 헤더를 한 번만 전송하기 위한 상태 플래그
    send_header_data = True
    
    print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            # 플래그를 사용하여 루프 당 최소 한번의 전송을 보장
            data_send_in_loop = False
            send_header_data = False
            
            # 1. AI 분석 최종 데이터 큐에서 AI_SCHIMA에 따른 데이터를 가져와 처리
            try:
                ai_result = final_output_queue.get_nowait()

                print(ai_result)

                # 2. AI_SCHIMA를 문자열로 변환 및 UTF-8 인코딩
                json_string = json.dumps(ai_result)
                sock.send(json_string.encode('utf-8'))
                # print(f"situationDetector: [{addr}] AI 분석 데이터 전송 완료")
                data_send_in_loop = True
            except queue.Empty:
                # 큐가 비어있으면 통과
                pass
            
            # 2. 이벤트 영상 데이터 전송
            try:
                # 2-1 : 이벤트 영상 1개 가져오기
                    # video_buffer 형식 : b''
                video_metadata, video_size, video_buffer = event_video_queue.get_nowait()
                """
                VideoSize   : unsigned int                          : I - 비디오 영상 바이트 전체 크기
                Video       : 1024 byte per single communication    : 모두 송신하면 b"DONE" 전송

                수신부 비디오 영상 수신 시작 신호 감지 : video_data 값이 있을 경우
                수신부 비디오 영상 수신 종료 신호 감지 : video_data['video']의 값이 b'DONE'일 경우
                """

                # 2.2 : send_header_data 시그널을 이용해 영상 헤더 데이터를 1회 전송
                if send_header_data:
                    video_meta_string = json.dumps(video_metadata)
                    sock.send(video_meta_string.encode('utf-8'))
                    # 헤더 전송 후, 현재 영상의 픽셀 데이터 전송이 끝날 때까지 플래그를 False로 변경
                    send_header_data = False


                # 2-3 : 각 청크를 넣어 바이너리 스트림 영상 데이터를 생성 및 전송 
                bytes_send = 0
                chunk_size = 1024
                while bytes_send < video_size:
                    chunk = video_buffer[bytes_send : bytes_send + chunk_size]

                    # 1. chunk(bytes)를 Base64로 인코딩하여 문자열로 변환
                    encoded_chunk_str = base64.b64encode(chunk).decode('ascii')
                
                    video_data = {
                        "videosize" : video_size,
                        "video" : encoded_chunk_str
                    }
                    print(video_data)

                    video_data_string = json.dumps(video_data) # json -> str
                    sock.send(video_data_string.encode('utf-8'))
                    bytes_send += len(chunk)

                    # print(f"situationDetector: [{addr}] 이벤트 영상 조각 데이터 전송 완료")

                # 이벤트 영상 전송 완료 신호 전송
                video_data_end = {
                    "videosize" : video_size,
                    "video" : "DONE",
                }
                video_data_end_string = json.dumps(video_data_end) # json -> str
                sock.send(video_data_end_string.encode('udf-8'))
                print(f"situationDetector: [{addr}] 이벤트 영상 조각 데이터 전송 완료")
                data_send_in_loop = True

                # 현재 영상 전송이 모두 완료되었으므로, 다음 영상 전송을 위해 플래그를 True로 리셋
                send_header_data = True

            except queue.Empty:
                # 큐가 비어있으면 통과
                pass

            # 처리할 데이터가 없는 경우 CPU 부하 방지
            if not data_send_in_loop:
                time.sleep(0.01)

    except Exception as e:
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 종료.")


def ds_server_run(event_video_queue : queue.Queue,
                        final_output_queue: queue.Queue,
                        shutdown_event: threading.Event):
    """
    deviceManager 클라이언트의 연결을 수락하고,
    각 클라이언트에 대해 양방향 통신(수신/송신) 스레드를 생성 및 관리.
    """
    while not shutdown_event.is_set():
        try:
            # 1. TCP 클라이언트 소켓 생성
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            print(f"situationDetector (TCP dS Communicator) : dS 서버 {(TCP_HOST, TCP_PORT)}에 연결 시도")
            client_sock.connect((TCP_HOST, TCP_PORT))
            print(f"situationDetector (TCP dS Communicator) : dS 서버에 성공적으로 연결되었습니다.")

            # 2. 연결된 소켓으로 데이터를 보내는 스레드 시작
            sender_thread = threading.Thread(target=_handle_send, args=(client_sock, (TCP_HOST, TCP_PORT), final_output_queue, event_video_queue, shutdown_event))
            sender_thread.daemon = True
            sender_thread.start()
            
            # 스레드가 모두 종료될 때까지 대기 (즉, 연결이 끊어질 때까지)
            sender_thread.join()

        except ConnectionRefusedError:
            print(f"situationDetector (TCP dS Communicator) : 서버가 연결을 거부했습니다. 5초 후 재시도합니다.")
        except socket.gaierror:
            print(f"situationDetector (TCP dS Communicator) : 호스트 이름을 찾을 수 없습니다: {TCP_HOST}. 5초 후 재시도합니다.")
        except Exception as e:
            print(f"situationDetector (TCP dS Communicator) : 클라이언트 메인 루프 오류: {e}")
        finally:
            # 루프가 끝나면 (연결이 끊어졌거나 오류 발생 시) 소켓을 정리
            if client_sock:
                client_sock.close()

        # 종료 신호가 없을 경우에만 재연결 대기
        if not shutdown_event.is_set():
            time.sleep(5)

    print("situationDetector (TCP dS Communicator) : 통신 서버를 종료합니다.")