# situationDetector/server/dataService.py
"""
situationDetector <-> dataService 양방향 TCP 통신 모듈
"""
import socket
import threading
import queue
import time
import struct

# 통신 설정
TCP_HOST = '192.168.0.86'  # situationDetector 자신의 IP 주소
TCP_PORT = 2301         # dataService와 통신할 단일 포트

def _handle_send(conn: socket.socket, addr: tuple, final_output_queue: queue.Queue, shutdown_event: threading.Event):
    """
    하나의 dataServer 클라이언트에 데이터를 지속적으로 송신하는 스레드.
    송신 데이터 : 합산 데이터 (6가지 모델 분석 결과 + 해제 요청이 있으면 30초간 무시)
    """
    print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 시작")
    try:
        while not shutdown_event.is_set():
            try:
                ai_result = final_output_queue.get() 
                conn.send(ai_result)
                print(f"situationDetector (TCP dS Communicator) : [{addr}] 테스트 데이터 전송 완료 ({len(ai_result)} bytes)")

            except queue.Empty:
                # 큐가 비어있는 것은 정상적인 상황이므로 계속 진행
                continue
            except (socket.error, BrokenPipeError) as e:
                print(f"situationDetector (TCP dS Communicator) : [{addr}] 소켓 오류 발생: {e}. 송신 스레드를 종료합니다.")
                break # 소켓 오류 시 루프 탈출
            finally:
                # 테스트 코드 : 전송한 데이터 그대로 출력
                print(ai_result)
    except Exception as e:
        if not shutdown_event.is_set():
            print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 오류: {e}")
    finally:
        print(f"situationDetector (TCP dS Communicator) : [{addr}] 송신 스레드 종료.")

def ds_server_run(final_output_queues: queue.Queue,
                        shutdown_event: threading.Event):
    """
    deviceManager 클라이언트의 연결을 수락하고,
    각 클라이언트에 대해 양방향 통신(수신/송신) 스레드를 생성 및 관리.
    """
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
            print(f"situationDetector (TCP dS Communicator) : dS 통신 서버가 {TCP_HOST}:{TCP_PORT}에서 연결 대기 중")

            conn, addr = None, None
            while not shutdown_event.is_set():
                try:
                    # 2. 클라이언트 연결 대기
                    conn, addr = server_sock.accept()
                    print(f"situationDetector (TCP dS Communicator) : dS 클라이언트 연결됨: {addr}")

                    # 3. 연결된 클라이언트를 위한 수신/송신 스레드 생성 및 시작
                    sender = threading.Thread(target=_handle_send, args=(conn, addr, final_output_queues, shutdown_event))
                    sender.daemon = True
                    sender.start()
                    
                    # 스레드가 모두 종료될 때까지 대기 (즉, 연결이 끊어질 때까지)
                    sender.join()
                    print(f"situationDetector (TCP dS Communicator) : [{addr}] 클라이언트와의 세션이 종료되었습니다. 새 연결을 대기합니다.")

                except socket.timeout:
                    # 타임아웃은 정상 동작이므로 루프 계속
                    continue
                finally:
                    if conn:
                        conn.close()

        except Exception as e:
            print(f"situationDetector (TCP dS Communicator) : 서버 메인 루프 오류: {e}")
        finally:
            if server_sock:
                server_sock.close()
            # 종료 신호가 없을 경우에만 재시작
            if not shutdown_event.is_set():
                print("situationDetector (TCP dS Communicator) : 5초 후 서버 재시작을 시도합니다.")
                time.sleep(5)

    print("situationDetector (TCP dS Communicator) : 통신 서버를 종료합니다.")