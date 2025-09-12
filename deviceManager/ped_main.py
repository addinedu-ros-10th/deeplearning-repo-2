# ped_main.py
import threading
from deviceManager.tcp_ai_sender import manage_tcp_connection
from deviceManager.udp_ai_sender import stream_video_udp

def main():
    """
    TCP 및 UDP 핸들러 스레드를 생성하고 관리
    """
    # 스레드 간 상태 공유를 위한 Event 객체
    # tcp_connected_event: TCP 연결 상태 (set: 연결됨, clear: 끊어짐)
    # shutdown_event: 모든 스레드를 종료하기 위한 신호
    tcp_connected_event = threading.Event()
    shutdown_event = threading.Event()

    # 1. TCP 핸들러 스레드 생성
    tcp_thread = threading.Thread(
        target=manage_tcp_connection,
        args=(tcp_connected_event, shutdown_event),
        daemon=True  # 메인 스레드 종료 시 함께 종료
    )

    # 2. UDP 핸들러 스레드 생성
    udp_thread = threading.Thread(
        target=stream_video_udp,
        args=(tcp_connected_event, shutdown_event),
        daemon=True
    )

    print("PED MAIN : TCP 및 UDP 전송 스레드 시작")
    tcp_thread.start()
    udp_thread.start()

    try:
        # 스레드가 살아있는 동안 메인 스레드는 대기
        # is_alive()를 사용하면 KeyboardInterrupt를 받을 수 있음
        while tcp_thread.is_alive() and udp_thread.is_alive():
            tcp_thread.join(timeout=0.1)
            udp_thread.join(timeout=0.1)
            
    except KeyboardInterrupt:
        print("\nPED MAIN : 종료 신호 감지, 모든 스레드를 정리")
        # 모든 스레드에 종료 신호 전송
        shutdown_event.set()
    finally:
        # 스레드가 완전히 종료될 때까지 대기
        tcp_thread.join()
        udp_thread.join()
        print("PED MAIN : 모든 스레드 종료 완료. 프로그램 종료.")


if __name__ == "__main__":
    main()