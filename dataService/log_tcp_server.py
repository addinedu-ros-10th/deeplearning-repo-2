
import socket
import json
import threading
from datetime import datetime, timedelta

# --- 서버 설정 ---
HOST = '127.0.0.1'  # 로컬호스트 IP
PORT = 9999         # 사용할 포트 번호

def handle_client(conn, addr):
    """
    클라이언트 연결을 처리하는 함수 (쓰레드에서 실행됨)
    - conn: 클라이언트와 연결된 소켓 객체
    - addr: 클라이언트의 주소 정보 (IP, 포트)
    """
    print(f"✅ [서버] 클라이언트 연결됨: {addr}")
    try:
        # 클라이언트로부터 데이터를 받음 (최대 4096바이트)
        data = conn.recv(4096)
        if not data:
            print(f"❌ [서버] 클라이언트로부터 데이터를 받지 못함: {addr}")
            return

        # 받은 데이터(bytes)를 UTF-8 문자열로 디코딩
        request_str = data.decode('utf-8')
        # JSON 문자열을 파이썬 딕셔너리로 변환
        request_json = json.loads(request_str)
        
        print(f"🔵 [서버] 수신한 요청:\n{json.dumps(request_json, indent=2, ensure_ascii=False)}")

        # --- 가짜 로그 데이터 생성 로직 ---
        # 실제 애플리케이션에서는 이 부분에서 DB를 조회합니다.
        response_logs = []
        base_time = datetime.now()
        
        # 요청받은 detection_types에 따라 다른 메시지를 생성
        event_map = {
            '0': '화재', '1': '폭행', '2': '누워있는 사람', 
            '3': '실종자', '4': '무단 투기', '5': '흡연자'
        }
        
        for i in range(25): # 25개의 샘플 로그 생성
            # 요청된 유형 중 하나를 랜덤하게 선택하여 로그 생성
            event_type_code = request_json['detection_types'][(i % len(request_json['detection_types']))] if request_json['detection_types'] else '0'
            event_name = event_map.get(event_type_code, "알 수 없음")
            
            log_time = base_time - timedelta(hours=i*2)
            response_logs.append({
                'timestamp': log_time.strftime("%Y-%m-%d %H:%M:%S"),
                'message': f'[자동] 테스트용 {event_name} 감지됨 (샘플 {i+1})'
            })
        
        # 정렬 순서에 따라 데이터를 정렬
        if request_json.get('orderby') == 'latest':
            response_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        else:
            response_logs.sort(key=lambda x: x['timestamp'], reverse=False)

        # 응답 데이터를 JSON 문자열로 변환
        response_str = json.dumps(response_logs, ensure_ascii=False)
        # JSON 문자열을 UTF-8 바이트로 인코딩하여 클라이언트에 전송
        conn.sendall(response_str.encode('utf-8'))
        print(f"🟢 [서버] 응답 완료: {addr}")

    except json.JSONDecodeError:
        print(f"🔥 [서버] JSON 디코딩 오류: {addr}")
    except Exception as e:
        print(f"🔥 [서버] 오류 발생: {e}")
    finally:
        # 모든 처리가 끝나면 클라이언트와의 연결을 닫음
        conn.close()
        print(f"🔌 [서버] 클라이언트 연결 종료: {addr}")


def start_server():
    """서버를 시작하는 메인 함수"""
    # TCP 소켓 생성
    # AF_INET: IPv4 주소 체계 사용, SOCK_STREAM: TCP 프로토콜 사용
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 소켓 주소 재사용 옵션 설정 (서버를 재시작할 때 주소 충돌 방지)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 소켓을 지정된 호스트와 포트에 바인딩(연결)
    server_socket.bind((HOST, PORT))
    
    # 클라이언트의 연결 요청을 기다리는 상태로 전환 (최대 5개까지 대기)
    server_socket.listen(5)
    print(f"🚀 [서버] 서버 시작! 클라이언트 연결 대기 중... (주소: {HOST}:{PORT})")

    try:
        while True:
            # 클라이언트의 연결 요청을 수락 (연결이 될 때까지 여기서 대기)
            # conn: 클라이언트와 통신할 새로운 소켓, addr: 클라이언트 주소
            conn, addr = server_socket.accept()
            
            # 각 클라이언트를 별도의 쓰레드에서 처리하여 동시 접속 지원
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
    except KeyboardInterrupt:
        print("\n🛑 [서버] 서버를 종료합니다.")
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()

