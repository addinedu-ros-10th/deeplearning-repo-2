
import socket
import json

# --- 접속할 서버 정보 ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 9999

def fetch_logs_from_network(request_payload):
    """
    TCP 서버에 로그 데이터를 요청하고 응답을 받아 파이썬 객체로 반환합니다.
    
    Args:
        request_payload (dict): 서버에 보낼 검색 조건이 담긴 딕셔너리.
                                (예: {'start_date': '...', 'orderby': '...', ...})

    Returns:
        list or None: 성공 시 로그 데이터(딕셔너리 리스트), 실패 시 None을 반환.
    """
    try:
        # 1. TCP 소켓 생성
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 2. 서버에 연결 시도 (Timeout을 5초로 설정)
        client_socket.settimeout(5)
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("✅ [클라이언트] 서버에 연결 성공")

        # 3. 요청 데이터(딕셔너리)를 JSON 문자열로 변환 후, UTF-8 바이트로 인코딩
        message = json.dumps(request_payload).encode('utf-8')
        
        # 4. 인코딩된 데이터를 서버에 전송
        client_socket.sendall(message)
        print(f"🔵 [클라이언트] 요청 전송:\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}")

        # 5. 서버로부터 응답 데이터를 받기 위한 버퍼 초기화
        response_bytes = b""
        while True:
            # 4096 바이트씩 데이터를 받음
            chunk = client_socket.recv(4096)
            if not chunk:
                # 더 이상 받을 데이터가 없으면 루프 종료
                break
            response_bytes += chunk
        
        # 6. 받은 데이터(bytes)를 UTF-8 문자열로 디코딩 후, JSON 파싱
        response_str = response_bytes.decode('utf-8')
        response_data = json.loads(response_str)
        print("🟢 [클라이언트] 응답 수신 완료")
        
        return response_data

    except socket.timeout:
        print("🔥 [클라이언트] 오류: 서버 연결 시간 초과")
        return None
    except ConnectionRefusedError:
        print("🔥 [클라이언트] 오류: 서버가 연결을 거부했습니다. (서버가 실행 중인지 확인하세요)")
        return None
    except json.JSONDecodeError:
        print("🔥 [클라이언트] 오류: 서버로부터 받은 데이터가 올바른 JSON 형식이 아닙니다.")
        return None
    except Exception as e:
        print(f"🔥 [클라이언트] 알 수 없는 오류 발생: {e}")
        return None
    finally:
        # 7. 모든 작업이 끝나면 소켓 연결을 반드시 닫음
        if 'client_socket' in locals():
            client_socket.close()
            print("🔌 [클라이언트] 연결 종료")
