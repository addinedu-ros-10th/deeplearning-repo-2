# test_db_server.py
import socket

# tcp_ds_sender.py에 정의된 HOST와 PORT와 동일하게 설정합니다.
TCP_HOST = 'localhost'
TCP_PORT = 6602

def run_test_server():
    """
    TCP 서버를 실행하여 클라이언트(tcp_ds_sender)의 연결을 수신하고,
    전송받은 데이터를 터미널에 출력합니다.
    """
    # AF_INET: IPv4 주소 체계, SOCK_STREAM: TCP 소켓을 의미합니다.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 서버가 종료된 후에도 주소를 즉시 재사용할 수 있도록 설정합니다.
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        # 지정된 호스트와 포트로 소켓을 바인딩합니다.
        server_socket.bind((TCP_HOST, TCP_PORT))

        # 클라이언트의 연결을 기다리도록 설정합니다. ( backlog: 1 )
        server_socket.listen(1)
        print(f"[*] 테스트 DB 서버가 {TCP_HOST}:{TCP_PORT} 에서 연결을 기다리고 있습니다.")

        while True:
            # 클라이언트의 연결 요청을 수락합니다.
            # conn: 데이터 송수신에 사용될 새로운 소켓 객체
            # addr: 연결된 클라이언트의 주소
            print("\n[*] 새 클라이언트의 연결을 기다립니다...")
            conn, addr = server_socket.accept()
            print(f"[*] {addr} 에서 클라이언트가 연결되었습니다.")

            try:
                # 클라이언트가 연결을 유지하는 동안 데이터를 계속 수신합니다.
                while True:
                    # 최대 1024 바이트의 데이터를 수신합니다.
                    data = conn.recv(1024)
                    
                    # 수신된 데이터가 없으면 클라이언트가 연결을 종료한 것입니다.
                    if not data:
                        print(f"[*] 클라이언트({addr})와의 연결이 종료되었습니다.")
                        break
                    
                    # 수신된 데이터는 바이트(bytes) 형식이므로,
                    # 사람이 읽을 수 있는 문자열(string)로 디코딩하여 출력합니다.
                    # tcp_ds_sender에서 별도의 인코딩을 지정하지 않았으므로, 기본 'utf-8'을 사용합니다.
                    print(f" -> 수신 데이터 ({addr}): {data.decode('utf-8')}")
            
            except ConnectionResetError:
                # 클라이언트가 비정상적으로 연결을 종료했을 때 발생하는 예외를 처리합니다.
                print(f"[!] 클라이언트({addr})와의 연결이 비정상적으로 끊어졌습니다.")
            
            finally:
                # 현재 클라이언트와의 연결 소켓을 닫습니다.
                conn.close()

    except KeyboardInterrupt:
        # Ctrl+C 입력 시 서버를 정상적으로 종료합니다.
        print("\n[*] 서버를 종료합니다.")
    
    finally:
        # 서버 소켓을 닫습니다.
        server_socket.close()

if __name__ == "__main__":
    run_test_server()