# test_db_server.py
import socket
import struct
import pickle
import cv2

# tcp_ds_sender.py에 정의된 HOST와 PORT와 동일하게 설정합니다.
TCP_HOST = '192.168.0.180'
TCP_PORT = 1201

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

        # 클라이언트의 연결 요청을 수락합니다.
        # conn: 데이터 송수신에 사용될 새로운 소켓 객체
        # addr: 연결된 클라이언트의 주소
        print("\n[*] 새 클라이언트의 연결을 기다립니다...")
        conn, addr = server_socket.accept()
        print(f"[*] {addr} 에서 클라이언트가 연결되었습니다.")

        a = int(input("Input data: "))
        conn.send(struct.pack("BBBB", 2, 1, 1, a))


        data = conn.recv(36)
        print(struct.unpack('BBBIIIIIII', data))
        file = open("data.avi", "wb")
        video_size = int(struct.unpack("I",data[27:-1])[0])
        video = b""
        while True:
            data = conn.recv(1024)
            if (data == b"DONE"):
                print("Receiving Done")
                break
            file.write(data)
        file.close()
        conn.close()

    except KeyboardInterrupt:
        # Ctrl+C 입력 시 서버를 정상적으로 종료합니다.
        print("\n[*] 서버를 종료합니다.")
    
    finally:
        # 서버 소켓을 닫습니다.
        server_socket.close()

if __name__ == "__main__":
    run_test_server()