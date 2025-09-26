import socket
import struct
import time 
HOST = '127.0.0.1' 
PORT = 2401        

print("테스트 서버를 시작합니다...")
print(f"IP: {HOST}, PORT: {PORT} 에서 연결을 기다리는 중... (Ctrl+C로 종료)")

# 1. 소켓 생성
# [수정] A_INET -> AF_INET 으로 오타를 수정했습니다.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # 2. IP와 포트를 소켓에 바인딩
    server_socket.bind((HOST, PORT))
    
    # 3. 클라이언트의 연결을 기다리는 '수신 대기' 모드로 전환
    server_socket.listen()
    
    # --- 서버가 계속해서 새로운 클라이언트의 접속을 받을 수 있도록 무한 루프를 추가합니다. ---
    while True: 
        print("\n=========================================")
        print("새로운 클라이언트의 접속을 기다립니다...")
        
        # 4. 클라이언트의 연결 요청을 수락 (새 클라이언트가 올 때까지 여기서 대기)
        conn, addr = server_socket.accept()
        
        # with 구문을 사용하면 이 블록이 끝날 때 conn 소켓(클라이언트와의 연결)이 자동으로 닫힙니다.
        with conn:
            print(f"{addr} 에서 연결되었습니다.")
            
            # 한 클라이언트가 여러 데이터를 보낼 경우를 대비해 내부 루프를 유지합니다.
            while True:
                # 5. 클라이언트로부터 데이터를 수신
                data = conn.recv(4)
                time.sleep(1)
                
                if (not len(data)):
                    continue
                # --- 수신된 데이터 처리 ---
                print("-" * 30)
                print(f"수신된 원본 데이터 (hex): {data.hex()}")

                if len(data) == 4:
                    # 6. struct.unpack으로 바이트 데이터를 파이썬 숫자로 변환
                    unpacked_data = struct.unpack('BBBB', data)
                    
                    print(f"데이터 해석 (튜플): {unpacked_data}")
                    
                    source_id, destination_id, command, alarm_type = unpacked_data

                    print("--- 상세 정보 ---")
                    print(f"  보낸 곳 (Source ID): {source_id}")
                    print(f"  받는 곳 (Destination ID): {destination_id}")
                    print(f"  명령 (Command): {command}")
                    print(f"  알람 타입 (Alarm Type): {alarm_type}")
                    print("-" * 30)

                else:
                    print(f"경고: 예상치 못한 길이({len(data)} bytes)의 데이터가 수신되었습니다.")