# data_service.py
import socket
import json
from datetime import datetime

def run_data_service():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 7777))
        s.listen()
        print(f"[DataService] 대시보드 연결 대기 중... (포트: 7777)")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[DataService] 대시보드 연결됨: {addr}")
                try:
                    data = conn.recv(1024)
                    if not data: continue
                    
                    event_data = json.loads(data.decode('utf-8'))
                    print(f"[DataService] 수신: '{event_data['event_type']}' 상황 종료 신호")
                    
                    response = { "status": "OK" }
                    conn.sendall(json.dumps(response).encode('utf-8'))

                except (BrokenPipeError, ConnectionResetError):
                    print(f"[DataService] 대시보드 연결 끊김: {addr}")

if __name__ == '__main__':
    run_data_service()