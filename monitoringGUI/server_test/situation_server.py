# server.py
import socket, threading, time, json, random, cv2, numpy as np
from datetime import datetime

GUI_CLIENT_IP = "127.0.0.1"

def video_streamer():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        gui_addr = (GUI_CLIENT_IP, 9999)
        print(f"[서버/영상] GUI({GUI_CLIENT_IP}:9999)로 영상 전송 시작.")
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, "LIVE FROM SERVER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, now_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            sock.sendto(buffer, gui_addr)
            time.sleep(1/30)

def event_sender():
    events = [
        {"type": "화재", "alert": "HIGH"},
        {"type": "폭행", "alert": "HIGH"},
    ]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 8888))
        s.listen()
        print(f"[서버/이벤트] GUI 클라이언트 연결 대기 중... (포트: 8888)")
        conn, addr = s.accept()
        print(f"[서버/이벤트] GUI 연결됨: {addr}")
        while True:
            try:
                time.sleep(10) # 10초마다 이벤트 발생
                event_info = random.choice(events)
                event_data = { "event_type": event_info["type"], "alert_level": event_info["alert"] }
                print(f"[서버/이벤트] 이벤트 전송: {event_info['type']}")
                conn.sendall(json.dumps(event_data).encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError):
                print(f"[서버/이벤트] GUI 연결 끊김. 다시 대기합니다.")
                conn, addr = s.accept()
                print(f"[서버/이벤트] GUI 재연결됨: {addr}")

if __name__ == '__main__':
    threading.Thread(target=video_streamer, daemon=True).start()
    threading.Thread(target=event_sender, daemon=True).start()
    while True: time.sleep(60)