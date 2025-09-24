# receiver/clear_receiver.py
import socket, threading, json

HOST = "0.0.0.0"
PORT = 50556

def handle_clear(msg: dict, addr):
    # 프로토콜 체크
    if msg.get("type")!="situationDetector" or msg.get("event")!="patrol_clear":
        print(f"[RX][{addr}] unsupported: {msg}")
        return
    # 여기서 실제 해제 처리(대시보드 상태 갱신/DB 업데이트 등)
    print(f"[RX] CLEAR patrol={msg.get('patrol_number')} ts={msg.get('timestamp')} data={msg.get('data')}")

def _client(conn, addr):
    buf=b""
    try:
        with conn:
            while chunk := conn.recv(4096):
                buf+=chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n",1)
                    if not line.strip(): continue
                    try:
                        handle_clear(json.loads(line.decode("utf-8")), addr)
                    except Exception as e:
                        print(f"[RX][{addr}] JSON error: {e} | raw={line[:200]!r}")
    except Exception as e:
        print(f"[RX][{addr}] conn error: {e}")

def main(host=HOST, port=PORT):
    print(f"[RX] listening on {host}:{port}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen()
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[RX] client connected: {addr}")
            threading.Thread(target=_client, args=(conn,addr), daemon=True).start()
    finally:
        srv.close()

if __name__ == "__main__":
    main()
