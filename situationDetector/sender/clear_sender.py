# sender/clear_sender.py
import argparse, socket, json
from datetime import datetime, timezone

def send_patrol_clear(patrol_number:int, event_id:str, track_id:int|None=None,
                      reason:str|None=None, note:str|None=None,
                      host:str="127.0.0.1", port:int=50556, source:str="fall_detection"):
    payload = {
        "type":"situationDetector",     # 고정
        "event":"patrol_clear",         # '해제' 이벤트 식별자
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "patrol_number": patrol_number, # 순찰차/패트롤 번호
        "source": source,               # 발생 소스(모듈명)
        "data": { "event_id": event_id }
    }
    if track_id is not None: payload["data"]["track_id"]=int(track_id)
    if reason: payload["data"]["reason"]=reason
    if note: payload["data"]["note"]=note

    line = (json.dumps(payload, ensure_ascii=False)+"\n").encode("utf-8")
    with socket.create_connection((host,port), timeout=2.0) as s:
        s.sendall(line)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50556)
    ap.add_argument("--patrol", type=int, default=0)
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()
    send_patrol_clear(args.patrol, args.event_id, args.track_id, args.reason, args.note, args.host, args.port)

if __name__ == "__main__":
    cli()
