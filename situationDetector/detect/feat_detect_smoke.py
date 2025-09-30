import queue
import threading
import time
import json

SMOKE_MODEL_PATH = "situationDetector/detect/feat_detect_smoke/best.pt"

SMOKE_LABELS = {
    0: "smoke",
    1: "not_smoke",
}

def _load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)

def _build_detection_item(cls_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    """
    흡연 감지 스키마:
      class_name: SMOKE_LABELS에서 가져옴 (기본 "smoke"/"not_smoke")
    """
    name = SMOKE_LABELS.get(int(cls_id), str(int(cls_id)))
    return {
        "class_id": int(cls_id),
        "class_name": name,
        "confidence": float(conf),
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
    }

def run_smoke_detect(
    analysis_frame_queue: queue.Queue,
    aggregation_queue: queue.Queue,
    analyzer_name: str,
    shutdown_event: threading.Event,
):
    """
    입력(analysis_frame_queue):
      - dict: {"frame","frame_count","timestamp","patrol_number"}
      - tuple/list: (frame_count, frame_time, frame)

    출력(aggregation_queue):
      {
        "detection": [ {...}, ... ],
        "timestamp": <float|str>,
        "analyzer_name": analyzer_name,   # 보통 "feat_detect_smoke"
        "detection_count": <int>,
        "patrol_number": <int>,
      }
    """
    print("feat_detect_smoke : worker start, loading model...")
    try:
        model = _load_model(SMOKE_MODEL_PATH)
    except Exception as e:
        print(f"feat_detect_smoke : model load failed -> {e}")
        model = None  # 모델 없어도 루프는 유지

    while not shutdown_event.is_set():
        # ---- 입력 꺼내기 (dict/tuple 호환) ----
        try:
            item = analysis_frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"feat_detect_smoke : queue error -> {e}")
            break

        frame = None
        frame_count = None
        frame_time = None
        patrol_no = 1

        if isinstance(item, dict):
            frame       = item.get("frame")
            frame_count = item.get("frame_count")
            frame_time  = item.get("timestamp")
            patrol_no   = item.get("patrol_number", 1)
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            frame_count, frame_time, frame = item[:3]
            patrol_no = 1
        else:
            continue

        if frame_time is None:
            frame_time = time.time()
        if frame is None:
            continue

        # ---- 추론 ----
        det_list = []
        if model is not None:
            try:
                res = model(frame, verbose=False)[0]
                if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                    for b in res.boxes.data.tolist():  # x1,y1,x2,y2,conf,cls
                        x1, y1, x2, y2, conf, cls = b
                        det_list.append(
                            _build_detection_item(
                                cls_id=int(cls), conf=conf,
                                x1=x1, y1=y1, x2=x2, y2=y2
                            )
                        )
            except Exception as e:
                print(f"feat_detect_smoke : inference error -> {e}")

        # ---- 결과 패키징 ----
        result_package = {
            "detection": det_list,
            "timestamp": frame_time,
            "analyzer_name": analyzer_name,   # 보통 "feat_detect_smoke"
            "detection_count": len(det_list),
            "patrol_number": patrol_no,
        }

        try:
            aggregation_queue.put(result_package, timeout=0.1)
        except queue.Full:
            # 취합 큐가 가득하면 해당 프레임 결과는 드롭
            pass

    print("feat_detect_smoke : worker stop")


# ---- 단독 실행(옵션) ----
def _standalone():
    """간단 단독 테스트 (카메라 0에서 1프레임만 추론)"""
    import cv2
    q_in = queue.Queue()
    q_out = queue.Queue()
    ev = threading.Event()

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("feat_detect_smoke standalone: cannot grab frame")
        return

    q_in.put({
        "frame": frame,
        "frame_count": 1,
        "timestamp": time.time(),
        "patrol_number": 1,
    })

    t = threading.Thread(target=run_smoke_detect, args=(q_in, q_out, "feat_detect_smoke", ev), daemon=True)
    t.start()
    time.sleep(1.0)
    ev.set()
    t.join()

    while not q_out.empty():
        print(json.dumps(q_out.get(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _standalone()

