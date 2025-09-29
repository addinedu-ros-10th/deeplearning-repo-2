# # situationDetector/detect/feat_detect_fall.py

# import queue
# import threading
# import time
# import json

# # 모델은 함수 안에서 lazy import (임포트 시 예외로 함수 정의 못하는 문제 방지)
# FALL_MODEL_PATH = "situationDetector/detect/feat_detect_fall/yolov8n.pt"  # 필요 시 변경

# def _load_model(weights_path: str):
#     from ultralytics import YOLO
#     return YOLO(weights_path)

# def _build_detection_item(cls_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
#     # 요구한 스키마: class_name은 고정 "detect_fall"
#     return {
#         "class_id": int(cls_id),
#         "class_name": "detect_fall",
#         "confidence": float(conf),
#         "bbox": {
#             "x1": float(x1),
#             "y1": float(y1),
#             "x2": float(x2),
#             "y2": float(y2),
#         },
#     }

# def run_fall_detect(
#     analysis_frame_queue: queue.Queue,
#     aggregation_queue: queue.Queue,
#     analyzer_name: str,
#     shutdown_event: threading.Event,
# ):
#     """
#     analysis_frame_queue -> { "frame", "frame_count", "timestamp", "patrol_number" }
#     aggregation_queue <- {
#         "detection": [ {...}, ... ],
#         "timestamp": <float or str>,
#         "analyzer_name": analyzer_name,
#         "detection_count": <int>,
#         "patrol_number": <int>,
#     }
#     """
#     print("feat_detect_fall : worker start, loading model...")
#     try:
#         model = _load_model(FALL_MODEL_PATH)
#     except Exception as e:
#         print(f"feat_detect_fall : model load failed -> {e}")
#         # 모델이 없어도 프로세스가 죽지 않도록, 빈 루프(패스스루)로 동작
#         model = None

#     while not shutdown_event.is_set():
#         try:
#             item = analysis_frame_queue.get(timeout=1.0)
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"feat_detect_fall : queue error -> {e}")
#             break

#         # 입력 dict 언패킹 (키 이름은 상황Detector 쪽 설명에 맞춤)
#         frame       = item.get("frame")
#         frame_count = item.get("frame_count")
#         frame_time  = item.get("timestamp")      # float(time.time()) 또는 문자열
#         patrol_no   = item.get("patrol_number", 1)

#         if frame is None:
#             continue

#         det_list = []

#         if model is not None:
#             try:
#                 # 필요 시 conf/imgsz 등 추가 가능
#                 res = model(frame, verbose=False)[0]
#                 if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
#                     # xyxy, conf, cls 텐서를 파이썬 값으로
#                     for b in res.boxes.data.tolist():
#                         x1, y1, x2, y2, conf, cls = b
#                         det_list.append(_build_detection_item(cls_id=int(cls), conf=conf,
#                                                              x1=x1, y1=y1, x2=x2, y2=y2))
#             except Exception as e:
#                 # 모델 추론 오류는 해당 프레임만 스킵
#                 print(f"feat_detect_fall : inference error -> {e}")

#         # 결과 패키징
#         result_package = {
#             "detection": det_list,                # 감지 결과 리스트
#             "timestamp": frame_time,              # 수신 프레임의 시간 그대로
#             "analyzer_name": analyzer_name,       # "feat_detect_fall" 로 들어옴
#             "detection_count": len(det_list),     # 감지 개수
#             "patrol_number": patrol_no,           # 입력에서 받은 순찰차 번호
#         }

#         try:
#             aggregation_queue.put(result_package, timeout=0.1)
#         except queue.Full:
#             # 취합 큐가 가득 찬 경우 해당 프레임 결과는 드롭
#             pass

#     print("feat_detect_fall : worker stop")


# # ===== 단독 테스트 실행용(Optional) =====
# # 모듈로 실행(-m situationDetector.situationDetector)할 때는 실행되지 않음
# def _standalone():
#     """
#     간단한 단독 테스트 러너 (카메라/영상은 여기서 다루지 않음).
#     프로젝트 통합 실행은 situationDetector.situationDetector 를 사용하세요.
#     """
#     import cv2
#     import threading
#     import queue
#     q_in = queue.Queue()
#     q_out = queue.Queue()
#     ev = threading.Event()

#     # 더미 프레임 하나만 밀어넣고 종료
#     cap = cv2.VideoCapture(0)
#     ok, frame = cap.read()
#     cap.release()
#     if not ok:
#         print("feat_detect_fall standalone: cannot grab frame")
#         return

#     q_in.put({
#         "frame": frame,
#         "frame_count": 1,
#         "timestamp": time.time(),
#         "patrol_number": 1,
#     })

#     t = threading.Thread(target=run_fall_detect, args=(q_in, q_out, "feat_detect_fall", ev), daemon=True)
#     t.start()
#     time.sleep(1.0)
#     ev.set()
#     t.join()

#     while not q_out.empty():
#         print(json.dumps(q_out.get(), ensure_ascii=False, indent=2))

# if __name__ == "__main__":
#     _standalone()


# situationDetector/detect/feat_detect_fall.py
import queue
import threading
import time
import json

# 필요 시 경로 변경
FALL_MODEL_PATH = "situationDetector/detect/feat_detect_fall/best.pt"

def _load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)

def _build_detection_item(cls_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    # 요구 스키마에 맞춰 class_name은 'detect_fall'로 고정
    return {
        "class_id": int(cls_id),
        "class_name": "detect_fall",
        "confidence": float(conf),
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
    }

def run_fall_detect(
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
        "analyzer_name": analyzer_name,
        "detection_count": <int>,
        "patrol_number": <int>,
      }
    """
    print("feat_detect_fall : worker start, loading model...")
    try:
        model = _load_model(FALL_MODEL_PATH)
    except Exception as e:
        print(f"feat_detect_fall : model load failed -> {e}")
        model = None  # 모델 없어도 루프는 유지

    while not shutdown_event.is_set():
        # ---- 입력 꺼내기 (dict/tuple 호환) ----
        try:
            item = analysis_frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"feat_detect_fall : queue error -> {e}")
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
                            _build_detection_item(cls_id=int(cls), conf=conf,
                                                  x1=x1, y1=y1, x2=x2, y2=y2)
                        )
            except Exception as e:
                print(f"feat_detect_fall : inference error -> {e}")

        # ---- 결과 패키징 ----
        result_package = {
            "detection": det_list,
            "timestamp": frame_time,
            "analyzer_name": analyzer_name,   # 보통 "feat_detect_fall"
            "detection_count": len(det_list),
            "patrol_number": patrol_no,
        }

        try:
            aggregation_queue.put(result_package, timeout=0.1)
        except queue.Full:
            # 취합 큐가 가득하면 해당 프레임 결과는 드롭
            pass

    print("feat_detect_fall : worker stop")


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
        print("feat_detect_fall standalone: cannot grab frame")
        return

    q_in.put({
        "frame": frame,
        "frame_count": 1,
        "timestamp": time.time(),
        "patrol_number": 1,
    })

    t = threading.Thread(target=run_fall_detect, args=(q_in, q_out, "feat_detect_fall", ev), daemon=True)
    t.start()
    time.sleep(1.0)
    ev.set()
    t.join()

    while not q_out.empty():
        print(json.dumps(q_out.get(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _standalone()

