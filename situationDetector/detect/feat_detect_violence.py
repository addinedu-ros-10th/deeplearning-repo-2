import queue
import threading
import time
import json
from collections import deque

VIOLENCE_MODEL_PATH = "situationDetector/detect/feat_detect_violence/lstm_pose_v8_best.pt"

def _build_detection_item_det(cls_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    return {
        "class_id": int(cls_id),
        "class_name": "detect_violence",  # 폭력 상황 감지 이벤트명 고정
        "confidence": float(conf),
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
    }

def _build_detection_item_cls(cls_id: int, conf: float):
    # 분류형 결과 (bbox 없음)
    return {
        "class_id": int(cls_id),
        "class_name": "detect_violence",
        "confidence": float(conf),
        "bbox": None,
    }

def _try_load_yolo(weights_path: str):
    """YOLO 포맷 시도"""
    try:
        from ultralytics import YOLO
        m = YOLO(weights_path)
        _ = m.predict(source=None, imgsz=32, verbose=False)  # 더미콜(에러 유도 X)
        return m
    except Exception:
        return None

def _try_load_torch(weights_path: str, device: str = "cpu"):
    """TorchScript 또는 일반 nn.Module 시도 (I3D/LSTM 같은 분류형 가정)"""
    try:
        import torch
        # 1) TorchScript
        try:
            m = torch.jit.load(weights_path, map_location=device).eval()
            return ("torchscript", m, device)
        except Exception:
            pass
        # 2) 일반 nn.Module 저장본
        m = torch.load(weights_path, map_location=device)
        if hasattr(m, "eval"):
            m.eval()
        return ("nnmodule", m, device)
    except Exception:
        return None

def _preprocess_clip(frames, resize=224, device="cpu"):
    """frames: list of BGR uint8 -> tensor (1, T, 3, H, W) in [0,1]"""
    import cv2, numpy as np, torch
    imgs = []
    for f in frames:
        im = cv2.resize(f, (resize, resize))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(im)
    x = np.stack(imgs, 0)              # (T,H,W,3)
    x = np.transpose(x, (0,3,1,2))     # (T,3,H,W)
    x = np.expand_dims(x, 0)           # (1,T,3,H,W)
    return torch.from_numpy(x).to(device)

def _softmax_logits(y):
    import torch
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, torch.Tensor) and y.ndim > 2:
        # (B,T,C) 형태면 시간 평균
        y = y.mean(dim=1)
    return torch.softmax(y, dim=-1)

def run_violence_detect(
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
    print("feat_detect_violence : worker start, loading model...")

    # 1) 우선 YOLO 포맷 시도
    yolo_model = _try_load_yolo(VIOLENCE_MODEL_PATH)

    # 2) 실패하면 분류형(I3D/LSTM) 시도
    clip_mode = None
    clip_model = None
    device = "cpu"
    SEQ_LEN = 32     # 학습에 맞춰 조정
    STRIDE  = 2      # 프레임 샘플링 간격
    clip_buf = deque(maxlen=SEQ_LEN)

    if yolo_model is None:
        loaded = _try_load_torch(VIOLENCE_MODEL_PATH, device="cpu")
        if loaded is not None:
            clip_mode, clip_model, device = loaded  # ('torchscript'|'nnmodule', model, device)
            print(f"feat_detect_violence : loaded {clip_mode} classifier on {device}")
        else:
            print("feat_detect_violence : model load failed (YOLO/Classifier both). Running passthrough.")
            # 모델 없어도 루프는 유지

    while not shutdown_event.is_set():
        # ---- 입력 꺼내기 (dict/tuple 호환) ----
        try:
            item = analysis_frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"feat_detect_violence : queue error -> {e}")
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

        det_list = []

        # ---- (A) YOLO 탐지형 ----
        if yolo_model is not None:
            try:
                res = yolo_model(frame, verbose=False, conf=0.25)[0]
                if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                    for b in res.boxes.data.tolist():  # x1,y1,x2,y2,conf,cls
                        x1, y1, x2, y2, conf, cls = b
                        cls = int(cls)
                        # 폭력(id=0)만 리포트 (필요하면 NonFight도 리포트하도록 변경)
                        if cls == 0:
                            det_list.append(_build_detection_item_det(cls, conf, x1, y1, x2, y2))
            except Exception as e:
                print(f"feat_detect_violence : YOLO inference error -> {e}")

        # ---- (B) 분류형(I3D/LSTM) ----
        elif clip_model is not None:
            try:
                # 프레임 버퍼링 (STRIDE 간격으로 채움)
                if not hasattr(run_violence_detect, "_stride"):
                    run_violence_detect._stride = 0
                run_violence_detect._stride = (run_violence_detect._stride + 1) % STRIDE
                if run_violence_detect._stride == 0:
                    clip_buf.append(frame.copy())

                if len(clip_buf) >= SEQ_LEN:
                    frames = list(clip_buf)[-SEQ_LEN:]
                    x = _preprocess_clip(frames, resize=224, device=device)
                    import torch
                    with torch.no_grad():
                        probs = _softmax_logits(clip_model(x))[0].detach().cpu().numpy()
                    cls_id = int(probs.argmax())
                    conf   = float(probs.max())

                    # 폭력(Fight)만 리포트 (비폭력은 드롭)
                    if cls_id == 0:
                        det_list.append(_build_detection_item_cls(cls_id, conf))
            except Exception as e:
                print(f"feat_detect_violence : classifier inference error -> {e}")

        # ---- 결과 패키징 ----
        result_package = {
            "detection": det_list,
            "timestamp": frame_time,
            "analyzer_name": analyzer_name,   # 보통 "feat_detect_violence"
            "detection_count": len(det_list),
            "patrol_number": patrol_no,
        }

        try:
            aggregation_queue.put(result_package, timeout=0.1)
        except queue.Full:
            pass  # 취합 큐가 가득하면 드롭

    print("feat_detect_violence : worker stop")

# ---- 단독 실행(옵션) ----
def _standalone():
    """카메라 0에서 1~2초 정도 돌려보고 결과 큐 출력"""
    import cv2
    q_in = queue.Queue()
    q_out = queue.Queue()
    ev = threading.Event()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("feat_detect_violence standalone: cannot open camera")
        return
    for i in range(40):  # 잠깐 몇 프레임만 밀어넣기
        ok, frame = cap.read()
        if not ok: break
        q_in.put({
            "frame": frame,
            "frame_count": i+1,
            "timestamp": time.time(),
            "patrol_number": 1,
        })
        time.sleep(0.02)
    cap.release()

    t = threading.Thread(target=run_violence_detect, args=(q_in, q_out, "feat_detect_violence", ev), daemon=True)
    t.start()
    time.sleep(1.5)
    ev.set()
    t.join()

    while not q_out.empty():
        print(json.dumps(q_out.get(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _standalone()

