import queue
import threading
import time
import json
from collections import deque

TRASH_MODEL_PATH = "situationDetector/detect/feat_detect_trash/trash_detector_i3d_full_model.pt"

SEQ_LEN = 32       # 입력 프레임 수
RESIZE  = 224      # 네트워크 입력 해상도
STRIDE  = 2        # 프레임 샘플링 간격
INFER_DEVICE = "cpu"   

TRASH_LABELS = {
    0: "trash_dumping",
    1: "not_trash",
}

def _try_load_yolo(weights_path: str):
    """YOLO 포맷일 경우 모델 로드 (아니면 예외)"""
    try:
        from ultralytics import YOLO
        m = YOLO(weights_path)
        return m
    except Exception:
        return None

def _try_load_classifier(weights_path: str, device: str = INFER_DEVICE):
    """TorchScript 또는 일반 nn.Module 로드 (I3D/LSTM 등 분류형)"""
    import torch
    # TorchScript
    try:
        m = torch.jit.load(weights_path, map_location=device).eval()
        return ("torchscript", m, device)
    except Exception:
        pass
    # nn.Module
    try:
        m = torch.load(weights_path, map_location=device)
        if hasattr(m, "eval"):
            m.eval()
        return ("nnmodule", m, device)
    except Exception:
        return None

def _preprocess_clip(frames, resize=RESIZE, device: str = INFER_DEVICE):
    """frames(list of BGR uint8) -> (1, T, 3, H, W) float32[0..1] tensor"""
    import cv2, numpy as np, torch
    imgs = []
    for f in frames:
        im = cv2.resize(f, (resize, resize))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        imgs.append(im)
    x = np.stack(imgs, 0)              # (T,H,W,3)
    x = np.transpose(x, (0,3,1,2))     # (T,3,H,W)
    x = x[None, ...]                    # (1,T,3,H,W)
    return torch.from_numpy(x).to(device)

def _softmax_logits(y):
    """다양한 모양의 logits를 소프트맥스로 확률화 (B,T,C -> T 평균)"""
    import torch
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, torch.Tensor) and y.ndim > 2:
        y = y.mean(dim=1)  # 시간 평균
    return torch.softmax(y, dim=-1)

def _build_detection_item_det(cls_id: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    """탐지형 결과(박스 포함)"""
    return {
        "class_id": int(cls_id),
        "class_name": TRASH_LABELS.get(int(cls_id), str(int(cls_id))),
        "confidence": float(conf),
        "bbox": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
    }

def _build_detection_item_cls(cls_id: int, conf: float):
    """분류형 결과(박스 없음)"""
    return {
        "class_id": int(cls_id),
        "class_name": TRASH_LABELS.get(int(cls_id), str(int(cls_id))),
        "confidence": float(conf),
        "bbox": None,
    }

def run_trash_detect(
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
    print("feat_detect_trash : worker start, loading model...")

    # 1) YOLO 시도
    yolo_model = _try_load_yolo(TRASH_MODEL_PATH)

    cls_mode = None
    cls_model = None
    cls_device = INFER_DEVICE
    if yolo_model is None:
        # 2) 분류형(I3D/LSTM) 시도
        loaded = _try_load_classifier(TRASH_MODEL_PATH, INFER_DEVICE)
        if loaded is not None:
            cls_mode, cls_model, cls_device = loaded
            print(f"feat_detect_trash : classifier loaded ({cls_mode}) on {cls_device}")
        else:
            print("feat_detect_trash : model load failed (neither YOLO nor classifier)")
    else:
        print("feat_detect_trash : YOLO model loaded")

    clip_buf = deque(maxlen=SEQ_LEN)
    stride_tick = 0

    while not shutdown_event.is_set():
        # ---- 입력 꺼내기 (dict/tuple 호환) ----
        try:
            item = analysis_frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"feat_detect_trash : queue error -> {e}")
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

        # ---- 추론 ----
        try:
            if yolo_model is not None:
                # 탐지형 경로
                res = yolo_model(frame, verbose=False)[0]
                if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                    for x1, y1, x2, y2, conf, cls in res.boxes.data.tolist():
                        det_list.append(_build_detection_item_det(int(cls), conf, x1, y1, x2, y2))
            elif cls_model is not None:
                # 분류형 경로: 시퀀스 누적 후 분류
                stride_tick = (stride_tick + 1) % max(1, STRIDE)
                if stride_tick == 0:
                    clip_buf.append(frame.copy())

                if len(clip_buf) >= SEQ_LEN:
                    frames = list(clip_buf)[-SEQ_LEN:]
                    x = _preprocess_clip(frames, resize=RESIZE, device=cls_device)

                    import torch
                    with torch.no_grad():
                        probs = _softmax_logits(cls_model(x))[0].detach().cpu().numpy()
                    cls_id = int(probs.argmax())
                    conf   = float(probs.max())
                    # 분류형은 bbox가 없으므로 한 개의 아이템만 추가
                    det_list.append(_build_detection_item_cls(cls_id, conf))
        except Exception as e:
            print(f"feat_detect_trash : inference error -> {e}")

        # ---- 결과 패키징 ----
        result_package = {
            "detection": det_list,
            "timestamp": frame_time,
            "analyzer_name": analyzer_name,   # 보통 "feat_detect_trash"
            "detection_count": len(det_list),
            "patrol_number": patrol_no,
        }

        try:
            aggregation_queue.put(result_package, timeout=0.1)
        except queue.Full:
            # 취합 큐가 가득하면 해당 프레임 결과는 드롭
            pass

    print("feat_detect_trash : worker stop")


# -------- 단독 실행(옵션) --------
def _standalone():
    """카메라 0으로 간단 단독 테스트 (분류형은 클립 누적 필요)"""
    import cv2
    q_in = queue.Queue()
    q_out = queue.Queue()
    ev = threading.Event()

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("feat_detect_trash standalone: cannot grab frame")
        return

    # 분류형 테스트를 위해 동일 프레임을 여러 번 밀어 넣어도 동작은 함(정확도 X)
    for i in range(SEQ_LEN):
        q_in.put({
            "frame": frame,
            "frame_count": i + 1,
            "timestamp": time.time(),
            "patrol_number": 1,
        })

    t = threading.Thread(target=run_trash_detect, args=(q_in, q_out, "feat_detect_trash", ev), daemon=True)
    t.start()
    time.sleep(1.0)
    ev.set()
    t.join()

    while not q_out.empty():
        print(json.dumps(q_out.get(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _standalone()
