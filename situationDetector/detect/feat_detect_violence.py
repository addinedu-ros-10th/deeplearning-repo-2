import cv2
import time
import math
import json
import queue
import threading
import numpy as np
import os

# ---------------------------------------------------------
# 모델 파일 경로 설정 (우선순위: ONNX -> PyTorch)
#   - 본인 모델 경로/파일명을 실제 상황에 맞게 수정하세요.
#   - 예) best.onnx 또는 best.pt
# ---------------------------------------------------------
VIOLENCE_MODEL_ONNX = "situationDetector/detect/feat_detect_violence/best.onnx"
VIOLENCE_MODEL_PT   = "situationDetector/detect/feat_detect_violence/best.pt"

# 분류 라벨 (학습 라벨 순서와 반드시 맞춰주세요!)
# NOTE: 학습 시 클래스 인덱스가 [0:Fight, 1:NonFight]가 아니면 순서를 바꾸세요.
VIOLENCE_CLASSES = ["Fight", "NonFight"]

# 전처리 파라미터 (자신의 모델에 맞게 조정)
IMG_SIZE = 224  # 예: 224x224 입력
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # torchvision 기준
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _preprocess_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR(OpenCV) -> 모델 입력 텐서(1,3,H,W), float32"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))        # HWC -> CHW
    x = np.expand_dims(x, 0).astype(np.float32)  # (1,3,H,W)
    return x

class ViolenceClassifier:
    """
    ONNX(권장) 또는 PyTorch 모델을 로드해 프레임 단위로
    Fight/NonFight 확률을 반환합니다.
    """
    def __init__(self):
        self.kind = None
        self.session = None
        self.input_name = None
        self.model_pt = None

        # 1) ONNX 우선
        if os.path.exists(VIOLENCE_MODEL_ONNX):
            try:
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(VIOLENCE_MODEL_ONNX, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.kind = "onnx"
                print("[violence] ONNX model loaded:", VIOLENCE_MODEL_ONNX)
                return
            except Exception as e:
                print("[violence] ONNX load failed:", e)

        # 2) PyTorch(.pt) 대안
        if os.path.exists(VIOLENCE_MODEL_PT):
            try:
                import torch
                self.model_pt = torch.jit.load(VIOLENCE_MODEL_PT, map_location="cpu") \
                                if VIOLENCE_MODEL_PT.endswith(".pt") else None
                if self.model_pt is None:
                    # 일반 state_dict 형태일 경우 – 사용자 코드에 맞게 로더 구현 필요
                    # 여기선 간단히 실패로 처리
                    raise RuntimeError("Unsupported .pt format; provide TorchScript .pt")
                self.model_pt.eval()
                self.kind = "torch"
                print("[violence] TorchScript model loaded:", VIOLENCE_MODEL_PT)
                return
            except Exception as e:
                print("[violence] Torch load failed:", e)

        # 3) 둘 다 없으면 더미(항상 0.5/0.5) – 파이프라인 동작 확인용
        self.kind = "dummy"
        print("[violence] WARNING: No model found. Using dummy 0.5/0.5 outputs.")

    def predict_probs(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns probs (2,) for [Fight, NonFight]
        """
        x = _preprocess_bgr(frame_bgr)

        if self.kind == "onnx":
            ort_out = self.session.run(None, {self.input_name: x})
            logits = ort_out[0]  # (1,2) 가정
            probs = softmax(logits, axis=-1)[0]
            return probs.astype(np.float32)

        elif self.kind == "torch":
            import torch
            with torch.no_grad():
                t = torch.from_numpy(x)  # (1,3,H,W) float32
                logits = self.model_pt(t)  # (1,2) 가정
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                return probs.astype(np.float32)

        # dummy
        return np.array([0.5, 0.5], dtype=np.float32)

def run_violence_detect(
    analysis_frame_queue: queue.Queue,
    aggregation_queue: queue.Queue,     # 취합 큐
    analyzer_name: str,                 # 모델/분석기 이름
    shutdown_event: threading.Event,
):
    """
    1) analysis_frame_queue에서 (frame_count, frame_time, frame) 가져옴
    2) 폭력 상황 분류 모델 수행 (프레임 단위)
    3) 결과를 JSON 스키마에 맞게 생성하여 aggregation_queue로 put
    """
    print("situationDetector (VIOLENCE) : 스레드 시작, 모델 로드")
    classifier = ViolenceClassifier()

    while not shutdown_event.is_set():
        try:
            frame_count, frame_time, frame = analysis_frame_queue.get(timeout=1.0)
            if frame is None:
                continue

            # 분류 확률 얻기: probs[0]=Fight, probs[1]=NonFight  (라벨 순서 확인 필수!)
            probs = classifier.predict_probs(frame)  # shape (2,)
            # 안전가드
            if probs.shape[0] != 2 or not np.isfinite(probs).all():
                probs = np.array([0.0, 0.0], dtype=np.float32)

            # 스키마에 맞게 두 클래스 모두 기록
            detection_list = []
            for class_id, (cls_name, p) in enumerate(zip(VIOLENCE_CLASSES, probs)):
                detection_list.append({
                    "class_id": int(class_id),   # 0=Fight, 1=NonFight (학습 라벨 순서와 일치시킬 것)
                    "class_name": cls_name,
                    "confidence": float(p),      # 확률(0~1)
                })

            result_package = {
                "detection": {
                    "feat_detect_violence": detection_list  # <- 기능명 키 아래 리스트
                },
                "timestamp": frame_time,        # 프레임 타임스탬프(송출 포맷에 맞게 사용)
                "analyzer_name": analyzer_name, # 분석기/모델 이름
                "detection_count": len(detection_list),
                "patrol_number": 1,
            }

            # 디버그 출력
            # print(json.dumps(result_package, ensure_ascii=False))

            try:
                aggregation_queue.put(result_package)
            except queue.Full:
                print("situationDetector (VIOLENCE) : DB 큐가 가득 참 / 분석 결과 버림")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"situationDetector (VIOLENCE) : 처리 중 오류 발생: {e}")
            break

    print("situationDetector (VIOLENCE) : 스레드 종료")
