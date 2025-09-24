import os
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# =========================
# 설정
# =========================
USE_GPU = torch.cuda.is_available()
MODEL_NAME = "buffalo_s"   # 가벼운 번들: buffalo_s (또는 "buffalo_m", 기존은 "buffalo_l")
DET_SIZE = (320, 320)      # 감지 해상도(작을수록 빠름; 320~640 사이에서 FPS/정확도 타협)
REF_IMG_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_person.jpg"
REF_EMB_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_embedding.npy"

THRESH_COS_DIST = 0.35     # 코사인 거리 임계값 (작을수록 같다고 판단). 0.32~0.40 사이 튜닝 권장
FRAME_SKIP = 1             # 0 또는 1: 매 프레임, 2: 격프레임 등

# =========================
# 유틸
# =========================
def l2_normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    n = np.linalg.norm(vec)
    if n < eps:
        return vec
    return vec / n

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    # a, b: L2 정규화 전제. 혹은 내부에서 정규화해도 됨.
    a = l2_normalize(a, eps)
    b = l2_normalize(b, eps)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    return 1.0 - cosine_similarity(a, b, eps)

# =========================
# 참조 임베딩 준비(없으면 생성)
# =========================
def build_or_load_ref_embedding(app: FaceAnalysis, ref_img_path: str, ref_emb_path: str) -> np.ndarray:
    if os.path.exists(ref_emb_path):
        ref_emb = np.load(ref_emb_path).astype(np.float32)
        return l2_normalize(ref_emb)

    img = cv2.imread(ref_img_path)
    if img is None:
        raise FileNotFoundError(f"참조 이미지가 없습니다: {ref_img_path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("참조 이미지에서 얼굴을 찾지 못했습니다. 더 선명한 정면 사진을 사용해 주세요.")

    # 가장 큰 얼굴 선택
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    ref_emb = faces[0].embedding
    ref_emb = l2_normalize(ref_emb)
    np.save(ref_emb_path, ref_emb)
    print(f"[INFO] 참조 임베딩 저장 완료: {ref_emb_path}")
    return ref_emb

# =========================
# 메인
# =========================
def main():
    # FaceAnalysis 준비
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(
        ctx_id=0 if USE_GPU else -1,
        det_size=DET_SIZE,   # 감지기 입력 크기 고정(속도 안정)
    )
    print(f"[INFO] Device: {'GPU' if USE_GPU else 'CPU'}, Model: {MODEL_NAME}, det_size={DET_SIZE}")

    # 참조 임베딩 로드/생성
    ref_emb = build_or_load_ref_embedding(app, REF_IMG_PATH, REF_EMB_PATH)

    cap = cv2.VideoCapture(0)  # 카메라 인덱스 필요시 조정
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 장치 인덱스 또는 권한을 확인하세요.")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] 프레임을 읽지 못했습니다.")
                break

            # 필요시 다운스케일(속도↑) — 예: 긴 변 960 제한
            h, w = frame.shape[:2]
            max_side = max(h, w)
            if max_side > 960:
                scale = 960.0 / max_side
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

            # 프레임 스킵
            if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
                frame_idx += 1
                cv2.imshow("Live Match", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            faces = app.get(frame)

            # 결과 표시
            for f in faces:
                emb = l2_normalize(f.embedding)
                cos_sim = cosine_similarity(ref_emb, emb)
                cos_dist = 1.0 - cos_sim

                x1, y1, x2, y2 = map(int, f.bbox)
                if cos_dist <= THRESH_COS_DIST:
                    label = f"MATCH ✓  sim={cos_sim:.3f}  dist={cos_dist:.3f}"
                    color = (0, 200, 0)
                else:
                    label = f"NO MATCH  sim={cos_sim:.3f}  dist={cos_dist:.3f}"
                    color = (60, 60, 220)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            cv2.imshow("Live Match", frame)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
