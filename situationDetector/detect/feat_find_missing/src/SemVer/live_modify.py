import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# --- 설정 ---
MODEL_NAME = "buffalo_s"
DET_SIZE = (320, 320)
REF_EMB_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_embedding.npy"
THRESH_DIST = 0.35  # 코사인 거리 임계값 (작을수록 동일인)

def l2_normalize(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

# 1) ArcFace 로드
use_gpu = False
try:
    use_gpu = torch.cuda.is_available()
except Exception:
    pass
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if use_gpu else -1, det_size=DET_SIZE)

# 2) 참조 임베딩 로드(+정규화)
ref_emb = np.load(REF_EMB_PATH).astype(np.float32)
ref_emb = l2_normalize(ref_emb)

# 3) 카메라
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = app.get(frame)
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = l2_normalize(f.embedding)

        cos_sim = float(np.dot(ref_emb, emb))
        cos_dist = 1.0 - cos_sim

        is_match = cos_dist <= THRESH_DIST
        color = (0, 200, 0) if is_match else (50, 50, 230)
        label = f"{'FOUND' if is_match else 'UNKNOWN'}  sim={cos_sim:.2f} dist={cos_dist:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("ArcFace Live (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
