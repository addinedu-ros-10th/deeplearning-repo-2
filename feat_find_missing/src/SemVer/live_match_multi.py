# live_match_multi.py
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_s"
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"
THRESH_SIM = 0.65  # 코사인 유사도 임계값(0.60~0.75부터 시작해 튜닝 권장)

def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)

embs = np.load(EMB_PATH).astype(np.float32)   # (N, D) — 저장 시 L2 정규화됨
names = np.load(NAMES_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ok, frame = cap.read()
    if not ok: break

    for f in app.get(frame):
        x1, y1, x2, y2 = f.bbox.astype(int)
        q = l2n(f.embedding)
        sims = embs @ q                              # (N,)
        j = int(np.argmax(sims))
        sim = float(sims[j])

        match = sim >= THRESH_SIM
        color = (0, 200, 0) if match else (60, 60, 230)
        label = f"{names[j]} ({sim:.2f})" if match else f"UNKNOWN ({sim:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("ArcFace Multi (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
