# live_match_multi.py  (교체)
import cv2, numpy as np, torch
from insightface.app import FaceAnalysis

MODEL_NAME = "buffalo_s"   # 필요시 "buffalo_m"로 변경해서 비교
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

BASE_THRESH = 0.625  #  임계값

def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def tta_query_emb(app, img):
    faces = app.get(img)
    if not faces: return []
    outs = []
    for f in faces:
        e1 = l2n(f.embedding)
        flip = cv2.flip(img, 1)
        faces_f = app.get(flip)
        if faces_f:
            faces_f.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
            e2 = l2n(faces_f[0].embedding)
            e = l2n((e1 + e2) / 2.0)
        else:
            e = e1
        outs.append((f, e))
    return outs

app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)

embs  = np.load(EMB_PATH).astype(np.float32)  # (P, D), 사람별 대표
names = np.load(NAMES_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ok, frame = cap.read()
    if not ok: break

    fe_list = tta_query_emb(app, frame)  # [(face, emb), ...]
    for f, q in fe_list:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = (x2 - x1) * (y2 - y1) + 1e-6
        # 동적 임계값: 프레임 크기 대비 얼굴이 작을수록(멀수록) 임계값 ↑
        H, W = frame.shape[:2]
        face_ratio = area / (H * W)
        thresh = BASE_THRESH + (0.04 if face_ratio < 0.04 else 0.0)  # 필요시 더 조정

        sims = embs @ q  # (P,)
        j = int(np.argmax(sims))
        sim = float(sims[j])
        match = sim >= thresh

        # 디버깅: Top-3 후보 로깅
        top3_idx = np.argsort(-sims)[:3]
        print("Top3:", [(str(names[i]), float(sims[i])) for i in top3_idx])

        color = (0, 200, 0) if match else (60, 60, 230)
        label = f"{names[j]} ({sim:.2f})" if match else f"UNKNOWN ({sim:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("ArcFace Multi (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
