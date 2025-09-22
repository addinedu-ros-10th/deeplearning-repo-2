# live_match_multi.py
# 실시간: (a) 일반 임베딩 + (b) ROI 하관 가림(가상 마스크) 임베딩을 병렬 산출하여 더 높은 유사도로 판정.
# 작은 얼굴엔 임계값을 약간 상향. Top-3 로깅으로 현장 튜닝에 도움.

import cv2, numpy as np, torch
from insightface.app import FaceAnalysis

# ====== 설정 ======
MODEL_NAME = "buffalo_s"   # 필요시 "buffalo_m"로 변경하여 비교
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

BASE_THRESH = 0.625        # 기본 임계값(현장에 맞춰 0.60~0.68 탐색 권장)
SMALL_FACE_INC = 0.04      # 얼굴이 매우 작을 때 임계값 가산치

# ====== 유틸 ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def tta_faces(app, img):
    faces = app.get(img)
    if not faces: return []
    outs = []
    H, W = img.shape[:2]
    flip = cv2.flip(img, 1)
    faces_f = app.get(flip)

    for f in faces:
        e1 = l2n(f.embedding)
        # flip-TTA: flip 이미지에서도 가장 큰 얼굴 사용(근사)
        if faces_f:
            faces_f.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
            e2 = l2n(faces_f[0].embedding)
            e = l2n((e1 + e2) / 2.0)
        else:
            e = e1
        outs.append((f, e))
    return outs

def masked_query_emb(app, frame, f):
    """탐지된 얼굴 ROI에 '가상 마스크'를 씌우고 재임베딩(하관 가림 대응)"""
    x1, y1, x2, y2 = f.bbox.astype(int)
    roi = frame[max(0,y1):y2, max(0,x1):x2]
    if roi.size == 0: return None

    h, w = roi.shape[:2]
    y_top  = int(h * 0.58)
    y_bot  = int(h * 0.90)
    x_l    = int(w * 0.05)
    x_r    = int(w * 0.95)
    roi_masked = roi.copy()
    if y_bot > y_top and x_r > x_l:
        fill = int(np.mean(roi_masked[y_top:y_bot, x_l:x_r]))
        cv2.rectangle(roi_masked, (x_l, y_top), (x_r, y_bot), (fill, fill, fill), -1)

    faces = app.get(roi_masked)
    if not faces: return None
    faces.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
    e1 = l2n(faces[0].embedding)

    flip = cv2.flip(roi_masked, 1)
    faces_f = app.get(flip)
    if faces_f:
        faces_f.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
        e2 = l2n(faces_f[0].embedding)
        return l2n((e1 + e2)/2)
    return e1

# ====== 앱/데이터 ======
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)

embs  = np.load(EMB_PATH).astype(np.float32)  # (P, D) — 사람별 대표 임베딩
names = np.load(NAMES_PATH)

# ====== 카메라 ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 다른 index를 사용하거나 권한을 확인하세요.")

# ====== 루프 ======
while True:
    ok, frame = cap.read()
    if not ok: break

    fe_list = tta_faces(app, frame)  # [(face, emb), ...]
    H, W = frame.shape[:2]

    for f, q_full in fe_list:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = max(1, (x2 - x1) * (y2 - y1))
        face_ratio = area / (H * W)

        # 동적 임계값: 얼굴이 매우 작으면 보수적으로
        thresh = BASE_THRESH + (SMALL_FACE_INC if face_ratio < 0.04 else 0.0)

        # 1) 일반 경로
        sims_full = embs @ q_full
        j_full = int(np.argmax(sims_full))
        sim_full = float(sims_full[j_full])

        # 2) 마스크 경로(가능하면)
        q_mask = masked_query_emb(app, frame, f)
        if q_mask is not None:
            sims_mask = embs @ q_mask
            j_mask = int(np.argmax(sims_mask))
            sim_mask = float(sims_mask[j_mask])
        else:
            j_mask, sim_mask = j_full, -1.0

        # 3) 더 강한 신호 채택(마스크 경로는 미세 보수화)
        if sim_mask > sim_full:
            j, sim, used, thr = j_mask, sim_mask, "mask", (thresh + 0.01)
        else:
            j, sim, used, thr = j_full, sim_full, "full", thresh

        match = sim >= thr

        # 디버깅: Top-3 로깅(튜닝 시 유용)
        top3_full = [(str(names[i]), float(sims_full[i])) for i in np.argsort(-sims_full)[:3]]
        if q_mask is not None:
            top3_mask = [(str(names[i]), float(sims_mask[i])) for i in np.argsort(-sims_mask)[:3]]
            print(f"Top3 full: {top3_full} | Top3 mask: {top3_mask}")
        else:
            print(f"Top3 full: {top3_full} | mask: N/A")

        # 시각화
        color = (0, 200, 0) if match else (60, 60, 230)
        label = f"{names[j]} ({sim:.2f})/{used} thr{thr:.2f}" if match else f"UNKNOWN ({sim:.2f})/{used}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    cv2.imshow("ArcFace Multi (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
