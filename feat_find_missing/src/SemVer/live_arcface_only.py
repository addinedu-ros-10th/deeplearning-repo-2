#4) 실시간 매칭(ArcFace 단독 버전: 가장 간단)

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis



# 1) ArcFace 로드
use_gpu = torch.cuda.is_available()
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0 if use_gpu else -1)

# 2) 참조 임베딩 로드
ref_emb = np.load("/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_embedding.npy")

# 3) 카메라
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없어요.")

THRESH = 0.35  # 시작 임계값(0.30~0.45 사이에서 나중에 튜닝)

while True:
    ok, frame = cap.read()
    if not ok: break

    faces = app.get(frame)  # 얼굴 검출 + 임베딩
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = f.embedding
        emb = emb / np.linalg.norm(emb)

        # 코사인 유사도
        sim = float(np.dot(ref_emb, emb))

        if sim >= THRESH:
            color = (0, 255, 0)
            label = f"FOUND {sim:.2f}"
        else:
            color = (0, 0, 255)
            label = f"UNKNOWN {sim:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow("ArcFace Live (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()

