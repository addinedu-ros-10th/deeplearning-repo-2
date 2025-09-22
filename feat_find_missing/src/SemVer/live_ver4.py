# live_match_multi.py
# - 매 프레임 처리 함수 infer_once(frame) 추가: 매치된 사람만 리스트로 반환
# - 반환 스키마:
#   {
#     "name": <str>,
#     "birthday": <YYYYMMDD>,
#     "mask_type": "mask" | "full",
#     "confidence": <float>  # 선택된 경로의 코사인 유사도
#   }

import cv2, numpy as np, torch, re
from insightface.app import FaceAnalysis

# ====== 설정 ======
MODEL_NAME = "buffalo_s"   # 필요시 "buffalo_m"로 변경하여 비교
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

BASE_THRESH     = 0.625    # 기본 임계값(현장에 맞춰 0.60~0.68 탐색 권장)
SMALL_FACE_INC  = 0.04     # 얼굴이 매우 작을 때 임계값 가산치
MASK_PATH_DELTA = 0.01     # mask 경로 선택 시 약간 더 보수적인 임계값

# ====== 유틸 ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def parse_name_birthday(name_token: str):
    """
    names.npy 의 항목이 'YYYYMMDD_name' 형태라고 가정.
    예) '19980317_choi' -> (name='choi', birthday='19980317')
    규칙과 다르면 생년월일은 빈 문자열로 반환.
    """
    m = re.match(r"^(\d{8})_(.+)$", str(name_token))
    if not m:
        return str(name_token), ""  # 포맷 불일치 시 생년월일 빈 값
    yyyymmdd, name = m.groups()
    return name, yyyymmdd

def tta_faces(app, img):
    faces = app.get(img)
    if not faces: return []
    outs = []
    flip = cv2.flip(img, 1)
    faces_f = app.get(flip)
    if faces_f:
        faces_f.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
        e2_global = l2n(faces_f[0].embedding)
    else:
        e2_global = None

    for f in faces:
        e1 = l2n(f.embedding)
        if e2_global is not None:
            e = l2n((e1 + e2_global) / 2.0)
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
names = np.load(NAMES_PATH)                   # (P,) — 등록 이름 토큰(예: '19980317_choi')

# ====== 핵심: 한 프레임 추론 함수(리턴 제공) ======
def infer_once(frame):
    """
    입력 프레임에서 매치된 사람만 반환.
    반환: List[Dict]  (각 딕셔너리는 아래 4개 필드만 포함)
      - "name": <str>
      - "birthday": <YYYYMMDD> (규칙 불일치 시 빈 문자열 "")
      - "mask_type": "mask" | "full"
      - "confidence": <float>  # 선택된 경로의 코사인 유사도
    """
    results = []
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
            j, sim, used, thr = j_mask, sim_mask, "mask", (thresh + MASK_PATH_DELTA)
        else:
            j, sim, used, thr = j_full, sim_full, "full", thresh

        match = sim >= thr
        if not match:
            continue  # 임계 미만은 반환하지 않음(요청 스키마만 반환)

        reg_token = names[j]
        name, birthday = parse_name_birthday(reg_token)

        # 화면 표시(원하면 주석 처리)
        color = (0, 200, 0)
        label = f"{name} ({sim:.2f})/{used}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        results.append({
            "name": name,
            "birthday": birthday,       # 'YYYYMMDD' 또는 ''
            "mask_type": used,          # "full" or "mask"
            "confidence": float(sim)    # 코사인 유사도
        })
    return results

# ====== 데모 루프 (원하면 그대로 사용) ======
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 index를 사용하거나 권한을 확인하세요.")

    while True:
        ok, frame = cap.read()
        if not ok: break

        # 핵심: 리턴값 사용
        matched = infer_once(frame)
        # 여기서 matched 리스트를 원하는 곳(서버 전송 등)에 바로 사용하면 됨.
        # 예: for item in matched: send(item)
        if matched:  # 비어 있지 않으면
            print("현재 프레임 매칭 결과:", matched)
        else:
            print("매칭 없음")
    

        cv2.imshow("ArcFace Multi (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
