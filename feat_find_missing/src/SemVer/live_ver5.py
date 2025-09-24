# live_match_multi.py
# - 정렬된 얼굴 크롭에서 바로 임베딩 산출 (재탐지 X)
# - 정렬 크롭의 하관만 가려 mask 임베딩 생성
# - infer_once(frame) -> 매치된 사람만 [{name,birthday,mask_type,confidence}] 반환

import cv2, numpy as np, torch, re
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ====== 설정 ======
MODEL_NAME = "buffalo_s"   # 필요시 "buffalo_m" 비교
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

BASE_THRESH     = 0.625    # 기본 임계값(현장에 맞춰 0.60~0.68 탐색 권장)
SMALL_FACE_INC  = 0.04     # 얼굴이 매우 작을 때 임계값 가산치
ALIGN_SIZE      = 112      # norm_crop 기본 크기

# ====== 유틸 ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def parse_name_birthday(name_token: str):
    """
    names.npy 항목이 'YYYYMMDD_name' 형태라고 가정.
    예) '19980317_choi' -> (name='choi', birthday='19980317')
    규칙과 다르면 생년월일은 "".
    """
    m = re.match(r"^(\d{8})_(.+)$", str(name_token))
    if not m:
        return str(name_token), ""
    yyyymmdd, name = m.groups()
    return name, yyyymmdd

def rec_embed(rec_model, crop_bgr, do_flip_tta=True):
    """
    정렬된 얼굴 크롭(BGR, 112x112)을 받아 임베딩 반환.
    - ArcFaceONNX.get_feat(crop_bgr) 사용 (정렬 크롭에서 직접 임베딩)
    - flip-TTA: 좌우반전 이미지를 추가로 임베딩 후 평균
    """
    # crop_bgr는 uint8 BGR (112x112) 이어야 함
    if crop_bgr.dtype != np.uint8:
        crop_bgr = np.clip(crop_bgr, 0, 255).astype(np.uint8)

    emb1 = rec_model.get_feat(crop_bgr).astype(np.float32).squeeze()

    if do_flip_tta:
        crop_bgr_f = cv2.flip(crop_bgr, 1)
        emb2 = rec_model.get_feat(crop_bgr_f).astype(np.float32).squeeze()
        emb = l2n((emb1 + emb2) / 2.0)
    else:
        emb = l2n(emb1)
    return emb


def make_masked_lower(crop_bgr, top_ratio=0.58, bot_ratio=0.90, pad_ratio=0.05):
    """
    정렬된 얼굴 크롭(112x112) 하부를 사각형으로 덮어 하관 가림 효과.
    """
    h, w = crop_bgr.shape[:2]
    y_top = int(h * top_ratio)
    y_bot = int(h * bot_ratio)
    x_l   = int(w * pad_ratio)
    x_r   = int(w * (1 - pad_ratio))
    out = crop_bgr.copy()
    if y_bot > y_top and x_r > x_l:
        fill = int(np.mean(out[y_top:y_bot, x_l:x_r]))  # 주변 톤으로 채우기
        cv2.rectangle(out, (x_l, y_top), (x_r, y_bot), (fill, fill, fill), -1)
    return out

def detect_and_align(app, frame):
    """
    프레임에서 얼굴 탐지 -> (face_obj, 정렬크롭BGR) 리스트 반환.
    face_align.norm_crop(frame, f.kps, image_size=ALIGN_SIZE) 사용.
    """
    faces = app.get(frame)
    if not faces:
        return []
    outputs = []
    for f in faces:
        # f.kps: (5,2) 랜드마크; 정렬된 112x112 크롭 생성
        crop = face_align.norm_crop(frame, landmark=f.kps, image_size=ALIGN_SIZE)
        outputs.append((f, crop))
    return outputs

# ====== 앱/데이터 ======
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)
# 인식 모델 핸들
rec = app.models.get('recognition', None)
if rec is None:
    raise RuntimeError("InsightFace recognition model not found in app. Check model name or installation.")

embs  = np.load(EMB_PATH).astype(np.float32)  # (P, D)
names = np.load(NAMES_PATH)                   # (P,)

# ====== 핵심: 한 프레임 추론 함수(리턴 제공) ======
def infer_once(frame):
    """
    입력 프레임에서 매치된 사람만 반환.
    반환: List[Dict] (각 항목은 아래 4개 필드만 포함)
      - "name": <str>
      - "birthday": <YYYYMMDD> (규칙 불일치 시 "")
      - "mask_type": "mask" | "full"
      - "confidence": <float>  # 선택 경로의 코사인 유사도
    """
    results = []
    detections = detect_and_align(app, frame)  # [(face_obj, aligned_crop_bgr), ...]
    H, W = frame.shape[:2]

    for f, crop in detections:
        # 얼굴 크기에 비례해 임계값 동적 조정
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = max(1, (x2 - x1) * (y2 - y1))
        face_ratio = area / (H * W)
        thresh = BASE_THRESH + (SMALL_FACE_INC if face_ratio < 0.04 else 0.0)

        # 1) full 임베딩 (정렬 크롭 기준)
        q_full = rec_embed(rec, crop, do_flip_tta=True)
        sims_full = embs @ q_full
        j_full = int(np.argmax(sims_full))
        sim_full = float(sims_full[j_full])

        # 2) mask 임베딩 (정렬 크롭 하관 가림 후)
        crop_masked = make_masked_lower(crop)
        q_mask = rec_embed(rec, crop_masked, do_flip_tta=True)
        sims_mask = embs @ q_mask
        j_mask = int(np.argmax(sims_mask))
        sim_mask = float(sims_mask[j_mask])

        # 3) 경로 선택: mask가 더 높으면 mask 채택 (동률이면 full 유지)
        if sim_mask > sim_full:
            j, sim, used = j_mask, sim_mask, "mask"
        else:
            j, sim, used = j_full, sim_full, "full"

        # 임계값 통과 여부
        if sim < thresh:
            continue

        reg_token = names[j]
        name, birthday = parse_name_birthday(reg_token)

        # (원하면 시각화 — 필요 없으면 이 블록을 주석 처리)
        color = (0, 200, 0)
        label = f"{name} ({sim:.2f})/{used}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        results.append({
            "name": name,
            "birthday": birthday,
            "mask_type": used,         # "full" or "mask"
            "confidence": float(sim)   # 코사인 유사도
        })

    return results

# ====== 데모 루프 ======
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 index를 사용하거나 권한을 확인하세요.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        matched = infer_once(frame)
        # matched 를 그대로 서버 전송 등에 사용하면 됨.
        # 예) for item in matched: send_json(item)
        if matched:  # 비어 있지 않으면
            print("현재 프레임 매칭 결과:", matched)
        else:
            print("매칭 없음")

        cv2.imshow("ArcFace Multi (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
