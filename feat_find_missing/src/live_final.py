# live_match_multi_refactored.py
import cv2
import numpy as np
import torch
import re
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from pathlib import Path

# ====== 1. 설정 관리 ======
class Config:
    # 모델 및 경로
    MODEL_NAME = "buffalo_s"
    BASE_DIR = Path("/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src")
    EMB_PATH = BASE_DIR / "embeddings.npy"
    NAMES_PATH = BASE_DIR / "names.npy"

    # 탐지 및 정렬 설정
    DET_SIZE = (320, 320)
    ALIGN_SIZE = 112

    # 매칭 임계값
    BASE_THRESH = 0.625
    SMALL_FACE_INC = 0.04
    SMALL_FACE_AREA_RATIO = 0.04 # H*W 대비 얼굴 면적 비율

    # 마스크 생성 파라미터
    MASK_TOP_RATIO = 0.58
    MASK_BOT_RATIO = 0.90
    MASK_PAD_RATIO = 0.05

# ====== 유틸리티 함수 (변경 없음) ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def parse_name_birthday(name_token: str) -> tuple[str, str]:
    m = re.match(r"^(\d{8})_(.+)$", str(name_token))
    if not m:
        return str(name_token), ""
    yyyymmdd, name = m.groups()
    return name, yyyymmdd

def rec_embed(rec_model, crop_bgr: np.ndarray, do_flip_tta=True) -> np.ndarray:
    if crop_bgr.dtype != np.uint8:
        crop_bgr = np.clip(crop_bgr, 0, 255).astype(np.uint8)
    
    emb1 = rec_model.get_feat(crop_bgr).astype(np.float32).squeeze()
    if not do_flip_tta:
        return l2n(emb1)

    crop_bgr_f = cv2.flip(crop_bgr, 1)
    emb2 = rec_model.get_feat(crop_bgr_f).astype(np.float32).squeeze()
    return l2n((emb1 + emb2) / 2.0)

def make_masked_lower(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    y_top = int(h * Config.MASK_TOP_RATIO)
    y_bot = int(h * Config.MASK_BOT_RATIO)
    x_l = int(w * Config.MASK_PAD_RATIO)
    x_r = int(w * (1 - Config.MASK_PAD_RATIO))
    
    out = crop_bgr.copy()
    if y_bot > y_top and x_r > x_l:
        fill_area = out[y_top:y_bot, x_l:x_r]
        fill = int(np.mean(fill_area)) if fill_area.size > 0 else 128
        cv2.rectangle(out, (x_l, y_top), (x_r, y_bot), (fill, fill, fill), -1)
    return out

def detect_and_align(app: FaceAnalysis, frame: np.ndarray) -> list[tuple[object, np.ndarray]]:
    faces = app.get(frame)
    if not faces:
        return []
    
    return [
        (f, face_align.norm_crop(frame, landmark=f.kps, image_size=Config.ALIGN_SIZE))
        for f in faces
    ]

# ====== 2. 로직 개선: 매칭 함수 분리 ======
def get_best_match(query_emb: np.ndarray, ref_embs: np.ndarray) -> tuple[int, float]:
    """쿼리 임베딩과 참조 임베딩들 간의 최고 유사도 및 인덱스 반환"""
    sims = ref_embs @ query_emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    return best_idx, best_sim

# ====== 앱/데이터 로드 ======
try:
    app = FaceAnalysis(name=Config.MODEL_NAME)
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=Config.DET_SIZE)
    rec_model = app.models.get('recognition')
    if rec_model is None:
        raise RuntimeError("InsightFace recognition model not found.")
    
    ref_embeddings = np.load(Config.EMB_PATH).astype(np.float32)
    ref_names = np.load(Config.NAMES_PATH)
except Exception as e:
    print(f"초기화 실패: {e}")
    exit()

# ====== 핵심: 한 프레임 추론 함수 (개선됨) ======
def infer_once(frame: np.ndarray) -> list[dict]:
    results = []
    detections = detect_and_align(app, frame)
    H, W = frame.shape[:2]

    for face_obj, aligned_crop in detections:
        # 동적 임계값 계산
        x1, y1, x2, y2 = face_obj.bbox.astype(int)
        area = max(1, (x2 - x1) * (y2 - y1))
        face_ratio = area / (H * W)
        thresh = Config.BASE_THRESH + (Config.SMALL_FACE_INC if face_ratio < Config.SMALL_FACE_AREA_RATIO else 0.0)

        # 1) Full-face 임베딩으로 최고 매칭 찾기
        q_full = rec_embed(rec_model, aligned_crop)
        j_full, sim_full = get_best_match(q_full, ref_embeddings)

        # 2) Masked-face 임베딩으로 최고 매칭 찾기
        crop_masked = make_masked_lower(aligned_crop)
        q_mask = rec_embed(rec_model, crop_masked)
        j_mask, sim_mask = get_best_match(q_mask, ref_embeddings)

        # 3) 더 나은 경로 선택
        if sim_mask > sim_full:
            best_idx, best_sim, match_type = j_mask, sim_mask, "mask"
        else:
            best_idx, best_sim, match_type = j_full, sim_full, "full"

        # 최종 임계값 통과 시 결과에 추가
        if best_sim >= thresh:
            name, birthday = parse_name_birthday(ref_names[best_idx])
            results.append({
                "name": name,
                "birthday": birthday,
                "mask_type": match_type,
                "confidence": best_sim
            })
            
            # (시각화)
            label = f"{name} ({best_sim:.2f})/{match_type}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    return results

# ====== 데모 루프 (개선됨) ======
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            matched_results = infer_once(frame)
            
            if matched_results:
                print(f"현재 프레임 매칭 결과: {matched_results}")
            else:
                print("매칭 없음")

            cv2.imshow("Real-time Face Match (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 4. 루프 종료 시 리소스 자동 해제
        cap.release()
        cv2.destroyAllWindows()