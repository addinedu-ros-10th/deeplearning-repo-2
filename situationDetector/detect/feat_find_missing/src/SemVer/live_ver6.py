# live_ver6.py
# - 실종자 매칭 시 서버로 최소 데이터 전송
#   1) /missing/pre (사전 메타데이터)
#   2) TCP 파일 업로드 (샘플 MP4)
#   3) /missing/post (실종자 결과 JSON: 첫 1건)

import cv2, numpy as np, torch, re, time, json, socket, struct, hashlib, requests
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ====== 설정 (서버/차량/위치/파일) ======
SERVER_HTTP = "http://127.0.0.1:8000"   # http_api.py 서버
TCP_HOST, TCP_PORT = "127.0.0.1", 50060 # media_tcp_server.py 서버
PATROL_CAR_NAME = "Patrol_Car_1"
DEFAULT_LAT, DEFAULT_LNG = 37.5665, 126.9780  # 임시 위치(서울시청 좌표 예시)
LOCAL_CLIP_PATH = "/home/choi/test.mp4"       # 임시 샘플 MP4(전/후 10초 병합 파일로 교체 예정)

# ====== 모델 설정 ======
MODEL_NAME = "buffalo_s"
DET_SIZE   = (320, 320)
EMB_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

BASE_THRESH     = 0.625
SMALL_FACE_INC  = 0.04
ALIGN_SIZE      = 112

# ====== 유틸 ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def parse_name_birthday(name_token: str):
    m = re.match(r"^(\d{8})_(.+)$", str(name_token))
    if not m:
        return str(name_token), ""
    yyyymmdd, name = m.groups()
    return name, yyyymmdd

def rec_embed(rec_model, crop_bgr, do_flip_tta=True):
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
    h, w = crop_bgr.shape[:2]
    y_top = int(h * top_ratio)
    y_bot = int(h * bot_ratio)
    x_l   = int(w * pad_ratio)
    x_r   = int(w * (1 - pad_ratio))
    out = crop_bgr.copy()
    if y_bot > y_top and x_r > x_l:
        fill = int(np.mean(out[y_top:y_bot, x_l:x_r]))
        cv2.rectangle(out, (x_l, y_top), (x_r, y_bot), (fill, fill, fill), -1)
    return out

def detect_and_align(app, frame):
    faces = app.get(frame)
    if not faces:
        return []
    outputs = []
    for f in faces:
        crop = face_align.norm_crop(frame, landmark=f.kps, image_size=ALIGN_SIZE)
        outputs.append((f, crop))
    return outputs

# ====== 서버 통신 헬퍼 ======
def create_event(timestamp: float, lat: float, lng: float, patrol_car_name: str):
    body = {"timestamp": float(timestamp),
            "location": [float(lat), float(lng)],
            "patrol_car_name": patrol_car_name}
    r = requests.post(SERVER_HTTP + "/missing/pre", json=body, timeout=5)
    r.raise_for_status()
    return r.json()  # {event_id, media_id, rel_path}

def send_file_tcp(media_id: str, rel_path: str, local_mp4: str):
    with open(local_mp4, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    header = json.dumps({
        "media_id": media_id,
        "rel_path": rel_path,
        "file_size": len(data),
        "sha256": sha
    }).encode("utf-8")
    s = socket.create_connection((TCP_HOST, TCP_PORT), timeout=10)
    s.sendall(struct.pack(">I", len(header)))
    s.sendall(header)
    s.sendall(data)
    resp = s.recv(2)
    s.close()
    return resp == b"OK"

def post_missing_result(event_id: str, name: str, birthday: str, mask_type: str, confidence: float):
    body = {"event_id": event_id, "name": name, "birthday": birthday,
            "mask_type": mask_type, "confidence": float(confidence)}
    r = requests.post(SERVER_HTTP + "/missing/post", json=body, timeout=5)
    r.raise_for_status()
    return r.json()

# ====== 앱/데이터 로드 ======
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)
rec = app.models.get('recognition', None)
if rec is None:
    raise RuntimeError("InsightFace recognition model not found. Check model name/install.")

embs  = np.load(EMB_PATH).astype(np.float32)
names = np.load(NAMES_PATH)

# ====== 프레임 추론 ======
def infer_once(frame):
    results = []
    detections = detect_and_align(app, frame)
    H, W = frame.shape[:2]

    for f, crop in detections:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = max(1, (x2 - x1) * (y2 - y1))
        face_ratio = area / (H * W)
        thresh = BASE_THRESH + (SMALL_FACE_INC if face_ratio < 0.04 else 0.0)

        q_full = rec_embed(rec, crop, do_flip_tta=True)
        sims_full = embs @ q_full
        j_full = int(np.argmax(sims_full))
        sim_full = float(sims_full[j_full])

        crop_masked = make_masked_lower(crop)
        q_mask = rec_embed(rec, crop_masked, do_flip_tta=True)
        sims_mask = embs @ q_mask
        j_mask = int(np.argmax(sims_mask))
        sim_mask = float(sims_mask[j_mask])

        if sim_mask > sim_full:
            j, sim, used = j_mask, sim_mask, "mask"
        else:
            j, sim, used = j_full, sim_full, "full"

        if sim < thresh:
            continue

        reg_token = names[j]
        name, birthday = parse_name_birthday(reg_token)

        color = (0, 200, 0)
        label = f"{name} ({sim:.2f})/{used}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        results.append({
            "name": name,
            "birthday": birthday,
            "mask_type": used,
            "confidence": float(sim)
        })

    return results

# ====== 메인 루프 ======
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 index 또는 권한 확인.")

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        matched = infer_once(frame)

        if matched:
            print("현재 프레임 매칭 결과:", matched)

            # ====== 여기서 서버로 전송(최소 흐름) ======
            try:
                ts = time.time()
                lat, lng = DEFAULT_LAT, DEFAULT_LNG  # TODO: 실제 위치 연동시 교체
                ticket = create_event(ts, lat, lng, PATROL_CAR_NAME)
                print("[ticket]", ticket)

                ok_upload = send_file_tcp(ticket["media_id"], ticket["rel_path"], LOCAL_CLIP_PATH)
                print("[media]", "OK" if ok_upload else "FAIL")

                # 첫 번째 매칭만 전송 (필요하면 전체 반복 전송도 가능)
                hit = matched[0]
                resp = post_missing_result(ticket["event_id"],
                                           hit["name"], hit["birthday"],
                                           hit["mask_type"], hit["confidence"])
                print("[post]", resp)
            except Exception as e:
                print("[ERROR] 전송 실패:", e)
            # =======================================

        else:
            print("매칭 없음")

        cv2.imshow("ArcFace Multi (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
