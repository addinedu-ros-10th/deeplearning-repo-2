# make_ref_embeddings_multi.py
# 단일 이미지든 사람별 폴더든 OK. 강건 증강 + flip-TTA로 1장만 있어도 템플릿을 확장해 평균 임베딩 저장.

import os, glob, cv2, numpy as np, torch, random
from insightface.app import FaceAnalysis

# ====== 설정 ======
MODEL_NAME   = "buffalo_s"   # 필요시 "buffalo_m"로 변경하여 비교
DET_SIZE     = (320, 320)
IMG_DIR      = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/data"
EMB_PATH     = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

AUG_PER_IMG  = 12            # 원본 1장당 증강 샘플 수(8~24 권장, 현장에 맞춰 조정)

# ====== 유틸 ======
def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

def aug_color_blur(img):
    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[...,1] = np.clip(hsv[...,1] + np.random.randint(-20, 21), 0, 255)  # 채도
    hsv[...,2] = np.clip(hsv[...,2] + np.random.randint(-25, 31), 0, 255)  # 밝기
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.5:
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out, (k,k), 0)
    return out

def aug_occlude_lower(img):
    """하관 가림(마스크 근사)"""
    h, w, _ = img.shape
    y1 = int(h * random.uniform(0.55, 0.68))
    y2 = int(h * random.uniform(0.78, 0.90))
    x1 = int(w * random.uniform(0.05, 0.15))
    x2 = int(w * random.uniform(0.85, 0.95))
    out = img.copy()
    fill = int(np.mean(out[max(0,y1):min(h,y2), max(0,x1):min(w,x2)])) if y2>y1 and x2>x1 else 128
    cv2.rectangle(out, (x1, y1), (x2, y2), (fill, fill, fill), -1)
    return out

def aug_affine(img):
    """소규모 회전/스케일/시프트"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), random.uniform(-7,7), random.uniform(0.95,1.05))
    M[:,2] += np.random.uniform(-0.02*w, 0.02*w, size=2)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def tta_embed(app, img):
    faces = app.get(img)
    if not faces: return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    e1 = l2n(faces[0].embedding)
    flip = cv2.flip(img, 1)
    faces_f = app.get(flip)
    if faces_f:
        faces_f.sort(key=lambda g: (g.bbox[2]-g.bbox[0])*(g.bbox[3]-g.bbox[1]), reverse=True)
        e2 = l2n(faces_f[0].embedding)
        return l2n((e1 + e2)/2)
    return e1

# ====== 앱 준비 ======
app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)

# ====== 대상 수집: 폴더 모드 or 파일 나열 모드 지원 ======
entries = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
is_dir_mode = any(os.path.isdir(p) for p in entries)

targets = []
if is_dir_mode:
    # 사람별 폴더: data/이름/이미지들
    for d in entries:
        if not os.path.isdir(d): continue
        imgs = [p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".jpg",".jpeg",".png"))]
        if imgs: targets.append( (os.path.basename(d), imgs) )
else:
    # 이미지 나열: 파일명을 이름으로 사용
    imgs = [p for p in entries if p.lower().endswith((".jpg",".jpeg",".png"))]
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        targets.append( (name, [p]) )

# ====== 임베딩 생성 ======
person_embs, person_names = [], []

for name, img_paths in targets:
    embs = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None: continue
        # 원본
        e = tta_embed(app, img)
        if e is not None: embs.append(e)
        # 증강 다수
        for _ in range(AUG_PER_IMG):
            cand = img.copy()
            if random.random() < 0.9: cand = aug_color_blur(cand)
            if random.random() < 0.6: cand = aug_occlude_lower(cand)
            if random.random() < 0.6: cand = aug_affine(cand)
            e = tta_embed(app, cand)
            if e is not None: embs.append(e)

    embs = [e for e in embs if e is not None]
    if not embs: 
        print(f"[WARN] {name}: 얼굴을 찾지 못해 스킵.")
        continue

    ref = l2n(np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32))  # 평균 후 L2 정규화
    person_embs.append(ref)
    person_names.append(name)

if person_embs:
    person_embs = np.stack(person_embs).astype(np.float32)
    np.save(EMB_PATH, person_embs)
    np.save(NAMES_PATH, np.asarray(person_names))
    print(f"[OK] Saved {len(person_names)} persons →")
    print("     EMB:", EMB_PATH)
    print("     NMS:", NAMES_PATH)
else:
    print("[ERR] No embeddings saved.")
