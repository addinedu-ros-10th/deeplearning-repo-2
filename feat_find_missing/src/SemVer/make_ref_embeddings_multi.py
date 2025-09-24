# make_ref_embeddings_multi.py
import os, glob
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

MODEL_NAME   = "buffalo_s"
DET_SIZE     = (320, 320)
IMG_DIR      = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/data"
EMB_PATH     = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/embeddings.npy"
NAMES_PATH   = "/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/names.npy"

def l2n(x, eps=1e-9):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x)
    return x if n < eps else x / n

app = FaceAnalysis(name=MODEL_NAME)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=DET_SIZE)

embs, names = [], []
for path in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
    if not path.lower().endswith((".jpg", ".jpeg", ".png")): continue
    img = cv2.imread(path)
    if img is None: continue
    faces = app.get(img)
    if not faces: continue
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    emb = l2n(faces[0].embedding)
    embs.append(emb)
    names.append(os.path.splitext(os.path.basename(path))[0])

if embs:
    embs = np.stack(embs).astype(np.float32)   # (N, D), 이미 L2 정규화됨
    np.save(EMB_PATH, embs)
    np.save(NAMES_PATH, np.asarray(names))
