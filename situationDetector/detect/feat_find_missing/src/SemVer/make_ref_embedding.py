# 참조 임베딩 만들기(한 번만)

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# 1) ArcFace 로드
use_gpu = torch.cuda.is_available()
app = FaceAnalysis(name="buffalo_l")  # 검증 잘 되는 사전학습 세트
app.prepare(ctx_id=0 if use_gpu else -1)

# 2) 실종자 사진 읽기
img = cv2.imread("/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_person.jpg")
if img is None:
    raise FileNotFoundError("missing_person.jpg 파일을 같은 폴더에 두세요.")

# 3) 얼굴 검출 & 임베딩
faces = app.get(img)
if len(faces) == 0:
    raise ValueError("실종자 얼굴을 찾지 못했어요. 더 선명한 사진을 써보세요.")
# 가장 큰 얼굴 1개 선택
faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
ref_emb = faces[0].embedding
# L2 정규화(코사인 유사도 계산 안정화)
ref_emb = ref_emb / np.linalg.norm(ref_emb)

# 4) 임베딩 저장
np.save("/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/feat_find_missing/src/missing_embedding.npy", ref_emb)
print("저장 완료: missing_embedding.npy")
