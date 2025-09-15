import os
import cv2
import pandas as pd
import random
import numpy as np

# --- 설정값 ---
ANNOTATIONS_FILE = 'trash_dumping_annotations.csv'
VIDEO_DIR = './data'
OUTPUT_DIR = './clips'
CLIP_LENGTH = 16  # 클립 당 프레임 수
NUM_NEGATIVE_SAMPLES_PER_VIDEO = 10 # 영상 하나당 추출할 Negative 클립 수

# --- 출력 폴더 생성 ---
DUMPING_DIR = os.path.join(OUTPUT_DIR, 'dumping')
NON_DUMPING_DIR = os.path.join(OUTPUT_DIR, 'non_dumping')
os.makedirs(DUMPING_DIR, exist_ok=True)
os.makedirs(NON_DUMPING_DIR, exist_ok=True)

# --- 어노테이션 파일 로드 ---
df = pd.read_csv(ANNOTATIONS_FILE)

# --- 메인 루프: 각 영상에 대해 클립 추출 ---
for index, row in df.iterrows():
    video_filename = row['video_filename']
    start_frame = row['start_frame']
    end_frame = row['end_frame']
    
    video_path = os.path.join(VIDEO_DIR, video_filename)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: '{video_path}' 영상을 열 수 없습니다.")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[{video_filename}] 처리 중 (총 {total_frames} 프레임)")

    # 1. Positive 클립 (투기 구간) 추출
    clip_count = 0
    for frame_start in range(start_frame, end_frame - CLIP_LENGTH + 1, CLIP_LENGTH // 2): # 8프레임씩 겹치며 추출
        clip_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        for i in range(CLIP_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames.append(frame)
        
        if len(clip_frames) == CLIP_LENGTH:
            clip_name = f"{os.path.splitext(video_filename)[0]}_positive_{clip_count}"
            clip_folder = os.path.join(DUMPING_DIR, clip_name)
            os.makedirs(clip_folder, exist_ok=True)
            for i, frame in enumerate(clip_frames):
                cv2.imwrite(os.path.join(clip_folder, f"frame_{i:04d}.jpg"), frame)
            clip_count += 1
    print(f"  -> Positive 클립 {clip_count}개 생성 완료.")

    # 2. Negative 클립 (정상 구간) 추출
    negative_ranges = [(0, start_frame - CLIP_LENGTH), (end_frame, total_frames - CLIP_LENGTH)]
    valid_negative_starts = []
    for r_start, r_end in negative_ranges:
        if r_end > r_start:
            valid_negative_starts.extend(range(r_start, r_end))
    
    if not valid_negative_starts:
        print(f"  -> Negative 클립을 추출할 유효한 구간이 없습니다.")
        cap.release()
        continue

    selected_starts = random.sample(valid_negative_starts, min(NUM_NEGATIVE_SAMPLES_PER_VIDEO, len(valid_negative_starts)))
    
    clip_count = 0
    for frame_start in selected_starts:
        clip_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        for i in range(CLIP_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames.append(frame)

        if len(clip_frames) == CLIP_LENGTH:
            clip_name = f"{os.path.splitext(video_filename)[0]}_negative_{clip_count}"
            clip_folder = os.path.join(NON_DUMPING_DIR, clip_name)
            os.makedirs(clip_folder, exist_ok=True)
            for i, frame in enumerate(clip_frames):
                cv2.imwrite(os.path.join(clip_folder, f"frame_{i:04d}.jpg"), frame)
            clip_count += 1
    print(f"  -> Negative 클립 {clip_count}개 생성 완료.")

    cap.release()

print("\n--- 모든 영상에 대한 클립 추출 작업 완료! ---")