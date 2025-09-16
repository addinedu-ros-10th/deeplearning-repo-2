import os
import cv2
import random
import shutil
from sklearn.model_selection import train_test_split

# 경로 설정
LITTERING_DIR = "/media/momo/MOMO/littering_FHD/"
DATE_DIR = "/media/momo/MOMO/"
OUTPUT_DIR = "/media/momo/MOMO/dataset/"
FRAME_INTERVAL = 30  # 초당 1프레임 추출 (30fps 기준)

# 디렉토리 생성
for split in ["train", "test"]:
    for label in ["dump", "non-dump"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# 동영상 파일 목록 수집
littering_videos = [os.path.join(LITTERING_DIR, f) for f in os.listdir(LITTERING_DIR) if f.endswith((".mp4", ".avi"))]
date_videos = [os.path.join(DATE_DIR, f) for f in os.listdir(DATE_DIR) if f.startswith("20250915_") and f.endswith((".mp4", ".avi"))]
all_videos = littering_videos + date_videos

print(f"littering_videos: {len(littering_videos)}")
print(f"date_videos: {len(date_videos)}")
print(f"all_videos: {len(all_videos)}")

if not all_videos:
    print("No videos found. Check directory paths and file formats.")
    exit(1)

# 라벨링 (임의로 예시, 실제로는 동영상 내용에 따라 수동 라벨링 필요)
labels = [random.choice(["dump", "non-dump"]) for _ in range(len(all_videos))]

# TEST 데이터셋에 20250915_ 동영상 5개 포함
test_videos = date_videos  # 20250915_ 5개
remaining_videos = [v for v in littering_videos if v not in test_videos]
test_videos += random.sample(remaining_videos, 15)  # 추가 15개
train_videos = [v for v in all_videos if v not in test_videos]

print(f"train_videos: {len(train_videos)}")
print(f"test_videos: {len(test_videos)}")

# 비디오에서 이미지 클립 추출
def extract_frames(video_path, output_dir, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {video_path}, FPS: {fps}")
    frame_count = 0
    success = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(fps * FRAME_INTERVAL) == 0:  # 동적 FPS 반영
            frame_filename = f"{os.path.basename(video_path).split('.')[0]}_frame{frame_count}.jpg"
            output_path = os.path.join(output_dir, label, frame_filename)
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            success = True
        frame_count += 1
    cap.release()
    return success

# TRAIN 데이터셋 처리
for i, video in enumerate(train_videos):
    label = random.choice(["dump", "non-dump"])  # 실제 라벨링 필요
    print(f"Processing train video {i+1}/{len(train_videos)}: {video}")
    if not extract_frames(video, os.path.join(OUTPUT_DIR, "train"), label):
        print(f"Failed to process: {video}")

# TEST 데이터셋 처리
for i, video in enumerate(test_videos):
    label = random.choice(["dump", "non-dump"])  # 실제 라벨링 필요
    print(f"Processing test video {i+1}/{len(test_videos)}: {video}")
    if not extract_frames(video, os.path.join(OUTPUT_DIR, "test"), label):
        print(f"Failed to process: {video}")

print(f"TRAIN: {len(train_videos)} videos, TEST: {len(test_videos)} videos")