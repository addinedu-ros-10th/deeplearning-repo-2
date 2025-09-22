import os
import shutil
import random

# --- 설정 ---
# 24개의 영상이 들어있는 폴더 (이전 작업으로 dump 폴더에 모두 있음)
SOURCE_DIRECTORY = '/home/momo/Desktop/video/dump'

# 데이터를 나눌 상위 폴더
BASE_DIRECTORY = '/home/momo/Desktop/video'

# 학습용 데이터와 테스트용 데이터의 비율 (80:20)
TRAIN_RATIO = 0.8

# --- 스크립트 시작 ---

def split_videos_for_learning():
    """SOURCE_DIRECTORY의 영상들을 train/test 폴더로 무작위 분할합니다."""
    
    # 1. train, test 폴더 경로 설정 및 생성
    train_folder = os.path.join(BASE_DIRECTORY, 'train')
    test_folder = os.path.join(BASE_DIRECTORY, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    print(f"📁 '{train_folder}'와 '{test_folder}' 폴더를 확인/생성했습니다.")

    # 2. 원본 디렉토리에서 .mp4 파일 목록 가져오기
    try:
        video_files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('.mp4')]
        if not video_files:
            print(f"⚠️ 경고: '{SOURCE_DIRECTORY}' 폴더에 영상 파일이 없습니다.")
            return
    except FileNotFoundError:
        print(f"⛔ 오류: '{SOURCE_DIRECTORY}' 폴더를 찾을 수 없습니다.")
        return

    # 3. 파일 목록을 무작위로 섞기
    random.shuffle(video_files)
    
    # 4. 학습용과 테스트용으로 나눌 기준점 계산
    total_files = len(video_files)
    split_point = int(total_files * TRAIN_RATIO)
    
    train_files = video_files[:split_point]
    test_files = video_files[split_point:]

    # 5. 파일들을 각각의 폴더로 이동
    def move_files(files, destination_folder):
        count = 0
        for filename in files:
            try:
                shutil.move(os.path.join(SOURCE_DIRECTORY, filename), 
                            os.path.join(destination_folder, filename))
                count += 1
            except Exception as e:
                print(f"'{filename}' 파일 이동 중 오류 발생: {e}")
        return count

    train_count = move_files(train_files, train_folder)
    test_count = move_files(test_files, test_folder)

    print("-" * 30)
    print("✅ 작업 완료!")
    print(f"  - 총 {total_files}개의 영상을 분할했습니다.")
    print(f"  - {train_count}개의 파일을 'train' 폴더로 이동했습니다.")
    print(f"  - {test_count}개의 파일을 'test' 폴더로 이동했습니다.")

if __name__ == '__main__':
    split_videos_for_learning()