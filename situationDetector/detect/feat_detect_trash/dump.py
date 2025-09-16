import os
import shutil
import pandas as pd

# --- 설정 ---
# 원본 동영상 파일이 있는 디렉토리
SOURCE_DIRECTORY = '/home/momo/Desktop/video'

# 'dump' 영상 목록이 담긴 CSV 파일 경로
CSV_FILE_PATH = '/home/momo/dev_ws/deeplearning-repo-2/situationDetector/detect/feat_detect_trash/all_trash_dumping_annotations.csv'

# --- 스크립트 시작 ---

def sort_videos_into_folders():
    """
    SOURCE_DIRECTORY의 영상들을 CSV 목록을 기준으로
    'dump'와 'non dump' 폴더로 분류합니다.
    """
    # 1. 분류할 폴더 경로 설정 및 생성
    dump_folder = os.path.join(SOURCE_DIRECTORY, 'dump')
    nondump_folder = os.path.join(SOURCE_DIRECTORY, 'non dump')
    
    os.makedirs(dump_folder, exist_ok=True)
    os.makedirs(nondump_folder, exist_ok=True)
    print("📁 'dump', 'non dump' 폴더를 확인/생성했습니다.")

    # 2. CSV 파일을 읽어 'dump' 파일 목록을 set으로 만들기 (빠른 조회를 위해)
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        dump_filenames = set(df['video_filename'])
        print(f"📖 CSV 파일에서 {len(dump_filenames)}개의 'dump' 영상 목록을 읽었습니다.")
    except FileNotFoundError:
        print(f"⚠️ 경고: CSV 파일 '{CSV_FILE_PATH}'을(를) 찾을 수 없습니다.")
        print("모든 영상을 'non dump'로 분류합니다.")
        dump_filenames = set()

    # 3. 원본 디렉토리의 파일들을 순회하며 분류 작업 수행
    dump_count = 0
    nondump_count = 0
    
    for filename in os.listdir(SOURCE_DIRECTORY):
        source_path = os.path.join(SOURCE_DIRECTORY, filename)
        
        # 파일이고, .mp4 확장자를 가진 경우에만 처리
        if os.path.isfile(source_path) and filename.endswith('.mp4'):
            try:
                # 'dump' 목록에 파일 이름이 있는지 확인
                if filename in dump_filenames:
                    shutil.move(source_path, os.path.join(dump_folder, filename))
                    dump_count += 1
                else:
                    shutil.move(source_path, os.path.join(nondump_folder, filename))
                    nondump_count += 1
            except Exception as e:
                print(f"'{filename}' 파일 이동 중 오류 발생: {e}")

    print("-" * 30)
    print("✅ 작업 완료!")
    print(f"  - {dump_count}개의 파일을 'dump' 폴더로 이동했습니다.")
    print(f"  - {nondump_count}개의 파일을 'non dump' 폴더로 이동했습니다.")


if __name__ == '__main__':
    sort_videos_into_folders()