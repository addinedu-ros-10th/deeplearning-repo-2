import os
import pandas as pd
import xml.etree.ElementTree as ET

# --- 설정 ---
# 영상과 XML 파일이 함께 있는 디렉토리 경로
DATA_DIRECTORY = '/home/momo/Desktop/video'

# 업데이트할 CSV 파일 경로
CSV_FILE_PATH = '/home/momo/dev_ws/deeplearning-repo-2/situationDetector/detect/feat_detect_trash/all_trash_dumping_annotations.csv'

# --- 스크립트 시작 ---

def get_frames_from_xml(xml_path):
    """
    XML 파일 경로를 받아 start_frame과 end_frame을 추출합니다.
    action 태그가 없거나 파일이 없으면 None을 반환합니다.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        start_element = root.find('.//action/frame/start')
        end_element = root.find('.//action/frame/end')
        
        if start_element is not None and end_element is not None:
            start_frame = start_element.text
            end_frame = end_element.text
            # end_frame이 비어있는 경우도 처리
            if start_frame and end_frame:
                return int(start_frame), int(end_frame)
    except (FileNotFoundError, ET.ParseError):
        # 파일이 없거나 XML 파싱 오류가 나면 무시
        return None, None
        
    return None, None


def main():
    """CSV 파일을 읽고 XML의 정보로 업데이트합니다."""
    try:
        # 1. CSV 파일을 pandas DataFrame으로 읽기
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"오류: CSV 파일 '{CSV_FILE_PATH}'을(를) 찾을 수 없습니다.")
        return

    print("CSV 파일 업데이트를 시작합니다...")
    updated_count = 0

    # 2. CSV의 각 행을 순회하며 작업 수행
    for index, row in df.iterrows():
        video_filename = row['video_filename']
        
        # XML 파일 이름 및 경로 생성
        xml_filename = os.path.splitext(video_filename)[0] + '.xml'
        xml_path = os.path.join(DATA_DIRECTORY, xml_filename)
        
        # XML에서 프레임 정보 추출
        start, end = get_frames_from_xml(xml_path)
        
        # 유효한 프레임 정보를 가져왔고, 현재 값이 0인 경우에만 업데이트
        if start is not None and end is not None:
            if df.loc[index, 'start_frame'] != start or df.loc[index, 'end_frame'] != end:
                df.loc[index, 'start_frame'] = start
                df.loc[index, 'end_frame'] = end
                print(f"  - {video_filename}: start={start}, end={end} (업데이트됨)")
                updated_count += 1
        else:
            print(f"  - {video_filename}: XML에서 Action 정보를 찾을 수 없음 (통과)")


    # 3. 변경된 내용을 다시 CSV 파일에 저장
    df.to_csv(CSV_FILE_PATH, index=False)
    
    print("-" * 30)
    print(f"✅ 완료! 총 {updated_count}개의 행이 업데이트되었습니다.")


if __name__ == '__main__':
    main()