import os
import pandas as pd
import xml.etree.ElementTree as ET

# XML 파일들이 있는 디렉토리 경로
xml_dir = './data' # 현재 디렉토리라고 가정

# 추출한 정보를 저장할 리스트
action_data = []

# 디렉토리 내의 모든 파일을 순회
print(f"'{xml_dir}' 디렉토리에서 XML 파일을 스캔합니다...")
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        xml_path = os.path.join(xml_dir, filename)
        
        try:
            # XML 파일 파싱
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 1. 영상 파일명 추출
            video_filename = root.find('filename').text
            
            # 2. 행동(action)의 시작 및 종료 프레임 추출
            # <object> -> <action> -> <frame> 경로를 따라 검색
            action_frame = root.find('object/action/frame')
            if action_frame is not None:
                start_frame = int(action_frame.find('start').text)
                end_frame = int(action_frame.find('end').text)
                
                # 추출한 정보를 리스트에 추가
                action_data.append({
                    'video_filename': video_filename,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                })
            else:
                print(f"경고: '{filename}' 파일에서 <action> 정보를 찾을 수 없습니다.")

        except Exception as e:
            print(f"오류: '{filename}' 파일 처리 중 문제 발생 - {e}")

# 리스트를 Pandas DataFrame으로 변환
df = pd.DataFrame(action_data)

# 결과 출력
print("\n[추출 완료된 데이터]")
print(df)

# (선택) 결과를 CSV 파일로 저장하여 나중에 쉽게 불러올 수 있도록 함
df.to_csv('trash_dumping_annotations.csv', index=False)
print("\n'trash_dumping_annotations.csv' 파일로 저장 완료!")