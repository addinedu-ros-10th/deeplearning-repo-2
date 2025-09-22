import cv2
import os
import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

# 원하는 행동 목록을 이곳에 추가/수정하세요
ACTION_LIST = ['drop', 'carry', 'walk', 'run', 'stand']

# --- 전역 변수 및 데이터 저장소 ---
objects_data = {}
active_object_id = None
mouse_click_pos = None

def prettify_xml(elem):
    """XML을 보기 좋게 정렬하여 문자열로 반환합니다."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def mouse_callback(event, x, y, flags, param):
    """마우스 콜백 함수: 더블클릭 시 좌표 저장"""
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_click_pos = (x, y)
        print(f"마우스 위치 저장: ({x}, {y})")

def format_timedelta_to_hhmmss(seconds_float):
    """초(float)를 'HH:MM:SS.s' 형식의 문자열로 변환합니다."""
    total_seconds = int(seconds_float)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    tenths = int((seconds_float % 1) * 10)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{tenths}"

def main():
    global active_object_id, mouse_click_pos

    # 1. 사전 정보 입력 받기
    video_path = input("라벨링할 동영상 파일의 경로를 입력하세요: ")
    if not os.path.exists(video_path):
        print(f"오류: '{video_path}' 파일을 찾을 수 없습니다."); return

    print("\n--- 영상 기본 정보(Header)를 입력하세요 ---")
    header_info = {
        'location': input("Location (e.g., PLACE01): "),
        'season': input("Season (e.g., SPRING): "),
        'weather': input("Weather (e.g., SUNNY): "),
        'time': input("Time (e.g., DAY): "),
        'population': input("Population (e.g., 2): "),
        'character': input("Character (e.g., M40,F40): "),
    }

    # 2. 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: '{video_path}' 영상을 열 수 없습니다."); return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xml_filename = os.path.splitext(video_path)[0] + ".xml"
    video_basename = os.path.basename(video_path)

    main_event = {}
    is_paused = True
    
    window_name = f'Annotation Tool for {video_basename}'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- 라벨링을 시작합니다 ---")
    print("Space:재생/일시정지, s/e:이벤트 시작/종료, 1-9:객체선택, a/d:행동 시작/종료")
    print("마우스더블클릭:위치지정, .:다음프레임, ,:이전프레임, q:저장&종료")

    ret, frame = cap.read()

    while cap.isOpened() and ret:
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time_sec = (current_frame_num - 1) / fps if fps > 0 else 0

        if mouse_click_pos and active_object_id:
            objects_data[active_object_id]['position'] = { 'keyframe': current_frame_num - 1, 'x': mouse_click_pos[0], 'y': mouse_click_pos[1] }
            print(f"'{active_object_id}'의 위치를 프레임 {current_frame_num - 1}에 저장했습니다.")
            mouse_click_pos = None

        display_frame = frame.copy()
        info_text = f"Frame: {current_frame_num-1}/{total_frames} Time: {current_time_sec:.2f}s"
        if is_paused: info_text += " (Paused)"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if active_object_id:
            cv2.putText(display_frame, f"Active Object: {active_object_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(30 if not is_paused else 0) & 0xFF
        
        if key == ord('q'): break
        elif key == ord('.'):
            is_paused = True
            ret, frame = cap.read()
        elif key == ord(','):
            is_paused = True
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            prev_frame = max(0, current_frame - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
            ret, frame = cap.read()
        elif key == ord(' '): 
            is_paused = not is_paused
        elif key >= ord('1') and key <= ord('9'):
            obj_num = key - ord('0')
            active_object_id = f'person_{obj_num}'
            if active_object_id not in objects_data: objects_data[active_object_id] = {'position': {}, 'action': {}}
            print(f"'{active_object_id}' 선택됨.")
        elif key == ord('s'):
            main_event['starttime'] = current_time_sec
            print(f"메인 이벤트 시작: {current_time_sec:.2f}s")
        elif key == ord('e'):
            main_event['endtime'] = current_time_sec
            print(f"메인 이벤트 종료: {current_time_sec:.2f}s")
        
        # ▼▼▼▼▼ [수정된 부분] 터미널에서 바로 입력받도록 변경 ▼▼▼▼▼
        elif key == ord('a') and active_object_id:
            is_paused = True # 행동 선택 중에는 영상이 멈추도록 보장
            print(f"\n--- '{active_object_id}'의 행동을 선택하세요 ---")
            for i, action in enumerate(ACTION_LIST, 1):
                print(f"  {i}: {action}")
            
            choice = input("원하는 행동의 번호를 입력하고 Enter를 누르세요: ")
            
            try:
                action_num = int(choice)
                if 1 <= action_num <= len(ACTION_LIST):
                    action_name = ACTION_LIST[action_num - 1]
                    objects_data[active_object_id]['action'] = {
                        'actionname': action_name,
                        'start': current_frame_num - 1
                    }
                    print(f"--> '{active_object_id}'의 행동 '{action_name}' 시작 프레임: {current_frame_num - 1}")
                else:
                    print("--> 잘못된 번호입니다. 행동이 지정되지 않았습니다.")
            except ValueError:
                print("--> 숫자를 입력해야 합니다. 행동이 지정되지 않았습니다.")
        # ▲▲▲▲▲ [수정된 부분] 여기까지 ▲▲▲▲▲
        
        elif key == ord('d') and active_object_id:
            if 'start' in objects_data.get(active_object_id, {}).get('action', {}):
                objects_data[active_object_id]['action']['end'] = current_frame_num - 1
                print(f"'{active_object_id}'의 행동 종료 프레임: {current_frame_num - 1}")
        
        if not is_paused:
            ret, frame = cap.read()
            if not ret: is_paused = True

    cap.release()
    cv2.destroyAllWindows()

    # 3. XML 파일 생성 (이하 코드 동일)
    print("\nXML 파일을 생성합니다...")
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'dump'
    ET.SubElement(annotation, 'filename').text = video_basename
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'NIA2019 Database v1'
    ET.SubElement(source, 'annotation').text = 'NIA2019'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    header = ET.SubElement(annotation, 'header')
    total_duration_sec = total_frames / fps if fps > 0 else 0
    ET.SubElement(header, 'duration').text = format_timedelta_to_hhmmss(total_duration_sec)
    ET.SubElement(header, 'fps').text = str(int(fps))
    ET.SubElement(header, 'frames').text = str(total_frames)
    ET.SubElement(header, 'inout').text = 'OUT'
    for k, v in header_info.items(): ET.SubElement(header, k).text = v
    if 'starttime' in main_event and 'endtime' in main_event:
        event = ET.SubElement(annotation, 'event')
        ET.SubElement(event, 'eventname').text = 'dump'
        event_start = main_event['starttime']
        event_duration = main_event['endtime'] - event_start
        ET.SubElement(event, 'starttime').text = format_timedelta_to_hhmmss(event_start)
        ET.SubElement(event, 'duration').text = format_timedelta_to_hhmmss(event_duration)
    for obj_name, obj_data in objects_data.items():
        has_position = 'position' in obj_data and obj_data['position']
        has_action = 'action' in obj_data and 'start' in obj_data['action']
        if has_position or has_action:
            object_elem = ET.SubElement(annotation, 'object')
            ET.SubElement(object_elem, 'objectname').text = obj_name
            if has_position:
                pos = ET.SubElement(object_elem, 'position')
                ET.SubElement(pos, 'keyframe').text = str(obj_data['position']['keyframe'])
                keypoint = ET.SubElement(pos, 'keypoint')
                ET.SubElement(keypoint, 'x').text = str(obj_data['position']['x'])
                ET.SubElement(keypoint, 'y').text = str(obj_data['position']['y'])
            if has_action:
                act = ET.SubElement(object_elem, 'action')
                ET.SubElement(act, 'actionname').text = obj_data['action'].get('actionname', 'unknown')
                frame_elem = ET.SubElement(act, 'frame')
                ET.SubElement(frame_elem, 'start').text = str(obj_data['action']['start'])
                ET.SubElement(frame_elem, 'end').text = str(obj_data['action'].get('end', ''))
    with open(xml_filename, "w", encoding="utf-8") as f:
        f.write(prettify_xml(annotation))
    print(f"\n성공: 라벨링 정보({xml_filename})가 저장되었습니다.")

if __name__ == '__main__':
    main()