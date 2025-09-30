# --- 필요한 라이브러리들을 불러옵니다 ---
import time  # 코드 실행 시간, FPS 계산 등 시간 관련 기능을 사용하기 위해 필요합니다.
import torch  # 파이토치(PyTorch) 라이브러리. 딥러닝 모델을 다루기 위해 필수적입니다.
import torchvision  # 파이토치의 일부로, 이미지 관련 처리를 도와줍니다.
import cv2  # OpenCV 라이브러리. 웹캠을 켜고, 비디오 프레임을 다루고, 화면에 그림을 그리는 등 시각적인 부분을 담당합니다.
import numpy as np  # Numpy 라이브러리. 숫자, 특히 배열이나 행렬 계산을 빠르고 쉽게 하기 위해 사용됩니다.
from PIL import Image  # Python Image Library. 이미지를 열고 다루는 데 사용됩니다. OpenCV와 함께 이미지 형식을 변환할 때 유용합니다.
import torchvision.transforms.functional as F  # 이미지의 형태를 모델이 이해할 수 있는 '텐서(Tensor)' 형태로 바꿔주는 등 변환 기능을 제공합니다.
from collections import OrderedDict, deque  # 파이썬의 기본 라이브-러리. OrderedDict는 딕셔너리의 순서를 보장해주고, deque는 리스트보다 빠르고 효율적으로 데이터를 추가/삭제할 수 있어 위치 기록 저장에 사용합니다.

# 우리가 만든 train.py 파일에서 get_model 함수를 불러옵니다.
# 이 함수는 Faster R-CNN 모델의 구조를 정의해줍니다.
from situationDetector.detect.feat_detect_trash.hold.train_human_trash import get_model

# --- 1. 객체 추적기 (Centroid Tracker) 클래스 ---
# 이 클래스는 화면에 나타난 객체들에게 각각 번호표(ID)를 붙여주고,
# 프레임이 바뀌어도 같은 객체를 계속 따라가는 역할을 합니다.
class CentroidTracker:
    # 추적기를 처음 만들 때 실행되는 초기 설정 함수입니다.
    def __init__(self, maxDisappeared=50):
        # 'nextObjectID': 다음에 새로 등록될 객체에게 부여할 ID 번호입니다. 0번부터 시작합니다.
        self.nextObjectID = 0
        # 'objects': 현재 추적 중인 객체들의 ID와 그 중심점 좌표를 저장하는 딕셔너리입니다. 예: {0: (100, 150), 1: (320, 240)}
        self.objects = OrderedDict()
        # 'disappeared': 특정 객체가 몇 프레임 동안 화면에서 사라졌는지 세는 딕셔너리입니다. 예: {0: 2, 1: 0}
        self.disappeared = OrderedDict()
        # 'maxDisappeared': 객체가 화면에서 사라졌을 때, 몇 프레임까지 기다려줄지 정하는 한계값입니다.
        # 이 값을 넘으면 추적 목록에서 삭제합니다. (사람이 잠깐 기둥 뒤에 숨었다 나와도 같은 사람으로 인식하게 해줌)
        self.maxDisappeared = maxDisappeared

    # 새로운 객체가 나타났을 때, 목록에 등록하는 함수입니다.
    def register(self, centroid):
        # 현재 객체의 ID와 중심점 좌표를 'objects'에 저장합니다.
        self.objects[self.nextObjectID] = centroid
        # 방금 등록했으므로, 사라진 횟수를 0으로 초기화합니다.
        self.disappeared[self.nextObjectID] = 0
        # 다음 객체를 위해 ID 번호를 1 증가시킵니다.
        self.nextObjectID += 1

    # 너무 오래 사라진 객체를 추적 목록에서 삭제하는 함수입니다.
    def deregister(self, objectID):
        # 'objects'와 'disappeared' 딕셔너리에서 해당 ID의 정보를 삭제합니다.
        del self.objects[objectID]
        del self.disappeared[objectID]

    # 매 프레임마다 호출되는 가장 중요한 함수입니다.
    # 현재 프레임에서 발견된 객체들(rects)을 기반으로 추적 정보를 업데이트합니다.
    def update(self, rects):
        # 만약 현재 프레임에서 아무 객체도 발견되지 않았다면,
        if len(rects) == 0:
            # 기존에 추적하던 모든 객체들의 'disappeared' 카운트를 1씩 증가시킵니다.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # 만약 어떤 객체가 'maxDisappeared' 한계보다 더 오래 사라졌다면,
                if self.disappeared[objectID] > self.maxDisappeared:
                    # 추적 목록에서 삭제합니다.
                    self.deregister(objectID)
            # 현재 추적 중인 객체 정보를 반환하고 함수를 종료합니다.
            return self.objects

        # 현재 프레임에서 발견된 객체들의 중심점 좌표를 계산하여 저장할 준비를 합니다.
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # 바운딩 박스의 x, y 좌표를 이용해 중심점을 계산합니다.
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # 만약 아직 추적 중인 객체가 하나도 없다면,
        if len(self.objects) == 0:
            # 현재 프레임에서 발견된 모든 객체들을 새로 등록합니다.
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # 이미 추적 중인 객체가 있다면,
        else:
            # 기존에 추적하던 객체들의 ID와 중심점 좌표를 가져옵니다.
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # 기존 객체들과 새로 발견된 객체들 사이의 모든 거리 조합을 계산합니다.
            # (유클리드 거리를 사용하여 누가 누구랑 가장 가까운지 알아보기 위함)
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            for i in range(len(objectCentroids)):
                for j in range(len(inputCentroids)):
                    D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))

            # 각 기존 객체에 대해 가장 가까운 새 객체를 찾고, 그 거리들을 오름차순으로 정렬합니다.
            # 이를 통해 가장 거리가 가까운 쌍부터 먼저 짝을 지어줄 수 있습니다.
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # 가장 가까운 쌍부터 차례대로 짝을 지어줍니다.
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # 짝이 맺어졌으므로, 이 객체는 동일한 객체로 판단합니다.
                objectID = objectIDs[row]
                # 중심점 위치를 새로운 위치로 업데이트하고,
                self.objects[objectID] = inputCentroids[col]
                # 'disappeared' 카운트를 0으로 리셋합니다. (다시 나타났으니까!)
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # 짝을 찾지 못한 기존 객체와 새로운 객체를 처리합니다.
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # 만약 기존 객체의 수가 새 객체보다 많거나 같다면 (짝 못찾은 기존 객체는 사라진 것으로 간주)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # 만약 새 객체의 수가 더 많다면 (짝 못찾은 새 객체는 새로 나타난 것으로 간주)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        # 최종적으로 업데이트된 추적 객체 정보를 반환합니다.
        return self.objects

def run_inference():
    # --- 1. 환경 설정 및 모델 로딩 ---
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    num_classes = 3
    CLASSES = ['BACKGROUND', 'human', 'trash']
    CONF_THRESHOLD = 0.65
    ASSOCIATION_THRESHOLD = 100
    STATIC_THRESHOLD = 15
    DEPARTURE_TIME_WINDOW = 30
    
    print("모델을 불러오는 중입니다...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("모델을 성공적으로 불러왔습니다.")

    # --- 2. 추적기, 웹캠, 상태 변수 초기화 ---
    tracker = CentroidTracker(maxDisappeared=40)
    tracked_info = {} 

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("에러: 비디오 소스를 열 수 없습니다.")
        return None, None # <-- 반환 형식 통일

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # --- 3. 이벤트 녹화 및 줌인 기능 변수 추가 ---
    PRE_EVENT_SECONDS, POST_EVENT_SECONDS, ZOOM_DURATION = 5, 5, 5
    frame_buffer = deque(maxlen=FPS * PRE_EVENT_SECONDS)
    is_recording_event = False
    event_video_writer = None
    event_frames_to_record = 0
    event_video_filename = None
    event_zoom_active = False
    event_zoom_start_time = 0
    last_event_roi = None # <-- 반환할 ROI 변수 초기화

    print("추론을 시작합니다... 'q' 키를 누르면 종료됩니다.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_buffer.append(frame.copy())
        
        # --- (객체 탐지, 추적, 행위 분석 로직은 기존과 동일) ---
        img_tensor = F.to_tensor(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad(): prediction = model(img_tensor)
        detected_rects, detection_details = [], []
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > CONF_THRESHOLD and CLASSES[label.item()] in ['human', 'trash']:
                box_np = box.cpu().numpy().astype(int)
                detected_rects.append(box_np)
                detection_details.append({'box': box_np, 'label': CLASSES[label.item()]})
        tracked_objects = tracker.update(detected_rects)
        current_tracked_ids = set(tracked_objects.keys())
        for obj_id, centroid in tracked_objects.items():
            if obj_id not in tracked_info: tracked_info[obj_id] = {'label': None, 'positions': deque(maxlen=DEPARTURE_TIME_WINDOW), 'associated_with': None, 'last_association_time': 0, 'dumping_alert': False}
            tracked_info[obj_id]['positions'].append(centroid)
            for det in detection_details:
                box_center = ((det['box'][0] + det['box'][2]) // 2, (det['box'][1] + det['box'][3]) // 2)
                if np.linalg.norm(np.array(box_center) - np.array(centroid)) < 20: tracked_info[obj_id]['label'] = det['label']; break
        human_ids = [k for k, v in tracked_info.items() if v['label'] == 'human' and k in current_tracked_ids]
        trash_ids = [k for k, v in tracked_info.items() if v['label'] == 'trash' and k in current_tracked_ids]
        for h_id in human_ids:
            human_pos, is_associated = tracked_info[h_id]['positions'][-1], False
            for t_id in trash_ids:
                trash_pos = tracked_info[t_id]['positions'][-1]
                if np.linalg.norm(np.array(human_pos) - np.array(trash_pos)) < ASSOCIATION_THRESHOLD:
                    tracked_info[h_id]['associated_with'], tracked_info[h_id]['last_association_time'], is_associated = t_id, time.time(), True
                    break
            if not is_associated and tracked_info[h_id]['associated_with'] is not None:
                associated_trash_id = tracked_info[h_id]['associated_with']
                if associated_trash_id in tracked_info and tracked_info[associated_trash_id]['label'] == 'trash':
                    trash_positions = tracked_info[associated_trash_id]['positions']
                    if len(trash_positions) > 10:
                        movement = np.std(trash_positions, axis=0).sum()
                        if movement < STATIC_THRESHOLD and time.time() - tracked_info[h_id]['last_association_time'] > 1.0:
                            if not is_recording_event:
                                print(f"--- 무단 투기 행위 감지! 녹화를 시작합니다. ---")
                                event_zoom_active, event_zoom_start_time = True, time.time()
                                h_box = next((d['box'] for d in detection_details if tracked_info.get(h_id, {}).get('label') == d['label']), None)
                                t_box = next((d['box'] for d in detection_details if tracked_info.get(associated_trash_id, {}).get('label') == d['label']), None)
                                if h_box is not None and t_box is not None:
                                    x1, y1 = min(h_box[0], t_box[0]), min(h_box[1], t_box[1])
                                    x2, y2 = max(h_box[2], t_box[2]), max(h_box[3], t_box[3])
                                    last_event_roi = [max(0, x1-50), max(0, y1-50), min(W, x2+50), min(H, y2+50)]
                                is_recording_event = True
                                event_frames_to_record = FPS * POST_EVENT_SECONDS
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                event_video_filename = f"event_{timestamp}.mp4"
                                event_video_writer = cv2.VideoWriter(event_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
                                for buffered_frame in frame_buffer: event_video_writer.write(buffered_frame)
                tracked_info[h_id]['associated_with'] = None

        if is_recording_event:
            event_video_writer.write(frame)
            event_frames_to_record -= 1
            if event_frames_to_record <= 0:
                print(f"--- 녹화 완료: {event_video_filename} ---")
                event_video_writer.release()
                is_recording_event = False
                break # <-- 녹화 완료 시 루프 탈출

        display_frame = frame
        if event_zoom_active:
            if time.time() - event_zoom_start_time < ZOOM_DURATION:
                if last_event_roi:
                    x1, y1, x2, y2 = last_event_roi
                    zoomed_area = frame[y1:y2, x1:x2]
                    if zoomed_area.size > 0:
                        display_frame = cv2.resize(zoomed_area, (W, H))
                        cv2.putText(display_frame, "EVENT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else: event_zoom_active = False

        cv2.imshow('Illegal Dumping Action Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("추론을 중지합니다.")
    cap.release()
    if event_video_writer:
        event_video_writer.release()
    cv2.destroyAllWindows()
    
    # --- 7. 최종 결과 반환 ---
    # 영상 파일 경로와 이벤트 발생 시의 ROI 좌표를 함께 반환
    return event_video_filename, last_event_roi

if __name__ == "__main__":
    # run_inference 함수가 이제 두 개의 값을 반환합니다.
    saved_video_path, event_roi = run_inference()
    
    # 두 값 모두 정상적으로 반환되었는지 확인합니다.
    if saved_video_path and event_roi:
        print("\n--- 최종 결과 ---")
        print(f"이벤트 영상이 다음 경로에 저장되었습니다: {os.path.abspath(saved_video_path)}")
        print(f"이벤트 발생 영역(ROI) 좌표: {event_roi}")
    else:
        print("\n이벤트가 발생하지 않았거나, 사용자에 의해 프로그램이 종료되었습니다.")