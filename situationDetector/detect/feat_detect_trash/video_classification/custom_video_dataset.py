# custom_video_dataset.py
import torch
from torch.utils.data import Dataset
import json
import os
import cv2
from PIL import Image

class CustomActionDataset(Dataset):
    def __init__(self, annotation_file, clip_len=16, transform=None):
        self.annotation_file = annotation_file
        self.clip_len = clip_len
        self.transform = transform
        
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 'events' 목록이 실제 학습 데이터가 됩니다.
        self.samples = self.annotations['events']
        
        # 클래스 레이블을 숫자로 변환 (예: normal -> 0, littering -> 1)
        # 실제 데이터셋의 클래스 정의에 맞게 수정해야 합니다.
        self.class_map = {"정상행위": 0, "불법행위": 1} 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_info = self.samples[idx]
        
        # events 정보에서 비디오 파일 경로와 프레임 정보 추출
        # (이 부분은 info 섹션의 파일 이름과 연결해야 할 수 있습니다)
        # 예시: video_path = os.path.join(VIDEO_ROOT_DIR, self.annotations['info']['file_name'])
        video_path = "path/to/videos/" + self.annotations['info']['file_name'] # <-- 실제 비디오 폴더 경로 필요
        
        start_frame = event_info['ev_start_frame']
        end_frame = event_info['ev_end_frame']
        label_str = event_info['class_description'] # "불법행위" 또는 "정상행위"
        label = self.class_map[label_str]
        
        # 비디오 클립에서 프레임들을 샘플링하는 로직 (HMDB51Dataset과 유사)
        frames = self._load_frames(video_path, start_frame, end_frame)
        
        # 데이터 변환 적용
        if self.transform:
            video_tensor = self.transform(frames)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def _load_frames(self, video_path, start_frame, end_frame):
        # cv2.VideoCapture를 사용하여 비디오를 열고
        # start_frame과 end_frame 사이에서 self.clip_len 만큼 프레임을 균일하게 추출하는 로직 구현
        # HMDB51Dataset의 __getitem__ 내부 로직을 참고하여 작성
        frames = []
        # ... 로직 구현 ...
        return frames