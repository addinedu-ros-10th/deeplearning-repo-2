import sys
import cv2
import os
from datetime import datetime
from collections import deque
import random
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QTextBrowser, QDialog, QGridLayout, QLineEdit, QPushButton,
                               QComboBox, QMessageBox, QListWidget, QSizePolicy, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QFont

# --- 유틸리티 및 AI 모델 클래스 ---

# 영상 내 행동 분류를 위한 AI 모델 정의
class ActionClassifier(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2):
        super(ActionClassifier, self).__init__()
        # 사전 훈련된 EfficientNet-B0 모델을 CNN 특징 추출기로 사용
        self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.cnn.classifier[1].in_features
        # CNN의 마지막 분류 레이어는 제거 (특징 벡터만 필요)
        self.cnn.classifier = nn.Identity()
        # CNN으로 추출된 프레임별 특징들의 시간적 패턴을 학습하기 위한 LSTM
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        # LSTM의 출력을 최종 클래스로 분류하는 분류기
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 입력 텐서 형태: (배치, 채널, 시간(프레임 수), 높이, 너비)
        b, c, t, h, w = x.shape
        # LSTM 처리를 위해 (배치, 시간, 채널, 높이, 너비) 형태로 변경
        x = x.permute(0, 2, 1, 3, 4)
        # 각 프레임을 독립적으로 CNN에 통과시키기 위해 (배치*시간, 채널, 높이, 너비)로 reshape
        x = x.reshape(b * t, c, h, w)
        features = self.cnn(x)
        # CNN 출력을 다시 (배치, 시간, 특징) 형태로 복원
        features = features.view(b, t, -1)
        # LSTM으로 특징 시퀀스 처리
        lstm_out, _ = self.lstm(features)
        # 마지막 시간 스텝의 LSTM 출력을 최종 분류에 사용
        out = self.classifier(lstm_out[:, -1, :])
        return out

# 비디오 화면의 가로 세로 비율(16:9)을 유지해주는 커스텀 위젯
class AspectRatioWidget(QWidget):
    def __init__(self, parent=None, aspect_ratio=16.0/9.0):
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.video_label = QLabel(self)  # 비디오 프레임이 표시될 라벨
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: rgb(40, 40, 40); color: white;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    # 위젯 크기가 변경될 때마다 호출되는 이벤트 핸들러
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        # 위젯의 비율에 따라 비디오 라벨의 크기와 위치를 조정하여 16:9 비율 유지
        if w / h > self.aspect_ratio:
            new_w = int(h * self.aspect_ratio)
            self.video_label.setGeometry((w - new_w) // 2, 0, new_w, h)
        else:
            new_h = int(w / self.aspect_ratio)
            self.video_label.setGeometry(0, (h - new_h) // 2, w, new_h)

# 롤링 녹화(오래된 파일을 자동 삭제하며 계속 녹화)를 담당하는 클래스
class RollingRecorder:
    def __init__(self, output_dir="recordings", chunk_seconds=10, fps=30, frame_size=(1280, 720)):
        self.output_dir = output_dir  # 녹화 파일 저장 디렉토리
        self.chunk_seconds = chunk_seconds  # 파일 분할 시간 (초)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 비디오 코덱
        self.fps = fps
        self.frame_size = frame_size
        # 녹화 파일 경로를 저장하는 덱(deque). 최대 6개 파일 유지 (약 1분)
        self.file_deque = deque(maxlen=6)
        self.out = None  # VideoWriter 객체
        self.frame_count = 0
        self.frames_per_chunk = self.chunk_seconds * self.fps  # 파일당 프레임 수
        os.makedirs(self.output_dir, exist_ok=True) # 저장 디렉토리 생성

    # 새로운 녹화 파일(청크) 시작
    def _start_new_chunk(self):
        # 덱이 꽉 차면 가장 오래된 파일 경로를 꺼내서 실제 파일 삭제
        if len(self.file_deque) == self.file_deque.maxlen:
            try:
                os.remove(self.file_deque.popleft())
            except OSError:
                pass
        # 현재 시간을 파일명으로 하여 새 녹화 파일 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chunk_path = os.path.join(self.output_dir, f"chunk_{timestamp}.mp4")
        self.out = cv2.VideoWriter(chunk_path, self.fourcc, self.fps, self.frame_size)
        self.file_deque.append(chunk_path)

    # 프레임을 받아와 파일에 쓰기
    def write_frame(self, frame):
        # 새 파일이 필요하거나 현재 파일이 꽉 찼으면 새 청크 시작
        if self.out is None or self.frame_count >= self.frames_per_chunk:
            if self.out:
                self.out.release()
            self._start_new_chunk()
            self.frame_count = 0
        # 프레임 크기가 지정된 크기와 다르면 리사이즈
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        self.out.write(frame)
        self.frame_count += 1

    # 녹화 종료
    def stop(self):
        if self.out:
            self.out.release()
            self.out = None

# 긴급 상황 발생 시 나타나는 경고 대화상자
class AlertDialog(QDialog):
    def __init__(self, parent, event_type, prob):
        super().__init__(parent)
        self.setWindowTitle(f"{event_type} 경고")
        self.setModal(False) # 다른 창과 상호작용 가능하도록 비-모달 설정
        self.setWindowModality(Qt.NonModal)
        # 사용자가 닫기 버튼을 누르지 못하도록 설정
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.event_type = event_type
        self.parent_dashboard = parent
        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)
        message = QLabel(f"{event_type} 감지됨! (확률: {prob:.2f})")
        layout.addWidget(message)
        stop_button = QPushButton("상황 종료")
        layout.addWidget(stop_button)
        stop_button.clicked.connect(self.confirm_stop)

    # '상황 종료' 버튼 클릭 시 확인 메시지 박스 표시
    def confirm_stop(self):
        if QMessageBox.question(self, "확인", "상황을 종료하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.parent_dashboard.resolve_alert(self.event_type) # 메인 윈도우에 상황 종료 알림
            self.accept() # 대화상자 닫기

# 로그 및 녹화된 동영상 뷰어 대화상자
class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None, event_colors=None):
        super().__init__(parent)
        self.setWindowTitle("로그 뷰어 및 동영상 재생")
        self.setMinimumSize(1000, 700)
        self.log_entries = log_entries or []
        self.event_colors = event_colors or {}
        self.video_capture = None
        self.playing_video = False
        self.setupUi()
        self.connect_signals()
        self.update_video_list()
        self.filter_and_display_logs()

    # UI 구성
    def setupUi(self):
        main_layout = QHBoxLayout(self)
        
        # 왼쪽: 비디오 패널
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_display = AspectRatioWidget() # 비율 유지 위젯 사용
        self.video_display.video_label.setText("재생할 동영상을 선택하세요.")
        video_layout.addWidget(self.video_display)
        main_layout.addWidget(video_panel, 7) # 70% 너비 차지

        # 오른쪽: 컨트롤 패널
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.addWidget(QLabel("저장된 동영상 목록"))
        self.video_list = QListWidget() # 녹화 파일 목록
        control_layout.addWidget(self.video_list)
        self.play_button = QPushButton("재생")
        control_layout.addWidget(self.play_button)
        
        # 로그 필터링 UI
        filter_layout = QGridLayout()
        self.filter_event_type = QComboBox()
        self.filter_event_type.addItem("모든 이벤트")
        self.filter_event_type.addItems(self.event_colors.keys())
        filter_layout.addWidget(QLabel("이벤트 유형:"), 0, 0)
        filter_layout.addWidget(self.filter_event_type, 0, 1)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("키워드 검색")
        filter_layout.addWidget(self.search_input, 1, 0, 1, 2)
        self.search_button = QPushButton("로그 검색")
        filter_layout.addWidget(self.search_button, 2, 0, 1, 2)
        control_layout.addLayout(filter_layout)

        control_layout.addWidget(QLabel("필터링된 로그"))
        self.log_browser = QTextBrowser()
        control_layout.addWidget(self.log_browser)
        main_layout.addWidget(control_panel, 3) # 30% 너비 차지

    # 시그널-슬롯 연결
    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.play_button.clicked.connect(self.play_selected_video)
        self.search_button.clicked.connect(self.filter_and_display_logs)
        self.filter_event_type.currentIndexChanged.connect(self.filter_and_display_logs)

    # 'recordings' 폴더의 mp4 파일 목록을 읽어와 리스트 위젯 업데이트
    def update_video_list(self):
        self.video_list.clear()
        self.video_list.addItems(sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True))

    # 선택된 비디오 재생/정지 토글
    def play_selected_video(self):
        if self.playing_video:
            self.video_timer.stop()
            self.video_capture.release()
            self.playing_video = False
            self.play_button.setText("재생")
            return

        item = self.video_list.currentItem()
        if item:
            path = os.path.join("recordings", item.text())
            self.video_capture = cv2.VideoCapture(path)
            if self.video_capture.isOpened():
                fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
                self.playing_video = True
                self.play_button.setText("정지")
                self.video_timer.start(int(1000 / fps))

    # 타이머에 의해 주기적으로 호출되어 비디오 프레임 업데이트
    def update_video_frame(self):
        if self.playing_video and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
                self.video_display.video_label.setPixmap(QPixmap.fromImage(qt_image))
            else: # 비디오 끝에 도달하면 재생 중지
                self.video_timer.stop()
                self.video_capture.release()
                self.playing_video = False
                self.play_button.setText("재생")

    # 필터 조건에 따라 로그를 표시
    def filter_and_display_logs(self):
        keyword = self.search_input.text().strip().lower()
        event_type = self.filter_event_type.currentText()
        self.log_browser.clear()
        for entry in self.log_entries:
            if (event_type == "모든 이벤트" or event_type in entry['message']) and \
               (not keyword or keyword in entry['message'].lower()):
                self.log_browser.append(f"{entry['timestamp']} - {entry['message']}")

    # 대화상자가 닫힐 때 비디오 관련 자원 해제
    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        super().closeEvent(event)

# 메인 애플리케이션 윈도우 클래스
class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) # UI 생성
        pygame.mixer.init() # 경고음 재생을 위한 pygame mixer 초기화
        self.alert_sound, self.alert_channel, self.alert_active = None, None, False
        self.open_alerts = [] # 현재 열려있는 경고창 목록
        self.log_entries = [] # 모든 로그 기록

        # AI 모델 및 전처리 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dumping_model = self.smoker_model = None # 실제 모델 로딩은 비활성화
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.frame_buffer = deque(maxlen=16) # AI 추론에 사용할 프레임 버퍼
        self.frame_counter = 0

        self.recorder = RollingRecorder() # 롤링 녹화 객체 생성

        # # --- 네트워크 비디오 스트림 초기화 --- 실제 사용할 서버
        # self.video_source = "http://192.168.0.180:5000/stream?src=0"
        # self.cap = cv2.VideoCapture(self.video_source)

        # --- 네트워크 비디오 스트림 초기화 시험용 로컬서버 ---
        self.video_source = "http://127.0.0.1:5000/video_feed"
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "스트림 오류", f"비디오 스트림을 열 수 없습니다:\n{self.video_source}")
            sys.exit() # 스트림 열기 실패 시 프로그램 종료

        # 메인 타이머 설정: 30fps로 프레임 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)

        # 이벤트 유형별 색상 및 경고음 설정
        self.event_colors = {"화재":"red", "폭행":"orange", "누워있는 사람":"yellow", "실종자":"cyan", "무단 투기":"lightgray", "흡연자":"lightgray"}
        if os.path.exists("alert.wav"):
            self.alert_sound = pygame.mixer.Sound("alert.wav")
            self.alert_channel = pygame.mixer.Channel(0)

    # 프로그램 방식의 UI 생성
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("AURA 관제 대시보드 (단일 화면)")
        MainWindow.resize(1600, 900)

        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)

        # 왼쪽: 단일 비디오 위젯
        self.video_widget = AspectRatioWidget()
        self.main_layout.addWidget(self.video_widget, 7) # 전체 너비의 70%

        # 오른쪽: 컨트롤 패널
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        self.main_layout.addWidget(control_panel, 3) # 전체 너비의 30%

        self.log_open_button = QPushButton("로그 열기 및 동영상 재생")
        self.control_layout.addWidget(self.log_open_button)

        # 수동 이벤트 발생 버튼 그룹
        trigger_groupbox = QGroupBox("수동 이벤트 발생")
        button_layout = QVBoxLayout(trigger_groupbox)
        self.trigger_buttons = {}
        event_names = ["폭행", "화재", "누워있는 사람", "실종자", "무단 투기", "흡연자"]
        for name in event_names:
            btn = QPushButton(f"{name} 발생")
            self.trigger_buttons[name] = btn
            button_layout.addWidget(btn)
        self.control_layout.addWidget(trigger_groupbox)

        # 통합 이벤트 로그
        log_label = QLabel("통합 이벤트 로그")
        font = QFont(); font.setPointSize(12); font.setBold(True)
        log_label.setFont(font)
        self.control_layout.addWidget(log_label)

        self.log_browser = QTextBrowser()
        self.control_layout.addWidget(self.log_browser)

        # 시그널-슬롯 연결
        self.log_open_button.clicked.connect(self.open_log_viewer)
        for name, btn in self.trigger_buttons.items():
            btn.clicked.connect(lambda checked, n=name: self.trigger_event(n))

    # 타이머에 의해 주기적으로 호출되는 메인 루프
    def update_frame(self):
        # --- [수정] 네트워크 스트림에서 프레임 읽기 ---
        ret, frame = self.cap.read()

        if not ret: # 프레임 읽기 실패 시
            # "NO SIGNAL" 메시지를 표시할 검은색 프레임 생성
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else: # 프레임 읽기 성공 시
            # 프레임에 현재 시간과 녹화 중 아이콘 표시
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, now_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 녹화기에 프레임 전달
            self.recorder.write_frame(frame)
            
            # AI 추론 시뮬레이션
            self.frame_buffer.append(self.preprocess_frame(frame))
            self.frame_counter += 1
            if self.frame_counter >= 30: # 1초(30프레임)에 한번씩 추론 시뮬레이션
                if random.random() < 0.1: # 10% 확률로 무단투기 자동 감지
                    self.run_inference(self.dumping_model, "무단 투기")
                if random.random() < 0.1: # 10% 확률로 흡연자 자동 감지
                    self.run_inference(self.smoker_model, "흡연자")
                self.frame_counter = 0

        # 비디오 위젯에 현재 프레임 표시
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_widget.video_label.setPixmap(pixmap)

    # AI 모델 입력을 위해 프레임 전처리
    def preprocess_frame(self, frame):
        return self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # AI 추론 실행 (현재는 시뮬레이션)
    def run_inference(self, model, event_type):
        # 실제 모델 추론 대신 랜덤 확률로 시뮬레이션
        prob = random.uniform(0.7, 0.99)
        self.trigger_event(event_type, prob, is_auto=True)

    # 이벤트 발생 처리 (수동 또는 자동)
    def trigger_event(self, event_type, prob=None, is_auto=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = prob if prob is not None else random.random()
        color = self.event_colors.get(event_type, "white")
        log_message = f"[{'자동' if is_auto else '수동'}] {event_type} 감지됨 (확률: {prob:.2f})"

        # 통합 로그 브라우저에 HTML 형식으로 로그 추가
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})

        # 특정 중요 이벤트에 대해서는 경고창 및 경고음 발생
        if event_type in ["폭행", "화재", "누워있는 사람"]:
            if self.alert_sound and self.alert_channel and not self.alert_active:
                self.alert_channel.play(self.alert_sound, loops=-1) # 무한 반복 재생
                self.alert_active = True
            dialog = AlertDialog(self, event_type, prob)
            self.open_alerts.append(dialog)
            self.place_alert_dialog(dialog) # 경고창 위치 자동 배치
            dialog.show()

    # 경고 대화상자가 겹치지 않도록 위치 계산
    def place_alert_dialog(self, dialog):
        screen_rect = QApplication.primaryScreen().geometry()
        margin = 20
        d_w, d_h = 300, 150
        positions = {(d.pos().x(), d.pos().y()) for d in self.open_alerts if d is not dialog}
        # 화면 좌측 상단부터 빈 공간을 찾아 배치
        for r in range((screen_rect.height() - margin) // (d_h + margin)):
            for c in range((screen_rect.width() - margin) // (d_w + margin)):
                x = margin + c * (d_w + margin)
                y = margin + r * (d_h + margin)
                if (x, y) not in positions:
                    dialog.move(x, y)
                    return
        dialog.move(margin, margin) # 빈 공간이 없으면 기본 위치에 배치

    # 경고창에서 '상황 종료'가 확인되었을 때 호출
    def resolve_alert(self, event_type):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"<font color='green'>{timestamp} - {event_type} 상황 종료됨</font>"
        self.log_browser.append(log_message)
        self.log_entries.append({"timestamp": timestamp, "message": f"{event_type} 상황 종료됨"})
        # 약간의 딜레이 후 경고음 중지 여부 체크 (여러 경고창이 있을 수 있으므로)
        QTimer.singleShot(100, self.check_and_stop_sound)

    # 모든 경고창이 닫혔는지 확인하고 경고음 중지
    def check_and_stop_sound(self):
        # 현재 화면에 보이는 AlertDialog가 하나도 없으면
        if not any(isinstance(w, AlertDialog) and w.isVisible() for w in QApplication.topLevelWidgets()) and self.alert_active:
            if self.alert_channel:
                self.alert_channel.stop()
            self.alert_active = False

    # 로그 뷰어 대화상자 열기
    def open_log_viewer(self):
        dialog = LogViewerDialog(self, self.log_entries, self.event_colors)
        dialog.exec()

    # 메인 윈도우가 닫힐 때 호출되는 이벤트
    def closeEvent(self, event):
        self.timer.stop()
        self.recorder.stop()
        pygame.mixer.quit()
        if self.cap: # --- [수정] 비디오 캡처 객체 해제 ---
            self.cap.release()
        
        # 열려있는 모든 대화상자 닫기
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog):
                w.close()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.show()
    sys.exit(app.exec())