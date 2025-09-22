import sys
import cv2
import os
from datetime import datetime, timedelta
from collections import deque
import random
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import re
import json
import socket
import threading
from queue import Queue

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QTextBrowser, QDialog, QGridLayout, QLineEdit, QPushButton,
                               QComboBox, QMessageBox, QListWidget, QSizePolicy, QGroupBox, QTabWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit, QCalendarWidget,
                               QAbstractItemView, QCheckBox)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPoint, QRect, QDate, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

# --- 유틸리티 및 AI 모델 클래스 (이전과 동일) ---

class ActionClassifier(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2):
        super(ActionClassifier, self).__init__()
        self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden_size, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b * t, c, h, w)
        features = self.cnn(x)
        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        out = self.classifier(lstm_out[:, -1, :])
        return out

class AspectRatioWidget(QWidget):
    def __init__(self, parent=None, aspect_ratio=16.0/9.0):
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True) # 비율 할당 3,7 위해서 필요
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)
        self.setLayout(layout)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        if w / h > self.aspect_ratio:
            new_w = int(h * self.aspect_ratio)
            self.video_label.setGeometry((w - new_w) // 2, 0, new_w, h)
        else:
            new_h = int(w / self.aspect_ratio)
            self.video_label.setGeometry(0, (h - new_h) // 2, w, new_h)

class RollingRecorder:
    def __init__(self, output_dir="recordings", chunk_seconds=10, fps=30, frame_size=(1280, 720)):
        self.output_dir = output_dir
        self.chunk_seconds = chunk_seconds
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = fps
        self.frame_size = frame_size
        self.file_deque = deque(maxlen=6)
        self.out = None
        self.frame_count = 0
        self.frames_per_chunk = self.chunk_seconds * self.fps


        os.makedirs(self.output_dir, exist_ok=True)
    def _start_new_chunk(self):
        if len(self.file_deque) == self.file_deque.maxlen:
            try:
                os.remove(self.file_deque.popleft())
            except OSError:
                pass
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chunk_path = os.path.join(self.output_dir, f"chunk_{timestamp}.mp4")
        self.out = cv2.VideoWriter(chunk_path, self.fourcc, self.fps, self.frame_size)
        self.file_deque.append(chunk_path)
    def write_frame(self, frame):
        if self.out is None or self.frame_count >= self.frames_per_chunk:
            if self.out:
                self.out.release()
            self._start_new_chunk()
            self.frame_count = 0

        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        self.out.write(frame)
        self.frame_count += 1
    def stop(self):
        if self.out:
            self.out.release()
            self.out = None

class AlertDialog(QDialog):
    def __init__(self, parent, event_type, prob):
        super().__init__(parent)
        self.setWindowTitle(f"{event_type} 경고")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.event_type = event_type
        self.parent_dashboard = parent
        self.setFixedSize(300, 150)
        layout = QVBoxLayout(self)
        message = QLabel(f"{event_type} 감지됨! (확률: {prob:.2f})")
        layout.addWidget(message)
        stop_button = QPushButton("확인")
        layout.addWidget(stop_button)
        stop_button.clicked.connect(self.confirm_stop)
    def confirm_stop(self):
        if QMessageBox.question(self, "확인", "경고을 종료하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.parent_dashboard.resolve_alert(self.event_type)
            self.accept()

# --- [수정] LogViewerDialog 클래스 전체를 아래 코드로 교체 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None, event_colors=None):
        super().__init__(parent)
        self.setWindowTitle("로그 뷰어 및 동영상 재생")
        self.setMinimumSize(1200, 800)

        # 데이터 및 상태 변수
        self.all_log_entries = log_entries or [] # 초기 데이터 (시뮬레이션용)
        self.filtered_log_entries = []
        self.event_colors = event_colors or {}
        
        # [추가] 이벤트 이름과 검색용 코드 매핑
        self.event_type_map = {
            "화재": "0", "폭행": "1", "누워있는 사람": "2", 
            "실종자": "3", "무단 투기": "4", "흡연자": "5"
        }

        # 녹화 파일 목록 로드
        self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        self.video_capture = None
        self.playing_video = False
        
        # 페이지네이션
        self.current_page = 1
        self.items_per_page = 20

        self.setupUi()
        self.connect_signals()
        self.request_logs_from_server() # 시작 시 기본값으로 로그 요청
        self.add_example_log() # 예시 로그 추가


    def setupUi(self):
        main_layout = QHBoxLayout(self)

        # 1. 왼쪽: 비디오 패널
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        
        # [수정] 비디오 디스플레이를 640*480 고정 크기 QLabel로 변경
        self.video_display = QLabel("재생할 동영상을 선택하세요.")
        self.video_display.setFixedSize(640, 480)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: rgb(40, 40, 40); color: white;")
        self.video_display.setScaledContents(True) # 이미지가 라벨 크기에 맞게 조절되도록 설정
        
        video_layout.addWidget(self.video_display)
        video_layout.addStretch() # 비디오 위젯이 상단에 고정되도록
        main_layout.addWidget(video_panel) # 비율 없이 추가

        # 2. 오른쪽: 컨트롤 및 로그 테이블 패널
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 2a. 검색 조건 UI
        search_groupbox = QGroupBox("검색 조건")
        search_layout = QGridLayout(search_groupbox)
        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7)) # 기본값: 7일 전
        self.end_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.orderby_combo = QComboBox()
        self.orderby_combo.addItems(["최신순", "오래된 순"])
        
        search_layout.addWidget(QLabel("시작일:"), 0, 0)
        search_layout.addWidget(self.start_date_edit, 0, 1)
        search_layout.addWidget(QLabel("종료일:"), 1, 0)
        search_layout.addWidget(self.end_date_edit, 1, 1)
        search_layout.addWidget(QLabel("정렬:"), 2, 0)
        search_layout.addWidget(self.orderby_combo, 2, 1)
        
        # 이벤트 유형 체크박스
        event_groupbox = QGroupBox("이벤트 종류")
        self.event_checkboxes = {}
        event_layout = QGridLayout(event_groupbox)
        
        # 체크박스를 2열로 배치
        event_names = list(self.event_type_map.keys())
        for i, name in enumerate(event_names):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True) # 기본값은 모두 선택
            self.event_checkboxes[name] = checkbox
            event_layout.addWidget(checkbox, i // 2, i % 2)
            
        search_layout.addWidget(event_groupbox, 3, 0, 1, 2)
        
        self.search_button = QPushButton("로그 검색")
        search_layout.addWidget(self.search_button, 4, 0, 1, 2)
        control_layout.addWidget(search_groupbox)

        # 2b. 로그 테이블
        self.log_table = QTableWidget()
        # [수정] 컬럼 개수 및 이름 변경
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["발생 시각", "이벤트 종류", "영상 재생"])
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_table.setSortingEnabled(True)
        control_layout.addWidget(self.log_table)

        # 2c. 페이지네이션 컨트롤
        pagination_layout = QHBoxLayout()
        self.prev_button = QPushButton("이전")
        self.page_label = QLabel("1 / 1")
        self.next_button = QPushButton("다음")
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_button)
        pagination_layout.addStretch()
        control_layout.addLayout(pagination_layout)

        main_layout.addWidget(control_panel, 1) # 비율 1 할당

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)

    # [수정] 서버에 로그를 요청하는 메서드 (JSON 생성 및 시뮬레이션)
    def request_logs_from_server(self):
        start_date = self.start_date_edit.date().toString("yyyyMMdd")
        end_date = self.end_date_edit.date().toString("yyyyMMdd")
        
        orderby = True if self.orderby_combo.currentText() == "최신순" else False
        
        detection_types = []
        for name, checkbox in self.event_checkboxes.items():
            if checkbox.isChecked():
                detection_types.append(self.event_type_map[name])

        # Search_JSON 생성
        search_json = {
            "time_start": start_date,
            "time_end": end_date,
            "time_orderby": orderby,
            "detection_type": detection_types
        }

        # --- 실제로는 이 부분에서 TCP 통신으로 서버에 search_json을 보냅니다 ---
        print("--- 데이터 요청 JSON ---")
        import json
        print(json.dumps(search_json, indent=2))
        
        # --- 시뮬레이션: 현재 가지고 있는 로그를 서버에서 받은 것처럼 필터링 ---
        # 실제 구현 시, 아래 로직은 서버 응답을 받은 후 실행됩니다.
        self.filtered_log_entries = []
        start_dt = self.start_date_edit.dateTime().toPython().date()
        end_dt = self.end_date_edit.dateTime().toPython().date()
        
        for entry in self.all_log_entries:
            try:
                log_dt = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                if not (start_dt <= log_dt <= end_dt):
                    continue

                # 이벤트 유형 필터링
                is_target_event = False
                for name, code in self.event_type_map.items():
                    if code in detection_types and name in entry['message']:
                        is_target_event = True
                        break
                if is_target_event:
                    self.filtered_log_entries.append(entry)
            except ValueError:
                continue
        
        # 정렬
        self.filtered_log_entries.sort(key=lambda x: x['timestamp'], reverse=orderby)
        
        self.current_page = 1
        self.update_table_display()

    def update_table_display(self):
        self.log_table.setSortingEnabled(False)
        self.log_table.setRowCount(0)

        start_index = (self.current_page - 1) * self.items_per_page
        end_index = start_index + self.items_per_page
        logs_to_display = self.filtered_log_entries[start_index:end_index]

        for row, entry in enumerate(logs_to_display):
            self.log_table.insertRow(row)

            # 이벤트 종류 파싱
            message = entry['message']
            event_type = "정보" # 기본값
            # 정규식을 사용하여 대괄호 뒤의 첫 단어를 이벤트 종류로 간주
            match = re.search(r"\]\s*([\w\s]+?)\s*(감지 됨|확인 됨)", message)
            if match:
                event_type = match.group(1).strip()
            
            # [수정] 3개 열에 데이터 채우기
            self.log_table.setItem(row, 0, QTableWidgetItem(entry['timestamp']))
            self.log_table.setItem(row, 1, QTableWidgetItem(event_type))

            play_button = QPushButton("재생")
            play_button.clicked.connect(lambda chk, ts=entry['timestamp']: self.play_video_for_log(ts))
            self.log_table.setCellWidget(row, 2, play_button)

        self.log_table.setSortingEnabled(True)
        self.update_pagination_controls()

    # (이하 나머지 메서드는 이전과 동일)
    def update_pagination_controls(self):
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        self.page_label.setText(f"{self.current_page} / {total_pages}")
        self.prev_button.setEnabled(self.current_page > 1)
        self.next_button.setEnabled(self.current_page < total_pages)

    def go_to_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_table_display()

    def go_to_next_page(self):
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        if self.current_page < total_pages:
            self.current_page += 1
            self.update_table_display()

    def play_video_for_log(self, timestamp_str):
        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        best_match = None
        min_diff = timedelta.max
        for filename in self.recording_files:
            try:
                file_time_str = filename.replace('chunk_', '').replace('.mp4', '').replace('_', ' ', 1)
                file_time = datetime.strptime(file_time_str, "%Y-%m-%d %H:%M:%S")
                if log_time >= file_time:
                    diff = log_time - file_time
                    if diff < min_diff:
                        min_diff = diff
                        best_match = filename
            except ValueError:
                continue
        if best_match:
            self.play_video(os.path.join("recordings", best_match))
        else:
            self.video_display.setText("해당 시간에 녹화된 영상이 없습니다.")

    def play_video(self, path):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
        self.video_capture = cv2.VideoCapture(path)
        if self.video_capture.isOpened():
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.playing_video = True
            self.video_timer.start(int(1000 / fps))
        else:
            self.playing_video = False

    def update_video_frame(self):
        if self.playing_video and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                # [수정] 고정된 QLabel 크기에 맞춰 이미지 리사이즈
                qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_display.setPixmap(pixmap) # setPixmap으로 변경
            else:
                self.video_timer.stop()
                if self.video_capture: self.video_capture.release()
                self.playing_video = False
    
    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        super().closeEvent(event)

    def add_example_log(self):
        # 예시 데이터
        example_timestamp = "2025-09-19 18:02:25"
        example_event_type = "화재 감지"
        
        # 테이블의 가장 첫 번째 행(0번 인덱스)에 새로운 행을 삽입합니다.
        self.log_table.insertRow(0)
        
        # 각 셀에 예시 데이터를 채웁니다.
        self.log_table.setItem(0, 0, QTableWidgetItem(example_timestamp))
        self.log_table.setItem(0, 1, QTableWidgetItem(example_event_type))
        
        # '재생' 버튼을 생성하고 시그널을 연결합니다.
        play_button = QPushButton("재생")
        # 예시 데이터이므로, 버튼을 누르면 가장 최근 영상을 재생하도록 연결합니다.
        if self.recording_files:
            latest_video = os.path.join("recordings", self.recording_files[0])
            play_button.clicked.connect(lambda: self.play_video(latest_video))
        
        # 버튼을 테이블 셀에 추가합니다.
        self.log_table.setCellWidget(0, 2, play_button)

#============ TCP 수신 스레드 (TcpReceiver)::========================================================================================
class TcpReceiver(QThread):
    data_received = Signal(dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.running = True

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = ('localhost', 2401)
        server_socket.bind(server_address)
        server_socket.listen(1)
        print(f"TCP 서버 시작 - 포트: {server_address[1]}")
        
        while self.running:
            client_socket, client_address = server_socket.accept()
            print(f"클라이언트 연결: {client_address}")
            while self.running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    json_data = json.loads(data.decode('utf-8'))
                    self.data_received.emit(json_data)
                except (json.JSONDecodeError, ConnectionError) as e:
                    print(f"데이터 수신 오류: {e}")
                    break
            client_socket.close()
        server_socket.close()

    def stop(self):
        self.running = False

# --- 메인 애플리케이션 윈도우 클래스 (이전과 거의 동일) ---
class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) # UI 생성
        
        self.log_viewer_dialog = None
        
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
        self.event_colors = {"화재":"red", "폭행":"orange", "누워있는 사람":"purple", "실종자":"cyan", "무단 투기":"lightgray", "흡연자":"lightgray"}
        if os.path.exists("alert.wav"):
            self.alert_sound = pygame.mixer.Sound("alert.wav")
            self.alert_channel = pygame.mixer.Channel(0)

# ======    TCP 수신 스레드 초기화 및 시작      =================================
        self.tcp_receiver = TcpReceiver(self)
        self.tcp_receiver.data_received.connect(self.process_tcp_data)
        self.tcp_receiver.start()
# ==========================================================================


    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("AURA 관제 대시보드")
        
        screen_rect = QApplication.primaryScreen().availableGeometry()
        MainWindow.showMaximized()

        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        margin = 10
        self.main_layout.setContentsMargins(margin, margin, margin, margin)
        self.main_layout.setSpacing(margin)

        # 비디오 위젯의 너비를 화면 너비의 일부로 설정하여 유연성 확보
        self.video_widget = AspectRatioWidget()
        video_width = int(screen_rect.width() * 0.6) # 화면 너비의 60% 사용
        self.video_widget.setMinimumWidth(video_width)
        self.main_layout.addWidget(self.video_widget, 7) # 7의 비율을 할당

        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        self.main_layout.addWidget(control_panel, 3) # 3의 비율을 할당

        self.log_open_button = QPushButton("로그 뷰어 및 동영상 재생")
        self.control_layout.addWidget(self.log_open_button)

        trigger_groupbox = QGroupBox("수동 이벤트 발생")
        button_layout = QVBoxLayout(trigger_groupbox)
        self.trigger_buttons = {}
        event_names = ["폭행", "화재", "누워있는 사람", "실종자 발견", "무단 투기", "흡연자"]
        for name in event_names:
            btn = QPushButton(f"{name} 발생")
            self.trigger_buttons[name] = btn
            button_layout.addWidget(btn)
        self.control_layout.addWidget(trigger_groupbox)

        log_label = QLabel("통합 이벤트 로그")
        font = QFont(); font.setPointSize(12); font.setBold(True)
        log_label.setFont(font)
        self.control_layout.addWidget(log_label)

        self.log_browser = QTextBrowser()
        self.control_layout.addWidget(self.log_browser)

        self.log_open_button.clicked.connect(self.open_log_viewer)
        for name, btn in self.trigger_buttons.items():
            btn.clicked.connect(lambda checked, n=name: self.trigger_event(n))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, now_str, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            # cv2.putText(frame, "REC", (50, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # 녹화하지 않으므로 삭제
            self.recorder.write_frame(frame)
            self.frame_buffer.append(self.preprocess_frame(frame))
            self.frame_counter += 1

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_widget.video_label.setPixmap(pixmap)

    def preprocess_frame(self, frame):
        return self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def run_inference(self, model, event_type):
        prob = random.uniform(0.7, 0.99)
        self.trigger_event(event_type, prob, is_auto=True)

    def trigger_event(self, event_type, prob=None, is_auto=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = prob if prob is not None else random.random()
        color = self.event_colors.get(event_type, "red")
        log_message = f"[{'자동' if is_auto else '수동'}] {event_type} 감지됨 (확률: {prob:.2f})"
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})


        if event_type in ["폭행", "화재", "누워있는 사람", "실종자 발견"]:
            
            if self.alert_sound and self.alert_channel:
            # and not self.alert_active:
                self.alert_channel.set_endevent(pygame.USEREVENT + 1)
                pygame.time.set_timer(pygame.USEREVENT + 1, 5000, loops=1) # 5000ms = 5초
                self.alert_channel.play(self.alert_sound)

                self.alert_channel.play(self.alert_sound, loops=-1)
                self.alert_active = True
            dialog = AlertDialog(self, event_type, prob)
            self.open_alerts.append(dialog)
            self.place_alert_dialog(dialog)
            dialog.show()

    def place_alert_dialog(self, dialog):
        # 1. 기본 위치 및 간격 설정
        start_pos = QPoint(60, 20)      # 첫 번째 팝업의 시작 위치
        cascade_offset = QPoint(40, 40) # 같은 줄에서 대각선으로 이동할 간격
        row_y_offset = 100               # 다음 줄로 넘어갈 때 추가할 y 간격
        cascade_per_row = 10            # 한 줄에 표시할 최대 팝업 수

        # 2. 현재 열려있는 팝업 수 확인
        num_open_alerts = len(self.open_alerts) - 1

        # 3. 현재 팝업의 행과 열 위치 계산
        current_row = num_open_alerts // cascade_per_row
        current_col = num_open_alerts % cascade_per_row

        # 4. 최종 위치 계산
        # 기본 시작 위치에서 (열 * 대각선 간격) 만큼 이동
        new_pos = start_pos + (current_col * cascade_offset)
        # 행이 바뀌었다면 (행 * 줄 간격) 만큼 y축으로 추가 이동
        new_pos.setY(new_pos.y() + (current_row * row_y_offset))
        
        # 5. 다이얼로그를 계산된 위치로 이동
        dialog.move(new_pos)

    def resolve_alert(self, event_type):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"<font color='green'>{timestamp} - {event_type} 상황 종료됨</font>"
        self.log_browser.append(log_message)
        self.log_entries.append({"timestamp": timestamp, "message": f"{event_type} 상황 종료됨"})
        QTimer.singleShot(100, self.check_and_stop_sound)

    def check_and_stop_sound(self):
        if not any(isinstance(w, AlertDialog) and w.isVisible() for w in QApplication.topLevelWidgets()) and self.alert_active:
            if self.alert_channel:
                self.alert_channel.stop()
            self.alert_active = False

    def open_log_viewer(self):
        if self.log_viewer_dialog and self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog.activateWindow()
            return
            
        self.log_viewer_dialog = LogViewerDialog(self, self.log_entries, self.event_colors)
        
        main_window_pos = self.geometry().topRight()
        self.log_viewer_dialog.move(main_window_pos)
        
        self.log_viewer_dialog.show()

#==== class PatrolDashboard(QMainWindow): 안에 있음    =============================================
#====: TCP 데이터 처리 (process_tcp_data):              =============================================

    def process_tcp_data(self, json_data):
        print("TCP 데이터 처리 시작:", json_data)  # 디버깅 로그 추가
        timestamp = json_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for detection in json_data.get("detection", []):
            event_type = detection["class_name"]
            confidence = detection["confidence"]
            color = self.event_colors.get(event_type, "red")
            log_message = f"[TCP] {event_type} 감지됨 (확률: {confidence:.2f})"
            html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
            self.log_browser.append(html_log)
            self.log_entries.append({"timestamp": timestamp, "message": log_message})
            if event_type in ["폭행", "화재", "누워있는 사람"] and confidence > 0.7:
                if self.alert_sound and self.alert_channel and not self.alert_active:
                    self.alert_channel.play(self.alert_sound, loops=-1)
                    self.alert_active = True
                dialog = AlertDialog(self, event_type, confidence)
                self.open_alerts.append(dialog)
                self.place_alert_dialog(dialog)
                dialog.show()
        if self.log_viewer_dialog and self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog.all_log_entries = self.log_entries
            self.log_viewer_dialog.request_logs_from_server()

    def closeEvent(self, event):
        print("Closing application...")
        
        # 1. 모든 타이머 중지
        self.timer.stop()
        
        # 2. 모든 자식 다이얼로그 창 닫기
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog):
                w.close()

        # 3. 백그라운드 스레드 (TCP 서버) 안전하게 종료
        print("Stopping TCP receiver thread...")
        self.tcp_receiver.stop()
        
        # --- 스레드가 accept()에서 대기하는 것을 깨우기 위한 트릭 ---
        try:
            # 로컬호스트의 서버 소켓으로 임시 접속하여 accept() 대기 상태를 해제
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 2401))
        except ConnectionRefusedError:
            # 서버가 이미 닫히고 있는 경우 정상
            pass
        except Exception as e:
            print(f"Error while trying to unblock TCP server: {e}")

#=======    TCP     ===================================                
        self.tcp_receiver.wait(2000) # 스레드가 종료될 때까지 최대 2초 대기
        if self.tcp_receiver.isRunning():
            print("TCP thread is still running, terminating...")
            self.tcp_receiver.terminate() # 최후의 수단으로 강제 종료

        print("TCP thread stopped.")
        
        # 4. OpenCV 및 Pygame 리소스 해제
        print("Releasing resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.recorder.stop()
        pygame.mixer.quit()
        print("Resources released.")

        # 5. 부모 클래스의 closeEvent 호출하여 창 닫기 완료
        super().closeEvent(event)
#=======    TCP     ===================================                


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.show()
    sys.exit(app.exec())
