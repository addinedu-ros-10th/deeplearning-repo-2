
from log_viewer import LogViewerDialog

import time
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
import struct  # 
from queue import Queue

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QTextBrowser, QDialog, QGridLayout, QLineEdit, QPushButton,
                               QComboBox, QMessageBox, QListWidget, QSizePolicy, QGroupBox, QTabWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit, QCalendarWidget,
                               QAbstractItemView, QCheckBox)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPoint, QRect, QDate, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

# --- 유틸리티 및 AI 모델 클래스 ---

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
        self.video_label.setScaledContents(True)
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
        self.frames_per_chunk = int(self.chunk_seconds * self.fps)
        os.makedirs(self.output_dir, exist_ok=True)

    def _start_new_chunk(self):
        if len(self.file_deque) == self.file_deque.maxlen:
            try: os.remove(self.file_deque.popleft())
            except OSError: pass
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chunk_path = os.path.join(self.output_dir, f"chunk_{timestamp}.mp4")
        self.out = cv2.VideoWriter(chunk_path, self.fourcc, self.fps, self.frame_size)
        self.file_deque.append(chunk_path)

    def write_frame(self, frame):
        if self.out is None or self.frame_count >= self.frames_per_chunk:
            if self.out: self.out.release()
            self._start_new_chunk()
            self.frame_count = 0
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        self.out.write(frame)
        self.frame_count += 1

    def stop(self):
        if self.out: self.out.release()
        self.out = None

class AlertDialog(QDialog):
    # __init__ 메서드에 timestamp 인자를 추가합니다.
    def __init__(self, parent, event_type, prob, timestamp):
        super().__init__(parent)
        self.setWindowTitle(f"{event_type} 발생")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.event_type = event_type
        self.parent_dashboard = parent
        
        # [수정] 텍스트가 두 줄로 늘어나므로 높이를 약간 늘려줍니다.
        self.setFixedSize(300, 170)
        self.blink_timer = None

        layout = QVBoxLayout(self)
        
        # [수정] 메시지에 '발생 시각'을 포함하도록 변경합니다.
        message_text = f"{event_type} 감지됨! (확률: {prob:.2f})\n\n발생 시각: {timestamp}"
        message = QLabel(message_text)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter) # 텍스트 가운데 정렬
        
        layout.addWidget(message)
        stop_button = QPushButton("확인")
        layout.addWidget(stop_button)
        stop_button.clicked.connect(self.confirm_stop)

    def confirm_stop(self):
        if QMessageBox.question(self, "확인", "경고를 종료하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            # confirm_stop을 호출할 때 dialog 인스턴스(self)를 전달하도록 수정
#================================= 연결고리 1 =============================
            self.parent_dashboard.resolve_alert(self, self.event_type)
#================================= 연결고리 1 =============================
            self.accept()



class TcpReceiver(QThread):
    """
    백그라운드에서 TCP 소켓 통신을 통해 JSON 데이터를 수신하는 스레드 클래스.
    데이터를 수신하면 'data_received' 시그널을 발생시켜 메인 스레드로 전달합니다.
    """
    data_received = Signal(dict)  # 수신된 데이터를 담아 보낼 시그널 (dict 타입)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

        # [수정] 접속할 서버의 주소와 포트로 변경
        self.server_host = '192.168.0.86'  # Server(situationDectector)의 IP
        self.server_port = 2401           # Server(situationDectector)의 Result_JSON 수신 포트

    def run(self):
        """스레드가 시작될 때 실행되는 메인 루프"""
        while self.running:
            try:
                # [수정] 클라이언트 소켓 생성 및 서버에 연결 시도
                print(f"Connecting to server {self.server_host}:{self.server_port}...")
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                client_socket.connect((self.server_host, self.server_port))
                
                print("Server connected.")
                
                # 연결이 성공하면, 연결이 끊길 때까지 계속 데이터 수신
                with client_socket:
                    full_data = b""
                    while self.running:
                        # 1024 바이트씩 데이터 수신
                        data = client_socket.recv(1024)
                        # 서버가 연결을 끊으면 data는 비어있게 됨
                        if not data:
                            print("Server disconnected.")
                            break
                        
                        # [추가] 여러 JSON이 붙어서 오는 경우를 대비한 처리
                        # 실제로는 데이터의 끝을 알리는 구분자(delimiter)가 필요하지만,
                        # 여기서는 수신된 데이터를 바로 처리한다고 가정합니다.
                        full_data += data
                        
                        # 수신된 데이터가 있으면 처리
                        if full_data:
                            try:
                                json_str = full_data.decode('utf-8')
                                json_data = json.loads(json_str)
                                self.data_received.emit(json_data)
                                full_data = b"" # 처리가 끝났으므로 버퍼 비우기
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                # 아직 JSON 데이터가 완성되지 않았을 수 있으므로 오류 대신 계속 데이터를 받습니다.
                                # print(f"Incomplete data received: {e}")
                                pass

            except ConnectionRefusedError:
                # 서버가 닫혀있거나 연결을 거부할 경우
                print("Connection refused. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                # 그 외 다른 네트워크 오류
                print(f"TCP client error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        
        print("TCP Receiver (Client Mode) stopped.")

    def stop(self):
        """스레드를 안전하게 종료하기 위한 메서드"""
        self.running = False
        print("Stopping TCP client thread...")

class VideoThread(QThread):
    # 프레임을 전달하기 위한 시그널 정의
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, video_source, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.video_source = video_source

    def run(self):
        # 스레드가 시작되면 이 루프가 실행됨
        cap = cv2.VideoCapture(self.video_source)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 프레임을 성공적으로 읽으면 시그널을 통해 메인 스레드로 전송
                self.change_pixmap_signal.emit(frame)
            else:
                # 프레임 읽기 실패 (연결 끊김 등)
                print(f"스트림 연결 끊김. 5초 후 재연결 시도...")
                cap.release() # 자원 해제
                time.sleep(5) # 5초 대기
                cap = cv2.VideoCapture(self.video_source) # 재연결 시도
        
        # 루프 종료 시 자원 해제
        cap.release()
        print("VideoThread 종료.")

    def stop(self):
        """스레드를 안전하게 종료"""
        self._run_flag = False
        self.wait() # 스레드가 완전히 종료될 때까지 대기


# --- 메인 대시보드 클래스 ---
class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.log_viewer_dialog = None

          # Pygame 믹서 초기화
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # --- [수정] 여러 소리를 동시에 재생하기 위해 8개의 채널을 생성합니다 ---
        pygame.mixer.set_num_channels(16)
        
        # --- [수정] 단일 채널 변수(self.alert_channel)를 삭제하고, 사운드 딕셔너리만 남깁니다 ---
        self.alert_sounds = {} 
        
        # 이벤트 이름과 실제 사운드 파일 이름을 매핑하는 딕셔너리
        sound_map = {
            "화재": "fire.mp3",
            "폭행": "violence.mp3",
            "쓰러진 사람": "faint.mp3",
            "실종자 발견": "missing_person.mp3"
        }

        # 매핑된 사운드 파일들을 미리 로드
        print("--- 알람 사운드 로딩 시작 ---")
        for event, filename in sound_map.items():
            path = os.path.join("alert", filename)
            if os.path.exists(path):
                self.alert_sounds[event] = pygame.mixer.Sound(path)
                print(f"'{path}' 로드 성공.")
            else:
                print(f"경고: 사운드 파일을 찾을 수 없습니다 - '{path}'")
        print("--- 알람 사운드 로딩 완료 ---")

        self.open_alerts = []
        self.log_entries = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dumping_model = self.smoker_model = None
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.frame_buffer = deque(maxlen=16)
        self.frame_counter = 0
        self.recorder = RollingRecorder()


        # # --- 로컬 웹캠으로 직접 연결 ---
        # self.camera_index = 0  # 사용할 카메라 번호 (0:기본, 1:두번째)
        # self.cap = cv2.VideoCapture(self.camera_index)

        # if not self.cap.isOpened():
        #     QMessageBox.critical(self, "카메라 오류", f"{self.camera_index}번 카메라를 열 수 없습니다.")
        #     sys.exit()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_frame)
        # self.timer.start(1000 // 30)
        # # --- 로컬 웹캠으로 직접 연결 ---

        # --- 새로운 VideoThread 설정 ---
        self.video_source = "http://192.168.0.180:5000/stream?src=0"

        self.thread = VideoThread(self.video_source) # 비디오 스레드 생성
        self.thread.change_pixmap_signal.connect(self.update_image) # 시그널과 슬롯 연결
        self.thread.start() # 스레드 시작
        # --- 새로운 VideoThread 설정 ---

        self.event_colors = {"화재":"red", "폭행":"orange", "쓰러진 사람":"purple", "실종자":"cyan", "무단 투기":"lightgray", "흡연자":"lightgray"}
        if os.path.exists("alert.mp3"):
            self.alert_sound = pygame.mixer.Sound("alert.mp3")
            self.alert_channel = pygame.mixer.Channel(0)

        self.event_to_code_map = {
            "쓰러진 사람": 0,
            "화재": 1,
            "폭행": 2,
            "실종자 발견": 3
        }
    
        self.tcp_receiver = TcpReceiver(self)
        self.tcp_receiver.data_received.connect(self.process_tcp_data)
        self.tcp_receiver.start()

        # --- [추가] AI 모델의 영문 클래스 이름을 UI의 한글 이벤트 이름으로 변환하기 위한 딕셔너리 ---
        self.class_name_map = {
            # 화재 관련 이벤트
            "detect_fire": "화재",
            "detect_fire_danger_smoke": "화재",
            "detect_fire_general_smoke": "화재",
            
            # 폭행 관련 이벤트 (예시)
            "violence": "폭행",
            
            # 쓰러진 사람 관련 이벤트 (예시)
            "fallen": "쓰러진 사람",
            
            # (필요에 따라 다른 AI 탐지 클래스 이름과 한글 이벤트 이름을 여기에 추가)
        }




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
        self.video_widget = AspectRatioWidget()
        video_width = int(screen_rect.width() * 0.6)
        self.video_widget.setMinimumWidth(video_width)
        self.main_layout.addWidget(self.video_widget, 7)
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        self.main_layout.addWidget(control_panel, 3)
        self.log_open_button = QPushButton("로그 뷰어 및 동영상 재생")
        self.control_layout.addWidget(self.log_open_button)
        trigger_groupbox = QGroupBox("수동 이벤트 발생")
        button_layout = QVBoxLayout(trigger_groupbox)
        self.trigger_buttons = {}
        event_names = ["폭행", "화재", "쓰러진 사람", "실종자 발견", "무단 투기", "흡연자"]
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

    # def update_frame(self):
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    #         cv2.putText(frame, "NO SIGNAL", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    #     else:
    #         now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #         cv2.putText(frame, now_str, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #         cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    #         cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #         self.recorder.write_frame(frame)
    #         self.frame_buffer.append(self.preprocess_frame(frame))
    #         self.frame_counter += 1
    #     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb_image.shape
    #     qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
    #     pixmap = QPixmap.fromImage(qt_image)
    #     self.video_widget.video_label.setPixmap(pixmap)


    # update_frame 메서드를 update_image로 변경하거나 새로 만듦
    def update_image(self, frame):
        """VideoThread로부터 받은 프레임을 화면에 업데이트하는 슬롯"""
        # 이 메서드는 더 이상 self.cap.read()를 호출하지 않음
        
        # 프레임이 유효한지 확인
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            # 기존 update_frame에 있던 로직을 여기에 적용
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, now_str, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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

    # 30초 후 정지
    def stop_sound_on_channel(self, channel):
        """지정된 채널에서 재생 중인 사운드를 중지하는 슬롯"""
        if channel and channel.get_busy():
            channel.stop()
            print(f"Channel {channel} 재생이 30초 후 자동으로 중지되었습니다.")        

    def trigger_event(self, event_type, prob=None, is_auto=False):
        # timestamp를 메서드 시작 부분에서 생성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = prob if prob is not None else random.random()
        color = self.event_colors.get(event_type, "red")
        log_message = f"[{'자동' if is_auto else '수동'}] {event_type} 감지됨 (확률: {prob:.2f})"
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})

        if event_type in self.alert_sounds:
            # [수정] AlertDialog를 생성할 때 timestamp 변수를 전달합니다.
            dialog = AlertDialog(self, event_type, prob, timestamp)
            
            channel = pygame.mixer.find_channel()
            if channel:
                sound_to_play = self.alert_sounds[event_type]
                channel.play(sound_to_play, loops=-1)
                print(f"Channel {channel}에서 '{event_type}' 알람 시작.")
                dialog.alert_channel = channel
            else:
                print("경고: 모든 오디오 채널이 사용 중입니다.")
                dialog.alert_channel = None
            
            self.open_alerts.append(dialog)
            self.place_alert_dialog(dialog)
            dialog.show()

    def update_dialog_blink(self, dialog):
        if not hasattr(dialog, 'blinking') or not dialog.blinking:
            return
        if dialog.isActiveWindow():
            dialog.blinking = False
            dialog.setWindowTitle(dialog.original_title)
            if dialog.blink_timer: dialog.blink_timer.stop()
            return
        if "!" in dialog.windowTitle():
            dialog.setWindowTitle(dialog.original_title)
        else:
            dialog.setWindowTitle(f"! {dialog.original_title} !")

    def place_alert_dialog(self, dialog):
        # 1. 첫 번째 경고창이 표시될 시작 위치
        start_pos = QPoint(100, 20)
        # 2. 다음 경고창이 이전 창으로부터 얼마나 떨어질지 결정하는 대각선 간격
        cascade_offset = QPoint(110, 100)
        
        # --- [수정] 아래 변수들을 추가 및 변경합니다 ---
        # 3. 한 줄에 표시할 최대 창 개수
        cascade_per_row = 10 # 예시로 10개로 줄임 (화면에 맞게 조절)
        
        # 4. 다음 줄로 넘어갈 때의 좌표 변화량
        row_x_offset = -30   # 왼쪽으로 30px 이동
        row_y_offset = 60    # 아래쪽으로 60px 이동
        # --- 여기까지 ---

        # 현재 몇 번째 창인지 계산
        num_open_alerts = len(self.open_alerts) - 1
        # 몇 번째 줄, 몇 번째 칸에 위치해야 하는지 계산
        current_row = num_open_alerts // cascade_per_row
        current_col = num_open_alerts % cascade_per_row

        # --- [수정] 최종 위치 계산 로직 변경 ---
        # 1. 기본 대각선 위치 계산
        new_pos = start_pos + (current_col * cascade_offset)
        
        # 2. 줄바꿈에 따른 x, y 좌표 추가 이동
        #    - current_row가 1이면(두 번째 줄), x는 -30, y는 +60 만큼 추가로 이동합니다.
        new_pos.setX(new_pos.x() + (current_row * row_x_offset))
        new_pos.setY(new_pos.y() + (current_row * row_y_offset))
        
        # 최종 계산된 위치로 경고창을 이동
        dialog.move(new_pos)



    def resolve_alert(self, dialog_to_remove, event_type):

#============= --- 연결 고리 2 ----- [추가] 알람 종료 명령을 보내는 메서드 호출 ---
        self.send_stop_alarm_command(event_type)
#============= --- 연결 고리 2 -----=======================================        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"<font color='green'>{timestamp} - {event_type} 상황 종료됨</font>"
        self.log_browser.append(log_message)
        self.log_entries.append({"timestamp": timestamp, "message": f"{event_type} 상황 종료됨"})
        
        # --- [수정] 닫히는 다이얼로그에 연결된 채널의 소리만 중지 ---
        # dialog_to_remove 객체에 alert_channel 속성이 있는지 확인
        if hasattr(dialog_to_remove, 'alert_channel') and dialog_to_remove.alert_channel:
            dialog_to_remove.alert_channel.stop()
            print(f"Channel {dialog_to_remove.alert_channel}의 알람을 중지했습니다.")
        # --- 여기까지 ---

        if dialog_to_remove in self.open_alerts:
            self.open_alerts.remove(dialog_to_remove)
        
        # [삭제] 더 이상 check_and_stop_sound 함수를 호출할 필요가 없습니다.
        # QTimer.singleShot(100, self.check_and_stop_sound)


    def check_and_stop_sound(self):
        alert_dialogs_open = any(isinstance(d, AlertDialog) and d.isVisible() for d in self.open_alerts)
        if not alert_dialogs_open and self.alert_channel and self.alert_channel.get_busy():
            self.alert_channel.stop()

        # --- [추가] 알람[경고] 종료 명령을 보내는 클래스 메서드 ---
    def send_stop_alarm_command(self, event_type):
        """monitoringGUI에서 situationDetector로 '알람 종료' 명령을 전송합니다."""
        TARGET_IP = "192.168.0.86"
        TARGET_PORT = 2401

        SOURCE_ID = 4          # 보내는 곳: monitoringGUI (0x04)
        DESTINATION_ID = 2     # 받는 곳: situationDetector (0x02)
        COMMAND_STOP_ALARM = 1 # 명령: 알람 종료 (0x01)


        alarm_type_code = self.event_to_code_map.get(event_type, 255)
        if alarm_type_code == 255:
            print(f"경고: {event_type}에 해당하는 알람 종료 코드를 찾을 수 없습니다.")
            return

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2) 
                sock.connect((TARGET_IP, TARGET_PORT))
                
                # [수정] 'BBBB': 4개의 unsigned char 형식으로 데이터를 패킹
                payload = struct.pack('BBBB', SOURCE_ID, DESTINATION_ID, COMMAND_STOP_ALARM, alarm_type_code)
                
                sock.send(payload)
                print(f"알람 종료 명령 전송 성공: {payload.hex()}")

        except Exception as e:
            print(f"알람 종료 명령 전송 실패: {e}")
        # --- [추가] 알람[경고] 종료 명령을 보내는 클래스 메서드 ---


    def open_log_viewer(self):
        if self.log_viewer_dialog is None or not self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog = LogViewerDialog(self, self.log_entries, self.event_colors)
            self.log_viewer_dialog.show()
        self.log_viewer_dialog.activateWindow()

    def process_tcp_data(self, json_data):
        """수신된 TCP JSON 데이터를 처리하여 이벤트를 발생시키는 슬롯"""
        print("TCP 데이터 처리 시작:", json_data)
        timestamp = json_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # 1. 'detection' 딕셔너리를 가져옵니다.
        detection_features = json_data.get("detection", {})

        # 2. 딕셔너리 내부의 모든 값(탐지 결과 리스트)을 순회합니다.
        #    (예: "feat_detect_fire" 리스트, "feat_detect_violence" 리스트 등)
        for feature_results in detection_features.values():
            # feature_results가 리스트 형태일 경우에만 처리
            if isinstance(feature_results, list):
                # 3. 각 탐지 결과 리스트 내부의 개별 객체를 순회합니다.
                for detection in feature_results:
                    # AI 모델이 보낸 영문 클래스 이름을 가져옵니다.
                    eng_class_name = detection.get("class_name")
                    
                    # 4. __init__에서 정의한 맵을 사용해 한글 이벤트 이름으로 변환합니다.
                    event_type = self.class_name_map.get(eng_class_name)
                    
                    # 5. 매핑된 한글 이벤트 이름이 있을 경우에만 로그/경고를 생성합니다.
                    if event_type:
                        confidence = detection.get("confidence", 0.0)
                        # trigger_event를 호출하여 로그 기록, 팝업, 알람 등을 처리합니다.
                        self.trigger_event(event_type, confidence, is_auto=True)

        # 로그 뷰어가 열려있을 경우, 새로운 로그를 반영하여 업데이트합니다.
        if self.log_viewer_dialog and self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog.all_log_entries = self.log_entries
            self.log_viewer_dialog.request_logs_from_server()


    # closeEvent 수정
    def closeEvent(self, event):
        print("Closing application...")
        # self.timer.stop() # 기존 타이머는 없으므로 삭제 또는 주석 처리
        self.thread.stop() # 비디오 스레드를 안전하게 종료
        self.tcp_receiver.stop()

        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog):
                w.close()

        print("Stopping TCP receiver thread...")
        self.tcp_receiver.stop()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(('localhost', 2401))
        except Exception: pass
        self.tcp_receiver.wait(1000)
        if self.tcp_receiver.isRunning():
            self.tcp_receiver.terminate()
        print("TCP thread stopped.")
        print("Releasing resources...")
        # if self.cap and self.cap.isOpened():
        #     self.cap.release()
        self.recorder.stop()
        pygame.mixer.quit()
        print("Resources released.")

        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    # window.show() # 보통 크기
    window.showMaximized()  # 이 줄을 추가하거나, 주석을 해제해주세요.
    sys.exit(app.exec())