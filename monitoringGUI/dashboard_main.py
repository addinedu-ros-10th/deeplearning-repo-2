
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
        self.blink_timer = None
        layout = QVBoxLayout(self)
        message = QLabel(f"{event_type} 감지됨! (확률: {prob:.2f})")
        layout.addWidget(message)
        stop_button = QPushButton("확인")
        layout.addWidget(stop_button)
        stop_button.clicked.connect(self.confirm_stop)

    def confirm_stop(self):
        if QMessageBox.question(self, "확인", "경고를 종료하시겠습니까?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.parent_dashboard.resolve_alert(self, self.event_type)
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
        self.host = 'localhost'  # 수신할 호스트 주소
        self.port = 2401          # 수신할 포트 번호

    def run(self):
        """스레드가 시작될 때 실행되는 메인 루프"""
        try:
            # 서버 소켓 생성 및 설정
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 주소 재사용 옵션 설정
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            # 타임아웃을 설정하여 self.running 플래그를 주기적으로 확인할 수 있게 함
            server_socket.settimeout(1.0)
            print(f"TCP Receiver listening on {self.host}:{self.port}")

            while self.running:
                try:
                    # 클라이언트 연결 대기
                    client_socket, addr = server_socket.accept()
                    with client_socket:
                        print(f"Accepted connection from {addr}")
                        full_data = b""
                        while True:
                            # 1024 바이트씩 데이터 수신
                            data = client_socket.recv(1024)
                            if not data:
                                break
                            full_data += data
                        
                        # 수신된 데이터가 있으면 처리
                        if full_data:
                            try:
                                # 수신된 바이트 데이터를 UTF-8 문자열로 디코딩
                                json_str = full_data.decode('utf-8')
                                # JSON 문자열을 파이썬 딕셔너리로 변환
                                json_data = json.loads(json_str)
                                # 파싱된 데이터를 시그널에 담아 발생시킴
                                self.data_received.emit(json_data)
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"Error processing received data: {e}")

                except socket.timeout:
                    # 타임아웃 발생 시 루프를 계속 진행하여 self.running 상태를 확인
                    continue
                except Exception as e:
                    print(f"TCP receiver error: {e}")
        
        finally:
            # 스레드 종료 시 서버 소켓 정리
            server_socket.close()
            print("TCP Receiver stopped.")

    def stop(self):
        """스레드를 안전하게 종료하기 위한 메서드"""
        self.running = False
        print("Stopping TCP receiver thread...")

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
        pygame.mixer.init()
        self.alert_sound, self.alert_channel, self.alert_active = None, None, False
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


        self.event_colors = {"화재":"red", "폭행":"orange", "누워있는 사람":"purple", "실종자":"cyan", "무단 투기":"lightgray", "흡연자":"lightgray"}
        if os.path.exists("alert.mp3"):
            self.alert_sound = pygame.mixer.Sound("alert.mp3")
            self.alert_channel = pygame.mixer.Channel(0)
        
        self.tcp_receiver = TcpReceiver(self)
        self.tcp_receiver.data_received.connect(self.process_tcp_data)
        self.tcp_receiver.start()

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

    def trigger_event(self, event_type, prob=None, is_auto=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = prob if prob is not None else random.random()
        color = self.event_colors.get(event_type, "red")
        log_message = f"[{'자동' if is_auto else '수동'}] {event_type} 감지됨 (확률: {prob:.2f})"
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})

        # [수정] 아래 부분을 변경합니다.
        if event_type in ["폭행", "화재", "누워있는 사람", "실종자 발견"]:
            # 현재 경고음이 울리고 있지 않을 때만 새로 재생
            if self.alert_sound and self.alert_channel and not self.alert_channel.get_busy():
                # maxtime=5000 옵션으로 5초 동안만 재생
                self.alert_channel.play(self.alert_sound, maxtime=5000)
            
                
            dialog = AlertDialog(self, event_type, prob)
            self.open_alerts.append(dialog)
            self.place_alert_dialog(dialog)
            dialog.blink_timer = QTimer(dialog)
            dialog.blinking = True
            dialog.original_title = dialog.windowTitle()
            dialog.blink_timer.timeout.connect(lambda d=dialog: self.update_dialog_blink(d))
            dialog.blink_timer.start(500)
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
        start_pos = QPoint(100, 20)
        cascade_offset = QPoint(40, 40)
        row_y_offset = 60
        cascade_per_row = 10
        num_open_alerts = len(self.open_alerts) - 1
        current_row = num_open_alerts // cascade_per_row
        current_col = num_open_alerts % cascade_per_row
        new_pos = start_pos + (current_col * cascade_offset)
        new_pos.setY(new_pos.y() + (current_row * row_y_offset))
        dialog.move(new_pos)

    def resolve_alert(self, dialog_to_remove, event_type):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"<font color='green'>{timestamp} - {event_type} 상황 종료됨</font>"
        self.log_browser.append(log_message)
        self.log_entries.append({"timestamp": timestamp, "message": f"{event_type} 상황 종료됨"})
        if dialog_to_remove in self.open_alerts:
            self.open_alerts.remove(dialog_to_remove)
        QTimer.singleShot(100, self.check_and_stop_sound)

    def check_and_stop_sound(self):
        alert_dialogs_open = any(isinstance(d, AlertDialog) and d.isVisible() for d in self.open_alerts)
        if not alert_dialogs_open and self.alert_channel and self.alert_channel.get_busy():
            self.alert_channel.stop()

    def open_log_viewer(self):
        if self.log_viewer_dialog is None or not self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog = LogViewerDialog(self, self.log_entries, self.event_colors)
            self.log_viewer_dialog.show()
        self.log_viewer_dialog.activateWindow()

    def process_tcp_data(self, json_data):
        print("TCP 데이터 처리 시작:", json_data)
        timestamp = json_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for detection in json_data.get("detection", []):
            event_type = detection["class_name"]
            confidence = detection["confidence"]
            self.trigger_event(event_type, confidence, is_auto=True)
        if self.log_viewer_dialog and self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog.all_log_entries = self.log_entries
            self.log_viewer_dialog.request_logs_from_server()

    # def closeEvent(self, event):
    #     print("Closing application...")
    #     self.timer.stop()
    #     for w in QApplication.topLevelWidgets():
    #         if isinstance(w, QDialog):
    #             w.close()

    # closeEvent 수정
    def closeEvent(self, event):
        print("Closing application...")
        # self.timer.stop() # 기존 타이머는 없으므로 삭제 또는 주석 처리
        self.thread.stop() # 비디오 스레드를 안전하게 종료
        
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
        if self.cap and self.cap.isOpened():
            self.cap.release()
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