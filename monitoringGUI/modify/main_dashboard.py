# main_dashboard.py

import sys
import os
import time
from datetime import datetime
from collections import deque
import random
import json
import socket
import threading
import select
import struct
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                               QTextBrowser, QDialog, QMessageBox, QPushButton, QListWidgetItem)
from PyQt6.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

from log_viewer import LogViewerDialog

# --- 상수 정의 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_UI_PATH = os.path.join(BASE_DIR, "main_dashboard.ui")

# 네트워크 설정
TCP_SERVER_HOST = '192.168.0.86'
TCP_SERVER_PORT = 2401
VIDEO_STREAM_URL = "http://192.168.0.180:5000/stream?src=0"
RECONNECT_DELAY_SECONDS = 5

# 미디어 및 레코딩 설정
ALERT_SOUND_DIR = os.path.join(BASE_DIR, "alert")
RECORDING_DIR = os.path.join(BASE_DIR, "recordings")
VIDEO_CHUNK_SECONDS = 10
VIDEO_FPS = 20
VIDEO_FRAME_SIZE = (1280, 720)

# AI 및 이벤트 설정
FRAME_BUFFER_SIZE = 16
EVENT_PROB_THRESHOLD = 0.6

# --- UI 클래스 로드 ---
form_class = uic.loadUiType(MAIN_UI_PATH)[0]


# --- 헬퍼 클래스 ---
class AspectRatioWidget(QWidget):
    def __init__(self, parent: QWidget = None, aspect_ratio: float = 16.0/9.0):
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        if w / h > self.aspect_ratio:
            new_w = int(h * self.aspect_ratio)
            self.video_label.setGeometry((w - new_w) // 2, 0, new_w, h)
        else:
            new_h = int(w / self.aspect_ratio)
            self.video_label.setGeometry(0, (h - new_h) // 2, w, new_h)

class RollingRecorder:
    def __init__(self):
        self.output_dir = RECORDING_DIR
        self.chunk_seconds = VIDEO_CHUNK_SECONDS
        self.fps = VIDEO_FPS
        self.frame_size = VIDEO_FRAME_SIZE
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.file_deque = deque(maxlen=6)
        self.out = None
        self.frame_count = 0
        self.frames_per_chunk = int(self.chunk_seconds * self.fps)
        os.makedirs(self.output_dir, exist_ok=True)

    def _start_new_chunk(self):
        if len(self.file_deque) == self.file_deque.maxlen:
            try:
                os.remove(self.file_deque.popleft())
            except OSError as e:
                print(f"Failed to delete old chunk: {e}")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chunk_path = os.path.join(self.output_dir, f"chunk_{timestamp}.mp4")
        self.out = cv2.VideoWriter(chunk_path, self.fourcc, self.fps, self.frame_size)
        self.file_deque.append(chunk_path)

    def write_frame(self, frame: np.ndarray):
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
    def __init__(self, parent: QWidget, event_type: str, prob: float, timestamp: str):
        super().__init__(parent)
        self.parent_dashboard = parent
        self.event_type = event_type
        
        self.setWindowTitle(f"{event_type} 발생")
        self.setFixedSize(300, 170)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.CustomizeWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        
        layout = QVBoxLayout(self)
        message_text = f"{event_type} 감지됨! (확률: {prob:.2f})\n\n발생 시각: {timestamp}"
        message_label = QLabel(message_text)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        confirm_button = QPushButton("확인")
        confirm_button.clicked.connect(self.confirm_and_close)
        
        layout.addWidget(message_label)
        layout.addWidget(confirm_button)

    def confirm_and_close(self):
        reply = QMessageBox.question(self, "확인", "경고를 종료하시겠습니까?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.parent_dashboard.resolve_alert(self, self.event_type)
            self.accept()

class TcpReceiver(QThread):
    data_received = pyqtSignal(dict)

    def __init__(self, command_queue: Queue, parent: QWidget = None):
        super().__init__(parent)
        self.command_queue = command_queue
        self.running = True
        self.server_host = TCP_SERVER_HOST
        self.server_port = TCP_SERVER_PORT
        self.client_socket = None

    def run(self):
        while self.running:
            connected = False
            while not connected and self.running:
                try:
                    print(f"Connecting to server {self.server_host}:{self.server_port}...")
                    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.client_socket.connect((self.server_host, self.server_port))
                    connected = True
                    print("Server connected.")
                except Exception as e:
                    print(f"Connection failed: {e}")
                    self.client_socket.close()
                    time.sleep(RECONNECT_DELAY_SECONDS)

            if not self.running:
                break

            buffer = ""
            decoder = json.JSONDecoder()
            
            while self.running and connected:
                try:
                    readable, writable, exceptional = select.select(
                        [self.client_socket], [self.client_socket], [self.client_socket], 0.1)

                    if exceptional:
                        print("Socket connection error.")
                        connected = False
                        continue

                    if readable:
                        data = self.client_socket.recv(4096)
                        if not data:
                            print("Server connection closed.")
                            connected = False
                            continue
                        
                        buffer += data.decode('utf-8', errors='ignore')
                        while True:
                            try:
                                json_data, end_pos = decoder.raw_decode(buffer)
                                self.data_received.emit(json_data)
                                buffer = buffer[end_pos:].lstrip()
                            except json.JSONDecodeError:
                                break
                    
                    if writable and not self.command_queue.empty():
                        command = self.command_queue.get_nowait()
                        self.client_socket.sendall(command)
                        print(f"Command sent to server: {command.hex()}")

                except (socket.error, Empty) as e:
                    print(f"Data exchange error: {e}")
                    connected = False
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    connected = False

            if self.client_socket:
                self.client_socket.close()

        print("TCP Receiver thread stopped.")

    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except OSError:
                pass
        print("Stopping TCP client thread...")


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.video_source = VIDEO_STREAM_URL
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
            else:
                print(f"Stream disconnected. Reconnecting in {RECONNECT_DELAY_SECONDS} seconds...")
                cap.release()
                time.sleep(RECONNECT_DELAY_SECONDS)
                cap = cv2.VideoCapture(self.video_source)
        cap.release()
        print("VideoThread stopped.")

    def stop(self):
        self._run_flag = False
        self.wait()


# --- 메인 대시보드 클래스 ---
class PatrolDashboard(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self._init_settings()
        self._init_custom_widgets()
        self._init_pygame()
        self._init_threads()
        self._connect_signals()

    def _init_settings(self):
        self.log_viewer_dialog = None
        self.command_queue = Queue()
        self.open_alerts = []
        self.log_entries = []
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.recorder = RollingRecorder()
        
        # [수정] '연기 감지' 색상 제거
        self.event_colors = {
            "화재": "red", 
            "폭행": "orange", 
            "쓰러진 사람": "purple",
            "실종자 발견": "blue", 
            "무단 투기": "gray", 
            "흡연자": "darkgray",
        }
        
        # [수정] '연기 감지' 이벤트 코드 제거
        self.event_to_code_map = {
            "화재": 0, 
            "쓰러진 사람": 1, 
            "흡연자": 2,
            "무단 투기": 3, 
            "폭행": 4, 
            "실종자 발견": 255,
        }

    def _init_custom_widgets(self):
        self.video_widget = AspectRatioWidget()
        self.video_container_layout.removeWidget(self.video_widget_placeholder)
        self.video_widget_placeholder.deleteLater()
        self.main_layout.insertWidget(0, self.video_widget, 7)

    def _init_pygame(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(16)
        self.alert_sounds = {}
        
        # [수정] '연기 감지' 사운드 파일 제거
        sound_map = {
            "화재": "fire.mp3", # 연기 감지 시에도 이 사운드가 사용됩니다.
            "폭행": "violence.mp3",
            "쓰러진 사람": "faint.mp3", 
            "실종자 발견": "missing_person.mp3",
        }
        print("--- Loading alert sounds ---")
        for event, filename in sound_map.items():
            path = os.path.join(ALERT_SOUND_DIR, filename)
            if os.path.exists(path):
                self.alert_sounds[event] = pygame.mixer.Sound(path)
                print(f"'{path}' loaded successfully.")
            else:
                print(f"Warning: Sound file not found - '{path}'")
        print("--- Sound loading complete ---")
    
    def _init_threads(self):
        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        self.tcp_receiver = TcpReceiver(self.command_queue, self)
        self.tcp_receiver.data_received.connect(self.process_tcp_data)
        self.tcp_receiver.start()

    def _connect_signals(self):
        self.log_open_button.clicked.connect(self.open_log_viewer)
        self.btn_resolve_alert.clicked.connect(self._on_resolve_button_clicked)
        self.alert_list_widget.itemClicked.connect(self._on_alert_item_clicked)
        
    def update_image(self, frame: np.ndarray):
        if frame is None:
            frame = np.zeros((VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        else:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, now_str, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.recorder.write_frame(frame)
            
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_widget.video_label.setPixmap(pixmap)

    def trigger_event(self, event_type: str, prob: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self.event_colors.get(event_type, "red")
        log_message = f" {event_type} detected (probability: {prob:.2f})"
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})
        
        if event_type in self.alert_sounds:
            dialog = AlertDialog(self, event_type, prob, timestamp)
            channel = pygame.mixer.find_channel(True)
            if channel:
                sound = self.alert_sounds[event_type]
                channel.play(sound, loops=-1)
                dialog.alert_channel = channel
            
            self.open_alerts.append(dialog)
            self._place_alert_dialog(dialog)
            dialog.show()

            list_item_text = f"[{event_type}] {timestamp}"
            list_item = QListWidgetItem(list_item_text, self.alert_list_widget)
            list_item.setData(Qt.ItemDataRole.UserRole, dialog)
    
    def _place_alert_dialog(self, dialog: QDialog):
        start_pos = QPoint(100, 20)
        offset = QPoint(110, 100)
        per_row = 10
        row_offset = QPoint(-30, 60)
        
        num = len(self.open_alerts) - 1
        row, col = num // per_row, num % per_row
        
        new_pos = start_pos + (col * offset)
        new_pos.setX(new_pos.x() + (row * row_offset.x()))
        new_pos.setY(new_pos.y() + (row * row_offset.y()))
        dialog.move(new_pos)

    def resolve_alert(self, dialog_to_remove: QDialog, event_type: str):
        self.send_stop_alarm_command(event_type)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"<font color='green'>{timestamp} - {event_type} event resolved</font>"
        self.log_browser.append(log_message)
        
        if hasattr(dialog_to_remove, 'alert_channel'):
            dialog_to_remove.alert_channel.stop()
            
        if dialog_to_remove in self.open_alerts:
            self.open_alerts.remove(dialog_to_remove)

        for i in range(self.alert_list_widget.count()):
            item = self.alert_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == dialog_to_remove:
                self.alert_list_widget.takeItem(i)
                break

    def _on_resolve_button_clicked(self):
        selected_item = self.alert_list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "알림", "목록에서 확인할 알림을 먼저 선택하세요.")
            return
        
        dialog = selected_item.data(Qt.ItemDataRole.UserRole)
        dialog.confirm_and_close()

    def _on_alert_item_clicked(self, item: QListWidgetItem):
        dialog = item.data(Qt.ItemDataRole.UserRole)
        if dialog:
            dialog.activateWindow()

    def send_stop_alarm_command(self, event_type: str):
        code = self.event_to_code_map.get(event_type)
        if code is None:
            return
        payload = struct.pack('BBBB', 4, 2, 1, code)
        self.command_queue.put(payload)
        print(f"Queued stop alarm command for {event_type}: {payload.hex()}")

    def open_log_viewer(self):
        if self.log_viewer_dialog is None or not self.log_viewer_dialog.isVisible():
            self.log_viewer_dialog = LogViewerDialog(self)
            self.log_viewer_dialog.show()
        self.log_viewer_dialog.activateWindow()

    def process_tcp_data(self, data: dict):
        print("Processing TCP data:", data)
        detection_features = data.get("detection", {})
        for feature, results in detection_features.items():
            if not isinstance(results, list) or not results:
                continue
            
            obj_list = results[:-1] if isinstance(results[-1], dict) and "detection_count" in results[-1] else results
            if not obj_list:
                continue

            # [수정] 'feat_detect_smoke'가 더 이상 별도 이벤트가 아니므로 핸들러에서 제거
            handlers = {
                "feat_detect_fire": lambda: self.handle_fire_event(obj_list),
                "feat_detect_fall": lambda: self.handle_generic_event(obj_list, "쓰러진 사람", "score_event"),
                "feat_detect_violence": lambda: self.handle_violence_event(obj_list),
                "feat_detect_missing_person": lambda: self.handle_missing_person_event(obj_list),
                "feat_detect_trash": lambda: self.handle_trash_event(obj_list)
            }
            if feature in handlers:
                handlers[feature]()

    # [수정] handle_fire_and_smoke_event 함수의 이름을 바꾸고 로직을 통합
    def handle_fire_event(self, results: list):
        """'feat_detect_fire' 결과를 분석하여 화재 이벤트를 발생시킵니다. (연기 포함)"""
        for item in results:
            confidence = item.get("confidence", 0.0)
            if confidence > EVENT_PROB_THRESHOLD:
                # class_name이 무엇이든(연기, 불꽃 등) '화재'로 간주하고 이벤트 발생
                self.trigger_event("화재", confidence)

    def handle_generic_event(self, results: list, event_name: str, conf_key: str, target_class: str = None):
        for item in results:
            if target_class and item.get("class_name") != target_class:
                continue
            confidence = item.get(conf_key, 0.0)
            if confidence > EVENT_PROB_THRESHOLD:
                self.trigger_event(event_name, confidence)

    def handle_violence_event(self, results: list):
        for item in results:
            if item.get("class_name") == "Fight":
                confidence = item.get("confidence", 0.0)
                if confidence > EVENT_PROB_THRESHOLD:
                    self.trigger_event("폭행", confidence)

    def handle_missing_person_event(self, results: list):
        for item in results:
            confidence = item.get("confidence", 0.0)
            if confidence > EVENT_PROB_THRESHOLD:
                self.trigger_event("실종자 발견", confidence)

    def handle_trash_event(self, results: list):
        class_names = {item.get("class_name") for item in results}
        if "human" in class_names and "trash" in class_names:
            max_conf = max(item.get("confidence", 0.0) for item in results)
            if max_conf > EVENT_PROB_THRESHOLD:
                self.trigger_event("무단 투기", max_conf)

    def closeEvent(self, event):
        print("Closing application...")
        self.video_thread.stop()
        self.tcp_receiver.stop()
        
        for widget in QApplication.instance().topLevelWidgets():
            if isinstance(widget, QDialog):
                widget.close()
        
        self.tcp_receiver.wait(1000)
        self.recorder.stop()
        pygame.mixer.quit()
        print("Resources released.")
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.showMaximized()
    sys.exit(app.exec())