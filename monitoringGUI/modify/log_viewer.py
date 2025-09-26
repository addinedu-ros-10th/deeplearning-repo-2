# log_viewer.py

import sys
import os
import time
from datetime import datetime
import json
import socket
import pickle
import struct
import tempfile

import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QMessageBox, QSplitter)
from PyQt6.QtCore import QTimer, QDate, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

# --- 상수 정의 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_VIEWER_UI_PATH = os.path.join(BASE_DIR, "log_viewer.ui")

# 로그 및 비디오 요청 서버
LOG_SERVER_HOST = '192.168.0.23'
LOG_SERVER_PORT = 3401

# --- 네트워크 유틸리티 함수 ---
def send_msg(sock: socket.socket, data: dict) -> bool:
    try:
        pickled_data = pickle.dumps(data)
        msg = struct.pack('L', len(pickled_data)) + pickled_data
        sock.sendall(msg)
        return True
    except (OSError, pickle.PicklingError) as e:
        print(f"Message sending failed: {e}")
        return False

def recv_msg(sock: socket.socket) -> dict | None:
    try:
        packed_len = sock.recv(struct.calcsize('L'))
        if not packed_len:
            return None
        msg_len = struct.unpack('L', packed_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet:
                return None
            data += packet
        return pickle.loads(data)
    except (OSError, pickle.UnpicklingError, struct.error) as e:
        print(f"Message receiving failed: {e}")
        return None

# --- 로그 뷰어 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        uic.loadUi(LOG_VIEWER_UI_PATH, self)
        
        self._init_settings()
        self._setup_ui_details()
        self.connect_signals()
        
        self.request_logs_from_server()

    def _init_settings(self):
        self.server_host = LOG_SERVER_HOST
        self.server_port = LOG_SERVER_PORT
        self.filtered_log_entries = []
        self.video_capture = None
        self.temp_video_path = None
        self.playing_video = False
        self.current_page = 1
        self.items_per_page = 20
        self._initial_size_set = False
        
    def _setup_ui_details(self):
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        self.event_type_map = {
            "화재": ("detect_fire", self.cb_fire),
            "폭행": ("assault", self.cb_assault),
            "누워있는 사람": ("fallen", self.cb_fallen),
            "실종자": ("missing", self.cb_missing),
            "무단 투기": ("dumping", self.cb_dumping),
            "흡연자": ("smoke", self.cb_smoking),
        }
        self.start_date_edit.setDate(QDate.currentDate().addDays(-1))
        self.end_date_edit.setDate(QDate.currentDate())

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)
        
    def send_request_to_server(self, command: str, params: dict) -> dict | None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                request = {'command': command, 'params': params}
                if not send_msg(sock, request):
                    raise ConnectionError("Failed to send request to server")
                return recv_msg(sock)
        except socket.error as e:
            QMessageBox.critical(self, "Server Connection Error", f"Could not connect to server: {e}")
            return None

    def request_logs_from_server(self):
        params = {
            'start_date': self.start_date_edit.date().toString("yyyy-MM-dd"),
            'end_date': self.end_date_edit.date().toString("yyyy-MM-dd"),
            'orderby_latest': self.orderby_combo.currentText() == "최신순",
            'event_types': [
                name_in_db for name, (name_in_db, checkbox) in self.event_type_map.items() 
                if checkbox.isChecked()
            ]
        }
        
        response = self.send_request_to_server('get_logs', params)
        
        if response is None:
            print("Warning: Received no response from server for 'get_logs'.")
            self.filtered_log_entries = []
        elif not isinstance(response, list):
            print(f"Warning: Expected a list from server, but got {type(response)}. Response: {response}")
            self.filtered_log_entries = []
        else:
            self.filtered_log_entries = response

        self.current_page = 1
        self.update_table_display()
        
    def play_video_for_log(self, class_id: int):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                request = {'command': 'get_video_path', 'params': {'class_id': class_id}}
                if not send_msg(sock, request):
                    raise ConnectionError("Failed to send video request")

                initial_response = recv_msg(sock)
                if not initial_response or 'error' in initial_response:
                    error_msg = initial_response.get('error', 'Unknown error') if initial_response else "No response"
                    self.video_display.setText(f"Video request failed: {error_msg}")
                    return
                
                if initial_response.get('status') == 'streaming_start':
                    self._receive_and_play_stream(sock)
                else:
                    self.video_display.setText("Received unknown response from server.")

        except (socket.error, ConnectionError) as e:
            QMessageBox.critical(self, "Server Communication Error", f"Error during video streaming: {e}")

    def _receive_and_play_stream(self, sock: socket.socket):
        buffer = b''
        while True:
            chunk = sock.recv(4096)
            if not chunk or b'DONE' in chunk:
                if b'DONE' in chunk:
                    final_chunk, _ = chunk.split(b'DONE', 1)
                    buffer += final_chunk
                break
            buffer += chunk
        
        if self.temp_video_path and os.path.exists(self.temp_video_path):
             os.remove(self.temp_video_path)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            self.temp_video_path = temp_file.name
        
        with open(self.temp_video_path, 'wb') as f:
            f.write(buffer)
        
        print(f"Stream received and saved to {self.temp_video_path}")
        self._play_local_video(self.temp_video_path)
            
    def update_table_display(self):
        self.log_table.setSortingEnabled(False)
        self.log_table.setRowCount(0)
        
        start_index = (self.current_page - 1) * self.items_per_page
        paginated_entries = self.filtered_log_entries[start_index : start_index + self.items_per_page]

        for row, entry in enumerate(paginated_entries):
            self.log_table.insertRow(row)
            
            try:
                raw_result = json.loads(entry.get('raw_result', '{}'))
                events = {d['class_name'] for v in raw_result.values() if isinstance(v, list) for d in v if 'class_name' in d}
                event_type_str = ", ".join(sorted(list(events))) or "Info"
            except (json.JSONDecodeError, AttributeError):
                event_type_str = "Data format error"
            
            ts_obj = entry.get('timestamp')
            ts_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts_obj, datetime) else str(ts_obj)
            
            self.log_table.setItem(row, 0, QTableWidgetItem(ts_str))
            self.log_table.setItem(row, 1, QTableWidgetItem(event_type_str))
            
            play_button = QPushButton("Play")
            play_button.clicked.connect(lambda _, cid=entry.get('class_id'): self.play_video_for_log(cid))
            self.log_table.setCellWidget(row, 2, play_button)
            
        self.log_table.setSortingEnabled(True)
        self.update_pagination_controls()
    
    def update_pagination_controls(self):
        total_items = len(self.filtered_log_entries)
        total_pages = max(1, (total_items + self.items_per_page - 1) // self.items_per_page)
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

    def _play_local_video(self, path: str):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture:
                self.video_capture.release()

        self.video_capture = cv2.VideoCapture(path)
        if self.video_capture.isOpened():
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.playing_video = True
            self.video_timer.start(int(1000 / fps))
        else:
            self.playing_video = False
            self.video_display.setText(f"Could not open video file:\n{path}")

    def update_video_frame(self):
        """비디오 재생 중 프레임을 지속적으로 업데이트"""
        if not self.playing_video or not self.video_capture.isOpened():
            return

        ret, frame = self.video_capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(qt_image))
        else:
            # 영상 재생이 끝나면 (ret is False) 타이머와 캡쳐를 중지
            self.video_timer.stop()
            self.video_capture.release()
            self.playing_video = False
            
            # [추가] 영상 재생이 완료된 후 화면을 초기 상태로 되돌립니다.
            self.video_display.clear() # 현재 표시된 마지막 프레임 제거
            self.video_display.setText("재생할 동영상을 선택하세요.") # 초기 안내 문구 설정
            self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter) # 텍스트 가운데 정렬

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_size_set:
            self.main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
            self._initial_size_set = True

    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                print(f"Temporary file deleted: {self.temp_video_path}")
                self.temp_video_path = None
            except OSError as e:
                print(f"Error deleting temporary file: {e}")
        
        super().closeEvent(event)