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
                               QHeaderView, QMessageBox, QSplitter, QWidget, QHBoxLayout,
                               QVBoxLayout)
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

# --- 바운딩 박스가 그려진 프레임을 보여줄 전용 팝업창 클래스 ---
class AnnotatedFrameDialog(QDialog):
    def __init__(self, annotated_frame: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("detecting_image")
        
        # [수정] 모달 설정을 False로 변경하여 독립적으로 만듭니다.
        self.setModal(False)
        # [추가] 창을 닫을 때 메모리에서 자동으로 삭제되도록 설정합니다.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        main_layout = QVBoxLayout(self)
        image_label = QLabel()

        # OpenCV 이미지(BGR)를 PyQt 이미지(QPixmap)로 변환
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        image_label.setPixmap(pixmap)
        main_layout.addWidget(image_label)

        # 닫기 버튼 추가
        close_button = QPushButton("닫기")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button)

        self.setLayout(main_layout)


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
        # [추가] 독립적으로 띄운 이미지 창들을 관리하기 위한 리스트
        self.open_frame_dialogs = []
        
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
        self.filtered_log_entries = response if response is not None else []
        self.current_page = 1
        self.update_table_display()
            
    def update_table_display(self):
        self.log_table.setSortingEnabled(False)
        self.log_table.setRowCount(0)
        
        start_index = (self.current_page - 1) * self.items_per_page
        paginated_entries = self.filtered_log_entries[start_index : start_index + self.items_per_page]

        for row, entry in enumerate(paginated_entries):
            self.log_table.insertRow(row)
            
            try:
                raw_result_json = json.loads(entry.get('raw_result', '{}'))
                event_meta = raw_result_json.get('event_video_meta', {})
                events = {d['class_name'] for v in event_meta.values() if isinstance(v, list) for d in v if isinstance(d, dict) and 'class_name' in d}
                event_type_str = ", ".join(sorted(list(events))) or entry.get('class_name', 'Info')
            except (json.JSONDecodeError, AttributeError):
                event_type_str = entry.get('class_name', "Data format error")
            
            ts_obj = entry.get('timestamp')
            ts_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts_obj, datetime) else str(ts_obj)
            
            self.log_table.setItem(row, 0, QTableWidgetItem(ts_str))
            self.log_table.setItem(row, 1, QTableWidgetItem(event_type_str))
            
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(5, 0, 5, 0)
            button_layout.setSpacing(5)

            play_button = QPushButton("재생")
            play_button.clicked.connect(lambda _, cid=entry.get('class_id'): self.play_video_for_log(cid))

            frame_button = QPushButton("이미지")
            frame_button.clicked.connect(lambda _, cid=entry.get('class_id'), raw=entry.get('raw_result'): self.show_annotated_frame_dialog(cid, raw))
            
            button_layout.addWidget(play_button)
            button_layout.addWidget(frame_button)
            
            self.log_table.setCellWidget(row, 2, button_container)
            
        self.log_table.setSortingEnabled(True)
        self.update_pagination_controls()

    def play_video_for_log(self, class_id: int):
        self.video_display.setText("서버로부터 영상 다운로드 중...")
        QApplication.processEvents()
        
        video_path = self._get_video_as_temp_file(class_id)
        if video_path:
            self._play_local_video(video_path)
        else:
            self.video_display.setText("영상을 가져오는 데 실패했습니다.")

    def _play_local_video(self, path: str):
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
            self.video_display.setText(f"Could not open video file:\n{path}")

    def update_video_frame(self):
        if not self.playing_video or not self.video_capture.isOpened():
            return
        ret, frame = self.video_capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(qt_image))
        else:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
            self.playing_video = False
            self.video_display.clear()
            self.video_display.setText("재생할 동영상을 선택하세요.")
            self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def show_annotated_frame_dialog(self, class_id: int, raw_result_str: str):
        if not class_id or not raw_result_str:
            QMessageBox.warning(self, "오류", "표시할 데이터가 없습니다.")
            return

        bboxes = self._parse_bboxes_from_raw(raw_result_str)
        
        video_path = self._get_video_as_temp_file(class_id)
        if not video_path:
            QMessageBox.critical(self, "오류", "서버에서 영상을 가져오는 데 실패했습니다.")
            return

        annotated_frame = self._extract_and_draw_frame(video_path, bboxes)
        if annotated_frame is None:
            QMessageBox.warning(self, "오류", "15초 지점의 프레임을 추출할 수 없습니다.\n(영상이 15초보다 짧을 수 있습니다)")
            return
            
        dialog = AnnotatedFrameDialog(annotated_frame, self)
        
        # [수정] 팝업창이 사라지는 것을 막기 위해 리스트에 추가하여 참조를 유지합니다.
        self.open_frame_dialogs.append(dialog)
        # [수정] exec() 대신 show()를 호출하여 독립적인 창으로 띄웁니다.
        dialog.show()

    def _parse_bboxes_from_raw(self, raw_result_str: str) -> list:
        bboxes = []
        try:
            raw_json = json.loads(raw_result_str)
            event_meta = raw_json.get('event_video_meta', {})
            for result_list in event_meta.values():
                if isinstance(result_list, list):
                    for detection in result_list:
                        if isinstance(detection, dict) and 'bbox' in detection:
                            bboxes.append(detection['bbox'])
        except json.JSONDecodeError:
            print("Error: Failed to parse raw_result JSON.")
        return bboxes

    def _get_video_as_temp_file(self, class_id: int) -> str | None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                request = {'command': 'get_video_path', 'params': {'class_id': class_id}}
                if not send_msg(sock, request):
                    raise ConnectionError("Failed to send video request")

                initial_response = recv_msg(sock)
                if not initial_response or 'error' in initial_response:
                    return None
                
                if initial_response.get('status') == 'streaming_start':
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
                    
                    return self.temp_video_path
        except (socket.error, ConnectionError):
            return None

    def _extract_and_draw_frame(self, video_path: str, bboxes: list) -> np.ndarray | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = 30 if fps == 0 else fps
            target_frame_pos = int(15 * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_pos)
            ret, frame = cap.read()

            if ret:
                for bbox in bboxes:
                    x1, y1 = int(bbox['x1']), int(bbox['y1'])
                    x2, y2 = int(bbox['x2']), int(bbox['y2'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                return frame
            return None
        finally:
            cap.release()

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

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_size_set:
            self.main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
            self._initial_size_set = True

    def closeEvent(self, event):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
        
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                self.temp_video_path = None
            except OSError as e:
                print(f"Error deleting temporary file: {e}")
        
        # [추가] 로그 뷰어를 닫을 때 열려있는 모든 이미지 창도 함께 닫습니다.
        for dialog in self.open_frame_dialogs:
            dialog.close()

        super().closeEvent(event)