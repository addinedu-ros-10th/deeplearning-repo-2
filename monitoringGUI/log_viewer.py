# client/log_viewer.py

import sys
import os
import cv2
import json
import socket
import pickle
import struct
from datetime import datetime
import tempfile

from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QMessageBox, QSplitter)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap

# --- TCP 통신 프로토콜 헬퍼 함수 ---
def send_msg(sock, data):
    try:
        pickled_data = pickle.dumps(data)
        msg = struct.pack('L', len(pickled_data)) + pickled_data
        sock.sendall(msg)
        return True
    except Exception as e:
        print(f"메시지 전송 실패: {e}")
        return False

def recv_msg(sock):
    try:
        packed_len = sock.recv(struct.calcsize('L'))
        if not packed_len: return None
        msg_len = struct.unpack('L', packed_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet: return None
            data += packet
        return pickle.loads(data)
    except Exception as e:
        print(f"메시지 수신 실패: {e}")
        return None

# --- 로그 뷰어 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_ui_and_find_widgets()

        self.server_host = "192.168.0.23"
        self.server_port = 3401
        self.filtered_log_entries = []
        
        self.video_capture = None
        self.playing_video = False
        self.current_page = 1
        self.items_per_page = 20
        self._initial_size_set = False

        self.connect_signals()
        self.request_logs_from_server()

    def send_request_to_server(self, command, params):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                request = {'command': command, 'params': params}
                if not send_msg(sock, request):
                    raise ConnectionError("서버에 요청 전송 실패")
                response = recv_msg(sock)
                if response is None:
                    raise ConnectionError("서버로부터 응답 수신 실패")
                return response
        except socket.error as e:
            QMessageBox.critical(self, "서버 연결 오류", f"서버에 연결할 수 없습니다: {e}")
            return None

    def request_logs_from_server(self):
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        orderby_latest = self.orderby_combo.currentText() == "최신순"
        
        detection_types = [
            name for name, checkbox in self.event_type_map.values() 
            if checkbox.isChecked()
        ]

        params = {
            'start_date': start_date,
            'end_date': end_date,
            'orderby_latest': orderby_latest,
            'event_types': detection_types
        }
        
        response = self.send_request_to_server('get_logs', params)
        self.filtered_log_entries = response if response is not None else []
        self.current_page = 1
        self.update_table_display()
        
    def play_video_for_log(self, class_id):
        # 스트리밍은 요청/응답 방식이 다르므로 소켓을 직접 관리
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                
                # 1. 영상 요청 보내기 (pickle)
                request = {'command': 'get_video_path', 'params': {'class_id': class_id}}
                if not send_msg(sock, request):
                    raise ConnectionError("서버에 영상 요청 실패")

                # 2. 서버의 첫 응답 받기 (pickle) - 스트리밍 시작 여부 확인
                initial_response = recv_msg(sock)
                if initial_response is None:
                    raise ConnectionError("서버로부터 첫 응답 수신 실패")

                if 'error' in initial_response:
                    self.video_display.setText(f"영상 요청 실패: {initial_response['error']}")
                    return
                
                if initial_response.get('status') == 'streaming_start':
                    # 3. 임시 파일에 영상 데이터 스트림 저장
                    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    self.temp_video_path = temp_video_file.name
                    
                    print(f"파일 스트리밍 수신 시작 -> {self.temp_video_path}")
                    
                    buffer = b''
                    while True:
                        # 종료 마커(b'DONE')가 수신될 때까지 데이터를 받음
                        chunk = sock.recv(4096)
                        if not chunk: break 
                        
                        # 수신 데이터에 종료 마커가 포함되어 있는지 확인
                        if b'DONE' in chunk:
                            # 마커 이전 데이터까지만 파일에 씀
                            final_chunk, _ = chunk.split(b'DONE', 1)
                            buffer += final_chunk
                            break
                        else:
                            buffer += chunk

                    temp_video_file.write(buffer)
                    temp_video_file.close()
                    print(f"스트리밍 수신 완료.")
                    
                    # 4. 저장된 임시 파일을 재생
                    self.play_video(self.temp_video_path)
                else:
                    self.video_display.setText("서버로부터 알 수 없는 응답을 받았습니다.")

        except (socket.error, ConnectionError) as e:
            QMessageBox.critical(self, "서버 통신 오류", f"영상 스트리밍 중 오류 발생: {e}")
            
    def _load_ui_and_find_widgets(self):
        loader = QUiLoader()
        ui_file_path = os.path.join(os.path.dirname(__file__), "log_viewer.ui")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QIODevice.ReadOnly):
            QMessageBox.critical(self, "UI 파일 오류", f"UI 파일을 열 수 없습니다: {ui_file.errorString()}"); sys.exit(1)
        ui_widget = loader.load(ui_file, self); ui_file.close()
        self.setLayout(ui_widget.layout())
        self.video_display = self.findChild(QLabel, "video_display")
        self.start_date_edit = self.findChild(QDateEdit, "start_date_edit")
        self.end_date_edit = self.findChild(QDateEdit, "end_date_edit")
        self.orderby_combo = self.findChild(QComboBox, "orderby_combo")
        self.search_button = self.findChild(QPushButton, "search_button")
        self.log_table = self.findChild(QTableWidget, "log_table")
        self.prev_button = self.findChild(QPushButton, "prev_button")
        self.page_label = self.findChild(QLabel, "page_label")
        self.next_button = self.findChild(QPushButton, "next_button")
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.event_type_map = {
            "화재": ("detect_fire", self.findChild(QCheckBox, "cb_fire")),
            "폭행": ("assault", self.findChild(QCheckBox, "cb_assault")),
            "누워있는 사람": ("fallen", self.findChild(QCheckBox, "cb_fallen")),
            "실종자": ("missing", self.findChild(QCheckBox, "cb_missing")),
            "무단 투기": ("dumping", self.findChild(QCheckBox, "cb_dumping")),
            "흡연자": ("smoke", self.findChild(QCheckBox, "cb_smoking")),
        }
        self.start_date_edit.setDate(QDate.currentDate().addDays(-1))
        self.end_date_edit.setDate(QDate.currentDate())

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)

    def update_table_display(self):
        self.log_table.setSortingEnabled(False); self.log_table.setRowCount(0)
        start_index = (self.current_page - 1) * self.items_per_page
        end_index = start_index + self.items_per_page
        for row_idx, entry in enumerate(self.filtered_log_entries[start_index:end_index]):
            self.log_table.insertRow(row_idx)
            try:
                raw_result = json.loads(entry['raw_result'])
                detected_events = []
                for key, value in raw_result.items():
                    if key.startswith('feat_') and isinstance(value, list):
                        for d in value:
                            if 'class_name' in d:
                                detected_events.append(d['class_name'])
                event_type_str = ", ".join(list(set(detected_events))) or "정보"
            except (json.JSONDecodeError, AttributeError): event_type_str = "데이터 형식 오류"
            timestamp_obj = entry['timestamp'];
            if isinstance(timestamp_obj, str): timestamp_obj = datetime.fromisoformat(timestamp_obj)
            timestamp_str = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')
            self.log_table.setItem(row_idx, 0, QTableWidgetItem(timestamp_str))
            self.log_table.setItem(row_idx, 1, QTableWidgetItem(event_type_str))
            play_button = QPushButton("재생")
            play_button.clicked.connect(lambda chk, cid=entry['class_id']: self.play_video_for_log(cid))
            self.log_table.setCellWidget(row_idx, 2, play_button)
        self.log_table.setSortingEnabled(True); self.update_pagination_controls()
    
    def update_pagination_controls(self):
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        self.page_label.setText(f"{self.current_page} / {total_pages}")
        self.prev_button.setEnabled(self.current_page > 1)
        self.next_button.setEnabled(self.current_page < total_pages)

    def go_to_prev_page(self):
        if self.current_page > 1: self.current_page -= 1; self.update_table_display()

    def go_to_next_page(self):
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        if self.current_page < total_pages: self.current_page += 1; self.update_table_display()

    def play_video(self, path):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
        self.video_capture = cv2.VideoCapture(path)
        if self.video_capture.isOpened():
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.playing_video = True; self.video_timer.start(int(1000 / fps))
        else: self.playing_video = False; self.video_display.setText(f"영상 파일을 열 수 없습니다:\n{path}")

    def update_video_frame(self):
        if self.playing_video and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.flip(frame, 0)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
                self.video_display.setPixmap(QPixmap.fromImage(qt_image))
            else:
                self.video_timer.stop()
                if self.video_capture: self.video_capture.release()
                self.playing_video = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_size_set:
            splitter = self.findChild(QSplitter, "main_splitter")
            if splitter: splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
            self._initial_size_set = True

    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture: self.video_capture.release()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = LogViewerDialog()
    dialog.showMaximized()
    sys.exit(app.exec())