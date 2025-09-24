import sys
import os
import re
import cv2
from datetime import datetime, timedelta

from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QAbstractItemView, QMessageBox, QSplitter, QGroupBox)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap

# --- 로그 뷰어 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None, event_colors=None):
        super().__init__(parent)
        
        self._load_ui_and_find_widgets()

        self.all_log_entries = log_entries or []
        self.filtered_log_entries = []
        self.event_colors = event_colors or {}
        try:
            self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        except FileNotFoundError:
            self.recording_files = []
            print("Warning: 'recordings' directory not found.")
            
        self.video_capture = None
        self.playing_video = False
        self.current_page = 1
        self.items_per_page = 20
        self._initial_size_set = False

        self.connect_signals()
        self.request_logs_from_server()
        self.initial_load_actions()

    def _load_ui_and_find_widgets(self):
        loader = QUiLoader()
        ui_file_path = "log_viewer.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file.open(QIODevice.ReadOnly):
            QMessageBox.critical(self, "UI 파일 오류", f"UI 파일을 열 수 없습니다: {ui_file.errorString()}")
            QTimer.singleShot(0, self.close)
            return
        
        ui_widget = loader.load(ui_file, self)
        ui_file.close()
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
            "화재": ("0", self.findChild(QCheckBox, "cb_fire")),
            "폭행": ("1", self.findChild(QCheckBox, "cb_assault")),
            "누워있는 사람": ("2", self.findChild(QCheckBox, "cb_fallen")),
            "실종자": ("3", self.findChild(QCheckBox, "cb_missing")),
            "무단 투기": ("4", self.findChild(QCheckBox, "cb_dumping")),
            "흡연자": ("5", self.findChild(QCheckBox, "cb_smoking")),
        }
        self.event_checkboxes = {name: data[1] for name, data in self.event_type_map.items()}

        # --- [추가] '이벤트 종류' 그룹 박스의 최소 너비를 코드로 설정 ---
        event_groupbox = self.findChild(QGroupBox, "event_groupbox")
        if event_groupbox:
            event_groupbox.setMinimumWidth(310) # 최소 너비를 300px로 설정
        # ---------------------------------------------------------

        self.start_date_edit.setDate(QDate(2025, 9, 21))
        self.end_date_edit.setDate(QDate(2025, 9, 22))

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)
    
    def initial_load_actions(self):
        """프로그램 시작 시 첫 로그를 자동 선택하고 영상을 재생합니다."""
        if self.filtered_log_entries:
            self.log_table.selectRow(0)
            first_log_timestamp = self.filtered_log_entries[0]['timestamp']
            self.play_video_for_log(first_log_timestamp)

    def request_logs_from_server(self):
        start_date = self.start_date_edit.date().toString("yyyyMMdd")
        end_date = self.end_date_edit.date().toString("yyyyMMdd")
        orderby = self.orderby_combo.currentText() == "최신순"
        
        detection_types = []
        for name, (code, checkbox) in self.event_type_map.items():
            if checkbox.isChecked():
                detection_types.append(code)

        start_dt = self.start_date_edit.dateTime().toPython().date()
        end_dt = self.end_date_edit.dateTime().toPython().date()
        self.filtered_log_entries = []
        for entry in self.all_log_entries:
            try:
                log_dt = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                if not (start_dt <= log_dt <= end_dt): continue
                is_target_event = any(name in entry['message'] for name, (code, cb) in self.event_type_map.items() if code in detection_types)
                if is_target_event: self.filtered_log_entries.append(entry)
            except ValueError: continue
        
        self.filtered_log_entries.sort(key=lambda x: x['timestamp'], reverse=orderby)
        self.current_page = 1
        self.update_table_display()

    def update_table_display(self):
        self.log_table.setSortingEnabled(False)
        self.log_table.setRowCount(0)
        start_index = (self.current_page - 1) * self.items_per_page
        end_index = start_index + self.items_per_page
        for row, entry in enumerate(self.filtered_log_entries[start_index:end_index]):
            self.log_table.insertRow(row)
            message = entry['message']
            event_type = "정보"
            match = re.search(r"\]\s*([\w\s]+?)\s*(감지됨|확인됨|상황 종료됨)", message)
            if match: event_type = match.group(1).strip()
            self.log_table.setItem(row, 0, QTableWidgetItem(entry['timestamp']))
            self.log_table.setItem(row, 1, QTableWidgetItem(event_type))
            play_button = QPushButton("재생")
            play_button.clicked.connect(lambda chk, ts=entry['timestamp']: self.play_video_for_log(ts))
            self.log_table.setCellWidget(row, 2, play_button)
        self.log_table.setSortingEnabled(True)
        self.update_pagination_controls()

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
                file_time_str = filename.replace('chunk_', '').replace('.mp4', '')
                file_time = datetime.strptime(file_time_str, "%Y-%m-%d_%H-%M-%S")
                if log_time >= file_time:
                    diff = log_time - file_time
                    if diff < min_diff:
                        min_diff = diff
                        best_match = filename
            except ValueError: continue
        if best_match: self.play_video(os.path.join("recordings", best_match))
        else: self.video_display.setText("해당 시간에 녹화된 영상이 없습니다.")

    def play_video(self, path):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
        self.video_capture = cv2.VideoCapture(path)
        if self.video_capture.isOpened():
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.playing_video = True
            self.video_timer.start(int(1000 / fps))
        else: self.playing_video = False

    def update_video_frame(self):
        if self.playing_video and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
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
            if splitter:
                total_width = self.width()
                splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
            
            self._initial_size_set = True

    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture: self.video_capture.release()
        super().closeEvent(event)


# --- 스크립트가 직접 실행될 때만 run_test 함수를 호출 ---
if __name__ == '__main__':
    def run_test():
        app = QApplication(sys.argv)
        
        test_logs = [
            {'timestamp': '2025-09-21 14:10:30', 'message': '[자동] 화재 감지됨 (확률: 0.95)'},
            {'timestamp': '2025-09-22 11:05:00', 'message': '[수동] 폭행 감지됨 (확률: 1.00)'}
        ]
        
        dialog = LogViewerDialog(log_entries=test_logs)
        dialog.showMaximized()
        
        sys.exit(app.exec())
        
    run_test()