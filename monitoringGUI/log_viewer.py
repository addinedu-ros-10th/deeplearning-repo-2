# log_viewer.py

import os
import re
from datetime import datetime, timedelta

# PySide6 관련 모든 import 구문을 이 파일에도 추가해야 합니다.
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QDialog, QGridLayout, QPushButton, QComboBox, QGroupBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
                               QAbstractItemView, QCheckBox)
from PySide6.QtCore import Qt, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap

# cv2는 영상 재생에만 필요하므로 이 파일에서도 import합니다.
import cv2

class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None, event_colors=None):
        super().__init__(parent)
        self.setWindowTitle("로그 뷰어 및 동영상 재생")
        self.setMinimumSize(1200, 800)

        # 데이터 및 상태 변수
        self.all_log_entries = log_entries or []
        self.filtered_log_entries = []
        self.event_colors = event_colors or {}
        
        self.event_type_map = {
            "화재": "0", "폭행": "1", "누워있는 사람": "2", 
            "실종자": "3", "무단 투기": "4", "흡연자": "5"
        }

        self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        self.video_capture = None
        self.playing_video = False
        
        self.current_page = 1
        self.items_per_page = 20

        self.setupUi()
        self.connect_signals()
        self.request_logs_from_server()

    # setupUi, connect_signals 및 LogViewerDialog의 모든 메서드는
    # 기존 코드와 동일하게 이 클래스 내부에 그대로 둡니다.
    # ... (기존에 작성하신 LogViewerDialog의 모든 메서드를 여기에 붙여넣기) ...
    def setupUi(self):
        main_layout = QHBoxLayout(self)
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_display = QLabel("재생할 동영상을 선택하세요.")
        self.video_display.setFixedSize(640, 480)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: rgb(40, 40, 40); color: white;")
        self.video_display.setScaledContents(True)
        video_layout.addWidget(self.video_display)
        video_layout.addStretch()
        main_layout.addWidget(video_panel)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        search_groupbox = QGroupBox("검색 조건")
        search_layout = QGridLayout(search_groupbox)
        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
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
        event_groupbox = QGroupBox("이벤트 종류")
        self.event_checkboxes = {}
        event_layout = QGridLayout(event_groupbox)
        event_names = list(self.event_type_map.keys())
        for i, name in enumerate(event_names):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.event_checkboxes[name] = checkbox
            event_layout.addWidget(checkbox, i // 2, i % 2)
        search_layout.addWidget(event_groupbox, 3, 0, 1, 2)
        self.search_button = QPushButton("로그 검색")
        search_layout.addWidget(self.search_button, 4, 0, 1, 2)
        control_layout.addWidget(search_groupbox)
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(3)
        self.log_table.setHorizontalHeaderLabels(["발생 시각", "이벤트 종류", "영상 재생"])
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_table.setSortingEnabled(True)
        control_layout.addWidget(self.log_table)
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
        main_layout.addWidget(control_panel, 1)

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)

    def request_logs_from_server(self):
        start_date = self.start_date_edit.date().toString("yyyyMMdd")
        end_date = self.end_date_edit.date().toString("yyyyMMdd")
        orderby = self.orderby_combo.currentText() == "최신순"
        detection_types = [self.event_type_map[name] for name, cb in self.event_checkboxes.items() if cb.isChecked()]
        start_dt = self.start_date_edit.dateTime().toPython().date()
        end_dt = self.end_date_edit.dateTime().toPython().date()
        self.filtered_log_entries = []
        for entry in self.all_log_entries:
            try:
                log_dt = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                if not (start_dt <= log_dt <= end_dt): continue
                is_target_event = any(name in entry['message'] for name, code in self.event_type_map.items() if code in detection_types)
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
    
    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture: self.video_capture.release()
        super().closeEvent(event)