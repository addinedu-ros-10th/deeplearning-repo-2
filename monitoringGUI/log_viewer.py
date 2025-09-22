# log_viewer.py

import sys
import os
import re
import cv2
from datetime import datetime, timedelta

from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QAbstractItemView, QMessageBox)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap




# --- 테스트를 위한 실행 코드 ---   python3  log_viewer.py 시에 테스트용 로그 데이터를 확인할 수 있음
def run_test():
    """테스트를 위한 LogViewerDialog 실행 함수"""
    app = QApplication(sys.argv)
    
    # 테스트용 로그 데이터
    test_logs = [
        {'timestamp': '2025-09-21 14:10:30', 'message': '[자동] 화재 감지됨 (확률: 0.95)'},
        {'timestamp': '2025-09-22 11:05:00', 'message': '[수동] 폭행 감지됨 (확률: 1.00)'}
    ]
    
    # LogViewerDialog 인스턴스 생성 및 실행
    dialog = LogViewerDialog(log_entries=test_logs)
    dialog.show()
    
    sys.exit(app.exec())


# --- 로그 뷰어 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None, event_colors=None):
        super().__init__(parent)
        
        # UI 파일 로드 및 위젯 초기화
        self._load_ui_and_find_widgets()

        # 데이터 및 상태 변수 초기화
        self.all_log_entries = log_entries or []
        self.filtered_log_entries = []
        self.event_colors = event_colors or {}
        self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        self.video_capture = None
        self.playing_video = False
        self.current_page = 1
        self.items_per_page = 20

        # 시그널 연결 및 초기 데이터 로드
        self.connect_signals()
        self.request_logs_from_server()

    def _load_ui_and_find_widgets(self):
        # .ui 파일을 동적으로 로드
        loader = QUiLoader()
        ui_file_path = "log_viewer.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file.open(QIODevice.ReadOnly):
            # [수정] 에러 메시지를 간결한 형태로 변경
            QMessageBox.critical(self, "UI 파일 오류", f"UI 파일을 열 수 없습니다: {ui_file.errorString()}")
            QTimer.singleShot(0, self.close)
            return
        
        # self를 부모로 하여 위젯을 로드하면 findChild로 찾을 수 있게 됨
        ui_widget = loader.load(ui_file, self)
        ui_file.close()

        # 메인 레이아웃을 다이얼로그에 설정
        self.setLayout(ui_widget.layout())

        # .ui 파일에 정의된 위젯들을 objectName으로 찾아 self의 속성으로 만듦
        self.video_display = self.findChild(QLabel, "video_display")
        self.start_date_edit = self.findChild(QDateEdit, "start_date_edit")
        self.end_date_edit = self.findChild(QDateEdit, "end_date_edit")
        self.orderby_combo = self.findChild(QComboBox, "orderby_combo")
        self.search_button = self.findChild(QPushButton, "search_button")
        self.log_table = self.findChild(QTableWidget, "log_table")
        self.prev_button = self.findChild(QPushButton, "prev_button")
        self.page_label = self.findChild(QLabel, "page_label")
        self.next_button = self.findChild(QPushButton, "next_button")

        # 이벤트 이름과 코드, 체크박스 위젯을 매핑
        self.event_type_map = {
            "화재": ("0", self.findChild(QCheckBox, "cb_fire")),
            "폭행": ("1", self.findChild(QCheckBox, "cb_assault")),
            "누워있는 사람": ("2", self.findChild(QCheckBox, "cb_fallen")),
            "실종자": ("3", self.findChild(QCheckBox, "cb_missing")),
            "무단 투기": ("4", self.findChild(QCheckBox, "cb_dumping")),
            "흡연자": ("5", self.findChild(QCheckBox, "cb_smoking")),
        }
        self.event_checkboxes = {name: data[1] for name, data in self.event_type_map.items()}

        self.start_date_edit.setDate(QDate(2025, 9, 21))
        self.end_date_edit.setDate(QDate(2025, 9, 22))

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
    
    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture: self.video_capture.release()
        super().closeEvent(event)


# --- 스크립트가 직접 실행될 때만 run_test 함수를 호출 ---
if __name__ == '__main__':
    run_test()