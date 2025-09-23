
import sys
import os
import re
import cv2
import json # [추가] JSON 라이브러리 임포트
from datetime import datetime, timedelta

from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QAbstractItemView, QMessageBox)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap

# [추가] 새로 작성한 네트워크 클라이언트 모듈을 import 합니다.
from network_client import fetch_logs_from_network

# --- 테스트를 위한 실행 코드 ---
def run_test():
    """테스트를 위한 LogViewerDialog 실행 함수"""
    app = QApplication(sys.argv)
    
    # [수정] LogViewerDialog는 이제 스스로 네트워크를 통해 데이터를 가져오므로
    #        테스트용 로그 데이터를 직접 주입할 필요가 없습니다.
    dialog = LogViewerDialog()
    dialog.show()
    
    sys.exit(app.exec())


# --- 로그 뷰어 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    # [수정] 생성자에서 log_entries 파라미터를 제거합니다.
    def __init__(self, parent=None, event_colors=None):
        super().__init__(parent)
        
        self._load_ui_and_find_widgets()
        
        # [수정] all_log_entries 제거, filtered_log_entries만 사용
        self.filtered_log_entries = []
        self.event_colors = event_colors or {}
        self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        self.video_capture = None
        self.playing_video = False
        self.current_page = 1
        self.items_per_page = 20

        self.connect_signals()
        self.request_logs_from_server() # 초기 데이터 로드를 위해 호출

    def _load_ui_and_find_widgets(self):
        loader = QUiLoader()
        ui_file_path = "/home/geonpc/dev-ws/deeplearning-repo-2/dataService/log_viewer.ui"
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

        self.event_type_map = {
            "화재": ("0", self.findChild(QCheckBox, "cb_fire")),
            "폭행": ("1", self.findChild(QCheckBox, "cb_assault")),
            "누워있는 사람": ("2", self.findChild(QCheckBox, "cb_fallen")),
            "실종자": ("3", self.findChild(QCheckBox, "cb_missing")),
            "무단 투기": ("4", "cb_dumping"),
            "흡연자": ("5", self.findChild(QCheckBox, "cb_smoking")),
        }
        self.event_checkboxes = {name: data[1] for name, data in self.event_type_map.items()}

        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate(2025, 9, 21))
        self.end_date_edit.setDate(QDate.currentDate())

    def connect_signals(self):
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)
    
    # ----------------------------------------------------------------------------------
    # [핵심 수정] 2번 요구사항을 반영하여 TCP 통신으로 데이터를 요청하도록 수정한 함수
    # ----------------------------------------------------------------------------------
    def request_logs_from_server(self):
        """
        UI에서 선택된 조건들을 JSON으로 만들어 TCP 서버에 로그를 요청하고,
        그 결과를 테이블에 표시하는 함수입니다.
        """
        # 1. UI 위젯에서 검색 조건 값들을 가져옵니다.
        #    (날짜 형식은 서버와 약속된 'yyyy-MM-dd'로 맞춥니다)
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        
        # 서버와 약속된 명확한 키워드('latest', 'oldest')로 변환합니다.
        orderby = "latest" if self.orderby_combo.currentText() == "최신순" else "oldest"
        
        detection_types = [
            code for name, (code, checkbox) in self.event_type_map.items() 
            if checkbox and checkbox.isChecked()
        ]
        
        # 2. 서버에 전송할 요청 데이터를 딕셔너리(JSON) 형태로 만듭니다.
        request_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "orderby": orderby,
            "detection_types": detection_types
        }
        
        # 3. network_client.py에 정의된 함수를 호출하여 서버에 데이터를 요청합니다.
        #    UI 코드는 복잡한 TCP 통신 과정을 알 필요가 없습니다.
        response_data = fetch_logs_from_network(request_payload)
        
        # 4. 서버 응답 결과에 따라 UI를 처리합니다.
        if response_data is not None:
            # 성공적으로 데이터를 받으면, 테이블에 표시할 데이터로 설정
            self.filtered_log_entries = response_data
            if not response_data:
                 QMessageBox.information(self, "검색 결과", "조건에 맞는 로그가 없습니다.")
        else:
            # 통신 실패 시, 사용자에게 오류를 알리고 테이블을 비웁니다.
            QMessageBox.critical(self, "네트워크 오류", "서버로부터 로그를 가져오는 데 실패했습니다.\n서버 상태를 확인해주세요.")
            self.filtered_log_entries = []

        # 5. 조회된 데이터로 테이블을 업데이트하고 페이지를 1페이지로 리셋합니다.
        self.current_page = 1
        self.update_table_display()

    # --- 이하 코드는 기존과 동일 (수정 없음) ---
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

if __name__ == '__main__':
    run_test()


### 실행 방법

# 1.  위 3개의 파일을 모두 같은 폴더에 저장합니다.
# 2.  터미널(명령 프롬프트)을 열고 `tcp_server.py`를 먼저 실행합니다.
#     ```bash
#     python tcp_server.py
#     ```
#     서버가 실행되면 "서버 시작! 클라이언트 연결 대기 중..." 메시지가 표시됩니다.
# 3.  **다른** 터미널을 열고 `log_viewer_tcp.py`를 실행합니다.
#     ```bash
#     python log_viewer_tcp.py
    

