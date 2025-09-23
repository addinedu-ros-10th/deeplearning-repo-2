
# log_viewer.py
# --- 필요한 라이브러리들을 가져옵니다 (Import) ---
import sys      # 시스템 관련 기능 (프로그램 실행 인자, 종료 등)
import os       # 운영체제 관련 기능 (파일/폴더 경로 처리 등)
import re       # 정규 표현식 (문자열에서 특정 패턴을 찾기 위함)
import cv2      # OpenCV 라이브러리 (영상 파일을 읽고 처리하기 위함)
from datetime import datetime, timedelta # 날짜와 시간 관련 작업을 위한 클래스
# PySide6: Qt GUI 프레임워크를 위한 파이썬 라이브러리
from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QCheckBox, QDateEdit, QComboBox,
                               QHeaderView, QAbstractItemView, QMessageBox, QSplitter, QGroupBox)
from PySide6.QtUiTools import QUiLoader # .ui 디자인 파일을 동적으로 로드하는 클래스
from PySide6.QtCore import QFile, QIODevice, QTimer, QDate # 파일, 입출력 장치, 타이머, 날짜 관련 클래스
from PySide6.QtGui import QImage, QPixmap # GUI에서 이미지를 다루기 위한 클래스
# --- 로그 뷰어 다이얼로그(창) 클래스 정의 ---
class LogViewerDialog(QDialog):
    """
    로그를 테이블 형태로 보여주고, 관련 영상을 재생하는 다이얼로그(창) 클래스입니다.
    Qt Designer로 만든 log_viewer.ui 파일과 연동하여 동작합니다.
    """
    def _init_(self, parent=None, log_entries=None, event_colors=None):
        # 부모 클래스인 QDialog의 생성자를 호출하여 기본적인 창의 속성을 설정합니다.
        super()._init_(parent)
        # .ui 파일을 불러와 화면을 구성하고 내부 위젯들을 초기화하는 메서드를 호출합니다.
        self._load_ui_and_find_widgets()
        # --- 데이터 및 상태 변수 초기화 ---
        # 외부에서 전달받은 전체 로그 목록을 저장합니다. (없으면 빈 리스트로 초기화)
        self.all_log_entries = log_entries or []
        # 현재 필터링된 로그 목록을 저장할 리스트입니다.
        self.filtered_log_entries = []
        # 이벤트 종류별 색상 정보 (현재 코드에서는 직접 사용되지 않음)
        self.event_colors = event_colors or {}
        # 'recordings' 폴더에서 .mp4 파일 목록을 찾아 최신순으로 정렬하여 저장합니다.
        try:
            self.recording_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
        except FileNotFoundError:
            # 'recordings' 폴더가 없으면 예외가 발생하므로, 빈 리스트로 초기화하고 경고 메시지를 출력합니다.
            self.recording_files = []
            print("Warning: 'recordings' directory not found.")
        self.video_capture = None   # OpenCV의 비디오 캡처 객체를 저장할 변수
        self.playing_video = False  # 현재 비디오가 재생 중인지 상태를 나타내는 플래그
        self.current_page = 1       # 로그 테이블의 현재 페이지 번호
        self.items_per_page = 20    # 한 페이지에 보여줄 로그의 수
        self._initial_size_set = False # 창이 처음 열릴 때 스플리터 크기를 딱 한 번만 설정하기 위한 플래그
        # UI 요소들의 이벤트(시그널)를 처리 함수(슬롯)에 연결합니다.
        self.connect_signals()
        # 초기 데이터로 로그 목록을 필터링하고 테이블에 표시합니다.
        self.request_logs_from_server()
        # 프로그램 시작 시 첫 번째 로그를 자동으로 선택하고 영상을 재생하는 초기 동작을 수행합니다.
        self.initial_load_actions()
    def _load_ui_and_find_widgets(self):
        """ .ui 파일을 읽어 화면을 구성하고, 코드에서 사용할 위젯들을 이름으로 찾아 멤버 변수에 할당합니다. """
        # QUiLoader: .ui 파일을 읽어 Qt 위젯 객체로 변환해주는 클래스
        loader = QUiLoader()
        ui_file_path = "log_viewer.ui"
        ui_file = QFile(ui_file_path)
        # .ui 파일을 읽기 전용으로 엽니다.
        if not ui_file.open(QIODevice.ReadOnly):
            # 파일 열기에 실패하면 에러 메시지 팝업을 띄우고 프로그램을 종료합니다.
            QMessageBox.critical(self, "UI 파일 오류", f"UI 파일을 열 수 없습니다: {ui_file.errorString()}")
            QTimer.singleShot(0, self.close) # 즉시 닫으면 에러가 날 수 있어, 0초 후 닫도록 스케줄링
            return
        # .ui 파일을 로드하여 위젯을 생성합니다. self를 부모로 지정하여 위젯들이 이 다이얼로그에 속하게 합니다.
        ui_widget = loader.load(ui_file, self)
        ui_file.close()
        # 로드한 위젯의 레이아웃을 현재 다이얼로그(self)의 메인 레이아웃으로 설정합니다.
        self.setLayout(ui_widget.layout())
        # .ui 파일에 설정된 objectName을 기준으로 각 위젯을 찾아 멤버 변수에 할당합니다.
        self.video_display = self.findChild(QLabel, "video_display")
        self.start_date_edit = self.findChild(QDateEdit, "start_date_edit")
        self.end_date_edit = self.findChild(QDateEdit, "end_date_edit")
        self.orderby_combo = self.findChild(QComboBox, "orderby_combo")
        self.search_button = self.findChild(QPushButton, "search_button")
        self.log_table = self.findChild(QTableWidget, "log_table")
        self.prev_button = self.findChild(QPushButton, "prev_button")
        self.page_label = self.findChild(QLabel, "page_label")
        self.next_button = self.findChild(QPushButton, "next_button")
        # 테이블의 가로 헤더(컬럼명)를 가져와 너비 조절 정책을 설정합니다.
        header = self.log_table.horizontalHeader()
        # 0번 열('발생 시각')은 내용 길이에 자동으로 맞춰집니다.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        # 1번 열('이벤트 종류')은 남는 공간을 모두 채우도록 확장됩니다. (이것이 컬럼 잘림을 방지합니다)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # 2번 열('영상 재생')은 내용 길이에 자동으로 맞춰집니다.
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        # 이벤트 종류 이름과 .ui 파일의 체크박스 위젯을 매핑(연결)합니다.
        self.event_type_map = {
            "화재": ("0", self.findChild(QCheckBox, "cb_fire")),
            "폭행": ("1", self.findChild(QCheckBox, "cb_assault")),
            "누워있는 사람": ("2", self.findChild(QCheckBox, "cb_fallen")),
            "실종자": ("3", self.findChild(QCheckBox, "cb_missing")),
            "무단 투기": ("4", self.findChild(QCheckBox, "cb_dumping")),
            "흡연자": ("5", self.findChild(QCheckBox, "cb_smoking")),
        }
        # 위젯 객체만 따로 딕셔너리로 관리하여 코드에서 쉽게 접근할 수 있도록 합니다.
        self.event_checkboxes = {name: data[1] for name, data in self.event_type_map.items()}
        # '이벤트 종류' 그룹 박스를 찾아 최소 너비를 설정하여 UI가 너무 좁아져 깨지는 것을 방지합니다.
        event_groupbox = self.findChild(QGroupBox, "event_groupbox")
        if event_groupbox:
            event_groupbox.setMinimumWidth(310)
        # 테스트용 데이터가 화면에 보이도록 날짜 위젯의 기본값을 특정 날짜로 설정합니다.
        self.start_date_edit.setDate(QDate(2025, 9, 21))
        self.end_date_edit.setDate(QDate(2025, 9, 22))
    def connect_signals(self):
        """UI 위젯들의 시그널(사용자 이벤트)을 해당 기능을 수행하는 슬롯(함수)에 연결합니다."""
        # 동영상 프레임 업데이트를 위한 타이머를 생성하고, 타임아웃 시 update_video_frame 함수를 호출하도록 연결합니다.
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        # 버튼 클릭 이벤트들을 각각의 처리 함수에 연결합니다.
        self.search_button.clicked.connect(self.request_logs_from_server)
        self.prev_button.clicked.connect(self.go_to_prev_page)
        self.next_button.clicked.connect(self.go_to_next_page)
3:39
def initial_load_actions(self):
        """프로그램이 시작될 때 첫 번째 로그를 자동으로 선택하고 해당 영상을 재생하여 사용자 경험을 향상시킵니다."""
        # 필터링된 로그가 하나라도 있을 경우에만 실행됩니다.
        if self.filtered_log_entries:
            # 테이블의 0번 행(가장 첫 번째 행)을 선택된 상태로 만듭니다.
            self.log_table.selectRow(0)
            # 첫 번째 로그의 타임스탬프를 가져옵니다.
            first_log_timestamp = self.filtered_log_entries[0]['timestamp']
            # 해당 타임스탬프의 영상을 재생하는 함수를 호출합니다.
            self.play_video_for_log(first_log_timestamp)
    def request_logs_from_server(self):
        """UI의 검색 조건에 따라 로그를 필터링하고 테이블을 업데이트합니다. (현재는 로컬 데이터를 필터링하는 시뮬레이션)"""
        # --- 실제로는 이 부분에서 서버에 보낼 JSON 데이터를 생성합니다. ---
        start_date = self.start_date_edit.date().toString("yyyyMMdd")
        end_date = self.end_date_edit.date().toString("yyyyMMdd")
        orderby = self.orderby_combo.currentText() == "최신순"
        detection_types = []
        for name, (code, checkbox) in self.event_type_map.items():
            if checkbox.isChecked():
                detection_types.append(code)
        # --- 여기까지가 서버 요청 데이터 생성 부분 ---
        # --- 아래는 서버에서 응답을 받았다고 가정하고, 로컬 데이터를 필터링하는 시뮬레이션 로직입니다. ---
        start_dt = self.start_date_edit.dateTime().toPython().date()
        end_dt = self.end_date_edit.dateTime().toPython().date()
        self.filtered_log_entries = []
        for entry in self.all_log_entries:
            try:
                log_dt = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                # 날짜 범위 확인: 로그의 날짜가 선택된 시작일과 종료일 사이에 있는지 검사합니다.
                if not (start_dt <= log_dt <= end_dt): continue
                # 선택된 이벤트 종류 확인: 체크된 이벤트 종류가 로그 메시지에 포함되어 있는지 검사합니다.
                is_target_event = any(name in entry['message'] for name, (code, cb) in self.event_type_map.items() if code in detection_types)
                if is_target_event: self.filtered_log_entries.append(entry)
            except ValueError: continue
        # 필터링된 결과를 '최신순' 또는 '오래된 순'으로 정렬합니다.
        self.filtered_log_entries.sort(key=lambda x: x['timestamp'], reverse=orderby)
        # 첫 페이지로 이동한 후 테이블 내용을 업데이트합니다.
        self.current_page = 1
        self.update_table_display()
    def update_table_display(self):
        """필터링된 로그 데이터를 현재 페이지에 맞게 테이블에 표시합니다."""
        # 데이터를 채우기 전에 정렬 기능을 잠시 비활성화하여 성능 저하를 막습니다.
        self.log_table.setSortingEnabled(False)
        # 기존 테이블 내용을 모두 지웁니다.
        self.log_table.setRowCount(0)
        # 현재 페이지에 해당하는 데이터의 시작/끝 인덱스를 계산합니다.
        start_index = (self.current_page - 1) * self.items_per_page
        end_index = start_index + self.items_per_page
        # 현재 페이지의 로그만 순회하며 테이블에 행을 추가합니다.
        for row, entry in enumerate(self.filtered_log_entries[start_index:end_index]):
            self.log_table.insertRow(row)
            message = entry['message']
            # 정규표현식을 사용해 전체 로그 메시지에서 '이벤트 종류' 부분만 추출합니다.
            event_type = "정보" # 기본값
            match = re.search(r"\]\s*([\w\s]+?)\s*(감지됨|확인됨|상황 종료됨)", message)
            if match: event_type = match.group(1).strip()
            # 각 셀(칸)에 데이터를 채웁니다.
            self.log_table.setItem(row, 0, QTableWidgetItem(entry['timestamp']))
            self.log_table.setItem(row, 1, QTableWidgetItem(event_type))
            # '재생' 버튼을 동적으로 생성하여 테이블의 세 번째 칸에 추가합니다.
            play_button = QPushButton("재생")
            # lambda 함수를 사용해, 버튼 클릭 시 해당 로그의 타임스탬프를 인자로 전달하여 play_video_for_log 함수가 호출되도록 연결합니다.
            play_button.clicked.connect(lambda chk, ts=entry['timestamp']: self.play_video_for_log(ts))
            self.log_table.setCellWidget(row, 2, play_button)
        # 데이터 채우기가 끝나면 다시 정렬 기능을 활성화합니다.
        self.log_table.setSortingEnabled(True)
        # 페이지 번호 UI("1 / 5" 등)를 업데이트합니다.
        self.update_pagination_controls()
    def update_pagination_controls(self):
        """페이지 번호와 이전/다음 버튼의 활성화 상태를 업데이트합니다."""
        # 전체 페이지 수를 계산합니다. (나머지가 없도록 올림 계산)
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        self.page_label.setText(f"{self.current_page} / {total_pages}")
        # 현재 페이지 위치에 따라 버튼을 활성화하거나 비활성화합니다.
        self.prev_button.setEnabled(self.current_page > 1)
        self.next_button.setEnabled(self.current_page < total_pages)
    def go_to_prev_page(self):
        """'이전' 버튼 클릭 시 호출되어 이전 페이지의 로그를 보여줍니다."""
        if self.current_page > 1:
            self.current_page -= 1
            self.update_table_display()
    def go_to_next_page(self):
        """'다음' 버튼 클릭 시 호출되어 다음 페이지의 로그를 보여줍니다."""
        total_pages = max(1, (len(self.filtered_log_entries) + self.items_per_page - 1) // self.items_per_page)
        if self.current_page < total_pages:
            self.current_page += 1
            self.update_table_display()
    def play_video_for_log(self, timestamp_str):
        """주어진 타임스탬프와 가장 가까운 시간의 녹화 영상을 찾아 재생합니다."""
        log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        best_match = None
        min_diff = timedelta.max
        # 모든 녹화 파일을 순회합니다.
        for filename in self.recording_files:
            try:
                # 파일 이름에서 시간 정보를 추출하여 datetime 객체로 변환합니다.
                file_time_str = filename.replace('chunk_', '').replace('.mp4', '')
                file_time = datetime.strptime(file_time_str, "%Y-%m-%d_%H-%M-%S")
                # 로그 발생 시간보다 이전에 녹화된 파일 중에서,
                if log_time >= file_time:
                    diff = log_time - file_time
                    # 시간 차이가 가장 적은 파일을 선택합니다.
                    if diff < min_diff:
                        min_diff = diff
                        best_match = filename
            except ValueError: continue
        if best_match:
            self.play_video(os.path.join("recordings", best_match))
        else:
            self.video_display.setText("해당 시간에 녹화된 영상이 없습니다.")
    def play_video(self, path):
        """주어진 경로의 비디오 파일을 실제로 재생합니다."""
        # 이미 다른 영상이 재생 중이면 먼저 중지하고 자원을 해제합니다.
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
        # OpenCV를 사용해 비디오 파일을 엽니다.
        self.video_capture = cv2.VideoCapture(path)
        if self.video_capture.isOpened():
            # 영상의 초당 프레임 수(FPS)를 가져와 타이머의 주기를 설정하고 시작합니다.
            fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            self.playing_video = True
            self.video_timer.start(int(1000 / fps))
        else:
            self.playing_video = False
            print(f"Error: Cannot open video file at {path}")
    def update_video_frame(self):
        """타이머에 의해 주기적으로 호출되어 비디오의 다음 프레임을 읽어와 화면에 표시합니다."""
        if self.playing_video and self.video_capture:
            ret, frame = self.video_capture.read()
            # 프레임을 성공적으로 읽었을 경우
            if ret:
                # OpenCV의 BGR 색상 순서를 Qt에서 사용하는 RGB 순서로 변환합니다.
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                # NumPy 배열(frame)을 QImage 형식으로 변환합니다.
                qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
                # QImage를 QPixmap으로 변환하여 QLabel에 표시합니다.
                self.video_display.setPixmap(QPixmap.fromImage(qt_image))
            # 영상의 끝에 도달했을 경우
            else:
                # 타이머와 비디오 캡처를 중지하고 자원을 해제합니다.
                self.video_timer.stop()
                if self.video_capture: self.video_capture.release()
                self.playing_video = False
    def showEvent(self, event):
        """창이 처음 화면에 표시되기 직전에 자동으로 호출되는 이벤트 핸들러입니다."""
        super().showEvent(event)
        # 창이 처음 열릴 때 딱 한 번만 실행하기 위한 로직
        if not self._initial_size_set:
            splitter = self.findChild(QSplitter, "main_splitter")
            if splitter:
                # 현재 창의 전체 너비를 기준으로 6:4 비율로 스플리터(좌우 패널)를 나눕니다.
                total_width = self.width()
                splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
            # 플래그를 True로 바꿔 다시는 이 로직이 실행되지 않도록 합니다.
            self._initial_size_set = True
    def closeEvent(self, event):
        """창이 닫힐 때 호출되어 비디오 관련 자원을 안전하게 해제합니다."""
        self.video_timer.stop()
        if self.video_capture: self.video_capture.release()
        super().closeEvent(event)
# --- 스크립트가 직접 실행될 때만 아래 코드가 동작하도록 하는 파이썬 표준 구문 ---
if _name_ == '_main_':
    def run_test():
        """이 파일을 단독으로 실행했을 때 테스트용 창을 띄우는 함수입니다."""
        app = QApplication(sys.argv)
        test_logs = [
            {'timestamp': '2025-09-21 14:10:30', 'message': '[자동] 화재 감지됨 (확률: 0.95)'},
            {'timestamp': '2025-09-22 11:05:00', 'message': '[수동] 폭행 감지됨 (확률: 1.00)'}
        ]
        dialog = LogViewerDialog(log_entries=test_logs)
        # 창을 일반 크기가 아닌, 모니터에 꽉 채운 최대화 상태로 엽니다.
        dialog.showMaximized()
        sys.exit(app.exec())
    run_test()






