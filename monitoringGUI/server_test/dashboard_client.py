# dashboard_client.py
import sys, cv2, os, json, socket, threading, time, random, numpy as np, pygame
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

SERVER_IP = "127.0.0.1"
DATA_SERVICE_IP = "127.0.0.1"

class Communicate(QObject):
    video_frame_ready = Signal(np.ndarray)
    event_data_ready = Signal(dict)
    connection_status = Signal(str, str)
    log_message = Signal(str, str)

class AlertDialog(QDialog):
    def __init__(self, parent, event_type):
        super().__init__(parent)
        self.setWindowTitle(f"{event_type} 경고")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.event_type = event_type
        self.parent_dashboard = parent

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{event_type} 상황이 감지되었습니다."))
        stop_button = QPushButton("상황 종료")
        layout.addWidget(stop_button)
        stop_button.clicked.connect(self.confirm_stop)

    def confirm_stop(self):
        if QMessageBox.question(self, "확인", "상황을 종료하시겠습니까?") == QMessageBox.StandardButton.Yes:
            self.parent_dashboard.resolve_alert(self.event_type)
            self.accept()

class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        pygame.mixer.init()

        self.communicator = Communicate()
        self.communicator.video_frame_ready.connect(self.update_frame)
        self.communicator.event_data_ready.connect(self.handle_event)
        self.communicator.connection_status.connect(self.update_connection_status)
        self.communicator.log_message.connect(self.add_log)
        
        self.event_colors = {"화재":"red", "폭행":"orange"}
        self.alert_sound = pygame.mixer.Sound("alert.wav") if os.path.exists("alert.wav") else None

        self.start_network_listeners()

    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("AURA 관제 대시보드 (클라이언트)")
        MainWindow.resize(1600, 900)
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        self.video_label = QLabel("서버 연결 대기 중..."); self.video_label.setAlignment(Qt.AlignCenter); self.video_label.setStyleSheet("background-color:black; color:white;")
        self.main_layout.addWidget(self.video_label, 7)
        control_panel = QWidget(); self.control_layout = QVBoxLayout(control_panel); self.main_layout.addWidget(control_panel, 3)
        info_groupbox = QGroupBox("연결 상태"); info_layout = QGridLayout(info_groupbox)
        info_layout.addWidget(QLabel("영상 서버:"), 0, 0); self.video_status_label = QLabel("-"); info_layout.addWidget(self.video_status_label, 0, 1)
        info_layout.addWidget(QLabel("이벤트 서버:"), 1, 0); self.event_status_label = QLabel("-"); info_layout.addWidget(self.event_status_label, 1, 1)
        self.control_layout.addWidget(info_groupbox)
        log_label = QLabel("통합 이벤트 로그"); font = QFont(); font.setPointSize(12); font.setBold(True); log_label.setFont(font); self.control_layout.addWidget(log_label)
        self.log_browser = QTextBrowser(); self.control_layout.addWidget(self.log_browser)

    def start_network_listeners(self):
        threading.Thread(target=self._video_receiver_worker, daemon=True).start()
        threading.Thread(target=self._event_receiver_worker, daemon=True).start()

    def _video_receiver_worker(self):
        # ... (이전과 동일)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("0.0.0.0", 9999)); sock.settimeout(5.0)
            while True:
                try:
                    packet, _ = sock.recvfrom(65535)
                    self.communicator.connection_status.emit("video", "ok")
                    frame = cv2.imdecode(np.frombuffer(packet, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None: self.communicator.video_frame_ready.emit(frame)
                except socket.timeout: self.communicator.connection_status.emit("video", "lost")

    def _event_receiver_worker(self):
        # ... (이전과 동일)
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((SERVER_IP, 8888))
                    self.communicator.connection_status.emit("event", "ok")
                    while True:
                        data = sock.recv(4096)
                        if not data: break
                        self.communicator.event_data_ready.emit(json.loads(data.decode('utf-8')))
            except Exception: self.communicator.connection_status.emit("event", "lost"); time.sleep(5)

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def handle_event(self, event_data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_type = event_data.get("event_type", "알 수 없음")
        color = self.event_colors.get(event_type, "white")
        self.add_log("black", f"<font color='{color}'>{timestamp} - [경고] {event_type} 상황 발생</font>")
        if event_data.get("alert_level") in ["HIGH", "MEDIUM"]:
            if self.alert_sound: self.alert_sound.play()
            AlertDialog(self, event_type).show()

    def resolve_alert(self, event_type):
        """상황 종료 버튼 클릭 시 호출되는 함수"""
        threading.Thread(target=self._resolve_alert_worker, args=(event_type,), daemon=True).start()

    def _resolve_alert_worker(self, event_type):
        """dataService와 통신을 처리하는 워커 스레드"""
        # 3. dataService로 알림 해제 신호 전송 및 로그
        self.communicator.log_message.emit("gray", f"-> '{event_type}' 상황 종료 신호를 dataService로 전송합니다.")
        payload = {"event_type": event_type, "action": "resolve"}
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((DATA_SERVICE_IP, 7777))
                s.sendall(json.dumps(payload).encode('utf-8'))
                response = s.recv(1024)
            # 4. dataService로부터 해제 확인 신호 수신 및 로그
            if json.loads(response.decode('utf-8')).get("status") == "OK":
                self.communicator.log_message.emit("green", f"<- '{event_type}' 상황 종료 확인을 dataService로부터 수신했습니다.")
        except Exception as e:
            self.communicator.log_message.emit("red", f"[오류] dataService 통신 실패: {e}")

    def add_log(self, color, message):
        self.log_browser.append(f"<font color='{color}'>{message}</font>")

    def update_connection_status(self, service, status):
        label = self.video_status_label if service == "video" else self.event_status_label
        if status == "ok": label.setText("<font color='green'>연결됨</font>")
        else: label.setText("<font color='red'>연결 끊김</font>")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.show()
    sys.exit(app.exec())