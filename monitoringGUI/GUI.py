import sys
import cv2
import os
from datetime import datetime
from collections import deque
import random
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                               QTextBrowser, QDialog, QGridLayout, QLineEdit, QPushButton,
                               QComboBox, QMessageBox, QListWidget, QSizePolicy, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QFont

# --- 유틸리티 및 AI 모델 클래스 (이전과 동일) ---

class ActionClassifier(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2):
        super(ActionClassifier, self).__init__(); self.cnn=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT); num_features=self.cnn.classifier[1].in_features; self.cnn.classifier=nn.Identity(); self.lstm=nn.LSTM(input_size=num_features,hidden_size=lstm_hidden_size,num_layers=lstm_layers,batch_first=True); self.classifier=nn.Sequential(nn.Linear(lstm_hidden_size,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,num_classes))
    def forward(self,x):
        b,c,t,h,w=x.shape; x=x.permute(0,2,1,3,4); x=x.reshape(b*t,c,h,w); features=self.cnn(x); features=features.view(b,t,-1); lstm_out,_=self.lstm(features); out=self.classifier(lstm_out[:,-1,:]); return out

class AspectRatioWidget(QWidget):
    def __init__(self, parent=None, aspect_ratio=16.0/9.0):
        super().__init__(parent); self.aspect_ratio=aspect_ratio; self.video_label=QLabel(self); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setStyleSheet("background-color: rgb(40, 40, 40); color: white;"); layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.video_label); self.setLayout(layout)
    def resizeEvent(self,event):
        super().resizeEvent(event); w=self.width(); h=self.height()
        if w/h > self.aspect_ratio: new_w=int(h*self.aspect_ratio); self.video_label.setGeometry((w-new_w)//2,0,new_w,h)
        else: new_h=int(w/self.aspect_ratio); self.video_label.setGeometry(0,(h-new_h)//2,w,new_h)

class RollingRecorder:
    def __init__(self, output_dir="recordings", chunk_seconds=10, fps=30, frame_size=(1280, 720)):
        self.output_dir=output_dir; self.chunk_seconds=chunk_seconds; self.fourcc=cv2.VideoWriter_fourcc(*'mp4v'); self.fps=fps; self.frame_size=frame_size; self.file_deque=deque(maxlen=6); self.out=None; self.frame_count=0; self.frames_per_chunk=self.chunk_seconds*self.fps; os.makedirs(self.output_dir,exist_ok=True)
    def _start_new_chunk(self):
        if len(self.file_deque) == self.file_deque.maxlen:
            try: os.remove(self.file_deque.popleft())
            except OSError: pass
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); chunk_path=os.path.join(self.output_dir,f"chunk_{timestamp}.mp4"); self.out=cv2.VideoWriter(chunk_path,self.fourcc,self.fps,self.frame_size); self.file_deque.append(chunk_path)
    def write_frame(self,frame):
        if self.out is None or self.frame_count>=self.frames_per_chunk:
            if self.out: self.out.release()
            self._start_new_chunk(); self.frame_count=0
        if frame.shape[1]!=self.frame_size[0] or frame.shape[0]!=self.frame_size[1]: frame=cv2.resize(frame, self.frame_size)
        self.out.write(frame); self.frame_count+=1
    def stop(self):
        if self.out: self.out.release(); self.out=None

class AlertDialog(QDialog):
    def __init__(self, parent, event_type, prob):
        super().__init__(parent); self.setWindowTitle(f"{event_type} 경고"); self.setModal(False); self.setWindowModality(Qt.NonModal); self.setWindowFlags(self.windowFlags()&~Qt.WindowCloseButtonHint|Qt.CustomizeWindowHint); self.setAttribute(Qt.WA_DeleteOnClose,True); self.event_type=event_type; self.parent_dashboard=parent; self.setFixedSize(300,150); layout=QVBoxLayout(self); message=QLabel(f"{event_type} 감지됨! (확률: {prob:.2f})"); layout.addWidget(message); stop_button=QPushButton("상황 종료"); layout.addWidget(stop_button); stop_button.clicked.connect(self.confirm_stop)
    def confirm_stop(self):
        if QMessageBox.question(self,"확인","상황을 종료하시겠습니까?",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)==QMessageBox.StandardButton.Yes: self.parent_dashboard.resolve_alert(self.event_type); self.accept()

class LogViewerDialog(QDialog):
    def __init__(self,parent=None,log_entries=None,event_colors=None): super().__init__(parent); self.setWindowTitle("로그 뷰어 및 동영상 재생"); self.setMinimumSize(1000,700); self.log_entries=log_entries or[]; self.event_colors=event_colors or{}; self.video_capture=None; self.playing_video=False; self.setupUi(); self.connect_signals(); self.update_video_list(); self.filter_and_display_logs()
    def setupUi(self): main_layout=QHBoxLayout(self); video_panel=QWidget(); video_layout=QVBoxLayout(video_panel); self.video_display=AspectRatioWidget(); self.video_display.video_label.setText("재생할 동영상을 선택하세요."); video_layout.addWidget(self.video_display); main_layout.addWidget(video_panel,7); control_panel=QWidget(); control_layout=QVBoxLayout(control_panel); control_layout.addWidget(QLabel("저장된 동영상 목록")); self.video_list=QListWidget(); control_layout.addWidget(self.video_list); self.play_button=QPushButton("재생"); control_layout.addWidget(self.play_button); filter_layout=QGridLayout(); self.filter_event_type=QComboBox(); self.filter_event_type.addItem("모든 이벤트"); self.filter_event_type.addItems(self.event_colors.keys()); filter_layout.addWidget(QLabel("이벤트 유형:"),0,0); filter_layout.addWidget(self.filter_event_type,0,1); self.search_input=QLineEdit(); self.search_input.setPlaceholderText("키워드 검색"); filter_layout.addWidget(self.search_input,1,0,1,2); self.search_button=QPushButton("로그 검색"); filter_layout.addWidget(self.search_button,2,0,1,2); control_layout.addLayout(filter_layout); control_layout.addWidget(QLabel("필터링된 로그")); self.log_browser=QTextBrowser(); control_layout.addWidget(self.log_browser); main_layout.addWidget(control_panel,3)
    def connect_signals(self): self.video_timer=QTimer(self); self.video_timer.timeout.connect(self.update_video_frame); self.play_button.clicked.connect(self.play_selected_video); self.search_button.clicked.connect(self.filter_and_display_logs); self.filter_event_type.currentIndexChanged.connect(self.filter_and_display_logs)
    def update_video_list(self): self.video_list.clear(); self.video_list.addItems(sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')],reverse=True))
    def play_selected_video(self):
        if self.playing_video: self.video_timer.stop(); self.video_capture.release(); self.playing_video=False; self.play_button.setText("재생"); return
        if item:=self.video_list.currentItem():
            path=os.path.join("recordings",item.text()); self.video_capture=cv2.VideoCapture(path)
            if self.video_capture.isOpened(): fps=self.video_capture.get(cv2.CAP_PROP_FPS)or 30; self.playing_video=True; self.play_button.setText("정지"); self.video_timer.start(int(1000/fps))
    def update_video_frame(self):
        if self.playing_video and self.video_capture:
            ret,frame=self.video_capture.read()
            if ret: rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,ch=rgb_image.shape; qt_image=QImage(rgb_image.data,w,h,w*ch,QImage.Format_RGB888); self.video_display.video_label.setPixmap(QPixmap.fromImage(qt_image))
            else: self.video_timer.stop(); self.video_capture.release(); self.playing_video=False; self.play_button.setText("재생")
    def filter_and_display_logs(self):
        keyword=self.search_input.text().strip().lower(); event_type=self.filter_event_type.currentText(); self.log_browser.clear()
        for entry in self.log_entries:
            if (event_type=="모든 이벤트" or event_type in entry['message']) and (not keyword or keyword in entry['message'].lower()):
                self.log_browser.append(f"{entry['timestamp']} - {entry['message']}")
    def closeEvent(self,event): self.video_timer.stop(); self.video_capture.release() if self.video_capture else None; super().closeEvent(event)

class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) # UI 프로그래밍 방식으로 생성
        pygame.mixer.init()
        self.alert_sound, self.alert_channel, self.alert_active = None, None, False
        self.open_alerts = []
        self.log_entries = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dumping_model = self.smoker_model = None # 실제 모델 파일 로딩은 비활성화
        self.transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        self.frame_buffer = deque(maxlen=16)
        self.frame_counter = 0

        self.recorder = RollingRecorder()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)

        self.event_colors = {"화재":"red","폭행":"orange","누워있는 사람":"yellow","실종자":"cyan","무단 투기":"lightgray","흡연자":"lightgray"}
        if os.path.exists("alert.wav"): self.alert_sound=pygame.mixer.Sound("alert.wav"); self.alert_channel=pygame.mixer.Channel(0)
        
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("AURA 관제 대시보드 (단일 화면)")
        MainWindow.resize(1600, 900)
        
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)

        # --- 수정: 탭 위젯을 단일 비디오 위젯으로 변경 ---
        self.video_widget = AspectRatioWidget()
        self.main_layout.addWidget(self.video_widget, 7) # 전체의 70% 차지

        # Right: Control Panel
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        self.main_layout.addWidget(control_panel, 3) # 전체의 30% 차지

        self.log_open_button = QPushButton("로그 열기 및 동영상 재생")
        self.control_layout.addWidget(self.log_open_button)

        trigger_groupbox = QGroupBox("수동 이벤트 발생")
        button_layout = QVBoxLayout(trigger_groupbox)
        self.trigger_buttons = {}
        event_names = ["폭행","화재","누워있는 사람","실종자","무단 투기","흡연자"]
        for name in event_names:
            btn = QPushButton(f"{name} 발생")
            self.trigger_buttons[name] = btn
            button_layout.addWidget(btn)
        self.control_layout.addWidget(trigger_groupbox)

        # --- 수정: 탭 위젯을 단일 로그 브라우저로 변경 ---
        log_label = QLabel("통합 이벤트 로그")
        font = QFont(); font.setPointSize(12); font.setBold(True)
        log_label.setFont(font)
        self.control_layout.addWidget(log_label)
        
        self.log_browser = QTextBrowser()
        self.control_layout.addWidget(self.log_browser)
        
        # Connect Signals
        self.log_open_button.clicked.connect(self.open_log_viewer)
        for name, btn in self.trigger_buttons.items():
            btn.clicked.connect(lambda checked, n=name: self.trigger_event(n))
    
    def update_frame(self):
        # 가상 비디오 프레임 생성
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, f"VIRTUAL STREAM @ {now_str}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1); cv2.putText(frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.recorder.write_frame(frame)
        
        # --- 수정: 단일 비디오 위젯에만 프레임 표시 ---
        rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,ch=rgb_image.shape
        qt_image = QImage(rgb_image.data,w,h,w*ch,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_widget.video_label.setPixmap(pixmap)
            
        # AI 추론 시뮬레이션
        self.frame_buffer.append(self.preprocess_frame(frame)); self.frame_counter+=1
        if self.frame_counter >= 30: # 1초에 한번씩 추론 시뮬레이션
            if random.random() < 0.1: # 10% 확률로 무단투기 자동 감지
                self.run_inference(self.dumping_model, "무단 투기")
            if random.random() < 0.1: # 10% 확률로 흡연자 자동 감지
                self.run_inference(self.smoker_model, "흡연자")
            self.frame_counter = 0
            
    def preprocess_frame(self, frame): return self.transform(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))

    def run_inference(self, model, event_type):
        # 실제 모델 추론 대신 랜덤 확률로 시뮬레이션
        prob = random.uniform(0.7, 0.99)
        self.trigger_event(event_type, prob, is_auto=True)

    def trigger_event(self, event_type, prob=None, is_auto=False):
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prob = prob if prob is not None else random.random()
        color = self.event_colors.get(event_type, "white")
        log_message = f"[{'자동' if is_auto else '수동'}] {event_type} 감지됨 (확률: {prob:.2f})"
        
        # --- 수정: 단일 통합 로그에만 기록 ---
        html_log = f"<font color='{color}'>{timestamp} - {log_message}</font>"
        self.log_browser.append(html_log)
        self.log_entries.append({"timestamp": timestamp, "message": log_message})
        
        if event_type in ["폭행","화재","누워있는 사람"]:
            if self.alert_sound and self.alert_channel and not self.alert_active:
                self.alert_channel.play(self.alert_sound, loops=-1); self.alert_active=True
            dialog=AlertDialog(self,event_type,prob); self.open_alerts.append(dialog)
            self.place_alert_dialog(dialog); dialog.show()

    def place_alert_dialog(self, dialog):
        screen_rect=QApplication.primaryScreen().geometry(); margin=20; d_w, d_h=300, 150
        positions = {(d.pos().x(), d.pos().y()) for d in self.open_alerts if d is not dialog}
        for r in range((screen_rect.height() - margin) // (d_h + margin)):
            for c in range((screen_rect.width() - margin) // (d_w + margin)):
                x = margin + c * (d_w + margin); y = margin + r * (d_h + margin)
                if (x, y) not in positions: dialog.move(x, y); return
        dialog.move(margin, margin)

    def resolve_alert(self, event_type):
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"); log_message=f"<font color='green'>{timestamp} - {event_type} 상황 종료됨</font>"
        self.log_browser.append(log_message); self.log_entries.append({"timestamp":timestamp, "message":f"{event_type} 상황 종료됨"})
        QTimer.singleShot(100, self.check_and_stop_sound)

    def check_and_stop_sound(self):
        if not any(isinstance(w,AlertDialog)and w.isVisible()for w in QApplication.topLevelWidgets()) and self.alert_active:
            if self.alert_channel: self.alert_channel.stop()
            self.alert_active=False
            
    def open_log_viewer(self):
        LogViewerDialog(self, self.log_entries, self.event_colors).exec()
        
    def closeEvent(self,event):
        self.timer.stop(); self.recorder.stop(); pygame.mixer.quit()
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog): w.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.show()
    sys.exit(app.exec())