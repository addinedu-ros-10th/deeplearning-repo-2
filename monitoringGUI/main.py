import sys
import cv2
import os
from datetime import datetime
from collections import deque
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QTextBrowser, QDialog, QGridLayout, QLineEdit, QPushButton, QDateTimeEdit,
                              QComboBox, QMessageBox, QListWidget)
from PySide6.QtCore import Qt, QTimer, QDateTime, QPoint
from PySide6.QtGui import QImage, QPixmap
import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# police.py 파일에서 UI 설계도를 가져옵니다.
try:
    from police import Ui_MainWindow
except ImportError:
    print("오류: police.py 파일을 찾을 수 없습니다. 같은 디렉토리에 있는지 확인해주세요.")
    sys.exit(1)

# --- 유틸리티 클래스들 ---
class ActionClassifier(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2):
        super(ActionClassifier, self).__init__(); self.cnn=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT); num_features=self.cnn.classifier[1].in_features; self.cnn.classifier=nn.Identity(); self.lstm=nn.LSTM(input_size=num_features,hidden_size=lstm_hidden_size,num_layers=lstm_layers,batch_first=True); self.classifier=nn.Sequential(nn.Linear(lstm_hidden_size,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,num_classes))
    def forward(self,x): b,c,t,h,w=x.shape; x=x.permute(0,2,1,3,4); x=x.reshape(b*t,c,h,w); features=self.cnn(x); features=features.view(b,t,-1); lstm_out,_=self.lstm(features); last_output=lstm_out[:,-1,:]; out=self.classifier(last_output); return out

class AspectRatioWidget(QWidget):
    def __init__(self, parent=None, aspect_ratio=16.0/9.0):
        super().__init__(parent); self.aspect_ratio=aspect_ratio; self.video_label=QLabel(self); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setStyleSheet("background-color: rgb(40, 40, 40); color: white;"); layout=QVBoxLayout(self); layout.addWidget(self.video_label); self.setLayout(layout)
    def resizeEvent(self,event):
        super().resizeEvent(event); w=self.width(); h=self.height()
        if w/h > self.aspect_ratio: new_w=int(h*self.aspect_ratio); self.video_label.setGeometry((w-new_w)//2,0,new_w,h)
        else: new_h=int(w/self.aspect_ratio); self.video_label.setGeometry(0,(h-new_h)//2,w,new_h)

class RollingRecorder:
    def __init__(self, output_dir="recordings", chunk_seconds=10, total_duration_seconds=60, fourcc='mp4v', fps=30, frame_size=(640,480)):
        self.output_dir=output_dir; self.chunk_seconds=chunk_seconds; self.total_duration_seconds=total_duration_seconds; self.fourcc=cv2.VideoWriter_fourcc(*fourcc); self.fps=fps; self.frame_size=frame_size; self.max_chunks=self.total_duration_seconds//self.chunk_seconds; self.file_deque=deque(maxlen=self.max_chunks); self.out=None; self.frame_count=0; self.frames_per_chunk=self.chunk_seconds*self.fps; os.makedirs(self.output_dir,exist_ok=True)
    def _start_new_chunk(self):
        if len(self.file_deque)==self.max_chunks: oldest_file=self.file_deque.popleft(); os.remove(oldest_file)
        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); chunk_path=os.path.join(self.output_dir,f"chunk_{timestamp}.mp4"); self.out=cv2.VideoWriter(chunk_path,self.fourcc,self.fps,self.frame_size); self.file_deque.append(chunk_path)
    def write_frame(self,frame):
        if self.out is None or self.frame_count>=self.frames_per_chunk:
            if self.out is not None: self.out.release()
            self._start_new_chunk(); self.frame_count=0
        self.out.write(frame); self.frame_count+=1
    def stop(self):
        if self.out is not None: self.out.release(); self.out=None

class AlertDialog(QDialog):
    def __init__(self,parent,event_type,prob):
        super().__init__(parent); self.setWindowTitle(f"{event_type} 경고"); self.setModal(False); self.setWindowModality(Qt.NonModal); self.setWindowFlags(self.windowFlags()&~Qt.WindowCloseButtonHint|Qt.CustomizeWindowHint); self.setAttribute(Qt.WA_DeleteOnClose,True); self.event_type=event_type; self.parent=parent; self.setFixedSize(300,150); layout=QVBoxLayout(); self.setLayout(layout); message=QLabel(f"{event_type} 감지됨! (확률: {prob:.2f})"); layout.addWidget(message); stop_button=QPushButton("상황 종료"); layout.addWidget(stop_button); stop_button.clicked.connect(self.confirm_stop)
    def confirm_stop(self):
        reply=QMessageBox.question(self,"확인","상황을 종료하시겠습니까?",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        if reply==QMessageBox.StandardButton.Yes:
            if self.parent.alert_channel: self.parent.alert_channel.stop()
            self.parent.alert_active=False; timestamp=QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"); termination_message=f"{timestamp} - {self.event_type} 상황 종료됨"; self.parent.ui.log_browser.append(termination_message); self.parent.log_entries.append({"timestamp":timestamp,"message":termination_message}); self.accept()

# --- 로그 보기 및 동영상 재생을 위한 새 다이얼로그 클래스 ---
class LogViewerDialog(QDialog):
    def __init__(self, parent=None, log_entries=None):
        super().__init__(parent)
        self.setWindowTitle("로그 뷰어 및 동영상 재생")
        self.setMinimumSize(1000, 700)
        self.log_entries = log_entries or []
        self.video_capture = None
        self.playing_video = False

        # --- UI 구성 ---
        main_layout = QHBoxLayout(self)
        
        # Left: Video playback
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        self.video_display = AspectRatioWidget(aspect_ratio=16.0/9.0)
        self.video_display.video_label.setText("재생할 동영상을 선택하세요.")
        video_layout.addWidget(self.video_display)
        main_layout.addWidget(video_panel, 7)

        # Right: Controls and logs
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        control_layout.addWidget(QLabel("저장된 동영상 목록"))
        self.video_list = QListWidget()
        control_layout.addWidget(self.video_list)
        
        self.play_button = QPushButton("재생")
        control_layout.addWidget(self.play_button)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("키워드 검색")
        search_layout.addWidget(self.search_input)
        self.search_button = QPushButton("로그 검색")
        search_layout.addWidget(self.search_button)
        control_layout.addLayout(search_layout)

        self.log_browser = QTextBrowser()
        control_layout.addWidget(self.log_browser)
        main_layout.addWidget(control_panel, 3)

        # --- 타이머 및 시그널 연결 ---
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.play_button.clicked.connect(self.play_selected_video)
        self.search_button.clicked.connect(self.search_logs)
        
        self.update_video_list()
        self.search_logs() # 처음 열 때 전체 로그 표시

    def update_video_list(self):
        self.video_list.clear()
        if os.path.exists("recordings"):
            video_files = sorted([f for f in os.listdir("recordings") if f.endswith('.mp4')], reverse=True)
            self.video_list.addItems(video_files)

    def play_selected_video(self):
        if self.playing_video:
            self.video_timer.stop()
            if self.video_capture: self.video_capture.release()
            self.playing_video = False
            self.play_button.setText("재생")
            return

        selected_item = self.video_list.currentItem()
        if selected_item:
            video_path = os.path.join("recordings", selected_item.text())
            self.video_capture = cv2.VideoCapture(video_path)
            if self.video_capture.isOpened():
                fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
                self.playing_video = True
                self.play_button.setText("정지")
                self.video_timer.start(int(1000 / fps))
            else:
                QMessageBox.warning(self, "오류", f"영상을 열 수 없습니다: {video_path}")

    def update_video_frame(self):
        if self.playing_video and self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_display.video_label.setPixmap(QPixmap.fromImage(qt_image))
            else:
                self.video_timer.stop()
                self.video_capture.release()
                self.playing_video = False
                self.play_button.setText("재생")

    def search_logs(self):
        keyword = self.search_input.text().strip().lower()
        self.log_browser.clear()
        for entry in self.log_entries:
            if not keyword or keyword in entry['message'].lower():
                self.log_browser.append(f"{entry['timestamp']} - {entry['message']}")
    
    def closeEvent(self, event):
        if self.video_capture: self.video_capture.release()
        self.video_timer.stop()
        super().closeEvent(event)

# --- 메인 애플리케이션 클래스 ---
class PatrolDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        pygame.mixer.init()
        self.alert_sound=None; self.alert_channel=None; self.alert_active=False; self.open_alerts=[]
        
        placeholder=self.ui.video_widget
        container_layout=self.ui.main_layout
        index=container_layout.indexOf(placeholder)
        container_layout.removeWidget(placeholder)
        placeholder.hide(); placeholder.deleteLater()
        self.video_widget=AspectRatioWidget(aspect_ratio=16.0/9.0)
        container_layout.insertWidget(index,self.video_widget)
        container_layout.setStretch(index,7); container_layout.setStretch(1,3)
        
        self.video_capture=cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.video_capture=None
            self.video_widget.video_label.setText("카메라를 열 수 없습니다.")
        
        frame_width, frame_height, fps = 640, 480, 30
        if self.video_capture:
            frame_width=int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height=int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps=int(self.video_capture.get(cv2.CAP_PROP_FPS)) or 30

        self.recorder=None
        if self.video_capture: self.recorder=RollingRecorder(frame_size=(frame_width,frame_height),fps=fps)
        
        self.log_entries=[]
        
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000//fps)
        
        self.ui.log_open_button.clicked.connect(self.open_log_viewer)
        self.ui.assault_trigger_button.clicked.connect(lambda:self.trigger_event("폭행"))
        self.ui.fire_trigger_button.clicked.connect(lambda:self.trigger_event("화재"))
        self.ui.lying_down_trigger_button.clicked.connect(lambda:self.trigger_event("누워있는 사람"))
        self.ui.missing_trigger_button.clicked.connect(lambda:self.trigger_event("실종자"))
        self.ui.dumping_trigger_button.clicked.connect(lambda:self.trigger_event("무단 투기"))
        self.ui.smoker_trigger_button.clicked.connect(lambda:self.trigger_event("흡연자"))
        
        self.cascade_pos=QPoint(50,50); self.cascade_offset=QPoint(40,40); self.initial_cascade_pos_x=50

    def open_log_viewer(self):
        self.log_viewer = LogViewerDialog(self, self.log_entries)
        self.log_viewer.show()

    def update_frame(self):
        if not self.video_capture: return
        
        ret, frame = self.video_capture.read()
        if not ret: return
            
        if self.recorder: self.recorder.write_frame(frame)
        
        rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb_image.shape
        bytes_per_line=ch*w
        qt_image=QImage(rgb_image.data,w,h,bytes_per_line,QImage.Format.Format_RGB888)
        self.video_widget.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def trigger_event(self, event_type):
        timestamp=QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        prob=random.random()
        log_message=f"{event_type} 감지됨 (확률: {prob:.2f})"
        self.ui.log_browser.append(f"{timestamp} - {log_message}")
        self.log_entries.append({"timestamp":timestamp,"message":log_message})
        if event_type in["폭행","화재","누워있는 사람"]:
            if not hasattr(self, 'alert_sound'):
                self.alert_sound_file="alert.wav"
                if os.path.exists(self.alert_sound_file): self.alert_sound=pygame.mixer.Sound(self.alert_sound_file); self.alert_channel=pygame.mixer.Channel(0)
                else: self.alert_sound = None
            if self.alert_sound and self.alert_channel and not self.alert_active: self.alert_channel.play(self.alert_sound,loops=-1); self.alert_active=True
            dialog=AlertDialog(self,event_type,prob)
            screen_rect=QApplication.primaryScreen().geometry()
            if self.cascade_pos.y()+dialog.height()>screen_rect.height(): self.initial_cascade_pos_x+=self.cascade_offset.x()*2; self.cascade_pos.setX(self.initial_cascade_pos_x); self.cascade_pos.setY(50)
            if self.cascade_pos.x()+dialog.width()>screen_rect.width(): self.initial_cascade_pos_x=50; self.cascade_pos.setX(self.initial_cascade_pos_x); self.cascade_pos.setY(50)
            dialog.move(self.cascade_pos); dialog.show(); self.open_alerts.append(dialog); self.cascade_pos+=self.cascade_offset
            dialog.finished.connect(lambda result,d=dialog:self.on_alert_finished(d))

    def on_alert_finished(self, dialog):
        if dialog in self.open_alerts: self.open_alerts.remove(dialog)
        if not self.open_alerts and self.alert_channel: self.alert_channel.stop(); self.alert_active=False

    def closeEvent(self, event):
        if self.recorder: self.recorder.stop()
        if self.video_capture: self.video_capture.release()
        pygame.mixer.quit()
        if hasattr(self, 'log_viewer') and self.log_viewer.isVisible():
            self.log_viewer.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatrolDashboard()
    window.setWindowState(Qt.WindowState.WindowMaximized)
    window.show()
    sys.exit(app.exec())