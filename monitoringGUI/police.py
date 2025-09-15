# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QMetaObject, Qt)
from PySide6.QtGui import (QFont)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QSizePolicy, QStatusBar,
    QWidget, QLineEdit, QPushButton, QVBoxLayout, QTextBrowser)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1280, 720)
        
        self.centralwidget = QWidget()
        self.gridLayout = QGridLayout(self.centralwidget)

        # Main layout
        self.main_layout = QHBoxLayout()
        
        # Left Panel: Video Container (placeholder)
        self.video_widget = QWidget()
        sizePolicy_video = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy_video.setHorizontalStretch(7)
        self.video_widget.setSizePolicy(sizePolicy_video)
        self.main_layout.addWidget(self.video_widget)

        # Right Panel: Control Panel
        self.control_panel = QWidget()
        sizePolicy_control = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy_control.setHorizontalStretch(3)
        self.control_panel.setSizePolicy(sizePolicy_control)
        self.control_layout = QVBoxLayout(self.control_panel)

        # Top: Log Open Button
        self.log_open_button = QPushButton(self.control_panel)
        self.log_open_button.setText("로그 열기 및 동영상 재생") # 버튼 텍스트 변경
        self.control_layout.addWidget(self.log_open_button)

        # Middle: Event Trigger Buttons
        self.button_layout = QVBoxLayout()
        self.assault_trigger_button = QPushButton("폭행 발생", self.control_panel)
        self.button_layout.addWidget(self.assault_trigger_button)
        self.fire_trigger_button = QPushButton("화재 발생", self.control_panel)
        self.button_layout.addWidget(self.fire_trigger_button)
        self.lying_down_trigger_button = QPushButton("누워있는 사람 발생", self.control_panel)
        self.button_layout.addWidget(self.lying_down_trigger_button)
        self.missing_trigger_button = QPushButton("실종자 발생", self.control_panel)
        self.button_layout.addWidget(self.missing_trigger_button)
        self.dumping_trigger_button = QPushButton("무단 투기 발생", self.control_panel)
        self.button_layout.addWidget(self.dumping_trigger_button)
        self.smoker_trigger_button = QPushButton("흡연자 발생", self.control_panel)
        self.button_layout.addWidget(self.smoker_trigger_button)
        self.control_layout.addLayout(self.button_layout)

        # Bottom: Info Panel
        self.info_label = QLabel("상황 정보", self.control_panel)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.info_label.setFont(font)
        self.control_layout.addWidget(self.info_label)

        self.login_status = QLabel("로그인 상태: 로그인됨", self.control_panel)
        self.control_layout.addWidget(self.login_status)

        self.log_browser = QTextBrowser(self.control_panel)
        self.control_layout.addWidget(self.log_browser)
        
        self.control_layout.addStretch()
        self.main_layout.addWidget(self.control_panel)

        self.gridLayout.addLayout(self.main_layout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"​AURA (Autonomous Urban Risk Analyzer)", None))
