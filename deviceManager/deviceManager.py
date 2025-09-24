from threading import Thread
from datetime import datetime
import cv2
import numpy as np
import time
import os
import pygame
from Tcp_client_manager import Tcp_client_manager
import struct
import sys
import pickle
import multiprocessing

class deviceManager(Tcp_client_manager):
    def __init__(self):
        super().__init__()
        self.file_path = os.path.dirname(__file__)
        self.before_video = list()
        self.after_video = list()
        self.capture_time = 15
        self.capture_frame = 30
        self.event_video = list()
        self.event_video_ready = False

    def event_data_send(self):
        time_list = [self.current_time.year, self.current_time.month, self.current_time.day, self.current_time.hour, self.current_time.minute, self.current_time.second]
        
        if (self.alarm == 0):
            return

        if (self.event_video_ready):
            data = struct.pack("BBBIIIIIII", self.deviceManager_ID, self.situationDetector_ID, 1, *time_list, sys.getsizeof(self.event_video))
            self.send_data(data)

            with open(self.file_path+f"/embedmediaStorage/{self.current_time.strftime("%Y%m%d%H%M%S")}.avi", "rb") as video:
                buffer = video.read(1024)
                while (buffer):
                    self.send_data(buffer)
                    buffer = video.read(1024)
                self.send_data(b"DONE")
            self.event_video_ready = False
        else:
            return

    def event_video_setup(self, frame):    
        if (len(self.before_video) < self.capture_frame*self.capture_time - 200):
            self.before_video.append(frame)
        else:
            if (self.alarm == 0):
                self.after_video.clear()
                self.before_video.pop(0)
                self.before_video.append(frame)
            else:
                if (len(self.after_video) < self.capture_frame*self.capture_time + 200):
                    self.after_video.append(frame)
                    
                else:
                    self.event_video = self.before_video + self.after_video
                    self.current_time = datetime.now()

                    out = cv2.VideoWriter(self.file_path+f'/embedmediaStorage/{self.current_time.strftime("%Y%m%d%H%M%S")}.avi', cv2.VideoWriter_fourcc(*'DIVX'), self.capture_frame, (640, 480))
                    for i in range(len(self.event_video)):
                        out.write(self.event_video[i])
                    
                    print(len(self.before_video), len(self.after_video))
                    self.before_video = self.after_video[400:]
                    self.after_video.clear()
                    self.event_video_ready = True
                    
                    self.event_data_send()
                    out.release()
        return
    
    def alarm_set(self):
        pygame.mixer.init()
        pygame.mixer.set_num_channels(1)
        sound = pygame.mixer.Channel(0)
        while True:
            if self.alarm == 0:
                print("None")
                continue

            elif self.alarm == 1:
                alarm = pygame.mixer.Sound(self.file_path+'/embedmediaStorage/fire.mp3')
                print("화재용 LED")
            elif self.alarm == 2:
                alarm = pygame.mixer.Sound(self.file_path+'/embedmediaStorage/fight.mp3')
                print("폭행 LED")
            elif self.alarm == 3:
                alarm = pygame.mixer.Sound(self.file_path+'/embedmediaStorage/trash.mp3')
                print("무단투기 LED")
            elif self.alarm == 4:
                alarm = pygame.mixer.Sound(self.file_path+'/embedmediaStorage/smoke.mp3')
                print("흡연자 LED")
            else:
                print("Wrong Alarm Type")

            if (sound.get_busy()):
                pass
            else:
                sound.play(alarm)
            time.sleep(0.1)

    def media_init(self):
        while True:
            try:
                print("VideoCapture Waiting for Broadcasting Server")
                self.cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")
                time.sleep(1)
                if (self.cap.isOpened()):
                    print("Video Capture Start")
                    break
                
            except Exception as e:
                print(f"영상 수신 오류: {e}")

    def backup_media(self):
        while True:
            ret, frame = self.cap.read()
            if (ret):
                self.event_video_setup(frame)                  
                time.sleep(1/self.capture_frame)

            else:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def main(self):
        self.socket_init()
        self.media_init()

        self.Tcp_thread = Thread(target=self.receive_data)
        self.Tcp_thread.daemon = True

        self.media_thread = Thread(target=self.backup_media)
        self.media_thread.daemon = True

        self.alarm_thread = Thread(target=self.alarm_set)
        self.alarm_thread.daemon = True

        self.Tcp_thread.start()
        self.media_thread.start()
        self.alarm_thread.start()

        self.media_thread.join()
        self.Tcp_thread.join()
        self.alarm_thread.join()


if __name__ == "__main__":
    main = deviceManager()
    main.main()