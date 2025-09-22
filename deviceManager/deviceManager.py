from threading import Thread
from datetime import datetime
import cv2
import numpy as np
import time
import os
from playsound import playsound
from Tcp_client_manager import Tcp_client_manager
import struct
import sys

class deviceManager(Tcp_client_manager):
    def __init__(self):
        super().__init__()
        self.file_path = os.path.dirname(__file__)
        self.before_video = list()
        self.after_video = list()
        self.capture_time = 15
        self.capture_frame = 25
        self.event_video = list()
        self.event_video_ready = False

    def event_data_send(self):
        
        while True:
            current_time = datetime.now()
            time_list = [current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second]
            
            if (self.alarm == 0):
                data = struct.pack("BBBIIIIIIBI", self.deviceManager_ID, self.situationDetector_ID, 1, time_list, 0)
                self.send_data(data)

            if (self.event_video_ready):
                data = struct.pack("BBBIIIIIIBI", self.deviceManager_ID, self.situationDetector_ID, 1, time_list, sys.getsizeof(self.event_video))
                self.send_data(data)

                with open("test.mp4", "rb") as video:
                    buffer = video.read()
                    print(buffer)
                    self.send_data_all(buffer)
                    self.event_video_ready = False
            else:
                pass
            time.sleep(0.1)

    def event_video_setup(self, frame):    
        if (len(self.before_video) < self.capture_frame*self.capture_time):
            self.before_video.append(frame)
        else:
            if (self.alarm == 0):
                self.before_video.pop(0)
                self.before_video.append(frame)
            else:
                if (len(self.after_video) < self.capture_frame*self.capture_time):
                    self.after_video.append(frame)
                else:
                    self.event_video = self.before_video + self.after_video
                    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

                    out = cv2.VideoWriter(self.file_path+f'/embedmediaStorage/{current_time}.avi', cv2.VideoWriter_fourcc(*'DIVX'), self.capture_frame, (640, 480))
                    for i in range(len(self.event_video)):
                        out.write(self.event_video[i])
                    self.before_video = self.after_video
                    self.after_video.clear()
                    self.event_video_ready = True 
        return

        # print(self.before_video)

    def alarm_set(self):
        while True:
            if self.alarm == 0:
                print("None")
            elif self.alarm == 1:
                playsound(self.file_path+'/embedmediaStorage/fire.mp3')
                print("화재용 LED")
            elif self.alarm == 2:
                playsound(self.file_path+'/embedmediaStorage/smoking.mp3')
                print("폭행 LED")
            elif self.alarm == 3:
                playsound(self.file_path+'/embedmediaStorage/fire.mp3')
                print("무단투기 LED")
            elif self.alarm == 4:
                playsound(self.file_path+'/embedmediaStorage/fire.mp3')
                print("흡연자 LED")
            else:
                print("Wrong Alarm Type")
                
            time.sleep(1)

    def backup_media(self):
        while True:
            try:
                cap = cv2.VideoCapture("http://192.168.0.180:5000/stream?src=0")
                time.sleep(1)
                if (cap.isOpened()):
                    break
            except Exception as e:
                print(f"영상 수신 오류: {e}")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(self.file_path+'/embedmediaStorage/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), self.capture_frame, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if (ret):
                out.write(frame)
                self.event_video_setup(frame)                    
                time.sleep(0.01)
                #!@ Need to Delete Start
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                #!@ Need to Delete End

            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def main(self):
        self.socket_init()

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