from threading import Thread
from datetime import datetime
import cv2
import numpy as np
import time
import os
# from playsound import playsound
from Tcp_client_manager import Tcp_client_manager


class deviceManager(Tcp_client_manager):
    def __init__(self):
        super().__init__()
        self.file_path = os.path.dirname(__file__)

    
    # def alarm_set(self):
    #     while (self.alarm):
    #         if self.alarm == 1:
    #             playsound(self.file_path+'/embedmediaStorage/fire.mp3')
    #         elif self.alarm == 2:
    #             playsound(self.file_path+'/embedmediaStorage/fire.mp3')
    #         elif self.alarm == 3:
    #             playsound(self.file_path+'/embedmediaStorage/fire.mp3')
    #         elif self.alarm == 4:
    #             playsound(self.file_path+'/embedmediaStorage/fire.mp3')
    #         else:
    #             print("Wrong Alarm Type")
                
    #         time.sleep(0.1)

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

        out = cv2.VideoWriter(self.file_path+'/embedmediaStorage/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if (ret):
                out.write(frame)
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


        # self.Tcp_thread.start()
        self.media_thread.start()
        
        self.media_thread.join()
        self.Tcp_thread.join()

if __name__ == "__main__":
    main = deviceManager()
    main.main()