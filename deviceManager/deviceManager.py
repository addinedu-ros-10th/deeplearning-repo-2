from threading import Thread
from datetime import datetime
import cv2
import numpy as np
import time
import os
from Tcp_client_manager import Tcp_client_manager

class deviceManager(Tcp_client_manager):
    def __init__(self):
        super().__init__()
        self.file_path = os.path.dirname(__file__)
    
    def make_backup_dir(self):
        try:
            os.makedirs(self.file_path + "/video")
        except:
            pass

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

        out = cv2.VideoWriter(self.file_path+'/video/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (frame_width, frame_height))

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

        self.Tcp_thread = Thread(target=self.receive_data())
        self.Tcp_thread.daemon = True

        self.make_backup_dir()
        self.media_therad = Thread(target=self.backup_media)
        self.media_therad.daemon = True


        self.Tcp_thread.start()
        self.Tcp_thread.join()

        self.media_therad.start()
        self.media_therad.join()

if __name__ == "__main__":
    main = deviceManager()
    main.main()