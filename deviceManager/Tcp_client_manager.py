import socket
import json
from datetime import datetime
import struct
import time

class Tcp_client_manager():
    def __init__(self):
        self.TCP_HOST = "192.168.0.86"
        # self.TCP_HOST = "172.20.10.8"
        self.TCP_PORT = 1201
        self.PATROL_NUMBER = 1
        self.deviceManager_ID = 0x01
        self.situationDetector_ID = 0x02
        self.recv_data = ""
        self.alarm = 0
        self.last_alarm = 0
        
    def socket_init(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while (not connected):
            try:
                print("Socket Waiting for Connection from situationDetector ")
                self.client_socket.connect((self.TCP_HOST, self.TCP_PORT))
                connected = True
                print("situationDetector Connected")
            except Exception as e:
                print(f"Socket Initiate Error Occured : {e}")
                time.sleep(5)

    def send_data(self, data):
        self.client_socket.send(data)

    def send_data_all(self, data):
        self.client_socket.sendall(data)
        
    def receive_data(self):
        data_size = 4
        while True:
            try:
                self.client_socket.settimeout(2.0)
                self.recv_data = self.client_socket.recv(data_size)
                self.data_validation()
                time.sleep(0.1)
            except Exception:
                pass    
            except KeyboardInterrupt:
                print("Keyboard Interruption")
                break
    
    def data_validation(self):
        if (self.recv_data[0] != self.situationDetector_ID):
            print("Wrong Data Source Input")
            self.recv_data = ""
            return
        if (self.recv_data[1] != self.deviceManager_ID):
            print("Wrong Data Destination Input")
            self.recv_data = ""
            return
        if (self.recv_data[2] != self.PATROL_NUMBER):
            print("Wrong Patrol Number")
            self.recv_data = ""
            return
        self.alarm = int(self.recv_data[3])
        
        if (self.alarm != 0):
            self.last_alarm = self.alarm


    def socket_close(self):
        self.client_socket.close()

    def __exit__(self):
        print("tcp close")
        self.socket_close()