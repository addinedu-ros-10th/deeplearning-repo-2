import socket
import time
import queue
import threading

TCP_HOST = '192.168.0.180' # deviceManager IP 주소
TCP_PORT = 1201            # deviceManager TCP 발신 포트 주소

def send_tcp_data_to_dm(event_video_queue : queue.Queue,
                        ):
  """
  Alarm type을 받아서 다음 데이터를 deviceManager에 발신하는 기능
  1. source
  2. Destination
  3. Patrol number
  4. Alarm type
  """
  