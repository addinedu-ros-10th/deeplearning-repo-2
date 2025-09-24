# tcp_client.py

# test_webcam 에서 import 함

import socket
import json
import logging

def send_event_data(data_json, host='localhost', port=2401):
    """
    이벤트 데이터를 JSON으로 변환하여 TCP 서버에 전송합니다.
    """
    try:
        # JSON 객체를 문자열로 변환 (UTF-8 인코딩)
        message = json.dumps(data_json)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(message.encode('utf-8'))
            logging.info(f"Successfully sent event data to {host}:{port}")
            # print("Sent:", message) # 디버깅 시 사용
            
    except ConnectionRefusedError:
        logging.error(f"Connection refused. Is the server running on {host}:{port}?")
    except Exception as e:
        logging.error(f"Failed to send data: {e}")