# server/data_service_tcp_server.py

import socket
import pickle
import struct
import data_service
import os

def send_msg(sock, data):
    try:
        pickled_data = pickle.dumps(data)
        msg = struct.pack('L', len(pickled_data)) + pickled_data
        sock.sendall(msg)
        return True
    except Exception as e:
        print(f"메시지 전송 실패: {e}")
        return False

def recv_msg(sock):
    try:
        packed_len = sock.recv(struct.calcsize('L'))
        if not packed_len: return None
        msg_len = struct.unpack('L', packed_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(msg_len - len(data))
            if not packet: return None
            data += packet
        return pickle.loads(data)
    except Exception as e:
        print(f"메시지 수신 실패: {e}")
        return None

def handle_client_connection(client_socket, addr):
    """클라이언트와의 전체 통신을 처리하는 함수 (파일 스트리밍 기능 개선)"""
    print(f"{addr}에서 클라이언트가 접속했습니다.")
    is_streaming = False
    try:
        while True:
            # 스트리밍 중이 아닐 때만 pickle 메시지를 기다림
            if not is_streaming:
                request = recv_msg(client_socket)
                if request is None: break
                
                print(f"수신한 요청: {request}")
                
                command = request.get('command')
                params = request.get('params', {})
                
                if command == 'get_logs':
                    response_data = data_service.get_logs(**params)
                    send_msg(client_socket, response_data)

                elif command == 'get_video_path':
                    path = data_service.get_video_path_for_log(**params)
                    
                    if path and os.path.exists(path):
                        file_size = os.path.getsize(path)
                        # 1. 스트리밍 시작을 알리는 성공 메시지 전송
                        start_response = {"status": "streaming_start", "size": file_size}
                        send_msg(client_socket, start_response)
                        is_streaming = True # 스트리밍 모드 활성화

                        # 2. 실제 파일 데이터 스트리밍
                        print(f"'{path}' 파일 스트리밍 시작...")
                        with open(path, 'rb') as f:
                            while True:
                                chunk = f.read(4096)
                                if not chunk: break
                                client_socket.sendall(chunk)
                        
                        # 3. 종료 마커 전송
                        client_socket.sendall(b'DONE')
                        print("파일 스트리밍 완료.")
                        break # 파일 전송 후 연결 종료

                    else:
                        error_response = {"error": "서버에 해당 영상 파일이 존재하지 않습니다."}
                        send_msg(client_socket, error_response)
                else:
                    error_response = {"error": "알 수 없는 명령어입니다."}
                    send_msg(client_socket, error_response)

    except (ConnectionResetError, ConnectionError) as e:
        print(f"{addr}에서 연결이 끊어졌습니다: {e}")
    except Exception as e:
        print(f"통신 중 에러 발생: {e}")
    finally:
        print(f"{addr} 클라이언트 접속이 종료되었습니다.")
        client_socket.close()



def run_server():
    host = '0.0.0.0'
    port = 3401
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"TCP 서버가 {host}:{port}에서 실행 중입니다...")
    while True:
        client_socket, addr = server_socket.accept()
        handle_client_connection(client_socket, addr)

if __name__ == '__main__':
    run_server()