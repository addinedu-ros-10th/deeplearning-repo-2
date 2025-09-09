# fire-detect-server.py
import socket
import json
import cv2
import numpy as np
import base64

"""
기능
1. 소켓 생성 / 소켓 옵션 설정
2. (주소, 포트)에 소켓 할당
3. 클라이언트 연결 대기 및 연결 수락
4. 데이터 수신
3. 수신한 데이터 변환 (byte) -> (str) -> (json)

"""

HOST = 'localhost' # 로컬호스트 (테스트환경)
PORT = 6600 # 6600번 포트 사용

def main():
  # 1. 소켓 생성
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # 2. 소켓 옵션 설정
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 3. bind : (주소, 포트)에 소켓 할당
    s.bind((HOST, PORT))
    print(f"서버 : 소켓을 ({HOST}:{PORT})에 할당")

    # 4. listen : 클라이언트의 연결을 기다림 (대기)
    s.listen()
    print(f"서버 : ({HOST}:{PORT})에서 연결 대기.")

    # 5. accept : 연결 수락
    conn, addr = s.accept()
    buffer = ""
    
    with conn:
      print(f"서버 : {addr} 주소의 클라이언트와 연결됨")
      try:
        # 데이터수신 루프
        while True:
          # 6. recv : 클라이언트로부터 데이터 수신
          data_bytes = conn.recv(4096)
          if not data_bytes:
            print("서버 : 수신된 데이터 없음. 연결 종료")
            break
        
          # 7. 수신 데이터 디코딩
          # 여러 JSON 객체가 붇어서 올 수 있으므로 버퍼 사용
          buffer += data_bytes.decode('utf-8')
        
          # 3. JSON 객체를 찾아 처리
          try:
            # 가장 바깥쪽 '{'과 '}'을 기준으로 객체 분리
            start_index = buffer.find('{')
            end_index = buffer.rfind('}')
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
              json_string = buffer[start_index : end_index + 1]
              
              # 4. 문자열을 python json으로 변환 (디코딩)
              received_data = json.loads(json_string)
              
              # (출력작업)
              print("\n[수신 완료]")
              # 테스트코드 : 단순 출력 작업
              print(json.dumps(received_data, indent=4, ensure_ascii=False))

              # 5. 처리된 부분은 버퍼에서 제거 작업
              buffer = buffer[end_index + 1:]
          except json.JSONDecodeError:
            # 아직 완전한 JSON 객체가 도착하지 않았을 수 있음
            # 다음 recv()를 위해 대기
            print("JSONDecodeError")
            print(data_bytes)
            continue
          """
          # 서버와 DB에서 이미지 데이터 자체를 처리하지 않고, 데이터 주소만 받아 처리함
          """
          # # 6. 이미지 출력
          # base64_img_str = json_received["orig_img"]
          # img_bytes = base64.b64decode(base64_img_str) # 6-2. Base64 문자열을 byte 데이터로 디코딩
          # np_arr = np.frombuffer(img_bytes, np.uint8) # 6-3. byte 데이터를 numpy 배열로 변환
          # img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # 6-4. numpy 배열을 OpenCV 이미지로 디코딩
          # if img_decoded is not None: # 6-5. 화면에 이미지 표시
          #   cv2.imshow("Received Image", img_decoded)

          # # 6-6. 키 입력 대기 (GUI 창 업데이트 및 유지를 위해 필수)
          # # 1ms 동안 키 입력을 기다리며, ESC(ASCII 27)를 누르면 루프 종료
          # if cv2.waitKey(1) & 0xFF == 27:
          #   break

      except ConnectionRefusedError:
          print(f"서버({HOST}:{PORT}) 연결 불가")
      except KeyboardInterrupt:
          print("\n사용자입력 : 프로그램 종료")
      finally:
          print("프로그램을 종료합니다.")

if __name__ == '__main__':
  main()