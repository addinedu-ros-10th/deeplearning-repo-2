import socket
import json

HOST = 'localhost' # 로컬호스트
PORT = 6600 # 6600번 포트 사용

def main():
  # 1. 소켓 생성
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
      # 2. 소켓 옵션 설정
      s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

      # 3. bind : (주소, 포트)에 소켓 할당
      s.bind((HOST, PORT))
      print(f"[*] 서버({HOST}:{PORT})에 연결되었습니다.")

      # 4. listen : 클라이언트의 연결을 기다림 (대기)
      s.listen()
      print("서버가 클라이언트 연결을 대기하는 중")

      # 5. accept : 연결 수락
      conn, addr = s.accept()
      buffer = ""
      
      with conn:
        print(f"{addr} 주소의 클라이언트와 연결됨")
        # 데이터수신 루프
        while True:
          # 6. recv : 클라이언트로부터 데이터 수신
          data_bytes = conn.recv(4096)
          if not data_bytes:
            print("수신된 데이터 없음. 연결 종료")
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
              # 예쁘게 출력하기 위해 다시 json.dumps 사용
              print(json.dumps(received_data, indent=4, ensure_ascii=False))

              
              # 5. 처리된 부분은 버퍼에서 제거 작업
              buffer = buffer[end_index + 1:]
          except json.JSONDecodeError:
            # 아직 완전한 JSON 객체가 도착하지 않았을 수 있음
            # 다음 recv()를 위해 대기
            print("JSONDecodeError")
            print(data_bytes)
            continue
    
    except ConnectionRefusedError:
        print(f"서버({HOST}:{PORT}) 연결 불가")
    except KeyboardInterrupt:
        print("\n사용자입력 : 프로그램 종료")
    finally:
        print("프로그램을 종료합니다.")

if __name__ == '__main__':
  main()