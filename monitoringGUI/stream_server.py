from flask import Flask, Response
import cv2

# Flask 앱 초기화
app = Flask(__name__)

# 0번 카메라(보통 내장 웹캠)와 연결
# 만약 다른 카메라를 사용하려면 숫자를 1, 2 등으로 변경
camera = cv2.VideoCapture(0)

def generate_frames():
    """카메라 프레임을 지속적으로 읽어와 JPEG 형식으로 인코딩하고 스트리밍합니다."""
    while True:
        # 카메라에서 프레임 읽기
        success, frame = camera.read()
        if not success:
            break
        else:
            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # 인코딩된 프레임을 바이트 스트림으로 변환
            frame_bytes = buffer.tobytes()
            
            # HTTP multipart response 형식으로 프레임 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 경로입니다."""
    # generate_frames 함수가 생성하는 프레임들을 응답으로 보냅니다.
    # mimetype='multipart/x-mixed-replace; boundary=frame'은
    # 서버가 클라이언트에게 지속적으로 데이터를 보내는 스트리밍 방식임을 알려줍니다.
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 서버 실행 (IP 주소 0.0.0.0은 모든 네트워크 인터페이스에서 접속 허용)
    # port=5000은 5000번 포트를 사용하겠다는 의미
    app.run(host='0.0.0.0', port=5000)