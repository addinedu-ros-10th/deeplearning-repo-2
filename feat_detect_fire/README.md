## 불 / 연기 감지 기능 서버,클라이언트 구현

## 0. 사용법

### 1. 쉘 파일(sh) 사용 (자동)
1. run_client.sh / run_server.sh 파일 가상환경 경로 / 프로젝트 절대경로 지정
2. 터미널1 : run_client.sh
```
$ source run_client.sh
```
3. 터미널2 : run_server.sh
```
$ source run_server.sh
```

### 2. 가상환경 설정 및 파이썬 파일 실행 (수동)
1. 가상환경 활성화
2. 터미널1 : 서버 실행
```
(경로 : 프로젝트루트/feat_detect_fire_smoking)
$ python -m fire-detect-server
```
3. 터미널2 : 클라이언트 실행
```
(경로 : 프로젝트루트/feat_detect_fire_smoking)
$ python -m fire-detect-client
```

## 1. 개요

### 1. 서버
  - 기능
    1. 데이터 길이 수신 (이미지 데이터의 온전한 수신을 위해 데이터 길이를 받아옴)
    2. 실제 데이터 수신
    3. 수신한 데이터 디코딩 및 python json으로 변환
      (바이트 -> str -> )
    4. 받아온 영상을 cv2로 출력

  - 특징
    1. (데이터 변환) 빅 엔디안 사용
    2. 소켓 **생성** (연결x)

### 2. 클라이언트
  - 기능
    1. 카메라 생성 (cv2.VideoCapture(0))
    2. 소켓 연결
    3. YOLO모델을 이용해서 cv로 읽어온 이미지에서 객체 정보 추출 (detect)
    4. 전송에 필요한 시간 정보만 추출
    ```
    transform = {
      "timestamp" : time.time(),
      "detections" : result[0].names,
      "orig_img" : base64_image, # 
    }
    ```
    5. str 정보로 변환 및 인코딩 / 전송
  
  - 특징
    1. (데이터 변환) 빅 엔디안 사용
    2. 소켓 **연결** (생성x)