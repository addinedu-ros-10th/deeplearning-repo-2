# 화재 연기 detect기능 및 서버통신 기능

### 1. sh파일 경로를 환경에 맞게 지정
  1. Server/run_fire_server.sh 파일
  2. feat_detect_fire/run_fire_client.sh 파일
  
  - 각각의 sh 파일에서
    - 프로젝트 경로 지정 : PROJECT_DIR
    - 가상환경 기본경로 지정

### 2. 서버 실행 (터미널 1)
```
~/dev_ws/deeplearning-repo-2$ source Server/run_fire_server.sh 
서버 : 소켓을 (localhost:6600)에 할당
서버 : (localhost:6600)에서 연결 대기.
```

### 3. 클라이언트(YOLO 카메라) 실행 (터미널 2)
```
:~/dev_ws/deeplearning-repo-2$ source feat_detect_fire/run_fire_client.sh 
시스템 : 연결 대기 - localhost : 6600
시스템 : 연결 완료 - localhost : 6600
```

### 4. 결과
- 터미널 1
```
[수신 완료]
{
    "timestamp": 1757406959.5229077,
    "boxes": [
        {
            "class_id": 1,
            "class_name": "fire_smoke",
            "detect_count": 1,
            "confidence": 0.1739780753850937,
            "box_xyxy": [
                [
                    258.0130615234375,
                    47.83088302612305,
                    522.800537109375,
                    306.9483337402344
                ]
            ]
        }
    ],
    "video_path": "/patrol_car/video"
}
```


- 터미널 2
```
0: 480x640 1 fire_smoke, 9.1ms
Speed: 1.4ms preprocess, 9.1ms inference, 4.2ms postprocess per image at shape (1, 3, 480, 640)

클라이언트 : 전송 완료
클라이언트 종료
```

### 5. 수동 실행

- 기준경로 : 프로젝트 루트 경로

#### 서버 실행
```
~/dev_ws/deeplearning-repo-2$ python -m Server/fire-detect-server
```

#### 클라이언트 실행
```
:~/dev_ws/deeplearning-repo-2$ python -m feat_detect_fire/fire-detect-client
```