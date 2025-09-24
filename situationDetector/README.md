# situationDetector 설계


## 순찰 이벤트 감지 동작 시나리오

### 1. 통신 관련 기능 정의

| 인덱스 | 기능 이름 | 상호작용 | 통신방식 | 컴포넌트 명칭 |
|---|---|---|---|---|
| 1 | 실시간 영상 전송 | (dM) → (sD) | 수신 (TCP/UDP) | deviceManager, situationDetector |
| 2 | 순찰 이벤트 판단 | 자체기능 | - | situationDetector |
| 3 | 순찰 이벤트 영상요청/정보 발신 | (sD) → (dM) | 발신 (TCP) | situationDetector, deviceManager |
| 4 | 순찰 이벤트 영상 / 차량 정보 수신 | (dM) → (sD) | 수신 (TCP/UDP) | deviceManager, situationDetector |
| 5 | 순찰 이벤트 영상/정보 발신 | (sD) → (dS) | 발신 (TCP/UDP) | situationDetector, dataService |
| 6 | 순찰 이벤트 정보 전송 | (sD) → (mGUI) | 발신 (TCP) | situationDetector, monitoringGUI |
| 7 | 실시간 영상 전송 | (sD) → (mGUI) | 발신 (UDP) | situationDetector, monitoringGUI |

---

### 2. `situationDetector` 디렉터리 구조

- `[x]` 구현 완료 / `[ ]` 구현 미완료

```
situationDetector
├── receiver
│   ├──  tcp_dm_receiver.py # TCP 영상 메타데이터 / 차량 정보 수신 소켓 파일 [ ]
│   └──  udp_dm_receiver.py # UDP 영상 픽셀데이터 / 순찰 이벤트 영상 수신 소켓 파일 [ ]
├── sender
│   ├──  tcp_dm_sender.py # TCP 순찰 이벤트 영상요청/정보 발신 [ ]
│   ├──  udp_ds_sender.py # UDP 순찰 이벤트 영상요청/정보 발신 [ ]
│   ├──  tcp_ds_sender.py # TCP 분석결과 데이터 발신 파일 [ ]
│   ├──  tcp_mgui_sender.py # TCP 순찰 이벤트 정보 발신 파일 [ ]
│   └──  udp_mgui_sender.py # UDP 실시간 영상 전송 파일 [ ]
├── detect
│   ├── detect_main.py # detect기능 main파일
│   └── feat
│       ├──  feat_detect_fall.py [ ]
│       ├──  feat_detect_fire.py [ ]
│       ├──  feat_detect_smoke.py [ ]
│       ├──  feat_detect_trash.py [ ]
│       ├──  feat_detect_violance.py [ ]
│       └──  feat_find_missing.py [ ]
└── sd_main.py # situationDetector 메인파일
```

---

### 3. 기능 상세 설명

1.  **실시간 영상 전송 (dM)**: 순찰차에서 보낸 영상 메타데이터를 1초마다 TCP로, 영상 픽셀데이터를 UDP로 수신
2.  **순찰 이벤트 판단**: `situationDetector`의 자체 기능으로, 수신한 영상을 분석 모델을 이용해 분석
3.  **순찰 이벤트 영상요청/정보 발신 (dM)**: 이벤트 감지 시 `deviceManager`에 30초 분량의 이벤트 영상 데이터를 요청 (TCP)
4.  **순찰 이벤트 영상 / 차량 정보 수신 (dM)**: `deviceManager`로부터 순찰 이벤트 영상(UDP)과 순찰 차량 정보(TCP)를 수신
5.  **순찰 이벤트 영상/정보 발신 (dS)**: 원본 영상(UDP)과 분석 결과를 `dataService`로 전송 (TCP)
6.  **순찰 이벤트 정보 전송 (mGUI)**: 이벤트 발생 시, 해당 이벤트 정보를 `monitoringGUI`에 보냄 (TCP)
7.  **실시간 영상 전송 (mGUI)**: 순찰차에서 받은 영상을 `monitoringGUI`에 그대로 스트리밍 (UDP)