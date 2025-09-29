#!/bin/bash

# --- 설정: 실행할 파이썬 스크립트의 절대 경로를 지정합니다. ---
SITUATION_DETECTOR_SERVER="/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/dataService/dS_sD_Server_3.py"
DATA_SERVICE_SERVER="/home/choi/dev_ws/project_deepL_ws/deeplearning-repo-2/dataService/server/data_service_tcp_server.py"

# --- 설정: 로그 및 PID 파일을 저장할 폴더 ---
LOG_DIR="./logs"
PID_DIR="./pids"

# --- 각 서버의 PID 파일 경로 ---
PID_FILE_SITUATION="$PID_DIR/situation_detector.pid"
PID_FILE_DATASERVICE="$PID_DIR/data_service.pid"


# --- 함수: 서버 시작 ---
start() {
    # 로그와 PID 폴더가 없으면 생성
    mkdir -p $LOG_DIR
    mkdir -p $PID_DIR

    echo "Starting SituationDetector Server..."
    # 1. SituationDetector 서버를 백그라운드(&)로 실행
    # 2. 모든 출력(>)과 에러(2>&1)를 로그 파일로 저장
    python3 $SITUATION_DETECTOR_SERVER > "$LOG_DIR/situation_detector.log" 2>&1 &
    # 3. 방금 실행한 프로세스의 ID(PID)를 파일에 저장
    echo $! > $PID_FILE_SITUATION

    sleep 2 # 다음 서버 실행 전 잠시 대기

    echo "Starting DataService Server..."
    python3 $DATA_SERVICE_SERVER > "$LOG_DIR/data_service.log" 2>&1 &
    echo $! > $PID_FILE_DATASERVICE

    echo "All servers started."
}

# --- 함수: 서버 종료 ---
stop() {
    echo "Stopping servers..."

    # PID 파일이 존재하면, 해당 PID를 읽어서 프로세스 종료
    if [ -f $PID_FILE_SITUATION ]; then
        kill $(cat $PID_FILE_SITUATION)
        rm $PID_FILE_SITUATION
        echo "SituationDetector Server stopped."
    else
        echo "SituationDetector Server is not running."
    fi

    if [ -f $PID_FILE_DATASERVICE ]; then
        kill $(cat $PID_FILE_DATASERVICE)
        rm $PID_FILE_DATASERVICE
        echo "DataService Server stopped."
    else
        echo "DataService Server is not running."
    fi
}

# --- 메인 로직: 스크립트 실행 시 입력된 명령어(start, stop 등)를 확인 ---
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

exit 0