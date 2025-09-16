# 1. 데이터 서비스 서버를 백그라운드에서 실행
echo "🚀 데이터 서비스를 시작합니다..."
python3 data_service.py &
DATA_SERVICE_PID=$!
sleep 2 # 서버가 포트를 열 때까지 잠시 대기

# 2. 메인 서버(영상/이벤트)를 백그라운드에서 실행
echo "🚀 메인 서버를 시작합니다..."
python3 situation_server.py &
SERVER_PID=$!
sleep 2 # 서버가 포트를 열 때까지 잠시 대기

# 3. GUI 클라이언트를 포그라운드에서 실행 (이 창이 닫히면 스크립트가 계속 진행됨)
echo "🚀 GUI 클라이언트를 시작합니다. (GUI 창을 닫으면 모든 서버가 함께 종료됩니다)"
python3 dashboard_client.py

# 4. GUI 클라이언트가 종료된 후, 백그라운드에서 실행되던 서버들을 모두 종료
echo "🧹 모든 백그라운드 서버 프로세스를 종료합니다..."
kill $SERVER_PID
kill $DATA_SERVICE_PID

echo "✅ 모든 작업이 종료되었습니다."


# 실행 권한 부여
# chmod +x start_all.sh


# 스크립트 실행
# ./start_all.sh