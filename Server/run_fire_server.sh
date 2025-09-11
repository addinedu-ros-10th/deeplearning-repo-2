#!/bin/bash

# 프로젝트 절대 경로 지정
# PROJECT_DIR="/home/<user_name>/dev_ws/deeplearning-repo-2/"

PROJECT_DIR="/home/park/dev_ws/deeplearning-repo-2"
cd $PROJECT_DIR

# 가상환경 기본경로 : 프로젝트 경로/.venv (다를 경우 수정)

# 가상환경 활성화
# source <venv_dir>/bin/activate

source $PROJECT_DIR/.venv/bin/activate

FIRE_DETECT_SERVER_SCRIPT="$PROJECT_DIR/Server/fire-detect-server.py"

# 파이썬 프로그램 실행 (server)
python $FIRE_DETECT_SERVER_SCRIPT