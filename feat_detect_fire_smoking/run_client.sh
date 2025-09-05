#!/bin/bash

# 프로젝트 절대 경로 지정
# e.g.
# PROJECT_DIR="/home/park/dev_ws/deeplearning-repo-2/"
PROJECT_DIR="/home/park/dev_ws/deeplearning-repo-2"
cd $PROJECT_DIR

# 가상환경 기본경로 : 프로젝트 경로/.venv (다를 경우 수정)

# 가상환경 활성화
source $PROJECT_DIR/.venv/bin/activate

FIRE_DETECT_CLIENT_SCRIPT="$PROJECT_DIR/feat_detect_fire_smoking/fire-detect-client.py"

# 파이썬 프로그램 실행 (client)
python $FIRE_DETECT_CLIENT_SCRIPT