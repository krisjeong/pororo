#!/usr/bin/env bash

# 홈 디렉토리에서 실행 (예. ./maru_sample/asr.sh)
AUDIO_PATH="korean_sample3.wav"
LOG_FILE="./logs/asr.log"
python -m maru_sample.asr \
    -audio_path ${AUDIO_PATH} \
	-log_file ${LOG_FILE}
