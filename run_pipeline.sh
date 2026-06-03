#!/bin/bash
# 수동 실행용 (run_pipeline.bat 대체)
cd "$(dirname "$0")"

LOG=output/pipeline_run.log
LOG_BAK=output/pipeline_run.log.bak

# 로그 로테이션: 5MB 초과 시 백업
if [ -f "$LOG" ]; then
    SIZE=$(stat -c%s "$LOG" 2>/dev/null || stat -f%z "$LOG" 2>/dev/null)
    if [ "$SIZE" -gt 5242880 ]; then
        [ -f "$LOG_BAK" ] && rm "$LOG_BAK"
        mv "$LOG" "$LOG_BAK"
    fi
fi

source "$(dirname "$0")/venv/bin/activate"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 파이프라인 시작" >> "$LOG"
python oil_risk_mvp.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 파이프라인 종료" >> "$LOG"
