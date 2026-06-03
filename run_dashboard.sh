#!/bin/bash
# 대시보드 서버 실행 (포트 8501)
cd "$(dirname "$0")"
source "$(dirname "$0")/venv/bin/activate"
streamlit run dashboard_v2.py --server.port 8501 --server.address 0.0.0.0
