@echo off
cd /d "%~dp0"
echo 대시보드 시작 중...
python -m streamlit run dashboard_v2.py
pause
