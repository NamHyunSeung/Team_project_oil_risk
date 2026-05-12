@echo off
cd /d "%~dp0"
python oil_risk_mvp.py >> output\pipeline_run.log 2>&1
