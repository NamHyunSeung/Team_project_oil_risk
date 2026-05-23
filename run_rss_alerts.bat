@echo off
cd /d "%~dp0"

set LOG=output\rss_alerts.log
set LOG_BAK=output\rss_alerts.log.bak

setlocal enabledelayedexpansion
if exist "%LOG%" (
    for /f %%A in ('powershell -command "(Get-Item \"%LOG%\").length"') do set SIZE=%%A
    if !SIZE! GTR 1048576 (
        if exist "%LOG_BAK%" del "%LOG_BAK%"
        move "%LOG%" "%LOG_BAK%"
    )
)

echo [%date% %time%] RSS 스캔 시작 >> "%LOG%"
python oil_risk_mvp.py --rss-alerts >> "%LOG%" 2>&1
echo [%date% %time%] RSS 스캔 종료 >> "%LOG%"
