@echo off
cd /d "%~dp0"

:: 로그 로테이션: 5MB 초과 시 backup으로 이동
set LOG=output\pipeline_run.log
set LOG_BAK=output\pipeline_run.log.bak

if exist "%LOG%" (
    for /f %%A in ('powershell -command "(Get-Item \"%LOG%\").length"') do set SIZE=%%A
    if !SIZE! GTR 5242880 (
        if exist "%LOG_BAK%" del "%LOG_BAK%"
        move "%LOG%" "%LOG_BAK%"
    )
)

setlocal enabledelayedexpansion
echo [%date% %time%] 파이프라인 시작 >> "%LOG%"
python oil_risk_mvp.py >> "%LOG%" 2>&1
echo [%date% %time%] 파이프라인 종료 >> "%LOG%"
