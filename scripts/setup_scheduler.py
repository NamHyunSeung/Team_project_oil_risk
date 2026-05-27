"""
Windows Task Scheduler에 파이프라인 자동 실행 등록/해제
사용법:
  python setup_scheduler.py install   # 매일 오전 7시 등록
  python setup_scheduler.py remove    # 등록 해제
  python setup_scheduler.py status    # 등록 상태 확인
"""
import sys, subprocess
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

TASK_NAME       = "OilRiskPipeline"
TASK_NAME_RSS   = "OilRiskRSSAlert"
BASE_DIR        = Path(__file__).parent
SCRIPT          = BASE_DIR / "oil_risk_mvp.py"
LOG_FILE        = BASE_DIR / "output" / "scheduler.log"
RSS_LOG_FILE    = BASE_DIR / "output" / "rss_alert.log"
RUN_HOUR        = 7
RUN_MINUTE      = 0


def _python():
    return sys.executable


def install(hour=RUN_HOUR, minute=RUN_MINUTE):
    cmd = [
        "schtasks", "/Create", "/F",
        "/TN", TASK_NAME,
        "/TR", f'"{_python()}" "{SCRIPT}" >> "{LOG_FILE}" 2>&1',
        "/SC", "DAILY",
        "/ST", f"{hour:02d}:{minute:02d}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ 스케줄 등록 완료 — 매일 {hour:02d}:{minute:02d} 실행")
    else:
        print(f"❌ 등록 실패: {r.stderr.strip()}")


def remove():
    r = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("✅ 스케줄 해제 완료")
    else:
        print(f"❌ 해제 실패: {r.stderr.strip()}")


def install_rss():
    """매시 정각 RSS 경보 모니터링 등록 (파이프라인과 독립)"""
    rss_script = f'"{sys.executable}" -c "import oil_risk_mvp; oil_risk_mvp.monitor_rss_alerts()"'
    cmd = [
        "schtasks", "/Create", "/F",
        "/TN", TASK_NAME_RSS,
        "/TR", f'{rss_script} >> "{RSS_LOG_FILE}" 2>&1',
        "/SC", "HOURLY",
        "/MO", "1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print("✅ RSS 경보 스케줄 등록 완료 — 매시간 실행")
    else:
        print(f"❌ RSS 등록 실패: {r.stderr.strip()}")


def remove_rss():
    r = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME_RSS, "/F"],
        capture_output=True, text=True
    )
    print("✅ RSS 경보 스케줄 해제" if r.returncode == 0 else f"❌ {r.stderr.strip()}")


def status():
    for tn in [TASK_NAME, TASK_NAME_RSS]:
        r = subprocess.run(
            ["schtasks", "/Query", "/TN", tn, "/FO", "LIST"],
            capture_output=True, text=True, encoding="cp949", errors="replace"
        )
        print(r.stdout if r.returncode == 0 else f"등록 없음 ({tn})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install":
        hour   = int(sys.argv[2]) if len(sys.argv) > 2 else RUN_HOUR
        minute = int(sys.argv[3]) if len(sys.argv) > 3 else RUN_MINUTE
        install(hour, minute)
    elif cmd == "install-rss":
        install_rss()
    elif cmd == "remove":
        remove()
    elif cmd == "remove-rss":
        remove_rss()
    else:
        status()
