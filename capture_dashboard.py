"""Playwright로 Streamlit 대시보드 스크린샷 캡처
출력: output/ppt/ 폴더에 PNG 저장
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

OUT = Path("output/ppt")
OUT.mkdir(exist_ok=True)
BASE = "http://localhost:8501"

def shot(page, name, scroll_y=0, full=False):
    page.evaluate(f"window.scrollTo(0, {scroll_y})")
    time.sleep(1.2)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print(f"  saved output/ppt/{name}.png")

def click_tab(page, partial):
    tabs = page.locator('button[role="tab"]')
    for i in range(tabs.count()):
        if partial in tabs.nth(i).inner_text():
            tabs.nth(i).click()
            page.wait_for_timeout(3500)
            return True
    print(f"  [WARN] tab '{partial}' not found")
    return False

def do_login(page, username, password):
    try:
        pw = page.locator('input[type="password"]')
        if pw.count() == 0:
            return False
        txt = page.locator('input[type="text"]')
        if txt.count() > 0:
            txt.first.fill(username)
        pw.first.fill(password)
        try:
            page.locator('button:has-text("로그인")').click(timeout=3000)
        except Exception:
            page.keyboard.press("Enter")
        page.wait_for_timeout(5000)
        page.wait_for_load_state("networkidle", timeout=15000)
        return True
    except Exception as e:
        print(f"  login error: {e}")
        return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── 관리자 뷰 ─────────────────────────────────────────────────────
        print("=== 관리자 뷰 캡처 ===")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        pg_admin = ctx.new_page()
        pg_admin.goto(BASE, timeout=25000)
        pg_admin.wait_for_load_state("networkidle", timeout=25000)
        pg_admin.wait_for_timeout(5000)

        if do_login(pg_admin, "admin", "admin123"):
            print("  admin 로그인 성공")
        pg_admin.wait_for_timeout(2000)

        # 탭 목록 확인
        tabs = pg_admin.locator('button[role="tab"]')
        n = tabs.count()
        if n > 0:
            print(f"  tab count: {n}")

            print("[1] 관리자 - 리스크 현황")
            if click_tab(pg_admin, "리스크 현황"):
                shot(pg_admin, "10_admin_risk_top")
                shot(pg_admin, "10b_admin_risk_charts", scroll_y=700)

            print("[2] 관리자 - 모델 모니터링")
            if click_tab(pg_admin, "모델 모니터링"):
                shot(pg_admin, "13_admin_monitoring_top")
                shot(pg_admin, "13b_admin_monitoring_full", full=True)

            print("[3] 관리자 - 예측 오차")
            if click_tab(pg_admin, "예측 오차"):
                shot(pg_admin, "14_admin_error_top")
                shot(pg_admin, "14b_admin_error_full", full=True)

            print("[4] 관리자 - 파이프라인")
            if click_tab(pg_admin, "파이프라인"):
                shot(pg_admin, "15_admin_pipeline")
        else:
            print("  [INFO] 탭 없음 - admin 로그인 실패 또는 사용자 뷰")
            shot(pg_admin, "10_admin_top")
            shot(pg_admin, "10b_admin_full", full=True)

        ctx.close()

        # ── 일반 사용자 뷰 ────────────────────────────────────────────────
        print("\n=== 사용자 뷰 캡처 ===")
        ctx2 = browser.new_context(viewport={"width": 1440, "height": 900})
        pg_user = ctx2.new_page()
        pg_user.goto(BASE, timeout=25000)
        pg_user.wait_for_load_state("networkidle", timeout=25000)
        pg_user.wait_for_timeout(5000)

        if do_login(pg_user, "user1", "user1234"):
            print("  user1 로그인 성공")
        pg_user.wait_for_timeout(2000)

        tabs2 = pg_user.locator('button[role="tab"]')
        if tabs2.count() > 0:
            print("  [INFO] user1도 탭 있음 - 관리자 계정과 동일 뷰")
        else:
            print("[5] 사용자 뷰 - 상단")
            shot(pg_user, "20_user_top")
            print("[6] 사용자 뷰 - 차트")
            shot(pg_user, "21_user_charts", scroll_y=700)
            print("[7] 사용자 뷰 - 전체")
            shot(pg_user, "22_user_fullpage", full=True)

        ctx2.close()
        browser.close()

    print("\n완료: output/ppt/ 확인")
    import os
    files = sorted(Path("output/ppt").glob("*.png"))
    print(f"총 {len(files)}개 PNG:")
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb}KB)")

if __name__ == "__main__":
    main()
