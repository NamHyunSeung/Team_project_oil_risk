"""
국제 유가 리스크 예측 시스템 — GUI 런처
실행: python launcher.py  또는 더블클릭
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import webbrowser
import sys
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
DASHBOARD_URL = "http://localhost:8501"

# ── 다크 테마 색상 ─────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TXT     = "#e6edf3"
TXT_SUB = "#8b949e"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
BLUE    = "#58a6ff"


class OilRiskLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("국제 유가 리스크 예측 시스템")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._center(420, 320)

        self._proc      = None   # streamlit 프로세스
        self._running   = False
        self._log_lines = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 구성 ───────────────────────────────────────────────
    def _build_ui(self):
        # 헤더
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(20, 6))
        tk.Label(hdr, text="🛢", font=("Segoe UI Emoji", 28),
                 bg=BG, fg=TXT).pack(side="left")
        title_f = tk.Frame(hdr, bg=BG)
        title_f.pack(side="left", padx=10)
        tk.Label(title_f, text="국제 유가 리스크 예측 시스템",
                 font=("맑은 고딕", 13, "bold"), bg=BG, fg=TXT).pack(anchor="w")
        tk.Label(title_f, text="XGBoost-HAR + SARIMAX Ensemble",
                 font=("맑은 고딕", 9), bg=BG, fg=TXT_SUB).pack(anchor="w")

        # 구분선
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)

        # 상태 표시
        status_f = tk.Frame(self, bg=PANEL, relief="flat")
        status_f.pack(fill="x", padx=20, pady=4)
        status_f.pack_propagate(False)
        status_f.configure(height=38)

        tk.Label(status_f, text="상태", font=("맑은 고딕", 9),
                 bg=PANEL, fg=TXT_SUB).pack(side="left", padx=12, pady=8)
        self._status_dot = tk.Label(status_f, text="●", font=("Arial", 11),
                                    bg=PANEL, fg=YELLOW)
        self._status_dot.pack(side="left")
        self._status_lbl = tk.Label(status_f, text="대기 중",
                                    font=("맑은 고딕", 9, "bold"),
                                    bg=PANEL, fg=TXT)
        self._status_lbl.pack(side="left", padx=6)

        # 로그 창
        log_f = tk.Frame(self, bg=PANEL, relief="flat")
        log_f.pack(fill="both", expand=True, padx=20, pady=6)

        self._log = tk.Text(log_f, height=7, bg=PANEL, fg=TXT_SUB,
                            font=("Consolas", 8), relief="flat",
                            state="disabled", wrap="word",
                            insertbackground=TXT, bd=0)
        self._log.pack(fill="both", expand=True, padx=8, pady=6)
        self._log_append("대시보드 시작 버튼을 눌러주세요.")

        # 버튼
        btn_f = tk.Frame(self, bg=BG)
        btn_f.pack(fill="x", padx=20, pady=(4, 20))

        self._btn_start = tk.Button(
            btn_f, text="▶  대시보드 시작",
            font=("맑은 고딕", 10, "bold"),
            bg=GREEN, fg="#000", activebackground="#2ea043",
            relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
            command=self._start
        )
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._btn_browser = tk.Button(
            btn_f, text="🌐 브라우저",
            font=("맑은 고딕", 10),
            bg=PANEL, fg=BLUE, activebackground=BORDER,
            relief="flat", bd=0, padx=12, pady=8, cursor="hand2",
            state="disabled", command=self._open_browser
        )
        self._btn_browser.pack(side="left", padx=(0, 6))

        self._btn_stop = tk.Button(
            btn_f, text="■  종료",
            font=("맑은 고딕", 10),
            bg=PANEL, fg=RED, activebackground=BORDER,
            relief="flat", bd=0, padx=12, pady=8, cursor="hand2",
            state="disabled", command=self._stop
        )
        self._btn_stop.pack(side="left")

    # ── 유틸 ─────────────────────────────────────────────────
    def _center(self, w, h):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _log_append(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")
        self._log_lines.append(msg)

    def _set_status(self, text, color):
        self._status_dot.configure(fg=color)
        self._status_lbl.configure(text=text)

    # ── 시작 ─────────────────────────────────────────────────
    def _start(self):
        if self._running:
            return
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._set_status("시작 중...", YELLOW)
        self._log_append("\n[시작] Streamlit 대시보드 실행 중...")
        threading.Thread(target=self._run_streamlit, daemon=True).start()

    def _run_streamlit(self):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run",
                 str(BASE_DIR / "dashboard.py"),
                 "--server.headless", "true",
                 "--server.port", "8501"],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env,
            )
            self._running = True

            # 서버 준비 대기 후 브라우저 열기
            threading.Thread(target=self._wait_and_open, daemon=True).start()

            # 로그 스트리밍
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    self.after(0, self._log_append, line)

            self._proc.wait()

        except FileNotFoundError:
            self.after(0, self._log_append, "[오류] streamlit을 찾을 수 없습니다.")
            self.after(0, self._log_append, "  pip install streamlit 을 실행하세요.")
        except Exception as e:
            self.after(0, self._log_append, f"[오류] {e}")
        finally:
            self._running = False
            self.after(0, self._on_stopped)

    def _wait_and_open(self):
        time.sleep(3)
        if self._running:
            self.after(0, self._set_status, "실행 중", GREEN)
            self.after(0, self._btn_browser.configure, {"state": "normal"})
            self.after(0, self._log_append, "[준비] 브라우저를 엽니다 → " + DASHBOARD_URL)
            webbrowser.open(DASHBOARD_URL)

    def _on_stopped(self):
        self._set_status("중지됨", RED)
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._btn_browser.configure(state="disabled")
        self._log_append("[종료] 대시보드가 종료됐습니다.")

    # ── 브라우저 / 종료 ───────────────────────────────────────
    def _open_browser(self):
        webbrowser.open(DASHBOARD_URL)

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_append("[종료 중] 프로세스 종료 요청...")

    def _on_close(self):
        if self._running and messagebox.askyesno(
            "종료 확인", "대시보드를 종료하고 창을 닫을까요?",
            default="yes"
        ):
            self._stop()
            self.after(800, self.destroy)
        elif not self._running:
            self.destroy()


if __name__ == "__main__":
    app = OilRiskLauncher()
    app.mainloop()
