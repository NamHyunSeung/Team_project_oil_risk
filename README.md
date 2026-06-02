# 국제 유가 리스크 예측 시스템

WTI 원유 가격의 D+1~D+7 예측 및 리스크 등급 자동 분류 파이프라인.

---

## 빠른 시작

```bash
pip install -r requirements.txt
python oil_risk_mvp.py
streamlit run dashboard_v2.py
```

---

## 현재 모델 구성

| 역할 | 모델 | 성능 |
|------|------|------|
| 가격 예측 (D+1~D+7) | SARIMAX(0,1,0) | MASE=0.602 |
| 변동성 예측 (vol_5d) | XGBoost-HAR | hold-out R²=0.4681 |
| 방향성 예측 | XGBoost-Classifier | dir_acc=61.1%, wf=53.6% |
| 비교용 | VAR(9) | MASE=0.9998 |
| 비교용 | ETS(HW-Damped) | MASE=1.0 |

---

## 설치

### 요구사항

- Python 3.9+, Windows 10/11

### 환경변수

프로젝트 루트에 `.env` 생성:

```env
EIA_API_KEY=your_key
FRED_API_KEY=0a1d6c8b56c44eff8716c204f0aa49bf
GUARDIAN_API_KEY=3a287cda-6e49-49f0-8998-3092657e209e
NEWSAPI_KEY=your_key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=16자리_앱비밀번호
ALERT_TO=recipient@email.com
```

### 수동 데이터

`data/data_gpr_daily_recent.xls` — https://www.policyuncertainty.com/gpr.html 에서 수동 다운로드

---

## 실행

```bash
# 메인 파이프라인
python oil_risk_mvp.py

# 대시보드
streamlit run dashboard_v2.py

# Windows 배치
run_pipeline.bat
run_rss_alerts.bat

# 작업 스케줄러 등록
python scripts/setup_scheduler.py
```

---

## 출력 파일 (`output/`)

| 파일 | 내용 |
|------|------|
| `forecast_7days.csv` | D+1~D+7 예측가, 75% CI, VaR, bias 교정값 |
| `latest_risk_signal.csv` | 리스크 등급(SURGE/DROP/CAUTION/NORMAL), OVX, 방향 편향 |
| `prediction_log.csv` | backtest+live 예측 오차 누적 (방향성 정확도 포함) |
| `forecast_snapshots.csv` | run_date별 D+1~D+7 예측 이력 (신뢰도 산출 기반) |
| `model_performance.csv` | RMSE/MAE/MASE/R²/dir_acc 모델별 성능 지표 |
| `oil_forecast_plot.png` | 예측 차트 (6-panel) |
| `wordcloud.png` | 뉴스 키워드 워드클라우드 |
| `pipeline_run.log` | 실행 로그 (5MB 자동 로테이션) |

---

## 리스크 등급

```
SURGE_RISK  — risk_score 높음 + 상승 편향
DROP_RISK   — risk_score 높음 + 하락 편향
CAUTION     — risk_score 중간
NORMAL      — risk_score 낮음
```

SURGE_RISK / DROP_RISK 발생 시 이메일 자동 알림.

---

## 자동 재훈련

- **REFIT_STALE_DAYS=7**: 7일마다 Optuna 캐시 초기화 → 하이퍼파라미터 재탐색
- **MASE > 임계값 3일 연속**: 모든 모델 캐시 초기화 → 강제 재훈련
  - 임계값: OVX≥60 → 2.0, OVX≥40 → 1.7, 기본 → 1.5

---

## 파일 구조

```
Team_project_oil_risk/
├── oil_risk_mvp.py          # 메인 파이프라인 (~7100줄)
├── dashboard_v2.py          # Streamlit 대시보드
├── requirements.txt
├── run_pipeline.bat / run_rss_alerts.bat / run_dashboard.bat
├── OilRisk.spec             # PyInstaller EXE 스펙
├── .env                     # API 키 (git 제외)
├── config/                  # 작업 스케줄러 XML, 인증 설정
├── data/                    # CFTC annual.txt, GPR xls
├── docs/                    # 개발 히스토리, 논문 링크
├── experiments/             # 실험 스크립트
├── scripts/                 # 스케줄러 등록 유틸
└── output/                  # 자동 생성
```

---

## 상세 문서

- [개발 히스토리 (초기~현재)](docs/DEVELOPMENT_HISTORY.md) — 전체 실험 기록, 채택/롤백 결정 이유
