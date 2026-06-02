# 국제 유가 리스크 예측 시스템

WTI 원유 가격의 단기 예측(D+1~D+7)과 리스크 분류를 수행하는 자동화 파이프라인입니다.  
매일 자동 실행되어 가격 예측, 변동성 예측, 방향성 예측, 리스크 등급을 산출하고 이메일 알림을 전송합니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [모델 구성](#모델-구성)
4. [데이터 소스](#데이터-소스)
5. [피처 엔지니어링](#피처-엔지니어링)
6. [성능 지표](#성능-지표)
7. [설치 및 실행](#설치-및-실행)
8. [출력 파일](#출력-파일)
9. [리스크 분류 체계](#리스크-분류-체계)
10. [개발 히스토리](#개발-히스토리)

---

## 프로젝트 개요

### 목적

- WTI 원유 가격 D+1~D+7 예측 (SARIMAX 기반)
- 5일 실현변동성(vol_5d) 예측 (XGBoost-HAR)
- 가격 방향성 예측 (XGBoost-Classifier, dir_acc=61.1%)
- 리스크 등급 분류: SURGE_RISK / DROP_RISK / CAUTION / NORMAL
- 이상 징후 감지 시 이메일 자동 알림

### 핵심 지표 (2026-06-02 기준)

| 항목 | 값 |
|------|-----|
| WTI 현재가 | $90.26 |
| 리스크 등급 | CAUTION |
| OVX | 61.2 (HIGH) |
| SARIMAX MASE | 0.602 (Persistence 대비 40% 우수) |
| 예측 신뢰도 | UNKNOWN (스냅샷 누적 중, 6월 중순 해소 예정) |

---

## 시스템 아키텍처

```
[데이터 수집]
  yfinance (WTI/Brent/DXY/VIX/OVX)
  EIA API (원유 재고)
  FRED API (경기 지표)
  뉴스 RSS (로이터, AP, BBC 등 50+ 소스)
  CFTC CoT (투기적 포지션)
  GPR (지정학 리스크 지수)
        ↓
[피처 엔지니어링]  §FEA
  FEATURE_COLS: 105개 (다중공선성 제거 후)
  HAR_FEATURE_COLS: 22개 (다중공선성 제거 후)
        ↓
[모델 훈련]  §MDL
  SARIMAX(0,1,0) ← 가격 예측 채택
  XGBoost-HAR    ← 변동성 예측 채택
  XGBoost-Classifier ← 방향성 예측
  VAR(9) / ETS(HW-Damped) ← 비교용
  Stacking (SARIMAX+XGB+VAR → Ridge) ← 미채택
        ↓
[앙상블 & 예측]  §ENS
  forecast_7days.csv (D+1~D+7)
  Bias 교정 (prediction_log 최근 5일 live 오차 기반)
        ↓
[리스크 분류]  §RSK
  risk_score, directional_bias → 등급 결정
  latest_risk_signal.csv
        ↓
[알림 & 저장]  §ALT, §LOG
  이메일 알림 (SMTP)
  prediction_log.csv (누적 오차 로그)
  forecast_snapshots.csv (예측 이력)
```

---

## 모델 구성

### 가격 예측: SARIMAX(0,1,0)

- **특징**: 랜덤워크 + 드리프트. 외생변수(뉴스 감성, OVX, 재고 충격 등) 포함
- **훈련 window**: 최근 3년 슬라이딩 (SARIMAX_WINDOW_YEARS=3)
- **Optuna**: 외생변수 선택 자동 탐색 (XGBoost-HAR 공유 캐시, 7일마다 리셋)
- **MASE**: 0.602 ← **채택 모델**
- **Bias 교정**: prediction_log.csv live 최근 5일 tail, |bias|>1.0 시 자동 보정

### 변동성 예측: XGBoost-HAR

- **타겟**: vol_5d (5일 실현변동성)
- **HAR 구조**: RV_1d + RV_5d + RV_21d 기반 피처 (22개)
- **Walk-forward**: 시계열 5-fold
- **Hold-out R²**: 0.4681 (Persistence R²=0.6285 대비 절대값 낮지만 잔차 구조 개선)

### 방향성 예측: XGBoost-Classifier

- **타겟**: 익일 가격 방향 (UP/DOWN/FLAT)
- **In-sample dir_acc**: 61.1%
- **Walk-forward dir_acc**: 53.6%
- **월별 추이**: 1월 100% → 2월 84% → 3월 95% → 4월 67% → 5월 56%
  - 4월 이후 고변동 레짐 전환 + 하락 편향 누적으로 저하

### 비교 모델 (미채택)

| 모델 | MASE | 비고 |
|------|------|------|
| VAR(9) WTI+Brent+DXY+VIX+OVX | 0.9998 | Persistence 수준 |
| ETS(HW-Damped) | 1.0 | Persistence 수준 |
| Stacking (SARIMAX+XGB+VAR→Ridge) | 0.7137 | SARIMAX 단독보다 열위 |

### 자동 재훈련 트리거

- **REFIT_STALE_DAYS = 7**: Optuna 캐시 7일마다 초기화 → 하이퍼파라미터 재탐색
- **MASE_RETRAIN_THRESH = 1.5** (OVX≥60 시 2.0): 연속 **MASE_RETRAIN_DAYS = 3**일 초과 시 강제 재훈련
  - 재훈련 시 XGB_OPTUNA_CACHE, SVM_CACHE, SVM_MODEL_CACHE, GARCH_CACHE 초기화

---

## 데이터 소스

| 소스 | 내용 | 기간 |
|------|------|------|
| yfinance CL=F | WTI 원유 일봉 (OHLCV) | 10년 (DATA_YEARS=10) |
| yfinance BZ=F | 브렌트 원유 일봉 | 10년 |
| yfinance DX-Y.NYB | 달러 인덱스 | 10년 |
| yfinance ^VIX | 주식시장 공포지수 | 10년 |
| yfinance OVX | 원유 변동성 지수 | 10년 |
| EIA API | 미국 원유 재고 주간 데이터 | EIA_SHIFT=1 (목요일 반영) |
| FRED API | 경기선행지수(CLI), 산업생산 등 | 10년 |
| 뉴스 RSS | 로이터, AP, BBC, Al Jazeera 등 50+ | 최근 |
| CFTC CoT | 비상업적(투기적) 롱/숏 포지션 | data/annual.txt |
| GPR | 지정학 리스크 지수 | data/data_gpr_daily_recent.xls |

---

## 피처 엔지니어링

총 **105개** FEATURE_COLS, **22개** HAR_FEATURE_COLS (2026-06-02 다중공선성 제거 후)

### 주요 피처 범주

- **가격**: WTI/Brent/DXY/VIX/OVX 레벨, 로그수익률, 스프레드
- **변동성**: RV_1d/5d/21d, GARCH, Garman-Klass (parkinson_vol 제거됨)
- **모멘텀**: SMA/EMA 크로스오버, RSI, MACD
- **수요/공급 충격**: EIA 재고 서프라이즈, 수요/공급 더미
- **계절성**: 월, 요일, OPEC 회의 더미, COVID 더미
- **뉴스 감성**: FinBERT/TextBlob/키워드 기반 news_sentiment, news_count
- **지정학**: GPR 지수, 지정학 더미
- **CoT**: 순 투기적 포지션, 변화량

### 다중공선성 제거 (G안, 2026-06-02)

FEATURE_COLS에서 제거: `parkinson_vol`, `return_neg`, `rv_mom_5_21`, `ewma_vol_10`, `ewma_vol_21`, `ewma_vol_63`, `rv_intraday_5d`, `rv_intraday_21d`, `rv_intraday_vs_close`

HAR_FEATURE_COLS에서 제거: `ewma_vol_10`, `ewma_vol_21`, `ewma_vol_63`, `rv_intraday_5d`, `rv_mom_5_21`

**결과**: HAR R² +27% (0.0719→0.0913), CLS F1 +0.7%

---

## 성능 지표

### 모델 성능 요약 (model_performance.csv)

| 모델 | 타겟 | RMSE | MAE | MASE | 비고 |
|------|------|------|-----|------|------|
| SARIMAX(0,1,0) | price | 3.381 | 1.834 | **0.602** | 채택 |
| XGBoost-HAR | vol_5d | 0.00826 | 0.00461 | 0.659 | 채택 (hold-out R²=0.4681) |
| XGBoost-Classifier | direction | - | - | - | dir_acc=61.1%, wf=53.6% |
| Stacking | price | 3.512 | 2.175 | 0.714 | 미채택 |
| VAR(9) | price | 4.460 | 3.047 | 0.9998 | 미채택 |
| ETS(HW-Damped) | price | 4.459 | 3.047 | 1.0 | Persistence 동급 |
| Persistence | price | 4.459 | 3.047 | 1.0 | 기준선 |

### MASE 해석

- MASE < 1.0: Persistence(어제 = 오늘) 대비 우수
- SARIMAX MASE=0.602: Persistence 대비 **40% 오차 감소**

### 예측 신뢰도 (forecast_reliability)

forecast_snapshots.csv에서 D+2~D+7 actual 채워지면 자동 산출.  
현재 UNKNOWN (2026-06-15 이후 자연 해소 예정).

---

## 설치 및 실행

### 요구사항

- Python 3.9+
- Windows 10/11 (Windows 작업 스케줄러 연동)

### 의존성 설치

```bash
pip install -r requirements.txt
```

주요 패키지:

```
numpy, pandas, scikit-learn          # 핵심
yfinance, xgboost, statsmodels       # 모델
streamlit, plotly                    # 대시보드
arch, pmdarima                       # 변동성/ARIMA
transformers, torch                  # FinBERT NLP
feedparser                           # RSS 뉴스
python-dotenv, apscheduler           # 환경설정/스케줄
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
EIA_API_KEY=your_eia_api_key
FRED_API_KEY=your_fred_api_key
GUARDIAN_API_KEY=your_guardian_api_key
NEWSAPI_KEY=your_newsapi_key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=your_app_password_16chars
ALERT_TO=recipient@email.com
```

> EIA API 키: https://www.eia.gov/opendata/ 에서 무료 발급  
> Gmail 앱 비밀번호: Google 계정 → 보안 → 앱 비밀번호

### GPR 데이터

`data/data_gpr_daily_recent.xls` 파일 필요 (지정학 리스크 지수).  
https://www.policyuncertainty.com/gpr.html 에서 수동 다운로드.

### 실행

#### 수동 실행

```bash
python oil_risk_mvp.py
```

#### 배치 실행 (Windows)

```bat
run_pipeline.bat         # 메인 파이프라인
run_rss_alerts.bat       # RSS 알림만
run_dashboard.bat        # 대시보드 시작
run_pipeline_night.bat   # 야간 실행용
```

#### 대시보드

```bash
streamlit run dashboard_v2.py
```

#### Windows 작업 스케줄러 등록

```bash
python scripts/setup_scheduler.py
```

또는 `config/task_schedule.xml`을 작업 스케줄러에 임포트.

---

## 출력 파일

모든 출력 파일은 `output/` 폴더에 저장됩니다.

### forecast_7days.csv

D+1~D+7 예측값. 매일 덮어씌움.

| 컬럼 | 설명 |
|------|------|
| date | 예측 대상 날짜 |
| forecast_price | 최종 앙상블 예측가 |
| lower_75ci / upper_75ci | 75% 신뢰구간 |
| sarimax_forecast | SARIMAX 원본 예측 |
| xgb_forecast | XGBoost 예측 |
| bias_correction | 적용된 바이어스 교정값 |
| var_5pct / var_95pct | 5%/95% VaR |

### latest_risk_signal.csv

최신 리스크 등급 및 지표. 매일 갱신.

| 컬럼 | 설명 |
|------|------|
| risk_level | SURGE_RISK / DROP_RISK / CAUTION / NORMAL |
| wti_price | 당일 WTI 종가 |
| volatility_5d | 5일 실현변동성 |
| ovx_level | OVX (HIGH≥50, EXTREME≥70) |
| directional_bias | 예측 방향 편향 |
| downside_risk_pct / upside_risk_pct | 하락/상승 리스크 비율 (live 오차 기반) |
| forecast_reliability | UNKNOWN / LOW / MEDIUM / HIGH |

### prediction_log.csv

누적 예측 오차 로그. backtest + live 구분.

| 컬럼 | 설명 |
|------|------|
| type | backtest / live |
| sarimax_pred | SARIMAX 예측가 |
| actual_price | 실제 종가 |
| price_error | actual - pred (양수=과소예측) |
| dir_correct | 방향성 정확도 (0/1) |

### forecast_snapshots.csv

run_date별 D+1~D+7 예측 이력. 예측 신뢰도 산출에 사용.

### model_performance.csv

모델별 RMSE, MAE, MASE, R², dir_acc 등 성능 지표.

### 기타 출력

| 파일 | 설명 |
|------|------|
| oil_forecast_plot.png | 예측 시각화 차트 |
| wordcloud.png | 뉴스 키워드 워드클라우드 |
| crisis_keywords.csv | 감지된 위기 키워드 |
| pipeline_run.log | 실행 로그 (5MB 초과 시 자동 로테이션) |

---

## 리스크 분류 체계

```
risk_score = f(변동성, 모멘텀, OVX, 뉴스 감성, 재고 충격, 지정학)

SURGE_RISK : risk_score > threshold AND directional_bias > 0
DROP_RISK  : risk_score > threshold AND directional_bias < 0
CAUTION    : threshold/2 < risk_score ≤ threshold
NORMAL     : risk_score ≤ threshold/2
```

리스크 등급에 따라 이메일 자동 알림 전송 (SURGE_RISK / DROP_RISK 시).

---

## 파일 구조

```
Team_project_oil_risk/
├── oil_risk_mvp.py          # 메인 파이프라인 (7100+ 줄)
├── dashboard_v2.py          # Streamlit 대시보드
├── requirements.txt         # 패키지 목록
├── run_pipeline.bat         # 파이프라인 실행 배치
├── run_rss_alerts.bat       # RSS 알림 배치
├── run_dashboard.bat        # 대시보드 실행 배치
├── run_pipeline_night.bat   # 야간 실행 배치
├── OilRisk.spec             # PyInstaller 실행파일 스펙
├── .env                     # API 키 (git 제외)
├── config/
│   ├── task_schedule.xml    # Windows 작업 스케줄러 설정
│   ├── task_schedule_rss.xml
│   └── auth_config.yaml
├── data/
│   ├── annual.txt           # CFTC 연간 데이터
│   └── data_gpr_daily_recent.xls  # GPR 지정학 리스크 (수동 다운로드)
├── docs/
│   ├── 오일 리스크 계획 및 실행 전략 완성본.pdf
│   └── 논문 링크.txt
├── experiments/
│   ├── sarimax_experiment.py    # SARIMAX 파라미터 실험
│   └── lstm_direction.py        # LSTM 방향성 실험
├── scripts/
│   └── setup_scheduler.py       # 작업 스케줄러 등록
└── output/                      # 자동 생성
    ├── forecast_7days.csv
    ├── latest_risk_signal.csv
    ├── prediction_log.csv
    ├── forecast_snapshots.csv
    ├── model_performance.csv
    ├── oil_forecast_plot.png
    ├── wordcloud.png
    └── pipeline_run.log
```

---

## 개발 히스토리

### 초기 설계 (2026-01~02)

- WTI D+1 예측을 목표로 시작
- yfinance + FRED + 뉴스 RSS 데이터 파이프라인 구축
- XGBoost-HAR (변동성), SARIMAX (가격), XGBoost-Classifier (방향성) 기본 구조 확립
- Windows 작업 스케줄러 연동으로 매일 자동 실행

### 모델 개선 (2026-03~04)

- VAR(9), ETS(HW-Damped), Stacking 앙상블 추가 실험
- SARIMAX MASE=0.602으로 모든 대안 모델 압도 → 단독 채택
- Stacking MASE=0.7137 (SARIMAX보다 열위) → 미채택
- OVX, GPR, CFTC CoT 피처 추가
- forecast_snapshots.csv 누적으로 예측 신뢰도 체계 구축

### 성능 안정화 (2026-05)

- **EIA 타임존 수정**: `pd.Timestamp.today()` → `pd.Timestamp.now('Asia/Seoul')` (UTC 환경 KST 목요일 판단 오류 수정)
- **Bias 교정 소스 개선**: live_gap_spikes.csv(스파이크 상위5) → prediction_log.csv live 최근 5일 tail (더 안정적)
- **MASE_RETRAIN_THRESH 동적화**: OVX 수준에 따라 1.5/1.7/2.0으로 분기

### 다중공선성 제거 (2026-06-02)

- 전체 피처 VIF/rank 분석
- FEATURE_COLS: 114→105개 (9개 제거)
- HAR_FEATURE_COLS: 27→22개 (5개 제거)
- 결과: HAR R² +27% (0.0719→0.0913), CLS F1 +0.7%

### 모니터링 체계

- prediction_log.csv 방향성 월별 분석: 4-5월 고변동 레짐 전환 확인
- 연속 MASE 열화 감지 → 자동 재훈련 트리거
- 7일마다 Optuna 캐시 초기화 → 하이퍼파라미터 재탐색

---

## 알려진 이슈 및 제한사항

### forecast_reliability = UNKNOWN

forecast_snapshots.csv에서 D+2~D+7 actual 데이터가 충분히 쌓이지 않으면 UNKNOWN.  
2026-06-15 이후 자연 해소 예정.

### downside/upside_risk_pct = 50/50

prediction_log.csv live 오차가 전부 양수(과소예측)인 경우,  
`_neg` 배열이 비어 `len(_neg) > 0` 조건 미충족 → 기본값 50.0 반환.  
의도된 폴백 동작이며 실제 의사결정에 직접 사용되지 않음.

### 방향성 워크포워드 저하 (53.6%)

4월 이후 고변동 레짐 전환으로 학습 패턴 불일치.  
REFIT_STALE_DAYS=7로 지속 업데이트 중.
