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

## 리스크 스코어 계산 방식

```
mom_ratio  = min(|mom_5d| / hist_mom_75, 3.5)       # 트렌드형 이동 포착
vol_ratio  = max(vol_5d / hist_vol_75, mom_ratio)    # 변동성·모멘텀 중 큰 값
risk_score = vol_ratio × news_amp × geo_amp × sentiment_amp × ovx_amp
```

- `hist_vol_75` / `hist_mom_75`: 훈련 구간(최근 60일 제외) 기준 75분위
- `cap=3.5`: 기존 SURGE_RISK 구간에서 mom_ratio 과도 증폭 방지
- 트렌드형 이동(매일 +2~3% 누적)과 스파이크형 이동 모두 포착

### mom_ratio 도입 배경 (방식F 채택)

기존 수식은 `vol_5d / hist_vol_75`만 사용해 스파이크형 급등락에는 반응했지만,
매일 +2~3%씩 누적되는 **트렌드형 이동**은 변동성이 낮아 `NORMAL`로 잘못 분류되는 문제가 있었다.

실험한 방식 비교:

| 방식 | 수식 | 문제 |
|------|------|------|
| A | `vol_ratio + 0.5×mom_ratio` | 가산 방식 → 정상 구간에서 오탐 발생 |
| B | `max(vol_ratio, mom_ratio)` cap 없음 | 기존 SURGE 구간에서 과도 증폭 |
| C | B + 별도 mom_cap 적용 | 임계값 튜닝 복잡, 코드 과적합 위험 |
| **F** | **`max(vol_ratio, min(mom_ratio, 3.5))`** | 채택 — 오탐 없이 불일치 전부 해결 |

**방식F 검증 결과**:
- 트렌드형 이동 불일치 구간 7개 → 전부 CAUTION+ 개선
- 기존 SURGE/DROP 구간 오탐 0건 (레벨 유지)
- risk_score vs |mom_5d| 상관계수 0.790 (방식 도입 전 0.44)
- 레벨 분포 변화: SURGE_RISK 39→47건, CAUTION 28→34건, NORMAL 유지

---

## 지정학 위험 감지

GPR(Geopolitical Risk) 데이터 지연 문제를 3단계로 보완:

1. **GPR 임계값 완화**: z-score > 0.5 (이전 > 1.0) — 상위 약 30% 수준에서 위기 감지
2. **지정학 키워드 확장**: iran, hormuz, blockade, houthi, naval blockade, oil embargo 등 추가
3. **뉴스 실시간 보완**: `latest_alerts.json`에서 geopolitical 뉴스 감지 시 GPR 데이터와 무관하게 geo_dummy=1 강제 적용

---

## risk_history.csv (리스크 이력)

- 최근 1년(365일) 분 기록 유지
- **공휴일·주말 포함**: 영업일·달력일 기준 전일 값 forward fill → 차트 빈 칸 없음
  - 거래일 데이터 없는 날(미국 공휴일, 토/일)은 직전 거래일의 vol_5d, mom_5d, WTI 값 사용
  - 뉴스·감성·OVX 등 독립 피처는 당일 값 반영

---

## 안정성 개선 및 버그 수정

### forecast_snapshots.csv — actual_price 중복 컬럼 수정

`forecast_snapshots.csv`와 `prediction_log.csv`를 merge할 때 두 파일에 `actual_price` 컬럼이 모두 존재하면
pandas가 `actual_price_x` / `actual_price_y`로 분리해 기존 값이 사라지는 버그 수정.
`_pl_actual`로 rename 후 coalesce 처리 → 실제 가격 데이터 손실 없음.

### _atomic_csv — UTF-8 인코딩 기본값 추가

`_atomic_csv()` 함수에 `encoding='utf-8'` 기본값 추가.
이전에는 일부 환경에서 CP949(Windows 기본)로 저장돼 한글 포함 CSV 읽기 오류 발생 가능.

### forecast_snapshots.csv — d2~d7 actual 역채움 개선

**버그**: `_snap_row[_ca] = _snap_row[_cd].map(_price_str_map)` 방식은 `_price_str_map`에 해당 날짜가 없으면 기존 actual 값을 NaN으로 덮어쓰는 문제 발생.

**수정**:
- `fillna` 방식으로 변경 → 기존 actual 값 보존, 없는 경우에만 채움
- 역채움 소스에 `prediction_log.csv`의 `actual_price`도 추가 → `full_df`에 없는 날짜 보완

### forecast_reliability — 다중 horizon MASE 계산 추가

기존에는 D+1 actual_price만 사용해 MASE를 계산했으나,
d2~d7 각 horizon의 (actual, forecast) 쌍도 포함해 다중 horizon 기반 MASE로 개선.

- D+1 행 수는 기준 유지, d2~d7 실제가 확인된 쌍을 전체 오차 pool에 합산
- naive baseline은 D+1 시계열 기준 (day-to-day diff 평균 절대값) 유지
- 로그에 `n` (D+1 기준)과 `multi-horizon` (전체 합산) 별도 출력

### 감성 충격 보정 — ±3% 캡 추가 (2026-06-04)

**버그**: `_shock_adj = _chg3_sh * 20.0` 계산 시 FinBERT 3일 감성 변화값(-0.665)이 극단적일 경우 -13.31$ 보정이 무제한 적용됨.

**증상**: SARIMAX=$95.31, XGB=$94.14인데 최종 D+1 예측가=$83.15 (현재가 $95.24 대비 -12.7%).

**수정**: `np.clip(_chg3_sh * 20.0, -last_price * 0.03, last_price * 0.03)`으로 ±3% 캡 적용.
최대 조정폭 현재가 기준 ±약 $2.85로 제한 → D+1 정상화 ($93.35).

### live bias + 모멘텀 추가 보정 — 합산 캡 추가 (2026-06-04)

**잠재 문제**: `compute_live_bias_correction()`의 ±5$ 캡 이후 모멘텀 추가 보정(-3$)이 별도로 더해져 합산 최대 -8$를 초과할 가능성.

**수정**: 모멘텀 += 이후, 방향 불일치 체크 이전에 `np.clip(bias, -8.0, 8.0)` 추가.
live(±5$) + momentum(±3$) 설계 의도 상한을 코드에서 명시적으로 보장. 초과 시 로그 출력.

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
