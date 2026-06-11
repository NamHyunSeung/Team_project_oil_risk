# 국제 유가 리스크 예측 시스템

WTI 원유 가격의 D+1~D+7 예측 및 리스크 등급 자동 분류 파이프라인.

---

## Background

Oil price volatility directly impacts the cost
structure of energy-dependent companies. This system
predicts WTI crude oil prices (D+1~D+7) and
automatically classifies risk levels (SURGE/DROP/
CAUTION/NORMAL) by combining time-series models,
volatility modeling, and news sentiment analysis.

---

## 연구 방법론 하이라이트

이 프로젝트는 단순 구현을 넘어 **가설 검증 → 실험 → 채택/롤백**의 반복 사이클로 진행되었으며,
주요 의사결정은 모두 [개발 히스토리](docs/DEVELOPMENT_HISTORY.md)와 본 README의 변경 로그에 근거와 함께 기록되어 있다.

- **데이터 누수(Data Leakage) 탐지·증명**: SARIMAX 백테스트 오차가 비정상적으로 낮아진 원인을
  추적해 `예측값 = 실제값 + 잡음` 형태로 정답이 누수되고 있음을 상관계수 0.9999999999999999로
  수학적으로 증명하고 수정 (→ [상세](#sarimax-백테스트-데이터-누수look-ahead-수정-2026-06-09))
- **Ablation 기반 의사결정**: 리스크 스코어 계산식을 A/B/C/F 4가지 방식으로 비교 실험,
  정량 지표(오탐 건수, |mom_5d| 상관계수 0.44→0.790)로 최종 방식을 채택
  (→ [상세](#mom_ratio-도입-배경-방식f-채택))
- **모델 간 정량 비교**: SARIMAX/XGBoost-HAR/XGBoost-Classifier/VAR/ETS를 동일 데이터·기간에서
  MASE·R²·방향성 정확도로 비교, 단일 모델이 아닌 역MAE 가중 앙상블로 보완
- **실패 사례의 구조적 분석**: 백테스트 오차 $6+ 케이스를 전수 분류해 "구조적 예측 불가(블랙스완)"와
  "개선 가능"을 구분하고, 후속 실험 우선순위를 도출 (→ [상세](#백테스트-예측-한계-및-개선-계획))
- **지속적 검증·자동 재훈련**: MASE 드리프트가 임계값을 초과하면 자동 재훈련되도록 설계,
  수치 변경 시 README/모델 성능표를 즉시 갱신하는 워크플로 유지

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

SURGE_RISK / DROP_RISK 발생 시 이메일 자동 알림 (구독자 전체 발송).

---

## 구독 플랜

| 플랜 | 월 구독료 | 가격 예측 차트 | 뉴스 키워드 | 워드클라우드 | 예측 성과 탭 | 이메일 알람 | 알람 임계값 |
|------|---------|--------------|-----------|------------|-------------|-----------|-----------|
| 무료 | ₩0 | — | 상위 5개 | — | — | — | — |
| 일반 | ₩490,000 | D+1~7 예측 + 신뢰구간 | 상위 10개 | ✓ | — | ✓ | 고정(0.7) |
| 프로 | ₩1,290,000 | 전체 + 모델별 상세 | 10개 | ✓ | MASE + 방향성 정확도 | ✓ | 커스텀(사이드바) |

`config/auth_config.yaml`의 `plan` 필드로 사용자별 플랜 설정.

### 동시 접속 제한

| 플랜 | 동시 접속 가능 기기 수 |
|------|---------------------|
| 무료 | 1대 |
| 일반 | 1대 |
| 프로 | 무제한 |

세션 정보는 `config/active_sessions.json`에 저장. TTL 8시간 초과 세션은 자동 만료.

---

## 사용자 계정 관리

### 회원가입

대시보드 로그인 화면의 **"📝 회원가입"** 탭에서 직접 가입:

- 아이디 (영문+숫자, 4~20자), 이름, 이메일, 비밀번호(8자 이상)
- 플랜 선택: 무료(즉시) / 일반·프로 → 7일 체험 후 만료
- 비밀번호는 bcrypt 해시로 `config/auth_config.yaml`에 저장

### 구독 만료 자동 처리

만료일이 지나면 로그인 시 자동으로 `free` 플랜으로 전환 (YAML 업데이트).
관리자 계정은 다운그레이드 대상 제외.

### 플랜 업그레이드 요청

1. 사용자: 사이드바 **"⬆ 플랜 업그레이드"** 버튼 클릭 → YAML에 `upgrade_request` 저장
2. 관리자: Tab5 상단의 요청 목록에서 **"승인"** 클릭 → `plan` 업데이트 + 구독 30일 연장

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

### 충격 후 레짐 캡 — D+1 변동폭 ±8% 제한 (2026-06-04)

**문제**: 4/8 -16.4% 급락 이후 극단 피처값(mom_5d, vol_5d)에 모델이 과반응해 4/13 오차 +$9.03 발생.
기존 `compute_error_spike_blend()`는 `|error|>$5 연속 3일` 조건이라 충격 직후 소액 오차 구간에서 미발동.

**수정**: 최근 5영업일 내 단일 일간 변동 ≥8% 감지 시 D+1 예측을 현재가 기준 ±8% 범위로 클리핑.

**효과**: 4/13 오차 +$9.03 → -$2.44 (D+1 $82.25 → $88.84, 실제 $91.28).
threshold=8% 시 전수 시뮬레이션에서 정확한 예측까지 악화시키는 문제 확인 → 14%로 상향 조정.
4/16(5영업일 내 최대 변동 7.87%)은 미발동으로 영향 없음.
`save_prediction_log()` 백테스트 루프에도 동일 로직 적용 → 백테스트 차트에도 캡 결과 반영.

### live bias + 모멘텀 추가 보정 — 합산 캡 추가 (2026-06-04)

**잠재 문제**: `compute_live_bias_correction()`의 ±5$ 캡 이후 모멘텀 추가 보정(-3$)이 별도로 더해져 합산 최대 -8$를 초과할 가능성.

**수정**: 모멘텀 += 이후, 방향 불일치 체크 이전에 `np.clip(bias, -8.0, 8.0)` 추가.
live(±5$) + momentum(±3$) 설계 의도 상한을 코드에서 명시적으로 보장. 초과 시 로그 출력.

### 백테스트 날짜-값 정렬 버그 수정 (2026-06-08)

**버그**: 5/20 백테스트 오차가 -11.64로 비정상적으로 컸음. `_to_bday()`로 영업일
캘린더로 확장된 `full_wti`에서 예측값을 포지션 슬라이싱(`values[-n:]`)으로 추출하는데,
비교 대상 실측값/날짜는 원본 `sx_test.index`(거래일 기준)에서 추출 — 두 시리즈
길이 차이로 예측값-날짜가 어긋나는 정렬 오류 발생 (예외 없이 조용히 발생).

**수정**: `pred_obj.predicted_mean`(DatetimeIndex 보유)과
`full_wti - full_wti.shift(1)`(날짜 자동 정렬)을 조합 후 `.reindex(sx_test.index)`로
명시적 날짜 매핑 — 길이 불일치에 의존하지 않는 정렬로 교체.

**효과**: 5/20 오차 -11.64 → -1.94로 정상화. 전체 backtest 89건 평균 절대오차도
3.567 → 2.994로 개선.

---

### 백테스트 차트에 다중모델 앙상블 예측 추가 (2026-06-08)

**배경**: 라이브는 SARIMAX+VAR+XGB를 역MAE 가중평균으로 결합하지만 백테스트
차트는 SARIMAX 단일 모델만 표시 — 라이브와 동일한 앙상블 예측을 백테스트에서도
보기 위해 개선. (참고: model_performance.csv의 "Stacking(+SARIMAX+XGB+VAR,
미채택)" MAE=3.14는 이미 유사 조합을 시도했으나 XGB 단독(3.00)보다 나빠 정확도
개선을 보장하진 않음을 사전 인지하고 진행)

**구현**: VAR/XGB 결과에 `test_dates` 추가 → SARIMAX·VAR·XGB의 `pred_price_test`를
날짜 교집합 기준 역MAE 가중평균(라이브 기본 앙상블과 동일 방식)으로 결합해 기존
`stacking_pred`/`stacking_error` 슬롯(실제 Stacking 미채택 시 폴백)에 채움 —
기존 `stacking_error` 우선 로직을 통해 대시보드 전체에 자동 반영. 한 모델의
날짜가 없으면 나머지 모델만으로 가중치 재정규화.

**효과**: 추가 학습/연산 거의 없이(기존 `pred_price_test` 배열 재사용) 백테스트
차트에 라이브와 동일한 방식의 다중모델 앙상블 예측선 표시. 라벨 "Stacking 예측" →
"앙상블 예측"으로 수정해 학습된 메타러너가 아닌 가중평균임을 명확히 함.

---

### SARIMAX 백테스트 정답값 정의 불일치 수정 (2026-06-08)

**버그**: 모델 모니터링 탭(약 80%)과 백테스트 탭(51.3%, 동전 던지기 수준)의 SARIMAX
방향성 정확도가 크게 어긋남. 원인은 `pred_price_test`(= "같은 날" `WTI_k` 나우캐스트)와
비교하는 정답값이 `target_price`(= `WTI_{k+1}`, 다음 날)로 채워져 있어 예측·정답이
하루 어긋난 채 비교되고 있었음 — `prediction_log` 실측 대조로 확인(`sarimax_pred`는
당일 실제가와 MAE 0.369로 거의 일치, `actual_price`는 다음날 실제가와 MAE 0.667로
일치). VAR/ETS/XGBoost-Return/Stacking은 예측·정답 정의가 내부적으로 일관됨을
전수 확인 — 이상 없음.

**수정**: `oil_risk_mvp.py:2932` 한 줄, `y_px_te_sx = sx_test['target_price']` →
`sx_test['WTI']`(같은 날 실제가)로 변경. SARIMAX 평가 지표(rmse/mae/r2/ci_calib_q75)와
`prediction_log` 백테스트 행(actual_price/price_error/dir_correct)이 모두 이 값에서
파생되므로 한 줄 수정으로 함께 정합성을 회복.

### SARIMAX 백테스트 데이터 누수(look-ahead) 수정 (2026-06-09)

**버그**: 위 정렬 버그 수정 직후 백테스트 MAE가 $0.39, 방향성 정확도 96.2%로
비현실적으로 좋아지고 "모델 드리프트(라이브/백테스트 MAE 비율 4.44×)" 경고가
새로 발생. 원인은 `oil_risk_mvp.py:2927`의
`pred_price_s = pred_obj.predicted_mean + _wti_returns_s`에서
`_wti_returns_s = full_wti - full_wti.shift(1)`(= **실제 확정 등락폭** `WTI_k - WTI_{k-1}`,
모델이 예측한 변화량이 아님)을 이미 가격 레벨 예측치인 `predicted_mean`(SARIMAX(d=1)의
`get_prediction`이 자동 역차분하여 반환하는 `E[WTI_k|WTI_{<k},exog_k]` 그 자체)에 더해
미래 시점 `k`의 실제 종가 정보가 예측값에 누수된 것. 그 결과
`pred_price_s[k] = WTI_k + (predicted_mean[k] - WTI_{k-1})`이 되어 "예측값"이 사실상
"정답 + 작은 잡음"이 됨.

**증거**: `price_error[k](= actual − pred)`와 `-(predicted_mean[k] - 전일종가)`의
상관계수 = 0.9999999999999999, 최대 절대 차이 7.1×10⁻¹⁵(부동소수점 오차 수준)로
완전히 동일 — 즉 "백테스트 오차"는 forecast 정확도가 아니라 "SARIMAX가 예측한 당일
가격 변동폭의 크기"(평균 ≈$0.40)만 측정하는 의미 없는 값이었음을 수학적으로 확정.
(이 누수는 위 정렬 버그보다 먼저 존재했으나, 두 버그가 서로 가려 "그럴듯한" $1~2대
오차로 보였음 — 정렬 버그를 고치자 누수가 그대로 노출되어 MAE가 $0.39로 떨어진 것)

**수정**: `_wti_returns_s` 가산 로직을 제거하고 `predicted_mean`을 그대로 가격
예측치로 사용 — `pred_price_s = pred_obj.predicted_mean.reindex(sx_test.index)`.
라이브 예측(`forecast_next_7days`, D+1~D+7 forward forecast)과 백테스트(나우캐스트)는
본질적으로 다른 과제이므로, 수정 후 백테스트 MAE가 라이브 MAE($1.50) 수준으로
정상화되고 "모델 드리프트" 거짓 경보가 해소될 것으로 예상.

---

### 뉴스 수집 정확도 개선 + 대시보드 안정성 개선 (2026-06-10)

**뉴스 수집**: Guardian 키워드 필터가 standalone "oil"을 포함해 팜유·엔진오일·
모터스포츠 등 비석유 기사를 다수 수집 — 복합어 기반 필터(`crude oil`,
`oil price`, `oil tanker` 등)로 교체해 노이즈 제거. 동시에 호르무즈 해협·이란
관련 지정학 이슈 전용 쿼리(`GUARDIAN_QUERY_GEO`)를 추가해 블랙스완 감지에
중요한 지정학 뉴스 누락을 보완. 동일 필터를 `forecast_next_7days`/
`save_prediction_log`의 뉴스 캐시 필터링에도 일괄 적용.

**대시보드**: 백테스트 탭의 중복된 "오차 스파이크" 테이블 제거(앙상블 추가로
오차 차트와 중복). 파이프라인 실행 상태를 boolean 플래그 대신 실제
`subprocess.Popen` 객체(`pipeline_proc`)와 `.poll()`로 판단하도록 변경 —
플래그가 프로세스 상태와 어긋나 "실행 중" 표시가 고착되는 문제 방지.

---

## 자동 재훈련

- **REFIT_STALE_DAYS=7**: 7일마다 Optuna 캐시 초기화 → 하이퍼파라미터 재탐색
- **MASE > 임계값 3일 연속**: 모든 모델 캐시 초기화 → 강제 재훈련
  - 임계값: OVX≥60 → 2.0, OVX≥40 → 1.7, 기본 → 1.5

---

## 파일 구조

```
Team_project_oil_risk/
├── oil_risk_mvp.py          # 메인 파이프라인 (~7314줄)
├── dashboard_v2.py          # Streamlit 대시보드
├── requirements.txt
├── run_pipeline.bat / run_rss_alerts.bat / run_dashboard.bat
├── OilRisk.spec             # PyInstaller EXE 스펙
├── .env                     # API 키 (git 제외)
├── config/                  # 작업 스케줄러 XML, 인증 설정
├── data/                    # CFTC annual.txt, GPR xls
├── docs/                    # 개발 히스토리, 논문 링크
│   ├── generate_timeline.py      # 기술 카테고리 타임라인 생성
│   ├── generate_timeline_biz.py  # 비즈니스 발표용 타임라인 생성
│   ├── timeline.png              # 기술 타임라인 이미지
│   └── timeline_biz.png          # 비즈니스 타임라인 이미지 (발표용)
├── experiments/             # 실험 스크립트
├── scripts/                 # 스케줄러 등록 유틸
└── output/                  # 자동 생성
```

---

## 백테스트 예측 한계 및 개선 계획

### 구조적 예측 불가 케이스 ($6+ 오차, 2026-04~05)

| 날짜 | 오차 | 패턴 | 해결 가능성 |
|------|------|------|-------------|
| 4/7  | -$15.17 | 주말 복합 블랙스완 (관세·지정학 이벤트 주말 발생 → 월요일 개장 반영) | 불가 |
| 4/16 | -$11.03 | 당일 급락 — 전일까지 시그널 없음 | 불가 |
| 4/17 | +$6.15  | 급락 직후 반등 미예측 | 전수 시뮬 후 판단 |
| 4/28 | +$6.94  | 블랙스완 급등 — 뉴스 감성 중립 | 불가 |
| 5/5  | -$7.17  | 블랙스완 급락 — 뉴스 감성 중립 | 불가 |

5건 모두 충격 레짐 캡 미발동. 현재 SARIMAX exog(`dxy_change`, `demand_shock`, `supply_shock`, `vix_change`)에 감성·OVX·GPR가 포함되지 않아 외부 충격이 예측값 자체에 반영되지 않는 구조적 한계.

### 다음 개선 실험 예정

| 우선순위 | 방법 | 리스크 |
|----------|------|--------|
| 1 | `vix_change` → `ovx_change` 교체 — 원유 특화 변동성으로 대체 | 낮음 |
| 2 | `news_sentiment_3d` exog 추가 — 감성을 예측값에 직접 반영 | 중간 (선행성 검증 필요) |

### 보류 아이디어 — 뉴스 감성 D+1 한정 반영 (2026-06-09)

장마감~다음날 개장 사이 뉴스 감성을 D+1 예측에만 실측값으로 반영하고 이후
스텝은 기존처럼 가정값(0)을 쓰는 아이디어. `fut_exog` 생성부
(`oil_risk_mvp.py:4633-4642`)에 이미 추출돼 있는 `last_exog`를 스텝0에만
적용하면 구조적으로 구현 가능 — 멀티스텝 미래값을 가정해야 하는 기존 한계가
이 범위에는 적용되지 않음.

**보류 사유**: `news_sentiment`는 `groupby('date')`(`oil_risk_mvp.py:2230`)로
달력일 단위 집계되는데, 캐시(`guardian_news_cache.csv`)의 `date` 컬럼이
`YYYY-MM-DD`뿐이라 "장마감 후~개장 전" 시간창과 정렬 불가. RSS 파싱부
(`oil_risk_mvp.py:1300`)는 `published_parsed`로 시:분:초까지 얻지만 캐싱 시
날짜만 남기고 버림(`oil_risk_mvp.py:1309`). 선행 작업으로 캐시 스키마를
datetime화하고 과거 뉴스 데이터를 재수집해야 적용 가능.

---

## 대시보드 구조

### 사용자 페이지 (`render_user_page`)

**고정 영역**: 리스크 히어로 카드 → 예측 신뢰도 경고 → 리스크 드라이버 → 핵심 지표 6개

**탭 영역**:
- `📈 가격 예측` — 가격 히스토리 + D+1~7 예측 차트
- `📰 뉴스 & 알람` — RSS 알람 + 뉴스 키워드 워드클라우드
- `📊 예측 성과` — 예측 vs 실제, 방향성 정확도, MASE 추이

---

## 상세 문서

- [개발 히스토리 (초기~현재)](docs/DEVELOPMENT_HISTORY.md) — 전체 실험 기록, 채택/롤백 결정 이유
