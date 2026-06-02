# 국제 유가 리스크 예측 시스템 — 개발 히스토리

초기 MVP부터 현재까지 전체 실험, 채택/롤백 결정, 성능 변화를 시간순으로 기록한 문서.

---

## Phase 0: 프로젝트 기획

**목표**: WTI 원유 가격의 단기 예측(D+1~D+7)과 리스크 등급 자동 분류  
**배경**: 국제 유가는 지정학, 수급, 거시경제, 뉴스 이벤트가 복합적으로 작용하는 비선형 시계열  
**접근**: 전통 시계열(SARIMAX) + 머신러닝(XGBoost)의 역할 분리 앙상블  
**참고 자료**: `docs/오일 리스크 계획 및 실행 전략 완성본.pdf`, `docs/논문 링크.txt`

---

## Phase 1: 초기 MVP (커밋 `e599ff1`, 2026-05-07)

### 초기 모델 구성

| 모델 | 역할 | 비고 |
|------|------|------|
| SARIMAX(2,1,1) | WTI 가격 D+1~D+7 예측 | statsmodels |
| XGBoost-HAR | 5일 실현변동성(vol_5d) 예측 | walk-forward 5-fold |

### 초기 피처 (28개)

```python
FEATURE_COLS = [
    # HAR 구성요소
    'RV_1d', 'RV_5d', 'RV_21d',
    # 모멘텀
    'return_1d', 'mom_5d', 'mom_21d',
    # 외생 거시변수
    'dxy_change', 'dxy_5d', 'demand_shock', 'supply_shock',
    'geo_dummy', 'gpr_zscore',
    # 뉴스
    'news_sentiment_smooth', 'news_count',
    'news_sentiment_lag1', 'news_count_lag1',
    'news_sentiment_lag2', 'news_count_lag2',
    # 기술적 지표
    'price_vs_ma5', 'price_vs_ma21', 'bb_position',
    'return_lag1', 'return_lag2', 'RV_lag1',
    'vol_5d', 'vol_10d', 'vol_21d', 'brent_wti_spread',
    # 특수
    'covid_dummy',
]
```

### 초기 데이터 소스

- yfinance: WTI(CL=F), Brent(BZ=F), DXY, VIX
- FRED API: 경기 지표
- Guardian API: 뉴스 감성 (keyword 기반)
- GPR: 지정학 리스크 지수 (`data/data_gpr_daily_recent.xls`, 수동 다운로드)

### 초기 파이프라인 구조 (1419줄)

```
fetch_data() → fetch_news() → build_features()
→ train_models() → forecast_next_7days() → save_prediction_log()
→ classify_risk() → extract_crisis_keywords() → generate_wordcloud()
→ plot_oil_forecast()
```

### 초기 출력

- `model_performance.csv` — RMSE/MAE/R² 성능표
- `forecast_7days.csv` — D+1~D+7 예측
- `latest_risk_signal.csv` — 리스크 등급
- `oil_forecast_plot.png` — 6-panel 차트
- `wordcloud.png` — 뉴스 키워드 워드클라우드

---

## Phase 2: 운영 인프라 구축 (2026-05-07 ~ 05-10)

### 예측 로그 시스템 (`dd05fc9`)

- `prediction_log.csv` 도입: run_date 기반 daily 오차 누적
- gap-fill 로직: actual_price 미수신 시 full_df에서 소급 채움
- backtest / live 구분 필드 추가

### Live 오차 피드백 루프 (`72f1161`)

- **Bias 교정 (A)**: 직전 5일 live 오차 평균으로 SARIMAX 예측 보정
- **Live-aware 앙상블 가중치 (B)**: live 오차 누적되면 가중치 동적 조정
- → 이후 지속 개선됨 (Phase 12에서 최종 수정)

### FRED WTI fallback (`f91f8a0`)

- yfinance rollover 시 CL=F 값이 튀는 문제 발생 (2026-05-06/08)
- FRED DCOILWTICO로 90일 이내 이상치 패치

### 이메일 알림 + 로그 로테이션 (`e172fad`)

- SMTP (Gmail) 이메일 알림: SURGE_RISK / DROP_RISK 발생 시 자동 전송
- 로그 파일 5MB 초과 시 `.bak`으로 자동 로테이션

### 대시보드 초기 기능 (`f87039b`)

- 한글 폰트 (Malgun Gothic)
- 피처 중요도 차트
- 5분 자동 새로고침
- EIA API 상태 표시

---

## Phase 3: Prophet 실험 → 제거 (2026-05-11)

### Prophet 추가 (`3690106`, `ec18ba7`)

- 트렌드 + 계절성 분해 모델로 D+1~D+7 앙상블에 편입 시도

### Prophet 제거 (`fae30a0`)

- **결과**: R² = -4.3 → 완전 실패
- **원인**: 유가는 트렌드+계절성 구조가 아닌 충격(jump) 기반 시계열
  - 2020 마이너스 유가, 2022 러시아 침공, 2026 관세 충격 등 극단값 대응 불가
- **결정**: Prophet 영구 제거. 이후 Prophet은 `benchmark_only` 모드로만 유지

---

## Phase 4: 훈련 윈도우 분리 + 앙상블 개선 (2026-05-11)

### 훈련 기간 분리 (`85ef78f`)

- 문제: XGBoost와 SARIMAX를 동일 window(10년)로 훈련 → 각 모델 특성 미반영
- **XGBoost**: 10년 데이터 + 지수감쇠 가중치 (최근 레짐 집중)
- **SARIMAX**: 5년 데이터 (중장기 트렌드 + 계절성)
- Bias 교정 임계값: |error| > 3 → > 2로 강화

### XGBoost-Return 모델 (`0e90396`)

- 변동성(vol_5d)만 예측하던 XGBoost를 수익률 예측으로 전환 시도
- R²-based 앙상블 가중치 추가 (`6452f0f`)
- → 이후 XGBoost-Classifier(방향성)로 최종 전환

---

## Phase 5: 변동성 모델 실험 라운드 (2026-05-12)

### Overfitting 문제 발견 (`fc94053`)

- XGBoost-HAR: train R²=0.80, hold-out R²=0.37 → overfit_gap = 0.44
- 원인: FEATURE_COLS(100+개)를 그대로 변동성 예측에 투입

### 변동성 모델 실험 4종 (`5d240e0`, `1466c3e`, `3d394fb`)

| 실험 | 내용 | 결과 |
|------|------|------|
| A | GARCH(1,1) 레버리지 효과 | 보조 지표로 활용 |
| B | Parkinson 범위 변동성 추가 | 다중공선성 유발 → 이후 제거 |
| C | EWMA 변동성 (λ=10/21/63) | 다중공선성 유발 → 이후 제거 |
| D | Intraday RV(1h) + VIX3M + SKEW | 데이터 수집 불안정 |

### HAR 전용 피처셋 도입 (`f2df09a`)

- HAR_FEATURE_COLS 분리: RV_1d/5d/21d 중심 27개로 제한
- 강한 정규화(L1+L2) 적용
- overfit_gap: 0.44 → 0.08로 개선

### HAR-Ridge vs XGBoost 비교 (`c669ba8`)

- HAR-Ridge가 XGBoost보다 일시적으로 hold-out 성능 우위
- → 최종적으로 XGBoost-HAR 유지 (Optuna 튜닝 후 역전)

---

## Phase 6: 뉴스 감성 파이프라인 개선 (2026-05-13)

### 감성 분석 다층 구조 (`0824403`)

- **TextBlob**: 기본 영문 감성
- **SENTIMENT_MAP**: 유가 도메인 전용 어휘 (Loughran-McDonald 기반, 300+ 단어)
- **PHRASE_SENTIMENT**: 복합 표현 처리 ("supply cut" → +0.6 등)
- **INTENSIFIERS**: 강조어 (significantly, sharply 등) 가중치

### 감성 소스 다양화

- 기존: Guardian API 단독
- 추가: NewsAPI, RSS 50+ 소스 (Reuters, AP, BBC, Al Jazeera 등)
- Guardian은 API 한도 제한으로 보조 소스로 전환

### SARIMAX exog 강화 (`e91554e`)

- 뉴스 감성 + OVX 변동성 + DXY momentum을 SARIMAX 외생변수로 추가
- 동적 앙상블 가중치 스무딩: α=0.3 EMA

---

## Phase 7: SARIMAX 파라미터 실험 (2026-05-14)

### 훈련 윈도우 축소 실험 (`866b752`)

- SARIMAX_YEARS: 5 → 3 테스트
- **결과**: R² 0.756 → 0.724 (악화)
- **롤백** (`256e325`): SARIMAX_YEARS=5 복구

### Optuna 하이퍼파라미터 최적화 (`638f2a4`)

- XGBoost-Return에 Optuna 적용: 외생변수 선택 자동화
- OPEC 회의 이벤트 피처 추가
- → 성능 개선 미미 → 롤백 (`f16304b`)
- → 이후 Optuna는 SARIMAX exog 선택으로 재활용

---

## Phase 8: FinBERT NLP 통합 (2026-05-15)

### FinBERT 기본 통합 (`a01cdae`)

- ProsusAI/finbert (금융 특화 BERT) 도입
- 뉴스 헤드라인 배치 감성 분류: POSITIVE/NEGATIVE/NEUTRAL → {+1, -1, 0}
- GPU 가용 시 자동 사용, 없으면 CPU fallback

### 동적 FinBERT 비중 + SARIMAX exog 편입 시도 (`c9e6bc4`) → 롤백 (`45e2c29`)

- **문제**: FinBERT가 "oil supply cuts" → NEGATIVE (가격엔 POSITIVE) 부호 역전
- SARIMAX exog로 편입 시 예측 방향 악화

### 공급/수요 뉴스 분리 감성 계산 (`2ad743e`) → 롤백 (`5eeb2e1`)

- 공급 관련 뉴스와 수요 관련 뉴스를 분리해 FinBERT 적용 시도
- 분리 기준 모호 + 샘플 수 부족 → 성능 불안정

### FinBERT WTI 방향 예측 fine-tuning (`7a01fd3`) → 롤백 (`657be2b`)

- WTI 가격 방향(UP/DOWN) 라벨로 FinBERT 파인튜닝 시도
- **문제**: 학습 데이터 부족 + 훈련 시간 과도 (일일 실행 불가)

### 최종 FinBERT 채택 형태

- 뉴스 헤드라인 감성 점수 보조 입력으로만 사용
- Sentence Embedding (all-MiniLM-L6-v2) + WTI 상관관계 피처로 대체 활용 (`21695e0`)

---

## Phase 9: 방향성 예측 개선 라운드 (2026-05-16~18)

### Oil Event 라이브러리 (`97b749e`, `490183d`)

- 유가 방향에 영향을 주는 이벤트 사전: 37 → 65개 확장
- "OPEC cut", "Iran sanctions", "Hurricane" 등 키워드 → 방향 스코어

### CFTC CoT 포지셔닝 피처 (`62ab80b`) → 롤백 (`46c3279`)

- Managed Money 롱/숏 포지션 비율을 방향성 피처로 시도
- **문제**: 주간 데이터라 일일 예측에 lag 발생 → 즉시 롤백
- → 이후 CFTC CoT는 raw 피처로만 포함 (신호 생성 아님)

### 방향성 타깃 변환 실험 (`fa65c85`) → 롤백 (`123a4bf`)

- UP/DOWN/FLAT 3-class → binary (UP vs DOWN) 변환
- FLAT 제거 후 정확도 개선 시도 → 성능 동일, 정보 손실

### XGBoost 이진 분류기 전환 (`2ee3e2b`)

- **결과**: dir_acc 55% → **60%** 달성
- 핵심 변경: 회귀→분류, class_weight='balanced', threshold 탐색

### 방향성 개선 실험 3종 롤백 (`ff7203e`)

- 임계값 최적화, 국면 분리, 보팅 앙상블 → 성능 개선 없음 → 전체 롤백

### SVM RBF 분류기 (`8e5b6ed`)

- SVM rbf kernel로 dir_acc **65%** 달성 (in-sample)
- Walk-forward: 54.2% (in-sample보다 대폭 낮음)
- **결정**: SVM 유지 (in-sample 기준 최고), XGBoost-Classifier와 병렬 운용

### 딥러닝 실험 (`582ab5e`, `1f3c042`)

- GRU-Attention, CNN1D 방향성 분류기 실험
- **문제**: 훈련 시간 10분+ → 일일 파이프라인 5분 목표와 충돌
- **결정**: 제거

---

## Phase 10: CEEMDAN 신호 분해 (2026-05-19)

### CEEMDAN 피처 추가 (`6ace118`, `9cd33ea`)

- Ensemble EMD(CEEMDAN)로 WTI 가격 시계열을 IMF 성분으로 분해
- 추세 IMF, 잡음 IMF를 SVM 전용 피처로 주입
- dir_acc 61.7% 달성 (CEEMDAN SVM)

### CEEMDAN deadlock 버그 수정 (`d87a820`)

- Windows에서 CEEMDAN `parallel=True` 시 멀티프로세싱 deadlock
- `parallel=False`로 수정

---

## Phase 11: 스태킹 앙상블 + LightGBM (2026-05-20)

### LightGBM 베이스 모델 추가 (`31a9df4`)

- SARIMAX + XGBoost + LightGBM 3개 베이스 모델
- 계절성 피처 (월별 더미) + SARIMAX 뉴스 감성 exog

### Stacking 메타러너 (`c156900`)

- Ridge 메타러너: SARIMAX + XGBoost + VAR → 최종 예측
- Walk-forward adaptive meta-learner 추가 (후속 커밋)
- 최종 WF-Adapt Stacking MASE = 0.7135

### Stacking 미채택 결정

- **SARIMAX(0,1,0) 단독 MASE = 0.602 < Stacking MASE = 0.7135**
- Stacking 추가 복잡도 대비 성능 열위 → 비교 지표로만 유지
- Stacking 채택 조건: Stacking MAE < SARIMAX MAE × 0.97 (롤링 30일 기준)

---

## Phase 12: 운영 기능 확장 (2026-05-21~25)

### 로그인 + EXE 런처 (`93fbddd`)

- streamlit-authenticator 기반 로그인 화면
- PyInstaller OilRisk.spec으로 Windows EXE 빌드
- 로그인 관련 버그 수정 5건 (`0e61ceb` ~ `8857acb`)

### 관리자 탭 (`fda8eff`, `cda52e8`, `719229b`)

- 사용자 관리 / 시스템 모니터링 / 파이프라인 실행
- user1 계정 추가, 탭별 관리자/사용자 뷰 분리
- 이메일 알림 설정 UI
- 자동 스케줄러 등록/해제 (`54b714f`)
- 관리자 비밀번호 초기화 UI (`d09f3e2`)
- 구독 만료일 관리 + 모바일 반응형 CSS (`9b3f783`)

### 피처 추가 (`d306c5b`, `2591ed6`)

- 천연가스(NG=F), RBOB 가솔린 모멘텀
- 3-2-1 크랙 스프레드 (HO/WTI 비율)
- EIA 재고 서프라이즈 (`inv_surprise`) + 4주 모멘텀 z-score (`inv_mom4_z`)

---

## Phase 13: VAR + ETS 앙상블 고도화 (2026-05-26~28)

### VAR 다변량 모델 추가 (`ad944c6`)

- VAR(9): WTI + Brent + DXY + VIX → OVX 추가 (`9a4ff41`)
- MASE = 0.9998 (Persistence 수준) → 비교 지표로만 사용

### ETS(HW-Damped) 추가 (`3ac36d4`)

- Holt-Winters Damped Exponential Smoothing
- MASE = 1.0 (Persistence 동급) → 비교 지표로만 사용

### Quantile XGBoost (`89910e0`)

- Q10/Q90 예측으로 신뢰구간 시각화
- 이후 75% CI 방식으로 통합

### Ridge-TW 동적 앙상블 (`ad944c6`)

- Temporal Weighting Ridge: 최근 오차에 지수감쇠 가중치
- Walk-forward 30일 rolling 앙상블 가중치

---

## Phase 14: 신뢰성 및 품질 강화 (2026-05-29 ~ 06-01)

### 데이터 누출 수정 2라운드 (`0cbb12b`, `1837d3d`)

- EIA 재고 서프라이즈 자기참조 누출 수정
- classify_risk 분위수 누출 수정
- Winsorization + Regime 라벨 누출 수정

### 모니터링 체계 구축

- 피처 드리프트 감지: 훈련 p01/p99 저장 → live OOD 경고 (`fd49854`)
- CI 보정 추적: 75% 신뢰구간 실제 커버리지 모니터링 (`c84afce`)
- Multi-window MAE: 전체/60일/45일 성능 분리 보고 (`a35f9f1`)
- 라이브 MASE 모니터링: backtest MAE 1.5배 초과 시 경고 (`959c3be`)
- 예측 이상 감지: D+1이 spot 대비 ±30% 초과 시 경고 (`b814b61`)

### MASE 기반 자동 재훈련 (`c12ff30`, `d4908ce`)

- MASE_RETRAIN_THRESH = 1.5
- MASE_RETRAIN_DAYS = 3 (연속 3일 열화)
- 조건 충족 시 Optuna/SVM/GARCH 캐시 전체 초기화
- OVX 수준별 임계값 동적 조정: OVX≥60 → 2.0, OVX≥40 → 1.7

### SARIMAX 최종 파라미터 확정

- SARIMAX(0,1,0): 랜덤워크+드리프트
  - (2,1,1)에서 변경: 고변동 레짐에서 AR/MA 항이 오히려 노이즈 유발
- SARIMAX_WINDOW_YEARS = 3: 최근 레짐 집중 (5년 대비 고변동 구간 대응 우수)
- MASE = **0.602** 확정

### forecast_snapshots.csv + 예측 신뢰도 (`3b38636`, `b450361`)

- run_date별 D+1~D+7 예측 이력 누적
- n=90 스냅샷 도달 시 MASE=0.606 산출 (D+1 기준)
- forecast_reliability: 스냅샷 수 기반 UNKNOWN → LOW → MEDIUM → HIGH

---

## Phase 15: 리스크 신호 고도화 (2026-06-01~02)

### Black Swan 탄력성 (`c48d179`, `c2cb2fb`)

- 관세 관련 키워드(tariff, sanction 등) 감지 → jump_flag 활성화
- jump_flag 활성 시: CI 1.2배 확장, surge_prob 패널티
- OPEC 긴급 회의 감지 → 리스크 점수 가중

### 실시간 RSS 이벤트 경보 (`72c315a`)

- 50+ RSS 소스 4시간 주기 모니터링
- 유가 관련 키워드 감지 시 즉시 이메일 알림
- `run_rss_alerts.bat` / `config/task_schedule_rss.xml`

### VaR + OVX alarm + hedge_ratio (`f2cf98b`, `351e7a9`)

- var_5pct / var_95pct: 5%/95% Value at Risk
- OVX alarm: HIGH(≥50), EXTREME(≥70)
- hedge_ratio: OVX + vol_5d 기반 헤지 비율 권고

---

## Phase 16: 최종 버그 수정 + 다중공선성 제거 (2026-06-02, 커밋 `6f8a980`)

### Bias 교정 소스 수정

- **이전**: live_gap_spikes.csv 스파이크 상위 5개 기반 → 극단값에 과민 반응
- **수정**: prediction_log.csv live 최근 5일 tail → 안정적 편향 추정
- 조건: `|bias| > 1.0`, `np.clip(-10, 10)`, `len >= 3`

### EIA 타임존 수정

- **이전**: `pd.Timestamp.today()` → UTC 서버 환경에서 한국 목요일 미감지
- **수정**: `pd.Timestamp.now('Asia/Seoul')` → KST 기준 요일 판단

### 다중공선성 제거 (G안)

전체 피처 VIF/rank 분석 결과:

**FEATURE_COLS에서 제거 (9개)**:
`parkinson_vol`, `return_neg`, `rv_mom_5_21`, `ewma_vol_10`, `ewma_vol_21`, `ewma_vol_63`, `rv_intraday_5d`, `rv_intraday_21d`, `rv_intraday_vs_close`

**HAR_FEATURE_COLS에서 제거 (5개)**:
`ewma_vol_10`, `ewma_vol_21`, `ewma_vol_63`, `rv_intraday_5d`, `rv_mom_5_21`

**결과**:

| 지표 | 이전 | 이후 | 변화 |
|------|------|------|------|
| FEATURE_COLS 수 | 114 | 105 | -9 |
| HAR_FEATURE_COLS 수 | 27 | 22 | -5 |
| HAR hold-out R² | 0.0719 | 0.0913 | +27% |
| CLS F1 | - | - | +0.7% |

---

## 현재 상태 (2026-06-02)

### 채택 모델

| 역할 | 모델 | 성능 |
|------|------|------|
| 가격 예측 | SARIMAX(0,1,0), 3년 window | MASE=0.602 |
| 변동성 예측 | XGBoost-HAR, HAR 22개 피처 | hold-out R²=0.4681 |
| 방향성 예측 | XGBoost-Classifier | dir_acc=61.1%, wf=53.6% |

### 미채택 모델 (비교용 유지)

| 모델 | MASE | 미채택 이유 |
|------|------|------------|
| Stacking (SARIMAX+XGB+VAR→Ridge) | 0.7135 | SARIMAX 단독보다 열위 |
| VAR(9) | 0.9998 | Persistence 수준 |
| ETS(HW-Damped) | 1.0 | Persistence 수준 |
| Prophet | R²=-4.3 | 충격 기반 시계열 부적합 |

### 주요 알려진 이슈

- **forecast_reliability = UNKNOWN**: 스냅샷 누적 중 → 2026-06-15 이후 자연 해소
- **downside/upside_risk_pct = 50/50**: live 오차 전부 양수 시 폴백값 반환 (의도된 동작)
- **방향성 월별 저하**: 4-5월 고변동 레짐 전환으로 wf_dir_acc 53.6% → REFIT_STALE_DAYS=7로 지속 업데이트 중

---

## 롤백 목록 요약

| 실험 | 롤백 이유 |
|------|-----------|
| Prophet | R²=-4.3, 충격 시계열 부적합 |
| 동적 FinBERT 비중 | FinBERT 감성 부호 역전 (supply cut → NEGATIVE) |
| 공급/수요 분리 감성 | 분리 기준 모호, 샘플 부족 |
| FinBERT fine-tuning | 학습 데이터 부족, 훈련 시간 과도 |
| CFTC CoT 방향 피처 | 주간 데이터 lag → 즉시 롤백 |
| 방향성 타깃 이진화 | FLAT 제거 시 정보 손실, 성능 동일 |
| 방향성 임계값 최적화/국면분리/보팅앙상블 | 성능 개선 없음 |
| GRU-Attn / CNN1D | 훈련 시간 10분+ → 일일 파이프라인 초과 |
| Optuna OPEC 이벤트 | 성능 개선 미미 |
| SARIMAX 3년 window (1차) | R² 0.756→0.724 악화 |
| SVM exponential decay weights | WF 52.8→53.1% (소폭 개선, 유지) |
| CEEMDAN+SVM (EIA CFTC crackspread) | 복잡도 대비 개선 미미 |
| 성능 개선 실험 5종 (세션9) | 전체 성능 열화 |
