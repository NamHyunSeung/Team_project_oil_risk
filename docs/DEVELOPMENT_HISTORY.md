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

### SARIMAX live forecast 버그 수정 (`f14a07c`)

- **버그**: SARIMAX가 train-only 모델로 라이브 예측 → 최근 데이터 반영 안 됨
- **수정**: full-data 모델(훈련+라이브 구간 포함)로 forecast 수행
- 이후 예측값이 최신 가격 레벨에 훨씬 근접

### Live 오차 피드백 루프 (`72f1161`)

- **Bias 교정 (A)**: 직전 5일 live 오차 평균으로 SARIMAX 예측 보정
- **Live-aware 앙상블 가중치 (B)**: live 오차 누적되면 가중치 동적 조정
- → 이후 Phase 16에서 최종 수정 (live tail 5일 median 방식으로 개선)

### FRED WTI fallback (`f91f8a0`)

- yfinance rollover 시 CL=F 값이 튀는 문제 발생 (2026-05-06/08)
- FRED DCOILWTICO로 90일 이내 이상치 패치
- Brent FRED 검증 + API 상태 추적 대시보드 표시 (`5d7a703`)

### 이메일 알림 + 로그 로테이션 (`e172fad`)

- SMTP (Gmail) 이메일 알림: SURGE_RISK / DROP_RISK 발생 시 자동 전송
- 로그 파일 5MB 초과 시 `.bak`으로 자동 로테이션

### 대시보드 초기 기능 + 한국어화 (`f87039b`, `28bd3fc` ~ `7689611`)

- 한글 폰트 (Malgun Gothic)
- 피처 중요도 차트, 5분 자동 새로고침, EIA API 상태 표시
- 한국어 키워드 번역 사전 구축 (300+ 단어): wordcloud 한국어 표시
- 워드클라우드 제목·범례 한국어화

---

## Phase 3: Prophet 실험 → 제거 (2026-05-11)

### Prophet 추가 (`3690106`, `ec18ba7`)

- 트렌드 + 계절성 분해 모델로 D+1~D+7 앙상블에 편입 시도
- 모델 브레이크다운 + 컨센서스 표시 추가

### Prophet 제거 (`fae30a0`)

- **결과**: R² = -4.3 → 완전 실패
- **원인**: 유가는 트렌드+계절성 구조가 아닌 충격(jump) 기반 시계열
  - 2020 마이너스 유가, 2022 러시아 침공, 2026 관세 충격 등 극단값 대응 불가
- **결정**: Prophet 영구 제거. 이후 `benchmark_only` 모드(CatBoost/Prophet monitoring)로만 유지

---

## Phase 4: 훈련 윈도우 분리 + 앙상블 개선 (2026-05-11)

### 훈련 기간 분리 (`85ef78f`)

- 문제: XGBoost와 SARIMAX를 동일 window(10년)로 훈련 → 각 모델 특성 미반영
- **XGBoost**: 10년 데이터 + 지수감쇠 가중치 (최근 레짐 집중)
- **SARIMAX**: 5년 데이터 (중장기 트렌드 + 계절성)
- Bias 교정 임계값: |error| > 3 → > 2로 강화 (`85038a4`)

### XGBoost-Return 모델 (`0e90396`)

- 변동성(vol_5d)만 예측하던 XGBoost를 수익률 예측으로 전환 시도
- R²-based 앙상블 가중치 추가 (`6452f0f`)
- → 이후 XGBoost-Classifier(방향성)로 최종 전환

---

## Phase 5: 변동성 모델 실험 라운드 (2026-05-12)

### Overfitting 문제 발견 (`fc94053`)

- XGBoost-HAR: train R²=0.80, hold-out R²=0.37 → overfit_gap = 0.44
- 원인: FEATURE_COLS(100+개)를 그대로 변동성 예측에 투입
- overfit_gap 컬럼 `model_performance.csv`에 추가 (`c0b4c0f`)

### 변동성 모델 실험 4종 (`5d240e0`, `1466c3e`, `3d394fb`)

| 실험 | 내용 | 결과 |
|------|------|------|
| A | GARCH(1,1) 레버리지 효과 | 보조 지표로 활용 |
| B | Parkinson 범위 변동성 추가 | 다중공선성 유발 → Phase 16에서 제거 |
| C | EWMA 변동성 (λ=10/21/63) | 다중공선성 유발 → Phase 16에서 제거 |
| D | Intraday RV(1h) + VIX3M + SKEW | 데이터 수집 불안정 → Phase 16에서 제거 |

### HAR 전용 피처셋 도입 (`f2df09a`)

- HAR_FEATURE_COLS 분리: RV_1d/5d/21d 중심 27개로 제한
- 강한 정규화(L1+L2) 적용
- overfit_gap: 0.44 → 0.08로 개선

### HAR-Ridge vs XGBoost 비교 (`c669ba8`)

- HAR-Ridge가 XGBoost보다 일시적으로 hold-out 성능 우위
- → 최종적으로 XGBoost-HAR 유지 (Optuna 튜닝 후 역전)

### holdout 평가 복구 (`0201ff0`)

- 이전 편집에서 실수로 삭제된 `rmse_ho`/`r2_ho` holdout 지표 재삽입
- model_performance.csv hold-out 행 정상화

---

## Phase 6: 뉴스 감성 파이프라인 개선 (2026-05-13)

### 감성 분석 다층 구조 (`0824403`)

- **TextBlob**: 기본 영문 감성
- **SENTIMENT_MAP**: 유가 도메인 전용 어휘 (초기 버전)
- **PHRASE_SENTIMENT**: 복합 표현 처리 ("supply cut" → +0.6 등)
- **INTENSIFIERS**: 강조어 (significantly, sharply 등) 가중치

### 감성 소스 다양화

- 기존: Guardian API 단독
- 추가: NewsAPI, RSS 50+ 소스 (Reuters, AP, BBC, Al Jazeera 등)
- Guardian은 API 한도 제한으로 보조 소스로 전환

### SARIMAX exog 강화 (`e91554e`)

- 뉴스 감성 + OVX 변동성 + DXY momentum을 SARIMAX 외생변수로 추가
- 동적 앙상블 가중치 스무딩: α=0.3 EMA

### 워드클라우드 한국어 번역 (`d182d1f`, `2f1fb1f`, `9ddf2d6`)

- 영문 키워드 → 한국어 번역 매핑 추가 (`d182d1f`)
- 모든 키워드 한국어 커버리지 완성 (`2f1fb1f`)
- england/industry/stop/thing/obituary 등 누락 카테고리 보완 + 대시보드 테이블 반영 (`9ddf2d6`)

---

## Phase 7: SARIMAX 파라미터 실험 (2026-05-14)

### 훈련 윈도우 축소 실험 1차 (`866b752`)

- SARIMAX_YEARS: 5 → 3 테스트
- **결과**: R² 0.756 → 0.724 (악화)
- **롤백** (`256e325`): SARIMAX_YEARS=5 복구
- → 이후 Phase 14에서 3년으로 재전환 시 성공 (고변동 레짐 데이터 포함 여부가 핵심)

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

### XGBoost-Return 2-pass 피처 선택 (`224ceb2`)

- 누적 중요도 90% 또는 상위 25개로 피처 자동 선택
- 이후 XGBoost-Classifier로 전환 시에도 동일 방식 유지

### 최종 FinBERT 채택 형태

- 뉴스 헤드라인 감성 점수 보조 입력으로만 사용
- Sentence Embedding (all-MiniLM-L6-v2) + WTI 상관관계 피처로 대체 활용 (`21695e0`)

---

## Phase 9: 방향성 예측 개선 라운드 (2026-05-16~19)

### 성능 개선 실험 5종 (`9dcb07c`) → 전체 롤백

- Ridge-TW 동적 앙상블 도입 직후 5가지 동시 실험
- **결과**: 전체 성능 열화 → 일괄 롤백

### Oil Event 라이브러리 (`97b749e`, `490183d`)

- 유가 방향에 영향을 주는 이벤트 사전: 37 → 65개 확장
- "OPEC cut", "Iran sanctions", "Hurricane" 등 키워드 → 방향 스코어

### CFTC CoT 포지셔닝 피처 (`62ab80b`) → 롤백 (`46c3279`)

- Managed Money 롱/숏 포지션 비율을 방향성 피처로 시도
- **문제**: 주간 데이터라 일일 예측에 lag 발생 → 즉시 롤백
- → 이후 CFTC CoT는 raw 피처로만 포함, Phase 13에서 auto-download 구현

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
- 캔들스틱 패턴 / 웨이블릿 / 거래량 프로파일 / PCA 차원 축소 실험 정리
- **결정**: 전체 제거

### BiLSTM 조건부 앙상블 + SVM C 그리드 서치 (`933a4e2`)

- 감성 재매핑(news_sentiment → 도메인 분류 기반 재산출)
- BiLSTM 조건부 앙상블: 고변동성 레짐에서 BiLSTM 가중치 상향
- SVM C 파라미터 그리드 서치 자동화
- → 이후 `11edc9c` 미사용 코드 정리 시 BiLSTM 제거 (운영 복잡도 증가 대비 개선 미미)

### Pseudo-Huber 손실 적용 (`104326b`)

- XGBoost Stacking 메타러너에 Pseudo-Huber 손실 적용
- **결과**: Stacking MAE 3.637 → **3.596** (채택)
- 이유: 극단 오차에 대한 민감도 줄이면서 L2 손실 특성 유지

### XGB-Cls blend + 역변동성 가중치 (`8d755d9`)

- XGBoost 분류기 blend ratio 도입 (soft probability 혼합)
- 역변동성(1/vol_5d) 가중치로 저변동 구간 예측에 집중
- 크로스에셋(금·구리) + Mutual Information 독립 채널 실험
- → `b41fca6`에서 XGB-Cls 블렌드 라이브 예측 미반영 버그 수정

### MI 기반 피처 선택 (`450e44c`)

- Mutual Information으로 XGBClassifier 방향성 피처 자동 평가
- MI 낮은 피처 제외 → 노이즈 억제 + 과적합 방지

### 분류기 임계값 최적화 실험 (`f40f473`, `bb2ee65`, `ccec2fc`)

| 커밋 | 변경 | 결과 |
|------|------|------|
| `f40f473` | threshold 탐색 0.38~0.62 (고정 0.5 대신 그리드 탐색) | 최적값 탐색 |
| `bb2ee65` | threshold 0.55로 상향 + horizon별 bias decay | 방향성 개선 기대 |
| `ccec2fc` | **0.55→0.51 롤백** | MASE 0.97→1.02 회귀 방지 |

- 임계값 상향으로 방향성 정확도 개선 기대했으나 가격 예측 MASE 악화
- horizon별 bias decay: 장기(D+4~7) 예측 편향 점진적 감쇠
- **결론**: threshold=0.51 확정

### Stacking 채택 기준 수정 (`4c70a0f`)

- 기존: MAE + 방향성 복합 조건 → 수정: **MAE-only** 비교
- actual_price_test 동일 기간 타겟 정렬 (threshold overfitting 수정)

### har_vol_pred 피처 주입 롤백 (`07aae91`)

- XGBoost-Return에 har_vol_pred(변동성 예측값) 피처 주입 실험 → 즉시 롤백
- **결과**: MAE 3.79→3.96 악화, dir_acc 50%→45% 하락 → 노이즈 피처로 판단

---

## Phase 10: SENTIMENT_MAP 확장 + CEEMDAN (2026-05-20)

### SENTIMENT_MAP 유가 도메인 어휘 확장 (`53e2bf1`)

- Loughran-McDonald 금융 어휘 기반 유가 특화 재매핑
- 총 500+ 단어: "crude draw" → +0.8, "refinery outage" → +0.6 등
- 유가 방향에 반하는 금융 감성 어휘 역전 처리

### CEEMDAN 피처 추가 (`6ace118`, `9cd33ea`)

- Ensemble EMD(CEEMDAN)로 WTI 가격 시계열을 IMF 성분으로 분해
- 추세 IMF, 잡음 IMF를 SVM 전용 피처로 주입
- dir_acc 61.7% 달성 (CEEMDAN SVM)

### CEEMDAN deadlock 버그 수정 (`d87a820`)

- Windows에서 CEEMDAN `parallel=True` 시 멀티프로세싱 deadlock
- `parallel=False`로 수정

### SVM 세부 피처 주입 실험 (`afddb30` ~ `b8235d8`)

단계적으로 SVM 전용 피처를 추가하며 walk-forward 방향성 정확도 개선:

| 커밋 | 추가 피처 | WF dir_acc |
|------|----------|-----------|
| `afddb30` | supply_event_score, demand_event_score, geo_event_score | 53.9% |
| `d5dc3eb` | news_uncertainty (뉴스 불확실성 점수) | 54.1% |
| `d097f31` | 지수감쇠 샘플 가중치 (최근 레짐 집중) | 53.1% (일시 악화) |
| `dbed0eb` | Dead-zone 레이블 필터링 (FLAT 구간 제외) | 53.9% |
| `b8235d8` | VRP(ovx_rv_spread, 변동성 위험 프리미엄) | **54.2%** |

- 최종 SVM WF dir_acc = 54.2% (채택)
- WF 5폴드 방향성 평가 상시 추가 (`828ad0e`)

---

## Phase 11: 스태킹 앙상블 + LightGBM (2026-05-21)

### LightGBM 베이스 모델 추가 (`31a9df4`)

- SARIMAX + XGBoost + LightGBM 3개 베이스 모델
- 계절성 피처 (월별 더미) + SARIMAX 뉴스 감성 exog

### Stacking 메타러너 (`c156900`)

- Ridge 메타러너: SARIMAX + XGBoost + VAR → 최종 예측
- Walk-forward adaptive meta-learner 추가 (`9b6d19d`)
- Stacking 채택 조건 개선: 비교 기준 XGBoost → SARIMAX MAE로 수정 (`03541f7`, `eab2098`)

### Stacking WF-Adapt 성능 개선 과정

| 변경 | Stacking MAE |
|------|-------------|
| 초기 Stacking | ~3.725 |
| Pseudo-Huber 손실 (`104326b`) | 3.596 |
| 뉴스 감성 모델 4가지 개선 (`39513cb`) | 3.691 → 3.596 |
| WF-Adapt split 45/45→30/60 (`1a65c0d`) | 2.630 → **2.536** (+3.5%) |
| inv_mom4_z SARIMAX exog (`25e4231`) | 2.536 → 2.532 |
| gold/copper ratio z-scores (`1e09d94`) | 2.532 → **2.482** |
| 최종 WF-Adapt Stacking MASE | 0.7135 |

### Stacking 미채택 결정

- **SARIMAX(0,1,0) 단독 MASE = 0.602 < Stacking MASE = 0.7135**
- Stacking 추가 복잡도 대비 성능 열위 → 비교 지표로만 유지
- Stacking 채택 조건: Stacking MAE < SARIMAX MAE × 0.97 (롤링 30일 기준)

### 파이프라인 실행 시간 최적화 (`ba398c7`, `e58d679`)

- 목표: 일일 파이프라인 < 5분
- 최적화 내용: SVM 캐시, CEEMDAN 캐시, Optuna 캐시 조건부 재사용
- 6개 캐시 최적화 → 5분 이내 달성 (`d891895`)

### 방향성 모델 선택 기준 개선 (`9b0e556`)

- 기존: in-sample dir_acc 기준으로 Classifier-adj 선택
- 수정: walk-forward dir_acc(wf_dir_acc) 기준으로 선택
- 이유: in-sample 과적합 방지

### GPR 파일 자동 갱신 (`ca4fd50`)

- 기존: `data/data_gpr_daily_recent.xls` 수동 다운로드 필요
- 수정: 7일 이상 오래된 경우 공식 URL에서 자동 다운로드
- URL: `https://www.policyuncertainty.com/gpr.html` 직접 갱신

### Session7 리스크 신호 인프라 (`296842f`)

- CI 교정: 예측-실제 잔차 기반 신뢰구간 자동 캘리브레이션
- Multi-step 평가: D+1~D+7 각 horizon별 독립 성능 추적
- Rolling-30d 앙상블 가중치: 30일 롤링 기준 동적 재조정
- direction filter: 임계 이하 방향 신호 제거 (false positive 억제)
- futures_spread proxy: 선물 스프레드 → 시장 방향 보조 신호 활용

### EMA 앙상블 가중치 스무딩 (`e2f209a`)

- live stacking 예측에 EMA(α=0.3) 가중치 스무딩 적용
- 단기 오차 급변 시 앙상블 가중치 과민 반응 억제

---

## Phase 12: 운영 기능 확장 (2026-05-21~25)

### 로그인 + EXE 런처 (`93fbddd`)

- streamlit-authenticator 기반 로그인 화면
- PyInstaller OilRisk.spec으로 Windows EXE 빌드
- 로그인 관련 버그 수정 5건 (`0e61ceb`, `0f55520`, `94c5fed`, `8857acb`): TypeError, 세션 유지 문제, 중복 폼 표시

### 구독 플랜 관리 (`9a44f34`, `fd35e7e`)

- 로그인 화면 구독 플랜 안내 추가
- Pro 단일 플랜으로 단순화

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

### Walk-forward CV 보고 지표 강화 (`059a026`)

- Walk-forward fold별 성능 지표(MAE/MASE/dir_acc) 전체 보고 추가
- Quantile XGBoost(Q10/Q90) 후보 평가 — 이후 Phase 13에서 채택

### RSS 경보 스케줄러 CLI (`f48e621`)

- `--rss-alerts` CLI 플래그 추가: 별도 프로세스로 RSS 모니터링 실행
- Windows Task Scheduler XML 기반 4시간 주기 자동 등록 기능
- → Phase 15 (`72c315a`)에서 실제 50+ 소스 RSS 감지 기능 완성

### 관리자/사용자 기능 보완 (`8d91ec5`, `a23c778`, `fa9434d`, `b04416f`, `4463899`, `5be50d6`)

| 커밋 | 내용 |
|------|------|
| `8d91ec5` | Tab4 SyntaxError 수정 (고아 else 제거 + admin 블록 들여쓰기) |
| `a23c778` | Tab2 리스크 레벨 히스토리 제거 |
| `fa9434d` | 리스크 히스토리를 관리자 로그인 시 Tab2에 표시 |
| `b04416f` | 이메일 알림 수정 + 관리자 탭 이메일 설정 UI |
| `4463899` | 예측 신뢰구간 시각화 개선 + 로그인 버그 수정 |
| `5be50d6` | 코드 최적화: 불필요 블록 제거 및 실행 속도 개선 |

---

## Phase 13: VAR + ETS 앙상블 고도화 (2026-05-26~28)

### VAR 다변량 모델 추가 (`ad944c6`)

- VAR(9): WTI + Brent + DXY + VIX → OVX 추가 (`9a4ff41`)
- MASE = 0.9998 (Persistence 수준) → 비교 지표로만 사용
- VAR fallback 로직: VAR 실패 시 SARIMAX 단독 사용 (`0a90f35`)

### Ridge-TW 동적 앙상블 (`ad944c6`)

- Temporal Weighting Ridge: 최근 오차에 지수감쇠 가중치
- Walk-forward 30일 rolling 앙상블 가중치

### Multi-step Direct SARIMAX (`3ac36d4`)

- D+2~D+7을 각각 별도 모델로 직접 예측 (Direct 방식)
- 기존 recursive 예측 대비 중장기 오차 누적 감소
- Optuna 재튜닝 + XGB 슬라이딩 윈도우도 동시 적용

### ETS(HW-Damped) 추가 (`3ac36d4`)

- Holt-Winters Damped Exponential Smoothing
- MASE = 1.0 (Persistence 동급) → 비교 지표로만 사용
- GARCH 1-step 평가도 추가 (단기 변동성 기준선)

### Quantile XGBoost (`89910e0`)

- Q10/Q90 예측으로 신뢰구간 시각화
- 이후 75% CI 방식으로 통합

### News-Sentiment XGBoost D4 → 정리 (`ab55dac`, `39513cb`, `11edc9c`)

- News-Sentiment XGBoost 베이스 모델 D4 추가: Stacking MAE 3.725→3.691
- 4가지 뉴스 감성 개선 실험으로 일부 개선
- → `11edc9c` 미사용 코드/피처/모델 정리 시 제거 (독립 모델로서 한계)

### CatBoost/Prophet monitoring 모델 (`9730ac2`)

- CatBoost + Prophet을 `benchmark_only` 모드로 재도입
- 목적: 메인 앙상블 성능 비교 기준선, regime 가중치 보조
- 스태킹 임계값 강화: tree-based stacking 채택 기준 조정

### OPEC meeting calendar + EIA direct-download (`0cf05f1`)

- OPEC 정기/긴급 회의 일정 캘린더 피처: `opec_meeting_flag`, `opec_days_to_next`
- EIA API 실패 시 EIA 공식 사이트 직접 다운로드 fallback

### DXY + gold/copper 거시 피처 강화 (`1b6a249`, `1e09d94`)

- DXY 장기 모멘텀 (60/120일) SARIMAX exog 추가
- Optuna 하이퍼파라미터 재탐색 (stale 캐시 초기화)
- 금/구리 비율 z-score SARIMAX exog 추가 (`1e09d94`): Stacking MAE 2.532→2.482

### OPEC + EIA 신호 NEWS_FEATS 통합 (`af80bc9`)

- OPEC 캘린더 + EIA 재고 신호를 NEWS_FEATS 채널로 통합
- 뉴스 피처와 거시 피처의 독립적 앙상블 경로 분리

### CFTC COT auto-download (`7807609`)

- `fetch_cot()` 함수: CFTC 공식 사이트에서 annual.txt 자동 다운로드
- 기존 수동 관리(`data/annualof.txt`) → 자동화

### Backtest window 확장 (`ddb7a86`)

- backtest 기간: 60일 → **90일**로 확장
- 더 긴 평가 구간으로 계절성/레짐 변화 포함

### EIA 제품 재고 + 크랙 스프레드 피처 추가 → 롤백 (`3c1cbab`, `398fc99`, `b2797e8`)

- EIA 휘발유·정제유 재고, CFTC COT, HO=F 3-2-1 크랙 스프레드 피처 추가
- **롤백 이유**: 데이터 수집 불안정 + 기존 크랙 스프레드와 중복 → 복잡도 증가 대비 개선 미미

### 완전 중복 피처 8개 제거 (`f2dd965`)

- FEATURE_COLS에서 완전 중복(Pearson r > 0.98) 피처 8개 제거
- 이후 Phase 16 다중공선성 제거(G안)의 전처리 단계

### Surge detector 단계별 개발 (`812a7ca`, `600ba58`, `fed6730`, `351e7a9`)

SURGE_RISK 신호 감지 엔진: 4단계 단계적 개발

| 커밋 | 변경 | 핵심 지표 |
|------|------|-----------|
| `812a7ca` | additive surge detector 최초 도입 (OVX+momentum+news 합산) | recall=13.5% |
| `600ba58` | threshold 5%, recall 임계 0.30 적용 | recall **13.5%→70.4%** |
| `fed6730` | OVX gate 추가 (낮은 OVX 구간 surge 억제), hedge_ratio 산출 시작 | 오탐 방지 |
| `351e7a9` | temporal filter + OVX calm CI reduction | 연속 오탐 억제 |

- MASE 0.8127 유지: surge recall 대폭 개선에도 가격 예측 열화 없음

### Session8 EIA lag 교정 중간 단계 (`f1595f1`)

- EIA 재고 발표 lag 교정: 발표 후 +3 거래일 시프트 (중간 단계)
- look-ahead audit log 추가: 잠재 데이터 누출 지점 전수 기록
- → Phase 15에서 `shift(1)`로 최종 단축 (EIA_SHIFT=1)

### SARIMAX residual correction 기준 개선 (`14f6f63`)

- 잔차 교정 채택 기준: R² → **MAE 기준**으로 변경 (더 직접적인 예측 오차 지표)
- bias regime reset: 레짐 전환 감지 시 누적 bias 초기화
- HAR overfit_gap: fold 모델 vs 최종 모델 동일 기준 비교 정렬

### Prophet benchmark_only 정렬 (`fdd539e`)

- Phase 13 도입 후 Prophet `benchmark_only` 모드 첫 실행 오류 수정
- HAR 최종 모델 파라미터를 fold 모델과 동기화

### Temporal decay half-life 단축 (`1d616a6`)

- 앙상블 시간 감쇠(temporal decay) half-life 단축
- 목적: 레짐 전환 시 빠른 가중치 적응 (2026 고변동 구간 대응)

### Naive persistence baseline 추가 (`274125e`)

- model_performance.csv에 naive persistence baseline 행 추가
- MASE(persistence 대비 상대 오차), MaxError 지표 추가
- 전체 모델 성능을 persistence 대비로 가시화

### 버그 수정 및 피처 개선 (`23f43a6`, `2452bfa`, `cde420b`)

| 커밋 | 내용 |
|------|------|
| `23f43a6` | news_uncertainty SVM 주입 후보 추가 (컬럼 미생성 시 자동 필터링) |
| `2452bfa` | feature_df→full_df 9곳, yfinance date+1, embedding dim 누출, wf_preds NaN 마스킹, gap_error 0나누기, AEC 재구성, CEEMDAN unlink, SMTP timeout, HAR_FEATURE_COLS 정리, ridge fallback 연결 (총 10건) |
| `cde420b` | 대시보드 모델 성능 표시 정리 |

---

## Phase 14: 신뢰성 및 품질 강화 (2026-05-29 ~ 06-01)

### 데이터 누출 수정 2라운드 (`5613f7a`, `0cbb12b`, `1837d3d`, `52c9081`)

- EIA 재고 서프라이즈 자기참조 누출 수정 (look-ahead 2건)
- classify_risk 분위수 누출 수정 (전체 데이터 분위수 → 훈련 구간 분위수)
- Winsorization + Regime 라벨 누출 수정
- 총 데이터 누출 수정: 4건 (2라운드)

### CI live recalibration (`40112f5`, `521da58`)

- prediction_log의 stacking_error 기반 CI 자동 보정
- live 오차 분포에 따라 75% CI 실시간 확장/축소
- `521da58`: live-type 오차만 사용, stacking_error 컬럼 채움
- `5e8986d`, `c9c61c9`: stacking_error 소급 backfill

### 모니터링 체계 구축

- 피처 드리프트 감지: 훈련 p01/p99 저장 → live OOD 경고 (`1c53f62`)
- CI 보정 추적: 75% 신뢰구간 실제 커버리지 모니터링 (`fd49854`)
- Multi-window MAE: 전체/60일/45일 성능 분리 보고 (`c84afce`)
- 라이브 MASE 모니터링 (`c12ff30`): backtest MAE 1.5배 초과 시 경고
- 라이브 예측 MAE 모니터링 (`959c3be`): 실시간 경고 로깅 (자동 재훈련과 별개)
- 예측 이상 감지: D+1이 spot 대비 ±30% 초과 시 경고 (`b814b61`)
- CI 자동 확장 1.2x (`a35f9f1`): D+1 모델 간 예측 불일치 std >$2 시 CI 확장
- Atomic CSV writes: temp파일 기록 후 rename (`54161a4`) — 파이프라인 중단 시 부분 기록 방지
- Optuna cache key에 피처 컬럼 목록 포함 (`52f1943`) — 피처 변경 시 Optuna 스테일 캐시 방지
- yfinance 데이터 staleness 검사: 최신 데이터가 >2 거래일 지연 시 경고 (`7a8d982`)
- stacking 가중치 이상 감지 로깅: 가중치 급변 시 경고 기록 (`93536f9`)
- meta-learner 최소 샘플 guard: 40 → **60**으로 상향 (`b35f71e`) — 불안정 가중치 방지
- SARIMAX 캐시 키에 exog 컬럼 목록 포함 (`075594c`) — 피처 변경 시 스테일 캐시 재사용 방지

### MASE 기반 자동 재훈련 (`c12ff30`, `d4908ce`)

- MASE_RETRAIN_THRESH = 1.5
- MASE_RETRAIN_DAYS = 3 (연속 3일 열화)
- 조건 충족 시 Optuna/SVM/GARCH 캐시 전체 초기화
- OVX 수준별 임계값 동적 조정: OVX≥60 → 2.0, OVX≥40 → 1.7

### SARIMAX 최종 파라미터 확정 (`036a856`)

- SARIMAX(0,1,0): 랜덤워크+드리프트
  - (2,1,1)에서 변경: 고변동 레짐에서 AR/MA 항이 오히려 노이즈 유발
- SARIMAX_WINDOW_YEARS = 3: 최근 레짐 집중 (5년 대비 2026 관세 충격 구간 집중)
  - 1차 실험(Phase 7)에서 실패했던 3년이 이번엔 성공: 훈련 데이터에 2026 고변동 레짐 포함
- MASE = **0.602** 확정

### D+2-7 예측 개선 (`036a856`, `062ea64`)

- D+2-7: SARIMAX flat forecast (동일값 반복) 제외 → persistence blend 적용
- D+2-7 momentum bias 추가 (`9881ff7`): 방향 신호 기반 drift 보정
- **결과**: D+2-7 예측 방향성 개선, flat forecast 문제 해소

### forecast_snapshots.csv + 예측 신뢰도 (`3b38636`, `3d38564`, `b450361`)

- run_date별 D+1~D+7 예측 이력 누적 (`3b38636`)
- D+1만 기록, run_date당 1행으로 dedup 수정 (`3d38564`)
- n=90 스냅샷 도달 시 MASE=0.606 산출 (D+1 기준) (`b450361`)
- forecast_reliability: 스냅샷 수 기반 UNKNOWN → LOW → MEDIUM → HIGH

### 버그 수정 (`474e34b`, `70817fe`, `22aee5e`, `5f13e15`)

| 커밋 | 내용 |
|------|------|
| `474e34b` | EIA 실패 시 inv_surprise/inv_mom4_z 누락 버그 수정 |
| `70817fe` | data_through를 full_df 기준으로 수정 (실제 최신 데이터 날짜 표시) |
| `22aee5e` | SARIMAX 캐시키 MD5 해시, df_full ffill().bfill(), ExtSVM 행별 CEEMDAN |
| `5f13e15` | Q10/Q90 dict kwargs TypeError, CEEMDAN 훈련 구간 데이터 누출, XGB-Cls blend 혼합 버그 수정 |

---

## Phase 15: 리스크 신호 고도화 (2026-06-01~02)

### VaR + OVX alarm + hedge_ratio (`f2cf98b`)

- var_5pct / var_95pct: 5%/95% Value at Risk
- OVX alarm: HIGH(≥50), EXTREME(≥70)
- hedge_ratio: OVX + vol_5d 기반 헤지 비율 권고
- asymmetric risk ratio: downside/upside_risk_pct 비대칭 리스크

### surge_prob_3d + Surge detector

- 3일 급등 확률 계산
- OVX gate + temporal filter로 오탐 방지
- momentum bias D+2-7 연동 (`9881ff7`)

### CI + momentum bias 개선 (`804008d`, `71c1a05`)

- CI tail coverage 실제 커버리지 추적
- hedge_ratio 강건성 개선
- momentum bias 과보정 방지 (deceleration scaling, `670d3d0`)

### Black Swan 탄력성 (`c48d179`, `c2cb2fb`)

- 관세 관련 키워드(tariff, sanction 등) 감지 → jump_flag 활성화
- jump_flag 활성 시: CI 1.2배 확장, surge_prob 패널티
- OPEC 긴급 회의 감지 → 리스크 점수 가중
- compound shock CI: 복합 충격 시 CI 추가 확장

### 실시간 RSS 이벤트 경보 (`72c315a`)

- 50+ RSS 소스 4시간 주기 모니터링
- 유가 관련 키워드 감지 시 즉시 이메일 알림
- `run_rss_alerts.bat` / `config/task_schedule_rss.xml`

### EIA 데이터 적시성 개선 (`47404c1`)

- EIA 재고 발표: 매주 목요일 KST 기준
- **이전**: `shift(3)` → 3거래일 후 반영
- **수정**: `shift(1)` → 다음 거래일 즉시 반영
- EIA_SHIFT=1: 목요일 발표 당일 반영으로 신호 적시성 향상

### Event-conditional direction thresholds (`294c611`)

- OPEC 회의 / EIA 발표 / jump_flag 활성 시 방향성 임계값 동적 변경
- 이벤트 종류별 threshold 조정 테이블:
  | 이벤트 | threshold 변경 | 목적 |
  |--------|---------------|------|
  | OPEC 긴급회의 감지 | 완화 (더 낮은 신뢰도 허용) | 공급 충격 방향 신호 민감도 ↑ |
  | EIA 발표일 | 유지 (기본값) | 재고 발표 전후 중립 유지 |
  | jump_flag 활성 | 강화 (높은 신뢰도 요구) | 급변 시 오탐 방지 |
- 방향 신호 품질 개선: 이벤트 없는 구간과 이벤트 구간 임계값 분리

### 운영 안정화 (`60ec37b`, `2496812`, `09fa760`, `c860cb9`, `46a0752`)

- regime-adaptive stacking 위기 감지 가중치 강화 (`60ec37b`)
- bias correction: 지수가중평균 → **중앙값**으로 변경 (`2496812`) — 이상치 포함 시 bias 과대추정 방지
- 리스크 신호 3종 개선 (`09fa760`): surge_prob 모멘텀 할인, DROP_RISK 트리거 조건, CI horizon scaling
- direction tracking 추가, forecast reliability flag, D+2-7 momentum drift 개선 (`c860cb9`)
- model versioning 도입, surge_prob 출력 투명성 개선 (`46a0752`)

---

## Phase 15.5: 운영 안정화 및 구조 재편 (2026-05-25~27)

### 대시보드 UX/버그 수정 (`81c909c`, `e3ed5b0`, `b8c9785`, `fc11398`, `3f0dedd`)

- 대시보드 버그 수정 + CI 명칭 80→75 변경 + HAR 감지 개선 (`81c909c`)
- 대시보드 UX 5가지 개선 (`e3ed5b0`)
- 대시보드 UX 4가지 이슈 수정 (`b8c9785`)
- 미사용 mpatches import 제거, Tab2에 RSS 트리거 섹션 추가 (`fc11398`)
- 리스크 신호 상단 RSS 경보 배너 제거 (`3f0dedd`)

### 파이프라인/크래시 안정화 (`b418278`, `df3e88e`, `984b267`, `9453233`, `c4743bc`, `2ef4a6a`, `02a9179`, `147717a`, `5db27cd`)

- stacking_rejected 키 KeyError 크래시 수정 (`b418278`)
- Stacking NameError 수정 (`df3e88e`)
- 크래시/divide-zero guard 6건 수정 (B1-B4/R1/R2) (`984b267`)
- Tab5 앞 관리자 전용 공지 추가 (`9453233`)
- rollback 임계값 수정, 파이프라인 UX 개선 (`c4743bc`)
- 파이프라인 UX 및 CEEMDAN 캐시 가시성 개선 (`2ef4a6a`)
- CL2=F NaN fallback 수정, FRED patch를 90일 이내로 제한 (`02a9179`)
- 파이프라인 파일 로그 핸들러 추가, 타이밍 예측값 수정 (`147717a`)
- dir_acc 및 운영 신뢰성 개선 (`5db27cd`)

### bias_correction / Stacking / VAR 개선 (`8cc06b9`)

- bias_correction decay 수정, Stacking backtest 추가, drift detection 수정, VAR forecast column 추가 (`8cc06b9`)

### 프로젝트 구조 재편 (`ee0de76`, `c9614c5`)

- 프로젝트 파일 구조 재편 (config/ 폴더 분리 등) (`ee0de76`)
- auth_config.yaml 경로를 config/로 업데이트 (`c9614c5`)

### 버그 수정 묶음 (`ba3edf0`, `78ebdb2`, `345a42d`, `76be03b`, `17cb80f`)

- EIA look-ahead audit 메시지 하드코딩 +3 수정 → EIA_SHIFT 변수 사용 (`ba3edf0`)
- CI fallback vol dropna().iloc[-1] all-NaN guard 추가 (`78ebdb2`)
- CEEMDAN 캐시 IMF 길이 < 훈련 윈도우 시 off-by-one 수정 (`345a42d`)
- 번역 캐시 저장 실패 원인인 json import 누락 수정 (`76be03b`)
- SVM 캐시 피처 불일치 수정 + overfit_gap 일관성 개선 (`17cb80f`)

---

## Phase 16: 최종 버그 수정 + 다중공선성 제거 (2026-06-02, 커밋 `6f8a980`)

### 대시보드 버그 17개 수정 + 예측 로그 캡 (`bd7dd3e`)

- 대시보드 크래시/divide-by-zero 17개 버그 수정
- prediction_log.csv 1000행 캡 추가 (무한 증가 방지)
- 관리자 전용 탭 접근 제어 강화

### 불필요 모델 제거 + 주간 Optuna 재탐색 (`cb1a625`)

- 죽은 모델(D2/D3/D4/D5) 제거: 미사용 베이스 모델 전부 정리
- REFIT_STALE_DAYS=7: 7일마다 Optuna 캐시 초기화 → 하이퍼파라미터 재탐색

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

### forecast_snapshots 수정 및 Phase 16 이후 버그 수정 (`787c14a`, `de0755e`, `8ae8f2a`, `743dfd3`, `4bb4499`)

- Stacking 거절 시 순수 SARIMAX fallback 보장 (`787c14a`)
- 대시보드 버그 수정 + tooltip + snapshot analysis panel 추가 (`de0755e`)
- direction accuracy 수정, backtest/live 행 dedup, 대시보드 정리 (`8ae8f2a`)
- monitor_rss_alerts 동기 실행으로 변경 (`743dfd3`)
- live_mae null 버그 수정, actual_direction backfill 추가 (`4bb4499`)

### 리스크 스코어 3가지 개선 (`b6f5afd`)

**방식F (mom_ratio) 채택**:
- `vol_ratio = max(vol_5d/hist_vol_75, min(|mom_5d|/hist_mom_75, 3.5))`
- 이전 방식A~C 대비 불일치 7개 CAUTION+ 개선, 오탐 0건, 상관계수 0.790
- 레벨 분포: SURGE 0→1, HIGH 5→6, CAUTION 49→50, NORMAL 311→308

**공휴일/주말 actual_price fill**:
- `risk_history`에서 공휴일·주말 행의 actual_price를 이전 거래일 값으로 forward fill

**GPR 임계값 완화**:
- `z-score > 1.0` → `z-score > 0.5` (지정학 감지 민감도 향상)

**지정학 키워드 확장 + 뉴스 실시간 보완**:
- `latest_alerts.json` 기반 실시간 geo_triggers 포함

### forecast_snapshots d2~d7 actual 역채움 버그 수정 + 다중 horizon MASE (`09a9ecc`)

**버그 수정**:
- 기존: `_snap_row[_ca] = _snap_row[_cd].map(_price_str_map)` → map에 없는 날짜를 NaN으로 덮어쓰는 버그
- 수정: `fillna` 방식으로 변경 → 기존 actual 값 보존
- 역채움 소스에 `prediction_log.csv` actual_price 추가 → full_df 누락 날짜 보완

**다중 horizon MASE**:
- 기존: D+1 actual_price만으로 MASE 계산
- 개선: d2~d7 각 horizon의 (actual, forecast) 쌍도 오차 pool에 합산해 MASE 계산
- naive baseline은 D+1 시계열 기준 유지, 로그에 `n` (D+1)과 `multi-horizon` (전체) 별도 출력

---

## Phase 17: 감성 충격 보정 버그 수정 (2026-06-04, 커밋 `TBD`)

### 문제

파이프라인 실행 후 D+1 예측가 $83.15 (현재가 $95.24 대비 -12.7%) 이상 발생.

- SARIMAX D+1=$95.31, XGB D+1=$94.14이지만 최종 앙상블=$83.15
- 로그: `⚡ 감성 충격 보정: raw_chg3=-0.665 raw_sent=-0.721 → -13.31$`

### 원인

```python
_shock_adj = _chg3_sh * 20.0  # -0.665 × 20 = -13.31$ 무제한 적용
```

sanity check는 ±30% 기준이라 -13.31$는 경고 없이 통과.

### 수정

```python
_shock_adj = float(np.clip(_chg3_sh * 20.0, -last_price * 0.03, last_price * 0.03))
```

최대 조정폭을 현재가 ±3%로 제한. 수정 후 D+1=$93.35 (정상 범위).

---

## 현재 상태 (2026-06-04)

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

- **forecast_reliability**: 다중 horizon MASE로 계산 범위 확장 (d2~d7 포함), 스냅샷 누적 중 → 2026-06-15 이후 자연 해소
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
| CFTC CoT 방향 피처 (1차) | 주간 데이터 lag → 즉시 롤백 |
| 방향성 타깃 이진화 | FLAT 제거 시 정보 손실, 성능 동일 |
| 방향성 임계값 최적화/국면분리/보팅앙상블 | 성능 개선 없음 |
| 성능 개선 실험 5종 (`9dcb07c`) | 전체 성능 열화 |
| GRU-Attn / CNN1D | 훈련 시간 10분+ → 일일 파이프라인 초과 |
| BiLSTM 조건부 앙상블 | 운영 복잡도 증가, `11edc9c` 정리 시 제거 |
| Optuna OPEC 이벤트 (1차) | 성능 개선 미미 |
| SARIMAX 3년 window (1차, Phase 7) | R² 0.756→0.724 악화 (고변동 레짐 데이터 미포함) |
| EIA 제품 재고 + CFTC COT + 크랙 스프레드 | 데이터 수집 불안정, 기존 피처와 중복 |
| News-Sentiment XGBoost D4 | `11edc9c` 미사용 모델 정리 시 제거 |
| 분류기 임계값 0.55 (`bb2ee65`) | MASE 0.97→1.02 회귀, 0.51로 복구 (`ccec2fc`) |
| Pseudo-Huber 손실 | *채택* (MAE 3.637→3.596 개선) |
