#!/usr/bin/env python3
"""SARIMAX RMSE/MAE 개선 변종 실험"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

sys.path.insert(0, '.')
from oil_risk_mvp import fetch_data, fetch_news, build_features

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("데이터 로드 중...")
price_df    = fetch_data()
news_df     = fetch_news()
feature_df, _ = build_features(price_df, news_df)

N_TEST  = 60
train_df = feature_df.iloc[:-N_TEST]
test_df  = feature_df.iloc[-N_TEST:]
N_TRAIN  = len(train_df)
y_te     = test_df['WTI'].values

EXOG_FULL   = [c for c in ['dxy_change','news_sentiment_smooth','news_count',
                            'demand_shock','supply_shock','geo_dummy',
                            'gpr_zscore','covid_dummy','vix_change']
               if c in feature_df.columns]
EXOG_SIMPLE = [c for c in ['dxy_change','demand_shock','supply_shock','vix_change']
               if c in feature_df.columns]

def _bday(s):
    try: return s.asfreq('B', method='ffill')
    except: return s

# ── 최적 파라미터 탐색 (베이스라인용) ─────────────────────────────────────────
print("\nauto_arima 파라미터 탐색 중...")
wti_tr    = _bday(train_df['WTI'])
exog_tr   = _bday(train_df[EXOG_FULL])
wti_full  = _bday(feature_df['WTI'])
exog_full = _bday(feature_df[EXOG_FULL])

aa = auto_arima(wti_tr, exogenous=exog_tr, d=1, seasonal=True, m=5, D=0,
                start_p=0, max_p=4, start_q=0, max_q=3,
                start_P=0, max_P=2, start_Q=0, max_Q=2,
                information_criterion='aic', stepwise=True,
                error_action='ignore', suppress_warnings=True, n_jobs=-1)
ORDER    = aa.order
S_ORDER  = aa.seasonal_order
print(f"최적 파라미터: {ORDER} × {S_ORDER}")


# ── 공통 평가 함수 ────────────────────────────────────────────────────────────
def run_variant(wti_train, exog_train, wti_full_s, exog_full_s,
                order, s_order, n_tr, log_space=False, bias_window=0):
    try:
        mdl_tr = SARIMAX(wti_train, exog=exog_train, order=order,
                         seasonal_order=s_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = mdl_tr.fit(disp=False, maxiter=300)

        mdl_full = SARIMAX(wti_full_s, exog=exog_full_s, order=order,
                           seasonal_order=s_order,
                           enforce_stationarity=False, enforce_invertibility=False)
        fit_full = mdl_full.filter(fit.params)
        pred_obj = fit_full.get_prediction(start=n_tr, dynamic=False)
        pred = pred_obj.predicted_mean.values[-N_TEST:]

        if log_space:
            pred = np.exp(pred)
        if bias_window > 0:
            bias = float(fit.resid.dropna().iloc[-bias_window:].mean())
            pred = pred - bias

        return {
            'rmse': float(np.sqrt(mean_squared_error(y_te, pred))),
            'mae':  float(mean_absolute_error(y_te, pred)),
            'r2':   float(r2_score(y_te, pred)),
        }
    except Exception as e:
        return {'error': str(e)}


# ── 변종 1: 베이스라인 (현재) ─────────────────────────────────────────────────
print("\n[1/4] 베이스라인 실행...")
r1 = run_variant(wti_tr, exog_tr, wti_full, exog_full, ORDER, S_ORDER, N_TRAIN)
print(f"      RMSE={r1.get('rmse',0):.4f}  MAE={r1.get('mae',0):.4f}  R²={r1.get('r2',0):.4f}")

# ── 변종 2: 로그 변환 ─────────────────────────────────────────────────────────
print("[2/4] 로그 변환 실행...")
wti_tr_log   = _bday(np.log(train_df['WTI']))
wti_full_log = _bday(np.log(feature_df['WTI']))

# 로그 스케일에 최적화된 파라미터 별도 탐색
aa_log = auto_arima(wti_tr_log, exogenous=exog_tr, d=1, seasonal=True, m=5, D=0,
                    start_p=0, max_p=4, start_q=0, max_q=3,
                    start_P=0, max_P=2, start_Q=0, max_Q=2,
                    information_criterion='aic', stepwise=True,
                    error_action='ignore', suppress_warnings=True, n_jobs=-1)
ORDER_LOG   = aa_log.order
S_ORDER_LOG = aa_log.seasonal_order
print(f"      로그용 파라미터: {ORDER_LOG} × {S_ORDER_LOG}")

r2 = run_variant(wti_tr_log, exog_tr, wti_full_log, exog_full,
                 ORDER_LOG, S_ORDER_LOG, N_TRAIN, log_space=True)
print(f"      RMSE={r2.get('rmse',0):.4f}  MAE={r2.get('mae',0):.4f}  R²={r2.get('r2',0):.4f}")

# ── 변종 3: Exog 간소화 ───────────────────────────────────────────────────────
print("[3/4] Exog 간소화 실행...")
exog_tr_s    = _bday(train_df[EXOG_SIMPLE])
exog_full_s  = _bday(feature_df[EXOG_SIMPLE])

aa_s = auto_arima(wti_tr, exogenous=exog_tr_s, d=1, seasonal=True, m=5, D=0,
                  start_p=0, max_p=4, start_q=0, max_q=3,
                  start_P=0, max_P=2, start_Q=0, max_Q=2,
                  information_criterion='aic', stepwise=True,
                  error_action='ignore', suppress_warnings=True, n_jobs=-1)
ORDER_S   = aa_s.order
S_ORDER_S = aa_s.seasonal_order
print(f"      간소화용 파라미터: {ORDER_S} × {S_ORDER_S}")

r3 = run_variant(wti_tr, exog_tr_s, wti_full, exog_full_s, ORDER_S, S_ORDER_S, N_TRAIN)
print(f"      RMSE={r3.get('rmse',0):.4f}  MAE={r3.get('mae',0):.4f}  R²={r3.get('r2',0):.4f}")

# ── 변종 4: COVID 구간 제외 학습 ─────────────────────────────────────────────
print("[4/4] COVID 제외 실행...")
covid_mask  = (train_df.index >= '2020-03-11') & (train_df.index <= '2021-06-30')
train_nc    = train_df[~covid_mask]
wti_tr_nc   = _bday(train_nc['WTI'])
exog_tr_nc  = _bday(train_nc[EXOG_FULL])
N_TRAIN_NC  = len(train_nc)

r4 = run_variant(wti_tr_nc, exog_tr_nc, wti_full, exog_full,
                 ORDER, S_ORDER, N_TRAIN_NC)
print(f"      RMSE={r4.get('rmse',0):.4f}  MAE={r4.get('mae',0):.4f}  R²={r4.get('r2',0):.4f}")

# ── 변종 5: 편향 보정 (30일 잔차 평균) ───────────────────────────────────────
print("[5/4] 편향 보정 실행...")
r5 = run_variant(wti_tr, exog_tr, wti_full, exog_full,
                 ORDER, S_ORDER, N_TRAIN, bias_window=30)
print(f"      RMSE={r5.get('rmse',0):.4f}  MAE={r5.get('mae',0):.4f}  R²={r5.get('r2',0):.4f}")

# ── 변종 6: Exog 간소화 + COVID 제외 (조합) ──────────────────────────────────
print("[6/6] 조합 (Exog 간소화 + COVID 제외) 실행...")
wti_tr_nc_s   = _bday(train_nc['WTI'])
exog_tr_nc_s  = _bday(train_nc[EXOG_SIMPLE])
exog_full_nc_s= _bday(feature_df[EXOG_SIMPLE])

aa_nc_s = auto_arima(wti_tr_nc_s, exogenous=exog_tr_nc_s, d=1, seasonal=True, m=5, D=0,
                     start_p=0, max_p=4, start_q=0, max_q=3,
                     start_P=0, max_P=2, start_Q=0, max_Q=2,
                     information_criterion='aic', stepwise=True,
                     error_action='ignore', suppress_warnings=True, n_jobs=-1)
ORDER_NCS   = aa_nc_s.order
S_ORDER_NCS = aa_nc_s.seasonal_order
print(f"      조합용 파라미터: {ORDER_NCS} × {S_ORDER_NCS}")

r6 = run_variant(wti_tr_nc_s, exog_tr_nc_s, wti_full, exog_full_nc_s,
                 ORDER_NCS, S_ORDER_NCS, N_TRAIN_NC)
print(f"      RMSE={r6.get('rmse',0):.4f}  MAE={r6.get('mae',0):.4f}  R²={r6.get('r2',0):.4f}")

# ── 최종 결과 출력 ────────────────────────────────────────────────────────────
results = {
    '베이스라인 (현재)':          r1,
    '로그 변환':                  r2,
    'Exog 간소화 (4개)':         r3,
    'COVID 제외 학습':            r4,
    '편향 보정 (30일)':           r5,
    '조합 (간소화+COVID제외)':    r6,
}

base_rmse = r1.get('rmse', 99)
base_mae  = r1.get('mae', 99)
base_r2   = r1.get('r2', 0)

print("\n" + "=" * 68)
print(f"{'방법':<26} {'RMSE':>8} {'vs기준':>6} {'MAE':>8} {'vs기준':>6} {'R²':>8}")
print("-" * 68)
for name, res in results.items():
    if 'error' in res:
        print(f"{name:<26} 오류: {res['error'][:40]}")
        continue
    d_rmse = res['rmse'] - base_rmse
    d_mae  = res['mae']  - base_mae
    d_r2   = res['r2']   - base_r2
    r2_warn = " ⚠" if d_r2 < -0.005 else ""
    print(f"{name:<26} {res['rmse']:>8.4f} {d_rmse:>+6.4f} "
          f"{res['mae']:>8.4f} {d_mae:>+6.4f} {res['r2']:>7.4f}{r2_warn}")
print("=" * 68)
print("⚠ = R² 0.5%p 이상 하락")


# ─────────────────────────────────────────────────────────────────────────────
# 과적합 검증
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 68)
print("  과적합 검증")
print("=" * 68)

# ── 1. 학습 내 vs 테스트 격차 ─────────────────────────────────────────────
print("\n[1] 학습 내(In-sample) vs 테스트(Hold-out) 격차")
print("    격차가 작을수록 과적합 가능성 낮음")
print("-" * 68)

def in_sample_rmse_mae(wti_train, exog_train, order, s_order):
    """학습 데이터에 대한 in-sample fitted values 기준 RMSE/MAE"""
    try:
        mdl = SARIMAX(wti_train, exog=exog_train, order=order,
                      seasonal_order=s_order,
                      enforce_stationarity=False, enforce_invertibility=False)
        fit = mdl.fit(disp=False, maxiter=300)
        fitted = fit.fittedvalues.dropna()
        actual = wti_train.loc[fitted.index]
        rmse = float(np.sqrt(mean_squared_error(actual, fitted)))
        mae  = float(mean_absolute_error(actual, fitted))
        return rmse, mae
    except:
        return None, None

configs = {
    '베이스라인':          (wti_tr,      exog_tr,      ORDER,     S_ORDER,     r1),
    'Exog 간소화 (단독)':  (wti_tr,      exog_tr_s,    ORDER_S,   S_ORDER_S,   r3),
    '조합(간소화+COVID제외)': (wti_tr_nc_s, exog_tr_nc_s, ORDER_NCS, S_ORDER_NCS, r6),
}
print(f"{'방법':<24} {'학습RMSE':>9} {'테스트RMSE':>10} {'격차':>8} {'학습MAE':>9} {'테스트MAE':>10} {'격차':>8}")
print("-" * 68)
for cname, (wt, et, od, sod, tres) in configs.items():
    ir, im = in_sample_rmse_mae(wt, et, od, sod)
    tr, tm = tres.get('rmse', 0), tres.get('mae', 0)
    if ir:
        print(f"{cname:<24} {ir:>9.4f} {tr:>10.4f} {tr-ir:>+8.4f} "
              f"{im:>9.4f} {tm:>10.4f} {tm-im:>+8.4f}")

# ── 2. 다중 윈도우 검증 (3개 구간) ───────────────────────────────────────────
print("\n[2] 다중 윈도우 검증 — 3개 구간에서 성능 일관성 확인")
print("    구간별로 성능이 일관되면 과적합 아님")
print("-" * 68)

WINDOWS = [
    ('최근 60일 (W1)',  -60,   None),
    ('61~120일 전 (W2)', -120, -60),
    ('121~180일 전 (W3)', -180, -120),
]

def run_window(wti_series, exog_series, order, s_order, n_tr, n_te):
    """단일 윈도우에 대한 SARIMAX 예측 평가"""
    try:
        tr_s = wti_series.iloc[:n_tr]
        te_s = wti_series.iloc[n_tr:n_tr + n_te]
        ex_tr = exog_series.iloc[:n_tr] if exog_series is not None else None
        ex_full = exog_series.iloc[:n_tr + n_te] if exog_series is not None else None

        mdl_tr = SARIMAX(tr_s, exog=ex_tr, order=order, seasonal_order=s_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = mdl_tr.fit(disp=False, maxiter=300)
        mdl_full = SARIMAX(wti_series.iloc[:n_tr + n_te], exog=ex_full,
                           order=order, seasonal_order=s_order,
                           enforce_stationarity=False, enforce_invertibility=False)
        fit_full = mdl_full.filter(fit.params)
        pred_obj = fit_full.get_prediction(start=n_tr, dynamic=False)
        pred = pred_obj.predicted_mean.values[-n_te:]
        actual = te_s.values[-n_te:]

        return {
            'rmse': float(np.sqrt(mean_squared_error(actual, pred))),
            'mae':  float(mean_absolute_error(actual, pred)),
            'r2':   float(r2_score(actual, pred)),
        }
    except Exception as e:
        return {'error': str(e)}

# 전체 WTI 시리즈 (B-freq, test 이전 전체)
wti_all   = _bday(feature_df['WTI'])
exog_all  = _bday(feature_df[EXOG_FULL])
exog_all_s= _bday(feature_df[EXOG_SIMPLE])
total_n   = len(wti_all)
W = 60  # 윈도우 크기

print(f"{'구간':<20} {'기준':>8} {'간소화':>8} {'조합':>8}   "
      f"{'기준':>7} {'간소화':>7} {'조합':>7}   "
      f"{'기준':>6} {'간소화':>6} {'조합':>6}")
print(f"{'':20} {'RMSE':>8} {'RMSE':>8} {'RMSE':>8}   "
      f"{'MAE':>7} {'MAE':>7} {'MAE':>7}   "
      f"{'R²':>6} {'R²':>6} {'R²':>6}")
print("-" * 80)

for wname, s, e in WINDOWS:
    te_end   = total_n + e if e else total_n
    te_start = total_n + s
    n_tr_w   = te_start
    n_te_w   = te_end - te_start

    if n_tr_w < 200 or n_te_w < 10:
        print(f"{wname:<20} 데이터 부족")
        continue

    wti_nc_w  = wti_all.iloc[:n_tr_w]
    c_mask    = (wti_nc_w.index >= '2020-03-11') & (wti_nc_w.index <= '2021-06-30')
    wti_nc_tr = wti_nc_w[~c_mask]
    ex_nc_tr  = exog_all_s.loc[wti_nc_tr.index]
    n_tr_nc_w = len(wti_nc_tr)

    rb  = run_window(wti_all, exog_all,   ORDER,     S_ORDER,     n_tr_w,    n_te_w)
    rs  = run_window(wti_all, exog_all_s, ORDER_S,   S_ORDER_S,   n_tr_w,    n_te_w)
    rc  = run_window(wti_all, exog_all_s, ORDER_NCS, S_ORDER_NCS, n_tr_nc_w, n_te_w)

    if any('error' in r for r in [rb, rs, rc]):
        print(f"{wname:<20} 오류 발생")
        continue

    print(f"{wname:<20} "
          f"{rb['rmse']:>8.4f} {rs['rmse']:>8.4f} {rc['rmse']:>8.4f}   "
          f"{rb['mae']:>7.4f} {rs['mae']:>7.4f} {rc['mae']:>7.4f}   "
          f"{rb['r2']:>6.4f} {rs['r2']:>6.4f} {rc['r2']:>6.4f}")

print("=" * 80)
print("※ 간소화가 3개 구간 모두에서 기준보다 나쁘지 않으면 안정적인 개선")
