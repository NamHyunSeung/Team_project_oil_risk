"""
뉴스량(news volume) 피처가 HAR 변동성 모델에 도움되는지 검증 (대시보드 미반영, 별도 실험)
- baseline: HAR_FEATURE_COLS
- +1피처: HAR_FEATURE_COLS + news_vol_ma20 (oilprice_sentiment_brent.csv의 n_articles 20일 이동평균)
- 비교: walk-forward TimeSeriesSplit(5-fold) RMSE/MAE/R2 (oil_risk_mvp.train_models()와 동일 설정)
- VIF 체크
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import oil_risk_mvp as orm

print("데이터 수집/피처 생성 중 (캐시 활용)...")
price_df = orm.fetch_data()
news_df = orm.fetch_news()
feature_df, full_df, aux = orm.build_features(price_df, news_df)
print(f"feature_df: {feature_df.shape}")

# ── 뉴스량 피처 병합 ────────────────────────────────────────────────────────
sent = pd.read_csv('output/oilprice_sentiment_brent.csv', index_col=0, parse_dates=True)
sent['n_articles_ff'] = sent['n_articles'].fillna(0)
sent['news_vol_ma20'] = sent['n_articles_ff'].rolling(20, min_periods=10).mean()

feature_df = feature_df.join(sent[['news_vol_ma20']], how='left')
feature_df['news_vol_ma20'] = feature_df['news_vol_ma20'].ffill().fillna(0)
print(f"news_vol_ma20 결측: {feature_df['news_vol_ma20'].isna().sum()} / {len(feature_df)}")

har_feats = [c for c in orm.HAR_FEATURE_COLS if c in feature_df.columns]
print(f"HAR 피처: {len(har_feats)}개")

n_test = 90
train_df = feature_df.iloc[:-n_test]
y_rv_tr = train_df['target_rv']


def walk_forward_cv(feat_cols, label):
    X_tr = train_df[feat_cols]
    _X = X_tr.values
    wf_preds = np.full(len(X_tr), np.nan)
    wf_actual = y_rv_tr.values.copy()
    tscv = TimeSeriesSplit(n_splits=5)
    for idx_tr, idx_va in tscv.split(_X):
        sc = StandardScaler()
        X_f = sc.fit_transform(_X[idx_tr])
        X_v = sc.transform(_X[idx_va])
        y_f = y_rv_tr.iloc[idx_tr]
        covid_w = (np.where(train_df['covid_dummy'].values[idx_tr] == 1, 0.35, 1.0)
                   if 'covid_dummy' in train_df.columns else None)
        m = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.025,
                              subsample=0.8, colsample_bytree=0.7,
                              min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
                              n_jobs=-1, random_state=42, verbosity=0)
        m.fit(X_f, y_f, sample_weight=covid_w)
        wf_preds[idx_va] = m.predict(X_v)
    mask = ~np.isnan(wf_preds)
    rmse = np.sqrt(mean_squared_error(wf_actual[mask], wf_preds[mask]))
    mae = mean_absolute_error(wf_actual[mask], wf_preds[mask])
    r2 = r2_score(wf_actual[mask], wf_preds[mask])
    print(f"  [{label}] RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}")
    return rmse, mae, r2


print("\n=== Walk-forward CV (5-fold) ===")
base_rmse, base_mae, base_r2 = walk_forward_cv(har_feats, "Baseline (HAR_FEATURE_COLS)")
new_feats = har_feats + ['news_vol_ma20']
new_rmse, new_mae, new_r2 = walk_forward_cv(new_feats, "+news_vol_ma20")

print(f"\nRMSE: {base_rmse:.5f} -> {new_rmse:.5f} ({(new_rmse-base_rmse)/base_rmse*100:+.2f}%)")
print(f"MAE : {base_mae:.5f} -> {new_mae:.5f} ({(new_mae-base_mae)/base_mae*100:+.2f}%)")
print(f"R2  : {base_r2:.4f} -> {new_r2:.4f}")

# ── VIF 체크 ────────────────────────────────────────────────────────────────
print("\n=== VIF 체크 (news_vol_ma20 vs 기존 HAR 피처) ===")
X_vif = train_df[new_feats].dropna()
vifs = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_pairs = sorted(zip(new_feats, vifs), key=lambda x: -x[1])
for name, v in vif_pairs[:6]:
    marker = " <- news_vol_ma20" if name == 'news_vol_ma20' else ""
    print(f"  {name:<20} VIF={v:8.2f}{marker}")
news_vif = dict(vif_pairs)['news_vol_ma20']
print(f"\nnews_vol_ma20 VIF = {news_vif:.2f} (보통 <5 양호, >10 다중공선성 우려)")
