"""
news_vol_ma20 추가 효과 재검증 — 고VIF 피처 제거 후 (대시보드 미반영)
- garch_vol(VIF=241.77), RV_21d(VIF=139.03) 등 news_vol_ma20과 겹치는 고VIF 피처를
  제거한 축소 피처셋에서 news_vol_ma20 추가 효과 재테스트
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

import oil_risk_mvp as orm

print("데이터 수집/피처 생성 중 (캐시 활용)...")
price_df = orm.fetch_data()
news_df = orm.fetch_news()
feature_df, full_df, aux = orm.build_features(price_df, news_df)

sent = pd.read_csv('output/oilprice_sentiment_brent.csv', index_col=0, parse_dates=True)
sent['n_articles_ff'] = sent['n_articles'].fillna(0)
sent['news_vol_ma20'] = sent['n_articles_ff'].rolling(20, min_periods=10).mean()
feature_df = feature_df.join(sent[['news_vol_ma20']], how='left')
feature_df['news_vol_ma20'] = feature_df['news_vol_ma20'].ffill().fillna(0)

har_feats = [c for c in orm.HAR_FEATURE_COLS if c in feature_df.columns]
n_test = 90
train_df = feature_df.iloc[:-n_test]
y_rv_tr = train_df['target_rv']


def walk_forward_cv(feat_cols):
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
    return rmse, mae, r2


def show(label, feats):
    rmse, mae, r2 = walk_forward_cv(feats)
    print(f"  [{label:<28}] n={len(feats):2d}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}")
    return rmse, mae, r2


print(f"\nHAR baseline 피처: {len(har_feats)}개")
print("\n=== 고VIF 피처 제거 후 news_vol_ma20 추가 효과 ===")

show("HAR baseline", har_feats)

drop1 = [c for c in har_feats if c != 'garch_vol']
show("-garch_vol", drop1)
show("-garch_vol +news_vol_ma20", drop1 + ['news_vol_ma20'])

drop2 = [c for c in drop1 if c != 'RV_21d']
show("-garch_vol-RV_21d", drop2)
show("-garch_vol-RV_21d +news_vol_ma20", drop2 + ['news_vol_ma20'])

drop3 = [c for c in har_feats if c not in ('garch_vol', 'RV_21d', 'RV_5d')]
show("-garch_vol-RV_21d-RV_5d", drop3)
show("-garch_vol-RV_21d-RV_5d +news_vol_ma20", drop3 + ['news_vol_ma20'])
