"""
뉴스감성 활용 4가지 방법 검증 (논문 기반, 대시보드 미반영)
1. GARCH-MIDAS 프록시: news_vol_ma60 (저빈도 장기성분)
2. 레짐 상호작용: news_vol_ma20 x high_vol_regime / x geo_dummy
3. 시장심리 대신 뉴스 "감성 극성" (volume 아닌 sent_ma20)
4. 이벤트/충격 기반: news_vol_shock (60일 대비 급증 z-score)
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
sent['news_vol_ma60'] = sent['n_articles_ff'].rolling(60, min_periods=20).mean()
roll_mean60 = sent['n_articles_ff'].rolling(60, min_periods=20).mean()
roll_std60 = sent['n_articles_ff'].rolling(60, min_periods=20).std()
sent['news_vol_shock'] = ((sent['n_articles_ff'].rolling(5, min_periods=3).mean() - roll_mean60)
                           / (roll_std60 + 1e-8))
sent['sent_ma20'] = sent['net_sent_ff'].rolling(20, min_periods=10).mean() if 'net_sent_ff' in sent.columns \
    else sent['net_sent'].fillna(0).rolling(20, min_periods=10).mean()

cols = ['news_vol_ma20', 'news_vol_ma60', 'news_vol_shock', 'sent_ma20']
feature_df = feature_df.join(sent[cols], how='left')
for c in cols:
    feature_df[c] = feature_df[c].ffill().fillna(0)

# 레짐 변수 (rolling 126일 기준 RV_21d 상위 25%)
feature_df['high_vol_regime'] = (feature_df['RV_21d'].rolling(126, min_periods=30)
                                  .rank(pct=True) > 0.75).astype(float).fillna(0)
feature_df['news_vol_x_highvol'] = feature_df['news_vol_ma20'] * feature_df['high_vol_regime']
feature_df['news_vol_x_geo'] = feature_df['news_vol_ma20'] * feature_df.get('geo_dummy', 0.0)

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
    print(f"  [{label:<32}] n={len(feats):2d}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:.4f}")
    return rmse, mae, r2


print(f"\nHAR baseline 피처: {len(har_feats)}개")
print("\n=== 4가지 방법 개별 검증 ===")

base = show("HAR baseline", har_feats)
r1 = show("1.GARCH-MIDAS proxy(+ma60)", har_feats + ['news_vol_ma60'])
r2a = show("2a.+news_vol x high_vol_regime", har_feats + ['news_vol_x_highvol'])
r2b = show("2b.+news_vol x geo_dummy", har_feats + ['news_vol_x_geo'])
r3 = show("3.+sent_ma20(감성극성)", har_feats + ['sent_ma20'])
r4 = show("4.+news_vol_shock(이벤트)", har_feats + ['news_vol_shock'])

print("\n=== 개선된 것들만 조합 ===")
results = {'1': r1, '2a': r2a, '2b': r2b, '3': r3, '4': r4}
feat_map = {'1': 'news_vol_ma60', '2a': 'news_vol_x_highvol', '2b': 'news_vol_x_geo',
            '3': 'sent_ma20', '4': 'news_vol_shock'}
improved = [k for k, v in results.items() if v[0] < base[0]]
print(f"  baseline 대비 RMSE 개선: {improved if improved else '없음'}")
if improved:
    combo_feats = har_feats + [feat_map[k] for k in improved]
    show("combo(개선된 것 전부)", combo_feats)

print("\n=== 요약 ===")
print(f"  {'baseline':<32} RMSE={base[0]:.5f} MAE={base[1]:.5f} R2={base[2]:.4f}")
for k, label in [('1', '1.GARCH-MIDAS proxy'), ('2a', '2a.regime x highvol'),
                  ('2b', '2b.regime x geo'), ('3', '3.sent_ma20'), ('4', '4.news_vol_shock')]:
    v = results[k]
    d_rmse = (v[0]-base[0])/base[0]*100
    print(f"  {label:<32} RMSE={v[0]:.5f}({d_rmse:+.2f}%) MAE={v[1]:.5f} R2={v[2]:.4f}")
