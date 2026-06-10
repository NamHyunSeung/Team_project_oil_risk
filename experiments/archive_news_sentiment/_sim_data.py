import pandas as pd, numpy as np, yfinance as yf
import sys
sys.stdout.reconfigure(encoding='utf-8')

corpus = pd.read_csv('output/guardian_training_corpus.csv')
main   = pd.read_csv('output/guardian_news_cache.csv')
OIL_KW = ['crude oil','brent crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker','crude inventory','oil inventory','oil price','oil supply','oil demand','oil production','oil tanker','oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas']
main = main[main['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]

combined = pd.concat([corpus[['date','title']], main[['date','title']]], ignore_index=True)
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
combined = combined.drop_duplicates(subset=['date','title']).sort_values('date').reset_index(drop=True)

brent = yf.download('BZ=F', start='2017-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret1 = brent.pct_change().shift(-1)
price_df = pd.DataFrame({'ret1': ret1}).reset_index()
price_df.columns = ['date_dt', 'ret1']
combined['date_dt'] = pd.to_datetime(combined['date'])
merged = combined.merge(price_df, on='date_dt', how='inner').dropna()

print("=== threshold 별 비교 (cap 없음) ===")
for thresh in [0.008, 0.010, 0.015]:
    df = merged.copy()
    df['label'] = df['ret1'].apply(lambda r: 'up' if r>thresh else ('down' if r<-thresh else 'neutral'))
    df = df[df['label']!='neutral']
    n_all  = len(df)
    n_old  = len(df[df['date_dt'].dt.year < 2026])
    n_2026 = len(df[df['date_dt'].dt.year >= 2026])
    print(f"thresh={thresh*100:.1f}%  total={n_all}  2018-2025={n_old}  2026={n_2026}")

print()
print("=== thresh=0.8%, cap=10 ===")
df = merged.copy()
df['label'] = df['ret1'].apply(lambda r: 'up' if r>0.008 else ('down' if r<-0.008 else 'neutral'))
df = df[df['label']!='neutral']
df['_rank'] = df.groupby('date').cumcount()
df = df[df['_rank']<10].drop(columns=['_rank'])
yr = df.groupby(df['date_dt'].dt.year)['label'].count()
print(f"total={len(df)}")
print(yr.to_string())
