import pandas as pd, os, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('output/finbert_training_data.csv')
df['date'] = pd.to_datetime(df['date'])
sz = df.groupby('date').size()
print('=== Current data ===')
print(f'Total: {len(df)}')
print(f'Unique dates: {df["date"].nunique()}')
print(f'Per-day avg: {len(df)/df["date"].nunique():.1f}')
print(f'Per-day max: {sz.max()}')
print()

OIL_KW = ['crude oil','brent crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker','crude inventory','oil inventory','oil price','oil supply','oil demand','oil production','oil tanker','oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas']

corpus = pd.read_csv('output/guardian_training_corpus.csv')
main   = pd.read_csv('output/guardian_news_cache.csv')
main   = main[main['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]

parts = [corpus[['date','title']], main[['date','title']]]
if os.path.exists('output/guardian_historical_cache.csv'):
    parts.append(pd.read_csv('output/guardian_historical_cache.csv')[['date','title']])

combined = pd.concat(parts, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
combined = combined.drop_duplicates(subset=['date','title']).sort_values('date').reset_index(drop=True)
print(f'Full corpus: {len(combined)} articles, {combined["date"].nunique()} days')
print()

import yfinance as yf
brent = yf.download('BZ=F', start='2017-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret1 = brent.pct_change().shift(-1)
price_df = pd.DataFrame({'date_dt': ret1.index, 'ret1': ret1.values})
combined['date_dt'] = pd.to_datetime(combined['date'])
merged = combined.merge(price_df, on='date_dt', how='inner').dropna()
merged['label'] = merged['ret1'].apply(lambda r: 'up' if r>0.008 else ('down' if r<-0.008 else 'neutral'))
non_neutral = merged[merged['label']!='neutral']

print('=== Cap simulation (thresh=0.8%) ===')
for cap in [10, 20, 50, None]:
    tmp = non_neutral.copy()
    if cap:
        tmp['_r'] = tmp.groupby('date').cumcount()
        tmp = tmp[tmp['_r']<cap]
    print(f'  cap={str(cap) if cap else "none":>5}: {len(tmp)} rows')
