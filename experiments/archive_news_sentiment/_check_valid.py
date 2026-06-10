import pandas as pd, os, sys
sys.stdout.reconfigure(encoding='utf-8')

OIL_KW = ['crude oil','brent crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker','crude inventory','oil inventory','oil price','oil supply','oil demand','oil production','oil tanker','oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas']

corpus = pd.read_csv('output/guardian_training_corpus.csv')
main   = pd.read_csv('output/guardian_news_cache.csv')

parts = [corpus[['date','title']]]
if os.path.exists('output/guardian_historical_cache.csv'):
    parts.append(pd.read_csv('output/guardian_historical_cache.csv')[['date','title']])
parts.append(main[['date','title']])

combined = pd.concat(parts, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
combined = combined.drop_duplicates(subset=['date','title']).sort_values('date').reset_index(drop=True)
print(f"1) 전체 corpus (중복제거): {len(combined)}건, {combined['date'].nunique()}일")

oil_filtered = combined[combined['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]
print(f"2) OIL_KW 필터 후:        {len(oil_filtered)}건, {oil_filtered['date'].nunique()}일")

import yfinance as yf
brent = yf.download('BZ=F', start='2017-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret1 = brent.pct_change().shift(-1)
price_df = pd.DataFrame({'date_dt': ret1.index, 'ret1': ret1.values})
oil_filtered = oil_filtered.copy()
oil_filtered['date_dt'] = pd.to_datetime(oil_filtered['date'])
merged = oil_filtered.merge(price_df, on='date_dt', how='inner').dropna()
print(f"3) 가격 매칭 후:          {len(merged)}건, {merged['date'].nunique()}일")

merged['label'] = merged['ret1'].apply(lambda r: 'up' if r>0.008 else ('down' if r<-0.008 else 'neutral'))
non_neutral = merged[merged['label']!='neutral']
neutral_cnt = (merged['label']=='neutral').sum()
print(f"4) neutral 제거 후 (±0.8%): {len(non_neutral)}건 (제거: {neutral_cnt}건)")
print(f"   up:{(non_neutral['label']=='up').sum()} / down:{(non_neutral['label']=='down').sum()}")

non_neutral = non_neutral.copy()
non_neutral['_r'] = non_neutral.groupby('date').cumcount()
capped = non_neutral[non_neutral['_r']<10]
print(f"5) cap=10 적용 후:        {len(capped)}건  ← 현재 학습 데이터")
