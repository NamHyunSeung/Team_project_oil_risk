import pandas as pd, os, sys
sys.stdout.reconfigure(encoding='utf-8')

OIL_KW = ['crude oil','brent crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker','crude inventory','oil inventory','oil price','oil supply','oil demand','oil production','oil tanker','oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas']

corpus = pd.read_csv('output/guardian_training_corpus.csv')
main   = pd.read_csv('output/guardian_news_cache.csv')
parts  = [corpus[['date','title']], main[['date','title']]]
if os.path.exists('output/guardian_historical_cache.csv'):
    parts.append(pd.read_csv('output/guardian_historical_cache.csv')[['date','title']])

combined = pd.concat(parts, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
combined = combined.drop_duplicates(subset=['date','title']).sort_values('date').reset_index(drop=True)
oil = combined[combined['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))].copy()

import yfinance as yf
brent = yf.download('BZ=F', start='2017-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret1 = brent.pct_change().shift(-1)
price_df = pd.DataFrame({'date_dt': ret1.index, 'ret1': ret1.values})
oil['date_dt'] = pd.to_datetime(oil['date'])
merged = oil.merge(price_df, on='date_dt', how='inner').dropna()

print(f"{'threshold':>12} {'cap':>6} {'건수':>8} {'up':>6} {'down':>6} {'비고'}")
print('-'*55)

for thresh in [0.008, 0.005, 0.003, 0.000]:
    for cap in [None, 30, 10]:
        tmp = merged.copy()
        if thresh > 0:
            tmp['label'] = tmp['ret1'].apply(lambda r: 'up' if r>thresh else ('down' if r<-thresh else 'neutral'))
            tmp = tmp[tmp['label']!='neutral']
        else:
            tmp['label'] = tmp['ret1'].apply(lambda r: 'up' if r>=0 else 'down')
        if cap:
            tmp['_r'] = tmp.groupby('date').cumcount()
            tmp = tmp[tmp['_r']<cap]
        up   = (tmp['label']=='up').sum()
        down = (tmp['label']=='down').sum()
        note = ' ← 현재' if thresh==0.008 and cap==10 else ''
        print(f"  ±{thresh*100:.1f}%      {str(cap) if cap else '없음':>6} {len(tmp):>8} {up:>6} {down:>6}{note}")
    print()
