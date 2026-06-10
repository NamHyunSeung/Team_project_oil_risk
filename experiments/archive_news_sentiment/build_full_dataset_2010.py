"""전체 코퍼스(2010-2026) 병합 + Brent D+1 레이블링
입력: guardian_training_corpus_2010.csv, guardian_training_corpus.csv,
      guardian_historical_cache.csv, guardian_news_cache.csv
출력: output/finbert_training_data_2010.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import yfinance as yf

OIL_KW = [
    'crude oil','brent crude','brent','wti','opec','barrel','petroleum',
    'refinery','shale','tanker','crude inventory','oil inventory',
    'oil price','oil supply','oil demand','oil production','oil tanker',
    'oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas',
]
LABEL_THRESHOLD = 0.005

# ── 1. 코퍼스 병합 ───────────────────────────────────────────────────────────
parts = []
for f in ['output/guardian_training_corpus_2010.csv',
          'output/guardian_training_corpus.csv',
          'output/guardian_historical_cache.csv']:
    d = pd.read_csv(f)
    parts.append(d[['date', 'title', 'source']] if 'source' in d.columns
                  else d[['date', 'title']].assign(source='Guardian'))
    print(f'{f}: {len(d)} rows')

main = pd.read_csv('output/guardian_news_cache.csv')
if 'title' in main.columns:
    main = main[main['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]
parts.append(main[['date', 'title', 'source']] if 'source' in main.columns
              else main[['date', 'title']].assign(source='Guardian'))
print(f'output/guardian_news_cache.csv: {len(main)} rows (OIL_KW 필터 후)')

corpus = pd.concat(parts, ignore_index=True)
corpus['date'] = pd.to_datetime(corpus['date']).dt.strftime('%Y-%m-%d')
corpus = corpus.drop_duplicates(subset=['date', 'title'])
corpus = corpus.sort_values('date').reset_index(drop=True)
print(f'\n병합 후: {len(corpus)} rows | {corpus["date"].min()} ~ {corpus["date"].max()}')

# ── 2. Brent 가격 (2009-12-01 ~) ────────────────────────────────────────────
print('\nFetching Brent prices (yfinance 2009-12-01 ~ 2026-06-10)...')
brent = yf.download('BZ=F', start='2009-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret_next = brent.pct_change().shift(-1)
ret_next.name = 'ret_next'

corpus['date_dt'] = pd.to_datetime(corpus['date'])
price_df = ret_next.reset_index()
price_df.columns = ['date_dt', 'ret_next']

merged = corpus.merge(price_df, on='date_dt', how='inner')
merged = merged.dropna(subset=['ret_next'])

def assign_label(r):
    if r > LABEL_THRESHOLD:
        return 'up'
    elif r < -LABEL_THRESHOLD:
        return 'down'
    return 'neutral'

merged['label'] = merged['ret_next'].apply(assign_label)
merged['ret_pct'] = (merged['ret_next'] * 100).round(3)

labeled = merged[merged['label'] != 'neutral'].copy()
labeled = labeled[['date', 'title', 'label', 'ret_pct']].reset_index(drop=True)

OUT = 'output/finbert_training_data_2010.csv'
labeled.to_csv(OUT, index=False)
print(f'\n저장: {len(labeled)} rows -> {OUT}')
print(f'Label 분포:\n{labeled["label"].value_counts().to_string()}')
print(f'Date range: {labeled["date"].min()} ~ {labeled["date"].max()}')

labeled['year'] = pd.to_datetime(labeled['date']).dt.year
print(f'\n연도별:\n{labeled.groupby("year").size().to_string()}')

cnt = labeled.groupby('date').size()
print(f'\n날짜당 최대 기사수: {cnt.max()}, 고유 날짜수: {labeled["date"].nunique()}')
