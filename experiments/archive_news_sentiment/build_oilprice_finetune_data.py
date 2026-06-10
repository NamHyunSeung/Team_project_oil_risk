"""
OilPrice.com 헤드라인(44,196건) 기반 FinBERT 파인튜닝용 라벨 데이터 생성 (7번째 시도)
입력: output/oilprice_headlines.csv
출력: output/oilprice_finetune_data.csv (date, title, label, ret_pct)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import yfinance as yf

LABEL_THRESHOLD = 0.005  # ±0.5%
OUT = 'output/oilprice_finetune_data.csv'

df = pd.read_csv('output/oilprice_headlines.csv')
df = df[df['category'].apply(lambda c: isinstance(c, str) and c.isascii())].copy()
df = df.dropna(subset=['date', 'title']).reset_index(drop=True)
print(f'헤드라인: {len(df)}건, {df["date"].min()} ~ {df["date"].max()}')

brent = yf.download('BZ=F', start='2009-12-01', end='2026-06-10', progress=False)['Close'].squeeze()
brent.index = pd.to_datetime(brent.index).tz_localize(None)
ret_next = brent.pct_change().shift(-1)  # D+1 수익률
ret_next.name = 'ret_next'

df['date_dt'] = pd.to_datetime(df['date'])
price_df = ret_next.reset_index()
price_df.columns = ['date_dt', 'ret_next']

merged = df.merge(price_df, on='date_dt', how='inner')
merged = merged.dropna(subset=['ret_next'])


def assign_label(r):
    if r > LABEL_THRESHOLD:
        return 'up'
    elif r < -LABEL_THRESHOLD:
        return 'down'
    else:
        return 'neutral'


merged['label'] = merged['ret_next'].apply(assign_label)
merged['ret_pct'] = (merged['ret_next'] * 100).round(3)

labeled = merged[merged['label'] != 'neutral'].copy()
labeled = labeled[['date', 'title', 'label', 'ret_pct']]
labeled = labeled.sort_values('date').reset_index(drop=True)
labeled.to_csv(OUT, index=False)

print(f'\n저장: {len(labeled)}행 -> {OUT}')
print(f'레이블 분포:\n{labeled["label"].value_counts().to_string()}')
print(f'기간: {labeled["date"].min()} ~ {labeled["date"].max()}')

labeled['year'] = pd.to_datetime(labeled['date']).dt.year
print(f'\n연도별:\n{labeled.groupby("year")["label"].count().to_string()}')
