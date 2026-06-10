"""GDELT DOC 2.0 API: 유가 관련 일별 톤(Average Tone)/볼륨 시계열 수집
- 2017-01-01 ~ 오늘 (단일 파이프라인, 일관된 수집 방식)
- timelinetone: 일별 평균 감성 톤
- timelinevol: 일별 매칭 기사 비율(%)
출력: output/gdelt_oil_tone.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import time
import pandas as pd

BASE = 'https://api.gdeltproject.org/api/v2/doc/doc'
QUERY = '(crude oil OR brent crude OR WTI OR OPEC OR "oil price")'
START = '20170101000000'
END = '20260610000000'


def call(mode, retries=10):
    params = {
        'query': QUERY,
        'mode': mode,
        'format': 'json',
        'startdatetime': START,
        'enddatetime': END,
    }
    for attempt in range(retries):
        r = requests.get(BASE, params=params, timeout=60)
        print(f'[{mode}] status={r.status_code} len={len(r.text)} (attempt {attempt+1})')
        if r.status_code == 200:
            return json.loads(r.text)
        time.sleep(20)
    raise RuntimeError(f'Failed to fetch {mode} after {retries} attempts')


tone = call('timelinetone')
time.sleep(20)
vol = call('timelinevol')

tone_data = tone['timeline'][0]['data']
vol_data = vol['timeline'][0]['data']

df_tone = pd.DataFrame(tone_data).rename(columns={'value': 'tone'})
df_vol = pd.DataFrame(vol_data).rename(columns={'value': 'vol_pct'})

df = df_tone.merge(df_vol, on='date')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%SZ')

print(f'\n총 {len(df)}일, {df["date"].min().date()} ~ {df["date"].max().date()}')
print(df.describe())
print(df.head())

df.to_csv('output/gdelt_oil_tone.csv', index=False)
print('\n저장: output/gdelt_oil_tone.csv')
