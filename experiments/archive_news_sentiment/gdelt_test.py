"""GDELT DOC 2.0 API 테스트: timelinetone / timelinevol 모드로
유가 관련 일별 톤(감성)/볼륨 시계열을 받아올 수 있는지 + 기간 제한 확인
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import json
import time

BASE = 'https://api.gdeltproject.org/api/v2/doc/doc'
QUERY = '(crude oil OR brent crude OR WTI OR OPEC OR "oil price")'

def call(mode, start, end, retries=8):
    params = {
        'query': QUERY,
        'mode': mode,
        'format': 'json',
        'startdatetime': start,
        'enddatetime': end,
    }
    for attempt in range(retries):
        r = requests.get(BASE, params=params, timeout=30)
        print(f'[{mode}] {start}~{end} status={r.status_code} len={len(r.text)} (attempt {attempt+1})')
        if r.status_code == 200:
            return r
        time.sleep(20)
    return r

# 2) 2010~2017 - 더 과거 데이터 커버리지 확인
r2 = call('timelinetone', '20100101000000', '20170101000000')
print(r2.text[:300])
if r2.status_code == 200:
    d2 = json.loads(r2.text)
    data2 = d2['timeline'][0]['data']
    print(f'  rows={len(data2)}  first={data2[0]["date"]}  last={data2[-1]["date"]}')
    nonzero = [x for x in data2 if x['value'] != 0]
    print(f'  nonzero values: {len(nonzero)}, first nonzero: {nonzero[0]["date"] if nonzero else None}')
