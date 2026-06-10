"""
FinBERT 파인튜닝용 학습 데이터 수집 확장 (2010-01-01 ~ 2018-01-06)
- 기존 corpus(2018-01-07~)와 합쳐 전체 기간을 2010년까지 확장
- 출력: output/guardian_training_corpus_2010.csv (신규 수집분)
"""
import sys, json, urllib.request, urllib.parse, time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

GUARDIAN_API_KEY = "3a287cda-6e49-49f0-8998-3092657e209e"
CORPUS_FILE = Path("output/guardian_training_corpus_2010.csv")

COLLECT_START = "2010-01-01"
COLLECT_END   = "2018-01-06"   # 기존 corpus 시작일(2018-01-07) 하루 전

GUARDIAN_QUERY = (
    '"crude oil" OR "brent crude" OR WTI OR OPEC OR "OPEC+" OR petroleum '
    'OR "oil price" OR "oil supply" OR "oil demand" OR "oil production" '
    'OR "oil sanctions" OR "crude inventory" OR "oil inventory" OR "oil tanker" '
    'OR "oil facility" OR "natural gas" OR LNG'
)
GUARDIAN_QUERY_GEO = (
    '"Strait of Hormuz" OR "Iran nuclear" OR "Iran attack" OR "Israel Iran" '
    'OR "Iran ceasefire" OR "Iran oil" OR "Saudi Arabia oil" OR "OPEC cut" '
    'OR "oil embargo" OR "Middle East oil" OR "oil supply cut"'
)

OIL_KW = [
    'crude oil','brent crude','brent','wti','opec','barrel','petroleum',
    'refinery','shale','tanker','crude inventory','oil inventory',
    'oil price','oil supply','oil demand','oil production','oil tanker',
    'oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas',
]


def fetch_chunk(from_dt: str, to_dt: str, query: str) -> list:
    articles, page = [], 1
    while True:
        params = urllib.parse.urlencode({
            'api-key':    GUARDIAN_API_KEY,
            'q':          query,
            'from-date':  from_dt,
            'to-date':    to_dt,
            'page':       page,
            'page-size':  200,
            'show-fields': 'headline,bodyText',
            'order-by':   'oldest',
        })
        url = f"https://content.guardianapis.com/search?{params}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read())['response']
        except Exception as e:
            print(f"  [WARN] {from_dt}~{to_dt} p{page}: {e}")
            break
        for item in data.get('results', []):
            fields = item.get('fields') or {}
            headline = fields.get('headline') or item.get('webTitle', '')
            body = (fields.get('bodyText') or '')[:300]
            articles.append({
                'date':   item['webPublicationDate'][:10],
                'title':  headline + (' ' + body if body else ''),
                'source': 'Guardian',
            })
        if page >= min(data.get('pages', 1), 10):
            break
        page += 1
        time.sleep(0.5)
    return articles


def collect() -> pd.DataFrame:
    if CORPUS_FILE.exists():
        print(f"Corpus cache found: {CORPUS_FILE}")
        return pd.read_csv(CORPUS_FILE)

    all_articles = []
    cur = date.fromisoformat(COLLECT_START)
    end_d = date.fromisoformat(COLLECT_END)
    chunk_n = 0

    while cur <= end_d:
        nxt = min(cur + timedelta(days=89), end_d)
        f, t = cur.isoformat(), nxt.isoformat()
        arts_m = fetch_chunk(f, t, GUARDIAN_QUERY)
        arts_g = fetch_chunk(f, t, GUARDIAN_QUERY_GEO)
        chunk_n += 1
        print(f"  chunk {chunk_n:>2} {f}~{t}: market={len(arts_m)} geo={len(arts_g)}")
        all_articles.extend(arts_m)
        all_articles.extend(arts_g)
        cur = nxt + timedelta(days=1)

    df = pd.DataFrame(all_articles).drop_duplicates(subset=['date', 'title'])
    df = df[df['title'].str.lower().apply(lambda t: any(kw in t for kw in OIL_KW))]
    df.to_csv(CORPUS_FILE, index=False)
    print(f"Saved {len(df)} articles -> {CORPUS_FILE}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    return df


if __name__ == '__main__':
    print(f"=== Collect {COLLECT_START} ~ {COLLECT_END} ===")
    collect()
