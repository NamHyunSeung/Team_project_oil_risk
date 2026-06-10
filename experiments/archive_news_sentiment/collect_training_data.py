"""
FinBERT 파인튜닝용 학습 데이터 수집 + 레이블링
- 수집 대상: 2018-01-01 ~ 2024-06-08 (기존 캐시와 분리)
- 기존 캐시 병합: guardian_historical_cache.csv, guardian_news_cache.csv
- 출력: output/guardian_training_corpus.csv (수집 원본)
         output/finbert_training_data.csv   (레이블 포함)
- 대시보드 파일 무변경
"""
import sys, json, urllib.request, urllib.parse, time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

sys.stdout.reconfigure(encoding='utf-8')

# ── Config ───────────────────────────────────────────────────────────────────
GUARDIAN_API_KEY = "3a287cda-6e49-49f0-8998-3092657e209e"
CORPUS_FILE  = Path("output/guardian_training_corpus.csv")
LABELED_FILE = Path("output/finbert_training_data.csv")

COLLECT_START = "2018-01-01"
COLLECT_END   = "2024-06-08"   # guardian_news_cache 시작일 하루 전

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

LABEL_THRESHOLD = 0.005   # ±0.5% 임계값


# ── Guardian fetch ────────────────────────────────────────────────────────────
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


# ── Step 1: 신규 수집 ─────────────────────────────────────────────────────────
def collect_new() -> pd.DataFrame:
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
    return df


# ── Step 2: 기존 캐시 병합 ────────────────────────────────────────────────────
def merge_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    parts = [new_df]

    # 역사적 캐시 (2020, 2022) — 이미 OIL_KW 필터됨
    hist = Path("output/guardian_historical_cache.csv")
    if hist.exists():
        h = pd.read_csv(hist)
        parts.append(h)
        print(f"Merged historical cache: {len(h)} rows")

    # 메인 캐시 (2024-06-09 이후) — OIL_KW 필터 적용
    main = Path("output/guardian_news_cache.csv")
    if main.exists():
        m = pd.read_csv(main)
        if 'title' in m.columns:
            m = m[m['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]
        parts.append(m[['date', 'title', 'source']] if 'source' in m.columns
                     else m[['date', 'title']].assign(source='Guardian'))
        print(f"Merged main cache: {len(m)} rows (after OIL_KW filter)")

    combined = pd.concat(parts, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
    combined = combined.drop_duplicates(subset=['date', 'title'])
    combined = combined.sort_values('date').reset_index(drop=True)
    print(f"Total after merge: {len(combined)} rows | {combined['date'].min()} ~ {combined['date'].max()}")
    return combined


# ── Step 3: 레이블링 ──────────────────────────────────────────────────────────
def label_data(corpus: pd.DataFrame) -> pd.DataFrame:
    if LABELED_FILE.exists():
        print(f"Labeled file exists: {LABELED_FILE}")
        existing = pd.read_csv(LABELED_FILE)
        print(f"  {len(existing)} rows | label dist:\n{existing['label'].value_counts().to_string()}")
        return existing

    print("\nFetching Brent prices (yfinance 2017-12-01 ~ 2026-06-10)...")
    brent = yf.download('BZ=F', start='2017-12-01', end='2026-06-10',
                        progress=False)['Close'].squeeze()
    brent.index = pd.to_datetime(brent.index).tz_localize(None)
    ret_next = brent.pct_change().shift(-1)   # D+1 수익률
    ret_next.name = 'ret_next'

    corpus['date_dt'] = pd.to_datetime(corpus['date'])

    # 날짜별 기사 집계 (title 첫 번째 기사 대표, 실제 파인튜닝시 모두 활용)
    # 레이블은 날짜 기준으로 부여
    price_df = ret_next.reset_index()
    price_df.columns = ['date_dt', 'ret_next']

    merged = corpus.merge(price_df, on='date_dt', how='inner')
    merged = merged.dropna(subset=['ret_next'])

    # 레이블 부여
    def assign_label(r):
        if r > LABEL_THRESHOLD:
            return 'up'
        elif r < -LABEL_THRESHOLD:
            return 'down'
        else:
            return 'neutral'

    merged['label'] = merged['ret_next'].apply(assign_label)
    merged['ret_pct'] = (merged['ret_next'] * 100).round(3)

    # neutral 제거 (노이즈)
    labeled = merged[merged['label'] != 'neutral'].copy()
    labeled = labeled[['date', 'title', 'label', 'ret_pct']]

    labeled = labeled.reset_index(drop=True)
    labeled.to_csv(LABELED_FILE, index=False)

    print(f"\nLabeled data saved: {len(labeled)} rows -> {LABELED_FILE}")
    print(f"Label distribution:\n{labeled['label'].value_counts().to_string()}")
    print(f"Date range: {labeled['date'].min()} ~ {labeled['date'].max()}")

    # 연도별 분포 확인
    labeled['year'] = pd.to_datetime(labeled['date']).dt.year
    print(f"\nBy year:\n{labeled.groupby('year')['label'].count().to_string()}")
    return labeled


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Step 1: Collect new articles (2018-01-01 ~ 2024-06-08) ===")
    new_df = collect_new()

    print("\n=== Step 2: Merge existing caches ===")
    corpus = merge_existing(new_df)

    print("\n=== Step 3: Label with Brent D+1 returns ===")
    labeled = label_data(corpus)

    print(f"\nDone. Training data ready: {LABELED_FILE}")
