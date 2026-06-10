"""
Historical Black Swan Signal Validation
- Periods: COVID (2020-01~05), Russia-Ukraine (2022-02~06)
- Signal: risk_cnt >= 3 (keyword-based, no FinBERT needed)
- Price: yfinance Brent (BZ=F)
- News: Guardian API -> output/guardian_historical_cache.csv (separate from main cache)
"""
import json, urllib.request, urllib.parse, time
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
GUARDIAN_API_KEY = "3a287cda-6e49-49f0-8998-3092657e209e"
CACHE_FILE = Path("output/guardian_historical_cache.csv")

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

PERIODS = [
    ("2020-01-01", "2020-05-31", "COVID"),
    ("2022-02-01", "2022-06-30", "Russia-Ukraine"),
]

OIL_KW = [
    'crude oil','brent crude','brent','wti','opec','barrel','petroleum',
    'refinery','shale','tanker','crude inventory','oil inventory',
    'oil price','oil supply','oil demand','oil production','oil tanker',
    'oil facility','oil sanctions','oil embargo','hormuz',
]

RISK_KW = [
    'attack','war','explos','blockade','halt','sanction','seize','missile',
    'bomb','escalat','conflict','shut','strike','crisis','hormuz',
]


# ── Guardian fetch ───────────────────────────────────────────────────────────
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
            'show-fields':'headline,bodyText',
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
        if page >= min(data.get('pages', 1), 5):
            break
        page += 1
        time.sleep(0.5)  # Guardian free tier: 12 req/s, 여유 있게
    return articles


# ── Collect or load cache ────────────────────────────────────────────────────
def collect_news() -> pd.DataFrame:
    if CACHE_FILE.exists():
        print(f"Cache found: {CACHE_FILE} — loading...")
        return pd.read_csv(CACHE_FILE)

    all_articles = []
    for start, end, label in PERIODS:
        print(f"\nCollecting {label} ({start} ~ {end})...")
        # 분기 단위 청크
        from datetime import date, timedelta
        cur = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        while cur <= end_d:
            nxt = min(cur + timedelta(days=89), end_d)
            f, t = cur.isoformat(), nxt.isoformat()
            arts_m = fetch_chunk(f, t, GUARDIAN_QUERY)
            arts_g = fetch_chunk(f, t, GUARDIAN_QUERY_GEO)
            print(f"  {f} ~ {t}: market={len(arts_m)} geo={len(arts_g)}")
            all_articles.extend(arts_m)
            all_articles.extend(arts_g)
            cur = nxt + timedelta(days=1)

    df = pd.DataFrame(all_articles).drop_duplicates(subset=['date','title'])
    # 복합어 필터
    df = df[df['title'].str.lower().apply(lambda t: any(kw in t for kw in OIL_KW))]
    df.to_csv(CACHE_FILE, index=False)
    print(f"\nSaved {len(df)} articles -> {CACHE_FILE}")
    return df


# ── Signal & accuracy ────────────────────────────────────────────────────────
def evaluate(nc: pd.DataFrame) -> None:
    nc['date'] = pd.to_datetime(nc['date'])
    nc['risk_cnt'] = nc['title'].str.lower().apply(
        lambda t: sum(1 for kw in RISK_KW if kw in t)
    )
    risk_daily = nc.groupby('date')['risk_cnt'].sum()

    # Brent 가격 수집
    print("\nFetching Brent prices (yfinance)...")
    price_start = min(p[0] for p in PERIODS)
    price_end   = max(p[1] for p in PERIODS)
    brent = yf.download('BZ=F', start=price_start, end=price_end,
                        progress=False)['Close'].squeeze()
    brent.index = pd.to_datetime(brent.index).tz_localize(None)

    ret_next = brent.pct_change().shift(-1) * 100

    df = pd.DataFrame({'price': brent, 'ret': ret_next}).join(
        risk_daily.rename('risk_cnt')
    ).dropna(subset=['ret'])
    df['risk_cnt'] = df['risk_cnt'].fillna(0)

    # 기존 방향 (예측 없으므로 "전일 방향 유지" 전략으로 대체)
    df['prev_ret'] = df['ret'].shift(1)
    naive_dir = (df['prev_ret'] > 0) == (df['ret'] > 0)

    # 신호 조정: risk >= 3 → 상향 → 상승 예측
    signal_days = df['risk_cnt'] >= 3
    adj_dir = naive_dir.copy()
    # 신호 발화 시: 상승 예측(ret > 0이면 맞음)
    adj_dir[signal_days] = df.loc[signal_days, 'ret'] > 0

    print(f"\n{'='*50}")
    print(f"{'Period':<20} {'Days':>5} {'Naive':>8} {'Signal':>8} {'Delta':>8}")
    print(f"{'-'*50}")

    for start, end, label in PERIODS:
        mask = (df.index >= start) & (df.index <= end)
        sub = df[mask]
        sig_m = signal_days[mask]
        naive_acc = ((sub['prev_ret'] > 0) == (sub['ret'] > 0)).mean()
        adj_acc   = adj_dir[mask].mean()
        print(f"{label:<20} {mask.sum():>5} {naive_acc:>8.1%} {adj_acc:>8.1%} {adj_acc-naive_acc:>+8.1%}p")

    print(f"{'-'*50}")
    total_naive = naive_dir.mean()
    total_adj   = adj_dir.mean()
    print(f"{'TOTAL':<20} {len(df):>5} {total_naive:>8.1%} {total_adj:>8.1%} {total_adj-total_naive:>+8.1%}p")

    # 블랙스완 커버
    bs = df[df['ret'].abs() > 4.0]
    sig_fires_bs = signal_days[bs.index].sum()
    print(f"\nBlack swan (|ret|>4%): {len(bs)}일")
    print(f"  signal fires on BS days: {sig_fires_bs}/{len(bs)} ({sig_fires_bs/len(bs):.1%})")
    print(f"  signal fires total: {signal_days.sum()} days")

    # 신호 발화일 예시 (BS 날짜)
    print("\nBS days with signal (risk_cnt >= 3):")
    for d, row in df[signal_days & (df['ret'].abs() > 4.0)].iterrows():
        print(f"  {d.date()} ret={row['ret']:+.1f}%  risk={int(row['risk_cnt'])}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    nc = collect_news()
    print(f"\nTotal articles after filter: {len(nc)}")
    print(f"Date range: {nc['date'].min()} ~ {nc['date'].max()}")
    evaluate(nc)
