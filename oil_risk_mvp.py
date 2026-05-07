#!/usr/bin/env python3
"""
국제 유가 리스크 예측 시스템 MVP
International Oil Price Risk Prediction System

Models  : XGBoost-HAR (volatility) + SARIMAX (price forecast)
Features: WTI/Brent, DXY, demand/supply shock, news sentiment, news count, geo-dummy
Output  : model_performance.csv, forecast_7days.csv, latest_risk_signal.csv,
          crisis_keywords.csv, oil_forecast_plot.png, wordcloud.png
"""

import sys, warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
# Windows 콘솔 UTF-8 설정 (emoji/한글 출력 안전화)
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
import os, re, logging
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# ── 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: fallback)
def _setup_korean_font():
    candidates = ['Malgun Gothic', 'Apple SD Gothic Neo', 'AppleGothic',
                  'NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams['font.family'] = c
            break
    plt.rcParams['axes.unicode_minus'] = False

_setup_korean_font()
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────────
try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False
    log.warning("yfinance 없음 → 더미 데이터 사용")

try:
    from fredapi import Fred as _Fred
    _FRED = True
except ImportError:
    _FRED = False
    log.warning("fredapi 없음 → 더미 충격변수 사용")

try:
    import feedparser as _feedparser_mod
    _FEED = True
except ImportError:
    _FEED = False
    log.warning("feedparser 없음 → 더미 뉴스 사용")

# ── API 키 & 파일 경로 설정
FRED_API_KEY     = "0a1d6c8b56c44eff8716c204f0aa49bf"
GUARDIAN_API_KEY = "3a287cda-6e49-49f0-8998-3092657e209e"
GPR_FILE         = "data_gpr_daily_recent.xls"   # 프로젝트 폴더에 위치
DATA_YEARS       = 10                             # 학습 데이터 기간

# ── COVID 특수처리 기간
COVID_START = '2020-03-11'   # WHO 팬데믹 선언일
COVID_END   = '2021-06-30'   # 백신 보급 안정화 시점

try:
    from textblob import TextBlob
    _TB = True
except ImportError:
    _TB = False

try:
    from wordcloud import WordCloud as _WC_Class
    _WC = True
except ImportError:
    _WC = False
    log.warning("wordcloud 없음 → 바 차트로 대체")

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _SARIMAX = True
except ImportError:
    _SARIMAX = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    _SKL = True
except ImportError:
    _SKL = False
    log.warning("scikit-learn 없음 → 일부 기능 제한")

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_LEVELS = {
    'NORMAL':     {'color': '#2ecc71', 'label': '정상',      'emoji': '🟢'},
    'CAUTION':    {'color': '#f39c12', 'label': '주의',      'emoji': '🟡'},
    'SURGE_RISK': {'color': '#e74c3c', 'label': '급등위험',  'emoji': '🔴'},
    'DROP_RISK':  {'color': '#3498db', 'label': '급락위험',  'emoji': '🔵'},
}

CRISIS_SEED = {
    'war', 'conflict', 'sanction', 'opec', 'supply', 'cut', 'shortage',
    'embargo', 'attack', 'explosion', 'disruption', 'geopolitical',
    'iran', 'russia', 'ukraine', 'israel', 'hamas', 'houthi',
    'pipeline', 'refinery', 'hurricane', 'recession', 'inflation',
    'strategic', 'reserve', 'spr', 'production', 'inventory',
    'energy', 'crisis', 'surge', 'crash', 'crude', 'oil', 'brent',
    'barrel', 'demand', 'risk', 'fear', 'concern', 'saudi', 'china',
    'export', 'price', 'tension',
}

STOP_WORDS = {
    'the','a','an','in','on','at','to','for','of','and','or','but','is',
    'are','was','were','be','been','has','have','had','will','would','could',
    'should','may','might','as','with','by','from','up','about','into',
    'through','before','after','that','this','these','those','it','its',
    'new','more','than','over','which','who','when','where','how','what',
    'not','no','said','says','amid','its','their','they','our','we','us',
    'amid','amid','its','after','amid'
}

SENTIMENT_MAP = {
    'war':-2,'attack':-2,'explosion':-2,'collapse':-2,'crash':-2,
    'crisis':-2,'shortage':-2,'embargo':-2,'sanction':-2,'shutdown':-2,
    'disruption':-2,'conflict':-2,'spike':-1.5,'surge':-1,
    'cut':-1,'fall':-1,'decline':-1,'risk':-1,'concern':-1,
    'fear':-1,'threat':-1,'tension':-1,'dispute':-1,'recession':-1,
    'recovery':1,'growth':1,'increase':1,'rise':1,'deal':1,
    'agreement':1,'stable':1,'boom':2,'peace':2,'resolution':2,
}

NEWS_RSS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.marketwatch.com/rss/topstories",
    "https://rss.cnn.com/rss/money_news_international.rss",
]

DUMMY_HEADLINES = [
    "OPEC+ agrees to extend production cuts amid falling oil prices",
    "Russia sanctions escalate, supply disruption fears rise sharply",
    "US crude oil inventories rise more than expected, demand concerns grow",
    "Iran nuclear deal talks collapse, geopolitical tensions spike in region",
    "Hurricane threatens Gulf of Mexico oil production platforms",
    "Fed raises interest rates again, oil demand outlook dims for 2024",
    "Ukraine conflict intensifies, energy market volatility surges to record",
    "China economic slowdown hits oil demand forecasts across Asia",
    "Houthi attacks on Red Sea shipping disrupt oil tanker routes",
    "Strategic Petroleum Reserve release announced to combat price surge",
    "OPEC production cut deeper than expected, bullish price outlook",
    "US recession fears grow as crude oil prices fall sharply",
    "Israel-Hamas conflict widens, Middle East oil risk premium rises",
    "Oil refinery fire causes supply shortage on East Coast",
    "OPEC meeting ends without agreement, crude price crashes 5%",
    "Saudi Arabia announces voluntary output cut extension into next quarter",
    "Global oil demand forecast lowered by IEA amid recession risk signals",
    "Pipeline sabotage disrupts European natural gas supply routes",
    "Libya oil fields shut down amid renewed civil conflict in south",
    "Dollar strengthens sharply, oil prices under pressure from DXY rally",
    "US shale production hits record high, supply surplus fears return",
    "Energy crisis deepens in Europe ahead of winter heating season",
    "Iraq oil exports suspended due to dispute with Kurdistan region",
    "Oil price war fears return as OPEC members clash on quotas again",
    "Nigerian oil production disrupted by militant attacks on pipelines",
    "Iran threatens to close Strait of Hormuz amid escalating sanctions",
    "Venezuela oil sanctions relaxed, supply expected to increase",
    "Brent crude falls below $70 as demand outlook weakens globally",
    "WTI crude spikes on unexpected massive inventory draw this week",
    "OPEC+ surprise cut sends oil prices surging above $90 per barrel",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  fetch_data()
# ─────────────────────────────────────────────────────────────────────────────

def _dummy_prices(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    np.random.seed(42)
    rets = np.random.normal(0.0003, 0.018, n)
    shocks = [int(n * r) for r in [0.18, 0.42, 0.67, 0.83]]
    for idx in shocks:
        direction = np.random.choice([-1, 1])
        rets[idx:idx+3] += direction * np.random.uniform(0.03, 0.06)
    wti = 78.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        'WTI':         wti,
        'Brent':       wti * 1.035 + np.random.normal(0, 0.4, n),
        'DXY':         103.0 + np.cumsum(np.random.normal(0, 0.25, n)),
        'demand_shock': np.random.normal(0, 2.0, n),   # Mbbl weekly inventory ∆
        'supply_shock': np.random.normal(0, 1.5, n),   # Mbbl/d production ∆
        'geo_dummy':   np.zeros(n),
    }, index=dates)
    for idx in shocks:
        s, e = max(0, idx-2), min(n, idx+10)
        df.iloc[s:e, df.columns.get_loc('geo_dummy')] = 1
    return df


def _attach_gpr(df: pd.DataFrame) -> pd.DataFrame:
    """
    GPR Index (Caldara & Iacoviello 2022) 로딩 후 df에 결합.
    - geo_dummy : GPRD z-score > 1.0 (상위 ~16%) 이면 1, 아니면 0
    - gpr_zscore: 연속형 표준화 GPR (피처로 추가 활용)
    """
    gpr_path = Path(GPR_FILE)
    if not gpr_path.exists():
        log.warning(f"    GPR 파일 없음({GPR_FILE}) → geo_dummy=0 사용")
        df['geo_dummy']  = 0.0
        df['gpr_zscore'] = 0.0
        return df

    try:
        gpr_raw = pd.read_excel(gpr_path, engine='xlrd', usecols=['date', 'GPRD'])
        gpr_raw['date'] = pd.to_datetime(gpr_raw['date'])
        gpr_raw = gpr_raw.set_index('date')['GPRD'].dropna()
        gpr_raw = gpr_raw[~gpr_raw.index.duplicated()]

        # 전체 기간 기준 z-score 정규화 (1985-2019 기준 100이므로 그대로도 의미있음)
        mu, sigma = gpr_raw.mean(), gpr_raw.std()
        gpr_z = (gpr_raw - mu) / sigma

        # df 인덱스에 맞춰 정렬 (없는 날은 앞날 값으로 채움)
        gpr_aligned = gpr_z.reindex(df.index).ffill().bfill().fillna(0)

        df['gpr_zscore'] = gpr_aligned
        df['geo_dummy']  = (gpr_aligned > 1.0).astype(float)  # 상위 16% 이상 = 지정학 위기

        n_events = int(df['geo_dummy'].sum())
        log.info(f"    GPR 연결 완료: 지정학 위기일 {n_events}일 "
                 f"({n_events/len(df)*100:.1f}%) / 기준 z>1.0")
    except Exception as exc:
        log.warning(f"    GPR 로딩 실패({exc}) → geo_dummy=0")
        df['geo_dummy']  = 0.0
        df['gpr_zscore'] = 0.0

    return df


def _attach_fred_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    FRED API로 실제 수요충격·공급충격 데이터 수집 후 df에 결합.
    실패 시 난수 더미로 폴백.

    사용 시리즈 (검증된 FRED 코드)
    ───────────────────────────────
    DCOILBRENTEU : Brent Crude Oil Prices (daily)  → Brent-WTI 스프레드 변화 = 수요충격
    DCOILWTICO   : WTI Crude Oil Prices (daily)    → 스프레드 계산 기준
    DHHNGSP      : Henry Hub Natural Gas (daily)   → 에너지 공급충격 proxy
    """
    n = len(df)

    if not _FRED or not FRED_API_KEY:
        log.info("    FRED 미설정 → 더미 충격변수 사용")
        np.random.seed(42)
        df['demand_shock'] = np.random.normal(0, 2.0, n)
        df['supply_shock']  = np.random.normal(0, 1.5, n)
        df = _attach_gpr(df)
        return df

    fred = _Fred(api_key=FRED_API_KEY)

    # ── 수요충격: Brent-WTI 스프레드 일변화 ───────────────────────────────
    # 스프레드 확대 = 국제 공급 타이트 = 수요 압력 신호
    try:
        log.info("    FRED 수집: Brent 가격 (DCOILBRENTEU)...")
        brent_fred = fred.get_series('DCOILBRENTEU',
                                     observation_start=start_date,
                                     observation_end=end_date)
        wti_fred = fred.get_series('DCOILWTICO',
                                   observation_start=start_date,
                                   observation_end=end_date)
        spread     = brent_fred - wti_fred          # Brent 프리미엄 ($/bbl)
        spread_chg = spread.diff()                  # 일별 스프레드 변화
        spread_daily = spread_chg.resample('B').ffill()
        df['demand_shock'] = spread_daily.reindex(df.index).ffill().bfill().fillna(0)
        log.info(f"      수요충격(스프레드Δ): μ={df['demand_shock'].mean():.3f}, "
                 f"σ={df['demand_shock'].std():.3f}")
    except Exception as exc:
        log.warning(f"      수요충격 FRED 실패({exc}) → 더미 사용")
        np.random.seed(1)
        df['demand_shock'] = np.random.normal(0, 2.0, n)

    # ── 공급충격: Henry Hub 천연가스 일변화율 ─────────────────────────────
    # 가스 가격 급등 = 에너지 공급 타이트 = 유가 상방 압력
    try:
        log.info("    FRED 수집: Henry Hub 천연가스 (DHHNGSP)...")
        gas = fred.get_series('DHHNGSP',
                              observation_start=start_date,
                              observation_end=end_date)
        gas_chg   = gas.pct_change() * 100          # 일별 변화율 (%)
        gas_daily = gas_chg.resample('B').ffill()
        df['supply_shock'] = gas_daily.reindex(df.index).ffill().bfill().fillna(0)
        log.info(f"      공급충격(HenryHubΔ%): μ={df['supply_shock'].mean():.3f}, "
                 f"σ={df['supply_shock'].std():.3f}")
    except Exception as exc:
        log.warning(f"      공급충격 FRED 실패({exc}) → 더미 사용")
        np.random.seed(2)
        df['supply_shock'] = np.random.normal(0, 1.5, n)

    # ── 지정학 더미: GPR Index (Caldara & Iacoviello) ─────────────────────
    df = _attach_gpr(df)

    log.info("    FRED 실제 데이터 연결 완료 ✓")
    return df


def fetch_data(start_date=None, end_date=None):
    """yfinance로 WTI·Brent·DXY 수집; 실패 시 더미 데이터 반환"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=365 * DATA_YEARS)).strftime('%Y-%m-%d')

    log.info(f"[1/9] 가격 데이터 수집: {start_date} ~ {end_date}")

    if not _YF:
        return _dummy_prices(start_date, end_date)

    try:
        def _dl(ticker):
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(raw, pd.DataFrame):
                col = 'Close' if 'Close' in raw.columns else raw.columns[0]
                s = raw[col]
                if isinstance(s, pd.DataFrame):   # MultiIndex ticker残留
                    s = s.iloc[:, 0]
            else:
                s = raw
            s.name = ticker
            return s

        wti   = _dl("CL=F")
        brent = _dl("BZ=F")
        dxy   = _dl("DX-Y.NYB")

        df = pd.DataFrame({'WTI': wti, 'Brent': brent, 'DXY': dxy})
        df = df.ffill().bfill()
        df.dropna(subset=['WTI'], inplace=True)

        if len(df) < 60:
            log.warning("데이터 부족 → 더미 데이터 사용")
            return _dummy_prices(start_date, end_date)

        # ── FRED 실제 데이터 연결 ──────────────────────────────────────────
        df = _attach_fred_data(df, start_date, end_date)

        log.info(f"    yfinance 성공: {len(df):,} rows")
        return df

    except Exception as e:
        log.warning(f"yfinance 오류({e}) → 더미 데이터 사용")
        return _dummy_prices(start_date, end_date)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  fetch_news()
# ─────────────────────────────────────────────────────────────────────────────

GUARDIAN_QUERY = (
    'oil OR "crude oil" OR "brent crude" OR WTI OR OPEC OR petroleum '
    'OR "energy crisis" OR "oil price" OR "oil supply" OR "oil demand" '
    'OR "natural gas" OR LNG OR "oil production" OR "oil sanction"'
)
NEWS_CACHE_FILE   = OUTPUT_DIR / 'guardian_news_cache.csv'


def _guardian_fetch_chunk(api_key: str, from_dt: str, to_dt: str) -> list:
    """Guardian API에서 특정 기간 뉴스 수집 (페이지네이션 포함)"""
    import urllib.request, json as _json, urllib.parse
    articles = []
    page = 1
    while True:
        params = urllib.parse.urlencode({
            'api-key':    api_key,
            'q':          GUARDIAN_QUERY,
            'from-date':  from_dt,
            'to-date':    to_dt,
            'page':       page,
            'page-size':  200,
            'show-fields':'headline',
            'order-by':   'oldest',
        })
        url = f"https://content.guardianapis.com/search?{params}"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = _json.loads(r.read())['response']
        except Exception as exc:
            log.debug(f"Guardian chunk 실패 ({from_dt}~{to_dt} p{page}): {exc}")
            break

        for item in data.get('results', []):
            title = (item.get('fields') or {}).get('headline') or item.get('webTitle', '')
            articles.append({
                'date':   item['webPublicationDate'][:10],
                'title':  title,
                'source': 'Guardian',
            })

        if page >= min(data.get('pages', 1), 5):   # 청크당 최대 5페이지(1000건)
            break
        page += 1
    return articles


def fetch_news(days_back: int = None):
    """
    Guardian API로 유가 뉴스 수집 (10년치 캐시 지원).
    캐시 파일이 있으면 신규 기사만 추가 수집.
    Guardian API 실패 시 RSS → 더미 순으로 폴백.
    """
    if days_back is None:
        days_back = 365 * DATA_YEARS

    log.info(f"[2/9] 뉴스 수집 중 (최근 {days_back//365}년, {days_back}일)...")

    start_dt = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_dt   = datetime.today().strftime('%Y-%m-%d')

    # ── 캐시 로드 ──────────────────────────────────────────────────────────
    cache_df = pd.DataFrame(columns=['date', 'title', 'source'])
    if NEWS_CACHE_FILE.exists():
        try:
            cache_df = pd.read_csv(NEWS_CACHE_FILE, parse_dates=['date'])
            cache_df['date'] = pd.to_datetime(cache_df['date']).dt.normalize()
            log.info(f"    캐시 로드: {len(cache_df)}건 (최근: {cache_df['date'].max().date()})")
            # 캐시 이후 날짜만 새로 수집
            last_cached = cache_df['date'].max().strftime('%Y-%m-%d')
            if last_cached >= end_dt:
                log.info("    캐시 최신 → API 호출 생략")
                return cache_df.sort_values('date').reset_index(drop=True)
            start_dt = last_cached   # 캐시 이후 구간만 수집
        except Exception:
            cache_df = pd.DataFrame(columns=['date', 'title', 'source'])

    # ── Guardian API 수집 (분기 단위 청크) ────────────────────────────────
    new_articles = []
    if GUARDIAN_API_KEY:
        try:
            chunk_start = datetime.strptime(start_dt, '%Y-%m-%d')
            chunk_end   = datetime.today()
            CHUNK_DAYS  = 90   # 분기 단위

            total_chunks = max(1, int((chunk_end - chunk_start).days / CHUNK_DAYS) + 1)
            log.info(f"    Guardian API 수집: {total_chunks}개 청크")

            cur = chunk_start
            while cur < chunk_end:
                nxt  = min(cur + timedelta(days=CHUNK_DAYS), chunk_end)
                arts = _guardian_fetch_chunk(
                    GUARDIAN_API_KEY,
                    cur.strftime('%Y-%m-%d'),
                    nxt.strftime('%Y-%m-%d'),
                )
                new_articles.extend(arts)
                log.info(f"      {cur.strftime('%Y-%m')} ~ {nxt.strftime('%Y-%m')}: "
                         f"+{len(arts)}건")
                cur = nxt + timedelta(days=1)

            log.info(f"    Guardian 신규 수집: {len(new_articles)}건")
        except Exception as exc:
            log.warning(f"    Guardian API 오류({exc}) → RSS 폴백")

    # ── RSS 폴백 ───────────────────────────────────────────────────────────
    if len(new_articles) < 20 and _FEED:
        cutoff = datetime.today() - timedelta(days=60)
        OIL_FILTER = {'oil','crude','brent','wti','opec','energy','barrel','petroleum','pipeline'}
        for url in NEWS_RSS:
            try:
                feed = _feedparser_mod.parse(url)
                for entry in feed.entries[:30]:
                    try:
                        pub = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else datetime.today()
                    except Exception:
                        pub = datetime.today()
                    title = entry.get('title', '')
                    if pub >= cutoff and any(kw in title.lower() for kw in OIL_FILTER):
                        new_articles.append({'date': pub.strftime('%Y-%m-%d'),
                                             'title': title, 'source': 'RSS'})
            except Exception:
                pass

    # ── 더미 폴백 ──────────────────────────────────────────────────────────
    if len(new_articles) < 10:
        log.info("    더미 뉴스 보충")
        np.random.seed(7)
        new_articles += [
            {'date': (datetime.today() - timedelta(days=int(np.random.randint(0, 30)))).strftime('%Y-%m-%d'),
             'title': h, 'source': 'dummy'}
            for h in DUMMY_HEADLINES
        ]

    # ── 캐시 병합 & 저장 ───────────────────────────────────────────────────
    new_df   = pd.DataFrame(new_articles)
    new_df['date'] = pd.to_datetime(new_df['date']).dt.normalize()
    combined = pd.concat([cache_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'title'])
    combined = combined.sort_values('date').reset_index(drop=True)

    try:
        combined.to_csv(NEWS_CACHE_FILE, index=False)
        log.info(f"    뉴스 캐시 저장: {len(combined)}건 → {NEWS_CACHE_FILE}")
    except Exception:
        pass

    log.info(f"    뉴스 총 {len(combined)}건 사용")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 3.  score_sentiment()
# ─────────────────────────────────────────────────────────────────────────────

def score_sentiment(text: str) -> float:
    """유가 특화 감성 점수 (-1 ~ +1); TextBlob 혼합 가중치 적용"""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = text.lower().split()
    raw = sum(SENTIMENT_MAP.get(w, 0) for w in tokens)
    score = float(np.clip(raw / max(len(tokens) * 0.3, 1), -1, 1))
    if _TB:
        try:
            tb = TextBlob(text).sentiment.polarity
            score = 0.55 * score + 0.45 * tb
        except Exception:
            pass
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 4.  build_features()
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # HAR 구성요소 (일·주·월 실현변동성)
    'RV_1d', 'RV_5d', 'RV_21d',
    # 모멘텀
    'return_1d', 'mom_5d', 'mom_21d',
    # 외생 거시변수
    'dxy_change', 'dxy_5d', 'demand_shock', 'supply_shock',
    'geo_dummy', 'gpr_zscore',              # GPR 더미 + 연속형
    # 뉴스 (현재 + 시차 1·2)
    'news_sentiment_smooth', 'news_count',
    'news_sentiment_lag1', 'news_count_lag1',
    'news_sentiment_lag2', 'news_count_lag2',
    # 기술적 지표
    'price_vs_ma5', 'price_vs_ma21', 'bb_position',
    'return_lag1', 'return_lag2', 'RV_lag1',
    'vol_5d', 'vol_10d', 'vol_21d', 'brent_wti_spread',
    # COVID 특수 변수
    'covid_dummy',
]


def build_features(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """가격 + 뉴스 데이터를 결합하여 피처 행렬 생성"""
    log.info("[3/9] 피처 생성 중...")
    df = price_df.copy()

    # ── COVID 더미 (WHO 팬데믹 선언 ~ 백신 보급 안정화)
    df['covid_dummy'] = (
        (df.index >= COVID_START) & (df.index <= COVID_END)
    ).astype(float)
    n_covid = int(df['covid_dummy'].sum())
    if n_covid > 0:
        log.info(f"    COVID 기간 {n_covid}일 더미 설정 ({COVID_START} ~ {COVID_END})")

    # ── WTI 마이너스 가격 처리 (2020-04-20 -$37 사태)
    df['WTI'] = df['WTI'].clip(lower=1.0)

    # ── 수익률 & 로그수익률
    df['return_1d']  = df['WTI'].pct_change()
    df['log_return'] = np.log(df['WTI'] / df['WTI'].shift(1))

    # ── 극단 수익률 Winsorization (상하위 0.5% 클리핑)
    lo = df['return_1d'].quantile(0.005)
    hi = df['return_1d'].quantile(0.995)
    df['return_1d']  = df['return_1d'].clip(lo, hi)
    df['log_return'] = df['log_return'].clip(lo * 1.05, hi * 1.05)

    # ── HAR 실현변동성 구성요소
    df['RV_1d']  = df['log_return'].abs()
    df['RV_5d']  = df['log_return'].rolling(5).std()
    df['RV_21d'] = df['log_return'].rolling(21).std()

    # ── 이동평균 & 모멘텀
    for w in [5, 10, 21]:
        df[f'ma_{w}d']  = df['WTI'].rolling(w).mean()
        df[f'vol_{w}d'] = df['log_return'].rolling(w).std()
    df['mom_5d']  = df['WTI'].pct_change(5)
    df['mom_21d'] = df['WTI'].pct_change(21)

    # ── DXY 변화율
    df['dxy_change'] = df['DXY'].pct_change()
    df['dxy_5d']     = df['DXY'].pct_change(5)

    # ── Brent-WTI 스프레드
    df['brent_wti_spread'] = (df['Brent'] - df['WTI']) if 'Brent' in df.columns else 0.0

    # ── 기술적 지표
    df['price_vs_ma5']  = df['WTI'] / (df['ma_5d']  + 1e-8) - 1
    df['price_vs_ma21'] = df['WTI'] / (df['ma_21d'] + 1e-8) - 1
    bb_mid = df['WTI'].rolling(20).mean()
    bb_std = df['WTI'].rolling(20).std()
    df['bb_position'] = (df['WTI'] - bb_mid) / (2 * bb_std + 1e-8)

    # ── 뉴스 집계
    if not news_df.empty:
        news_df = news_df.copy()
        news_df['sentiment'] = news_df['title'].apply(score_sentiment)
        daily = news_df.groupby('date').agg(
            news_count=('title', 'count'),
            news_sentiment=('sentiment', 'mean')
        )
        daily.index = pd.to_datetime(daily.index)
        df = df.join(daily, how='left')
        df['news_count']    = df['news_count'].fillna(0)
        df['news_sentiment'] = df['news_sentiment'].fillna(0)
    else:
        df['news_count']    = 0
        df['news_sentiment'] = 0

    # ── gpr_zscore 보정: 뉴스가 없는 날 GPR도 ffill로 유지됨 (이미 _attach_gpr에서 처리)
    if 'gpr_zscore' not in df.columns:
        df['gpr_zscore'] = 0.0
    if 'geo_dummy' not in df.columns:
        df['geo_dummy'] = 0.0

    # 지수가중 평활 + 시차
    df['news_sentiment_smooth'] = df['news_sentiment'].ewm(span=3, min_periods=1).mean()
    for lag in [1, 2]:
        df[f'news_sentiment_lag{lag}'] = df['news_sentiment'].shift(lag)
        df[f'news_count_lag{lag}']     = df['news_count'].shift(lag)

    # ── 시차 수익률 & 변동성
    for lag in [1, 2]:
        df[f'return_lag{lag}'] = df['return_1d'].shift(lag)
    df['RV_lag1'] = df['RV_1d'].shift(1)

    # ── 훈련 타깃 (다음 날 5일 실현변동성 & 가격)
    df['target_rv']    = df['RV_5d'].shift(-1)   # 5-day rolling vol (smoother, more predictable)
    df['target_price'] = df['WTI'].shift(-1)

    # 피처 행만 dropna (타깃 NaN 포함 시 훈련용으로만 제거)
    feat_na_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_full = df.copy()               # 마지막 행 보존용 (예측에 사용)
    df.dropna(subset=feat_na_cols + ['target_rv', 'target_price'], inplace=True)

    log.info(f"    피처 완성: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, df_full


# ─────────────────────────────────────────────────────────────────────────────
# 5.  train_models()
# ─────────────────────────────────────────────────────────────────────────────

def train_models(feature_df: pd.DataFrame):
    """XGBoost-HAR (변동성) + SARIMAX (가격) 훈련 및 성능 평가"""
    log.info("[4/9] 모델 훈련 중...")

    available_feats = [c for c in FEATURE_COLS if c in feature_df.columns]

    # ── 테스트셋: 최근 60 영업일 (원샷 장기예측 오차 제거)
    n_test   = 60
    train_df = feature_df.iloc[:-n_test]
    test_df  = feature_df.iloc[-n_test:]

    X_tr = train_df[available_feats]
    X_te = test_df[available_feats]
    y_rv_tr, y_rv_te = train_df['target_rv'],    test_df['target_rv']
    y_px_tr, y_px_te = train_df['target_price'], test_df['target_price']

    results = {}
    scaler  = None

    # ─────────────────────────────────────────────────────────────────────
    # Model A: XGBoost-HAR — walk-forward TimeSeriesSplit (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    if _SKL:
        scaler = StandardScaler()

        log.info("    [A] XGBoost-HAR (5-fold walk-forward CV) 학습 중...")
        if _XGB:
            modelA = xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
                random_state=42, verbosity=0,
            )
        else:
            modelA = GradientBoostingRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.8, random_state=42,
            )

        # ── walk-forward TimeSeriesSplit 평가 (5 fold)
        tscv      = TimeSeriesSplit(n_splits=5)
        wf_preds  = np.zeros(len(X_tr))
        wf_actual = y_rv_tr.values.copy()

        full_X = scaler.fit_transform(X_tr)   # 전체 훈련 데이터로 스케일러 적합

        for fold, (idx_tr, idx_va) in enumerate(tscv.split(full_X)):
            X_f, X_v = full_X[idx_tr], full_X[idx_va]
            y_f, y_v = y_rv_tr.iloc[idx_tr], y_rv_tr.iloc[idx_va]

            covid_w = (np.where(train_df['covid_dummy'].values[idx_tr] == 1, 0.35, 1.0)
                       if 'covid_dummy' in train_df.columns else None)

            m = (xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.7,
                                   min_child_weight=5, reg_alpha=0.05,
                                   random_state=42, verbosity=0)
                 if _XGB else
                 GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                           learning_rate=0.03, subsample=0.8,
                                           random_state=42))
            m.fit(X_f, y_f, sample_weight=covid_w)
            wf_preds[idx_va] = m.predict(X_v)

        rmse_cv = float(np.sqrt(mean_squared_error(wf_actual, wf_preds)))
        mae_cv  = float(mean_absolute_error(wf_actual, wf_preds))
        r2_cv   = float(r2_score(wf_actual, wf_preds))
        log.info(f"        Walk-forward CV → RMSE={rmse_cv:.5f}  MAE={mae_cv:.5f}  R²={r2_cv:.4f}")

        # ── 최종 모델: 전체 훈련셋으로 재학습
        X_tr_s = full_X
        X_te_s = scaler.transform(X_te)
        covid_w_full = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                        if 'covid_dummy' in train_df.columns else None)
        modelA.fit(X_tr_s, y_rv_tr, sample_weight=covid_w_full)

        # 홀드아웃 테스트셋 평가 (보고용)
        pred_rv  = modelA.predict(X_te_s)
        rmse_ho  = float(np.sqrt(mean_squared_error(y_rv_te, pred_rv)))
        mae_ho   = float(mean_absolute_error(y_rv_te, pred_rv))
        r2_ho    = float(r2_score(y_rv_te, pred_rv))
        log.info(f"        Hold-out 60d → RMSE={rmse_ho:.5f}  MAE={mae_ho:.5f}  R²={r2_ho:.4f}")

        results['xgb_har'] = {
            'model': modelA, 'scaler': scaler, 'features': available_feats,
            'type': 'vol_5d',
            'rmse': rmse_cv, 'mae': mae_cv, 'r2': r2_cv,
            'rmse_ho': rmse_ho, 'r2_ho': r2_ho,
            'name': 'XGBoost-HAR (WalkFwd)' if _XGB else 'GBM-HAR (WalkFwd)',
            'pred_rv_test':    pred_rv,
            'actual_rv_test':  y_rv_te.values,
            'test_dates':      test_df.index,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Model B: SARIMAX — 1-step ahead dynamic=False 평가 (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    exog_cols = [c for c in ['dxy_change', 'news_sentiment_smooth',
                              'news_count', 'demand_shock', 'supply_shock',
                              'geo_dummy', 'gpr_zscore', 'covid_dummy']
                 if c in feature_df.columns]
    log.info("    [B] SARIMAX 학습 + 1-step ahead 평가 중...")

    if _SARIMAX and len(train_df) > 60:
        try:
            # 전체 시리즈(훈련+테스트)로 모델 구성, 파라미터는 훈련에서만 추정
            full_wti  = feature_df['WTI']
            full_exog = feature_df[exog_cols] if exog_cols else None
            n_train   = len(train_df)

            # 훈련 데이터로만 파라미터 추정
            mdl_tr = SARIMAX(
                train_df['WTI'],
                exog=train_df[exog_cols] if exog_cols else None,
                order=(2, 1, 1),
                seasonal_order=(1, 0, 1, 5),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mdl_tr.fit(disp=False, maxiter=300)

            # ── 1-step ahead 평가: 추정된 파라미터 고정 + 전체 시리즈 적용
            mdl_full = SARIMAX(
                full_wti,
                exog=full_exog,
                order=(2, 1, 1),
                seasonal_order=(1, 0, 1, 5),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # 훈련 파라미터 고정, 칼만 필터로 1-step ahead 예측
            fit_full   = mdl_full.filter(fit.params)
            pred_obj   = fit_full.get_prediction(start=n_train, dynamic=False)
            pred_price = pred_obj.predicted_mean.values[-n_test:]

            rmse_b = float(np.sqrt(mean_squared_error(y_px_te, pred_price)))
            mae_b  = float(mean_absolute_error(y_px_te, pred_price))
            r2_b   = float(r2_score(y_px_te, pred_price))

            results['sarimax'] = {
                'model': fit, 'features': exog_cols, 'type': 'price',
                'rmse': rmse_b, 'mae': mae_b, 'r2': r2_b,
                'name': 'SARIMAX(2,1,1) 1-step',
                'pred_price_test':   pred_price,
                'actual_price_test': test_df['WTI'].values,
                'test_dates':        test_df.index,
            }
            log.info(f"        1-step ahead → RMSE={rmse_b:.4f}  MAE={mae_b:.4f}  R²={r2_b:.4f}")

        except Exception as exc:
            log.warning(f"SARIMAX 실패: {exc} → Ridge 대체")
            if _SKL and scaler:
                _ridge_fallback(results, X_tr_s, y_px_tr, X_te_s, y_px_te,
                                available_feats, scaler)
    elif _SKL and scaler:
        _ridge_fallback(results, X_tr_s, y_px_tr, X_te_s, y_px_te,
                        available_feats, scaler)

    # ── 성능 저장
    perf_df = pd.DataFrame([
        {'model': v['name'], 'target': v['type'],
         'rmse': round(v['rmse'], 5), 'mae': round(v['mae'], 5), 'r2': round(v['r2'], 4)}
        for v in results.values()
    ])
    perf_df.to_csv(OUTPUT_DIR / 'model_performance.csv', index=False)
    log.info("    model_performance.csv 저장")

    return results, test_df


def _ridge_fallback(results, Xtr, ytr, Xte, yte, feats, scaler):
    r = Ridge(alpha=1.0)
    r.fit(Xtr, ytr)
    p = r.predict(Xte)
    results['ridge'] = {
        'model': r, 'scaler': scaler, 'features': feats, 'type': 'price',
        'rmse': float(np.sqrt(mean_squared_error(yte, p))),
        'mae':  float(mean_absolute_error(yte, p)),
        'r2':   float(r2_score(yte, p)),
        'name': 'Ridge Regression',
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  forecast_next_7days()
# ─────────────────────────────────────────────────────────────────────────────

def forecast_next_7days(results: dict, feature_df: pd.DataFrame, full_df: pd.DataFrame):
    """SARIMAX + XGBoost 앙상블로 향후 7 영업일 유가 예측"""
    log.info("[5/9] 7일 예측 생성 중...")

    # 마지막 실제 가격 (dropna 전 full_df 사용)
    last_price = float(full_df['WTI'].dropna().iloc[-1])
    last_date  = feature_df.index[-1]
    fc_dates   = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='B')

    forecasts = {}

    # ── SARIMAX 예측
    if 'sarimax' in results:
        try:
            sfit     = results['sarimax']['model']
            ecols    = results['sarimax']['features']
            last_exog = feature_df[ecols].tail(5).mean()
            fut_exog  = pd.DataFrame([last_exog.values] * 7, columns=ecols)
            fc_vals   = sfit.forecast(steps=7, exog=fut_exog)
            forecasts['sarimax'] = np.array(fc_vals)
            log.info(f"    SARIMAX 7일 예측: {fc_vals.values.round(2)}")
        except Exception as exc:
            log.warning(f"SARIMAX 예측 실패: {exc}")

    # ── XGBoost 예측 (재귀적 수익률 시뮬레이션)
    if 'xgb_har' in results:
        try:
            info    = results['xgb_har']
            model   = info['model']
            scaler  = info['scaler']
            feats   = info['features']
            avail_f = [f for f in feats if f in feature_df.columns]

            last_row = feature_df[avail_f].iloc[-1:].values.copy()
            last_s   = scaler.transform(last_row)

            np.random.seed(0)
            path = [last_price]
            row  = last_s.copy()
            for _ in range(7):
                rv_pred = abs(float(model.predict(row)[0]))
                rv_pred = max(rv_pred, 0.004)
                ret = np.random.normal(0, rv_pred)
                path.append(path[-1] * (1 + ret))
                row = row * 0.98   # simple decay

            forecasts['xgb'] = np.array(path[1:])
        except Exception as exc:
            log.warning(f"XGBoost 예측 실패: {exc}")

    # ── 앙상블 or 폴백
    if 'sarimax' in forecasts and 'xgb' in forecasts:
        ensemble = 0.65 * forecasts['sarimax'] + 0.35 * forecasts['xgb']
    elif 'sarimax' in forecasts:
        ensemble = forecasts['sarimax']
    elif 'xgb' in forecasts:
        ensemble = forecasts['xgb']
    else:
        trend = feature_df['WTI'].diff().tail(5).mean()
        ensemble = np.array([last_price + trend * (i + 1) for i in range(7)])

    # ── 신뢰구간 (변동성 기반)
    recent_vol = float(feature_df['vol_5d'].dropna().iloc[-1]) if 'vol_5d' in feature_df.columns else 0.015
    t = np.arange(1, 8)
    ci_half = ensemble * recent_vol * 1.96 * np.sqrt(t)

    fc_df = pd.DataFrame({
        'date':          fc_dates.strftime('%Y-%m-%d'),
        'forecast_price': np.round(ensemble, 2),
        'lower_95ci':     np.round(ensemble - ci_half, 2),
        'upper_95ci':     np.round(ensemble + ci_half, 2),
    })
    if 'sarimax' in forecasts:
        fc_df['sarimax_forecast'] = np.round(forecasts['sarimax'], 2)
    if 'xgb' in forecasts:
        fc_df['xgb_forecast'] = np.round(forecasts['xgb'], 2)

    fc_df.to_csv(OUTPUT_DIR / 'forecast_7days.csv', index=False)
    log.info("    forecast_7days.csv 저장")
    return fc_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  save_prediction_log()
# ─────────────────────────────────────────────────────────────────────────────

PRED_LOG_FILE = OUTPUT_DIR / 'prediction_log.csv'

def save_prediction_log(results: dict, feature_df: pd.DataFrame, fc_df: pd.DataFrame):
    """예측 vs 실제 오차 로그 누적 저장
    - backtest: 60일 테스트셋 (매 실행마다 재구성)
    - live: 오늘 예측 → 다음 실행 시 실제값 자동 채움
    """
    log.info("    prediction_log.csv 업데이트 중...")

    # ── 백테스트 구간 (60일 테스트셋) ────────────────────────────────
    bt_rows = []
    sx = results.get('sarimax', {})
    xg = results.get('xgb_har', {})

    if 'pred_price_test' in sx and 'test_dates' in sx:
        dates        = sx['test_dates']
        pred_prices  = sx['pred_price_test']
        actual_prices = sx['actual_price_test']
        pred_vols    = xg.get('pred_rv_test',   np.full(len(dates), np.nan))
        actual_vols  = xg.get('actual_rv_test', np.full(len(dates), np.nan))

        for i, dt in enumerate(dates):
            pp = float(pred_prices[i])
            ap = float(actual_prices[i])
            pv = float(pred_vols[i])   if i < len(pred_vols)   else np.nan
            av = float(actual_vols[i]) if i < len(actual_vols) else np.nan

            p_err     = round(ap - pp, 2)
            p_err_pct = round((ap - pp) / ap * 100, 2) if ap != 0 else None
            v_err     = round(av - pv, 5) if not (np.isnan(av) or np.isnan(pv)) else None

            bt_rows.append({
                'date':           dt.strftime('%Y-%m-%d'),
                'sarimax_pred':   round(pp, 2),
                'actual_price':   round(ap, 2),
                'price_error':    p_err,
                'price_error_pct': p_err_pct,
                'xgb_pred_vol':   round(pv, 5) if not np.isnan(pv) else None,
                'actual_vol_5d':  round(av, 5) if not np.isnan(av) else None,
                'vol_error':      v_err,
                'type':           'backtest',
            })

    bt_df = pd.DataFrame(bt_rows)

    # ── 기존 live 기록 로드 + 실제값 업데이트 ────────────────────────
    live_rows = []
    if PRED_LOG_FILE.exists():
        existing  = pd.read_csv(PRED_LOG_FILE)
        old_live  = existing[existing['type'] == 'live'].copy()

        for idx, row in old_live.iterrows():
            try:
                dt = pd.to_datetime(row['date'])
                if pd.isna(row.get('actual_price')) and dt in feature_df.index:
                    ap = float(feature_df.loc[dt, 'WTI'])
                    pp = float(row['sarimax_pred'])
                    av = float(feature_df.loc[dt, 'RV_5d']) if 'RV_5d' in feature_df.columns else np.nan
                    pv = float(row['xgb_pred_vol']) if pd.notna(row.get('xgb_pred_vol')) else np.nan

                    old_live.at[idx, 'actual_price']    = round(ap, 2)
                    old_live.at[idx, 'price_error']     = round(ap - pp, 2)
                    old_live.at[idx, 'price_error_pct'] = round((ap - pp) / ap * 100, 2)
                    if not np.isnan(av):
                        old_live.at[idx, 'actual_vol_5d'] = round(av, 5)
                    if not (np.isnan(av) or np.isnan(pv)):
                        old_live.at[idx, 'vol_error'] = round(av - pv, 5)
            except Exception:
                pass

        live_rows = old_live.to_dict('records')

    # ── 오늘 새 예측 추가 ────────────────────────────────────────────
    if fc_df is not None and len(fc_df) > 0:
        next_date   = str(fc_df['date'].iloc[0])
        next_pred_p = round(float(fc_df['forecast_price'].iloc[0]), 2)

        already = any(r.get('date') == next_date for r in live_rows)
        if not already:
            xgb_pred_v = None
            try:
                model  = xg['model']
                scaler = xg['scaler']
                feats  = xg['features']
                last   = feature_df[feats].iloc[-1:].fillna(0)
                xgb_pred_v = round(float(model.predict(scaler.transform(last))[0]), 5)
            except Exception:
                pass

            live_rows.append({
                'date':            next_date,
                'sarimax_pred':    next_pred_p,
                'actual_price':    None,
                'price_error':     None,
                'price_error_pct': None,
                'xgb_pred_vol':    xgb_pred_v,
                'actual_vol_5d':   None,
                'vol_error':       None,
                'type':            'live',
            })

    live_df = pd.DataFrame(live_rows) if live_rows else pd.DataFrame()

    # ── 합치기 & 저장 ────────────────────────────────────────────────
    combined = pd.concat([bt_df, live_df], ignore_index=True) if not live_df.empty else bt_df
    combined.to_csv(PRED_LOG_FILE, index=False)
    n_live = len(live_df)
    n_filled = int(live_df['actual_price'].notna().sum()) if not live_df.empty else 0
    log.info(f"    prediction_log.csv 저장 "
             f"(백테스트 {len(bt_df)}일 | 실시간 {n_live}건 중 {n_filled}건 확인 완료)")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  classify_risk()
# ─────────────────────────────────────────────────────────────────────────────

def classify_risk(feature_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    """실시간 리스크 신호등: 정상 / 주의 / 급등위험 / 급락위험"""
    log.info("[6/9] 리스크 분류 중...")

    row = feature_df.iloc[-1]

    vol       = float(row.get('vol_5d',               0.015))
    mom       = float(row.get('mom_5d',               0.0))
    sentiment = float(row.get('news_sentiment_smooth', 0.0))
    n_count   = float(row.get('news_count',            0.0))
    bb        = float(row.get('bb_position',           0.0))
    geo       = float(row.get('geo_dummy',             0.0))

    hist_vol_75 = float(feature_df['vol_5d'].quantile(0.75)) if 'vol_5d' in feature_df.columns else 0.022

    # 리스크 점수 계산
    vol_ratio      = vol / (hist_vol_75 + 1e-8)
    news_amp       = 1 + min(n_count / 8, 1.0)      # 뉴스 건수 증폭
    geo_amp        = 1.35 if geo > 0.5 else 1.0     # 지정학 증폭
    sentiment_amp  = 1 + max(-sentiment, 0) * 0.5   # 부정 감성 증폭

    risk_score      = vol_ratio * news_amp * geo_amp * sentiment_amp
    directional     = mom + sentiment * 0.4 + bb * 0.2

    # 분류 규칙
    if   risk_score >= 2.2 and directional >  0.025:  level = 'SURGE_RISK'
    elif risk_score >= 2.2 and directional < -0.025:  level = 'DROP_RISK'
    elif risk_score >= 1.4 or abs(directional) > 0.02: level = 'CAUTION'
    else:                                              level = 'NORMAL'

    current_wti = float(full_df['WTI'].dropna().iloc[-1])

    signal = {
        'date':              feature_df.index[-1].strftime('%Y-%m-%d'),
        'risk_level':        level,
        'risk_label':        RISK_LEVELS[level]['label'],
        'wti_price':         round(current_wti, 2),
        'volatility_5d':     round(vol, 5),
        'momentum_5d':       round(mom, 5),
        'news_sentiment':    round(sentiment, 4),
        'news_count':        int(n_count),
        'geopolitical_alert': bool(geo > 0.5),
        'risk_score':        round(risk_score, 4),
        'directional_bias':  round(directional, 5),
    }

    pd.DataFrame([signal]).to_csv(OUTPUT_DIR / 'latest_risk_signal.csv', index=False)
    r = RISK_LEVELS[level]
    log.info(f"    {r['emoji']} {level} ({r['label']})  "
             f"WTI=${current_wti:.2f}  Vol={vol:.2%}  Score={risk_score:.3f}")
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# 8.  extract_crisis_keywords()
# ─────────────────────────────────────────────────────────────────────────────

def extract_crisis_keywords(news_df: pd.DataFrame, top_n: int = 60) -> pd.DataFrame:
    """뉴스 헤드라인에서 위기 키워드 추출 (빈도 + 시드 단어 부스팅)"""
    log.info("[7/9] 위기 키워드 추출 중...")

    all_text = ' '.join(news_df['title'].fillna('').tolist()).lower()
    tokens   = re.findall(r'\b[a-z][a-z+]{2,}\b', all_text)
    filtered = [w for w in tokens if w not in STOP_WORDS]
    counts   = Counter(filtered)

    # 위기 시드 단어 1.6× 부스팅
    for w in list(counts):
        if w in CRISIS_SEED:
            counts[w] = int(counts[w] * 1.6)

    top = counts.most_common(top_n)
    kw_df = pd.DataFrame(top, columns=['keyword', 'count'])
    kw_df['is_crisis_word'] = kw_df['keyword'].isin(CRISIS_SEED)
    kw_df['weight']         = (kw_df['count'] / kw_df['count'].max()).round(4)

    kw_df.to_csv(OUTPUT_DIR / 'crisis_keywords.csv', index=False)
    log.info(f"    키워드 {len(kw_df)}개 저장 (crisis={kw_df['is_crisis_word'].sum()}개)")
    return kw_df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  generate_wordcloud()
# ─────────────────────────────────────────────────────────────────────────────

def generate_wordcloud(kw_df: pd.DataFrame):
    """위기 키워드 워드클라우드 (없으면 바 차트 대체)"""
    log.info("[8/9] 워드클라우드 생성 중...")
    crisis_set = set(kw_df[kw_df['is_crisis_word']]['keyword'])

    if _WC:
        freq = dict(zip(kw_df['keyword'], kw_df['count']))

        def color_fn(word, **_):
            if word.lower() in crisis_set:
                return f"hsl({np.random.randint(0,25)}, 90%, {np.random.randint(42,58)}%)"
            return f"hsl({np.random.randint(195,245)}, 65%, {np.random.randint(45,62)}%)"

        wc = _WC_Class(
            width=1400, height=700, background_color='#12141a',
            color_func=color_fn, max_words=80, prefer_horizontal=0.68,
            min_font_size=11, max_font_size=130, random_state=42,
        )
        wc.generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#12141a')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Oil Market Crisis Keywords  (🔴 위기어  🔵 일반어)',
                     color='#e0e0e0', fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wordcloud.png', dpi=150, bbox_inches='tight', facecolor='#12141a')
        plt.close()
    else:
        # ── 바 차트 대체
        top20 = kw_df.head(20).iloc[::-1]
        colors = ['#e74c3c' if x else '#3d85c8' for x in top20['is_crisis_word']]

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#12141a')
        ax.set_facecolor('#1a1d24')
        ax.barh(top20['keyword'], top20['count'], color=colors)
        ax.set_xlabel('Frequency', color='#ccc')
        ax.set_title('Top Crisis Keywords (워드클라우드 대체 바 차트)', color='white', fontsize=13)
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_color('#333')
        p1 = mpatches.Patch(color='#e74c3c', label='Crisis Keyword')
        p2 = mpatches.Patch(color='#3d85c8', label='General Keyword')
        ax.legend(handles=[p1, p2], facecolor='#1a1d24', labelcolor='white')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wordcloud.png', dpi=150, bbox_inches='tight', facecolor='#12141a')
        plt.close()

    log.info("    wordcloud.png 저장")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  plot_oil_forecast()  — 통합 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_oil_forecast(feature_df: pd.DataFrame, fc_df: pd.DataFrame, signal: dict):
    """6-패널 대시보드 차트 생성"""
    log.info("[9/9] 차트 생성 중...")

    BG   = '#0d1117'
    PAN  = '#161b22'
    TXT  = '#e6edf3'
    GOLD = '#f0c040'
    CYAN = '#58a6ff'
    RED  = '#ff6b6b'
    GRN  = '#3fb950'

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32,
                            left=0.05, right=0.97, top=0.93, bottom=0.05)

    def style_ax(ax, title=''):
        ax.set_facecolor(PAN)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.grid(color='#21262d', linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=TXT, fontsize=10, pad=6, fontweight='bold')

    N = 180   # 차트에 표시할 과거 일수

    # ── Panel 1 (top, full width): Price + Forecast ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, 'WTI Crude Oil  —  Historical Price & 7-Day Forecast')

    hist = feature_df['WTI'].iloc[-N:]
    ax1.plot(hist.index, hist, color=GOLD, lw=1.5, label='WTI (historical)', zorder=3)

    if 'ma_21d' in feature_df.columns:
        ma = feature_df['ma_21d'].iloc[-N:]
        ax1.plot(ma.index, ma, color='#8b949e', lw=0.9, ls='--', alpha=0.7, label='MA-21')

    fd = pd.to_datetime(fc_df['date'])
    ax1.plot(fd, fc_df['forecast_price'], color=CYAN, lw=2, ls='--', label='Ensemble Forecast', zorder=4)
    if 'sarimax_forecast' in fc_df.columns:
        ax1.plot(fd, fc_df['sarimax_forecast'], color='#a371f7', lw=1, ls=':', alpha=0.8, label='SARIMAX')
    if 'xgb_forecast' in fc_df.columns:
        ax1.plot(fd, fc_df['xgb_forecast'], color='#ffa657', lw=1, ls=':', alpha=0.8, label='XGBoost')

    ax1.fill_between(fd, fc_df['lower_95ci'], fc_df['upper_95ci'],
                     alpha=0.13, color=CYAN, label='95% CI')

    rlevel = signal['risk_level']
    rcol   = RISK_LEVELS[rlevel]['color']
    ax1.axvspan(fd.iloc[0], fd.iloc[-1], alpha=0.07, color=rcol)
    ax1.annotate(f"{RISK_LEVELS[rlevel]['emoji']} {RISK_LEVELS[rlevel]['label']}",
                 xy=(fd.iloc[3], fc_df['forecast_price'].median()),
                 fontsize=10, color=rcol, fontweight='bold', ha='center')

    ax1.set_ylabel('USD / bbl', color=TXT, fontsize=9)
    ax1.legend(loc='upper left', facecolor=PAN, labelcolor=TXT, framealpha=0.6,
               fontsize=8, ncol=3)

    # ── Panel 2 (row2 col0): 5-Day Realized Volatility ───────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, 'Realized Volatility (5-day)')
    if 'vol_5d' in feature_df.columns:
        vdata = feature_df['vol_5d'].iloc[-N:] * 100
        ax2.plot(vdata.index, vdata, color=RED, lw=1.1)
        ax2.fill_between(vdata.index, 0, vdata, alpha=0.18, color=RED)
        thr = float(vdata.quantile(0.75))
        ax2.axhline(thr, color='#f39c12', lw=0.9, ls='--',
                    label=f'75th pct {thr:.2f}%')
        ax2.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)
    ax2.set_ylabel('%', color=TXT, fontsize=9)

    # ── Panel 3 (row2 col1): News Sentiment ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, 'News Sentiment (smoothed)')
    if 'news_sentiment_smooth' in feature_df.columns:
        sent = feature_df['news_sentiment_smooth'].iloc[-N:]
        cols = [GRN if s >= 0 else RED for s in sent]
        ax3.bar(sent.index, sent, color=cols, alpha=0.75, width=1.2)
        ax3.axhline(0, color='#8b949e', lw=0.7, ls='--')

    # ── Panel 4 (row2 col2): News Volume ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, 'Daily News Count')
    if 'news_count' in feature_df.columns:
        nc = feature_df['news_count'].iloc[-N:]
        ax4.bar(nc.index, nc, color=CYAN, alpha=0.65, width=1.2)
    ax4.set_ylabel('Articles / day', color=TXT, fontsize=9)

    # ── Panel 5 (row3 col0-1): Risk Signal Gauge ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(PAN)
    ax5.axis('off')
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.set_title('Current Risk Signal', color=TXT, fontsize=10, pad=6, fontweight='bold')

    order  = ['NORMAL', 'CAUTION', 'SURGE_RISK', 'DROP_RISK']
    labels = ['🟢 정상', '🟡 주의', '🔴 급등위험', '🔵 급락위험']
    xpos   = [0.06, 0.30, 0.54, 0.78]

    for i, (rl, lbl, xp) in enumerate(zip(order, labels, xpos)):
        is_cur = (rl == rlevel)
        rc = RISK_LEVELS[rl]['color']
        alpha = 0.95 if is_cur else 0.22
        rect = FancyBboxPatch((xp, 0.28), 0.20, 0.44,
                              boxstyle='round,pad=0.02',
                              facecolor=rc, alpha=alpha,
                              edgecolor='white',
                              linewidth=2.5 if is_cur else 0.5)
        ax5.add_patch(rect)
        ax5.text(xp + 0.10, 0.50, lbl, ha='center', va='center',
                 color='white', fontsize=9.5,
                 fontweight='bold' if is_cur else 'normal')
        if is_cur:
            ax5.text(xp + 0.10, 0.78, '▲ Current', ha='center',
                     color=rc, fontsize=8.5, fontweight='bold')

    details = (f"Risk Score: {signal['risk_score']:.3f}  │  "
               f"WTI: ${signal['wti_price']:.2f}  │  "
               f"Vol: {signal['volatility_5d']*100:.2f}%  │  "
               f"Mom(5d): {signal['momentum_5d']*100:+.2f}%  │  "
               f"Sentiment: {signal['news_sentiment']:+.3f}  │  "
               f"News: {signal['news_count']} │  "
               f"Geo: {'⚠' if signal['geopolitical_alert'] else '—'}")
    ax5.text(0.50, 0.10, details, ha='center', va='center',
             color='#8b949e', fontsize=8)

    # ── Panel 6 (row3 col2): Model Performance Table ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PAN)
    ax6.axis('off')
    ax6.set_title('Model Performance', color=TXT, fontsize=10, pad=6, fontweight='bold')
    try:
        pf = pd.read_csv(OUTPUT_DIR / 'model_performance.csv')
        tdata = [[r['model'], r['target'], f"{r['rmse']:.4f}",
                  f"{r['mae']:.4f}", f"{r['r2']:.3f}"] for _, r in pf.iterrows()]
        tbl = ax6.table(cellText=tdata, colLabels=['Model','Target','RMSE','MAE','R²'],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#21262d' if r == 0 else PAN)
            cell.set_text_props(color=TXT)
            cell.set_edgecolor('#30363d')
    except Exception:
        ax6.text(0.5, 0.5, 'No performance data', ha='center', va='center', color=TXT)

    # ── Super-title
    fig.suptitle(
        f'🛢  국제 유가 리스크 예측 시스템  │  {datetime.today().strftime("%Y-%m-%d %H:%M")}',
        color='white', fontsize=15, fontweight='bold', y=0.975
    )

    plt.savefig(OUTPUT_DIR / 'oil_forecast_plot.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    plt.close()
    log.info("    oil_forecast_plot.png 저장")


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline()  —  전체 파이프라인 오케스트레이터
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(start_date=None, end_date=None) -> dict:
    """전체 분석 파이프라인 실행 후 결과 dict 반환"""
    print("\n" + "=" * 65)
    print("  🛢  국제 유가 리스크 예측 시스템  MVP")
    print("=" * 65)

    price_df             = fetch_data(start_date, end_date)
    news_df              = fetch_news()   # 기본값: DATA_YEARS × 365일
    feature_df, full_df  = build_features(price_df, news_df)
    model_results, _     = train_models(feature_df)
    fc_df                = forecast_next_7days(model_results, feature_df, full_df)
    save_prediction_log(model_results, feature_df, fc_df)
    risk_signal          = classify_risk(feature_df, full_df)
    kw_df                = extract_crisis_keywords(news_df)
    generate_wordcloud(kw_df)
    plot_oil_forecast(feature_df, fc_df, risk_signal)

    # ── 결과 요약 출력
    rl = risk_signal['risk_level']
    r  = RISK_LEVELS[rl]
    print("\n" + "─" * 65)
    print(f"  리스크 신호 : {r['emoji']} {rl} ({r['label']})")
    print(f"  WTI 현재가  : ${risk_signal['wti_price']:.2f} / bbl")
    print(f"  5일 변동성  : {risk_signal['volatility_5d']*100:.2f}%")
    print(f"  뉴스 감성   : {risk_signal['news_sentiment']:+.4f}")
    print(f"  지정학 경보 : {'활성 ⚠' if risk_signal['geopolitical_alert'] else '없음 —'}")
    print(f"  리스크 점수 : {risk_signal['risk_score']:.4f}")
    print("\n  📁 저장된 파일:")
    for fname in ['model_performance.csv', 'forecast_7days.csv', 'latest_risk_signal.csv',
                  'crisis_keywords.csv', 'oil_forecast_plot.png', 'wordcloud.png']:
        p = OUTPUT_DIR / fname
        mark = "✓" if p.exists() else "✗"
        print(f"    {mark}  output/{fname}")
    print("─" * 65 + "\n")

    return {
        'risk_signal':    risk_signal,
        'forecast':       fc_df,
        'features':       feature_df,
        'models':         model_results,
        'keywords':       kw_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_pipeline()
