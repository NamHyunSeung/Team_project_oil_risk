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
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
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

# ── API 키 & 파일 경로 설정 (.env 파일 또는 환경변수에서 로드)
FRED_API_KEY     = os.getenv("FRED_API_KEY",     "0a1d6c8b56c44eff8716c204f0aa49bf")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "3a287cda-6e49-49f0-8998-3092657e209e")
EIA_API_KEY      = os.getenv("EIA_API_KEY",      "")
GPR_FILE         = "data_gpr_daily_recent.xls"   # 프로젝트 폴더에 위치
DATA_YEARS       = 10                             # 데이터 수집 기간 (XGBoost 학습용)
SARIMAX_YEARS    = 5                              # SARIMAX 학습 기간 (최근 가격 패턴 집중)

# ── 이메일 알림 설정 (.env 또는 환경변수)
# Gmail 사용 시: Google 계정 → 보안 → 앱 비밀번호 생성 후 SMTP_PASSWORD에 입력
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "")   # 발신 Gmail 주소
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")   # Gmail 앱 비밀번호 (16자리)
ALERT_TO      = os.getenv("ALERT_TO",      "")   # 수신 이메일 주소

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
    from pmdarima import auto_arima as _auto_arima
    _PMDARIMA = True
except ImportError:
    _PMDARIMA = False
    log.warning("pmdarima 없음 → 기본 SARIMAX 파라미터 사용")

try:
    from prophet import Prophet as _Prophet
    _PROPHET = True
except ImportError:
    _PROPHET = False
    log.warning("prophet 없음 → 2모델 앙상블 유지")

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
    # 관사/전치사/접속사
    'the','a','an','in','on','at','to','for','of','and','or','but','is',
    'are','was','were','be','been','has','have','had','will','would','could',
    'should','may','might','as','with','by','from','up','about','into',
    'through','before','after','that','this','these','those','it','its',
    # 대명사/의문사/부사
    'not','no','said','says','amid','their','they','our','we','us','you',
    'your','him','her','his','she','he','who','when','where','how','what',
    'why','which','than','more','most','very','also','just','even','still',
    # 일반 동사/형용사 (유가 무관)
    'make','made','say','get','got','take','took','come','came','know',
    'think','want','need','see','look','seem','give','put','use','find',
    'tell','ask','work','feel','try','leave','call','keep','let','mean',
    'first','last','next','one','two','three','all','both','can','will',
    'back','off','out','over','down','up','right','good','best','big',
    'old','long','little','own','same','other','new','high','low','day',
    'year','years','time','way','part','well','much','many','now','then',
    'here','there','where','ever','never','always','often','again','away',
    # 뉴스/미디어 관련 노이즈
    'happened','review','reviews','recipe','recipes','briefing','briefings',
    'podcast','newsletter','today','yesterday','week','month','morning',
    'guardian','report','reports','reporting','interview','exclusive',
    'live','update','updates','latest','breaking','watch','listen','read',
    'comment','opinion','analysis','explainer','guide','everything','shows',
    # 고유명사 노이즈 (인물명 등)
    'rachel','roddy','keir','starmer',
    # 무의미 명사
    'thing','things','stop','obituary','obituaries','life','lives',
    # 의미 없는 일반어
    'against','life','must','view','people','things','something','anything',
    'place','area','area','puts','set','sets','taking','being','doing',
    'goes','going','come','coming','getting','including','among','within',
    'across','around','between','while','since','until','unless','although',
}

# 4번: 뉴스 중요도 가중치 — 핵심 산유국/기관 언급 기사에 1.5× 가중치
HIGH_IMPACT_ENTITIES = {
    'opec', 'saudi', 'russia', 'iran', 'china', 'uae', 'iea',
    'venezuela', 'iraq', 'nigeria', 'libya', 'kuwait', 'qatar',
    'aramco', 'rosneft', 'kremlin', 'brics',
}

# 소스별 신뢰도 가중치 (EIA 공식 > Reuters 금융 > 에너지 전문 > 일반)
SOURCE_WEIGHTS = {
    'EIA':        2.0,
    'Reuters':    1.5,
    'OilPrice':   1.3,
    'Rigzone':    1.2,
    'Guardian':   1.2,
    'MarketWatch':1.1,
    'RSS':        1.0,
    'dummy':      0.3,
}

NEGATION_WORDS = {
    'not','no','never','without','halt','stop','end','cease',
    'avoid','prevent','block','reverse','reject','deny','fail',
}

SENTIMENT_MAP = {
    # 강한 부정 (-2)
    'war':-2,'attack':-2,'explosion':-2,'collapse':-2,'crash':-2,
    'crisis':-2,'shortage':-2,'embargo':-2,'sanction':-2,'shutdown':-2,
    'disruption':-2,'conflict':-2,'seizure':-2,'blockade':-2,
    # 중간 부정 (-1.5)
    'spike':-1.5,'plunge':-1.5,'slump':-1.5,'plummet':-1.5,
    'halt':-1.5,'freeze':-1.5,'restrict':-1.5,'glut':-1.5,
    # 약한 부정 (-1)
    'surge':-1,'cut':-1,'fall':-1,'decline':-1,'risk':-1,
    'concern':-1,'fear':-1,'threat':-1,'tension':-1,'dispute':-1,
    'recession':-1,'downgrade':-1,'weak':-1,'bearish':-1,
    'tumble':-1,'slip':-1,'drop':-1,'loss':-1,'warning':-1,
    'contango':-1,'oversupply':-1,'hawkish':-0.5,'tightening':-0.5,
    'withdrawal':-1,'evacuation':-0.5,'slowdown':-1,'shrink':-1,
    # 약한 긍정 (+1)
    'recovery':1,'growth':1,'increase':1,'rise':1,'deal':1,
    'agreement':1,'stable':1,'ease':1,'lift':1,'resume':1,
    'open':1,'rally':1,'rebound':1,'strong':1,'bullish':1,
    'draw':1,'deficit':1,'backwardation':1,'dovish':0.5,
    'ceasefire':1.5,'truce':1.5,'compliance':1,
    # 강한 긍정 (+2)
    'boom':2,'peace':2,'resolution':2,'surplus':2,'record':1.5,
}

# 구문 패턴 (바이그램) — 단어 단독보다 맥락이 중요한 표현
PHRASE_SENTIMENT = {
    'production cut':-2,'output cut':-2,'supply cut':-2,'capacity cut':-1.5,
    'deeper cut':-2,'extend cut':-1.5,'voluntary cut':-1.5,
    'supply disruption':-2,'supply shortage':-2,'supply crunch':-2,
    'demand destruction':-1.5,'demand weakness':-1.5,'demand slowdown':-1,
    'refinery shutdown':-1.5,'pipeline attack':-2,'pipeline shutdown':-1.5,
    'inventory draw':1.5,'stock draw':1.5,'crude draw':1.5,
    'inventory build':-1,'stock build':-1,'crude build':-1,
    'production increase':1.5,'output increase':1.5,'supply increase':1,
    'production boost':1.5,'output boost':1.5,
    'price cap':-1,'price ceiling':-1,'strategic release':-1,
    'nuclear deal':1,'sanctions lifted':2,'sanctions eased':1.5,
    'ceasefire deal':1.5,'peace deal':2,
    'demand recovery':1.5,'demand growth':1.5,'demand surge':1.5,
}

# 강화어 (인접 단어의 감성 1.4× 증폭)
INTENSIFIERS = {
    'record','massive','unprecedented','sharp','dramatic','significant',
    'major','severe','huge','deep','steep','rapidly','sharply','surging',
}

NEWS_RSS = [
    # 종합 경제/비즈니스
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/companyNews",
    "https://www.marketwatch.com/rss/topstories",
    "https://rss.cnn.com/rss/money_news_international.rss",
    # 에너지 전문
    "https://www.oilprice.com/rss/main",
    "https://www.eia.gov/rss/press_releases.xml",
    "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
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
        'VIX':         20.0  + np.cumsum(np.random.normal(0, 0.8, n)).clip(-15, 40),
        'OVX':         35.0  + np.cumsum(np.random.normal(0, 1.2, n)).clip(10, 100),
        'demand_shock': np.random.normal(0, 2.0, n),
        'supply_shock': np.random.normal(0, 1.5, n),
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
        df['demand_shock']   = np.random.normal(0, 2.0, n)
        df['supply_shock']   = np.random.normal(0, 1.5, n)
        df['inv_chg_zscore'] = np.random.normal(0, 1.0, n)
        df['inv_lvl_zscore'] = np.random.normal(0, 1.0, n)
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

    # ── 미국 원유 재고 (EIA API v2: WCESTUS1, SPR 제외 상업 재고) ───────────
    try:
        if not EIA_API_KEY:
            raise ValueError("EIA_API_KEY 미설정")
        import urllib.request as _ur, json as _json, urllib.parse as _up

        log.info("    EIA 수집: 미국 원유 재고 (WCESTUS1)...")
        params = _up.urlencode({
            'api_key':               EIA_API_KEY,
            'frequency':             'weekly',
            'data[0]':               'value',
            'facets[duoarea][]':     'NUS',
            'facets[product][]':     'EPC0',
            'facets[process][]':     'SAX',   # SPR 제외 상업 재고
            'start':                 start_date[:7],
            'end':                   end_date[:7],
            'sort[0][column]':       'period',
            'sort[0][direction]':    'asc',
            'length':                5000,
        })
        url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?{params}"
        with _ur.urlopen(url, timeout=20) as r:
            rows = _json.loads(r.read())['response']['data']

        inv_records = {row['period']: float(row['value'])
                       for row in rows if row['value'] is not None}
        inv = pd.Series(inv_records)
        inv.index = pd.to_datetime(inv.index)
        inv = inv.sort_index()

        # 주간 변화량 (천 배럴, 음수=감소=강세)
        inv_chg = inv.diff()

        # 주간 → 영업일 ffill
        inv_chg_bday   = inv_chg.resample('B').first().ffill()
        inv_level_bday = inv.resample('B').first().ffill()

        # z-score 정규화
        def _zscore(s, w=252):
            return ((s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-8)).fillna(0)

        df['inv_chg_zscore'] = _zscore(inv_chg_bday).reindex(df.index).ffill().bfill().fillna(0)
        df['inv_lvl_zscore'] = _zscore(inv_level_bday).reindex(df.index).ffill().bfill().fillna(0)
        log.info(f"      원유 재고 연결 완료: {len(inv)}주치 "
                 f"(변화 z-score μ={df['inv_chg_zscore'].mean():.3f})")
    except Exception as exc:
        log.warning(f"      원유 재고 수집 실패({exc}) → 0 사용")
        df['inv_chg_zscore'] = 0.0
        df['inv_lvl_zscore'] = 0.0

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
        def _dl(ticker, col='Close'):
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(raw, pd.DataFrame):
                if col in raw.columns:
                    s = raw[col]
                else:
                    s = raw.iloc[:, 0]
                if isinstance(s, pd.DataFrame):   # MultiIndex ticker残留
                    s = s.iloc[:, 0]
            else:
                s = raw
            s.name = ticker
            return s

        wti   = _dl("CL=F")
        brent = _dl("BZ=F")
        dxy   = _dl("DX-Y.NYB")
        try:
            vix = _dl("^VIX")
            log.info("    VIX(공포지수) 수집 완료")
        except Exception:
            vix = pd.Series(dtype=float, name="^VIX")

        try:
            ovx = _dl("^OVX")
            log.info("    OVX(원유 변동성 지수) 수집 완료")
        except Exception:
            ovx = pd.Series(dtype=float, name="^OVX")

        try:
            cl2 = _dl("CL2=F")
            futures_spread = (cl2 - wti).rename("futures_spread")
            log.info(f"    WTI 선물 커브 스프레드 수집 완료 (μ={futures_spread.mean():.3f})")
        except Exception:
            futures_spread = pd.Series(dtype=float, name="futures_spread")
            log.warning("    WTI 2번째 월물 수집 실패 → futures_spread=0")

        # Parkinson 추정을 위한 WTI High/Low 수집
        try:
            wti_high = _dl("CL=F", col='High').rename("WTI_High")
            wti_low  = _dl("CL=F", col='Low').rename("WTI_Low")
            log.info("    WTI High/Low 수집 완료 (Parkinson 추정용)")
        except Exception:
            wti_high = pd.Series(dtype=float, name="WTI_High")
            wti_low  = pd.Series(dtype=float, name="WTI_Low")

        # VIX 기간구조 + SKEW (파생상품 꼬리위험)
        try:
            vix3m = _dl("^VIX3M")
            log.info("    VIX3M(3개월 변동성) 수집 완료")
        except Exception:
            vix3m = pd.Series(dtype=float, name="^VIX3M")
        try:
            skew = _dl("^SKEW")
            log.info("    SKEW(꼬리위험) 수집 완료")
        except Exception:
            skew = pd.Series(dtype=float, name="^SKEW")

        df = pd.DataFrame({'WTI': wti, 'Brent': brent, 'DXY': dxy,
                           'VIX': vix, 'OVX': ovx, 'futures_spread': futures_spread,
                           'WTI_High': wti_high, 'WTI_Low': wti_low,
                           'VIX3M': vix3m, 'SKEW': skew})
        df = df.ffill().bfill()
        df.dropna(subset=['WTI'], inplace=True)

        if len(df) < 60:
            log.warning("데이터 부족 → 더미 데이터 사용")
            return _dummy_prices(start_date, end_date)

        # ── FRED 실제 데이터 연결 ──────────────────────────────────────────
        df = _attach_fred_data(df, start_date, end_date)

        # ── FRED WTI로 yfinance 이상값 패치 ───────────────────────────────
        df = _patch_wti_with_fred(df, start_date, end_date)

        log.info(f"    yfinance 성공: {len(df):,} rows")
        return df

    except Exception as e:
        log.warning(f"yfinance 오류({e}) → 더미 데이터 사용")
        return _dummy_prices(start_date, end_date)


def _patch_price_with_fred(df: pd.DataFrame, col: str, fred_series: str,
                           start_date: str, end_date: str,
                           fred_client, threshold: float = 0.10) -> pd.DataFrame:
    """FRED 공식 데이터로 yfinance 컬럼 이상값 감지·교체 (내부 공통 함수).

    threshold 비율 이상 괴리 시 FRED 값으로 교체. 오늘 날짜는 FRED 지연으로 제외.
    """
    try:
        fred_data = fred_client.get_series(fred_series,
                                           observation_start=start_date,
                                           observation_end=end_date)
        fred_data = fred_data.dropna()
        fred_data.index = pd.to_datetime(fred_data.index).normalize()

        if len(fred_data) < 30 or col not in df.columns:
            return df

        today     = pd.Timestamp(datetime.today().date())
        check_idx = df.index.intersection(fred_data.index)
        check_idx = check_idx[check_idx < today]

        if len(check_idx) == 0:
            return df

        yf_vals   = df.loc[check_idx, col]
        fred_vals = fred_data.loc[check_idx]
        diff_pct  = ((yf_vals - fred_vals) / fred_vals).abs()
        anomalies = diff_pct[diff_pct > threshold]

        if anomalies.empty:
            log.info(f"    {col} 이상값 없음 (FRED {fred_series} 검증 통과: {len(check_idx)}일)")
            return df

        log.warning(f"    ⚠ {col} 이상값 {len(anomalies)}건 → FRED {fred_series} 값으로 교체:")
        for dt, pct in anomalies.items():
            yf_v, fred_v = float(yf_vals[dt]), float(fred_vals[dt])
            log.warning(f"      {dt.date()}: yfinance={yf_v:.2f} → FRED={fred_v:.2f} (괴리 {pct*100:.1f}%)")
            df.loc[dt, col] = fred_v

        return df
    except Exception as exc:
        log.warning(f"    FRED {fred_series} 검증 실패({exc}) → yfinance 원본 유지")
        return df


def _patch_wti_with_fred(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """FRED로 WTI(DCOILWTICO)·Brent(DCOILBRENTEU) 이상값 패치."""
    if not _FRED or not FRED_API_KEY:
        return df
    try:
        fred = _Fred(api_key=FRED_API_KEY)
        df = _patch_price_with_fred(df, 'WTI',   'DCOILWTICO',   start_date, end_date, fred)
        df = _patch_price_with_fred(df, 'Brent',  'DCOILBRENTEU', start_date, end_date, fred)
    except Exception as exc:
        log.warning(f"    FRED 패치 초기화 실패({exc}) → yfinance 원본 유지")
    return df


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
            'show-fields':'headline,bodyText',
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
            fields   = item.get('fields') or {}
            headline = fields.get('headline') or item.get('webTitle', '')
            body_excerpt = (fields.get('bodyText') or '')[:300]
            # 제목 + 본문 앞 300자 결합 → 감성 분석 품질 향상
            full_text = headline + (' ' + body_excerpt if body_excerpt else '')
            articles.append({
                'date':   item['webPublicationDate'][:10],
                'title':  full_text,
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
        _RSS_DOMAIN_MAP = {
            'reuters.com':    'Reuters',
            'eia.gov':        'EIA',
            'oilprice.com':   'OilPrice',
            'rigzone.com':    'Rigzone',
            'marketwatch.com':'MarketWatch',
        }
        for url in NEWS_RSS:
            src_name = next((v for k, v in _RSS_DOMAIN_MAP.items() if k in url), 'RSS')
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
                                             'title': title, 'source': src_name})
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
    """유가 특화 감성 점수 (-1 ~ +1); 구문 패턴 + 강화어 + 부정어 반전 + TextBlob"""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower = text.lower()
    tokens = text_lower.split()
    raw = 0.0

    # 바이그램 구문 매칭 (단어 단독보다 높은 우선순위)
    matched_idx = set()
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in PHRASE_SENTIMENT:
            raw += PHRASE_SENTIMENT[bg]
            matched_idx |= {i, i + 1}

    # 단어 매칭 (구문에 이미 포함된 토큰은 건너뜀)
    for i, w in enumerate(tokens):
        if i in matched_idx:
            continue
        base = SENTIMENT_MAP.get(w, 0)
        if base != 0:
            context = tokens[max(0, i - 5):i]
            if any(neg in context for neg in NEGATION_WORDS):
                base *= -1
            if any(amp in context for amp in INTENSIFIERS):
                base *= 1.4
        raw += base

    score = float(np.clip(raw / max(len(tokens) * 0.3, 1), -1, 1))
    if _TB:
        try:
            tb = TextBlob(text).sentiment.polarity
            # TextBlob 비중 축소: 일반 코퍼스 기반으로 유가 뉴스에 부정확
            score = 0.75 * score + 0.25 * tb
        except Exception:
            pass
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 4.  build_features()
# ─────────────────────────────────────────────────────────────────────────────

# ── HAR 전용 피처셋 (변동성 예측 특화, ~25개, regime/뉴스/거시 제외)
# 과적합 원인: 96개 피처 중 regime(48%) 단독 지배 → 훈련R²=0.80 vs CV R²=0.36
HAR_FEATURE_COLS = [
    # 핵심 HAR 성분
    'RV_1d', 'RV_5d', 'RV_21d', 'RV_63d',
    # GARCH / Parkinson / EWMA
    'garch_vol', 'parkinson_vol', 'parkinson_vol_5d', 'parkinson_vol_21d',
    'ewma_vol_10', 'ewma_vol_21', 'ewma_vol_63',
    # 변동성 모멘텀
    'rv_term_slope', 'rv_5d_chg', 'rv_mom_5_21',
    # 레버리지 효과
    'leverage_effect', 'neg_return',
    # EIA 발표일 효과
    'dow_wednesday', 'dow_thursday', 'eia_vol_signal',
    # 장중 RV (정확한 측정)
    'rv_intraday', 'rv_intraday_5d',
    # 파생상품 (옵션 내재변동성)
    'ovx_zscore', 'ovx_change', 'ovx_rv_spread',
    'vix_change', 'vix_zscore',
    'vix_term_slope', 'vix_ts_zscore',
    'skew_zscore',
    # Brent 스필오버
    'brent_rv_1d_lag1', 'brent_rv_5d_lag1',
    # 기존 lag
    'RV_lag1',
]

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
    'news_sentiment_smooth7', 'sentiment_magnitude',
    'extreme_neg_news', 'news_count_pos', 'news_count_neg',
    # 기술적 지표
    'price_vs_ma5', 'price_vs_ma21', 'bb_position',
    'return_lag1', 'return_lag2', 'RV_lag1',
    'vol_5d', 'vol_10d', 'vol_21d', 'brent_wti_spread',
    # 추가 기술적 지표 (RSI, MACD, ATR, 가격 z-score)
    'rsi_14', 'macd', 'macd_signal', 'atr_14', 'price_zscore',
    # VIX 기반 피처
    'vix_zscore', 'vix_change',
    # VIX × 뉴스 감성 복합변수
    'fear_composite', 'vix_amplified', 'vix_sent_diverge',
    # OVX (원유 변동성 지수) 피처
    'ovx_zscore', 'ovx_change', 'ovx_rv_spread',
    # WTI 선물 커브 (contango/backwardation)
    'futures_spread', 'futures_spread_chg', 'contango_dummy',
    # EIA 미국 원유 재고 (실물 수급 지표)
    'inv_chg_zscore', 'inv_lvl_zscore',
    # HAR 장기 성분 + 레버리지 효과 + EIA 요일 효과
    'RV_63d', 'neg_return', 'return_neg', 'return_pos',
    'leverage_effect', 'dow_wednesday', 'dow_thursday', 'dow_monday',
    'eia_vol_signal',
    # A: GARCH 조건부 분산
    'garch_vol',
    # B: Parkinson 장중 범위 추정
    'parkinson_vol', 'parkinson_vol_5d', 'parkinson_vol_21d',
    # C: EWMA 변동성
    'ewma_vol_10', 'ewma_vol_21', 'ewma_vol_63',
    # D: 변동성 모멘텀
    'rv_term_slope', 'rv_5d_chg', 'rv_mom_5_21',
    # 장중 고빈도 실현분산 (1h)
    'rv_intraday', 'rv_intraday_5d', 'rv_intraday_21d', 'rv_intra_vs_close',
    # VIX 기간구조 + SKEW
    'vix_term_slope', 'vix_ts_zscore', 'skew_zscore', 'skew_chg',
    # 5번: 시장 국면(Regime) 피처
    'regime', 'regime_x_mom', 'regime_x_sent', 'regime_x_gpr',
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

    # ── HAR 실현변동성 구성요소 (일·주·월·분기)
    df['RV_1d']  = df['log_return'].abs()
    df['RV_5d']  = df['log_return'].rolling(5).std()
    df['RV_21d'] = df['log_return'].rolling(21).std()
    df['RV_63d'] = df['log_return'].rolling(63).std()   # 분기 변동성 (레짐 포착)

    # ── B: Parkinson 장중 범위 추정 (High-Low 기반, 종가보다 4~5배 효율적)
    if 'WTI_High' in df.columns and 'WTI_Low' in df.columns:
        _h = df['WTI_High'].replace(0, np.nan)
        _l = df['WTI_Low'].replace(0, np.nan).clip(lower=0.01)
        _hl_ratio = (_h / _l).clip(lower=1.0)
        df['parkinson_vol'] = (np.log(_hl_ratio) ** 2 / (4 * np.log(2))).apply(np.sqrt).fillna(0)
        df['parkinson_vol_5d']  = df['parkinson_vol'].rolling(5).mean().fillna(0)
        df['parkinson_vol_21d'] = df['parkinson_vol'].rolling(21).mean().fillna(0)
        log.info(f"    Parkinson vol 생성 (μ={df['parkinson_vol'].mean():.5f})")
    else:
        df['parkinson_vol']     = 0.0
        df['parkinson_vol_5d']  = 0.0
        df['parkinson_vol_21d'] = 0.0

    # ── C: EWMA 변동성 (RiskMetrics λ=0.94)
    df['ewma_vol_10']  = df['log_return'].ewm(span=10,  adjust=False).std().fillna(0)
    df['ewma_vol_21']  = df['log_return'].ewm(span=21,  adjust=False).std().fillna(0)
    df['ewma_vol_63']  = df['log_return'].ewm(span=63,  adjust=False).std().fillna(0)

    # ── D: 변동성 모멘텀 (term structure + 변화량)
    df['rv_term_slope']  = (df['RV_5d'] / df['RV_21d'].replace(0, np.nan)).fillna(1.0)  # >1: 단기>장기(변동성 상승 중)
    df['rv_5d_chg']      = df['RV_5d'].diff().fillna(0)   # 변동성 변화량
    df['rv_mom_5_21']    = (df['RV_5d'] - df['RV_21d']).fillna(0)  # 단기-장기 스프레드

    # ── A: GARCH(1,1) 조건부 분산 (변동성 클러스터링 명시적 모델링)
    try:
        from arch import arch_model as _arch_model
        _ret_pct = df['log_return'].dropna() * 100   # % 스케일
        _garch   = _arch_model(_ret_pct, vol='Garch', p=1, q=1,
                               dist='Normal', rescale=False)
        _res     = _garch.fit(disp='off', show_warning=False)
        _cond_vol = _res.conditional_volatility / 100   # 소수점 스케일 복원
        df['garch_vol'] = _cond_vol.reindex(df.index).ffill().bfill().fillna(0)
        log.info(f"    GARCH(1,1) 조건부 분산 생성 (μ={df['garch_vol'].mean():.5f})")
    except Exception as _ge:
        log.warning(f"    GARCH 실패({_ge}) → garch_vol=0")
        df['garch_vol'] = 0.0

    # ── 장중 고빈도 실현분산 (1h 데이터, 최근 730일 — 더 정확한 RV 추정)
    try:
        _raw_1h = yf.download("CL=F", period="730d", interval="1h",
                              progress=False, auto_adjust=True)
        if not _raw_1h.empty and len(_raw_1h) > 100:
            _c1h = _raw_1h['Close']
            if isinstance(_c1h, pd.DataFrame): _c1h = _c1h.iloc[:, 0]
            _r1h = np.log(_c1h / _c1h.shift(1)).dropna()
            _rv_1h = _r1h.resample('D').apply(
                lambda x: float(np.sqrt((x ** 2).sum())) if len(x) >= 4 else float('nan')
            ).dropna()
            _rv_1h.index = pd.to_datetime(_rv_1h.index).normalize()
            df['rv_intraday']     = _rv_1h.reindex(df.index)
            df['rv_intraday']     = df['rv_intraday'].fillna(df['parkinson_vol'])  # 과거는 Parkinson 대체
            df['rv_intraday_5d']  = df['rv_intraday'].rolling(5).mean().fillna(0)
            df['rv_intraday_21d'] = df['rv_intraday'].rolling(21).mean().fillna(0)
            df['rv_intra_vs_close'] = (df['rv_intraday'] /
                                       df['RV_1d'].replace(0, np.nan)).fillna(1.0).clip(0.3, 5.0)
            log.info(f"    장중 RV 생성: {_rv_1h.notna().sum()}일 (μ={float(_rv_1h.mean()):.5f})")
        else:
            for c in ['rv_intraday','rv_intraday_5d','rv_intraday_21d','rv_intra_vs_close']:
                df[c] = 0.0
    except Exception as _ie:
        log.warning(f"    장중 RV 실패({_ie}) → 0")
        for c in ['rv_intraday','rv_intraday_5d','rv_intraday_21d','rv_intra_vs_close']:
            df[c] = 0.0

    # ── VIX 기간구조 (VIX3M - VIX) + SKEW (꼬리위험)
    if 'VIX3M' in df.columns and df['VIX3M'].notna().sum() > 30:
        df['VIX3M']         = df['VIX3M'].ffill().bfill().fillna(0)
        df['vix_term_slope']= (df['VIX3M'] - df['VIX']).fillna(0)   # >0: 장기>단기(정상), <0: 역전(위험)
        df['vix_ts_zscore'] = ((df['vix_term_slope'] - df['vix_term_slope'].rolling(252).mean()) /
                               (df['vix_term_slope'].rolling(252).std() + 1e-8)).fillna(0)
        log.info(f"    VIX 기간구조 생성 (μ={df['vix_term_slope'].mean():.3f})")
    else:
        df['vix_term_slope'] = 0.0
        df['vix_ts_zscore']  = 0.0

    if 'SKEW' in df.columns and df['SKEW'].notna().sum() > 30:
        df['SKEW']       = df['SKEW'].ffill().bfill().fillna(100)
        df['skew_zscore']= ((df['SKEW'] - df['SKEW'].rolling(252).mean()) /
                            (df['SKEW'].rolling(252).std() + 1e-8)).fillna(0)
        df['skew_chg']   = df['SKEW'].diff().fillna(0)
        log.info(f"    SKEW 피처 생성 (μ={df['SKEW'].mean():.2f})")
    else:
        df['skew_zscore'] = 0.0
        df['skew_chg']    = 0.0

    # ── 레버리지 효과 (하락일 변동성 비대칭)
    df['neg_return']      = (df['return_1d'] < 0).astype(float)
    df['return_neg']      = df['return_1d'].clip(upper=0)   # 음수 수익률만
    df['return_pos']      = df['return_1d'].clip(lower=0)   # 양수 수익률만
    df['leverage_effect'] = df['neg_return'] * df['RV_5d']  # 하락×변동성 교호작용

    # ── EIA 발표 요일 효과 (수요일=재고발표일, 목요일=반응일)
    _dow = pd.to_datetime(df.index).dayofweek
    df['dow_wednesday']  = (_dow == 2).astype(float)
    df['dow_thursday']   = (_dow == 3).astype(float)
    df['dow_monday']     = (_dow == 0).astype(float)
    df['eia_vol_signal'] = df['dow_wednesday'] * df['inv_chg_zscore'].abs()

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

    # ── RSI (14일): 과매수/과매도 신호 (0~100)
    delta = df['WTI'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = (100 - (100 / (1 + gain / (loss + 1e-8)))).fillna(50)  # 초기값 → 중립(50)

    # ── MACD (12-26-9): 단기/장기 EMA 차이로 추세 강도 측정
    ema12 = df['WTI'].ewm(span=12, adjust=False).mean()
    ema26 = df['WTI'].ewm(span=26, adjust=False).mean()
    df['macd']        = (ema12 - ema26).fillna(0)
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)

    # ── ATR proxy (14일 평균 절대 로그수익률): 가격 범위 기반 변동성
    df['atr_14'] = df['log_return'].abs().rolling(14).mean().fillna(0)

    # ── 가격 z-score (252일 롤링): 역사적 수준 대비 현재 위치
    df['price_zscore'] = (
        (df['WTI'] - df['WTI'].rolling(252).mean()) /
        (df['WTI'].rolling(252).std() + 1e-8)
    ).fillna(0)   # 초기 252일 NaN → 0 (평균 수준으로 처리)

    # ── 뉴스 집계: 소스 신뢰도 × 핵심 기관 언급 복합 가중치
    if not news_df.empty:
        news_df = news_df.copy()
        news_df['sentiment'] = news_df['title'].apply(score_sentiment)

        def _impact_w(row):
            src_w = SOURCE_WEIGHTS.get(str(row.get('source', 'RSS')), 1.0)
            ent_w = 1.5 if any(k in str(row['title']).lower() for k in HIGH_IMPACT_ENTITIES) else 1.0
            return src_w * ent_w

        news_df['impact_w']    = news_df.apply(_impact_w, axis=1)
        news_df['w_sentiment'] = news_df['sentiment'] * news_df['impact_w']
        news_df['is_pos']      = (news_df['sentiment'] >  0.05).astype(float)
        news_df['is_neg']      = (news_df['sentiment'] < -0.05).astype(float)

        def _wavg_sent(g):
            return g['w_sentiment'].sum() / g['impact_w'].sum() if g['impact_w'].sum() > 0 else 0.0

        daily = news_df.groupby('date').apply(
            lambda g: pd.Series({
                'news_count':     len(g),
                'news_sentiment': _wavg_sent(g),
                'news_count_pos': g['is_pos'].sum(),
                'news_count_neg': g['is_neg'].sum(),
            })
        )
        daily.index = pd.to_datetime(daily.index)
        df = df.join(daily, how='left')
        df['news_count']     = df['news_count'].fillna(0)
        df['news_count_pos'] = df['news_count_pos'].fillna(0)
        df['news_count_neg'] = df['news_count_neg'].fillna(0)
        # 뉴스 없는 날: 0(중립) 대신 전날 감성 유지 → 지속적 이벤트 반영
        df['news_sentiment'] = df['news_sentiment'].ffill().fillna(0)
    else:
        df['news_count']     = 0
        df['news_count_pos'] = 0
        df['news_count_neg'] = 0
        df['news_sentiment'] = 0

    # ── gpr_zscore 보정: 뉴스가 없는 날 GPR도 ffill로 유지됨 (이미 _attach_gpr에서 처리)
    if 'gpr_zscore' not in df.columns:
        df['gpr_zscore'] = 0.0
    if 'geo_dummy' not in df.columns:
        df['geo_dummy'] = 0.0

    # ── 뉴스 감성 파생 피처
    df['news_sentiment_smooth']  = df['news_sentiment'].ewm(span=3, min_periods=1).mean()
    df['news_sentiment_smooth7'] = df['news_sentiment'].ewm(span=7, min_periods=1).mean()
    # 감성 강도: 절댓값 × log(뉴스수+1) — 큰 감성 + 많은 기사 = 강한 신호
    df['sentiment_magnitude']    = df['news_sentiment'].abs() * np.log1p(df['news_count'])
    # 극단 감성 더미 (EWM 평활 기준 ±0.35 초과)
    df['extreme_neg_news'] = (df['news_sentiment_smooth'] < -0.35).astype(float)
    df['extreme_pos_news'] = (df['news_sentiment_smooth'] >  0.35).astype(float)
    for lag in [1, 2]:
        df[f'news_sentiment_lag{lag}'] = df['news_sentiment'].shift(lag)
        df[f'news_count_lag{lag}']     = df['news_count'].shift(lag)

    # ── 시차 수익률 & 변동성
    for lag in [1, 2]:
        df[f'return_lag{lag}'] = df['return_1d'].shift(lag)
    df['RV_lag1'] = df['RV_1d'].shift(1)

    # ── VIX 기반 피처 (공포지수)
    if 'VIX' in df.columns and df['VIX'].notna().any():
        df['VIX'] = df['VIX'].ffill().bfill()
        vix_mu    = df['VIX'].rolling(252).mean()
        vix_sigma = df['VIX'].rolling(252).std()
        df['vix_zscore'] = ((df['VIX'] - vix_mu) / (vix_sigma + 1e-8)).fillna(0)
        df['vix_change']  = df['VIX'].pct_change().fillna(0)
    else:
        df['vix_zscore'] = 0.0
        df['vix_change']  = 0.0

    # ── OVX (CBOE 원유 변동성 지수) 피처
    if 'OVX' in df.columns and df['OVX'].notna().any():
        df['OVX'] = df['OVX'].ffill().bfill()
        ovx_mu    = df['OVX'].rolling(252).mean()
        ovx_sigma = df['OVX'].rolling(252).std()
        df['ovx_zscore'] = ((df['OVX'] - ovx_mu) / (ovx_sigma + 1e-8)).fillna(0)
        df['ovx_change'] = df['OVX'].pct_change().fillna(0)
        # 내재변동성 - 실현변동성 스프레드 (분산 리스크 프리미엄)
        # OVX: 연환산 %, vol_5d: 일별 소수점 → 동일 단위로 변환
        rv_annualized = df['vol_5d'] * np.sqrt(252) * 100   # 연환산 %
        df['ovx_rv_spread'] = (df['OVX'] - rv_annualized).fillna(0)
    else:
        df['ovx_zscore']   = 0.0
        df['ovx_change']   = 0.0
        df['ovx_rv_spread']= 0.0

    # ── WTI 선물 커브 스프레드 피처 (contango/backwardation)
    if 'futures_spread' in df.columns and df['futures_spread'].notna().sum() > 30:
        df['futures_spread']     = df['futures_spread'].ffill().bfill().fillna(0)
        df['futures_spread_chg'] = df['futures_spread'].diff().fillna(0)
        df['contango_dummy']     = (df['futures_spread'] > 0).astype(float)
        log.info(f"    선물 커브 스프레드 피처 생성 "
                 f"(contango 비율={df['contango_dummy'].mean()*100:.1f}%)")
    else:
        df['futures_spread']     = 0.0
        df['futures_spread_chg'] = 0.0
        df['contango_dummy']     = 0.0

    # ── VIX × 뉴스 감성 복합변수 (뉴스 집계 이후에 계산)
    neg_sent = (-df['news_sentiment_smooth']).clip(lower=0)   # 부정 감성만 추출

    # 공포 복합지수: VIX 높고 뉴스도 부정일 때만 발화 → 가장 신뢰도 높은 신호
    df['fear_composite']  = df['vix_zscore'] * neg_sent

    # VIX 증폭 감성: VIX 레벨로 뉴스 감성 자체를 스케일링
    df['vix_amplified']   = (df['news_sentiment_smooth']
                             * (1 + df['vix_zscore'].clip(lower=0)))

    # VIX-감성 괴리도: VIX↑ + 뉴스 긍정 → 과매도 반등, VIX↓ + 뉴스 부정 → 뉴스 과반응
    df['vix_sent_diverge']= df['vix_zscore'] - neg_sent

    # ── 5번: 시장 국면(Regime) 피처 — 변동성 75th pct 기준 고/저변동 구분
    vol_thresh = df['vol_5d'].quantile(0.75)
    df['regime']       = (df['vol_5d'] > vol_thresh).astype(float)
    df['regime_x_mom'] = df['regime'] * df['mom_5d']         # 국면 × 모멘텀
    df['regime_x_sent']= df['regime'] * df['news_sentiment_smooth']  # 국면 × 감성
    df['regime_x_gpr'] = df['regime'] * df['gpr_zscore']     # 국면 × 지정학

    # ── 훈련 타깃 (다음 날 5일 실현변동성 & 가격 & 수익률)
    df['target_rv']     = df['RV_5d'].shift(-1)
    df['target_rv_log'] = np.log(df['target_rv'].clip(lower=1e-8))
    df['target_price']  = df['WTI'].shift(-1)
    df['target_return'] = np.log(df['WTI'].shift(-1) / df['WTI'])   # 내일 log 수익률

    # 피처 행만 dropna (타깃 NaN 포함 시 훈련용으로만 제거)
    feat_na_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_full = df.copy()               # 마지막 행 보존용 (예측에 사용)
    df.dropna(subset=feat_na_cols + ['target_rv', 'target_rv_log', 'target_price', 'target_return'], inplace=True)

    log.info(f"    피처 완성: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, df_full


# ─────────────────────────────────────────────────────────────────────────────
# 5.  train_models()
# ─────────────────────────────────────────────────────────────────────────────

def train_models(feature_df: pd.DataFrame):
    """XGBoost-HAR (변동성) + SARIMAX (가격) 훈련 및 성능 평가"""
    log.info("[4/9] 모델 훈련 중...")

    available_feats = [c for c in FEATURE_COLS if c in feature_df.columns]
    # HAR 전용 피처: regime/뉴스/거시 제외로 과적합 방지
    har_feats = [c for c in HAR_FEATURE_COLS if c in feature_df.columns]
    log.info(f"    HAR 피처: {len(har_feats)}개 / 전체: {len(available_feats)}개")

    # ── 테스트셋: 최근 60 영업일 (원샷 장기예측 오차 제거)
    n_test   = 60
    train_df = feature_df.iloc[:-n_test]
    test_df  = feature_df.iloc[-n_test:]

    X_tr = train_df[har_feats]   # HAR 모델은 har_feats만 사용
    X_te = test_df[har_feats]
    # 3번: 로그 변환 타깃 사용 (훈련), 평가는 원래 스케일로 역변환
    y_rv_tr,     y_rv_te     = train_df['target_rv'],     test_df['target_rv']
    y_rv_log_tr, y_rv_log_te = train_df['target_rv_log'], test_df['target_rv_log']
    y_px_tr,     y_px_te     = train_df['target_price'],  test_df['target_price']
    y_ret_tr,    y_ret_te    = train_df['target_return'],  test_df['target_return']

    # 가격 모델(XGBoost-Return, SARIMAX)용 전체 피처
    X_tr_all = train_df[available_feats]
    X_te_all = test_df[available_feats]

    results = {}
    scaler  = None

    # ─────────────────────────────────────────────────────────────────────
    # Model A: XGBoost-HAR — walk-forward TimeSeriesSplit (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    if _SKL:
        scaler = StandardScaler()

        log.info("    [A] XGBoost-HAR (5-fold walk-forward CV) 학습 중...")
        if _XGB:
            # 정규화 대폭 강화 (과적합 gap=0.44 → 목표 gap<0.15)
            modelA = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6,
                min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0,
                n_jobs=-1, random_state=42, verbosity=0,
            )
        else:
            modelA = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.02,
                subsample=0.7, random_state=42,
            )

        # ── walk-forward TimeSeriesSplit 평가 (5 fold)
        tscv      = TimeSeriesSplit(n_splits=5)
        wf_preds  = np.zeros(len(X_tr))
        wf_actual = y_rv_tr.values.copy()

        full_X = scaler.fit_transform(X_tr)

        for fold, (idx_tr, idx_va) in enumerate(tscv.split(full_X)):
            X_f, X_v = full_X[idx_tr], full_X[idx_va]
            y_f      = y_rv_tr.iloc[idx_tr]

            covid_w = (np.where(train_df['covid_dummy'].values[idx_tr] == 1, 0.35, 1.0)
                       if 'covid_dummy' in train_df.columns else None)

            m = (xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.025,
                                   subsample=0.8, colsample_bytree=0.7,
                                   min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
                                   n_jobs=-1, random_state=42, verbosity=0)
                 if _XGB else
                 GradientBoostingRegressor(n_estimators=600, max_depth=5,
                                           learning_rate=0.025, subsample=0.8,
                                           random_state=42))
            m.fit(X_f, y_f, sample_weight=covid_w)
            wf_preds[idx_va] = m.predict(X_v)

        rmse_cv = float(np.sqrt(mean_squared_error(wf_actual, wf_preds)))
        mae_cv  = float(mean_absolute_error(wf_actual, wf_preds))
        r2_cv   = float(r2_score(wf_actual, wf_preds))
        log.info(f"        Walk-forward CV → RMSE={rmse_cv:.5f}  MAE={mae_cv:.5f}  R²={r2_cv:.4f}")

        # ── 최종 모델: 전체 훈련셋으로 재학습 (지수감쇠 + COVID 가중치)
        X_tr_s = full_X
        X_te_s = scaler.transform(X_te)

        _n = len(y_rv_tr)
        _time_w = np.exp(np.log(2) / 252 * np.arange(_n))  # 반감기 1년
        _time_w = _time_w / _time_w.mean()
        covid_w_full = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                        if 'covid_dummy' in train_df.columns else np.ones(_n))
        combined_w = covid_w_full * _time_w
        log.info(f"        지수감쇠 가중치: 최신/최고참 비율={_time_w[-1]/_time_w[0]:.1f}x")
        modelA.fit(X_tr_s, y_rv_tr, sample_weight=combined_w)

        # ── 홀드아웃 테스트셋 평가
        pred_rv = modelA.predict(X_te_s)
        rmse_ho = float(np.sqrt(mean_squared_error(y_rv_te, pred_rv)))
        mae_ho  = float(mean_absolute_error(y_rv_te, pred_rv))
        r2_ho   = float(r2_score(y_rv_te, pred_rv))
        log.info(f"        Hold-out 60d → RMSE={rmse_ho:.5f}  MAE={mae_ho:.5f}  R²={r2_ho:.4f}")

        # ── 과적합 감지 (훈련 R² vs CV R² 비교)
        r2_train = float(r2_score(y_rv_tr, modelA.predict(X_tr_s)))
        overfit_gap = r2_train - r2_cv
        if overfit_gap > 0.20:
            log.warning(f"    ⚠️ 과적합 의심: 훈련R²={r2_train:.4f} vs CV R²={r2_cv:.4f} "
                        f"(gap={overfit_gap:.3f})")
        else:
            log.info(f"        훈련R²={r2_train:.4f}  CV R²={r2_cv:.4f}  gap={overfit_gap:.3f} (정상)")

        # ── HAR-Ridge walk-forward CV 평가 후 XGBoost와 비교, 더 좋으면 채택
        try:
            _rf = [c for c in ['RV_1d','RV_5d','RV_21d','RV_63d',
                                'garch_vol','parkinson_vol','ewma_vol_10',
                                'leverage_effect','dow_wednesday',
                                'rv_intraday','ovx_zscore','vix_term_slope',
                                'skew_zscore','brent_rv_1d_lag1'] if c in har_feats]
            _sc_r  = StandardScaler()
            _Xr_tr = _sc_r.fit_transform(train_df[_rf])
            _Xr_te = _sc_r.transform(test_df[_rf])

            # Walk-forward CV로 Ridge 평가 (공정 비교)
            _ridge_preds = np.zeros(len(y_rv_tr))
            for _ti, _vi in TimeSeriesSplit(n_splits=5).split(_Xr_tr):
                _rm = Ridge(alpha=1.0)
                _rm.fit(_Xr_tr[_ti], y_rv_tr.values[_ti])
                _ridge_preds[_vi] = _rm.predict(_Xr_tr[_vi])
            _r2_ridge_cv = float(r2_score(y_rv_tr.values, _ridge_preds))

            # 최종 Ridge (전체 훈련셋)
            _ridge_final = Ridge(alpha=1.0)
            _ridge_final.fit(_Xr_tr, y_rv_tr)
            _r2_ridge_ho = float(r2_score(y_rv_te, _ridge_final.predict(_Xr_te)))
            _r2_xgb_ho   = float(r2_score(y_rv_te, modelA.predict(X_te_s)))

            log.info(f"        HAR-Ridge CV R²={_r2_ridge_cv:.4f}  Hold-out R²={_r2_ridge_ho:.4f}")
            log.info(f"        XGBoost  CV R²={r2_cv:.4f}   Hold-out R²={_r2_xgb_ho:.4f}")

            # Hold-out과 CV 모두 Ridge가 좋으면 채택
            if _r2_ridge_cv > r2_cv and _r2_ridge_ho > _r2_xgb_ho:
                modelA    = _ridge_final
                scaler    = _sc_r
                har_feats = _rf
                r2_cv     = _r2_ridge_cv
                pred_rv   = modelA.predict(_Xr_te)
                rmse_ho   = float(np.sqrt(mean_squared_error(y_rv_te, pred_rv)))
                mae_ho    = float(mean_absolute_error(y_rv_te, pred_rv))
                r2_ho     = _r2_ridge_ho
                log.info(f"    ✅ HAR-Ridge 채택: CV R²={r2_cv:.4f}  Hold-out R²={r2_ho:.4f}")
            else:
                log.info(f"    XGBoost 유지 (Ridge CV={_r2_ridge_cv:.4f} vs XGB CV={r2_cv:.4f})")
        except Exception as _he:
            log.warning(f"    HAR-Ridge 평가 실패({_he})")

        # ── 피처 중요도 저장 + 상위 30개 기록
        if hasattr(modelA, 'feature_importances_'):
            imp = sorted(zip(available_feats, modelA.feature_importances_),
                         key=lambda x: x[1], reverse=True)
            top_str = ', '.join(f"{n}({v:.3f})" for n, v in imp[:8])
            log.info(f"        피처 중요도 Top8: {top_str}")
            imp_df = pd.DataFrame(imp, columns=['feature', 'importance'])
            imp_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

        # ── 장중 RV를 타깃으로 한 별도 모델 (최근 730일, 더 정확한 측정값)
        if 'rv_intraday' in feature_df.columns:
            try:
                _intra_df = feature_df[feature_df['rv_intraday'] > 0].copy()
                _intra_df['target_intra'] = _intra_df['rv_intraday'].shift(-1)
                _intra_df = _intra_df.dropna(subset=['target_intra'])
                if len(_intra_df) > 120:
                    _n_te_i    = min(60, int(len(_intra_df) * 0.15))
                    _intra_tr  = _intra_df.iloc[:-_n_te_i]
                    _intra_te  = _intra_df.iloc[-_n_te_i:]
                    _avail_i   = [c for c in available_feats if c in _intra_tr.columns]
                    _sc_i      = StandardScaler()
                    _Xi_tr     = _sc_i.fit_transform(_intra_tr[_avail_i])
                    _Xi_te     = _sc_i.transform(_intra_te[_avail_i])
                    _mi        = (xgb.XGBRegressor(n_estimators=300, max_depth=3,
                                                   learning_rate=0.02, subsample=0.8,
                                                   n_jobs=-1, random_state=42, verbosity=0)
                                  if _XGB else Ridge(alpha=1.0))
                    _mi.fit(_Xi_tr, _intra_tr['target_intra'])
                    _pi        = _mi.predict(_Xi_te)
                    _r2_intra  = float(r2_score(_intra_te['target_intra'], _pi))
                    _r2_train_i = float(r2_score(_intra_tr['target_intra'], _mi.predict(_Xi_tr)))
                    log.info(f"        장중RV 타깃 모델: R²={_r2_intra:.4f}  "
                             f"훈련R²={_r2_train_i:.4f}  n_train={len(_intra_tr)}")
                    # 현재 모델보다 유의미하게 좋으면 교체
                    if _r2_intra > r2_cv + 0.05 and (_r2_train_i - _r2_intra) < 0.25:
                        modelA      = _mi
                        scaler      = _sc_i
                        available_feats = _avail_i
                        r2_cv       = _r2_intra
                        log.info(f"    ✅ 장중RV 타깃 모델 채택 (R²={_r2_intra:.4f})")
                    else:
                        log.info(f"    기존 모델 유지 (장중RV R²={_r2_intra:.4f} 미채택)")
            except Exception as _ie:
                log.debug(f"    장중RV 타깃 모델 실패({_ie})")

        results['xgb_har'] = {
            'model': modelA, 'scaler': scaler, 'features': har_feats,
            'type': 'vol_5d',
            'rmse': rmse_cv, 'mae': mae_cv, 'r2': r2_cv,
            'rmse_ho': rmse_ho, 'r2_ho': r2_ho,
            'train_r2': r2_train, 'overfit_gap': overfit_gap,
            'name': 'XGBoost-HAR (WalkFwd)' if _XGB else 'GBM-HAR (WalkFwd)',
            'pred_rv_test':    pred_rv,
            'actual_rv_test':  y_rv_te.values,
            'test_dates':      test_df.index,
        }

        # Step 3: HAR 예측 변동성 → XGBoost-Return 피처로 추가 (vol→price 인과 활용)
        try:
            _hf_avail = [c for c in har_feats if c in feature_df.columns]
            _hX_full  = scaler.transform(feature_df[_hf_avail])
            feature_df['har_vol_pred'] = np.abs(modelA.predict(_hX_full))
            if 'har_vol_pred' not in available_feats:
                available_feats.append('har_vol_pred')
            X_tr_all = feature_df.iloc[:-n_test][available_feats]
            X_te_all = feature_df.iloc[-n_test:][available_feats]
            log.info("    ✅ HAR vol 예측값 → XGB-Return 피처 추가 (har_vol_pred)")
        except Exception as _he3:
            log.debug(f"    HAR vol 피처 추가 실패({_he3})")

    # ─────────────────────────────────────────────────────────────────────
    # Model B: SARIMAX — 1-step ahead dynamic=False 평가 (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    # Exog: 거시(DXY/충격) + 원유 시장 구조(Brent스프레드/OVX/선물커브) + VIX
    exog_cols = [c for c in [
        'dxy_change', 'demand_shock', 'supply_shock', 'vix_change',
        'brent_wti_spread', 'ovx_change', 'futures_spread',
    ] if c in feature_df.columns]
    log.info("    [B] SARIMAX 학습 + 1-step ahead 평가 중...")

    if _SARIMAX and len(train_df) > 60:
        try:
            # 2번: 영업일(B) 주파수 명시 → SARIMAX 계절 패턴 인식 개선
            def _to_bday(s):
                try:
                    return s.asfreq('B', method='ffill')
                except Exception:
                    return s

            # SARIMAX는 최근 SARIMAX_YEARS 년치만 사용 (오래된 가격 레짐 영향 최소화)
            cutoff = feature_df.index[-1] - pd.DateOffset(years=SARIMAX_YEARS)
            sx_df  = feature_df[feature_df.index >= cutoff]
            n_test_sx = min(60, int(len(sx_df) * 0.15))
            sx_train  = sx_df.iloc[:-n_test_sx]
            sx_test   = sx_df.iloc[-n_test_sx:]

            full_wti  = _to_bday(sx_df['WTI'])
            full_exog = _to_bday(sx_df[exog_cols]) if exog_cols else None
            n_train   = len(sx_train)

            wti_train  = _to_bday(sx_train['WTI'])
            exog_train = _to_bday(sx_train[exog_cols]) if exog_cols else None

            # 1번: auto_arima로 최적 파라미터 탐색
            sarimax_order    = (2, 1, 1)
            sarimax_seasonal = (1, 0, 1, 5)
            if _PMDARIMA:
                try:
                    log.info("    [B0] auto_arima 파라미터 탐색 중 (stepwise)...")
                    aa = _auto_arima(
                        wti_train, exogenous=exog_train,
                        d=1, seasonal=True, m=5, D=0,
                        start_p=0, max_p=4, start_q=0, max_q=3,
                        start_P=0, max_P=2, start_Q=0, max_Q=2,
                        information_criterion='aic', stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        n_jobs=-1,
                    )
                    sarimax_order    = aa.order
                    sarimax_seasonal = aa.seasonal_order
                    log.info(f"    auto_arima 최적: {sarimax_order} × {sarimax_seasonal}")
                except Exception as e:
                    log.warning(f"    auto_arima 실패({e}) → 기본 파라미터 사용")

            # 훈련 데이터로만 파라미터 추정
            mdl_tr = SARIMAX(
                wti_train,
                exog=exog_train,
                order=sarimax_order,
                seasonal_order=sarimax_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mdl_tr.fit(disp=False, maxiter=300)

            # ── 1-step ahead 평가: 추정된 파라미터 고정 + 전체 시리즈 적용
            mdl_full = SARIMAX(
                full_wti,
                exog=full_exog,
                order=sarimax_order,
                seasonal_order=sarimax_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # 훈련 파라미터 고정, 칼만 필터로 1-step ahead 예측
            fit_full   = mdl_full.filter(fit.params)
            pred_obj   = fit_full.get_prediction(start=n_train, dynamic=False)
            pred_price = pred_obj.predicted_mean.values[-n_test:]

            y_px_te_sx = sx_test['target_price'].values
            rmse_b = float(np.sqrt(mean_squared_error(y_px_te_sx, pred_price)))
            mae_b  = float(mean_absolute_error(y_px_te_sx, pred_price))
            r2_b   = float(r2_score(y_px_te_sx, pred_price))

            # 라이브 예측용: 전체 데이터로 재학습 (훈련셋만 학습한 fit은 60일 전 상태)
            log.info("    [B1] 전체 데이터 SARIMAX 재학습 (라이브 예측용)...")
            mdl_live = SARIMAX(
                full_wti, exog=full_exog,
                order=sarimax_order, seasonal_order=sarimax_seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit_live = mdl_live.fit(disp=False, maxiter=300)

            results['sarimax'] = {
                'model': fit_live, 'features': exog_cols, 'type': 'price',
                'rmse': rmse_b, 'mae': mae_b, 'r2': r2_b,
                'name': f'SARIMAX{sarimax_order} 1-step',
                'pred_price_test':   pred_price,
                'actual_price_test': sx_test['WTI'].values,
                'test_dates':        sx_test.index,
                'order': sarimax_order, 'seasonal_order': sarimax_seasonal,
            }
            log.info(f"        1-step ahead → RMSE={rmse_b:.4f}  MAE={mae_b:.4f}  R²={r2_b:.4f}")

            # 3번: SARIMAX 잔차 교정 (Residual Correction)
            log.info("    [B2] SARIMAX 잔차 교정 모델 학습 중...")
            try:
                # 전체 데이터 fit_live의 잔차 사용 (훈련셋 구간만 추출)
                train_resid = fit_live.resid.reindex(train_df.index).dropna()
                rc_feat_cols = [c for c in
                    ['vol_5d', 'vix_change', 'news_sentiment_smooth', 'dxy_change', 'gpr_zscore']
                    if c in train_df.columns]

                resid_s = pd.Series(train_resid.values,
                                    index=train_df.index[-len(train_resid):])
                rc_df   = train_df.loc[resid_s.index, rc_feat_cols].copy().fillna(0)
                rc_df['resid_lag1'] = resid_s.shift(1)
                rc_df['resid_lag2'] = resid_s.shift(2)
                rc_target = resid_s.shift(-1)          # 다음 날 잔차가 타깃

                valid_idx = rc_df.dropna().index.intersection(rc_target.dropna().index)
                X_rc = rc_df.loc[valid_idx]
                y_rc = rc_target.loc[valid_idx]

                if len(X_rc) > 30:
                    rc_scaler = StandardScaler()
                    X_rc_s    = rc_scaler.fit_transform(X_rc)
                    rc_model  = Ridge(alpha=10.0)      # 강한 정규화 → 과적합 방지
                    rc_model.fit(X_rc_s, y_rc)

                    # 테스트셋 교정값 계산 (마지막 훈련 잔차 사용)
                    X_te_rc = test_df[rc_feat_cols].fillna(0).copy()
                    X_te_rc['resid_lag1'] = float(resid_s.iloc[-1])
                    X_te_rc['resid_lag2'] = float(resid_s.iloc[-2])
                    corrections = rc_model.predict(rc_scaler.transform(X_te_rc[X_rc.columns]))

                    corrected = pred_price + corrections
                    r2_c = float(r2_score(y_px_te, corrected))
                    r2_s = results['sarimax']['r2']

                    if r2_c > r2_s:
                        log.info(f"        잔차 교정 채택 ✓  R²: {r2_s:.4f} → {r2_c:.4f}")
                        # last_resid는 전체 데이터 기준 최신 잔차 사용
                        full_resid = fit_live.resid.reindex(feature_df.index).dropna()
                        results['resid_corrector'] = {
                            'model':       rc_model,
                            'scaler':      rc_scaler,
                            'features':    list(X_rc.columns),
                            'last_resid1': float(full_resid.iloc[-1]),
                            'last_resid2': float(full_resid.iloc[-2]),
                            'rc_feat_cols': rc_feat_cols,
                        }
                        results['sarimax']['r2']             = r2_c
                        results['sarimax']['pred_price_test'] = corrected
                    else:
                        log.info(f"        잔차 교정 미채택 (R²: {r2_s:.4f} ≥ {r2_c:.4f})")
            except Exception as e:
                log.warning(f"    잔차 교정 실패({e})")

        except Exception as exc:
            log.warning(f"SARIMAX 실패: {exc} → Ridge 대체")
            if _SKL and scaler:
                _ridge_fallback(results, X_tr_s, y_px_tr, X_te_s, y_px_te,
                                available_feats, scaler)
    elif _SKL and scaler:
        _ridge_fallback(results, X_tr_s, y_px_tr, X_te_s, y_px_te,
                        available_feats, scaler)

    # Prophet: WTI는 불연속 충격 기반이라 트렌드+계절성 모델 부적합 (R²=-4.3 확인)
    # → 학습 생략, 2모델 앙상블(SARIMAX+XGBoost) 유지

    # ─────────────────────────────────────────────────────────────────────
    # Model D: XGBoost 수익률 예측 (log_return 타깃)
    # vol 시뮬레이션 대체 — 방향성+크기 직접 학습
    # ─────────────────────────────────────────────────────────────────────
    if _XGB and _SKL and scaler is not None:
        log.info("    [D] XGBoost 수익률 예측 학습 중...")
        try:
            ret_scaler = StandardScaler()
            X_tr_ret   = ret_scaler.fit_transform(X_tr_all)   # 전체 피처 사용
            X_te_ret   = ret_scaler.transform(X_te_all)

            # 지수감쇠 + COVID 가중치
            _n_ret  = len(y_ret_tr)
            _tw_ret = np.exp(np.log(2) / 252 * np.arange(_n_ret))
            _tw_ret /= _tw_ret.mean()
            covid_w_ret = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                           if 'covid_dummy' in train_df.columns else np.ones(_n_ret))
            w_ret = covid_w_ret * _tw_ret

            modelD = xgb.XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.015,
                subsample=0.75, colsample_bytree=0.6,
                min_child_weight=8, reg_alpha=0.3, reg_lambda=3.0,
                n_jobs=-1, random_state=42, verbosity=0,
            )
            modelD.fit(X_tr_ret, y_ret_tr, sample_weight=w_ret)

            pred_ret   = modelD.predict(X_te_ret)
            # 수익률 → 가격 역변환 후 평가
            pred_px_d  = test_df['WTI'].values * np.exp(pred_ret)
            rmse_d     = float(np.sqrt(mean_squared_error(y_px_te, pred_px_d)))
            mae_d      = float(mean_absolute_error(y_px_te, pred_px_d))
            r2_d       = float(r2_score(y_px_te, pred_px_d))

            # 방향성 정확도 (상승/하락 예측 일치율)
            actual_dir  = np.sign(y_ret_te.values)
            pred_dir    = np.sign(pred_ret)
            dir_acc     = float((actual_dir == pred_dir).mean())

            log.info(f"        XGB-Return → RMSE={rmse_d:.4f}  MAE={mae_d:.4f}  "
                     f"R²={r2_d:.4f}  방향성={dir_acc*100:.1f}%")

            results['xgb_return'] = {
                'model': modelD, 'scaler': ret_scaler,
                'features': available_feats, 'type': 'price',
                'rmse': rmse_d, 'mae': mae_d, 'r2': r2_d,
                'dir_acc': dir_acc,
                'name': f'XGBoost-Return (방향성={dir_acc*100:.1f}%)',
            }
        except Exception as exc:
            log.warning(f"    XGBoost 수익률 예측 실패({exc})")

    # ── 성능 저장
    perf_rows = []
    for v in results.values():
        row = {'model': v['name'], 'target': v['type'],
               'rmse': round(v['rmse'], 5), 'mae': round(v['mae'], 5), 'r2': round(v['r2'], 4)}
        if 'dir_acc'     in v: row['dir_acc']     = round(v['dir_acc'], 4)
        if 'train_r2'    in v: row['train_r2']    = round(v['train_r2'], 4)
        if 'overfit_gap' in v: row['overfit_gap'] = round(v['overfit_gap'], 4)
        perf_rows.append(row)
    perf_df = pd.DataFrame(perf_rows)
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




def _train_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   exog_feats: list) -> dict | None:
    """4번: Prophet 모델 학습 및 hold-out 평가"""
    if not _PROPHET or not _SKL:
        return None
    try:
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(train_df.index),
            'y':  train_df['WTI'].values,
        })
        reg_cols = [c for c in exog_feats if c in train_df.columns]
        for c in reg_cols:
            prophet_df[c] = train_df[c].fillna(0).values

        m = _Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=5.0,
        )
        for c in reg_cols:
            m.add_regressor(c, standardize=True)
        m.fit(prophet_df)

        # hold-out 평가
        test_future = pd.DataFrame({'ds': pd.to_datetime(test_df.index)})
        for c in reg_cols:
            test_future[c] = test_df[c].fillna(0).values
        fc = m.predict(test_future)
        pred = fc['yhat'].values

        rmse_p = float(np.sqrt(mean_squared_error(test_df['WTI'].values, pred)))
        mae_p  = float(mean_absolute_error(test_df['WTI'].values, pred))
        r2_p   = float(r2_score(test_df['WTI'].values, pred))
        log.info(f"        Prophet Hold-out → RMSE={rmse_p:.4f}  R²={r2_p:.4f}")

        # R² < 0.3이면 실용성 없음 → 미사용
        if r2_p < 0.3:
            log.info(f"        Prophet 미채택 (R²={r2_p:.4f} < 0.3 기준)")
            return None

        return {
            'model': m, 'type': 'price',
            'rmse': rmse_p, 'mae': mae_p, 'r2': r2_p,
            'reg_cols': reg_cols,
            'pred_price_test': pred,
            'name': 'Prophet',
        }
    except Exception as exc:
        log.warning(f"    Prophet 학습 실패({exc}) → 미사용")
        return None


def compute_ensemble_weights(window: int = 30):
    """R² 기반 초기 가중치 + MAPE 미세조정으로 SARIMAX/XGBoost 동적 가중치 산출.

    1단계: model_performance.csv의 테스트셋 R²로 비례 가중치 계산
    2단계: 최근 backtest/live MAPE로 ±0.1 범위 미세조정
    R² 정보 없으면 기본값(0.65/0.35), 최종 클램프 [0.30, 0.70].
    """
    default = (0.65, 0.35)

    # ── 1단계: R² 기반 초기 가중치 ──────────────────────────────────────
    w_s_base = 0.65
    perf_path = OUTPUT_DIR / 'model_performance.csv'
    if perf_path.exists():
        try:
            pf = pd.read_csv(perf_path)
            sx  = pf[pf['model'].str.startswith('SARIMAX')]
            xgr = pf[pf['model'].str.startswith('XGBoost-Return')]
            if not sx.empty and not xgr.empty:
                r2_s = float(sx['r2'].iloc[0])
                r2_x = float(xgr['r2'].iloc[0])
                if (r2_s + r2_x) > 0 and r2_s > 0 and r2_x > 0:
                    w_s_base = float(np.clip(r2_s / (r2_s + r2_x), 0.30, 0.70))
                    log.info(f"    R² 기반 초기 가중치: SARIMAX={w_s_base:.2f} "
                             f"XGB={1-w_s_base:.2f} (R²_s={r2_s:.4f} R²_x={r2_x:.4f})")
        except Exception:
            pass

    # ── 2단계: MAPE 미세조정 ─────────────────────────────────────────────
    if not PRED_LOG_FILE.exists():
        return w_s_base, 1 - w_s_base
    try:
        pl = pd.read_csv(PRED_LOG_FILE)
        bt = pl[(pl['type'] == 'backtest') & pl['price_error'].notna()].tail(window)
        lv = pl[(pl['type'] == 'live') & pl['price_error'].notna()].tail(10)

        if len(bt) < 10 or bt['actual_price'].isna().all():
            return w_s_base, 1 - w_s_base

        bt_mape = (bt['price_error'].abs() / bt['actual_price'].mean() * 100).mean()
        if len(lv) >= 2:
            lv_mape = (lv['price_error'].abs() / lv['actual_price'].mean() * 100).mean()
            sarimax_mape = 0.3 * bt_mape + 0.7 * lv_mape
        else:
            sarimax_mape = bt_mape

        # MAPE 선형 보정: 3%→+0.05, 8%→-0.10 사이 선형 보간
        mape_adj = float(np.clip(np.interp(sarimax_mape, [3.0, 8.0], [0.05, -0.10]), -0.10, 0.05))

        w_s = float(np.clip(w_s_base + mape_adj, 0.30, 0.70))
        log.info(f"    최종 앙상블 가중치: SARIMAX={w_s:.2f} XGB={1-w_s:.2f} "
                 f"(R²기반={w_s_base:.2f} MAPE조정={mape_adj:+.2f} "
                 f"sarimax_mape={sarimax_mape:.2f}%)")
        return w_s, 1 - w_s
    except Exception:
        return w_s_base, 1 - w_s_base


def compute_live_bias_correction(window: int = 10, max_correction: float = 5.0) -> float:
    """최근 live 실측 오차의 지수가중 평균으로 bias correction 값 반환.

    예측이 지속적으로 한쪽으로 치우칠 때 다음 예측에 보정값을 더함.
    max_correction으로 과보정 방지. live 데이터 3건 미만이면 0 반환.
    price_error = actual - predicted 이므로 양수면 과소예측 → 더해야 함.
    """
    if not PRED_LOG_FILE.exists():
        return 0.0
    try:
        pl = pd.read_csv(PRED_LOG_FILE)
        lv = pl[(pl['type'] == 'live') & pl['price_error'].notna()].tail(window)
        if len(lv) < 2:
            return 0.0

        # 지수가중 평균 (최근일수록 더 반영)
        errors = lv['price_error'].values.astype(float)
        weights = np.exp(np.linspace(0, 1, len(errors)))
        weights /= weights.sum()
        bias = float(np.dot(weights, errors))

        # 과보정 방지
        bias = max(-max_correction, min(max_correction, bias))
        log.info(f"    Live bias correction: {bias:+.3f}$ (최근 {len(lv)}건 지수가중)")
        return bias
    except Exception:
        return 0.0


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

    # ── XGBoost 수익률 예측 → 가격 역변환 (xgb_return 우선, 없으면 vol 폴백)
    if 'xgb_return' in results:
        try:
            info    = results['xgb_return']
            model   = info['model']
            sc      = info['scaler']
            feats   = info['features']
            avail_f = [f for f in feats if f in feature_df.columns]
            last_row = feature_df[avail_f].iloc[-1:].values.copy()
            last_s   = sc.transform(last_row)

            pred_ret_d1 = float(model.predict(last_s)[0])   # D+1 log 수익률
            # D+1~7: 수익률 예측값에 불확실성 감쇠 적용 (멀수록 0에 수렴)
            decay = np.array([0.95 ** i for i in range(7)])
            ret_path = pred_ret_d1 * decay
            price_path = last_price * np.exp(np.cumsum(ret_path))
            forecasts['xgb'] = np.round(price_path, 2)
            log.info(f"    XGBoost-Return 예측 D+1: ret={pred_ret_d1:+.4f} "
                     f"→ ${price_path[0]:.2f} (방향: {'↑' if pred_ret_d1>0 else '↓'})")
        except Exception as exc:
            log.warning(f"XGBoost-Return 예측 실패({exc}) → vol 시뮬 폴백")

    if 'xgb' not in forecasts and 'xgb_har' in results:
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
                ret     = np.random.normal(0, rv_pred)
                path.append(path[-1] * (1 + ret))
                row = row * 0.98
            forecasts['xgb'] = np.array(path[1:])
            log.warning("    vol 시뮬레이션 폴백 사용")
        except Exception as exc:
            log.warning(f"XGBoost 예측 실패: {exc}")

    # ── 3번: SARIMAX 잔차 교정 적용
    if 'resid_corrector' in results and 'sarimax' in forecasts:
        try:
            rc   = results['resid_corrector']
            last = feature_df[rc['rc_feat_cols']].tail(1).fillna(0).copy()
            last['resid_lag1'] = rc['last_resid1']
            last['resid_lag2'] = rc['last_resid2']
            correction = float(rc['model'].predict(
                rc['scaler'].transform(last[rc['features']]))[0])
            forecasts['sarimax'] = forecasts['sarimax'] + correction
            log.info(f"    잔차 교정 적용: {correction:+.3f}")
        except Exception as e:
            log.warning(f"잔차 교정 예측 실패: {e}")

    # ── 4번: Prophet 예측
    if 'prophet' in results:
        try:
            pm      = results['prophet']['model']
            rcols   = results['prophet']['reg_cols']
            fut_p   = pd.DataFrame({'ds': fc_dates})
            for c in rcols:
                fut_p[c] = float(feature_df[c].tail(5).mean()) if c in feature_df.columns else 0.0
            forecasts['prophet'] = pm.predict(fut_p)['yhat'].values
            log.info(f"    Prophet 7일 예측: {forecasts['prophet'].round(2)}")
        except Exception as e:
            log.warning(f"Prophet 예측 실패: {e}")

    # ── 동적 앙상블 가중치 (최근 backtest 오차 기반)
    w_sarimax, w_xgb = compute_ensemble_weights()

    # ── 앙상블 — Prophet 있으면 3모델, 없으면 기존 2모델
    if 'sarimax' in forecasts and 'prophet' in forecasts and 'xgb' in forecasts:
        # Prophet 25% 고정, SARIMAX/XGBoost는 동적 가중치 75% 내 배분
        ensemble = (0.75 * (w_sarimax * forecasts['sarimax'] + w_xgb * forecasts['xgb'])
                    + 0.25 * forecasts['prophet'])
        log.info(f"    ✅ 3모델 앙상블 활성: SARIMAX×{w_sarimax*0.75:.2f} "
                 f"Prophet×0.25 XGB×{w_xgb*0.75:.2f}")
        log.info(f"       D+1 예측: SARIMAX={forecasts['sarimax'][0]:.2f} "
                 f"Prophet={forecasts['prophet'][0]:.2f} XGB={forecasts['xgb'][0]:.2f} "
                 f"→ 앙상블={ensemble[0]:.2f}")
    elif 'sarimax' in forecasts and 'xgb' in forecasts:
        ensemble = w_sarimax * forecasts['sarimax'] + w_xgb * forecasts['xgb']
        log.info(f"    ⚠️ 2모델 앙상블 (Prophet 미채택): SARIMAX×{w_sarimax:.2f} XGB×{w_xgb:.2f}")
        log.info(f"       D+1 예측: SARIMAX={forecasts['sarimax'][0]:.2f} "
                 f"XGB={forecasts['xgb'][0]:.2f} → 앙상블={ensemble[0]:.2f}")
    elif 'sarimax' in forecasts:
        ensemble = forecasts['sarimax']
    elif 'xgb' in forecasts:
        ensemble = forecasts['xgb']
    else:
        trend = feature_df['WTI'].diff().tail(5).mean()
        ensemble = np.array([last_price + trend * (i + 1) for i in range(7)])

    # ── A: live bias correction (실측 오차 피드백) ───────────────────
    bias = compute_live_bias_correction()
    if bias != 0.0:
        ensemble = ensemble + bias

    # ── 신뢰구간 (변동성 기반)
    recent_vol = float(feature_df['vol_5d'].dropna().iloc[-1]) if 'vol_5d' in feature_df.columns else 0.015
    t = np.arange(1, 8)
    ci_half = ensemble * recent_vol * 1.96 * np.sqrt(t)

    fc_df = pd.DataFrame({
        'date':           fc_dates.strftime('%Y-%m-%d'),
        'forecast_price': np.round(ensemble, 2),
        'lower_95ci':     np.round(ensemble - ci_half, 2),
        'upper_95ci':     np.round(ensemble + ci_half, 2),
        'bias_correction': round(bias, 3),
    })
    if 'sarimax' in forecasts:
        fc_df['sarimax_forecast'] = np.round(forecasts['sarimax'], 2)
    if 'xgb' in forecasts:
        fc_df['xgb_forecast'] = np.round(forecasts['xgb'], 2)
    if 'prophet' in forecasts:
        fc_df['prophet_forecast'] = np.round(forecasts['prophet'], 2)

    # 모델 합의도: 예측값들의 표준편차 (낮을수록 모델 간 일치)
    pred_cols = [c for c in ['sarimax_forecast', 'xgb_forecast', 'prophet_forecast']
                 if c in fc_df.columns]
    if len(pred_cols) >= 2:
        fc_df['model_std'] = fc_df[pred_cols].std(axis=1).round(2)

    fc_df.to_csv(OUTPUT_DIR / 'forecast_7days.csv', index=False)
    log.info("    forecast_7days.csv 저장")
    return fc_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  save_prediction_log()
# ─────────────────────────────────────────────────────────────────────────────

PRED_LOG_FILE = OUTPUT_DIR / 'prediction_log.csv'

def save_prediction_log(results: dict, feature_df: pd.DataFrame, fc_df: pd.DataFrame,
                        prev_fc_df: pd.DataFrame = None, full_df: pd.DataFrame = None):
    """예측 vs 실제 오차 로그 누적 저장
    - backtest: 60일 테스트셋 (매 실행마다 재구성)
    - live: 실행일 기준 entry + 미실행일은 직전 7일 예측으로 gap-fill
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

    # feature_df에 없는 최근 날짜는 full_df(raw 가격)로 fallback
    price_src = feature_df.copy()
    if full_df is not None and 'WTI' in full_df.columns:
        missing_idx = full_df.index.difference(feature_df.index)
        if not missing_idx.empty:
            price_src = pd.concat([price_src, full_df.loc[missing_idx, ['WTI']]])

    # ── 기존 live 기록 로드 + 실제값 업데이트 ────────────────────────
    live_rows = []
    if PRED_LOG_FILE.exists():
        existing  = pd.read_csv(PRED_LOG_FILE)
        old_live  = existing[existing['type'] == 'live'].copy()

        for idx, row in old_live.iterrows():
            try:
                dt = pd.to_datetime(row['date'])
                if pd.isna(row.get('actual_price')) and dt in price_src.index:
                    ap = float(price_src.loc[dt, 'WTI'])
                    pp = float(row['sarimax_pred'])
                    av = float(feature_df.loc[dt, 'RV_5d']) if (dt in feature_df.index and 'RV_5d' in feature_df.columns) else np.nan
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

    # ── 누락일 gap-fill (직전 실행의 7일 예측 활용) ──────────────────
    today = pd.Timestamp.today().normalize()
    existing_dates = {r.get('date') for r in live_rows}

    if prev_fc_df is not None and not prev_fc_df.empty and live_rows:
        try:
            prev_lookup = {str(row['date']): float(row['forecast_price'])
                           for _, row in prev_fc_df.iterrows()}
            last_live_date = pd.to_datetime(max(existing_dates))
            gap_dates = pd.bdate_range(
                start=last_live_date + timedelta(days=1),
                end=today - timedelta(days=1),
            )
            n_filled_gap = 0
            for gd in gap_dates:
                gd_str = gd.strftime('%Y-%m-%d')
                if gd_str in existing_dates or gd_str not in prev_lookup:
                    continue
                gap_pred = round(prev_lookup[gd_str], 2)
                gap_actual = gap_error = gap_error_pct = None
                if gd in price_src.index:
                    try:
                        gap_actual    = round(float(price_src.loc[gd, 'WTI']), 2)
                        gap_error     = round(gap_actual - gap_pred, 2)
                        gap_error_pct = round((gap_actual - gap_pred) / gap_actual * 100, 2)
                    except Exception:
                        pass
                live_rows.append({
                    'date':            gd_str,
                    'sarimax_pred':    gap_pred,
                    'actual_price':    gap_actual,
                    'price_error':     gap_error,
                    'price_error_pct': gap_error_pct,
                    'xgb_pred_vol':    None,
                    'actual_vol_5d':   None,
                    'vol_error':       None,
                    'type':            'live',
                })
                existing_dates.add(gd_str)
                n_filled_gap += 1
            if n_filled_gap:
                log.info(f"    gap-fill: {n_filled_gap}일 누락 채움")
        except Exception as e:
            log.warning(f"gap-fill 실패: {e}")

    # ── 실행일 기준 live entry 추가 / 덮어쓰기 ───────────────────────
    # 영업일이면 오늘 날짜, 주말/휴일이면 다음 영업일을 entry date로 사용
    if len(pd.bdate_range(today, today)) > 0:
        entry_date_str = today.strftime('%Y-%m-%d')
    else:
        entry_date_str = str(fc_df['date'].iloc[0]) if fc_df is not None and len(fc_df) > 0 else today.strftime('%Y-%m-%d')

    if fc_df is not None and len(fc_df) > 0:
        # entry_date에 해당하는 예측값 우선, 없으면 fc_df 첫 번째 값
        fc_match = fc_df[fc_df['date'] == entry_date_str]
        entry_pred = round(float(fc_match['forecast_price'].iloc[0]), 2) if not fc_match.empty \
                     else round(float(fc_df['forecast_price'].iloc[0]), 2)

        entry_actual = entry_error = entry_error_pct = None
        entry_ts = pd.Timestamp(entry_date_str)
        if entry_ts in feature_df.index:
            try:
                entry_actual    = round(float(feature_df.loc[entry_ts, 'WTI']), 2)
                entry_error     = round(entry_actual - entry_pred, 2)
                entry_error_pct = round((entry_actual - entry_pred) / entry_actual * 100, 2)
            except Exception:
                pass

        xgb_pred_v = None
        try:
            model  = xg['model']
            scaler = xg['scaler']
            feats  = xg['features']
            last   = feature_df[feats].iloc[-1:].fillna(0)
            xgb_pred_v = round(float(model.predict(scaler.transform(last))[0]), 5)
        except Exception:
            pass

        new_entry = {
            'date':            entry_date_str,
            'sarimax_pred':    entry_pred,
            'actual_price':    entry_actual,
            'price_error':     entry_error,
            'price_error_pct': entry_error_pct,
            'xgb_pred_vol':    xgb_pred_v,
            'actual_vol_5d':   None,
            'vol_error':       None,
            'type':            'live',
        }

        # 같은 날짜 entry가 있으면 최신 실행값으로 덮어쓰기
        today_idx = next((i for i, r in enumerate(live_rows) if r.get('date') == entry_date_str), None)
        if today_idx is not None:
            live_rows[today_idx] = new_entry
        else:
            live_rows.append(new_entry)

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

def send_risk_alert(risk_signal: dict, fc_df) -> bool:
    """HIGH/CRITICAL 리스크 시 이메일 알림 발송. 설정 미비 시 조용히 스킵."""
    if not SMTP_USER or not SMTP_PASSWORD or not ALERT_TO:
        log.debug("이메일 알림 미설정 (SMTP_USER/SMTP_PASSWORD/ALERT_TO 환경변수 필요)")
        return False

    level = risk_signal.get('risk_level', '')
    if level not in ('HIGH', 'CRITICAL'):
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        r       = RISK_LEVELS[level]
        wti     = risk_signal.get('wti_price', 0)
        score   = risk_signal.get('risk_score', 0)
        vol     = risk_signal.get('volatility_5d', 0) * 100
        sent    = risk_signal.get('news_sentiment', 0)
        geo     = '활성 ⚠' if risk_signal.get('geopolitical_alert') else '없음'
        today   = datetime.now().strftime('%Y-%m-%d %H:%M')

        fc_lines = ""
        if fc_df is not None and len(fc_df) > 0:
            fc_lines = "\n".join(
                f"  {row['date']}  ${row['forecast_price']:.2f}"
                for _, row in fc_df.head(3).iterrows()
            )

        body = f"""
[유가 리스크 시스템] {r['emoji']} {level} 경보 — {today}

━━━━━━━━━━━━━━━━━━━━━━━━━
리스크 레벨  : {r['emoji']} {level} ({r['label']})
리스크 점수  : {score:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━
WTI 현재가   : ${wti:.2f} / bbl
5일 변동성   : {vol:.2f}%
뉴스 감성    : {sent:+.4f}
지정학 경보  : {geo}

▶ 향후 3일 예측
{fc_lines}
━━━━━━━━━━━━━━━━━━━━━━━━━
국제 유가 리스크 예측 시스템 MVP
""".strip()

        msg = MIMEMultipart()
        msg['From']    = SMTP_USER
        msg['To']      = ALERT_TO
        msg['Subject'] = f"[유가 리스크] {r['emoji']} {level} 경보 — WTI ${wti:.2f}"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_TO, msg.as_string())

        log.info(f"    📧 리스크 알림 이메일 발송 완료 → {ALERT_TO}")
        return True

    except Exception as exc:
        log.warning(f"    이메일 발송 실패: {exc}")
        return False


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
    n_neg     = float(row.get('news_count_neg', 0.0))
    n_pos     = float(row.get('news_count_pos', 0.0))
    extreme_n = float(row.get('extreme_neg_news', 0.0))
    sent_mag  = float(row.get('sentiment_magnitude', 0.0))

    vol_ratio     = vol / (hist_vol_75 + 1e-8)
    # 부정 기사 수 기반 증폭 (긍정 기사로 상쇄)
    news_amp      = 1 + min((n_neg - n_pos * 0.5) / 8, 1.0)
    news_amp      = max(news_amp, 1.0)
    geo_amp       = 1.35 if geo > 0.5 else 1.0
    # 부정 감성 증폭 + 극단 감성 시 추가 10%
    sentiment_amp = 1 + max(-sentiment, 0) * 0.5 + extreme_n * 0.1

    risk_score    = vol_ratio * news_amp * geo_amp * sentiment_amp
    directional   = mom + sentiment * 0.4 + bb * 0.2

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

    # 리스크 히스토리 누적 저장
    _hist_path = OUTPUT_DIR / 'risk_history.csv'
    _hist_row  = pd.DataFrame([{
        'date':       signal['date'],
        'risk_level': level,
        'risk_score': signal['risk_score'],
        'wti_price':  signal['wti_price'],
        'volatility': signal['volatility_5d'],
    }])
    if _hist_path.exists():
        _hist = pd.read_csv(_hist_path)
        _hist = pd.concat([_hist, _hist_row], ignore_index=True)
        _hist = _hist.drop_duplicates(subset=['date'], keep='last').sort_values('date').tail(365)
    else:
        _hist = _hist_row
    _hist.to_csv(_hist_path, index=False)
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

_KW_TRANSLATE = {
    # 원자재/에너지
    'oil':'원유', 'crude':'원유', 'brent':'브렌트', 'wti':'WTI',
    'petroleum':'석유', 'barrel':'배럴', 'barrels':'배럴',
    'gas':'가스', 'lng':'LNG', 'lpg':'LPG', 'natural':'천연',
    'energy':'에너지', 'fuel':'연료', 'fuels':'연료',
    'refinery':'정유', 'refining':'정제', 'pipeline':'파이프라인',
    'shale':'셰일', 'offshore':'해양', 'rig':'굴착기',
    'fossil':'화석연료', 'coal':'석탄', 'nuclear':'핵/원자력',
    # 국가/기관
    'opec':'OPEC', 'opec+':'OPEC+', 'iea':'IEA', 'eia':'EIA',
    'russia':'러시아', 'russian':'러시아', 'kremlin':'크렘린',
    'saudi':'사우디', 'arabia':'아라비아', 'aramco':'아람코',
    'iran':'이란', 'iranian':'이란', 'iraq':'이라크', 'iraqi':'이라크',
    'china':'중국', 'chinese':'중국', 'beijing':'베이징',
    'usa':'미국', 'america':'미국', 'american':'미국',
    'uae':'UAE', 'emirates':'에미리트',
    'venezuela':'베네수엘라', 'nigeria':'나이지리아', 'angola':'앙골라',
    'kuwait':'쿠웨이트', 'libya':'리비아', 'algeria':'알제리',
    'qatar':'카타르', 'mexico':'멕시코', 'canada':'캐나다',
    'australia':'호주', 'norway':'노르웨이', 'uk':'영국', 'britain':'영국',
    'europe':'유럽', 'european':'유럽', 'india':'인도', 'japan':'일본',
    'korea':'한국', 'middle':'중동',
    # 지정학/안보
    'war':'전쟁', 'sanctions':'제재', 'sanction':'제재',
    'conflict':'분쟁', 'crisis':'위기', 'crises':'위기',
    'attack':'공격', 'attacks':'공격', 'military':'군사',
    'ukraine':'우크라이나', 'ukrainian':'우크라이나',
    'israel':'이스라엘', 'israeli':'이스라엘',
    'gaza':'가자', 'hamas':'하마스', 'iran':'이란',
    'geopolitical':'지정학', 'tension':'긴장', 'tensions':'긴장',
    'ceasefire':'휴전', 'invasion':'침공', 'missile':'미사일',
    'coup':'쿠데타', 'protest':'시위', 'strike':'파업/공습',
    'terrorism':'테러', 'drone':'드론',
    # 수급/시장
    'supply':'공급', 'supplies':'공급', 'demand':'수요',
    'production':'생산', 'output':'생산량', 'capacity':'생산능력',
    'inventory':'재고', 'inventories':'재고', 'stockpile':'비축',
    'reserve':'매장량', 'reserves':'매장량',
    'market':'시장', 'markets':'시장',
    'price':'가격', 'prices':'가격',
    'rally':'반등', 'slump':'급락', 'surge':'급등', 'drop':'급락',
    'cut':'감산', 'cuts':'감산', 'increase':'증산', 'quota':'쿼터',
    'export':'수출', 'exports':'수출', 'import':'수입', 'imports':'수입',
    'trade':'무역', 'deal':'협상', 'agreement':'합의',
    'rise':'상승', 'fall':'하락', 'decline':'하락', 'gain':'상승',
    'record':'기록', 'high':'고점', 'low':'저점', 'peak':'정점',
    'volatility':'변동성', 'swing':'급변',
    # 거시경제/금융
    'inflation':'인플레이션', 'recession':'경기침체',
    'gdp':'GDP', 'growth':'성장', 'economy':'경제', 'economic':'경제',
    'dollar':'달러', 'fed':'연준', 'federal':'연방', 'reserve':'준비/연준',
    'interest':'금리', 'rate':'금리', 'rates':'금리',
    'bank':'은행', 'banking':'금융', 'financial':'금융',
    'investment':'투자', 'investor':'투자자',
    'deficit':'적자', 'debt':'부채', 'currency':'통화',
    # 기후/에너지전환
    'climate':'기후', 'global':'세계', 'warming':'온난화',
    'renewable':'재생에너지', 'solar':'태양광', 'wind':'풍력',
    'electric':'전기차', 'carbon':'탄소', 'emission':'탄소배출',
    'emissions':'탄소배출', 'green':'친환경', 'clean':'청정',
    'transition':'전환', 'net':'탄소중립', 'zero':'제로',
    'cop':'기후회의', 'paris':'파리협정',
    # 주요 인물
    'trump':'트럼프', 'biden':'바이든', 'putin':'푸틴',
    'xi':'시진핑', 'mbs':'빈살만', 'powell':'파월',
    'zelensky':'젤렌스키', 'netanyahu':'네타냐후',
    # 기타 빈출 관련어
    'world':'세계', 'global':'세계', 'international':'국제',
    'plan':'계획', 'plans':'계획', 'policy':'정책', 'policies':'정책',
    'change':'변화', 'changes':'변화', 'reform':'개혁',
    'power':'전력/파워', 'grid':'전력망', 'storage':'저장',
    'hit':'타격', 'impact':'영향', 'effect':'효과',
    'security':'안보', 'risk':'리스크', 'threat':'위협',
    'forecast':'전망', 'outlook':'전망', 'prediction':'예측',
    'report':'보고서', 'data':'데이터', 'analysis':'분석',
    'opec+':'OPEC+', 'brexit':'브렉시트', 'covid':'코로나19',
    'pandemic':'팬데믹', 'recovery':'회복',
    'food':'식품', 'water':'수자원', 'war':'전쟁',
    # 지역명 추가
    'england':'영국', 'scotland':'스코틀랜드', 'wales':'웨일스',
    # 산업/기타
    'industry':'산업', 'industries':'산업', 'sector':'부문',
    # 추가 번역
    'government':'정부', 'governments':'정부',
    'election':'선거', 'elections':'선거',
    'warns':'경고', 'warning':'경고', 'warnings':'경고', 'fears':'우려',
    'labour':'노동당', 'labor':'노동', 'workers':'노동자',
    'australian':'호주', 'australia':'호주',
    'tax':'세금', 'taxes':'세금', 'taxation':'과세',
    'hits':'타격', 'hit':'타격',
    'sea':'해상', 'ocean':'해양', 'maritime':'해운',
    'london':'런던', 'paris':'파리', 'berlin':'베를린', 'tokyo':'도쿄',
    'washington':'워싱턴', 'moscow':'모스크바', 'riyadh':'리야드',
    'north':'북한/북쪽', 'south':'남쪽', 'east':'동쪽', 'west':'서방',
    'markets':'시장', 'economy':'경제', 'economies':'경제',
    'risk':'리스크', 'risks':'리스크',
    'emissions':'탄소배출', 'emission':'탄소배출',
}


def generate_wordcloud(kw_df: pd.DataFrame):
    """위기 키워드 워드클라우드 — 영어 키워드를 한글로 변환하여 표시"""
    log.info("[8/9] 워드클라우드 생성 중...")
    crisis_set = set(kw_df[kw_df['is_crisis_word']]['keyword'])

    # 영어 → 한글 변환 (매핑 없으면 원어 유지)
    kw_ko = kw_df.copy()
    kw_ko['keyword_display'] = kw_ko['keyword'].apply(
        lambda w: _KW_TRANSLATE.get(w.lower(), w)
    )
    # 한글 변환된 키워드끼리 빈도 합산
    ko_freq = kw_ko.groupby('keyword_display')['count'].sum()
    crisis_ko = set(
        kw_ko[kw_ko['is_crisis_word']]['keyword_display']
    )

    if _WC:
        freq = ko_freq.to_dict()

        def color_fn(word, **_):
            if word in crisis_ko or word.lower() in crisis_set:
                return f"hsl({np.random.randint(0,25)}, 90%, {np.random.randint(42,58)}%)"
            return f"hsl({np.random.randint(195,245)}, 65%, {np.random.randint(45,62)}%)"

        # 한글 폰트 경로 탐색 (WordCloud용)
        import matplotlib.font_manager as _fm
        _ko_candidates = ['Malgun Gothic', 'AppleGothic', 'NanumGothic']
        _ko_path = None
        for _n in _ko_candidates:
            _hits = [f.fname for f in _fm.fontManager.ttflist if f.name == _n]
            if _hits:
                _ko_path = _hits[0]
                break
        if _ko_path is None:
            _p = Path('C:/Windows/Fonts/malgun.ttf')
            if _p.exists():
                _ko_path = str(_p)

        wc_kwargs = dict(
            width=1400, height=700, background_color='#12141a',
            color_func=color_fn, max_words=80, prefer_horizontal=0.68,
            min_font_size=11, max_font_size=130, random_state=42,
        )
        if _ko_path:
            wc_kwargs['font_path'] = _ko_path
        wc = _WC_Class(**wc_kwargs)
        wc.generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#12141a')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('유가 시장 뉴스 키워드  (🔴 위기어  🔵 일반어)',
                     color='#e0e0e0', fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'wordcloud.png', dpi=150, bbox_inches='tight', facecolor='#12141a')
        plt.close()
    else:
        # ── 바 차트 대체 (한글 키워드 사용)
        top20 = kw_ko.head(20).iloc[::-1]
        colors = ['#e74c3c' if x else '#3d85c8' for x in top20['is_crisis_word']]

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#12141a')
        ax.set_facecolor('#1a1d24')
        ax.barh(top20['keyword_display'], top20['count'], color=colors)
        ax.set_xlabel('빈도', color='#ccc')
        ax.set_title('주요 위기 키워드 Top 20', color='white', fontsize=13)
        ax.tick_params(colors='#ccc')
        for sp in ax.spines.values(): sp.set_color('#333')
        p1 = mpatches.Patch(color='#e74c3c', label='위기 키워드')
        p2 = mpatches.Patch(color='#3d85c8', label='일반 키워드')
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

    api_status = {}  # API별 수집 성공 여부 추적

    # yfinance
    try:
        price_df = fetch_data(start_date, end_date)
        _is_dummy = (len(price_df) < 100 and price_df.index[0].year < 2010)
        api_status['yfinance'] = '❌ 더미' if _is_dummy else '✅ 정상'
    except Exception as e:
        price_df = fetch_data(start_date, end_date)
        api_status['yfinance'] = f'❌ 오류'

    # FRED
    api_status['FRED'] = '✅ 정상' if (_FRED and FRED_API_KEY) else '❌ 미설정'

    # Guardian 뉴스
    try:
        news_df = fetch_news()
        api_status['Guardian'] = '✅ 정상' if len(news_df) > 10 else '⚠️ 부족'
    except Exception:
        news_df = fetch_news()
        api_status['Guardian'] = '❌ 오류'

    # EIA
    api_status['EIA'] = '✅ 정상' if EIA_API_KEY else '❌ 미설정'

    feature_df, full_df  = build_features(price_df, news_df)
    model_results, _     = train_models(feature_df)

    # forecast_7days.csv 덮어쓰기 전에 이전 예측 로드 (gap-fill용)
    prev_fc_df = None
    _fc_csv = OUTPUT_DIR / 'forecast_7days.csv'
    if _fc_csv.exists():
        try:
            prev_fc_df = pd.read_csv(_fc_csv)
        except Exception:
            pass

    fc_df                = forecast_next_7days(model_results, feature_df, full_df)
    save_prediction_log(model_results, feature_df, fc_df, prev_fc_df, full_df)
    risk_signal          = classify_risk(feature_df, full_df)
    send_risk_alert(risk_signal, fc_df)
    kw_df                = extract_crisis_keywords(news_df)
    generate_wordcloud(kw_df)
    plot_oil_forecast(feature_df, fc_df, risk_signal)

    # ── 마지막 실행 시간 + API 상태 기록
    import json as _json
    _run_meta = {
        'last_run':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_through': feature_df.index[-1].strftime('%Y-%m-%d'),
        'n_live':      int((pd.read_csv(PRED_LOG_FILE)['type'] == 'live').sum()) if PRED_LOG_FILE.exists() else 0,
        'api_status':  api_status,
    }
    with open(OUTPUT_DIR / 'run_meta.json', 'w') as _f:
        _json.dump(_run_meta, _f)

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
def schedule_daily(hour: int = 6, minute: int = 0):
    """APScheduler로 매일 지정 시각에 파이프라인 자동 실행 (KST 기준)"""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        scheduler = BlockingScheduler(timezone='Asia/Seoul')
        scheduler.add_job(run_pipeline, 'cron', hour=hour, minute=minute)
        log.info(f"스케줄러 시작: 매일 {hour:02d}:{minute:02d}(KST) 자동 실행 — Ctrl+C로 중단")
        run_pipeline()   # 첫 실행은 즉시
        scheduler.start()
    except ImportError:
        log.error("APScheduler 미설치: pip install apscheduler")
    except (KeyboardInterrupt, SystemExit):
        log.info("스케줄러 종료")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='국제 유가 리스크 예측 시스템')
    parser.add_argument('--schedule', action='store_true',
                        help='매일 자동 실행 모드 (apscheduler 필요)')
    parser.add_argument('--hour',   type=int, default=6,
                        help='자동 실행 시각 0~23 (기본: 6)')
    parser.add_argument('--minute', type=int, default=0,
                        help='자동 실행 분  0~59 (기본: 0)')
    args = parser.parse_args()

    if args.schedule:
        schedule_daily(args.hour, args.minute)
    else:
        run_pipeline()
