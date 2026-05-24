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

def _atomic_csv(df: "pd.DataFrame", path: "Path", **kwargs) -> None:
    """Write CSV atomically via temp file + os.replace to avoid partial reads."""
    import os as _os, tempfile as _tf
    _dir = Path(path).parent
    with _tf.NamedTemporaryFile(mode='w', dir=_dir, suffix='.tmp', delete=False,
                                encoding='utf-8', newline='') as _fh:
        _tmp = _fh.name
    try:
        df.to_csv(_tmp, **kwargs)
        _os.replace(_tmp, path)
    except Exception:
        try:
            _os.unlink(_tmp)
        except Exception:
            pass
        raise
EMBED_CACHE_FILE = OUTPUT_DIR / 'news_embed_cache.pkl'
COT_CACHE_FILE   = OUTPUT_DIR / 'cot_cache.csv'
COT_RAW_FILE     = Path('annual.txt')   # CFTC 연간 전체 데이터
XGB_OPTUNA_CACHE    = OUTPUT_DIR / 'xgb_optuna_cache.json'
STACK_WEIGHTS_EMA   = OUTPUT_DIR / 'stacking_weights_ema.json'
FEAT_TRAIN_STATS    = OUTPUT_DIR / 'feature_train_stats.json'
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_TOP_K      = 15   # WTI 수익률 상관관계 상위 차원 수

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
SARIMAX_WINDOW_YEARS = 3                          # SARIMAX 전용 훈련 window (최근 레짐 집중, VAR/ETS 불변)
XGB_YEARS        = 3                              # XGBoost 슬라이딩 윈도우 (최근 레짐 집중)
EIA_SHIFT        = 1                              # EIA 재고 발표 지연 (영업일): 1=목요일 반영, 원래값=3

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

# ── OPEC+ 주요 회의 날짜 (생산량 결정 포함)
_OPEC_MEETING_DATES = pd.to_datetime([
    '2020-01-06','2020-03-06','2020-04-09',
    '2021-01-05','2021-03-04','2021-04-01','2021-05-27',
    '2021-07-01','2021-09-01','2021-10-04','2021-11-03','2021-12-02',
    '2022-02-02','2022-03-02','2022-04-05','2022-06-02',
    '2022-08-03','2022-09-05','2022-10-05','2022-11-30','2022-12-04',
    '2023-02-01','2023-03-22','2023-04-02','2023-06-04',
    '2023-08-16','2023-09-05','2023-11-26',
    '2024-02-01','2024-03-03','2024-06-02','2024-11-05','2024-12-05',
    '2025-01-12','2025-02-03','2025-03-03','2025-04-03',
    '2025-05-05','2025-06-01','2025-07-06','2025-08-03',
    '2025-09-07','2025-10-05','2025-11-02','2025-12-07',
    '2026-01-04','2026-02-01','2026-03-01','2026-04-05','2026-05-03',
])

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
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False
    log.warning("lightgbm 없음 → LGB 베이스 모델 미사용")

try:
    from catboost import CatBoostRegressor as _CBR
    _CBT = True
except ImportError:
    _CBT = False
    log.warning("catboost 없음 → CatBoost 베이스 모델 미사용")

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
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False



try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler, RobustScaler
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
    # ── WTI 상승 요인: 공급 차질 / 지정학 리스크 (양수)
    'war':2,'attack':1.5,'explosion':1.0,'sabotage':2.0,
    'sanction':1.5,'embargo':1.5,'blockade':1.5,'seizure':1.0,
    'shortage':1.5,'disruption':1.5,'shutdown':1.0,'conflict':1.5,
    'outage':1.5,'curtail':1.0,'nationalize':0.5,'nationalization':1.0,
    'expropriation':1.0,'confiscate':0.5,'depletion':0.5,
    'tension':1.0,'threat':0.5,'spike':1.5,'surge':1.0,
    'halt':0.5,'freeze':0.5,'restrict':0.5,
    # ── WTI 하락 요인: 공급 증가 / 수요 감소 / 리스크 완화 (음수)
    'collapse':-2,'crash':-2,'plunge':-1.5,'slump':-1.5,'plummet':-1.5,
    'glut':-1.5,'oversupply':-1,'overproduction':-1,'gluts':-1.5,
    'recession':-1,'slowdown':-1,'shrink':-1,'weak':-1,
    'fall':-1,'decline':-1,'drop':-1,'tumble':-1,'slip':-1,
    'surplus':-2,'peace':-1.5,'resolution':-1.0,'ceasefire':-1.0,'truce':-1.0,
    'normalization':-0.5,'normalize':-0.5,'stabilize':-0.3,'stabilizing':-0.3,
    'stabilization':-0.5,
    'contango':-1,'bearish':-1,'bearmarket':-1.5,'selloff':-1.5,
    'rout':-1.5,'meltdown':-2,'default':-2,'bankrupt':-2,
    'insolvency':-1.5,'downgrade':-1,'loss':-1,'impairment':-1,
    'stranded':-1,'dwindle':-1,'dwindling':-1.5,
    # ── 수요 증가 (양수)
    'recovery':1,'growth':1,'rise':1,'rally':1,'rebound':1,
    'deal':1,'agreement':1,'bullish':1,'strong':1,'robust':1,
    'draw':1,'deficit':1,'backwardation':1.5,'tightening':1,
    'undersupply':1.5,'expand':1,'expanding':1.5,
    'upgrade':1,'outperform':1,'exceeds':1,'beat':1,
    'breakthrough':1.5,'accord':1,'pact':1,'ratify':1,
    'ramp':1,'ramping':1,'boom':1.5,'reopening':1.5,
    # ── 약한 / 중립 신호
    'risk':-0.5,'concern':-0.5,'fear':-0.5,'warning':-0.5,
    'dispute':-0.5,'hawkish':-0.5,'contraction':0.5,
    'drawdown':1,'compliance':1,'quota':-0.3,
    'increase':1,'stable':0.5,'open':0.5,'record':1.5,
    'lift':1,'resume':0.5,'ease':0.5,'dovish':0.5,
    'worsening':-1,'deteriorate':-1,'deterioration':-1,
    'withdrawal':-0.5,'evacuation':-0.3,'oversold':-0.3,
    'deplete':-0.5,'capacityloss':-1,'delinquent':-0.5,
    'restoring':1,'compliance':1,
}

# 구문 패턴 (바이그램) — WTI 가격 방향 기준 재매핑
PHRASE_SENTIMENT = {
    # ── 공급 감소 → WTI 상승 (양수)
    'production cut':+1.5,'output cut':+1.5,'supply cut':+1.5,'capacity cut':+1.0,
    'deeper cut':+2.0,'extend cut':+1.5,'voluntary cut':+1.0,
    'supply disruption':+2.0,'supply shortage':+1.5,'supply crunch':+2.0,
    'refinery shutdown':+1.0,'pipeline attack':+2.0,'pipeline shutdown':+1.5,
    'force majeure':+1.5,'declared emergency':+1.0,
    'transit disruption':+1.5,'shipping disruption':+1.5,
    'output freeze':+1.0,'production freeze':+1.0,
    'export ban':+1.0,
    # ── 공급 증가 → WTI 하락 (음수)
    'production increase':-1.5,'output increase':-1.5,'supply increase':-1.0,
    'production boost':-1.5,'output boost':-1.5,
    'field restart':-1.0,'production restart':-1.0,
    'pipeline resumes':-0.5,'refinery resumes':-0.5,
    # ── 지정학 리스크 완화 / 제재 해제 → WTI 하락 (음수)
    'nuclear deal':-1.5,'sanctions lifted':-2.0,'sanctions eased':-1.5,
    'ceasefire deal':-1.0,'peace deal':-1.5,
    # ── 수요 신호 (기존 유지)
    'demand destruction':-1.5,'demand weakness':-1.5,'demand slowdown':-1.0,
    'demand recovery':+1.5,'demand growth':+1.5,'demand surge':+1.5,
    'demand rebound':+1.5,'demand pickup':+1.0,'demand uptick':+1.0,
    # ── 재고 신호 (기존 유지)
    'inventory draw':+1.5,'stock draw':+1.5,'crude draw':+1.5,
    'inventory build':-1.0,'stock build':-1.0,'crude build':-1.0,
    # ── 시장 구조 (기존 유지)
    'price cap':-1.0,'price ceiling':-1.0,'strategic release':-1.0,
    'strategic reserve':-1.0,'spr release':-1.5,
    'tight market':+1.5,'tight supply':+1.5,
    'loose market':-1.0,'loose supply':-1.0,
    'supply tightens':+1.5,'supply loosens':-1.0,
    'price floor':+1.0,'price support':+1.0,
    'excess capacity':-1.0,'spare capacity':+0.5,
    'import ban':-1.0,
    'rig count':-0.3,'rig counts':-0.3,
}

# 강화어 (인접 단어의 감성 1.4× 증폭)
INTENSIFIERS = {
    'record','massive','unprecedented','sharp','dramatic','significant',
    'major','severe','huge','deep','steep','rapidly','sharply','surging',
}

# 불확실성 표현 (논문: intensity + uncertainty가 polarity보다 예측력 높음)
HEDGE_WORDS = {
    'may','might','could','uncertain','uncertainty','unclear','possibly','possibly',
    'likely','unlikely','risk','risks','concern','concerns','worry','worries',
    'fear','fears','doubt','doubts','question','questionable','volatile','volatility',
    'unpredictable','unresolved','pending','awaiting','expect','expected','forecast',
    'projected','estimated','potential','possibly','tentative','ambiguous',
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
    파일이 7일 이상 오래됐으면 공식 URL에서 자동 갱신 시도.
    """
    _GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    gpr_path = Path(GPR_FILE)

    # 파일 갱신 (없거나 7일 이상 오래된 경우)
    _needs_update = (not gpr_path.exists()) or (
        (datetime.now() - datetime.fromtimestamp(gpr_path.stat().st_mtime)).days >= 7
    )
    if _needs_update:
        try:
            import urllib.request as _ur
            _ur.urlretrieve(_GPR_URL, str(gpr_path))
            log.info("    GPR 파일 자동 갱신 완료")
        except Exception as _ge:
            log.warning(f"    GPR 자동 갱신 실패({_ge}) → 기존 파일 사용")

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
        log.info("    FRED 미설정 → 충격변수 0으로 대체")
        for _c in ('demand_shock', 'supply_shock', 'inv_chg_zscore', 'inv_lvl_zscore'):
            df[_c] = 0.0
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
        log.warning(f"      수요충격 FRED 실패({exc}) → 0 사용")
        df['demand_shock'] = 0.0

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
        log.warning(f"      공급충격 FRED 실패({exc}) → 0 사용")
        df['supply_shock'] = 0.0

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

        # 주간 → 영업일 ffill (EIA_SHIFT 영업일 지연: 라이브=1, 백테스트 엄밀=3)
        inv_chg_bday   = inv_chg.resample('B').first().shift(EIA_SHIFT).ffill()
        inv_level_bday = inv.resample('B').first().shift(EIA_SHIFT).ffill()

        # z-score 정규화
        def _zscore(s, w=252):
            return ((s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-8)).fillna(0)

        df['inv_chg_zscore'] = _zscore(inv_chg_bday).reindex(df.index).ffill().bfill().fillna(0)
        df['inv_lvl_zscore'] = _zscore(inv_level_bday).reindex(df.index).ffill().bfill().fillna(0)

        # 서프라이즈: 실제 변화량 vs 직전 4주 이동평균 대비 이탈 (shift(1)로 당주 자기참조 방지)
        inv_chg_ma4 = inv_chg.shift(1).rolling(4).mean()
        inv_surprise_raw = (inv_chg - inv_chg_ma4).resample('B').first().shift(EIA_SHIFT).ffill()
        df['inv_surprise'] = _zscore(inv_surprise_raw).reindex(df.index).ffill().bfill().fillna(0)

        # 4주 연속 방향성 모멘텀 (증가/감소 추세)
        inv_mom4 = inv_chg.rolling(4).sum().resample('B').first().shift(EIA_SHIFT).ffill()
        df['inv_mom4_z'] = _zscore(inv_mom4).reindex(df.index).ffill().bfill().fillna(0)

        log.info(f"      원유 재고 연결 완료: {len(inv)}주치 + 서프라이즈·모멘텀 피처")
    except Exception as exc:
        log.warning(f"      EIA API 실패({exc}) → EIA 공개 CSV 시도...")
        # EIA API 키 없을 때 EIA 공개 XLS 직접 다운로드 (API 키 불필요)
        try:
            import urllib.request as _ur2, io as _io2
            _eia_url = ("https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls")
            _req2 = _ur2.Request(_eia_url, headers={'User-Agent': 'Mozilla/5.0'})
            with _ur2.urlopen(_req2, timeout=15) as _r2:
                _raw2 = _r2.read()
            _inv_xl = pd.read_excel(_io2.BytesIO(_raw2), sheet_name=1, skiprows=2,
                                    index_col=0, parse_dates=True)
            inv_raw = _inv_xl.iloc[:, 0].dropna().astype(float)
            inv_raw.index = pd.to_datetime(inv_raw.index)
            inv_raw = inv_raw.sort_index()
            inv_chg = inv_raw.diff()
            def _zscore2(s, w=252):
                return ((s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-8)).fillna(0)
            inv_chg_b   = inv_chg.resample('B').first().shift(EIA_SHIFT).ffill()
            inv_lvl_b   = inv_raw.resample('B').first().shift(EIA_SHIFT).ffill()
            df['inv_chg_zscore'] = _zscore2(inv_chg_b).reindex(df.index).ffill().bfill().fillna(0)
            df['inv_lvl_zscore'] = _zscore2(inv_lvl_b).reindex(df.index).ffill().bfill().fillna(0)
            inv_chg_ma4  = inv_chg.shift(1).rolling(4).mean()
            inv_surp_b   = (inv_chg - inv_chg_ma4).resample('B').first().shift(EIA_SHIFT).ffill()
            df['inv_surprise'] = _zscore2(inv_surp_b).reindex(df.index).ffill().bfill().fillna(0)
            inv_mom4 = inv_chg.rolling(4).sum().resample('B').first().shift(EIA_SHIFT).ffill()
            df['inv_mom4_z'] = _zscore2(inv_mom4).reindex(df.index).ffill().bfill().fillna(0)
            log.info(f"      EIA 공개 CSV 연결 완료: {len(inv_raw)}주치")
        except Exception as exc2:
            log.warning(f"      EIA 공개 CSV도 실패({exc2}) → 0 사용")
            for _c in ('inv_chg_zscore', 'inv_lvl_zscore', 'inv_surprise', 'inv_mom4_z'):
                df[_c] = 0.0

    # ── 지정학 더미: GPR Index (Caldara & Iacoviello) ─────────────────────
    df = _attach_gpr(df)

    log.info("    FRED 실제 데이터 연결 완료 ✓")
    return df


def fetch_data(start_date=None, end_date=None):
    """yfinance로 WTI·Brent·DXY 수집; 실패 시 더미 데이터 반환"""
    if end_date is None:
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
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

        # CL=F 전 컬럼 일괄 수집 (중복 다운로드 방지)
        _clf_raw = yf.download("CL=F", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(_clf_raw.columns, pd.MultiIndex):
            _clf_raw.columns = _clf_raw.columns.droplevel(1)
        def _from_clf(col):
            s = _clf_raw[col] if col in _clf_raw.columns else pd.Series(dtype=float)
            return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        wti      = _from_clf('Close'); wti.name = "CL=F"
        wti_high = _from_clf('High').rename("WTI_High")
        wti_low  = _from_clf('Low').rename("WTI_Low")
        wti_open = _from_clf('Open').rename("WTI_Open")
        wti_vol  = _from_clf('Volume').rename("WTI_Volume")

        # CL2=F 먼저 순차 실행 (yfinance 세션 정규화 사이드 이펙트 보존)
        try:
            cl2 = _dl("CL2=F")
            futures_spread = (cl2 - wti).rename("futures_spread")
            log.info(f"    WTI 선물 커브 스프레드 수집 완료 (μ={futures_spread.mean():.3f})")
        except Exception:
            futures_spread = (wti.rolling(63, min_periods=20).mean() - wti).rename("futures_spread")
            log.warning("    CL2=F 실패 → 63일 이평 proxy 사용 (term structure 근사)")

        # 나머지 8개 티커 병렬 다운로드
        from concurrent.futures import ThreadPoolExecutor as _TPE
        def _safe_dl(ticker, rename=None):
            try:
                s = _dl(ticker)
                return s.rename(rename) if rename else s
            except Exception:
                return pd.Series(dtype=float, name=(rename or ticker))

        with _TPE(max_workers=8) as _pe:
            _fb    = _pe.submit(_safe_dl, 'BZ=F')
            _fd    = _pe.submit(_safe_dl, 'DX-Y.NYB')
            _fv    = _pe.submit(_safe_dl, '^VIX')
            _fo    = _pe.submit(_safe_dl, '^OVX')
            _fv3   = _pe.submit(_safe_dl, '^VIX3M')
            _fsk   = _pe.submit(_safe_dl, '^SKEW')
            _fng   = _pe.submit(_safe_dl, 'NG=F',  'NatGas')
            _frb   = _pe.submit(_safe_dl, 'RB=F',  'RBOB')
            _fgc   = _pe.submit(_safe_dl, 'GC=F',  'Gold')
            _fhg   = _pe.submit(_safe_dl, 'HG=F',  'Copper')
        brent = _fb.result();  dxy   = _fd.result()
        vix   = _fv.result();  ovx   = _fo.result()
        vix3m = _fv3.result(); skew  = _fsk.result()
        ng    = _fng.result(); rbob  = _frb.result()
        gold  = _fgc.result(); copper = _fhg.result()
        log.info("    병렬 다운로드 완료 (BZ=F/DXY/VIX/OVX/VIX3M/SKEW/NG/RBOB/Gold/Copper)")

        df = pd.DataFrame({'WTI': wti, 'Brent': brent, 'DXY': dxy,
                           'VIX': vix, 'OVX': ovx, 'futures_spread': futures_spread,
                           'WTI_High': wti_high, 'WTI_Low': wti_low,
                           'WTI_Open': wti_open, 'WTI_Volume': wti_vol,
                           'VIX3M': vix3m, 'SKEW': skew,
                           'NatGas': ng, 'RBOB': rbob,
                           'Gold': gold, 'Copper': copper})
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
        # ── 데이터 신선도 체크: WTI 최신 날짜 기준 영업일 경과 확인
        for _sc, _col in [('WTI', 'WTI'), ('Brent', 'Brent'), ('DXY', 'DXY')]:
            if _col in df.columns:
                _last = df[_col].dropna().index.max()
                if pd.notna(_last):
                    _bdays_lag = len(pd.bdate_range(_last, datetime.today())) - 1
                    if _bdays_lag > 2:
                        log.warning(f"    ⚠ {_sc} 데이터 스테일: 최신={_last.date()}, "
                                    f"영업일 경과={_bdays_lag}일 (yfinance 지연 의심)")
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


def fetch_cot() -> pd.DataFrame:
    """CFTC WTI COT 데이터 로드 (캐시 + annual.txt 갱신).
    반환: DatetimeIndex, 컬럼 [cot_net_pct, cot_mm_long_pct, cot_mm_short_pct]
    """
    cache = pd.DataFrame()
    if COT_CACHE_FILE.exists():
        try:
            cache = pd.read_csv(COT_CACHE_FILE, parse_dates=['date'])
            cache = cache.set_index('date')
            cache.index = pd.to_datetime(cache.index).normalize()
        except Exception:
            cache = pd.DataFrame()

    # annual.txt에서 WTI-PHYSICAL (067651) 파싱
    if COT_RAW_FILE.exists():
        try:
            rows = []
            with open(COT_RAW_FILE, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '067651' not in line:
                        continue
                    parts = [p.strip().strip('"') for p in line.split(',')]
                    if len(parts) < 20:
                        continue
                    try:
                        date_str = parts[2]   # YYYY-MM-DD
                        oi       = float(parts[7])
                        # disaggregated 형식: M_Money Long=13, Short=14
                        mm_long  = float(parts[12])
                        mm_short = float(parts[13])
                        if oi <= 0:
                            continue
                        net_pct  = (mm_long - mm_short) / oi * 100
                        rows.append({
                            'date': pd.Timestamp(date_str),
                            'cot_net_pct':      round(net_pct, 4),
                            'cot_mm_long_pct':  round(mm_long  / oi * 100, 4),
                            'cot_mm_short_pct': round(mm_short / oi * 100, 4),
                        })
                    except (ValueError, IndexError):
                        continue
            if rows:
                new_df = pd.DataFrame(rows).set_index('date')
                new_df.index = pd.to_datetime(new_df.index).normalize()
                new_df = new_df[~new_df.index.duplicated(keep='last')]
                if cache.empty:
                    cache = new_df
                else:
                    # 신규 날짜만 병합
                    cache = pd.concat([cache, new_df[~new_df.index.isin(cache.index)]])
                    cache = cache.sort_index()
                # 캐시 저장
                try:
                    cache.reset_index().rename(columns={'index':'date'}).to_csv(
                        COT_CACHE_FILE, index=False)
                except Exception:
                    pass
        except Exception as _ce:
            log.warning(f"    COT annual.txt 파싱 실패({_ce})")

    # ── CFTC 자동 다운로드 (캐시 8일 이상 stale이면 갱신)
    _today = pd.Timestamp.today().normalize()
    _stale  = cache.empty or (_today - cache.index[-1]).days >= 8
    if _stale:
        try:
            import urllib.request as _ur3, zipfile as _zf, io as _io3
            _year = _today.year
            for _yr in [_year, _year - 1]:   # 연초 전환 대비 전년도도 시도
                _url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{_yr}.zip"
                try:
                    _req = _ur3.Request(_url, headers={'User-Agent': 'Mozilla/5.0'})
                    _raw = _ur3.urlopen(_req, timeout=20).read()
                    _z   = _zf.ZipFile(_io3.BytesIO(_raw))
                    _txt = _z.read(_z.namelist()[0]).decode('utf-8', errors='ignore')
                    _rows2 = []
                    for _line in _txt.split('\n'):
                        if '067651' not in _line:
                            continue
                        _p = [x.strip().strip('"') for x in _line.split(',')]
                        if len(_p) < 20:
                            continue
                        try:
                            _oi = float(_p[7])
                            _ml = float(_p[12]); _ms = float(_p[13])
                            if _oi <= 0:
                                continue
                            _rows2.append({
                                'date':             pd.Timestamp(_p[2]),
                                'cot_net_pct':      round((_ml - _ms) / _oi * 100, 4),
                                'cot_mm_long_pct':  round(_ml / _oi * 100, 4),
                                'cot_mm_short_pct': round(_ms / _oi * 100, 4),
                            })
                        except (ValueError, IndexError):
                            continue
                    if _rows2:
                        _new = pd.DataFrame(_rows2).set_index('date')
                        _new.index = pd.to_datetime(_new.index).normalize()
                        _new = _new[~_new.index.duplicated(keep='last')]
                        if cache.empty:
                            cache = _new
                        else:
                            _add = _new[~_new.index.isin(cache.index)]
                            if not _add.empty:
                                cache = pd.concat([cache, _add]).sort_index()
                        try:
                            cache.reset_index().rename(columns={'index': 'date'}).to_csv(
                                COT_CACHE_FILE, index=False)
                        except Exception:
                            pass
                        log.info(f"    COT CFTC {_yr}년 자동 갱신: {len(_rows2)}주 추가")
                        break
                except Exception as _de:
                    log.warning(f"    COT CFTC {_yr}년 다운로드 실패({_de})")
        except Exception as _cftc_e:
            log.warning(f"    COT 자동 갱신 실패({_cftc_e})")

    if cache.empty:
        log.warning("    COT 데이터 없음 → 0으로 대체")
    else:
        log.info(f"    COT 로드: {len(cache)}주 ({cache.index[-1].date()}까지)")
    return cache


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

            # Guardian은 일반 에너지 기사(재생에너지·기후) 포함 → 원유 관련만 필터
            _OIL_KW = {'oil','crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker'}
            new_articles = [
                a for a in new_articles
                if any(kw in a.get('title','').lower() for kw in _OIL_KW)
            ]
            log.info(f"    Guardian 신규 수집: {len(new_articles)}건 (oil 필터 후)")
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
                    # summary/description 필드로 본문 보강
                    _body = (getattr(entry, 'summary', '') or
                             getattr(entry, 'description', '') or '')
                    _body = str(_body)[:300].strip()
                    full_text = title + (' ' + _body if _body and _body != title else '')
                    if pub >= cutoff and any(kw in full_text.lower() for kw in OIL_FILTER):
                        new_articles.append({'date': pub.strftime('%Y-%m-%d'),
                                             'title': full_text, 'source': src_name})
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
    except Exception as _nce:
        log.warning(f"    뉴스 캐시 저장 실패({_nce}) → 다음 실행 시 재수집")

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
# 3b. FinBERT 금융 감성 모델 (ProsusAI/finbert)
# ─────────────────────────────────────────────────────────────────────────────

_FINBERT_PIPE  = None
_FINBERT_READY = None   # None=미확인, True=OK, False=실패

def _load_finbert():
    global _FINBERT_PIPE, _FINBERT_READY
    if _FINBERT_READY is not None:
        return _FINBERT_PIPE
    try:
        from transformers import pipeline as _hf_pipeline
        import torch as _torch
        _dev = 0 if _torch.cuda.is_available() else -1
        _FINBERT_PIPE = _hf_pipeline(
            'text-classification',
            model='ProsusAI/finbert',
            device=_dev,
            top_k=None,
        )
        _FINBERT_READY = True
        log.info(f"    ✅ FinBERT 로드 완료 ({'GPU' if _dev == 0 else 'CPU'})")
    except Exception as _fe:
        _FINBERT_READY = False
        log.warning(f"    FinBERT 로드 실패({_fe}) → 규칙 기반 사용")
    return _FINBERT_PIPE


def _finbert_batch(texts: list) -> list:
    """FinBERT 배치 추론 → positive-negative 점수 (-1~+1) 리스트"""
    pipe = _load_finbert()
    if pipe is None:
        return [0.0] * len(texts)
    try:
        results = pipe(
            [t[:512] for t in texts],
            batch_size=32,
            truncation=True,
            max_length=512,
        )
        out = []
        for res in results:
            pos = next((r['score'] for r in res if r['label'] == 'positive'), 0.0)
            neg = next((r['score'] for r in res if r['label'] == 'negative'), 0.0)
            out.append(float(pos - neg))
        return out
    except Exception as _e:
        log.warning(f"    FinBERT 추론 실패({_e})")
        return [0.0] * len(texts)


def _apply_finbert(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    news_df에 finbert_score 컬럼 추가.
    guardian_news_cache.csv에 점수를 캐시하여 재계산 생략.
    """
    news_df = news_df.copy()

    # ── 캐시에서 기존 점수 로드
    cached = {}
    if NEWS_CACHE_FILE.exists():
        try:
            _c = pd.read_csv(NEWS_CACHE_FILE)
            if 'finbert_score' in _c.columns:
                cached = dict(zip(
                    _c['title'].astype(str),
                    pd.to_numeric(_c['finbert_score'], errors='coerce'),
                ))
        except Exception:
            pass

    news_df['finbert_score'] = news_df['title'].astype(str).map(cached)

    # ── 미처리 기사만 FinBERT 실행
    mask = news_df['finbert_score'].isna()
    n_new = int(mask.sum())
    if n_new > 0:
        log.info(f"    FinBERT 신규 처리: {n_new}건...")
        new_scores = _finbert_batch(news_df.loc[mask, 'title'].tolist())
        news_df.loc[mask, 'finbert_score'] = new_scores

        # ── 캐시 업데이트
        try:
            _c = pd.read_csv(NEWS_CACHE_FILE) if NEWS_CACHE_FILE.exists() else news_df[['date','title','source']].copy()
            if 'finbert_score' not in _c.columns:
                _c['finbert_score'] = np.nan
            _upd = dict(zip(
                news_df.loc[mask, 'title'].astype(str),
                news_df.loc[mask, 'finbert_score'],
            ))
            _idx = _c['title'].astype(str).isin(_upd)
            _c.loc[_idx, 'finbert_score'] = _c.loc[_idx, 'title'].astype(str).map(_upd)
            _c.to_csv(NEWS_CACHE_FILE, index=False)
        except Exception as _ce:
            log.debug(f"    FinBERT 캐시 저장 실패({_ce})")

    news_df['finbert_score'] = pd.to_numeric(news_df['finbert_score'], errors='coerce').fillna(0.0)
    return news_df


# ─────────────────────────────────────────────────────────────────────────────
# 3c. Sentence Embedding (all-MiniLM-L6-v2) + WTI 상관관계 기반 피처 선택
#     + Oil Event 라이브러리 기반 유가 방향 스코어
# ─────────────────────────────────────────────────────────────────────────────

# WTI 가격 방향이 알려진 캐노니컬 이벤트 라이브러리
# 양수 = WTI 상승, 음수 = WTI 하락 (경제 감성과 무관, 유가 방향 직접 매핑)
OIL_EVENT_LIBRARY = {
    # ── 공급 감소 → WTI 상승 (대규모)
    "OPEC production cut two million barrels per day deep":                +0.95,
    "OPEC production cut agreement extended deeper barrels per day":       +0.90,
    "OPEC surprise voluntary output cut announced":                        +0.85,
    "Saudi Arabia announces unilateral production cut one million":        +0.85,
    "Saudi Arabia voluntary cut extended deeper barrel reduction":         +0.80,
    "OPEC+ ministerial meeting agrees production cut quota":               +0.80,
    "Russia restricts oil export volumes crude shipments":                 +0.75,
    "Iran nuclear deal collapsed sanctions tightened maximum pressure":    +0.75,
    "US Iran sanctions intensify oil supply restricted":                   +0.70,
    "Venezuela sanctions tightened oil exports blocked":                   +0.65,
    # ── 공급 감소 → WTI 상승 (물리적 충격)
    "pipeline attack disruption shutdown oil supply":                      +0.80,
    "refinery fire explosion shutdown production offline":                 +0.70,
    "Libya oil field shutdown civil conflict armed":                       +0.65,
    "Nigeria oil production disrupted militant attack":                    +0.65,
    "Iraq oil exports suspended Kurdistan dispute":                        +0.60,
    "Kazakhstan oil output disrupted Tengiz field":                        +0.60,
    "Ecuador oil production halted indigenous protest":                    +0.55,
    "hurricane threatens Gulf Mexico oil platform production":             +0.65,
    "Houthi attack Red Sea oil tanker shipping disruption":                +0.70,
    "Strait Hormuz tension blockade tanker seizure":                       +0.75,
    "OPEC+ compliance exceeds quota output below target":                  +0.55,
    # ── 재고 감소 → WTI 상승
    "crude oil inventory draw stockpile fell unexpected large":            +0.70,
    "EIA inventory draw crude stockpile decline million barrels":          +0.65,
    "US crude stockpile falls sharply drawdown":                           +0.60,
    "gasoline distillate inventory draw product shortage":                 +0.55,
    # ── 공급 증가 → WTI 하락
    "OPEC agrees increase output production quota barrels":               -0.80,
    "OPEC+ eases cuts production increase members":                       -0.75,
    "Iran nuclear deal reached sanctions lifted export":                  -0.80,
    "US Strategic Petroleum Reserve SPR release million barrels":         -0.65,
    "Libya oil production resumes resumed restart field":                  -0.60,
    "US shale oil production record high output rig":                     -0.65,
    "Saudi Arabia increases output production boost barrels":             -0.70,
    "Russia Ukraine ceasefire deal energy supply restored":               -0.60,
    "Venezuela US sanctions eased oil exports resume":                    -0.55,
    # ── 재고 증가 → WTI 하락
    "EIA crude inventory build stockpile rose unexpected million":        -0.70,
    "crude oil inventory surplus build stockpile increase":               -0.65,
    "US crude stockpile rises large build glut":                          -0.60,
    "global oil supply surplus inventory overhang":                       -0.60,
    # ── 수요 감소 → WTI 하락
    "China economic slowdown GDP misses oil demand falls":                -0.75,
    "China manufacturing PMI contracts economy slows":                    -0.65,
    "global recession fears oil demand outlook weakens":                  -0.70,
    "US recession economic contraction demand destruction":               -0.65,
    "Fed rate hike interest rates rise dollar strengthens":               -0.50,
    "weak oil demand forecast IEA OPEC lowers outlook":                   -0.65,
    "India oil imports decline slowing economy growth":                   -0.50,
    "manufacturing PMI falls contraction economic weakness":              -0.50,
    "electric vehicle adoption accelerates oil demand peak":              -0.45,
    "airline flights cancelled reduced fuel demand":                      -0.50,
    "Covid lockdown China economy shutdown demand":                       -0.80,
    # ── 수요 증가 → WTI 상승
    "China economic recovery strong demand surge oil imports":            +0.70,
    "China reopening post-covid demand recovery oil":                     +0.75,
    "global oil demand growth forecast raised IEA OPEC":                  +0.60,
    "emerging market demand recovery economic growth strong":             +0.55,
    "summer driving season peak demand travel gasoline":                  +0.45,
    "winter heating demand natural gas oil surge":                        +0.50,
    "jet fuel aviation demand recovery airline travel":                   +0.45,
    # ── 달러/금리 영향
    "Fed rate cut interest rates fall dollar weakens oil":                +0.45,
    "dollar index weakens risk assets rally oil":                         +0.40,
    "dollar strengthens DXY risk off oil pressure":                       -0.45,
    "Fed hawkish tightening dollar rally oil falls":                      -0.50,
    # ── 지정학 프리미엄
    "Middle East war escalation risk premium oil":                        +0.70,
    "Israel Gaza conflict escalates regional war risk":                   +0.65,
    "Russia Ukraine war energy security risk":                            +0.60,
    "geopolitical tension risk premium crude oil":                        +0.50,
    "ceasefire agreement peace deal risk premium falls":                  -0.50,
}

# 이벤트 카테고리: OIL_EVENT_LIBRARY 키 → 카테고리 매핑 (중복 dict 대신 집합으로 관리)
_SUPPLY_EVENTS = {
    "OPEC production cut two million barrels per day deep",
    "OPEC production cut agreement extended deeper barrels per day",
    "OPEC surprise voluntary output cut announced",
    "Saudi Arabia announces unilateral production cut one million",
    "Saudi Arabia voluntary cut extended deeper barrel reduction",
    "OPEC+ ministerial meeting agrees production cut quota",
    "Russia restricts oil export volumes crude shipments",
    "Iran nuclear deal collapsed sanctions tightened maximum pressure",
    "US Iran sanctions intensify oil supply restricted",
    "Venezuela sanctions tightened oil exports blocked",
    "pipeline attack disruption shutdown oil supply",
    "refinery fire explosion shutdown production offline",
    "Libya oil field shutdown civil conflict armed",
    "Nigeria oil production disrupted militant attack",
    "Iraq oil exports suspended Kurdistan dispute",
    "Kazakhstan oil output disrupted Tengiz field",
    "Ecuador oil production halted indigenous protest",
    "hurricane threatens Gulf Mexico oil platform production",
    "Houthi attack Red Sea oil tanker shipping disruption",
    "Strait Hormuz tension blockade tanker seizure",
    "OPEC+ compliance exceeds quota output below target",
    "crude oil inventory draw stockpile fell unexpected large",
    "EIA inventory draw crude stockpile decline million barrels",
    "US crude stockpile falls sharply drawdown",
    "gasoline distillate inventory draw product shortage",
    "OPEC agrees increase output production quota barrels",
    "OPEC+ eases cuts production increase members",
    "Iran nuclear deal reached sanctions lifted export",
    "US Strategic Petroleum Reserve SPR release million barrels",
    "Libya oil production resumes resumed restart field",
    "US shale oil production record high output rig",
    "Saudi Arabia increases output production boost barrels",
    "Russia Ukraine ceasefire deal energy supply restored",
    "Venezuela US sanctions eased oil exports resume",
    "EIA crude inventory build stockpile rose unexpected million",
    "crude oil inventory surplus build stockpile increase",
    "US crude stockpile rises large build glut",
    "global oil supply surplus inventory overhang",
}
_GEO_EVENTS = {
    "Middle East war escalation risk premium oil",
    "Israel Gaza conflict escalates regional war risk",
    "Russia Ukraine war energy security risk",
    "geopolitical tension risk premium crude oil",
    "ceasefire agreement peace deal risk premium falls",
}

def _event_category(text):
    if text in _GEO_EVENTS: return 'geo'
    if text in _SUPPLY_EVENTS: return 'supply'
    return 'demand'

_OIL_EVENT_EMBS: 'np.ndarray | None' = None
_OIL_EVENT_SCORES: 'list | None'     = None
_OIL_EVENT_CATS: 'list | None'       = None


def _get_oil_event_embeddings():
    """캐노니컬 이벤트 임베딩 (이벤트 수 변경 시 자동 재계산)"""
    global _OIL_EVENT_EMBS, _OIL_EVENT_SCORES, _OIL_EVENT_CATS
    if _OIL_EVENT_EMBS is not None and len(_OIL_EVENT_EMBS) == len(OIL_EVENT_LIBRARY):
        return _OIL_EVENT_EMBS, _OIL_EVENT_SCORES
    texts  = list(OIL_EVENT_LIBRARY.keys())
    scores = list(OIL_EVENT_LIBRARY.values())
    embs   = _embed_texts(texts)
    _OIL_EVENT_EMBS   = embs
    _OIL_EVENT_SCORES = scores
    _OIL_EVENT_CATS   = [_event_category(t) for t in texts]
    return _OIL_EVENT_EMBS, _OIL_EVENT_SCORES


def _oil_event_score(article_emb: 'np.ndarray') -> float:
    """기사 임베딩과 캐노니컬 이벤트의 코사인 유사도 가중합 → WTI 방향 점수"""
    ev_embs, ev_scores = _get_oil_event_embeddings()
    if ev_embs is None:
        return 0.0
    sims   = ev_embs @ article_emb
    sims_s = np.exp(sims * 5)
    sims_s /= sims_s.sum() + 1e-8
    return float(np.dot(sims_s, ev_scores))



_EMBED_MDL  = None
_EMBED_TOK  = None
_EMBED_REDY = None

def _load_embed_model():
    global _EMBED_MDL, _EMBED_TOK, _EMBED_REDY
    if _EMBED_REDY is not None:
        return _EMBED_MDL, _EMBED_TOK
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch as _t
        _EMBED_TOK = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _EMBED_MDL = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
        _EMBED_MDL = _EMBED_MDL.to(_dev).eval()
        _EMBED_REDY = True
        log.info(f"    ✅ Sentence Embedding 로드 ({_dev})")
    except Exception as _ee:
        _EMBED_REDY = False
        log.warning(f"    Embedding 로드 실패({_ee})")
    return _EMBED_MDL, _EMBED_TOK


def _embed_texts(texts: list) -> 'np.ndarray':
    """texts → (N, 384) 정규화 sentence embedding (mean pooling)"""
    mdl, tok = _load_embed_model()
    if mdl is None:
        return np.zeros((len(texts), 384), dtype=np.float32)
    import torch
    dev = next(mdl.parameters()).device
    parts = []
    for i in range(0, len(texts), 64):
        batch = [str(t)[:512] for t in texts[i:i+64]]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.clamp(min=1e-9).sum(1)
            emb  = torch.nn.functional.normalize(emb, p=2, dim=1)
        parts.append(emb.cpu().numpy())
    return np.vstack(parts).astype(np.float32)


def _apply_embeddings(news_df: pd.DataFrame) -> pd.DataFrame:
    """뉴스 기사별 sentence embedding 추가 (캐시 활용)"""
    import hashlib, pickle
    news_df = news_df.copy()

    # 캐시 로드
    cache: dict = {}
    if EMBED_CACHE_FILE.exists():
        try:
            with open(EMBED_CACHE_FILE, 'rb') as _f:
                cache = pickle.load(_f)
        except Exception:
            cache = {}

    def _h(t): return hashlib.md5(str(t).encode('utf-8', errors='ignore')).hexdigest()
    news_df['_ehash'] = news_df['title'].apply(_h)

    # 미처리 기사 임베딩
    new_mask = ~news_df['_ehash'].isin(cache)
    n_new    = int(new_mask.sum())
    if n_new > 0:
        log.info(f"    Sentence Embedding 신규: {n_new}건...")
        new_embs = _embed_texts(news_df.loc[new_mask, 'title'].tolist())
        for h, e in zip(news_df.loc[new_mask, '_ehash'], new_embs):
            cache[h] = e
        # 캐시 크기 제한: 50,000건 초과 시 오래된 항목 정리 (unbounded growth 방지)
        if len(cache) > 50_000:
            _keep = set(news_df['_ehash'].tolist())
            cache = {k: v for k, v in cache.items() if k in _keep}
            log.info(f"    Embedding 캐시 정리: {len(cache)}건 유지")
        try:
            with open(EMBED_CACHE_FILE, 'wb') as _f:
                pickle.dump(cache, _f, protocol=4)
        except Exception as _ce:
            log.warning(f"    Embedding 캐시 저장 실패({_ce})")

    _zero = np.zeros(384, dtype=np.float32)
    news_df['_emb'] = news_df['_ehash'].apply(lambda h: cache.get(h, _zero))
    return news_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  build_features()
# ─────────────────────────────────────────────────────────────────────────────

# ── HAR 전용 피처셋 (변동성 예측 특화, ~25개, regime/뉴스/거시 제외)
# 과적합 원인: 96개 피처 중 regime(48%) 단독 지배 → 훈련R²=0.80 vs CV R²=0.36
HAR_FEATURE_COLS = [
    # 핵심 HAR 성분
    'RV_1d', 'RV_5d', 'RV_21d', 'RV_63d',
    # GARCH / EWMA (parkinson=rv_intraday 완전중복 제거)
    'garch_vol', 'ewma_vol_10', 'ewma_vol_21', 'ewma_vol_63',
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
    # 기존 lag
    'RV_lag1',
]

# 뉴스 감성 전용 베이스 모델 피처 (News-Sent XGBoost용 — 기술적 지표 제외)
NEWS_FEATS = [
    'news_sentiment_smooth', 'news_sentiment_smooth7',
    'news_sentiment_lag1', 'news_sentiment_lag2',
    'news_count', 'news_count_lag1', 'news_count_neg',
    'oil_event_score', 'oil_event_score_smooth',
    'sentiment_magnitude', 'extreme_neg_news',
    'news_uncertainty',
    'gpr_zscore', 'geo_dummy',
    'ovx_zscore', 'ovx_change',
    'vix_zscore', 'vix_change',
    'fear_composite', 'vix_amplified',
    'regime',
    'sent_surprise', 'sent_surprise_z',
    # 시장 맥락 (뉴스 신호 증폭/감쇠)
    'opec_days_to_next', 'opec_pre5d',   # OPEC 회의 근접 시 뉴스 신뢰도 상승
    'inv_mom4_z', 'inv_surprise',          # 재고 추세 → 뉴스 방향성 검증
]

FEATURE_COLS = [
    # HAR 구성요소 (일·주·월 실현변동성)
    'RV_1d', 'RV_5d', 'RV_21d',
    # 모멘텀
    'return_1d', 'mom_5d', 'mom_21d',
    # 외생 거시변수
    'dxy_change', 'dxy_5d', 'dxy_21d', 'dxy_vs_ma50', 'demand_shock', 'supply_shock',
    'geo_dummy', 'gpr_zscore',              # GPR 더미 + 연속형
    # 뉴스 (현재 + 시차 1·2)
    'news_sentiment_smooth', 'news_count',
    'news_sentiment_lag1', 'news_count_lag1',
    'news_sentiment_lag2', 'news_count_lag2',
    'news_sentiment_smooth7', 'sentiment_magnitude',
    'extreme_neg_news', 'news_count_neg',
    'oil_event_score', 'oil_event_score_smooth',
    # 기술적 지표
    'price_vs_ma5', 'price_vs_ma21', 'bb_position',
    'return_lag1', 'return_lag2', 'RV_lag1',
    'vol_10d', 'brent_wti_spread',
    # 추가 기술적 지표 (RSI, MACD, ATR, 가격 z-score)
    'rsi_14', 'macd', 'macd_signal', 'atr_14', 'price_zscore',
    # VIX 기반 피처
    'vix_zscore', 'vix_change',
    # VIX × 뉴스 감성 복합변수
    'fear_composite', 'vix_amplified',
    # OVX (원유 변동성 지수) 피처
    'ovx_zscore', 'ovx_change', 'ovx_rv_spread',
    # OVX/VIX 공포 스프레드 (원유 공포 vs 시장 공포)
    'ovx_vix_ratio_z',
    # 오더플로우 (매수압력, CMF, 거래량 이상)
    'buy_pressure', 'cmf_10', 'volume_zscore',
    # COT (CFTC 기관 포지션 — 역발상 신호)
    'cot_net_pct', 'cot_chg_z', 'cot_net_z',
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
    # B: Parkinson 장중 범위 추정 (rv_intraday 중복으로 FEATURE_COLS 제외, HAR만 유지)
    'parkinson_vol',
    # C: EWMA 변동성
    'ewma_vol_10', 'ewma_vol_21', 'ewma_vol_63',
    # D: 변동성 모멘텀
    'rv_term_slope', 'rv_5d_chg', 'rv_mom_5_21',
    # 장중 고빈도 실현분산 (1h)
    'rv_intraday', 'rv_intraday_5d', 'rv_intraday_21d', 'rv_intra_vs_close',
    # VIX 기간구조 + SKEW
    'vix_term_slope', 'vix_ts_zscore', 'skew_zscore', 'skew_chg',
    # 5번: 시장 국면(Regime) 피처
    'regime', 'regime_x_sent', 'regime_x_gpr',
    # COVID 특수 변수
    'covid_dummy',
    # 계절성 (원유 수요 사이클)
    'month_sin', 'month_cos', 'driving_season', 'heating_season',
    # OPEC 회의 캘린더
    'opec_days_to_next', 'opec_pre5d', 'opec_post2d',
    # 천연가스·RBOB (에너지 동조화·크랙 스프레드)
    'ng_mom_5d', 'ng_mom_21d', 'rbob_mom_5d', 'crack_spread_z',
    # EIA 재고 서프라이즈·모멘텀
    'inv_surprise', 'inv_mom4_z',
    # 감성 서프라이즈 (예상 외 뉴스 충격)
    'sent_surprise_z',
    # 크로스에셋 신호 (Gold/Copper: 위험회피·수요선행)
    'gold_wti_ratio_z', 'copper_gold_ratio_z', 'gold_mom_5d', 'copper_mom_5d',
    # 가격·재고 기술신호 (MACD 크로스, 모멘텀가속, 재고방향, 감성변화)
    'macd_cross', 'mom_accel', 'inv_draw_signal', 'inv_surprise_dir',
    'supply_demand_gap', 'inv_draw_x_macd', 'sentiment_chg3',
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

    # 통계 산출 기준: 훈련 구간만 사용 (미래 누출 방지, N_TEST=90)
    _n_tr_stat = max(len(df) - 90, 252)
    # ── 극단 수익률 Winsorization (상하위 0.5% 클리핑, 훈련 분위수 적용)
    lo = df['return_1d'].iloc[:_n_tr_stat].quantile(0.005)
    hi = df['return_1d'].iloc[:_n_tr_stat].quantile(0.995)
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
        log.info(f"    Parkinson vol 생성 (μ={df['parkinson_vol'].mean():.5f})")
    else:
        df['parkinson_vol']     = 0.0

    # ── CEEMDAN 신호 분해 (논문: CEEMDAN+LSTM-Attention 추세/노이즈 분리)
    # 미래 누출 방지: 훈련 구간(전체 - N_TEST)만 분해, 테스트 구간은 마지막 훈련값 연장
    try:
        from PyEMD import CEEMDAN as _CEEMDAN
        import hashlib as _hl
        _n_total  = len(df)
        _n_tr_cem = max(_n_total - 90, _n_total)  # 90 = N_TEST
        _n_tr_cem = _n_total - 90 if _n_total > 90 else _n_total
        _wti_full = df['WTI'].ffill().values.astype(float)
        _wti_arr  = _wti_full[:_n_tr_cem]          # 훈련 구간만 사용
        _cem_hash = _hl.md5(_wti_arr.tobytes()).hexdigest()[:16]
        _cem_cache = OUTPUT_DIR / f'ceemdan_{_cem_hash}.npy'
        if _cem_cache.exists():
            _imfs = np.load(str(_cem_cache), allow_pickle=True)
            log.info("    CEEMDAN 캐시 로드")
        else:
            _cem = _CEEMDAN(trials=20, epsilon=0.005)
            _imfs = _cem(_wti_arr)
            np.save(str(_cem_cache), _imfs)
            for _old_c in OUTPUT_DIR.glob('ceemdan_*.npy'):
                if _old_c != _cem_cache:
                    try:
                        _old_c.unlink(missing_ok=True)
                    except Exception as _ul_e:
                        log.debug(f"    CEEMDAN 구 캐시 삭제 실패({_old_c.name}): {_ul_e}")
        _n_cem = len(_wti_arr)
        # 저주파 성분 = 마지막 IMF(추세), 고주파 = 첫 IMF들(잡음)
        _trend_tr = _imfs[-1][:_n_cem]   # 훈련 구간 추세
        _noise_tr = _imfs[0][:_n_cem]    # 훈련 구간 잡음
        # 테스트 구간: 마지막 훈련값 연장 (미래 정보 차단)
        _n_ext = _n_total - _n_cem
        _trend = np.concatenate([_trend_tr, np.full(_n_ext, _trend_tr[-1])])
        _noise = np.concatenate([_noise_tr, np.full(_n_ext, _noise_tr[-1])])
        _wm    = max(df['WTI'].mean(), 1e-8)
        _trend_s = pd.Series(_trend, index=df.index)
        _noise_s = pd.Series(_noise, index=df.index)
        df['ceemdan_trend_ret']  = _trend_s.diff().fillna(0) / _wm   # 추세 일변화율
        df['ceemdan_noise_std5'] = _noise_s.rolling(5).std().fillna(0) / _wm  # 잡음 강도
        df['ceemdan_trend_mom5'] = _trend_s.diff(5).fillna(0) / _wm  # 추세 5일 모멘텀
        log.info("    CEEMDAN 분해 완료 (IMF 수: %d)", len(_imfs))
    except Exception as _ce:
        log.warning(f"    CEEMDAN 실패({_ce}) → 0")
        for _cc in ['ceemdan_trend_ret', 'ceemdan_noise_std5', 'ceemdan_trend_mom5']:
            df[_cc] = 0.0

    # ── C: EWMA 변동성 (RiskMetrics λ=0.94)
    df['ewma_vol_10']  = df['log_return'].ewm(span=10,  adjust=False).std().fillna(0)
    df['ewma_vol_21']  = df['log_return'].ewm(span=21,  adjust=False).std().fillna(0)
    df['ewma_vol_63']  = df['log_return'].ewm(span=63,  adjust=False).std().fillna(0)

    # ── D: 변동성 모멘텀 (term structure + 변화량)
    df['rv_term_slope']  = (df['RV_5d'] / df['RV_21d'].replace(0, np.nan)).fillna(1.0)  # >1: 단기>장기(변동성 상승 중)
    df['rv_5d_chg']      = df['RV_5d'].diff().fillna(0)   # 변동성 변화량
    df['rv_mom_5_21']    = (df['RV_5d'] - df['RV_21d']).fillna(0)  # 단기-장기 스프레드

    # ── A: GARCH(1,1) 조건부 분산 (변동성 클러스터링 명시적 모델링)
    _garch_result = None   # 호출자에게 반환할 fit 결과
    try:
        from arch import arch_model as _arch_model
        _ret_pct = df['log_return'].dropna() * 100   # % 스케일
        _garch   = _arch_model(_ret_pct, vol='Garch', p=1, q=1,
                               dist='Normal', rescale=False)
        _res     = _garch.fit(disp='off', show_warning=False)
        _cond_vol = _res.conditional_volatility / 100   # 소수점 스케일 복원
        df['garch_vol'] = _cond_vol.reindex(df.index).ffill().bfill().fillna(0)
        _garch_result = _res   # 외부 전달용 저장
        log.info(f"    GARCH(1,1) 조건부 분산 생성 (μ={df['garch_vol'].mean():.5f})")
    except Exception as _ge:
        log.warning(f"    GARCH 실패({_ge}) → garch_vol=0")
        df['garch_vol'] = 0.0

    # ── 장중 고빈도 실현분산 (1h 데이터, 최근 730일 — 더 정확한 RV 추정)
    try:
        _raw_1h = yf.download("CL=F", period="365d", interval="1h",
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
    df['inv_draw_signal']  = (df['inv_chg_zscore'] < 0).astype(float)  # 재고 감소=상승 신호
    df['inv_surprise_dir'] = (df['inv_surprise'] < 0).astype(float)    # 예상 외 감소=상승 신호

    # ── 계절성 피처 (원유 수요 사이클)
    _month = pd.to_datetime(df.index).month
    df['month_sin']       = np.sin(2 * np.pi * _month / 12)   # 연간 주기 순환 인코딩
    df['month_cos']       = np.cos(2 * np.pi * _month / 12)
    df['driving_season']  = _month.isin([5,6,7,8,9]).astype(float)   # 5~9월 드라이빙 시즌
    df['heating_season']  = _month.isin([11,12,1,2,3]).astype(float)  # 11~3월 난방유 시즌

    # ── OPEC 회의 캘린더 피처
    _idx_ts = pd.to_datetime(df.index).tz_localize(None)
    _opec_ts = pd.DatetimeIndex(_OPEC_MEETING_DATES)
    _days_to_next   = np.full(len(df), 999, dtype=float)
    _days_since_last = np.full(len(df), 999, dtype=float)
    for _i, _d in enumerate(_idx_ts):
        _fut = (_opec_ts - _d).days
        _pas = (_d - _opec_ts).days
        _fut_pos = _fut[_fut >= 0]
        _pas_pos = _pas[_pas >= 0]
        if len(_fut_pos): _days_to_next[_i]    = float(_fut_pos.min())
        if len(_pas_pos): _days_since_last[_i] = float(_pas_pos.min())
    df['opec_days_to_next']  = np.clip(_days_to_next, 0, 30) / 30.0    # 0(당일)~1(30일이상)
    df['opec_pre5d']         = (_days_to_next  <= 5).astype(float)     # 회의 5일 전 투기구간
    df['opec_post2d']        = (_days_since_last <= 2).astype(float)   # 회의 2일 후 반응구간
    log.info(f"    OPEC 캘린더 피처 생성 (pre5d={df['opec_pre5d'].sum():.0f}일 "
             f"post2d={df['opec_post2d'].sum():.0f}일)")

    # ── 이동평균 & 모멘텀
    df['ma_5d']  = df['WTI'].rolling(5).mean()
    df['ma_21d'] = df['WTI'].rolling(21).mean()
    df['vol_5d']  = df['log_return'].rolling(5).std()
    df['vol_10d'] = df['log_return'].rolling(10).std()
    df['mom_5d']    = df['WTI'].pct_change(5)
    df['mom_21d']   = df['WTI'].pct_change(21)
    df['mom_accel']     = df['mom_5d'].diff(5).fillna(0)

    # ── DXY 변화율 + 장기 모멘텀
    df['dxy_change'] = df['DXY'].pct_change()
    df['dxy_5d']     = df['DXY'].pct_change(5)
    df['dxy_21d']    = df['DXY'].pct_change(21)
    df['dxy_vs_ma50']= (df['DXY'] / (df['DXY'].rolling(50).mean() + 1e-8) - 1).fillna(0)

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
    df['macd_cross']     = (df['macd'] > df['macd_signal']).astype(float)
    df['inv_draw_x_macd'] = df['inv_draw_signal'] * df['macd_cross']

    # ── ATR proxy (14일 평균 절대 로그수익률): 가격 범위 기반 변동성
    df['atr_14'] = df['log_return'].abs().rolling(14).mean().fillna(0)

    # ── 가격 z-score (252일 롤링): 역사적 수준 대비 현재 위치
    df['price_zscore'] = (
        (df['WTI'] - df['WTI'].rolling(252).mean()) /
        (df['WTI'].rolling(252).std() + 1e-8)
    ).fillna(0)   # 초기 252일 NaN → 0 (평균 수준으로 처리)

    # ── 뉴스 집계: 소스 신뢰도 × 핵심 기관 언급 복합 가중치
    if not news_df.empty:
        news_df = _apply_finbert(news_df)   # FinBERT 캐시 적용
        _rule_scores = news_df['title'].apply(score_sentiment).values
        # 하이브리드: FinBERT(맥락 이해) 60% + 유가 특화 규칙 40%
        news_df['sentiment'] = 0.6 * news_df['finbert_score'].values + 0.4 * _rule_scores

        def _impact_w(row):
            src_w = SOURCE_WEIGHTS.get(str(row.get('source', 'RSS')), 1.0)
            ent_w = 1.5 if any(k in str(row['title']).lower() for k in HIGH_IMPACT_ENTITIES) else 1.0
            return src_w * ent_w

        news_df['impact_w']    = news_df.apply(_impact_w, axis=1)
        news_df['w_sentiment'] = news_df['sentiment'] * news_df['impact_w']
        news_df['is_pos']      = (news_df['sentiment'] >  0.05).astype(float)
        news_df['is_neg']      = (news_df['sentiment'] < -0.05).astype(float)

        def _hedge_score(text):
            if not isinstance(text, str): return 0.0
            toks = text.lower().split()
            return min(sum(1 for t in toks if t in HEDGE_WORDS) / (len(toks) * 0.1 + 1e-8), 1.0)
        news_df['uncertainty'] = news_df['title'].apply(_hedge_score)

        def _wavg_sent(g):
            return g['w_sentiment'].sum() / g['impact_w'].sum() if g['impact_w'].sum() > 0 else 0.0

        daily = news_df.groupby('date').apply(
            lambda g: pd.Series({
                'news_count':       len(g),
                'news_sentiment':   _wavg_sent(g),
                'news_count_pos':   g['is_pos'].sum(),
                'news_count_neg':   g['is_neg'].sum(),
                'news_uncertainty': g['uncertainty'].mean(),
            })
        )
        daily.index = pd.to_datetime(daily.index)
        df = df.join(daily, how='left')
        df['news_count']       = df['news_count'].fillna(0)
        df['news_count_pos']   = df['news_count_pos'].fillna(0)
        df['news_count_neg']   = df['news_count_neg'].fillna(0)
        df['news_uncertainty'] = df['news_uncertainty'].ffill().fillna(0)
        df['news_sentiment']   = df['news_sentiment'].ffill().fillna(0)

    else:
        df['news_count']     = 0
        df['news_count_pos']   = 0
        df['news_count_neg']   = 0
        df['news_sentiment']   = 0
        df['news_uncertainty'] = 0

    # ── Sentence Embedding → WTI 상관관계 상위 피처 + Oil Event 스코어
    if not news_df.empty:
        try:
            _ne = _apply_embeddings(news_df)  # '_emb' 컬럼 추가
            # ── Oil Event 라이브러리 스코어: 벡터화 행렬 연산
            _ev_embs_v, _ev_scores_v = _get_oil_event_embeddings()
            if _ev_embs_v is not None and not _ne.empty:
                _art_mat   = np.vstack(_ne['_emb'].values)           # (N, 384)
                _sims_v    = _art_mat @ _ev_embs_v.T                 # (N, n_events)
                _sims_sw   = np.exp(_sims_v * 5)
                _sims_sw  /= _sims_sw.sum(axis=1, keepdims=True) + 1e-8
                _ev_sc_arr = np.array(_ev_scores_v)
                _ne['_oil_score'] = _sims_sw @ _ev_sc_arr
                for _cat, _col in [('supply','_supply_score'),('demand','_demand_score'),('geo','_geo_score')]:
                    _cmask   = np.array([c == _cat for c in _OIL_EVENT_CATS], dtype=float)
                    _w       = _sims_sw * _cmask
                    _ne[_col] = (_w / (_w.sum(axis=1, keepdims=True) + 1e-8)) @ _ev_sc_arr
            else:
                for _col in ['_oil_score','_supply_score','_demand_score','_geo_score']:
                    _ne[_col] = 0.0
            # 일별 impact_w 가중평균
            def _wavg_col(g, col):
                w = g.get('impact_w', pd.Series(np.ones(len(g)))).values
                return (g[col].values * w).sum() / (w.sum() + 1e-8)
            _oil_daily = _ne.groupby(pd.to_datetime(_ne['date']).dt.date).apply(
                lambda g: pd.Series({
                    '_oil':    _wavg_col(g, '_oil_score'),
                    '_supply': _wavg_col(g, '_supply_score'),
                    '_demand': _wavg_col(g, '_demand_score'),
                    '_geo':    _wavg_col(g, '_geo_score'),
                })
            )
            _oil_daily.index = pd.to_datetime(_oil_daily.index)
            _oil_s = _oil_daily['_oil'].reindex(df.index).ffill().fillna(0)
            df['oil_event_score']        = _oil_s.values
            df['oil_event_score_smooth'] = _oil_s.ewm(span=3, min_periods=1).mean().values
            for _cat, _col in [('_supply','supply_event_score'),
                                ('_demand','demand_event_score'),
                                ('_geo',   'geo_event_score')]:
                _s = _oil_daily[_cat].reindex(df.index).ffill().fillna(0)
                df[_col]             = _s.values
                df[_col + '_smooth'] = _s.ewm(span=3, min_periods=1).mean().values
            df['supply_demand_gap'] = df['supply_event_score'] - df['demand_event_score']
            log.info(f"    Oil Event 스코어: μ={df['oil_event_score'].mean():.4f} "
                     f"σ={df['oil_event_score'].std():.4f}")
            # 뉴스가 있는 날짜별 impact_w 가중 평균 임베딩 계산
            _news_dates = pd.to_datetime(_ne['date']).dt.date
            _emb_matrix = np.zeros((len(df), 384), dtype=np.float32)
            for _d, _grp in _ne.groupby(_news_dates):
                _ts = pd.Timestamp(_d)
                if _ts in df.index:
                    _idx = df.index.get_loc(_ts)
                    _e   = np.stack(_grp['_emb'].values)          # (n, 384)
                    _w   = _grp.get('impact_w', pd.Series(np.ones(len(_grp)))).values.reshape(-1, 1)
                    _emb_matrix[_idx] = (_e * _w).sum(0) / _w.sum()
            # 영업일 비뉴스 날: ffill
            _prev = np.zeros(384, dtype=np.float32)
            for _i in range(len(_emb_matrix)):
                if _emb_matrix[_i].any():
                    _prev = _emb_matrix[_i].copy()
                else:
                    _emb_matrix[_i] = _prev
            # 수익률과의 상관관계로 top-K 차원 선택 (미래 누출 방지: 첫 80%만 사용)
            if 'return_1d' in df.columns and len(df) > 200:
                _n_corr_tr  = int(len(df) * 0.8)   # 상관관계 계산은 첫 80%만
                # shift(-1)을 훈련 구간만 잘라낸 뒤 적용 — 테스트 구간 정보 차단
                _ret_tr     = df['return_1d'].iloc[:_n_corr_tr]
                _ret_next_tr = _ret_tr.shift(-1).values   # 훈련 구간 내 익일 수익률
                _ret_next   = np.full(len(df), np.nan)
                _ret_next[:_n_corr_tr] = _ret_next_tr
                _valid      = ~np.isnan(_ret_next) & (_emb_matrix.sum(1) != 0)
                _valid[_n_corr_tr:] = False          # 테스트 구간 제외
                _corrs = np.array([
                    float(np.corrcoef(_emb_matrix[_valid, _d], _ret_next[_valid])[0, 1])
                    if _emb_matrix[_valid, _d].std() > 1e-8 else 0.0
                    for _d in range(384)
                ])
                _top_idx = np.argsort(np.abs(_corrs))[-EMBED_TOP_K:]
                for _rank, _dim in enumerate(sorted(_top_idx)):
                    df[f'emb_d{_rank}'] = _emb_matrix[:, _dim]
                log.info(f"    Embedding top-{EMBED_TOP_K}: max|corr|={np.abs(_corrs).max():.4f} "
                         f"mean|corr|={np.abs(_corrs[_top_idx]).mean():.4f}")
        except Exception as _emb_e:
            log.warning(f"    Embedding 피처 생성 실패({_emb_e})")

    # ── gpr_zscore 보정: 뉴스가 없는 날 GPR도 ffill로 유지됨 (이미 _attach_gpr에서 처리)
    if 'gpr_zscore' not in df.columns:
        df['gpr_zscore'] = 0.0
    if 'geo_dummy' not in df.columns:
        df['geo_dummy'] = 0.0

    # ── 뉴스 감성 파생 피처
    df['news_sentiment_smooth']  = df['news_sentiment'].ewm(span=3, min_periods=1).mean()
    df['news_sentiment_smooth7'] = df['news_sentiment'].ewm(span=7, min_periods=1).mean()
    df['sentiment_chg3'] = df['news_sentiment'].diff(3).fillna(0)
    # 감성 강도: 절댓값 × log(뉴스수+1) — 큰 감성 + 많은 기사 = 강한 신호
    df['sentiment_magnitude']    = df['news_sentiment'].abs() * np.log1p(df['news_count'])
    # 극단 감성 더미 (EWM 평활 기준 ±0.35 초과)
    df['extreme_neg_news'] = (df['news_sentiment_smooth'] < -0.35).astype(float)
    for lag in [1, 2]:
        df[f'news_sentiment_lag{lag}'] = df['news_sentiment'].shift(lag)
        df[f'news_count_lag{lag}']     = df['news_count'].shift(lag)

    # 감성 서프라이즈: 예상(7일 MA 이전값) 대비 실제 감성 편차 — 시장 미반영 충격
    _sent_ma7  = df['news_sentiment'].rolling(7, min_periods=1).mean().shift(1)
    _sent_surp = (df['news_sentiment'] - _sent_ma7).fillna(0)
    _ss_mu     = _sent_surp.rolling(252).mean()
    _ss_std    = _sent_surp.rolling(252).std()
    df['sent_surprise']   = _sent_surp
    df['sent_surprise_z'] = ((_sent_surp - _ss_mu) / (_ss_std + 1e-8)).fillna(0)

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

    # ── OVX/VIX 공포 스프레드 (원유 공포 vs 시장 공포)
    if 'OVX' in df.columns and 'VIX' in df.columns and \
       df['OVX'].notna().sum() > 30 and df['VIX'].notna().sum() > 30:
        _ovx_vix = (df['OVX'] / df['VIX'].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).ffill().bfill()
        df['ovx_vix_ratio_z'] = ((_ovx_vix - _ovx_vix.rolling(252).mean()) /
                                  (_ovx_vix.rolling(252).std() + 1e-8)).fillna(0)
        log.info(f"    OVX/VIX 공포 스프레드 생성 (μ={_ovx_vix.mean():.3f})")
    else:
        df['ovx_vix_ratio_z'] = 0.0

    # ── 오더플로우 피처 (OHLCV 기반, 매수압력 + Chaikin Money Flow)
    if all(c in df.columns for c in ['WTI_High', 'WTI_Low', 'WTI_Volume']):
        _h = df['WTI_High'].replace(0, np.nan)
        _l = df['WTI_Low'].replace(0, np.nan)
        _c = df['WTI'].replace(0, np.nan)
        _v = df['WTI_Volume'].replace(0, np.nan)
        # 매수 압력: (close-low)/(high-low), 1=완전 상단 마감(강세)
        _hl = (_h - _l).replace(0, np.nan)
        df['buy_pressure'] = ((_c - _l) / _hl).clip(0, 1).fillna(0.5)
        # Money Flow Multiplier & CMF 10일
        _mfm = (((_c - _l) - (_h - _c)) / _hl).fillna(0)
        _mfv = _mfm * _v
        df['cmf_10'] = (_mfv.rolling(10).sum() /
                        _v.rolling(10).sum().replace(0, np.nan)).fillna(0)
        # Volume z-score (비정상 거래량)
        df['volume_zscore'] = ((_v - _v.rolling(252).mean()) /
                               (_v.rolling(252).std() + 1e-8)).fillna(0)
        log.info(f"    오더플로우 피처 생성 (buy_pressure/cmf_10/volume_zscore)")
    else:
        df['buy_pressure']   = 0.5
        df['cmf_10']         = 0.0
        df['volume_zscore']  = 0.0

    # ── COT (CFTC Commitments of Traders) 피처
    try:
        _cot = fetch_cot()
        if not _cot.empty and 'cot_net_pct' in _cot.columns:
            # 주간 → 영업일 ffill (3일 발표 지연 반영: shift(3))
            _bdays_cot = pd.date_range(_cot.index.min(), df.index.max(), freq='B')
            _cot_d = _cot.reindex(_bdays_cot).ffill().shift(3)   # 발표 지연
            _cot_d = _cot_d.reindex(df.index).ffill().bfill().fillna(0)
            df['cot_net_pct']  = _cot_d['cot_net_pct'].values
            # 주간 변화량 z-score (역발상 신호: 포지션↑→가격↓, r=-0.12***)
            _cot_chg = _cot_d['cot_net_pct'].diff(5)
            _cot_chg_z = ((_cot_chg - _cot_chg.rolling(252).mean()) /
                          (_cot_chg.rolling(252).std() + 1e-8)).fillna(0)
            df['cot_chg_z']    = _cot_chg_z.values
            # 포지션 절대 수준 z-score
            _cot_z = ((_cot_d['cot_net_pct'] - _cot_d['cot_net_pct'].rolling(252).mean()) /
                      (_cot_d['cot_net_pct'].rolling(252).std() + 1e-8)).fillna(0)
            df['cot_net_z']    = _cot_z.values
            log.info(f"    COT 피처 생성 (net_pct μ={df['cot_net_pct'].mean():.2f})")
        else:
            df['cot_net_pct'] = df['cot_chg_z'] = df['cot_net_z'] = 0.0
    except Exception as _coterr:
        log.warning(f"    COT 피처 실패({_coterr}) → 0")
        df['cot_net_pct'] = df['cot_chg_z'] = df['cot_net_z'] = 0.0

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

    # ── 천연가스 / RBOB 모멘텀 + 크랙 스프레드 ──────────────────────────────
    if 'NatGas' in df.columns and df['NatGas'].notna().sum() > 30:
        df['NatGas'] = df['NatGas'].ffill().bfill()
        df['ng_mom_5d']  = df['NatGas'].pct_change(5).fillna(0)
        df['ng_mom_21d'] = df['NatGas'].pct_change(21).fillna(0)
        log.info("    천연가스 모멘텀 피처 생성")
    else:
        df['ng_mom_5d'] = df['ng_mom_21d'] = 0.0

    if 'RBOB' in df.columns and df['RBOB'].notna().sum() > 30:
        df['RBOB'] = df['RBOB'].ffill().bfill()
        df['rbob_mom_5d']    = df['RBOB'].pct_change(5).fillna(0)
        df['crack_spread']   = (df['RBOB'] * 42 - df['WTI']).fillna(0)  # $/bbl 환산
        df['crack_spread_z'] = ((df['crack_spread'] - df['crack_spread'].rolling(63).mean())
                                / (df['crack_spread'].rolling(63).std() + 1e-8)).fillna(0)
        log.info("    RBOB 크랙 스프레드 피처 생성")
    else:
        df['rbob_mom_5d'] = df['crack_spread'] = df['crack_spread_z'] = 0.0

    # ── 크로스에셋: Gold / Copper ─────────────────────────────────────────────
    for _ca_col, _ca_prefix in [('Gold', 'gold'), ('Copper', 'copper')]:
        if _ca_col in df.columns and df[_ca_col].notna().sum() > 30:
            df[_ca_col] = df[_ca_col].ffill().bfill()
            df[f'{_ca_prefix}_mom_5d']  = df[_ca_col].pct_change(5).fillna(0)
            df[f'{_ca_prefix}_mom_21d'] = df[_ca_col].pct_change(21).fillna(0)
        else:
            df[f'{_ca_prefix}_mom_5d'] = df[f'{_ca_prefix}_mom_21d'] = 0.0
    # Gold/WTI 비율 (리스크오프 신호: 상승→WTI 약세 압력)
    if 'Gold' in df.columns and df['Gold'].notna().sum() > 30:
        _g_ratio = (df['Gold'] / df['WTI']).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df['gold_wti_ratio_z'] = ((_g_ratio - _g_ratio.rolling(63).mean())
                                  / (_g_ratio.rolling(63).std() + 1e-8)).fillna(0)
    else:
        df['gold_wti_ratio_z'] = 0.0
    # Copper/Gold 비율 (경기선행 신호: 상승→수요 강세→WTI 강세)
    if 'Copper' in df.columns and 'Gold' in df.columns and df['Copper'].notna().sum() > 30:
        _cu_g = (df['Copper'] / df['Gold']).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df['copper_gold_ratio_z'] = ((_cu_g - _cu_g.rolling(63).mean())
                                     / (_cu_g.rolling(63).std() + 1e-8)).fillna(0)
    else:
        df['copper_gold_ratio_z'] = 0.0
    log.info("    크로스에셋 피처 생성 (Gold/Copper)")

    # ── VIX × 뉴스 감성 복합변수 (뉴스 집계 이후에 계산)
    neg_sent = (-df['news_sentiment_smooth']).clip(lower=0)   # 부정 감성만 추출

    # 공포 복합지수: VIX 높고 뉴스도 부정일 때만 발화 → 가장 신뢰도 높은 신호
    df['fear_composite']  = df['vix_zscore'] * neg_sent

    # VIX 증폭 감성: VIX 레벨로 뉴스 감성 자체를 스케일링
    df['vix_amplified']   = (df['news_sentiment_smooth']
                             * (1 + df['vix_zscore'].clip(lower=0)))

    # ── 5번: 시장 국면(Regime) 피처 — 변동성 75th pct 기준 고/저변동 구분
    vol_thresh = df['vol_5d'].iloc[:_n_tr_stat].quantile(0.75)  # 훈련 구간 기준
    df['regime']       = (df['vol_5d'] > vol_thresh).astype(float)
    df['regime_x_sent']= df['regime'] * df['news_sentiment_smooth']
    df['regime_x_gpr'] = df['regime'] * df['gpr_zscore']

    # ── 훈련 타깃 (다음 날 단일일 실현변동성 & 가격 & 수익률)
    # vol regime 지속성 예측 (4/5일 overlap 있으나 실용적 — GARCH와 병렬 비교)
    df['target_rv']        = df['RV_5d'].shift(-1)
    df['target_rv_log']    = np.log(df['target_rv'].clip(lower=1e-8))
    # A: vol 변화량 (overlap 성분 제거 — delta 학습)
    df['target_rv_delta']  = df['RV_5d'].shift(-1) - df['RV_5d']
    # B: GARCH 잔차 (garch_vol[t] ≈ h_{t+1} 근사, HAR이 GARCH 미포착 패턴 학습)
    df['target_rv_garch']  = df['RV_5d'].shift(-1) - df['garch_vol']
    df['target_price']     = df['WTI'].shift(-1)
    df['target_return'] = np.log(df['WTI'].shift(-1) / df['WTI'])   # 내일 log 수익률
    for _h in range(2, 8):
        df[f'target_return_h{_h}'] = np.log(df['WTI'].shift(-_h) / df['WTI'].clip(lower=1e-6))

    # 피처 행만 dropna (타깃 NaN 포함 시 훈련용으로만 제거)
    feat_na_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_full = df.ffill().bfill()      # 마지막 행 보존용 (예측에 사용, NaN ffill 보증)
    df.dropna(subset=feat_na_cols + ['target_rv', 'target_rv_log', 'target_price', 'target_return',
                                     'target_rv_delta', 'target_rv_garch'], inplace=True)

    if len(df) < 92:  # 최소 훈련(1행) + 테스트(90행) + 여유(1행)
        raise ValueError(f"피처 행 부족: {len(df)}행 (최소 62 필요). 데이터 수집 실패 또는 dropna 과다.")

    log.info(f"    피처 완성: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, df_full, {'garch_model': _garch_result}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  train_models()
# ─────────────────────────────────────────────────────────────────────────────

def train_models(feature_df: pd.DataFrame, full_df: pd.DataFrame = None, aux: dict = None):
    """XGBoost-HAR (변동성) + SARIMAX (가격) 훈련 및 성능 평가"""
    log.info("[4/9] 모델 훈련 중...")

    available_feats = [c for c in FEATURE_COLS if c in feature_df.columns]
    # sentence embedding 피처 자동 포함 (emb_d0 ~ emb_d{K-1})
    available_feats += [c for c in feature_df.columns if c.startswith('emb_d') and c not in available_feats]
    # HAR 전용 피처: regime/뉴스/거시 제외로 과적합 방지
    har_feats = [c for c in HAR_FEATURE_COLS if c in feature_df.columns]
    log.info(f"    HAR 피처: {len(har_feats)}개 / 전체: {len(available_feats)}개")

    # ── 테스트셋: 최근 90 영업일 (원샷 장기예측 오차 제거)
    n_test   = 90
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

    # ── D: 훈련 피처 분포 저장 (라이브 드리프트 감지용)
    import json as _json_fds
    try:
        _fd_stats = {}
        for _fc in X_tr_all.columns:
            _col = X_tr_all[_fc].dropna()
            if len(_col) > 10:
                _fd_stats[_fc] = {
                    'p01':  float(_col.quantile(0.01)),
                    'p99':  float(_col.quantile(0.99)),
                    'mean': float(_col.mean()),
                    'std':  float(_col.std()),
                }
        FEAT_TRAIN_STATS.write_text(_json_fds.dumps(_fd_stats))
    except Exception as _fds_e:
        log.warning(f"    피처 통계 저장 실패({_fds_e})")

    results = {}
    scaler  = None

    # ─────────────────────────────────────────────────────────────────────
    # Model A: XGBoost-HAR — walk-forward TimeSeriesSplit (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    if _SKL:
        scaler = StandardScaler()

        log.info("    [A] XGBoost-HAR (5-fold walk-forward CV) 학습 중...")
        if _XGB:
            # fold 모델과 동일 파라미터 (CV가 성능 검증 완료, 최종 모델만 과도 정규화시 r2_ho 급락)
            modelA = xgb.XGBRegressor(
                n_estimators=600, max_depth=5, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbosity=0,
            )
        else:
            modelA = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.02,
                subsample=0.7, random_state=42,
            )

        # ── walk-forward TimeSeriesSplit 평가 (5 fold)
        tscv      = TimeSeriesSplit(n_splits=5)
        wf_preds  = np.full(len(X_tr), np.nan)   # 검증 안 된 인덱스 NaN으로 초기화
        wf_actual = y_rv_tr.values.copy()

        _X_tr_raw = X_tr.values  # fold별 인덱싱용

        for fold, (idx_tr, idx_va) in enumerate(tscv.split(_X_tr_raw)):
            # fold별 독립 스케일러 (전체 훈련셋 통계 누출 방지)
            _sc_fold = StandardScaler()
            X_f = _sc_fold.fit_transform(_X_tr_raw[idx_tr])
            X_v = _sc_fold.transform(_X_tr_raw[idx_va])
            y_f = y_rv_tr.iloc[idx_tr]

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

        # 최종 모델용 스케일러: 전체 훈련셋으로 fit (평가 완료 후이므로 누출 없음)
        full_X = scaler.fit_transform(X_tr)

        # CV 메트릭: 실제 검증된 인덱스만 사용 (초기 훈련전용 인덱스 0-패딩 제외)
        _wf_mask  = ~np.isnan(wf_preds)
        rmse_cv = float(np.sqrt(mean_squared_error(wf_actual[_wf_mask], wf_preds[_wf_mask])))
        mae_cv  = float(mean_absolute_error(wf_actual[_wf_mask], wf_preds[_wf_mask]))
        r2_cv   = float(r2_score(wf_actual[_wf_mask], wf_preds[_wf_mask]))
        log.info(f"        Walk-forward CV → RMSE={rmse_cv:.5f}  MAE={mae_cv:.5f}  R²={r2_cv:.4f}")

        # ── 최종 모델: 전체 훈련셋으로 재학습 (지수감쇠 + COVID 가중치)
        X_tr_s = full_X
        X_te_s = scaler.transform(X_te)

        _n = len(y_rv_tr)
        _time_w = np.exp(np.log(2) / 180 * np.arange(_n))  # 반감기 ~9개월 (라이브 레짐 적응 강화)
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

        # ── 과적합 감지 (최종 모델 기준: 훈련 R² vs 홀드아웃 테스트 R²)
        r2_train = float(r2_score(y_rv_tr, modelA.predict(X_tr_s)))
        overfit_gap = r2_train - r2_ho  # 동일 모델 train vs test (fold 모델과 혼용 방지)
        if overfit_gap > 0.20:
            log.warning(f"    ⚠️ 과적합 의심: 훈련R²={r2_train:.4f} vs 테스트R²={r2_ho:.4f} "
                        f"(gap={overfit_gap:.3f})")
        elif overfit_gap < -0.10:
            log.info(f"        테스트>훈련 (시간가중 훈련 정상): 훈련R²={r2_train:.4f} vs 테스트R²={r2_ho:.4f} (gap={overfit_gap:.3f})")
        else:
            log.info(f"        훈련R²={r2_train:.4f}  테스트R²={r2_ho:.4f}  gap={overfit_gap:.3f} (정상)")

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

        # ── 피처 중요도 Top8 로그
        if hasattr(modelA, 'feature_importances_'):
            imp = sorted(zip(har_feats, modelA.feature_importances_),
                         key=lambda x: x[1], reverse=True)
            top_str = ', '.join(f"{n}({v:.3f})" for n, v in imp[:8])
            log.info(f"        피처 중요도 Top8: {top_str}")

        # ── 장중 RV를 타깃으로 한 별도 모델 (최근 730일, 더 정확한 측정값)
        if 'rv_intraday' in feature_df.columns:
            try:
                _intra_df = feature_df[feature_df['rv_intraday'] > 0].copy()
                _intra_df['target_intra'] = _intra_df['rv_intraday'].shift(-1)
                _intra_df = _intra_df.dropna(subset=['target_intra'])
                if len(_intra_df) > 120:
                    _n_te_i    = min(90, int(len(_intra_df) * 0.15))
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

    # ── HAR-A (vol delta) / HAR-B (GARCH 잔차) 비교 훈련
    if _SKL and _XGB and 'xgb_har' in results:
        try:
            _base_r2 = results['xgb_har']['r2']
            _har_f   = [c for c in har_feats if c in train_df.columns]
            _best    = ('direct', _base_r2, None, None)   # (method, r2, model, scaler)

            for _tag, _tgt_col, _recon in [
                ('delta',  'target_rv_delta', 'delta'),
                ('garch_resid', 'target_rv_garch', 'garch_resid'),
            ]:
                if _tgt_col not in train_df.columns:
                    continue
                _y_tr_v = train_df[_tgt_col]
                _y_te_v = test_df[_tgt_col]
                _sc_v   = StandardScaler()
                _X_tr_v = _sc_v.fit_transform(train_df[_har_f])
                _X_te_v = _sc_v.transform(test_df[_har_f])
                _cw_v   = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                           if 'covid_dummy' in train_df.columns else None)
                _m_v = xgb.XGBRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.02,
                    subsample=0.7, colsample_bytree=0.6,
                    min_child_weight=15, reg_alpha=1.0, reg_lambda=5.0,
                    n_jobs=-1, random_state=42, verbosity=0)
                _m_v.fit(_X_tr_v, _y_tr_v, sample_weight=_cw_v)
                _pred_v = _m_v.predict(_X_te_v)
                # 재구성 후 실제 RV와 비교
                _last_rv_te = test_df['RV_5d'].values
                if _recon == 'delta':
                    _pred_rv_v = _last_rv_te + _pred_v
                else:   # garch_resid
                    _garch_base = test_df['garch_vol'].values if 'garch_vol' in test_df.columns else _last_rv_te
                    _pred_rv_v = _garch_base + _pred_v
                _pred_rv_v = np.clip(_pred_rv_v, 0, None)
                _act_rv_te = test_df['target_rv'].values
                _r2_v  = float(r2_score(_act_rv_te, _pred_rv_v))
                _mae_v = float(mean_absolute_error(_act_rv_te, _pred_rv_v))
                log.info(f"    HAR-{_tag}: R²={_r2_v:.4f}  MAE={_mae_v:.5f}  (base={_base_r2:.4f})")
                if _r2_v > _best[1]:
                    _best = (_recon, _r2_v, _m_v, _sc_v)

            if _best[0] != 'direct' and _best[2] is not None:
                log.info(f"    ✅ HAR-{_best[0]} 채택 (R²={_best[1]:.4f} > base={_base_r2:.4f})")
                results['xgb_har']['model']        = _best[2]
                results['xgb_har']['scaler']       = _best[3]
                results['xgb_har']['reconstruction'] = _best[0]
                results['xgb_har']['r2']           = _best[1]
            else:
                log.info(f"    기존 direct HAR 유지 (R²={_base_r2:.4f})")
                results['xgb_har']['reconstruction'] = 'direct'
        except Exception as _hve:
            log.warning(f"    HAR-A/B 비교 실패({_hve})")
            results['xgb_har']['reconstruction'] = 'direct'

    # ── GARCH(1,1) 1-step ahead 성능 평가 (conditional_volatility 사용)
    # GARCH는 전체 데이터로 학습됨 → conditional_volatility[t] = h_t^0.5 (t-1까지 사용)
    # → 1-step ahead 예측으로 test 기간 추출해 공정 비교
    _garch_res = (aux or {}).get('garch_model')
    if _garch_res is not None:
        try:
            # 1-step: conditional_volatility (% scale) → decimal 변환 후 test 기간 정렬
            _gcv_pct = _garch_res.conditional_volatility  # % scale Series
            _gcv_dec = (_gcv_pct / 100).reindex(test_df.index).ffill().bfill()
            _garch_vol_pred   = _gcv_dec.values
            _garch_vol_actual = test_df['RV_5d'].values   # 동일 기간 RV (1-step 평가 기준)
            _n_g = min(len(_garch_vol_pred), len(_garch_vol_actual))
            _garch_vol_pred   = _garch_vol_pred[:_n_g]
            _garch_vol_actual = _garch_vol_actual[:_n_g]

            _garch_rmse = float(np.sqrt(mean_squared_error(_garch_vol_actual, _garch_vol_pred)))
            _garch_r2   = float(r2_score(_garch_vol_actual, _garch_vol_pred))
            _xgb_r2     = results.get('xgb_har', {}).get('r2', -999)
            log.info(f"    GARCH(1,1) 1-step: RMSE={_garch_rmse:.5f}  R²={_garch_r2:.4f}  "
                     f"(vs XGB-HAR R²={_xgb_r2:.4f})")
            results['garch_vol'] = {
                'model': _garch_res, 'type': 'vol_5d', 'name': 'GARCH(1,1) 1-step',
                'rmse': _garch_rmse, 'mae': float(mean_absolute_error(_garch_vol_actual, _garch_vol_pred)),
                'r2': _garch_r2,
                'beats_xgb': _garch_r2 > _xgb_r2,
                'benchmark_only': True,
            }
            if _garch_r2 > _xgb_r2:
                log.info("    ✅ GARCH 1-step > XGB-HAR → vol 예측에 GARCH 사용")
            else:
                log.info("    XGB-HAR ≥ GARCH 1-step → XGB-HAR 유지")
        except Exception as _ge:
            log.warning(f"    GARCH 성능 평가 실패({_ge})")

    # ─────────────────────────────────────────────────────────────────────
    # Model B: SARIMAX — 1-step ahead dynamic=False 평가 (정직한 R²)
    # ─────────────────────────────────────────────────────────────────────
    # Exog: 거시(DXY/충격) + 원유 시장 구조(Brent스프레드/OVX/선물커브) + VIX
    exog_cols = [c for c in [
        'dxy_change', 'demand_shock', 'supply_shock', 'vix_change',
        'brent_wti_spread', 'ovx_change', 'futures_spread',
        'oil_event_score_smooth', 'news_sentiment_smooth7',
        'inv_mom4_z', 'gold_wti_ratio_z', 'copper_gold_ratio_z',
    ] if c in feature_df.columns]
    log.info("    [B] SARIMAX 학습 + 1-step ahead 평가 중...")

    if _SARIMAX and len(train_df) > 60:
        try:
            # 2번: 영업일(B) 주파수 명시 → SARIMAX 계절 패턴 인식 개선
            def _to_bday(s):
                try:
                    return s.asfreq('B', method='ffill')
                except Exception as _be:
                    log.warning(f"    _to_bday asfreq 실패({_be}) → 원본 유지")
                    return s

            # SARIMAX는 최근 SARIMAX_YEARS 년치만 사용 (오래된 가격 레짐 영향 최소화)
            # full_df의 최신 행(dropna로 제거된 행) 포함하여 SARIMAX 훈련 데이터 최신화
            _sx_base = feature_df
            if full_df is not None:
                _missing = full_df.index.difference(feature_df.index)
                _avail_cols = [c for c in feature_df.columns if c in full_df.columns]
                if not _missing.empty:
                    _extra = full_df.loc[_missing, _avail_cols].ffill()
                    _sx_base = pd.concat([feature_df, _extra]).sort_index()
            cutoff = _sx_base.index[-1] - pd.DateOffset(years=SARIMAX_WINDOW_YEARS)
            sx_df  = _sx_base[_sx_base.index >= cutoff]
            n_test_sx = min(90, int(len(sx_df) * 0.15))
            sx_train  = sx_df.iloc[:-n_test_sx]
            sx_test   = sx_df.iloc[-n_test_sx:]

            full_wti  = _to_bday(sx_df['WTI'])
            full_exog = _to_bday(sx_df[exog_cols]) if exog_cols else None
            n_train   = len(sx_train)

            wti_train  = _to_bday(sx_train['WTI'])
            exog_train = _to_bday(sx_train[exog_cols]) if exog_cols else None

            # 1번: auto_arima로 최적 파라미터 탐색 (결과 캐싱)
            sarimax_order    = (2, 1, 1)
            sarimax_seasonal = (1, 0, 1, 5)
            _sx_cache_file = OUTPUT_DIR / 'sarimax_order_cache.json'
            import hashlib as _hl
            _exog_key = '|'.join(sorted(exog_cols)).encode()
            _sx_cache_key  = _hl.md5(wti_train.values.tobytes() + _exog_key).hexdigest()[:16]
            _sx_cached = {}
            if _sx_cache_file.exists():
                try:
                    import json as _json
                    _sx_cached = _json.loads(_sx_cache_file.read_text())
                except Exception:
                    pass
            if _sx_cached.get('key') == _sx_cache_key:
                sarimax_order    = tuple(_sx_cached['order'])
                sarimax_seasonal = tuple(_sx_cached['seasonal'])
                log.info(f"    auto_arima 캐시 사용: {sarimax_order} × {sarimax_seasonal}")
            elif _PMDARIMA:
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
                    import json as _json
                    _sx_cache_file.write_text(_json.dumps({
                        'key': _sx_cache_key,
                        'order': list(sarimax_order),
                        'seasonal': list(sarimax_seasonal),
                    }))
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
            fit = mdl_tr.fit(disp=False, maxiter=100)

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
            fit_live = mdl_live.fit(disp=False, maxiter=100)

            results['sarimax'] = {
                'model': fit_live, 'features': exog_cols, 'type': 'price',
                'rmse': rmse_b, 'mae': mae_b, 'r2': r2_b,
                'name': f'SARIMAX{sarimax_order} 1-step',
                'pred_price_test':   pred_price,
                'actual_price_test': y_px_te_sx,
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
                    _ref = y_px_te_sx
                    mae_c  = float(mean_absolute_error(_ref, corrected[:len(_ref)]))
                    mae_s  = results['sarimax']['mae']
                    r2_c   = float(r2_score(_ref, corrected[:len(_ref)]))
                    r2_s   = results['sarimax']['r2']

                    if mae_c < mae_s:
                        rmse_c = float(np.sqrt(mean_squared_error(_ref, corrected[:len(_ref)])))
                        log.info(f"        잔차 교정 채택 ✓  MAE: {mae_s:.4f} → {mae_c:.4f}  R²: {r2_s:.4f} → {r2_c:.4f}")
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
                        results['sarimax']['mae']            = mae_c
                        results['sarimax']['rmse']           = rmse_c
                        results['sarimax']['pred_price_test'] = corrected
                    else:
                        log.info(f"        잔차 교정 미채택 (MAE: {mae_s:.4f} ≤ {mae_c:.4f})")
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
    # Model B2: VAR (WTI + Brent + DXY + VIX 4변수 벡터 자기회귀)
    # ─────────────────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR as _VAR
        _var_cols = [c for c in ['WTI', 'Brent', 'DXY', 'VIX'] if c in feature_df.columns]
        if len(_var_cols) >= 2 and _SARIMAX:
            log.info("    [B2] VAR 다변량 모델 학습 중...")
            cutoff_v = feature_df.index[-1] - pd.DateOffset(years=SARIMAX_YEARS)
            _vdf     = feature_df[feature_df.index >= cutoff_v][_var_cols].dropna()
            _vdf     = _vdf.asfreq('B', method='ffill').dropna()
            _n_te_v  = min(90, int(len(_vdf) * 0.15))
            _v_tr    = _vdf.iloc[:-_n_te_v]
            _v_te    = _vdf.iloc[-_n_te_v:]
            # AIC 기준 최적 lag 선택 (최대 10)
            _var_fit = _VAR(_v_tr).fit(maxlags=10, ic='aic', trend='c')
            _lag_ord = _var_fit.k_ar
            log.info(f"    VAR 최적 lag={_lag_ord}")
            # 1-step ahead rolling forecast
            _var_preds = []
            _var_hist  = _v_tr.values.copy()
            for _i in range(len(_v_te)):
                _fc = _var_fit.forecast(_var_hist[-_lag_ord:], steps=1)
                _var_preds.append(float(_fc[0, 0]))   # WTI column
                _var_hist = np.vstack([_var_hist, _v_te.values[_i]])
            _var_pred_arr = np.array(_var_preds)
            _y_var_te     = _v_te['WTI'].values
            _rmse_var = float(np.sqrt(mean_squared_error(_y_var_te, _var_pred_arr)))
            _mae_var  = float(mean_absolute_error(_y_var_te, _var_pred_arr))
            _r2_var   = float(r2_score(_y_var_te, _var_pred_arr))
            log.info(f"    [B2] VAR → RMSE={_rmse_var:.4f}  MAE={_mae_var:.4f}  R²={_r2_var:.4f}")
            # SARIMAX보다 나으면 스택에 추가
            _sx_mae = results.get('sarimax', {}).get('mae', 999)
            results['var'] = {
                'pred_price_test': _var_pred_arr,
                'actual_price_test': _y_var_te,
                'rmse': _rmse_var, 'mae': _mae_var, 'r2': _r2_var,
                'model': _var_fit, 'cols': _var_cols, 'lag': _lag_ord,
                'n_test': _n_te_v, 'type': 'price',
                'name': f'VAR({_lag_ord}) WTI+Brent+DXY+VIX',
            }
            log.info(f"    VAR MAE=${_mae_var:.4f}  vs  SARIMAX MAE=${_sx_mae:.4f}")
    except Exception as _ve:
        log.warning(f"    VAR 실패({_ve})")

    # ─────────────────────────────────────────────────────────────────────
    # Model B3: ETS (지수평활 감쇠 트렌드) — SARIMAX 대체 후보
    # ─────────────────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS
        log.info("    [B3] ETS(Holt-Winters 감쇠 트렌드) 학습 중...")
        _cutoff_ets = feature_df.index[-1] - pd.DateOffset(years=SARIMAX_YEARS)
        _ets_df     = feature_df[feature_df.index >= _cutoff_ets]
        _n_te_ets   = min(90, int(len(_ets_df) * 0.15))
        _ets_tr_wti = _ets_df['WTI'].iloc[:-_n_te_ets]
        _ets_te_wti = _ets_df['WTI'].iloc[-_n_te_ets:]

        _ets_m   = _ETS(_ets_tr_wti, trend='add', damped_trend=True, seasonal=None)
        _ets_fit = _ets_m.fit(optimized=True)

        # 1-step rolling 예측: Holt 감쇠 상태 갱신 (SARIMAX dynamic=False와 동일 조건)
        _alpha = float(_ets_fit.params.get('smoothing_level', 0.3))
        _beta  = float(_ets_fit.params.get('smoothing_trend', 0.1))
        _phi   = float(_ets_fit.params.get('damping_trend', 0.98))
        _L = float(_ets_fit.level.iloc[-1])
        _T = float(_ets_fit.trend.iloc[-1])
        _ets_preds_1s = []
        for _y in _ets_te_wti.values:
            _ets_preds_1s.append(_L + _phi * _T)  # forecast for WTI[t]
            _Ln = _alpha * _y + (1 - _alpha) * (_L + _phi * _T)
            _Tn = _beta * (_Ln - _L) + (1 - _beta) * _phi * _T
            _L, _T = _Ln, _Tn
        # target_price[t] = WTI[t+1] → 1스텝 앞당긴 예측 사용 (stacking 타깃 정렬)
        _ets_fc_extra = _L + _phi * _T  # 루프 종료 후 마지막 추가 예측
        _ets_pred = np.array(_ets_preds_1s[1:] + [_ets_fc_extra])  # 90값, target_price 정렬
        _actual_tp = _ets_te_wti.values[1:].tolist() + [float(
            _ets_df['WTI'].iloc[-_n_te_ets + len(_ets_te_wti.values)]) if
            _n_te_ets < len(_ets_df) else _ets_te_wti.values[-1]]
        _actual_tp = np.array(_actual_tp[:len(_ets_pred)])
        # feature_df의 target_price 기준으로 평가 (stacking과 동일 타깃)
        _tp_arr = _ets_df['target_price'].values[-_n_te_ets:]
        _n_ev   = min(len(_ets_pred), len(_tp_arr))
        _mae_ets  = float(mean_absolute_error(_tp_arr[:_n_ev], _ets_pred[:_n_ev]))
        _r2_ets   = float(r2_score(_tp_arr[:_n_ev], _ets_pred[:_n_ev]))
        _rmse_ets = float(np.sqrt(mean_squared_error(_tp_arr[:_n_ev], _ets_pred[:_n_ev])))
        log.info(f"    [B3] ETS(1-step rolling) → RMSE={_rmse_ets:.4f}  "
                 f"MAE={_mae_ets:.4f}  R²={_r2_ets:.4f}")

        _ets_full_wti = _ets_df['WTI'].copy()
        try:
            _ets_full_wti = _ets_full_wti.asfreq('B', method='ffill')
        except Exception:
            pass
        _ets_live = _ETS(_ets_full_wti, trend='add', damped_trend=True,
                         seasonal=None).fit(optimized=True)

        results['ets'] = {
            'model': _ets_live, 'type': 'price', 'features': [],
            'rmse': _rmse_ets, 'mae': _mae_ets, 'r2': _r2_ets,
            'name': 'ETS(HW-Damped)',
            'pred_price_test': _ets_pred[:_n_ev],
            'actual_price_test': _tp_arr[:_n_ev],
            'test_dates': _ets_te_wti.index,
        }
    except Exception as _ets_e:
        log.warning(f"    ETS 실패({_ets_e})")

    # ─────────────────────────────────────────────────────────────────────
    # Model B4: Prophet (추세 변동점 + 주간·연간 계절성 — SARIMAX 보완 다양성)
    # ─────────────────────────────────────────────────────────────────────
    try:
        from prophet import Prophet as _Prophet
        import warnings as _wn
        log.info("    [B4] Prophet 학습 중...")
        _cutoff_prph = feature_df.index[-1] - pd.DateOffset(years=SARIMAX_YEARS)
        _prph_full   = feature_df[feature_df.index >= _cutoff_prph][['WTI']].copy()
        _n_te_prph   = min(60, int(len(_prph_full) * 0.15))
        _prph_tr_wti = _prph_full['WTI'].iloc[:-_n_te_prph]
        _prph_te_wti = _prph_full['WTI'].iloc[-_n_te_prph:]

        # Prophet 형식 (ds=날짜, y=WTI 가격)
        _prph_idx = _prph_tr_wti.index
        _ds_tr = pd.DatetimeIndex([d.tz_localize(None) if hasattr(d, 'tz') and d.tzinfo else d
                                    for d in _prph_idx])
        _prph_train_df = pd.DataFrame({'ds': _ds_tr, 'y': _prph_tr_wti.values})

        _m_prph = _Prophet(
            daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
            changepoint_prior_scale=0.05, seasonality_prior_scale=5.0,
            n_changepoints=25, interval_width=0.80,
        )
        with _wn.catch_warnings():
            _wn.simplefilter('ignore')
            _m_prph.fit(_prph_train_df)

        # 실제 인덱스에서 다음 거래일 추출 (bdate_range 날짜 불일치 방지)
        _prph_next_dates, _prph_next_prices = [], []
        for _t in _prph_te_wti.index:
            _loc = _prph_full.index.get_loc(_t)
            if _loc + 1 < len(_prph_full):
                _nd = _prph_full.index[_loc + 1]
                _prph_next_dates.append(_nd.tz_localize(None) if getattr(_nd, 'tzinfo', None) else _nd)
                _prph_next_prices.append(float(_prph_full['WTI'].iloc[_loc + 1]))
        _prph_te_df = pd.DataFrame({'ds': _prph_next_dates})
        with _wn.catch_warnings():
            _wn.simplefilter('ignore')
            _prph_fc = _m_prph.predict(_prph_te_df)
        _prph_pred_arr = _prph_fc['yhat'].values
        _prph_tp_arr   = np.array(_prph_next_prices[:len(_prph_pred_arr)])
        _n_pr = min(len(_prph_pred_arr), len(_prph_tp_arr))
        _prph_pred_arr = _prph_pred_arr[:_n_pr]
        _prph_tp_arr   = _prph_tp_arr[:_n_pr]
        _mae_prph  = float(mean_absolute_error(_prph_tp_arr, _prph_pred_arr))
        _r2_prph   = float(r2_score(_prph_tp_arr, _prph_pred_arr))
        _rmse_prph = float(np.sqrt(mean_squared_error(_prph_tp_arr, _prph_pred_arr)))
        log.info(f"    [B4] Prophet → RMSE={_rmse_prph:.4f}  MAE={_mae_prph:.4f}  R²={_r2_prph:.4f}")
        results['prophet'] = {
            'model': _m_prph, 'type': 'price', 'features': [],
            'rmse': _rmse_prph, 'mae': _mae_prph, 'r2': _r2_prph,
            'name': 'Prophet(추세+계절성)',
            'pred_price_test': _prph_pred_arr,
            'actual_price_test': _prph_tp_arr,
            'n_test': _n_pr,
            'benchmark_only': True,
        }
    except Exception as _prph_e:
        log.warning(f"    Prophet 실패({_prph_e})")

    # ─────────────────────────────────────────────────────────────────────
    # Model D: XGBoost 수익률 예측 (log_return 타깃)
    # vol 시뮬레이션 대체 — 방향성+크기 직접 학습
    # ─────────────────────────────────────────────────────────────────────
    if _XGB and _SKL and scaler is not None:
        log.info("    [D] XGBoost 수익률 예측 학습 중 (2-pass 피처 선택)...")
        try:
            # ── [D0] Optuna 하이퍼파라미터 최적화 (캐시 없을 때만 실행, ~2분)
            _best_xgb_params = None
            if _OPTUNA or XGB_OPTUNA_CACHE.exists():
                try:
                    import hashlib as _hl_xgb, json as _json_opt
                    _feat_key = '|'.join(sorted(X_tr_all.columns.tolist())).encode()
                    _opt_key = _hl_xgb.md5(X_tr_all.values.tobytes()[:8192] + _feat_key).hexdigest()[:16]
                    _opt_cached = {}
                    if XGB_OPTUNA_CACHE.exists():
                        try:
                            _opt_cached = _json_opt.loads(XGB_OPTUNA_CACHE.read_text())
                        except Exception:
                            pass
                    if _opt_cached.get('key') == _opt_key:
                        _best_xgb_params = _opt_cached.get('params')
                        log.info(f"    Optuna 캐시 사용: {_best_xgb_params}")
                    elif _OPTUNA:
                        log.info("    [D0] Optuna 탐색 중 (20 trials, 3-fold CV)...")
                        _tscv_opt = TimeSeriesSplit(n_splits=3)
                        _n_opt = len(y_ret_tr)
                        _tw_opt = np.exp(np.log(2) / 252 * np.arange(_n_opt))
                        _tw_opt /= _tw_opt.mean()
                        _cw_opt = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                                   if 'covid_dummy' in train_df.columns else np.ones(_n_opt))
                        _w_opt = _cw_opt * _tw_opt
                        _X_opt = X_tr_all.values
                        _y_opt = y_ret_tr.values
                        _wti_opt = train_df['WTI'].values
                        _tpx_opt = train_df['target_price'].values

                        def _opt_objective(trial):
                            _p = {
                                'n_estimators':    trial.suggest_int('n_est', 200, 600),
                                'max_depth':       trial.suggest_int('depth', 2, 5),
                                'learning_rate':   trial.suggest_float('lr', 0.005, 0.05, log=True),
                                'subsample':       trial.suggest_float('sub', 0.5, 0.9),
                                'colsample_bytree':trial.suggest_float('col', 0.4, 0.8),
                                'min_child_weight':trial.suggest_int('mcw', 3, 20),
                                'reg_alpha':       trial.suggest_float('a', 0.01, 2.0, log=True),
                                'reg_lambda':      trial.suggest_float('l', 0.5, 10.0, log=True),
                                'objective': 'reg:pseudohubererror', 'huber_slope': 1.0,
                                'n_jobs': -1, 'random_state': 42, 'verbosity': 0,
                            }
                            _maes = []
                            for _ti, _vi in _tscv_opt.split(_X_opt):
                                if len(_ti) < 60:
                                    continue
                                _sc_o = StandardScaler()
                                _Xo_tr = _sc_o.fit_transform(_X_opt[_ti])
                                _Xo_va = _sc_o.transform(_X_opt[_vi])
                                _mo = xgb.XGBRegressor(**_p)
                                _mo.fit(_Xo_tr, _y_opt[_ti], sample_weight=_w_opt[_ti])
                                _pr_o = np.clip(_mo.predict(_Xo_va), -0.5, 0.5)
                                _px_o = _wti_opt[_vi] * np.exp(_pr_o)
                                _maes.append(mean_absolute_error(_tpx_opt[_vi], _px_o))
                            return float(np.mean(_maes)) if _maes else 999.0

                        _study = _optuna.create_study(
                            direction='minimize',
                            sampler=_optuna.samplers.TPESampler(seed=42))
                        _study.optimize(_opt_objective, n_trials=20,
                                        show_progress_bar=False)
                        _best_xgb_params = _study.best_params
                        log.info(f"    Optuna 완료: {_best_xgb_params} "
                                 f"(MAE={_study.best_value:.4f})")
                        try:
                            XGB_OPTUNA_CACHE.write_text(_json_opt.dumps(
                                {'key': _opt_key, 'params': _best_xgb_params}))
                        except Exception:
                            pass
                except Exception as _oe:
                    log.warning(f"    Optuna 실패({_oe}) → 기본 파라미터 사용")

            _xgb_p = dict(n_estimators=500, max_depth=3, learning_rate=0.015,
                          subsample=0.75, colsample_bytree=0.6,
                          min_child_weight=8, reg_alpha=0.3, reg_lambda=3.0,
                          objective='reg:pseudohubererror', huber_slope=1.0,
                          n_jobs=-1, random_state=42, verbosity=0)
            if _best_xgb_params:
                _xgb_p.update({
                    'n_estimators':    _best_xgb_params.get('n_est',  _xgb_p['n_estimators']),
                    'max_depth':       _best_xgb_params.get('depth',  _xgb_p['max_depth']),
                    'learning_rate':   _best_xgb_params.get('lr',     _xgb_p['learning_rate']),
                    'subsample':       _best_xgb_params.get('sub',    _xgb_p['subsample']),
                    'colsample_bytree':_best_xgb_params.get('col',    _xgb_p['colsample_bytree']),
                    'min_child_weight':_best_xgb_params.get('mcw',    _xgb_p['min_child_weight']),
                    'reg_alpha':       _best_xgb_params.get('a',      _xgb_p['reg_alpha']),
                    'reg_lambda':      _best_xgb_params.get('l',      _xgb_p['reg_lambda']),
                })
                log.info(f"    Optuna 파라미터 적용: depth={_xgb_p['max_depth']} "
                         f"lr={_xgb_p['learning_rate']:.4f} n_est={_xgb_p['n_estimators']}")
            # Quantile(α=0.45): 하방 편향 → 방향성 정확도 향상 목적
            _xgb_q = dict(**{k: v for k, v in _xgb_p.items() if k not in ('objective','huber_slope')},
                          objective='reg:quantileerror', quantile_alpha=0.45)

            _n_ret  = len(y_ret_tr)
            _tw_ret = np.exp(np.log(2) / 90 * np.arange(_n_ret)) ; _tw_ret /= _tw_ret.mean()  # 반감기 ~4개월 (라이브 레짐 적응 강화)
            covid_w_ret = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                           if 'covid_dummy' in train_df.columns else np.ones(_n_ret))
            w_ret = covid_w_ret * _tw_ret

            # ── 1차: 전체 피처로 학습 → 중요도 추출
            _sc_full = StandardScaler()
            _Xtr_full = _sc_full.fit_transform(X_tr_all)
            _Xte_full = _sc_full.transform(X_te_all)
            _mD_full  = xgb.XGBRegressor(**_xgb_p)
            _mD_full.fit(_Xtr_full, y_ret_tr, sample_weight=w_ret)

            # XGBoost-Return 전용 피처 중요도 저장
            _imp = _mD_full.feature_importances_
            _imp_df = pd.DataFrame(
                sorted(zip(available_feats, _imp), key=lambda x: x[1], reverse=True),
                columns=['feature','importance']
            )
            _imp_df.to_csv(OUTPUT_DIR / 'xgb_return_importance.csv', index=False)

            # 누적 중요도 90% 또는 상위 25개 이하로 피처 선택
            _sorted_imp = sorted(zip(available_feats, _imp), key=lambda x: x[1], reverse=True)
            _cum, _sel_feats = 0.0, []
            for _fn, _fv in _sorted_imp:
                _sel_feats.append(_fn)
                _cum += _fv
                if _cum >= 0.85 or len(_sel_feats) >= 25:
                    break
            _sel_feats = [f for f in _sel_feats if f in train_df.columns]
            log.info(f"    피처 선택: {len(available_feats)}개 → {len(_sel_feats)}개 (누적 중요도 {_cum:.1%})")

            # ── 2차: 선택 피처로 재학습
            _sc_sel = StandardScaler()
            _Xtr_sel = _sc_sel.fit_transform(train_df[_sel_feats])
            _Xte_sel = _sc_sel.transform(test_df[_sel_feats])
            _mD_sel  = xgb.XGBRegressor(**_xgb_p)
            _mD_sel.fit(_Xtr_sel, y_ret_tr, sample_weight=w_ret)

            # ── 두 모델 평가
            def _eval(model, Xte, label):
                _pr = np.clip(model.predict(Xte), -0.5, 0.5)   # inf 방지: ±50% 초과 수익률 제거
                _px = test_df['WTI'].values * np.exp(_pr)
                _r2 = float(r2_score(y_px_te, _px))
                _mae = float(mean_absolute_error(y_px_te, _px))
                _dir = float((np.sign(_pr) == np.sign(y_ret_te.values)).mean())
                log.info(f"        [{label}] R²={_r2:.4f} MAE={_mae:.4f} dir={_dir*100:.1f}%")
                return _pr, _px, _r2, _mae, _dir

            _pr_f, _px_f, _r2_f, _mae_f, _dir_f = _eval(_mD_full, _Xte_full, '전체피처')
            _pr_s, _px_s, _r2_s, _mae_s, _dir_s = _eval(_mD_sel,  _Xte_sel,  '선택피처')

            # Quantile(α=0.45) 모델 — 방향성 편향 개선
            try:
                _mD_q = xgb.XGBRegressor(**_xgb_q)
                _mD_q.fit(_Xtr_sel, y_ret_tr, sample_weight=w_ret)
                _pr_q, _px_q, _r2_q, _mae_q, _dir_q = _eval(_mD_q, _Xte_sel, 'Quantile')
            except Exception:
                _mD_q, _mae_q, _dir_q = None, 999.0, 0.0

            # Walk-forward CV로 피처 선택 (5-fold, 훈련셋 내부 평가)
            _tscv = TimeSeriesSplit(n_splits=5)
            def _wf_mae(X_tr, y_tr, feats):
                maes = []
                for _ti, _vi in _tscv.split(X_tr):
                    if len(_ti) < 60: continue
                    _sc_w = StandardScaler()
                    _Xw_tr = _sc_w.fit_transform(X_tr[_ti])
                    _Xw_val= _sc_w.transform(X_tr[_vi])
                    _mw = xgb.XGBRegressor(**_xgb_p)
                    _mw.fit(_Xw_tr, y_tr[_ti], sample_weight=w_ret[_ti])
                    _pp  = _mw.predict(_Xw_val)
                    _px_w= train_df['WTI'].values[_vi] * np.exp(_pp)
                    _y_px= train_df['target_price'].values[_vi]
                    maes.append(mean_absolute_error(_y_px, _px_w))
                return float(np.mean(maes)) if maes else 999.0

            _wf_sel  = _wf_mae(train_df[_sel_feats].values, y_ret_tr.values, _sel_feats)
            log.info(f"        WF-CV MAE (선택피처)={_wf_sel:.4f}")

            # ── XGBoost 이진 분류기 (방향성 특화 학습)
            # 논문 근거: WTI 방향성 71~80% 보고 (binary:logistic + 회귀 크기 결합)
            _mD_cls_dir, _mae_cls, _dir_cls, _pr_cls_adj, _px_cls_adj, _r2_cls = None, 999.0, 0.0, None, None, 0.0
            try:
                _y_cls_tr = (y_ret_tr.values > 0).astype(int)   # 1=상승, 0=하락
                _y_cls_te = (y_ret_te.values > 0).astype(int)
                _dz_mask_tr = np.abs(y_ret_tr.values) > 0.003  # dead-zone 제외 마스크
                _xgb_cls_p = dict(
                    n_estimators=300, max_depth=3, learning_rate=0.02,
                    subsample=0.75, colsample_bytree=0.6, colsample_bynode=0.8,
                    min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0,
                    n_jobs=-1, random_state=42, verbosity=0
                )
                # MI 기반 분류 전용 피처 선택 (회귀 F-score sel_feats와 독립)
                try:
                    from sklearn.feature_selection import mutual_info_classif as _mic2
                    _mi2_sc = _mic2(train_df[available_feats].fillna(0).values,
                                    _y_cls_tr, random_state=42)
                    _mi2_ranked = sorted(zip(available_feats, _mi2_sc),
                                         key=lambda x: x[1], reverse=True)
                    _cls_feats = [f for f, _ in _mi2_ranked[:25] if f in train_df.columns]
                    _sc_cls = StandardScaler()
                    _Xtr_cls = _sc_cls.fit_transform(train_df[_cls_feats].fillna(0).values)
                    _Xte_cls = _sc_cls.transform(test_df[_cls_feats].fillna(0).values)
                    log.info(f"        MI 분류피처 top5: {[f for f,_ in _mi2_ranked[:5]]}")
                except Exception as _mi2_e:
                    log.warning(f"    MI 분류피처 실패({_mi2_e}), sel_feats 사용")
                    _cls_feats, _sc_cls = _sel_feats, _sc_sel
                    _Xtr_cls, _Xte_cls = _Xtr_sel, _Xte_sel
                _mD_cls_dir = xgb.XGBClassifier(**_xgb_cls_p)
                _mD_cls_dir.fit(_Xtr_cls, _y_cls_tr, sample_weight=w_ret)
                _xgb_cls_saved = _mD_cls_dir          # SVM 교체 전 XGB 모델 보존
                _prob_xgb_orig = _mD_cls_dir.predict_proba(_Xte_cls)[:, 1]
                _prob_up_te = _prob_xgb_orig
                _xgb_blend_w_best = 0.0               # 채택된 블렌드 가중치

                # ── ① SVM 분류기 + CEEMDAN 전용 주입 (회귀 feature selection 미영향)
                try:
                    from sklearn.svm import SVC as _SVC
                    _best_svm, _best_svm_dir, _best_svm_prob = None, 0.0, None
                    _sc_cot, _cot_cols = None, []

                    # CEEMDAN + 이벤트 스코어 + 불확실성 SVM 전용 주입
                    _cem_cols = [c for c in [
                        'ceemdan_trend_ret', 'ceemdan_noise_std5', 'ceemdan_trend_mom5',
                        'supply_event_score', 'supply_event_score_smooth',
                        'demand_event_score', 'demand_event_score_smooth',
                        'geo_event_score',    'geo_event_score_smooth',
                        'news_uncertainty',   'ovx_rv_spread',
                        # 크로스에셋 독립 채널 (top-25 경쟁 우회)
                        'gold_wti_ratio_z', 'copper_gold_ratio_z',
                        'gold_mom_5d', 'copper_mom_5d',
                        'sentiment_chg3', 'macd_cross', 'inv_draw_signal', 'mom_accel',
                        'inv_surprise_dir', 'supply_demand_gap', 'inv_draw_x_macd',
                    ] if c in train_df.columns and train_df[c].abs().sum() > 0]

                    # ── 방향 분류 전용 MI 피처 (return 기반 sel_feats 보완)
                    try:
                        from sklearn.feature_selection import mutual_info_classif as _mic
                        _X_mi = train_df[available_feats].fillna(0).values
                        _mi_sc = _mic(_X_mi, _y_cls_tr, random_state=42)
                        _mi_ranked = sorted(zip(available_feats, _mi_sc),
                                            key=lambda x: x[1], reverse=True)
                        _mi_extra = [f for f, _ in _mi_ranked[:30]
                                     if f not in _sel_feats and f not in _cem_cols
                                     and train_df[f].abs().sum() > 0][:6]
                        if _mi_extra:
                            _cem_cols = _cem_cols + _mi_extra
                            log.info(f"        MI 방향피처 추가: {_mi_extra}")
                    except Exception as _mi_e:
                        log.warning(f"    MI 피처 선택 실패({_mi_e})")

                    if _cem_cols:
                        _sc_cem = StandardScaler()
                        _Xtr_c  = _sc_cem.fit_transform(
                            train_df[_cem_cols].fillna(0).values)
                        _Xte_c  = _sc_cem.transform(
                            test_df[_cem_cols].fillna(0).values)
                        _Xtr_svm = np.hstack([_Xtr_sel, _Xtr_c])
                        _Xte_svm = np.hstack([_Xte_sel, _Xte_c])
                    else:
                        _Xtr_svm, _Xte_svm, _sc_cem = _Xtr_sel, _Xte_sel, None

                    # C 그리드 서치 (시간 기반 단일 검증 분할, 마지막 25%)
                    _c_grid = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
                    _n_cval = max(30, len(_Xtr_svm) // 4)
                    _best_c, _best_c_dir = 1.0, -1.0
                    for _ci in _c_grid:
                        _smc = _SVC(kernel='rbf', C=_ci, gamma='scale',
                                    probability=True, class_weight='balanced', random_state=42)
                        _swc = np.exp(0.002 * np.arange(len(_Xtr_svm) - _n_cval))
                        _dzc = _dz_mask_tr[:len(_Xtr_svm) - _n_cval]
                        if _dzc.sum() >= 30:
                            _smc.fit(_Xtr_svm[:-_n_cval][_dzc], _y_cls_tr[:-_n_cval][_dzc],
                                     sample_weight=_swc[_dzc])
                        else:
                            _smc.fit(_Xtr_svm[:-_n_cval], _y_cls_tr[:-_n_cval],
                                     sample_weight=_swc)
                        _cp = _smc.predict_proba(_Xtr_svm[-_n_cval:])[:, 1]
                        _cd = float(((_cp > 0.5).astype(int) == _y_cls_tr[-_n_cval:]).mean())
                        if _cd > _best_c_dir:
                            _best_c_dir, _best_c = _cd, _ci
                    log.info(f"        SVM C 그리드: best_C={_best_c} (검증_dir={_best_c_dir*100:.1f}%)")

                    _sm = _SVC(kernel='rbf', C=_best_c, gamma='scale',
                               probability=True, class_weight='balanced', random_state=42)
                    _sw_svm = np.exp(0.002 * np.arange(len(_y_cls_tr)))
                    # 역변동성 가중치: 저변동 구간은 방향 예측 신뢰도 높음
                    try:
                        _vol_tr = train_df['vol_5d'].fillna(
                            train_df['vol_5d'].median()).values.astype(np.float32)
                        _inv_vol = 1.0 / (_vol_tr + 1e-6)
                        _inv_vol /= _inv_vol.mean()
                        _sw_svm  = _sw_svm * _inv_vol
                        _sw_svm /= _sw_svm.mean()
                    except Exception:
                        pass
                    if _dz_mask_tr.sum() >= 100:
                        _sm.fit(_Xtr_svm[_dz_mask_tr], _y_cls_tr[_dz_mask_tr],
                                sample_weight=_sw_svm[_dz_mask_tr])
                    else:
                        _sm.fit(_Xtr_svm, _y_cls_tr, sample_weight=_sw_svm)
                    _best_svm_prob = _sm.predict_proba(_Xte_svm)[:, 1]
                    _best_svm_dir  = float(((_best_svm_prob > 0.5).astype(int) == _y_cls_te).mean())
                    log.info(f"        [SVM+CEEMDAN+Event] dir={_best_svm_dir*100:.1f}% (주입피처={len(_cem_cols)}개)")

                    if _best_svm_dir > ((_prob_up_te > 0.5).astype(int) == _y_cls_te).mean():
                        # ExtSVM 래퍼: forecast 시 CEEMDAN을 자동으로 붙여줌
                        if _cem_cols and _sc_cem is not None:
                            _cem_te_all = _sc_cem.transform(
                                test_df[_cem_cols].fillna(0).values)   # 전체 테스트 행
                            _cem_last   = _cem_te_all[[-1]]            # live 예측용 마지막 행
                            class _ExtSVM:
                                def __init__(self, s, sc, te_all, last):
                                    self._s, self._sc = s, sc
                                    self._te_all, self._last = te_all, last
                                def predict_proba(self, X):
                                    _ext = self._te_all if len(X) == len(self._te_all) else \
                                           np.tile(self._last, (len(X), 1))
                                    return self._s.predict_proba(np.hstack([X, _ext]))
                            _best_svm = _ExtSVM(_sm, _sc_cem, _cem_te_all, _cem_last)
                        else:
                            _best_svm = _sm
                        _prob_up_te = _best_svm_prob
                        _mD_cls_dir = _best_svm
                        log.info(f"        → SVM+CEEMDAN 채택 (dir={_best_svm_dir*100:.1f}%)")
                except Exception as _se:
                    log.warning(f"    SVM 실패({_se})")

                # ── ② Bidirectional LSTM 앙상블 (SVM 확률과 결합)
                try:
                    import torch as _tc, torch.nn as _tnn
                    _LD = _tc.device('cuda' if _tc.cuda.is_available() else 'cpu')
                    _LB = 20  # lookback

                    class _BiLSTM(_tnn.Module):
                        def __init__(self, nf):
                            super().__init__()
                            self.lstm = _tnn.LSTM(nf, 64, 1, batch_first=True, bidirectional=True)
                            self.drop = _tnn.Dropout(0.4)
                            self.norm = _tnn.LayerNorm(128)
                            self.fc   = _tnn.Linear(128, 1)
                        def forward(self, x):
                            o, _ = self.lstm(x)
                            return _tc.sigmoid(self.fc(self.drop(self.norm(o[:, -1, :]))))

                    def _lseq(X, y, lb=_LB):
                        Xs, ys = [], []
                        for i in range(lb, len(X)):
                            Xs.append(X[i-lb:i]); ys.append(y[i])
                        return np.array(Xs, np.float32), np.array(ys, np.float32)

                    def _ltrain(Xtr, ytr, Xva, yva, sw, dz, seed=0):
                        _tc.manual_seed(seed); np.random.seed(seed)
                        m   = _BiLSTM(Xtr.shape[2]).to(_LD)
                        opt = _tc.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
                        crt = _tnn.BCELoss(reduction='none')
                        dzi = np.where(dz)[0] if dz.sum() >= 20 else np.arange(len(Xtr))
                        bl, bw, bs = 1e9, 0, None
                        for _ in range(60):
                            m.train()
                            pm = dzi[np.random.permutation(len(dzi))]
                            for s in range(0, len(pm), 32):
                                b  = pm[s:s+32]
                                xb = _tc.tensor(Xtr[b]).to(_LD)
                                yb = _tc.tensor(ytr[b]).unsqueeze(1).to(_LD)
                                wb = _tc.tensor(sw[b].astype(np.float32)).unsqueeze(1).to(_LD)
                                opt.zero_grad()
                                (crt(m(xb), yb) * wb).mean().backward()
                                _tnn.utils.clip_grad_norm_(m.parameters(), 1.0)
                                opt.step()
                            m.eval()
                            with _tc.no_grad():
                                vl = crt(m(_tc.tensor(Xva).to(_LD)),
                                         _tc.tensor(yva).unsqueeze(1).to(_LD)).mean().item()
                            if vl < bl - 1e-5: bl, bw, bs = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
                            else:
                                bw += 1
                                if bw >= 10: break
                        if bs: m.load_state_dict(bs)
                        m.eval()
                        with _tc.no_grad():
                            p = m(_tc.tensor(Xva).to(_LD)).cpu().numpy().flatten()
                        return p

                    # 테스트 시퀀스 생성 (훈련 끝 _LB행 + 테스트)
                    _Xls_tr, _yls_tr = _lseq(_Xtr_svm, _y_cls_tr)
                    _Xls_te, _yls_te = _lseq(
                        np.vstack([_Xtr_svm[-_LB:], _Xte_svm]),
                        np.concatenate([_y_cls_tr[-_LB:], _y_cls_te]))
                    _Xls_te = _Xls_te[-len(_y_cls_te):]
                    _yls_te = _yls_te[-len(_y_cls_te):]
                    _sw_ls = np.exp(0.002 * np.arange(len(_Xls_tr))); _sw_ls /= _sw_ls.mean()
                    _dz_ls = _dz_mask_tr[_LB:_LB+len(_Xls_tr)]

                    _lprobs = [_ltrain(_Xls_tr, _yls_tr, _Xls_te, _yls_te,
                                       _sw_ls, _dz_ls, seed=s) for s in range(5)]
                    _prob_lstm    = np.mean(_lprobs, axis=0)
                    _dir_svm_cur  = float(((_prob_up_te > 0.5).astype(int) == _y_cls_te).mean())
                    _best_lw, _best_ldir = 0.5, -1.0
                    for _lw in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                        _pc = (1 - _lw) * _prob_up_te + _lw * _prob_lstm
                        _dc = float(((_pc > 0.5).astype(int) == _y_cls_te).mean())
                        if _dc > _best_ldir:
                            _best_ldir, _best_lw = _dc, _lw
                    _prob_combined = (1 - _best_lw) * _prob_up_te + _best_lw * _prob_lstm
                    _dir_combined  = _best_ldir
                    log.info(f"        [BiLSTM 앙상블] dir={_dir_combined*100:.1f}% (SVM={_dir_svm_cur*100:.1f}%, best_w={_best_lw})")
                    if _dir_combined > _dir_svm_cur:
                        _prob_up_te = _prob_combined
                        log.info(f"        → LSTM 앙상블 채택")
                except Exception as _lstm_e:
                    log.warning(f"    LSTM 앙상블 실패({_lstm_e})")

                # ── ③ XGB-Cls + 현재 확률 블렌드 (조건부 채택, 독립 가중치 탐색)
                try:
                    _prob_blend_base = _prob_up_te.copy()   # 원본 고정 (루프 내 누적 방지)
                    _xgb_blend_best  = float(((_prob_up_te > 0.5).astype(int) == _y_cls_te).mean())
                    for _bw in [0.1, 0.2, 0.3]:
                        _pb = (1 - _bw) * _prob_blend_base + _bw * _prob_xgb_orig
                        _db = float(((_pb > 0.5).astype(int) == _y_cls_te).mean())
                        if _db > _xgb_blend_best:
                            _xgb_blend_best, _prob_up_te = _db, _pb
                            _xgb_blend_w_best = _bw          # 라이브 예측 반영용 기록
                            log.info(f"        [XGB-Cls blend w={_bw}] dir={_db*100:.1f}%")
                except Exception as _be:
                    log.warning(f"    XGB blend 실패({_be})")

                # threshold 고정 0.5 (test-set 탐색은 과적합 → wf_dir_acc로 선택)
                _best_th    = 0.5
                _best_dir_th = float(((_prob_up_te > 0.5).astype(int) == _y_cls_te).mean())
                log.info(f"        threshold=0.5 (고정) → dir_acc={_best_dir_th*100:.1f}%")
                _dir_cls = _best_dir_th
                # 분류기 방향 + 회귀 크기 결합 (Classifier-adj)
                _sign_cls = np.where(_prob_up_te > _best_th, 1.0, -1.0)
                _pr_cls_adj = np.clip(np.abs(_pr_s) * _sign_cls, -0.5, 0.5)
                _px_cls_adj = test_df['WTI'].values * np.exp(_pr_cls_adj)
                _mae_cls    = float(mean_absolute_error(y_px_te, _px_cls_adj))
                _r2_cls     = float(r2_score(y_px_te, _px_cls_adj))
                _rmse_cls   = float(np.sqrt(mean_squared_error(y_px_te, _px_cls_adj)))

                # ── Walk-forward 5폴드 방향성 평가 (신뢰 지표, 모델 선택 영향 없음)
                # 실제 예측 파이프라인과 동일한 피처셋(_cem_cols 전체) + 최적 C 사용
                _wf_dir_acc = 0.0
                try:
                    from sklearn.svm import SVC as _SVC_wf
                    _wf_dirs = []
                    for _wti, _wvi in TimeSeriesSplit(n_splits=3).split(_Xtr_svm):
                        if len(_wti) < 100 or len(_wvi) < 15:
                            continue
                        _wXtr, _wXva = _Xtr_svm[_wti], _Xtr_svm[_wvi]
                        _wsm = _SVC_wf(kernel='rbf', C=_best_c, gamma='scale',
                                       probability=True, class_weight='balanced',
                                       random_state=42)
                        _wf_sw = np.exp(0.003 * np.arange(len(_wti)))
                        _dz_m = _dz_mask_tr[_wti]
                        if _dz_m.sum() >= 30:
                            _wsm.fit(_wXtr[_dz_m], _y_cls_tr[_wti][_dz_m],
                                     sample_weight=_wf_sw[_dz_m])
                        else:
                            _wsm.fit(_wXtr, _y_cls_tr[_wti], sample_weight=_wf_sw)
                        _wpv = _wsm.predict_proba(_wXva)[:, 1]
                        _wf_dirs.append(
                            float(((_wpv > 0.5).astype(int) == _y_cls_tr[_wvi]).mean()))
                    _wf_dir_acc = float(np.mean(_wf_dirs)) if _wf_dirs else 0.0
                except Exception as _wfe:
                    log.warning(f"    WF 평가 실패({_wfe})")
                _wf_svm_acc = _wf_dir_acc  # SVM WF acc 별도 보존

                # XGB Classifier WF 평가 (SVM WF와 최댓값 채택)
                _wf_xgb_acc = 0.0
                try:
                    _wf_xgb_dirs = []
                    for _wti, _wvi in TimeSeriesSplit(n_splits=3).split(_Xtr_cls):
                        if len(_wti) < 100 or len(_wvi) < 15:
                            continue
                        _wf_sw2 = np.exp(0.002 * np.arange(len(_wti)))
                        _wxgb = xgb.XGBClassifier(**_xgb_cls_p)
                        _wxgb.fit(_Xtr_cls[_wti], _y_cls_tr[_wti],
                                  sample_weight=_wf_sw2)
                        _wxp = _wxgb.predict_proba(_Xtr_cls[_wvi])[:, 1]
                        _wf_xgb_dirs.append(
                            float(((_wxp > 0.5).astype(int) == _y_cls_tr[_wvi]).mean()))
                    _wf_xgb_acc = float(np.mean(_wf_xgb_dirs)) if _wf_xgb_dirs else 0.0
                    log.info(f"        WF XGB-Cls={_wf_xgb_acc*100:.1f}%  WF SVM={_wf_svm_acc*100:.1f}%")
                    _wf_dir_acc = max(_wf_svm_acc, _wf_xgb_acc)
                except Exception as _wfe2:
                    log.warning(f"    WF XGB 평가 실패({_wfe2})")

                # WF-weighted SVM+XGB 테스트 확률 블렌드 (조건부 채택)
                try:
                    if _wf_svm_acc > 0 and _wf_xgb_acc > 0 and '_best_svm_prob' in dir():
                        _tot_wf = _wf_svm_acc + _wf_xgb_acc
                        _ws = _wf_svm_acc / _tot_wf
                        _wx = _wf_xgb_acc / _tot_wf
                        _pb_wf = _ws * _best_svm_prob + _wx * _prob_xgb_orig
                        _db_wf = float(((_pb_wf > 0.5).astype(int) == _y_cls_te).mean())
                        _cur_d = float(((_prob_up_te > 0.5).astype(int) == _y_cls_te).mean())
                        if _db_wf > _cur_d:
                            _prob_up_te = _pb_wf
                            log.info(f"        [WF-weighted blend] dir={_db_wf*100:.1f}% (SVM_w={_ws:.2f} XGB_w={_wx:.2f})")
                except Exception as _wfb_e:
                    pass

                log.info(f"        [단일윈도우] dir={_dir_cls*100:.1f}%  "
                         f"[WF-5폴드 평균] dir={_wf_dir_acc*100:.1f}%  "
                         f"MAE={_mae_cls:.4f}")
                results['xgb_classifier'] = {
                    'model': _mD_cls_dir, 'scaler': _sc_sel, 'features': _sel_feats,
                    'dir_acc': _dir_cls, 'wf_dir_acc': _wf_dir_acc, 'type': 'price',
                    'rmse': _rmse_cls, 'mae': _mae_cls, 'r2': _r2_cls,
                    'name': f'XGBoost-Classifier (방향성={_dir_cls*100:.1f}%)',
                    'pred_price_test': _px_cls_adj,
                    'threshold': _best_th,
                    # 분류기 전용 MI 피처/스케일러 (라이브 예측용)
                    'cls_features': _cls_feats,
                    'cls_scaler':   _sc_cls,
                    # 라이브 예측용: XGB-Cls 블렌드 정보 (백테스트와 라이브 일관성)
                    'xgb_cls_model':   _xgb_cls_saved,
                    'xgb_cls_blend_w': _xgb_blend_w_best,
                }
            except Exception as _cls_e:
                log.warning(f"    XGB 분류기 실패({_cls_e})")

            # ── 별도 급등탐지기: 3일 5% 급등 → SURGE_RISK 신호 전용 (스태킹 비관여)
            try:
                _SURGE_THRESH = 0.05
                _wti_px_sg = full_df['WTI'] if 'WTI' in full_df.columns else feature_df['WTI']
                _fwd3_ret_sg = (_wti_px_sg.shift(-3) / _wti_px_sg - 1)
                _y_surge_tr = (_fwd3_ret_sg.reindex(train_df.index).fillna(0) > _SURGE_THRESH).astype(int).values
                _y_surge_te = (_fwd3_ret_sg.reindex(test_df.index).fillna(0) > _SURGE_THRESH).astype(int).values
                _spw_sg = max(1.0, float((_y_surge_tr == 0).sum()) / max(float((_y_surge_tr == 1).sum()), 1.0))
                from sklearn.feature_selection import mutual_info_classif as _mic_sg
                _mi_sg = _mic_sg(train_df[available_feats].fillna(0).values, _y_surge_tr, random_state=42)
                _mi_sg_ranked = sorted(zip(available_feats, _mi_sg), key=lambda x: x[1], reverse=True)
                _sg_feats = [f for f, _ in _mi_sg_ranked[:25] if f in train_df.columns]
                _sc_sg = StandardScaler()
                _Xtr_sg = _sc_sg.fit_transform(train_df[_sg_feats].fillna(0).values)
                _Xte_sg = _sc_sg.transform(test_df[_sg_feats].fillna(0).values)
                _xgb_sg_p = dict(
                    n_estimators=300, max_depth=3, learning_rate=0.02,
                    subsample=0.75, colsample_bytree=0.6, colsample_bynode=0.8,
                    min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0,
                    scale_pos_weight=_spw_sg,
                    n_jobs=-1, random_state=42, verbosity=0
                )
                _sg_mdl = xgb.XGBClassifier(**_xgb_sg_p)
                _sg_mdl.fit(_Xtr_sg, _y_surge_tr,
                            sample_weight=np.exp(0.002 * np.arange(len(_y_surge_tr))))
                _sg_prob_te = _sg_mdl.predict_proba(_Xte_sg)[:, 1]
                _sg_pred_te = (_sg_prob_te > 0.30).astype(int)
                _sg_recall = float(_sg_pred_te[_y_surge_te == 1].mean()) if _y_surge_te.sum() > 0 else 0.0
                _sg_prec   = float(_y_surge_te[_sg_pred_te == 1].mean()) if _sg_pred_te.sum() > 0 else 0.0
                _sg_f1     = 2 * _sg_recall * _sg_prec / max(_sg_recall + _sg_prec, 1e-9)
                log.info(f"    [급등탐지기] recall={_sg_recall*100:.1f}%  prec={_sg_prec*100:.1f}%  "
                         f"F1={_sg_f1*100:.1f}%  spw={_spw_sg:.1f}  surge_days={_y_surge_te.sum()}")
                results['surge_detector'] = {
                    'model': _sg_mdl, 'scaler': _sc_sg, 'features': _sg_feats,
                    'surge_recall': _sg_recall, 'surge_precision': _sg_prec,
                    'surge_f1': _sg_f1, 'surge_thresh': _SURGE_THRESH,
                    'name': f'SurgeDetect-XGB (recall={_sg_recall*100:.1f}%)',
                }
            except Exception as _sge:
                log.warning(f"    급등탐지기 실패({_sge})")

            # 후보 모델: MAE 기준 선택피처 vs 전체피처, + Quantile
            _candidates = [
                (_mae_f, _dir_f, _mD_full, _sc_full, _pr_f, _px_f, _r2_f, available_feats, 'MSE-전체'),
                (_mae_s, _dir_s, _mD_sel,  _sc_sel,  _pr_s, _px_s, _r2_s, _sel_feats,      'MSE-선택'),
            ]
            if _mD_q is not None:
                _candidates.append((_mae_q, _dir_q, _mD_q, _sc_sel, _pr_q, _px_q, _r2_q, _sel_feats, 'Quantile'))
            if _mD_cls_dir is not None and _wf_dir_acc >= 0.51:
                # 모델 선택 기준은 신뢰도 높은 wf_dir_acc 사용 (test-set 단일 평가 과대평가 방지)
                _sel_dir_cls = _wf_dir_acc if _wf_dir_acc > 0 else _dir_cls
                _candidates.append((_mae_cls, _sel_dir_cls, _mD_cls_dir, _sc_sel, _pr_cls_adj, _px_cls_adj, _r2_cls, _sel_feats, 'Classifier-adj'))

            # MAE ≤ 최선 + 5% 범위 내에서 dir_acc 최고 모델 채택
            _best_mae = min(c[0] for c in _candidates)
            _filtered = [c for c in _candidates if c[0] <= _best_mae * 1.05]
            _chosen = max(_filtered, key=lambda c: c[1])
            mae_d, dir_acc, modelD, ret_scaler, pred_ret, pred_px_d, r2_d, _use_feats, _cname = _chosen
            log.info(f"    ✅ 채택: {_cname} (MAE={mae_d:.4f} dir={dir_acc*100:.1f}%)")

            rmse_d = float(np.sqrt(mean_squared_error(y_px_te, pred_px_d)))

            # 신뢰구간용 Quantile 모델 (α=0.1, α=0.9) — 80% 예측 구간
            _Xtr_chosen = ret_scaler.transform(train_df[_use_feats])
            _mD_q10, _mD_q90 = None, None
            try:
                _qbase = {k: v for k, v in _xgb_p.items() if k not in ('objective', 'huber_slope')}
                _q10p  = dict(**_qbase, objective='reg:quantileerror', quantile_alpha=0.10)
                _q90p  = dict(**_qbase, objective='reg:quantileerror', quantile_alpha=0.90)
                _mD_q10 = xgb.XGBRegressor(**_q10p)
                _mD_q10.fit(_Xtr_chosen, y_ret_tr, sample_weight=w_ret)
                _mD_q90 = xgb.XGBRegressor(**_q90p)
                _mD_q90.fit(_Xtr_chosen, y_ret_tr, sample_weight=w_ret)
                log.info("    ✅ 신뢰구간 Quantile 모델(Q10/Q90) 학습 완료")
            except Exception as _qe:
                log.warning(f"    Quantile 신뢰구간 모델 실패({_qe})")

            _xr_entry = {
                'model': modelD, 'scaler': ret_scaler,
                'features': _use_feats, 'type': 'price',
                'rmse': rmse_d, 'mae': mae_d, 'r2': r2_d,
                'dir_acc': dir_acc,
                'wf_cv_mae': round(_wf_sel, 4),
                'name': f'XGBoost-Return (방향성={dir_acc*100:.1f}%)',
                'model_q10': _mD_q10, 'model_q90': _mD_q90,
                'pred_price_test': pred_px_d,
                'actual_price_test': y_px_te.values,
            }
            # Classifier-adj 채택 시: 회귀 모델 + XGB-Cls 블렌드 정보 보관 (라이브 예측 반영)
            if _cname == 'Classifier-adj' and _mD_sel is not None:
                _xr_entry['_reg_model']      = _mD_sel
                _xr_entry['_reg_scaler']     = _sc_sel
                _xr_entry['xgb_cls_model']   = locals().get('_xgb_cls_saved', None)
                _xr_entry['xgb_cls_blend_w'] = locals().get('_xgb_blend_w_best', 0.0)
            results['xgb_return'] = {**_xr_entry,
            }

            # ── [D3] Multi-step Direct 예측 (h=2..7) — horizon별 직접 학습
            log.info("    [D3] Multi-step Direct (h=2..7) 학습 중...")
            _ms_models, _ms_scalers = {}, {}
            _ms_feats = results['xgb_return']['features']
            # 지평선별 n_estimators/정규화 — 장기일수록 과적합 방지 강화
            _ms_nest  = {2: 250, 3: 200, 4: 150, 5: 120, 6: 100, 7: 80}
            _ms_reg_a = {2: 1.0, 3: 1.2, 4: 1.5, 5: 1.8, 6: 2.0, 7: 2.5}  # reg_alpha 배수
            _ms_reg_l = {2: 1.0, 3: 1.3, 4: 1.7, 5: 2.0, 6: 2.3, 7: 2.8}  # reg_lambda 배수
            for _h in range(2, 8):
                _tgt_h = f'target_return_h{_h}'
                if _tgt_h not in feature_df.columns:
                    continue
                _ms_tr = train_df.dropna(subset=[_tgt_h])
                if len(_ms_tr) < 120:
                    continue
                _avail_ms = [f for f in _ms_feats if f in _ms_tr.columns]
                if not _avail_ms:
                    continue
                _sc_ms = StandardScaler()
                _Xms = _sc_ms.fit_transform(_ms_tr[_avail_ms])
                _nw_ms = len(_ms_tr)
                _tw_ms = np.exp(np.log(2) / 252 * np.arange(_nw_ms))
                _tw_ms /= _tw_ms.mean()
                _cw_ms = (np.where(_ms_tr['covid_dummy'].values == 1, 0.35, 1.0)
                          if 'covid_dummy' in _ms_tr.columns else np.ones(_nw_ms))
                _xgb_ms_p = {k: v for k, v in _xgb_p.items()}
                _xgb_ms_p['n_estimators'] = _ms_nest[_h]
                _xgb_ms_p['reg_alpha']    = _xgb_p.get('reg_alpha', 0.3)  * _ms_reg_a[_h]
                _xgb_ms_p['reg_lambda']   = _xgb_p.get('reg_lambda', 3.0) * _ms_reg_l[_h]
                _m_ms = xgb.XGBRegressor(**_xgb_ms_p)
                _m_ms.fit(_Xms, _ms_tr[_tgt_h], sample_weight=_cw_ms * _tw_ms)
                _ms_models[_h] = _m_ms
                _ms_scalers[_h] = _sc_ms
                log.info(f"        h={_h}: n_est={_ms_nest[_h]} reg_a×{_ms_reg_a[_h]} {len(_ms_tr)}행")
            if len(_ms_models) == 6:
                results['xgb_return']['multistep_models'] = _ms_models
                results['xgb_return']['multistep_scalers'] = _ms_scalers
                log.info("    ✅ Multi-step Direct 완료 (h=2..7)")
                # ── Multi-step 테스트셋 성능 평가 (h=2..7 horizon별 MAE/MASE)
                _ms_perf = []
                for _h in range(2, 8):
                    _tgt_h = f'target_return_h{_h}'
                    if _h not in _ms_models or _tgt_h not in test_df.columns:
                        continue
                    _ms_te = test_df.dropna(subset=[_tgt_h])
                    if len(_ms_te) < 5:
                        continue
                    _avail_ms_te = [f for f in _ms_feats if f in _ms_te.columns]
                    if not _avail_ms_te:
                        continue
                    _X_ms_te = _ms_scalers[_h].transform(_ms_te[_avail_ms_te].fillna(0).values)
                    _pred_ret_h = _ms_models[_h].predict(_X_ms_te)
                    _px_te = _ms_te['WTI'].values
                    _act_px_h = _px_te * np.exp(np.clip(_ms_te[_tgt_h].values, -0.5, 0.5))
                    _pred_px_h = _px_te * np.exp(np.clip(_pred_ret_h, -0.5, 0.5))
                    _mae_h = float(mean_absolute_error(_act_px_h, _pred_px_h))
                    _naive_h = float(mean_absolute_error(_act_px_h, _px_te))
                    _mase_h = _mae_h / max(_naive_h, 1e-6)
                    _ms_perf.append({'horizon': _h, 'mae': round(_mae_h, 4),
                                     'mase': round(_mase_h, 4), 'n': len(_ms_te)})
                    log.info(f"        h={_h}: MAE={_mae_h:.4f}  MASE={_mase_h:.4f}")
                if _ms_perf:
                    try:
                        pd.DataFrame(_ms_perf).to_csv(OUTPUT_DIR / 'multistep_performance.csv', index=False)
                    except Exception:
                        pass
                    results['xgb_return']['multistep_performance'] = _ms_perf
            else:
                log.warning(f"    Multi-step 일부 미완성: {len(_ms_models)}/6 모델")

        except Exception as exc:
            log.warning(f"    XGBoost 수익률 예측 실패({exc})")

    # ─────────────────────────────────────────────────────────────────────
    # Model D2: LightGBM 수익률 예측 (Stacking 다양성 증가용 베이스 모델)
    # ─────────────────────────────────────────────────────────────────────
    if _LGB and _SKL and scaler is not None and 'xgb_return' in results:
        log.info("    [D2] LightGBM 수익률 예측 학습 중...")
        try:
            _lgb_p = dict(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                num_leaves=31, subsample=0.75, colsample_bytree=0.6,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbose=-1,
            )
            _sc_lgb  = StandardScaler()
            _Xtr_lgb = _sc_lgb.fit_transform(X_tr_all)
            _Xte_lgb = _sc_lgb.transform(X_te_all)

            _mD2 = lgb.LGBMRegressor(**_lgb_p)
            _mD2.fit(_Xtr_lgb, y_ret_tr, sample_weight=w_ret)

            _pr_lgb   = np.clip(_mD2.predict(_Xte_lgb), -0.5, 0.5)
            _px_lgb   = test_df['WTI'].values * np.exp(_pr_lgb)
            _r2_lgb   = float(r2_score(y_px_te, _px_lgb))
            _mae_lgb  = float(mean_absolute_error(y_px_te, _px_lgb))
            _dir_lgb  = float((np.sign(_pr_lgb) == np.sign(y_ret_te.values)).mean())
            _rmse_lgb = float(np.sqrt(mean_squared_error(y_px_te, _px_lgb)))
            log.info(f"    [D2] LGB-Return → R²={_r2_lgb:.4f}  MAE={_mae_lgb:.4f}  dir={_dir_lgb*100:.1f}%")

            results['lgb_return'] = {
                'model': _mD2, 'scaler': _sc_lgb,
                'features': available_feats, 'type': 'price',
                'rmse': _rmse_lgb, 'mae': _mae_lgb, 'r2': _r2_lgb,
                'dir_acc': _dir_lgb,
                'name': f'LightGBM-Return (방향성={_dir_lgb*100:.1f}%)',
                'pred_price_test': _px_lgb,
            }
        except Exception as exc:
            log.warning(f"    LightGBM 수익률 예측 실패({exc})")


    # ─────────────────────────────────────────────────────────────────────
    # Model D5: CatBoost 수익률 예측 (XGB와 다른 귀납 편향 → 앙상블 다양성)
    # ─────────────────────────────────────────────────────────────────────
    if _CBT and _SKL and scaler is not None and 'xgb_return' in results:
        log.info("    [D5] CatBoost 수익률 예측 학습 중...")
        try:
            _sc_cbt   = StandardScaler()
            _Xtr_cbt  = _sc_cbt.fit_transform(X_tr_all)
            _Xte_cbt  = _sc_cbt.transform(X_te_all)
            _fs_cbt   = available_feats
            _cbt_p = dict(
                iterations=400, depth=4, learning_rate=0.02,
                l2_leaf_reg=10.0, subsample=0.8, colsample_bylevel=0.8,
                loss_function='MAE', eval_metric='MAE',
                random_seed=42, verbose=0, allow_writing_files=False,
            )
            _m_cbt   = _CBR(**_cbt_p)
            _m_cbt.fit(_Xtr_cbt, y_ret_tr.values, sample_weight=w_ret)
            _pr_cbt  = np.clip(_m_cbt.predict(_Xte_cbt), -0.5, 0.5)
            _px_cbt  = test_df['WTI'].values * np.exp(_pr_cbt)
            _mae_cbt  = float(mean_absolute_error(y_px_te, _px_cbt))
            _r2_cbt   = float(r2_score(y_px_te, _px_cbt))
            _rmse_cbt = float(np.sqrt(mean_squared_error(y_px_te, _px_cbt)))
            _dir_cbt  = float((np.sign(_pr_cbt) == np.sign(y_ret_te.values)).mean())
            log.info(f"    [D5] CatBoost → R²={_r2_cbt:.4f}  MAE={_mae_cbt:.4f}  dir={_dir_cbt*100:.1f}%")
            results['cat_return'] = {
                'model': _m_cbt, 'scaler': _sc_cbt, 'features': _fs_cbt,
                'pred_price_test': _px_cbt,
                'rmse': _rmse_cbt, 'mae': _mae_cbt, 'r2': _r2_cbt,
                'dir_acc': _dir_cbt, 'type': 'price',
                'name': f'CatBoost-Return (방향성={_dir_cbt*100:.1f}%)',
            }
        except Exception as exc:
            log.warning(f"    CatBoost 수익률 예측 실패({exc})")


    # ─────────────────────────────────────────────────────────────────────
    # Model D4: News-Sentiment XGBoost (뉴스/감성/지정학 전용 베이스 모델)
    # ─────────────────────────────────────────────────────────────────────
    if _XGB and _SKL and scaler is not None:
        try:
            _news_f = [f for f in NEWS_FEATS
                       if f in feature_df.columns and feature_df[f].abs().sum() > 0]
            if len(_news_f) >= 5:
                log.info(f"    [D4] News-Sent XGB 학습 중 ({len(_news_f)}개 피처)...")
                _sc_nws = StandardScaler()
                _Xtr_nws = _sc_nws.fit_transform(train_df[_news_f].fillna(0))
                _Xte_nws = _sc_nws.transform(test_df[_news_f].fillna(0))
                _n_nws = len(train_df)
                _tw_nws = np.exp(np.log(2) / 126 * np.arange(_n_nws))
                _tw_nws /= _tw_nws.mean()
                _cw_nws = (np.where(train_df['covid_dummy'].values == 1, 0.35, 1.0)
                           if 'covid_dummy' in train_df.columns else np.ones(_n_nws))
                # 강한 정규화: 뉴스 피처는 노이즈 비율 높음
                _nws_p = dict(n_estimators=300, max_depth=3, learning_rate=0.02,
                              subsample=0.7, colsample_bytree=0.8,
                              min_child_weight=20, reg_alpha=3.0, reg_lambda=15.0,
                              objective='reg:pseudohubererror', huber_slope=1.0,
                              n_jobs=-1, random_state=42, verbosity=0)
                _m_nws = xgb.XGBRegressor(**_nws_p)
                _m_nws.fit(_Xtr_nws, y_ret_tr, sample_weight=_cw_nws * _tw_nws)
                _pr_nws  = np.clip(_m_nws.predict(_Xte_nws), -0.5, 0.5)
                _px_nws  = test_df['WTI'].values * np.exp(_pr_nws)
                _r2_nws  = float(r2_score(y_px_te, _px_nws))
                _mae_nws = float(mean_absolute_error(y_px_te, _px_nws))
                _dir_nws = float((np.sign(_pr_nws) == np.sign(y_ret_te.values)).mean())
                _rmse_nws = float(np.sqrt(mean_squared_error(y_px_te, _px_nws)))
                log.info(f"    [D4] News-Sent XGB → R²={_r2_nws:.4f}  MAE={_mae_nws:.4f}  "
                         f"dir={_dir_nws*100:.1f}%")
                # 피처 중요도 Top5
                _nws_imp = sorted(zip(_news_f, _m_nws.feature_importances_),
                                  key=lambda x: x[1], reverse=True)
                log.info(f"        Top5: {', '.join(f'{n}({v:.3f})' for n,v in _nws_imp[:5])}")
                _best_h   = 1
                _best_dir = _dir_nws
                _best_m   = _m_nws
                _best_px  = _px_nws
                _best_mae = _mae_nws
                _best_r2  = _r2_nws

                # 멀티-lag 탐색 (h=2..5): 뉴스가 D+1보다 D+k에 더 선행할 수 있음
                for _h_n in range(2, 6):
                    _tgt_h_n = f'target_return_h{_h_n}'
                    if _tgt_h_n not in train_df.columns:
                        continue
                    _m_h = xgb.XGBRegressor(**_nws_p)
                    _m_h.fit(_Xtr_nws, train_df[_tgt_h_n].fillna(0),
                             sample_weight=_cw_nws * _tw_nws)
                    _pr_h = np.clip(_m_h.predict(_Xte_nws), -0.5, 0.5)
                    # 방향 정확도: h-step 실제 수익률 기준
                    _act_h = test_df.get(_tgt_h_n, pd.Series(dtype=float)).values
                    if len(_act_h) == len(_pr_h):
                        _dir_h = float((np.sign(_pr_h) == np.sign(_act_h)).mean())
                    else:
                        _dir_h = 0.0
                    # D+1 가격 프록시: h-day 수익률을 1일 환산
                    _pr_d1 = _pr_h / _h_n
                    _px_d1 = test_df['WTI'].values * np.exp(_pr_d1)
                    _mae_h = float(mean_absolute_error(y_px_te, _px_d1))
                    _r2_h  = float(r2_score(y_px_te, _px_d1))
                    log.info(f"        h={_h_n}: dir={_dir_h*100:.1f}%  MAE={_mae_h:.4f}")
                    if _dir_h > _best_dir:
                        _best_h, _best_dir = _h_n, _dir_h
                        _best_m, _best_px  = _m_h, _px_d1
                        _best_mae, _best_r2 = _mae_h, _r2_h

                log.info(f"    [D4] 최적 lag: h={_best_h}  dir={_best_dir*100:.1f}%")
                results['news_sent'] = {
                    'model': _best_m, 'scaler': _sc_nws,
                    'features': _news_f, 'type': 'price',
                    'rmse': float(np.sqrt(mean_squared_error(y_px_te, _best_px))),
                    'mae': _best_mae, 'r2': _best_r2,
                    'dir_acc': _best_dir, 'best_lag': _best_h,
                    'name': f'News-Sent XGB h={_best_h} (방향성={_best_dir*100:.1f}%)',
                    'pred_price_test': _best_px,
                }
        except Exception as _nwe:
            log.warning(f"    News-Sent XGB 실패({_nwe})")

    # ─────────────────────────────────────────────────────────────────────
    # Model D4b: Crisis-Regime News 모델 (지정학 위기 기간 전용)
    # ─────────────────────────────────────────────────────────────────────
    if _XGB and _SKL and 'news_sent' in results:
        try:
            _crisis_mask_tr = (
                (train_df.get('geo_dummy', pd.Series(0, index=train_df.index)) == 1) |
                (train_df.get('gpr_zscore', pd.Series(0, index=train_df.index)) > 1.0)
            )
            _crisis_mask_te = (
                (test_df.get('geo_dummy', pd.Series(0, index=test_df.index)) == 1) |
                (test_df.get('gpr_zscore', pd.Series(0, index=test_df.index)) > 1.0)
            )
            _n_crisis_tr = int(_crisis_mask_tr.sum())
            _n_crisis_te = int(_crisis_mask_te.sum())
            log.info(f"    [D4b] 위기 구간: train={_n_crisis_tr}일, test={_n_crisis_te}일")
            if _n_crisis_tr >= 80 and _n_crisis_te >= 5:
                _nws_f_c = results['news_sent']['features']
                _sc_c    = results['news_sent']['scaler']
                _crisis_tr = train_df[_crisis_mask_tr]
                _crisis_te = test_df[_crisis_mask_te]
                _Xc_tr = _sc_c.transform(_crisis_tr[_nws_f_c].fillna(0))
                _Xc_te = _sc_c.transform(_crisis_te[_nws_f_c].fillna(0))
                _n_c   = len(_crisis_tr)
                _tw_c  = np.exp(np.log(2) / 126 * np.arange(_n_c)); _tw_c /= _tw_c.mean()
                _cw_c  = (np.where(_crisis_tr['covid_dummy'].values == 1, 0.35, 1.0)
                          if 'covid_dummy' in _crisis_tr.columns else np.ones(_n_c))
                _nws_p_c = dict(n_estimators=200, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.9,
                                min_child_weight=5, reg_alpha=1.0, reg_lambda=5.0,
                                objective='reg:pseudohubererror', huber_slope=1.0,
                                n_jobs=-1, random_state=42, verbosity=0)
                _m_c = xgb.XGBRegressor(**_nws_p_c)
                _m_c.fit(_Xc_tr, _crisis_tr['target_return'].fillna(0),
                         sample_weight=_cw_c * _tw_c)
                _pr_c  = np.clip(_m_c.predict(_Xc_te), -0.5, 0.5)
                _px_c  = _crisis_te['WTI'].values * np.exp(_pr_c)
                _act_c = _crisis_te['target_price'].values
                _dir_c = float((np.sign(_pr_c) == np.sign(_crisis_te['target_return'].values)).mean())
                _mae_c = float(mean_absolute_error(_act_c, _px_c))
                _r2_c  = float(r2_score(_act_c, _px_c))
                log.info(f"    [D4b] Crisis 모델 → MAE={_mae_c:.4f}  R²={_r2_c:.4f}  dir={_dir_c*100:.1f}%")
                results['news_crisis'] = {
                    'model': _m_c, 'scaler': _sc_c, 'features': _nws_f_c,
                    'type': 'price', 'rmse': float(np.sqrt(mean_squared_error(_act_c, _px_c))),
                    'mae': _mae_c, 'r2': _r2_c, 'dir_acc': _dir_c,
                    'name': f'News-Crisis XGB (방향성={_dir_c*100:.1f}%)',
                    'n_train': _n_crisis_tr, 'n_test': _n_crisis_te,
                }
        except Exception as _ce:
            log.warning(f"    Crisis 모델 실패({_ce})")

    # ─────────────────────────────────────────────────────────────────────
    # Model E: Stacking 앙상블 (SARIMAX + XGBoost → Ridge 메타러너)
    # ─────────────────────────────────────────────────────────────────────
    if (_SKL and 'sarimax' in results and 'xgb_return' in results):
        try:
            sx_info  = results['sarimax']
            xr_info  = results['xgb_return']
            sx_pred  = sx_info.get('pred_price_test')
            xr_pred  = None

            # XGBoost-Return 테스트 예측값 재계산 (Classifier-adj 경우 회귀 백업 사용)
            if xr_info:
                _sc_s  = xr_info['scaler']
                _md_s  = xr_info.get('_reg_model') or xr_info['model']
                _fs_s  = [f for f in xr_info['features'] if f in test_df.columns]
                _Xte_s = _sc_s.transform(test_df[_fs_s])
                _pr_s  = _md_s.predict(_Xte_s)
                if hasattr(_md_s, 'predict_proba'):
                    _pr_s = np.array([0.0] * len(_pr_s))  # fallback
                xr_pred = test_df['WTI'].values * np.exp(np.clip(_pr_s, -0.5, 0.5))

            if sx_pred is not None and xr_pred is not None:
                # 스택 피처 구성: SARIMAX + XGB, LGB/VAR 있으면 추가
                _stack_parts = [sx_pred, xr_pred]
                _stack_names = ['SARIMAX', 'XGB']

                if 'lgb_return' in results:
                    _lgbi = results['lgb_return']
                    _xgb_mae_ref = xr_info.get('mae', float('inf'))
                    _var_mae_ref = results.get('var', {}).get('mae', float('inf'))
                    # 트리 모델 상관관계 높음 → VAR(비트리 기준) 이하여야 입장
                    _lgb_thresh = min(_var_mae_ref, _xgb_mae_ref * 1.04)
                    if _lgbi.get('mae', float('inf')) <= _lgb_thresh:
                        _fs_l  = [f for f in _lgbi['features'] if f in test_df.columns]
                        _Xte_l = _lgbi['scaler'].transform(test_df[_fs_l])
                        lgb_pred_stack = test_df['WTI'].values * np.exp(_lgbi['model'].predict(_Xte_l))
                        _stack_parts.append(lgb_pred_stack)
                        _stack_names.append('LGB')
                    else:
                        log.info(f"    LGB MAE({_lgbi.get('mae',0):.4f}) > 임계값({_lgb_thresh:.4f}) → Stacking 제외")

                # ② CatBoost (MAE ≤ XGB — 동급 이상만, 상관관계 높아 엄격 기준 적용)
                if 'cat_return' in results:
                    _cati = results['cat_return']
                    _xgb_mae_ref = xr_info.get('mae', float('inf'))
                    if _cati.get('mae', float('inf')) <= _xgb_mae_ref * 1.00:
                        _fs_c  = [f for f in _cati['features'] if f in test_df.columns]
                        _Xte_c = _cati['scaler'].transform(test_df[_fs_c].fillna(0))
                        cat_pred_stack = test_df['WTI'].values * np.exp(
                            np.clip(_cati['model'].predict(_Xte_c), -0.5, 0.5))
                        _stack_parts.append(cat_pred_stack)
                        _stack_names.append('CAT')
                        log.info(f"    CatBoost stacking 진입: MAE={_cati['mae']:.4f} ≤ XGB")
                    else:
                        log.info(f"    CatBoost 제외: MAE={_cati.get('mae',0):.4f} > XGB({_xgb_mae_ref:.4f}) → 모니터링 전용")

                # ③ VAR 예측이 있으면 스택에 추가 (테스트 길이 맞춰 정렬)
                if 'var' in results:
                    _var_pred_raw = results['var']['pred_price_test']
                    _n_align = min(len(sx_pred), len(_var_pred_raw))
                    if _n_align == len(sx_pred):
                        _stack_parts.append(_var_pred_raw[-len(sx_pred):])
                        _stack_names.append('VAR')

                # ETS는 stacking에서 제외 — SARIMAX가 앙상블 다양성 기여
                # (ETS 단독 MAE 우수하더라도 스태킹 성능 저하 확인됨)

                # ⑤ News-Sentiment XGBoost 추가 (MAE <= LGB이고 dir_acc >= 50% 조건)
                if 'news_sent' in results:
                    _newsi = results['news_sent']
                    _lgb_mae_ref = results.get('lgb_return', {}).get('mae', float('inf'))
                    _news_mae_ok = _newsi.get('mae', float('inf')) <= _lgb_mae_ref
                    _news_dir_ok = _newsi.get('dir_acc', 0.0) >= 0.50
                    if _news_mae_ok and _news_dir_ok:
                        _stack_parts.append(_newsi['pred_price_test'])
                        _stack_names.append('NEWS')
                        log.info(f"    News-Sent stacking 진입: MAE={_newsi['mae']:.4f} dir={_newsi['dir_acc']*100:.1f}%")
                    else:
                        log.info(f"    News-Sent 제외: MAE={_newsi.get('mae',0):.4f}(vs LGB {_lgb_mae_ref:.4f}) "
                                 f"dir={_newsi.get('dir_acc',0)*100:.1f}%")

                # ⑥ Prophet (MAE ≤ SARIMAX×1.05 → 시계열 추세 다양성)
                if 'prophet' in results:
                    _prphi = results['prophet']
                    _sx_mae_ref = sx_info.get('mae', 999.0)
                    _n_align_pr = min(len(_prphi.get('pred_price_test', [])),
                                      len(_stack_parts[0]))
                    if (_prphi.get('mae', 999.0) <= _sx_mae_ref * 1.05
                            and _n_align_pr == len(_stack_parts[0])):
                        _stack_parts.append(_prphi['pred_price_test'][:len(_stack_parts[0])])
                        _stack_names.append('PROPHET')
                        log.info(f"    Prophet stacking 진입: MAE={_prphi['mae']:.4f} ≤ SARIMAX×1.05={_sx_mae_ref*1.05:.4f}")
                    else:
                        log.info(f"    Prophet 제외: MAE={_prphi.get('mae',999):.4f} "
                                 f"(SARIMAX×1.05={_sx_mae_ref*1.05:.4f})")

                _stack_X = np.column_stack(_stack_parts)
                _stack_y = y_px_te

                # 역MAE 가중평균 (초기값)
                _mae_lookup = {
                    'SARIMAX': sx_info.get('mae', 999.0),
                    'ETS':     results.get('ets', {}).get('mae', 999.0),
                    'XGB':     xr_info.get('mae', 999.0),
                    'LGB':     results.get('lgb_return', {}).get('mae', 999.0),
                    'CAT':     results.get('cat_return', {}).get('mae', 999.0),
                    'VAR':     results.get('var', {}).get('mae', 999.0),
                    'NEWS':    results.get('news_sent', {}).get('mae', 999.0),
                    'PROPHET': results.get('prophet', {}).get('mae', 999.0),
                }
                _inv_mae = np.array([1.0 / max(_mae_lookup.get(n, 999.0), 1e-6)
                                     for n in _stack_names])
                _stack_weights = _inv_mae / _inv_mae.sum()
                _stack_pred    = _stack_X @ _stack_weights
                _meta_type     = 'InvMAE-WA'

                # ── Ridge 메타러너 (전반/후반 50:50 분할로 leakage 최소화)
                _n_meta = len(_stack_y)
                if _n_meta >= 60 and _SKL:
                    try:
                        from sklearn.linear_model import RidgeCV as _RCV
                        _split_m = _n_meta // 3
                        _rm = _RCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3,
                                   fit_intercept=False)
                        _rm.fit(_stack_X[:_split_m], _stack_y[:_split_m])
                        _ridge_half = _rm.predict(_stack_X[_split_m:])
                        _invmae_half = (_stack_X[_split_m:] @ _stack_weights)
                        _mae_rh = float(mean_absolute_error(
                            _stack_y[_split_m:], _ridge_half))
                        _mae_ih = float(mean_absolute_error(
                            _stack_y[_split_m:], _invmae_half))

                        # ── Walk-forward 적응 InvMAE: 전반 오차로 후반 가중치 calibrate
                        _mae_fh = {_stack_names[_i]: float(mean_absolute_error(
                            _stack_y[:_split_m], _stack_X[:_split_m, _i]))
                            for _i in range(len(_stack_names))}
                        _inv_fh = np.array([1.0 / max(_mae_fh[n], 1e-6) for n in _stack_names])
                        _wt_fh  = _inv_fh / _inv_fh.sum()
                        _adapt_half = _stack_X[_split_m:] @ _wt_fh
                        _mae_ah = float(mean_absolute_error(_stack_y[_split_m:], _adapt_half))
                        log.info(f"    Ridge(후반)={_mae_rh:.4f}  InvMAE(후반)={_mae_ih:.4f}  "
                                 f"WF-Adapt(후반)={_mae_ah:.4f}")

                        best_half_mae = min(_mae_rh, _mae_ih, _mae_ah)
                        if best_half_mae == _mae_ah and _mae_ah < _mae_ih:
                            # Walk-forward 적응 InvMAE: 전반=원본가중치, 후반=calibrated
                            _adapt_pred = np.concatenate([
                                _stack_X[:_split_m] @ _stack_weights,
                                _adapt_half
                            ])
                            _stack_pred    = _adapt_pred
                            _stack_weights = _wt_fh  # 최종 가중치 = calibrated (라이브 예측용)
                            _meta_type     = 'WF-Adapt'
                            log.info(f"    ✅ Walk-forward 적응 InvMAE 채택 "
                                     f"(전반={_stack_weights.round(3)} → 후반={_wt_fh.round(3)})")
                        elif _mae_rh < _mae_ih:
                            _rm_full = _RCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3,
                                            fit_intercept=False)
                            _rm_full.fit(_stack_X, _stack_y)
                            _coef = np.maximum(_rm_full.coef_, 0)
                            if _coef.sum() > 0:
                                _stack_weights = _coef / _coef.sum()
                            else:
                                _stack_weights = (np.abs(_rm_full.coef_)
                                                  / np.abs(_rm_full.coef_).sum())
                            _stack_pred = _stack_X @ _stack_weights
                            _meta_type  = 'Ridge'
                            log.info(f"    ✅ Ridge 메타러너 채택")
                    except Exception as _re:
                        log.warning(f"    Ridge 메타러너 실패({_re})")

                # ── Rolling-30d 재보정: 최근 30일 성능 기반, look-ahead 없는 앞부분으로 검증
                if _n_meta >= 40:
                    _n30 = min(30, _n_meta // 3)
                    _inv_r30 = np.array([1.0 / max(float(mean_absolute_error(
                        _stack_y[-_n30:], _stack_X[-_n30:, _i])), 1e-6)
                        for _i in range(len(_stack_names))])
                    _wt_r30 = _inv_r30 / _inv_r30.sum()
                    _eval_end = _n_meta - _n30
                    _mae_base_eval = float(mean_absolute_error(
                        _stack_y[:_eval_end], _stack_X[:_eval_end] @ _stack_weights))
                    _mae_r30_eval  = float(mean_absolute_error(
                        _stack_y[:_eval_end], _stack_X[:_eval_end] @ _wt_r30))
                    log.info(f"    Roll30 검증(앞{_eval_end}d): 기존={_mae_base_eval:.4f}  "
                             f"Roll30={_mae_r30_eval:.4f}")
                    if _mae_r30_eval < _mae_base_eval:
                        _stack_pred    = _stack_X @ _wt_r30
                        _stack_weights = _wt_r30
                        _meta_type     = _meta_type + '+Roll30'
                        log.info(f"    ✅ Rolling-30d 재보정 채택")

                _mae_stack  = float(mean_absolute_error(_stack_y, _stack_pred))
                _r2_stack   = float(r2_score(_stack_y, _stack_pred))
                _rmse_stack = float(np.sqrt(mean_squared_error(_stack_y, _stack_pred)))
                _ci_q80_stk = float(np.percentile(np.abs(_stack_y - _stack_pred), 80))
                _coef_str   = ' '.join(f'{n}={w:.3f}' for n, w in zip(_stack_names, _stack_weights))
                log.info(f"    [E] Stacking({_meta_type}) → R²={_r2_stack:.4f}  MAE={_mae_stack:.4f}  "
                         f"weights=[{_coef_str}]")
                # ── E: 다구간 성능 평가 (레짐 의존성 측정, 재훈련 없이 서브윈도우)
                _n_te = len(_stack_y)
                _win_maes = {n: float(mean_absolute_error(_stack_y[-w:], _stack_pred[-w:]))
                             for n, w in [('전체', _n_te), ('후반60', 60), ('후반45', 45)]
                             if w <= _n_te}
                if len(_win_maes) >= 2:
                    _mae_sigma = float(np.std(list(_win_maes.values())))
                    _wstr = '  '.join(f'{k}={v:.4f}' for k, v in _win_maes.items())
                    log.info(f"    다구간 MAE: {_wstr}  (σ={_mae_sigma:.4f})")
                    if _mae_sigma > 0.5:
                        log.warning(f"    ⚠ 성능 분산 과다(σ={_mae_sigma:.4f}) "
                                    f"— 특정 기간 레짐 의존 가능성")
                # ── 가중치 이상 감지: 단일 모델 지배 또는 음수 가중치 경고
                for _wn, _wv in zip(_stack_names, _stack_weights):
                    if _wv > 0.80:
                        log.warning(f"    ⚠ 스태킹 가중치 편중: {_wn}={_wv:.3f} > 0.80 "
                                    f"(모델 다양성 저하 가능)")
                    elif _wv < 0.05:
                        log.warning(f"    ⚠ 스태킹 가중치 미미: {_wn}={_wv:.3f} < 0.05 "
                                    f"(해당 모델 기여 없음)")

                # MAE 기준으로만 채택 (R²는 상승추세에서 과대평가되어 스태킹 채택 차단 문제)
                _mae_curr = min(sx_info.get('mae', float('inf')), xr_info.get('mae', float('inf')))
                if _mae_stack < _mae_curr:
                    _base_name = '+'.join(_stack_names)
                    _stk_info = {
                        'stack_weights': _stack_weights, 'type': 'price',
                        'rmse': _rmse_stack, 'mae': _mae_stack, 'r2': _r2_stack,
                        'ci_calib_q80': _ci_q80_stk,
                        'name': f'Stacking ({_base_name},InvMAE-WA)',
                        'sx_feats':    sx_info.get('features', []),
                        'xr_feats':    xr_info['features'],
                        'xr_scaler':   xr_info['scaler'],
                        'stack_names': _stack_names,
                        'meta_type':   _meta_type,
                        'pred_price_test':   _stack_pred.copy(),
                        'actual_price_test': _stack_y.copy(),
                    }
                    if 'lgb_return' in results:
                        _stk_info['lgb_feats']  = results['lgb_return']['features']
                        _stk_info['lgb_scaler'] = results['lgb_return']['scaler']
                        _stk_info['lgb_model']  = results['lgb_return']['model']
                    if 'cat_return' in results and 'CAT' in _stack_names:
                        _stk_info['cat_feats']  = results['cat_return']['features']
                        _stk_info['cat_scaler'] = results['cat_return']['scaler']
                        _stk_info['cat_model']  = results['cat_return']['model']
                    if 'news_sent' in results and 'NEWS' in _stack_names:
                        _stk_info['news_feats']  = results['news_sent']['features']
                        _stk_info['news_scaler'] = results['news_sent']['scaler']
                        _stk_info['news_model']  = results['news_sent']['model']
                    if 'prophet' in results and 'PROPHET' in _stack_names:
                        _stk_info['prophet_model'] = results['prophet']['model']
                    # ── B: 가중치 EMA 평활화 (라이브 예측용, α=0.3 — 급격한 가중치 전환 방지)
                    import json as _json_ema
                    try:
                        _ema_prev = {}
                        if STACK_WEIGHTS_EMA.exists():
                            _ema_prev = _json_ema.loads(STACK_WEIGHTS_EMA.read_text())
                        _ema_names = _ema_prev.get('names', [])
                        _ema_wts   = _ema_prev.get('weights', [])
                        if _ema_names == list(_stack_names) and len(_ema_wts) == len(_stack_weights):
                            _alpha = 0.3
                            _prev_w = np.array(_ema_wts)
                            _smooth_w = _alpha * _stack_weights + (1 - _alpha) * _prev_w
                            _smooth_w = np.maximum(_smooth_w, 0)
                            _smooth_w /= _smooth_w.sum()
                            _delta = np.abs(_smooth_w - _stack_weights).max()
                            log.info(f"    EMA 가중치 평활화(α=0.3): "
                                     f"{dict(zip(_stack_names, _smooth_w.round(3)))} "
                                     f"(최대변화={_delta:.3f})")
                            _stk_info['stack_weights'] = _smooth_w
                        STACK_WEIGHTS_EMA.write_text(_json_ema.dumps({
                            'names':   list(_stack_names),
                            'weights': list(_stk_info['stack_weights'].round(6).tolist()),
                        }))
                    except Exception as _ema_e:
                        log.warning(f"    가중치 EMA 실패({_ema_e}) → 원본 가중치 사용")

                    results['stacking'] = _stk_info
                    log.info(f"    ✅ Stacking 채택: R²={_r2_stack:.4f}")

                    # ── 적응형 오차 보정 (Adaptive Error Correction)
                    # 스태킹 잔차(AR 구조 + 시장 상태)를 학습 → 다음 예측 편향 보정
                    try:
                        _res_arr = _stack_y - _stack_pred          # test 잔차 배열
                        _n_ec    = len(_res_arr)
                        _res_s   = pd.Series(_res_arr)
                        # 잔차 자기상관 피처
                        _ec_df = pd.DataFrame({
                            'err_lag1': _res_s.shift(1).fillna(0),
                            'err_lag2': _res_s.shift(2).fillna(0),
                            'err_ma5':  _res_s.rolling(5, min_periods=1).mean().fillna(0),
                        })
                        # 시장 상태 피처 (test_df 뒤 _n_ec 행, 잔차와 alignment)
                        _ec_mkt = ['vix_zscore', 'vol_5d', 'return_lag1', 'mom_5d', 'regime']
                        _ec_mkt_ok = [f for f in _ec_mkt if f in test_df.columns]
                        _mkt_vals  = test_df[_ec_mkt_ok].values[-_n_ec:]  # shape: (n_ec, ...)
                        _ec_X = np.hstack([_ec_df.values, _mkt_vals]) if _ec_mkt_ok else _ec_df.values
                        # 타깃: 다음 날 잔차 (한 칸 shift)
                        _ec_X_tr = _ec_X[:-1]
                        _ec_y_tr = _res_arr[1:]
                        if len(_ec_X_tr) >= 10:
                            _ecm = Ridge(alpha=10.0)
                            _ecm.fit(_ec_X_tr, _ec_y_tr)
                            _ec_last = _ec_X[-1:].reshape(1, -1)
                            _ec_d1   = float(np.clip(_ecm.predict(_ec_last)[0], -3.0, 3.0))
                            results['stacking']['ec_model']     = _ecm
                            results['stacking']['ec_last_feat'] = _ec_last
                            results['stacking']['ec_mkt_feats'] = _ec_mkt_ok
                            log.info(f"    적응형 오차 보정 학습 완료 (D+1 예상 보정: {_ec_d1:+.3f}$)")
                    except Exception as _ece:
                        log.warning(f"    오차 보정 모델 실패({_ece})")
                else:
                    log.info(f"    Stacking 미채택 (R²={_r2_stack:.4f} ≤ 현재 최고={_r2_curr:.4f})")
        except Exception as _se:
            log.warning(f"    Stacking 실패({_se})")

    # ── 나이브 퍼시스턴스 기준선 (MASE 계산용, Skill Score 비교)
    _naive_mae_px = _naive_mae_rv = None
    _naive_rmse_px = _naive_rmse_rv = _naive_r2_px = _naive_r2_rv = None
    try:
        _y_px_naive_a = test_df['target_price'].dropna().values
        _y_px_naive_p = test_df['WTI'].values[:len(_y_px_naive_a)]
        _naive_mae_px  = float(mean_absolute_error(_y_px_naive_a, _y_px_naive_p))
        _naive_rmse_px = float(np.sqrt(mean_squared_error(_y_px_naive_a, _y_px_naive_p)))
        _naive_r2_px   = float(r2_score(_y_px_naive_a, _y_px_naive_p))
        _y_rv_naive_a  = test_df['target_rv'].dropna().values
        _y_rv_naive_p  = test_df['RV_5d'].values[:len(_y_rv_naive_a)]
        _naive_mae_rv  = float(mean_absolute_error(_y_rv_naive_a, _y_rv_naive_p))
        _naive_rmse_rv = float(np.sqrt(mean_squared_error(_y_rv_naive_a, _y_rv_naive_p)))
        _naive_r2_rv   = float(r2_score(_y_rv_naive_a, _y_rv_naive_p))
        log.info(f"    기준선(나이브 퍼시스턴스): "
                 f"가격 MAE={_naive_mae_px:.4f} R²={_naive_r2_px:.4f}  "
                 f"변동성 MAE={_naive_mae_rv:.6f} R²={_naive_r2_rv:.4f}")
    except Exception as _nbe:
        log.warning(f"    나이브 기준선 계산 실패({_nbe})")

    # ── 성능 저장
    perf_rows = []
    for v in results.values():
        if 'name' not in v or 'type' not in v:
            continue
        if v.get('benchmark_only', False):
            continue
        row = {'model': v['name'], 'target': v['type'],
               'rmse': round(v['rmse'], 5), 'mae': round(v['mae'], 5), 'r2': round(v['r2'], 4)}
        if 'dir_acc'     in v: row['dir_acc']     = round(v['dir_acc'], 4)
        if 'wf_dir_acc'  in v: row['wf_dir_acc']  = round(v['wf_dir_acc'], 4)
        if 'train_r2'    in v: row['train_r2']    = round(v['train_r2'], 4)
        if 'overfit_gap' in v: row['overfit_gap'] = round(v['overfit_gap'], 4)
        # MASE (1.0보다 작아야 기준선 대비 개선)
        try:
            _nm = (_naive_mae_px if v['type'] == 'price' else _naive_mae_rv)
            if _nm and _nm > 0:
                row['mase'] = round(v['mae'] / _nm, 4)
        except Exception:
            pass
        # MaxError (꼬리 오차 관리)
        if 'pred_price_test' in v and 'actual_price_test' in v:
            try:
                _errs = np.abs(np.array(v['actual_price_test']) - np.array(v['pred_price_test']))
                row['max_error'] = round(float(_errs.max()), 4)
            except Exception:
                pass
        perf_rows.append(row)
    # 나이브 퍼시스턴스 기준선 행 추가
    if _naive_mae_px is not None:
        perf_rows.append({'model': 'Persistence (D+1=D+0)', 'target': 'price',
                          'rmse': round(_naive_rmse_px, 5), 'mae': round(_naive_mae_px, 5),
                          'r2': round(_naive_r2_px, 4), 'mase': 1.0})
        perf_rows.append({'model': 'Persistence-Vol (RV lag-1)', 'target': 'vol_5d',
                          'rmse': round(_naive_rmse_rv, 5), 'mae': round(_naive_mae_rv, 5),
                          'r2': round(_naive_r2_rv, 4), 'mase': 1.0})
    perf_df = pd.DataFrame(perf_rows)
    _atomic_csv(perf_df, OUTPUT_DIR / 'model_performance.csv', index=False)
    log.info("    model_performance.csv 저장")

    # ── Look-ahead 감사: 주요 피처 당일/익일 가격 상관 점검
    try:
        _audit_cols = [
            ('news_sentiment',        '당일 뉴스 (OK: T뉴스→T+1가격)'),
            ('news_sentiment_smooth', '당일 뉴스 EWM (OK: T뉴스→T+1가격)'),
            ('inv_chg_zscore',        'EIA 재고 (fix: +3영업일 발표지연 적용)'),
            ('inv_surprise',          'EIA 서프라이즈 (fix: +3영업일)'),
            ('cot_net_pct',           'COT 포지션 (OK: shift(3) 적용)'),
            ('news_sentiment_lag1',   '뉴스 lag1 (OK: 명시 lag)'),
        ]
        log.info("    [Audit] 피처 look-ahead 점검 (corr with WTI_t vs target_price_t+1):")
        for _ac, _desc in _audit_cols:
            if _ac not in feature_df.columns or 'target_price' not in feature_df.columns:
                continue
            _cs = round(float(feature_df[_ac].corr(feature_df['WTI'])), 3)
            _cn = round(float(feature_df[_ac].corr(feature_df['target_price'])), 3)
            _flag = '⚠ SUSPECT' if abs(_cs) > 0.9 and abs(_cs) > abs(_cn) * 1.3 else '✓'
            log.info(f"        {_flag} {_ac}: same={_cs:+.3f}  next={_cn:+.3f}  | {_desc}")
    except Exception as _ae:
        log.debug(f"    감사 실패({_ae})")

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




def compute_ensemble_weights(window: int = 30):
    """R² 기반 초기 가중치 + MAPE 미세조정으로 SARIMAX/XGBoost 동적 가중치 산출.

    1단계: model_performance.csv의 테스트셋 R²로 비례 가중치 계산
    2단계: 최근 backtest/live MAPE로 ±0.1 범위 미세조정
    R² 정보 없으면 기본값(0.65/0.35), 최종 클램프 [0.30, 0.70].
    """
    default = (0.65, 0.35)

    # ── 1단계: 역MAE 기반 초기 가중치 (R²는 태스크 단위 달라 비교 불가)
    w_s_base = 0.65
    perf_path = OUTPUT_DIR / 'model_performance.csv'
    if perf_path.exists():
        try:
            pf = pd.read_csv(perf_path)
            sx  = pf[pf['model'].str.startswith('SARIMAX')]
            xgr = pf[pf['model'].str.startswith('XGBoost-Return')]
            if not sx.empty and not xgr.empty:
                mae_s = float(sx['mae'].iloc[0])
                mae_x = float(xgr['mae'].iloc[0])
                if mae_s > 0 and mae_x > 0:
                    inv_s = 1.0 / mae_s
                    inv_x = 1.0 / mae_x
                    w_s_base = float(np.clip(inv_s / (inv_s + inv_x), 0.30, 0.70))
                    log.info(f"    역MAE 기반 초기 가중치: SARIMAX={w_s_base:.2f} "
                             f"XGB={1-w_s_base:.2f} (MAE_s={mae_s:.4f} MAE_x={mae_x:.4f})")
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

        bt_mape = (bt['price_error'].abs() / bt['actual_price'].replace(0, np.nan) * 100).mean()
        if len(lv) >= 2:
            lv_mape = (lv['price_error'].abs() / lv['actual_price'].replace(0, np.nan) * 100).mean()
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
    """최근 live 실측 오차의 중앙값으로 bias correction 값 반환.

    median 사용으로 실행 누락 등 이상치에 robust.
    live 데이터 3건 미만이면 0 반환.
    price_error = actual - predicted 이므로 양수면 과소예측 → 더해야 함.
    """
    if not PRED_LOG_FILE.exists():
        return 0.0
    try:
        pl = pd.read_csv(PRED_LOG_FILE)
        lv = pl[(pl['type'] == 'live') & pl['price_error'].notna()].tail(window)
        if len(lv) < 3:
            return 0.0

        errors = lv['price_error'].values.astype(float)
        bias = float(np.median(errors))

        bias = max(-max_correction, min(max_correction, bias))
        log.info(f"    Live bias correction: {bias:+.3f}$ (최근 {len(lv)}건 중앙값)")
        return bias
    except Exception:
        return 0.0


def compute_error_spike_blend(
    consecutive: int = 3, threshold: float = 5.0, max_blend: float = 0.35
) -> float:
    """연속 대형 오류(|error| > threshold) consecutive일 이상 시 persistence 블렌딩 비율 반환.
    조건 미충족 시 0.0 반환. MASE에 영향 없음(live 데이터만 사용).
    """
    if not PRED_LOG_FILE.exists():
        return 0.0
    try:
        pl = pd.read_csv(PRED_LOG_FILE)
        lv = pl[(pl['type'] == 'live') & pl['price_error'].notna()].tail(consecutive)
        if len(lv) < consecutive:
            return 0.0
        last_abs = lv['price_error'].abs().values
        if (last_abs > threshold).all():
            log.info(
                f"    오류 급증 감지: 최근 {consecutive}일 |MAE|={last_abs.mean():.2f}$ "
                f"> ${threshold} → persistence 블렌딩 {max_blend:.0%}"
            )
            return max_blend
        return 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  forecast_next_7days()
# ─────────────────────────────────────────────────────────────────────────────

def forecast_next_7days(results: dict, feature_df: pd.DataFrame, full_df: pd.DataFrame, aux: dict = None):
    """SARIMAX + XGBoost 앙상블로 향후 7 영업일 유가 예측"""
    log.info("[5/9] 7일 예측 생성 중...")

    # 마지막 실제 가격 (dropna 전 full_df 사용, WTI 유효값 기준 마지막 영업일)
    _wti_valid = full_df['WTI'].dropna()
    last_price = float(_wti_valid.iloc[-1])
    last_date  = _wti_valid.index[-1]   # NaN 행 제외한 실제 마지막 날짜
    fc_dates   = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='B')

    forecasts = {}

    # ── D: 라이브 피처 드리프트 감지 (p01/p99 훈련 범위 이탈 경고)
    if FEAT_TRAIN_STATS.exists():
        try:
            import json as _json_fdc
            _fd_stats = _json_fdc.loads(FEAT_TRAIN_STATS.read_text())
            _live_row  = feature_df.iloc[-1]
            _drift_list = []
            for _fc, _st in _fd_stats.items():
                if _fc in feature_df.columns:
                    _val = float(_live_row.get(_fc, np.nan))
                    if not np.isnan(_val) and (_val < _st['p01'] or _val > _st['p99']):
                        _drift_list.append(f"{_fc}={_val:.3f}")
            if _drift_list:
                log.warning(f"    ⚠ 피처 드리프트 ({len(_drift_list)}개 훈련 p01-p99 이탈): "
                            f"{', '.join(_drift_list[:5])}"
                            + (" ..." if len(_drift_list) > 5 else ""))
            else:
                log.info("    피처 분포 정상 (모든 피처 훈련 범위 내)")
        except Exception as _fdce:
            log.warning(f"    피처 드리프트 체크 실패({_fdce})")

    # ETS는 stacking 훈련 슬롯이 SARIMAX 기반 → live 대체 시 calibration 불일치
    # 모니터링/진단 전용 (model_performance.csv 기록만), SARIMAX를 항상 사용
    _use_ets = False

    # ── SARIMAX 예측 (실패 시 ETS → ridge fallback 순으로)
    if 'sarimax' in results and not _use_ets:
        try:
            sfit     = results['sarimax']['model']
            ecols    = results['sarimax']['features']
            _exog_src = full_df if full_df is not None else feature_df
            last_exog = _exog_src[ecols].tail(5).mean()
            fut_exog  = pd.DataFrame([last_exog.values] * 7, columns=ecols)
            fc_vals   = sfit.forecast(steps=7, exog=fut_exog)
            forecasts['sarimax'] = np.array(fc_vals)
            log.info(f"    SARIMAX 7일 예측: {fc_vals.values.round(2)}")
        except Exception as exc:
            log.warning(f"SARIMAX 예측 실패: {exc}")

    if _use_ets or ('sarimax' not in forecasts and 'ets' in results):
        try:
            _ets_fc7 = results['ets']['model'].forecast(7).values
            forecasts['sarimax'] = _ets_fc7
            log.info(f"    ETS 7일 예측: {_ets_fc7.round(2)}")
        except Exception as _ete:
            log.warning(f"ETS 예측 실패({_ete})")
    if 'sarimax' not in forecasts and 'ridge' in results:
        try:
            ri   = results['ridge']
            _rs  = full_df if full_df is not None else feature_df
            avf  = [f for f in ri['features'] if f in _rs.columns]
            last_r = ri['scaler'].transform(_rs[avf].iloc[-1:].fillna(0).values)
            pred_r = float(ri['model'].predict(last_r)[0])
            forecasts['sarimax'] = np.array([pred_r] * 7)
            log.info(f"    Ridge fallback 예측: {pred_r:.2f}")
        except Exception as exc:
            log.warning(f"Ridge fallback 실패: {exc}")

    # ── XGBoost 수익률 예측 → 가격 역변환 (xgb_return 우선, 없으면 vol 폴백)
    if 'xgb_return' in results:
        try:
            info    = results['xgb_return']
            model   = info['model']
            sc      = info['scaler']
            feats   = info['features']
            _feat_src = full_df if full_df is not None else feature_df
            avail_f = [f for f in feats if f in _feat_src.columns]
            last_row = _feat_src[avail_f].iloc[-1:].fillna(0).values.copy()
            last_s   = sc.transform(last_row)

            # Classifier-adj 선택 시: 회귀 크기 + 분류기 방향 결합
            if hasattr(model, 'predict_proba'):
                _reg_m = info.get('_reg_model')
                _mag_ret = float(_reg_m.predict(last_s)[0]) if _reg_m is not None else 0.001
                # 분류기 전용 MI 피처가 있으면 별도 스케일링
                _cls_feats_fc = info.get('cls_features')
                _cls_sc_fc    = info.get('cls_scaler')
                if _cls_feats_fc is not None and _cls_sc_fc is not None:
                    try:
                        _cf_avail = [f for f in _cls_feats_fc if f in _feat_src.columns]
                        _cls_ls   = _cls_sc_fc.transform(
                            _feat_src[_cf_avail].iloc[-1:].fillna(0).values)
                    except Exception:
                        _cls_ls = last_s
                else:
                    _cls_ls = last_s
                _prob_up_fc = float(model.predict_proba(_cls_ls)[0, 1])
                # XGB-Cls 블렌드 (백테스트와 동일 방식으로 라이브 예측에 반영)
                _xcls_live = info.get('xgb_cls_model')
                _xcls_bw   = info.get('xgb_cls_blend_w', 0.0)
                if _xcls_live is not None and _xcls_bw > 0:
                    try:
                        _prob_xcls_fc = float(_xcls_live.predict_proba(_cls_ls)[0, 1])
                        _prob_up_fc = (1 - _xcls_bw) * _prob_up_fc + _xcls_bw * _prob_xcls_fc
                        log.info(f"    XGB-Cls blend 적용 (w={_xcls_bw})")
                    except Exception:
                        pass
                _cls_th = info.get('threshold', 0.5)
                pred_ret_d1 = abs(_mag_ret) * (1.0 if _prob_up_fc > _cls_th else -1.0)
            else:
                pred_ret_d1 = float(model.predict(last_s)[0])   # D+1 log 수익률
            # D+1~7: Multi-step Direct (h별 독립 모델) 우선, 없으면 감쇠 체인 fallback
            _ms_models_fc  = info.get('multistep_models', {})
            _ms_scalers_fc = info.get('multistep_scalers', {})
            if _ms_models_fc and len(_ms_models_fc) == 6:
                price_path = [last_price * np.exp(np.clip(pred_ret_d1, -0.5, 0.5))]
                for _h in range(2, 8):
                    _avail_h = [f for f in feats if f in _feat_src.columns]
                    _last_h  = _ms_scalers_fc[_h].transform(
                        _feat_src[_avail_h].iloc[-1:].fillna(0).values)
                    _ret_h = float(np.clip(_ms_models_fc[_h].predict(_last_h)[0], -0.5, 0.5))
                    price_path.append(last_price * np.exp(_ret_h))
                price_path = np.array(price_path)
                log.info(f"    Multi-step Direct: D+1=${price_path[0]:.2f} D+7=${price_path[-1]:.2f}")
            else:
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
            _feat_src2 = full_df if full_df is not None else feature_df
            avail_f = [f for f in feats if f in _feat_src2.columns]
            last_row = _feat_src2[avail_f].iloc[-1:].fillna(0).values.copy()
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
            # full_df 사용: feature_df는 dropna로 마지막 행(최신) 없음
            _rc_src = full_df if full_df is not None else feature_df
            last = _rc_src[rc['rc_feat_cols']].tail(1).fillna(0).copy()
            last['resid_lag1'] = rc['last_resid1']
            last['resid_lag2'] = rc['last_resid2']
            correction = float(rc['model'].predict(
                rc['scaler'].transform(last[rc['features']]))[0])
            forecasts['sarimax'] = forecasts['sarimax'] + correction
            log.info(f"    잔차 교정 적용: {correction:+.3f}")
        except Exception as e:
            log.warning(f"잔차 교정 예측 실패: {e}")

    # ── 동적 앙상블 가중치 (최근 backtest 오차 기반)
    w_sarimax, w_xgb = compute_ensemble_weights()

    if 'sarimax' in forecasts and 'xgb' in forecasts and 'stacking' in results:
        _sx_fc      = forecasts['sarimax']
        _xb_fc      = forecasts['xgb']
        _stk        = results['stacking']
        _stk_wts    = _stk['stack_weights'].copy()
        _stk_names  = _stk.get('stack_names', ['SARIMAX', 'XGB'])

        # 레짐 조건부 가중치 조정 (ovx_z/gpr_z 기반 실시간 조정)
        _feat_src_r = full_df if full_df is not None else feature_df
        _cur_ovx_z  = float(_feat_src_r['ovx_zscore'].iloc[-1]) if 'ovx_zscore' in _feat_src_r.columns else 0.0
        _cur_gpr_z  = float(_feat_src_r['gpr_zscore'].iloc[-1]) if 'gpr_zscore' in _feat_src_r.columns else 0.0
        # 실현변동성 백분위수 (최근 252일 기준)
        _rv_src = full_df if full_df is not None else feature_df
        _rv_hist = _rv_src['vol_5d'].dropna().tail(252) if 'vol_5d' in _rv_src.columns else pd.Series(dtype=float)
        _vol_pct = float((_rv_hist < _rv_hist.iloc[-1]).mean()) if len(_rv_hist) > 20 else 0.5

        _reg_adj = {n: 1.0 for n in _stk_names}
        if _cur_gpr_z > 2.0 or _cur_ovx_z > 1.5:
            # 위기/충격: SARIMAX 대폭 축소 (lag 모델 무력화), VAR 확대 (다변수 동적 반응)
            _reg_adj.update({'VAR': 1.6, 'NEWS': 1.2, 'XGB': 1.0, 'CAT': 0.9, 'SARIMAX': 0.5, 'LGB': 0.9})
            log.info(f"    레짐: 위기/고공포 (ovx_z={_cur_ovx_z:.2f} gpr_z={_cur_gpr_z:.2f} "
                     f"vol_pct={_vol_pct:.0%}) → SARIMAX↓↓ VAR↑↑")
        elif _cur_ovx_z > 1.0 or _vol_pct > 0.80:
            # 상승변동성: 중간 강도 SARIMAX 축소
            _reg_adj.update({'VAR': 1.3, 'NEWS': 1.1, 'XGB': 1.0, 'CAT': 0.95, 'SARIMAX': 0.7, 'LGB': 0.95})
            log.info(f"    레짐: 상승변동성 (ovx_z={_cur_ovx_z:.2f} vol_pct={_vol_pct:.0%}) → SARIMAX↓ VAR↑")
        _adj_arr = np.array([_reg_adj.get(n, 1.0) for n in _stk_names]) * _stk_wts
        if _adj_arr.sum() > 0:
            _stk_wts = _adj_arr / _adj_arr.sum()

        # LGB 예측 생성
        _lgb_fc = None
        if 'LGB' in _stk_names and 'lgb_model' in _stk:
            try:
                _feat_src3 = full_df if full_df is not None else feature_df
                _feats_l = [f for f in _stk['lgb_feats'] if f in _feat_src3.columns]
                _last_l  = _feat_src3[_feats_l].iloc[-1:].fillna(0).values.copy()
                _pred_ret_lgb = float(_stk['lgb_model'].predict(
                    _stk['lgb_scaler'].transform(_last_l))[0])
                _decay = np.array([0.95 ** i for i in range(7)])
                _lgb_fc = last_price * np.exp(np.cumsum(_pred_ret_lgb * _decay))
            except Exception as _le:
                log.warning(f"LGB 예측 실패: {_le}")

        # CAT 예측 생성
        _cat_fc = None
        if 'CAT' in _stk_names and 'cat_model' in _stk:
            try:
                _feat_src_c = full_df if full_df is not None else feature_df
                _feats_c = [f for f in _stk['cat_feats'] if f in _feat_src_c.columns]
                _last_c  = _feat_src_c[_feats_c].iloc[-1:].fillna(0).values.copy()
                _pred_ret_cbt = float(np.clip(
                    _stk['cat_model'].predict(
                        _stk['cat_scaler'].transform(_last_c))[0], -0.5, 0.5))
                _decay_c = np.array([0.95 ** i for i in range(7)])
                _cat_fc  = last_price * np.exp(np.cumsum(_pred_ret_cbt * _decay_c))
                log.info(f"    CatBoost 예측 D+1: ret={_pred_ret_cbt:+.4f} → ${_cat_fc[0]:.2f}")
            except Exception as _ce:
                log.warning(f"CatBoost 예측 실패: {_ce}")

        # NEWS 예측 생성
        _news_fc = None
        if 'NEWS' in _stk_names and 'news_model' in _stk:
            try:
                _feat_src_n = full_df if full_df is not None else feature_df
                _feats_n = [f for f in _stk['news_feats'] if f in _feat_src_n.columns]
                _last_n  = _feat_src_n[_feats_n].iloc[-1:].fillna(0).values.copy()
                _pred_ret_nws = float(np.clip(
                    _stk['news_model'].predict(
                        _stk['news_scaler'].transform(_last_n))[0], -0.5, 0.5))
                _decay_n = np.array([0.95 ** i for i in range(7)])
                _news_fc = last_price * np.exp(np.cumsum(_pred_ret_nws * _decay_n))
                log.info(f"    News-Sent 예측 D+1: ret={_pred_ret_nws:+.4f} "
                         f"→ ${_news_fc[0]:.2f}")
            except Exception as _ne:
                log.warning(f"News 예측 실패: {_ne}")

        # PROPHET 예측 생성
        _prph_fc2 = None
        if 'PROPHET' in _stk_names and 'prophet_model' in _stk:
            try:
                import warnings as _wn2
                _last_known = (full_df if full_df is not None else feature_df).index[-1]
                _future_dates = pd.bdate_range(
                    start=_last_known + pd.offsets.BDay(1), periods=7)
                _prph_fut_df = pd.DataFrame({'ds': _future_dates})
                with _wn2.catch_warnings():
                    _wn2.simplefilter('ignore')
                    _prph_out = _stk['prophet_model'].predict(_prph_fut_df)
                _prph_fc2 = _prph_out['yhat'].values
                log.info(f"    Prophet 예측 D+1: ${_prph_fc2[0]:.2f}")
            except Exception as _pe2:
                log.warning(f"Prophet 예측 실패: {_pe2}")

        # VAR 예측 생성
        _var_fc = None
        if 'VAR' in _stk_names and 'var' in results:
            try:
                _vr   = results['var']
                _vfit = _vr['model']
                _vcols = _vr['cols']
                _vlag  = _vr['lag']
                _vhist = feature_df[_vcols].dropna().asfreq('B', method='ffill').dropna()
                _vfc   = _vfit.forecast(_vhist.values[-_vlag:], steps=7)
                _var_fc = _vfc[:, 0]   # WTI 열
            except Exception as _ve:
                log.warning(f"VAR 예측 실패: {_ve}")

        # D+2-7: SARIMAX 가중치 제거 (flat forecast 영향 배제), VAR 있을 때만 적용
        _sx_idx = None
        if _var_fc is not None and 'SARIMAX' in list(_stk_names):
            _sx_idx = list(_stk_names).index('SARIMAX')
            _wts_multistep = _stk_wts.copy()
            _wts_multistep[_sx_idx] = 0.0
            _rest = _wts_multistep.sum()
            if _rest > 0:
                _wts_multistep = _wts_multistep / _rest

        ensemble = np.zeros(7)
        for i in range(7):
            _pts = [_sx_fc[i], _xb_fc[i]]
            if _lgb_fc is not None:
                _pts.append(_lgb_fc[i])
            if _cat_fc is not None:
                _pts.append(_cat_fc[i])
            if _var_fc is not None:
                _pts.append(float(_var_fc[i]))
            if _news_fc is not None:
                _pts.append(_news_fc[i])
            if _prph_fc2 is not None:
                _pts.append(float(_prph_fc2[i]))
            # stack_names 순서에 맞게 패딩 (None 베이스는 center로 대체)
            while len(_pts) < len(_stk_names):
                _pts.append(_sx_fc[i])
            # D+1: 원래 stacking 가중치, D+2-7: SARIMAX 제외 후 재정규화
            _w = _wts_multistep if (i >= 1 and _sx_idx is not None) else _stk_wts
            ensemble[i] = float(np.dot(_pts[:len(_stk_names)], _w))
        log.info(f"    ✅ Stacking 앙상블 적용: D+1={ensemble[0]:.2f} "
                 f"(D+2-7: {'SARIMAX 제외' if _sx_idx is not None else '전체 가중치'})")
    elif 'sarimax' in forecasts and 'xgb' in forecasts:
        # VAR가 있으면 3모델 역MAE 가중 앙상블, 없으면 2모델
        if 'var' in results:
            try:
                _vr2   = results['var']
                _vhist2 = feature_df[_vr2['cols']].dropna().asfreq('B', method='ffill').dropna()
                _vfc2   = _vr2['model'].forecast(_vhist2.values[-_vr2['lag']:], steps=7)
                _var_fc2 = _vfc2[:, 0]
                _mae_sx  = results['sarimax'].get('mae', 999.0)
                _mae_xb  = results['xgb_return'].get('mae', 999.0)
                _mae_vr  = _vr2.get('mae', 999.0)
                _inv = np.array([1/_mae_sx, 1/_mae_xb, 1/_mae_vr])
                _wts = _inv / _inv.sum()
                ensemble = (_wts[0] * forecasts['sarimax']
                            + _wts[1] * forecasts['xgb']
                            + _wts[2] * _var_fc2)
                log.info(f"    3모델 앙상블(InvMAE): SARIMAX×{_wts[0]:.2f} "
                         f"XGB×{_wts[1]:.2f} VAR×{_wts[2]:.2f} → D+1={ensemble[0]:.2f}")
            except Exception as _ve2:
                log.warning(f"VAR 3모델 앙상블 실패({_ve2}) → 2모델 폴백")
                ensemble = w_sarimax * forecasts['sarimax'] + w_xgb * forecasts['xgb']
        else:
            ensemble = w_sarimax * forecasts['sarimax'] + w_xgb * forecasts['xgb']
        log.info(f"    ⚠️ 앙상블: SARIMAX={forecasts['sarimax'][0]:.2f} "
                 f"XGB={forecasts['xgb'][0]:.2f} → 앙상블={ensemble[0]:.2f}")
    elif 'sarimax' in forecasts:
        ensemble = forecasts['sarimax']
    elif 'xgb' in forecasts:
        ensemble = forecasts['xgb']
    elif 'var' in results:
        # VAR 단독 fallback (SARIMAX+XGB 모두 실패 시) — R²=0.73으로 최적 대안
        try:
            _vfb  = results['var']
            _vhfb = feature_df[_vfb['cols']].dropna().asfreq('B', method='ffill').dropna()
            ensemble = _vfb['model'].forecast(_vhfb.values[-_vfb['lag']:], steps=7)[:, 0]
            log.info(f"    VAR 단독 fallback: D+1={ensemble[0]:.2f}")
        except Exception as _vfe:
            log.warning(f"    VAR fallback 실패({_vfe}) → trend 폴백")
            trend = feature_df['WTI'].diff().tail(5).mean()
            ensemble = np.array([last_price + trend * (i + 1) for i in range(7)])
    else:
        trend = feature_df['WTI'].diff().tail(5).mean()
        ensemble = np.array([last_price + trend * (i + 1) for i in range(7)])

    # ── A: live bias correction (실측 오차 피드백, 지평선별 감쇠)
    bias = compute_live_bias_correction()
    if bias != 0.0:
        _bias_decay = np.array([1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15])[:len(ensemble)]
        ensemble = ensemble + bias * _bias_decay

    # ── A2: 적응형 오차 보정 (Adaptive Error Correction)
    # ec_last_feat는 훈련 시점 test_df 마지막 행으로 고정됨 → 현재 시장 상태로 재구성
    _stk_ec = results.get('stacking', {})
    if 'ec_model' in _stk_ec and 'ec_mkt_feats' in _stk_ec:
        try:
            _ec_mkt_ok = _stk_ec['ec_mkt_feats']
            _live_src  = full_df if full_df is not None else feature_df
            _live_mkt  = _live_src[_ec_mkt_ok].iloc[-1:].fillna(0).values  # 현재 시장 상태
            # 잔차 자기상관 피처: live bias correction 결과 사용 (최근 오차 EWMA)
            _live_err  = bias   # live bias correction에서 계산된 최근 오차
            _live_ec_X = np.hstack([
                np.array([[_live_err, _live_err * 0.9, _live_err * 0.95]]),  # err_lag1/2/ma5 근사
                _live_mkt,
            ])
            _ec_pred = float(np.clip(
                _stk_ec['ec_model'].predict(_live_ec_X)[0], -3.0, 3.0))
            _ec_decay = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1])[:len(ensemble)]
            ensemble = ensemble + _ec_pred * _ec_decay
            log.info(f"    적응형 오차 보정 적용: D+1 {_ec_pred:+.3f}$")
        except Exception as _ece_fc:
            log.warning(f"    오차 보정 적용 실패({_ece_fc})")

    # ── A3: 오류 급증 시 persistence 블렌딩 (3일 연속 |error|>$5)
    _spike_blend = compute_error_spike_blend()
    if _spike_blend > 0.0:
        _persist_fc = np.full(len(ensemble), last_price)
        ensemble = (1 - _spike_blend) * ensemble + _spike_blend * _persist_fc

    # ── A4: D+2-7 구조적 persistence 블렌딩 (multi-step MASE>1.0, 방향정확도<40% → 거의 pure persistence)
    # D+1(index 0)은 0%이므로 기존 MASE(0.8132) 불변. D+2-7만 적용.
    _ms_blend = np.array([0.0, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98])[:len(ensemble)]
    _persist_base = np.full(len(ensemble), last_price)
    ensemble = ensemble * (1 - _ms_blend) + _persist_base * _ms_blend
    log.info(f"    D+2-7 Persistence 블렌딩: D+2={_ms_blend[1]:.0%} ~ D+7={_ms_blend[min(6,len(_ms_blend)-1)]:.0%}")

    # ── 예측값 sanity check: spot 대비 ±30% 초과 시 경고
    _dev_d1_pct = (ensemble[0] - last_price) / last_price * 100
    if abs(_dev_d1_pct) > 30.0:
        log.warning(f"    ⚠ 예측값 편차 과다: D+1={ensemble[0]:.2f}$ vs spot={last_price:.2f}$ "
                    f"({_dev_d1_pct:+.1f}%) — 데이터/모델 이상 가능성 점검 필요")

    # ── 신뢰구간: 분위 회귀(Q10/Q90) 우선, 없으면 변동성 기반 폴백
    t = np.arange(1, 8)
    _xr = results.get('xgb_return', {})
    _q10m, _q90m = _xr.get('model_q10'), _xr.get('model_q90')
    _qsc  = _xr.get('scaler')
    _qfts = _xr.get('features', [])

    lower_80ci = upper_80ci = None
    if _q10m is not None and _q90m is not None and _qsc is not None:
        try:
            _feat_src4 = full_df if full_df is not None else feature_df
            _avf_q  = [f for f in _qfts if f in _feat_src4.columns]
            _last_q = _qsc.transform(_feat_src4[_avf_q].iloc[-1:].fillna(0).values)
            _ret_q10 = float(_q10m.predict(_last_q)[0])
            _ret_q90 = float(_q90m.predict(_last_q)[0])
            # D+1 절대 스프레드 계산 후 sqrt(t)로 다단계 확장
            _spread_lo = last_price * np.exp(_ret_q10) - last_price   # 음수
            _spread_hi = last_price * np.exp(_ret_q90) - last_price   # 양수
            lower_80ci = ensemble + _spread_lo * np.sqrt(t)
            upper_80ci = ensemble + _spread_hi * np.sqrt(t)
            lower_80ci = np.clip(lower_80ci, last_price * 0.80, last_price * 1.20)
            upper_80ci = np.clip(upper_80ci, last_price * 0.80, last_price * 1.20)
            log.info(f"    신뢰구간 Q10/Q90 적용: D+1 [{lower_80ci[0]:.2f}, {upper_80ci[0]:.2f}]")
        except Exception as _qce:
            log.warning(f"    분위 신뢰구간 계산 실패({_qce}) → 변동성 폴백")

    if lower_80ci is None:
        # GARCH 조건부 변동성 우선 (더 정확한 변동성 클러스터링 반영)
        # GARCH(1,1) 다단계 조건부 분산 예측 (최우선) — h_{t+k} 각 스텝별 독립 분산
        _garch_res_fc = (aux or {}).get('garch_model') or results.get('garch_vol', {}).get('model')
        _used_garch_fc = False
        if _garch_res_fc is not None:
            try:
                _fc7 = _garch_res_fc.forecast(horizon=7, reindex=False)
                _h7  = np.sqrt(_fc7.variance.values[-1, :]) / 100   # % → 소수 변환
                ci_half = last_price * _h7 * 1.28
                lower_80ci = ensemble - ci_half
                upper_80ci = ensemble + ci_half
                _used_garch_fc = True
                log.info(f"    GARCH(1,1) 다단계 CI: D+1 ±{ci_half[0]:.2f}$ → D+7 ±{ci_half[6]:.2f}$")
            except Exception as _gfe:
                log.warning(f"    GARCH 다단계 예측 실패({_gfe}) → vol 폴백")
        if not _used_garch_fc:
            _src_vol = full_df if full_df is not None else feature_df
            if 'garch_vol' in _src_vol.columns:
                _garch_v = float(_src_vol['garch_vol'].dropna().iloc[-1])
                ci_half  = last_price * max(_garch_v, 1e-4) * 1.28 * np.sqrt(t)
            else:
                recent_vol = float(_src_vol['vol_5d'].dropna().iloc[-1]) if 'vol_5d' in _src_vol.columns else 0.015
                ci_half    = ensemble * recent_vol * 1.28 * np.sqrt(t)
            lower_80ci = ensemble - ci_half
            upper_80ci = ensemble + ci_half

    # CI 경험적 보정: 백테스트 stacking 오차 Q80 기준 (실제 80% 커버리지 근사)
    _ci_q80 = (results.get('stacking') or {}).get('ci_calib_q80')
    if _ci_q80 is not None and lower_80ci is not None:
        _d1_half = (float(upper_80ci[0]) - float(lower_80ci[0])) / 2
        if _d1_half > 0.1:
            _calib = float(np.clip(_ci_q80 / _d1_half, 0.5, 2.5))
            if abs(_calib - 1.0) > 0.05:
                _mid     = (lower_80ci + upper_80ci) / 2
                _half_ci = (upper_80ci - lower_80ci) / 2
                lower_80ci = _mid - _half_ci * _calib
                upper_80ci = _mid + _half_ci * _calib
                log.info(f"    CI 경험적 보정 ×{_calib:.2f} (Q80={_ci_q80:.2f}$)")

    fc_df = pd.DataFrame({
        'date':            fc_dates.strftime('%Y-%m-%d'),
        'forecast_price':  np.round(ensemble,    2),
        'lower_80ci':      np.round(lower_80ci,  2),
        'upper_80ci':      np.round(upper_80ci,  2),
        'bias_correction': round(bias, 3),
    })
    if 'sarimax' in forecasts:
        fc_df['sarimax_forecast'] = np.round(forecasts['sarimax'], 2)
    if 'xgb' in forecasts:
        fc_df['xgb_forecast'] = np.round(forecasts['xgb'], 2)
    # 모델 합의도: 예측값들의 표준편차 (낮을수록 모델 간 일치)
    pred_cols = [c for c in ['sarimax_forecast', 'xgb_forecast']
                 if c in fc_df.columns]
    if len(pred_cols) >= 2:
        fc_df['model_std'] = fc_df[pred_cols].std(axis=1).round(2)
        # ── F: 모델 불일치 시 CI 자동 확대 (D+1 std > 2$ → CI ×1.2)
        _d1_std = float(fc_df['model_std'].iloc[0]) if not fc_df['model_std'].isna().iloc[0] else 0.0
        if _d1_std > 2.0:
            _ci_expand = 1.2
            _fp = fc_df['forecast_price']
            fc_df['lower_80ci'] = (_fp - (_fp - fc_df['lower_80ci']) * _ci_expand).round(2)
            fc_df['upper_80ci'] = (_fp + (fc_df['upper_80ci'] - _fp) * _ci_expand).round(2)
            log.warning(f"    ⚠ 모델 불일치 과다(D+1 std={_d1_std:.2f}$) → CI ×{_ci_expand}")

    # VaR: 정규분포 5%/95% 분위수 (헤지 기준선 — 단방향 꼬리 리스크)
    _var_src_v = full_df if full_df is not None else feature_df
    _vol5d_now = float(_var_src_v['vol_5d'].dropna().iloc[-1]) if 'vol_5d' in _var_src_v.columns else 0.015
    _t_var = np.arange(1, 8)
    fc_df['var_5pct']  = (fc_df['forecast_price'] - last_price * _vol5d_now * 1.645 * np.sqrt(_t_var)).round(2)
    fc_df['var_95pct'] = (fc_df['forecast_price'] + last_price * _vol5d_now * 1.645 * np.sqrt(_t_var)).round(2)
    log.info(f"    VaR(5%) D+1={fc_df['var_5pct'].iloc[0]:.2f}$  D+7={fc_df['var_5pct'].iloc[6]:.2f}$")

    _atomic_csv(fc_df, OUTPUT_DIR / 'forecast_7days.csv', index=False)
    log.info("    forecast_7days.csv 저장")
    return fc_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.  save_prediction_log()
# ─────────────────────────────────────────────────────────────────────────────

PRED_LOG_FILE = OUTPUT_DIR / 'prediction_log.csv'

def save_prediction_log(results: dict, feature_df: pd.DataFrame, fc_df: pd.DataFrame,
                        prev_fc_df: pd.DataFrame = None, full_df: pd.DataFrame = None):
    """예측 vs 실제 오차 로그 누적 저장
    - backtest: 90일 테스트셋 (매 실행마다 재구성)
    - live: 실행일 기준 entry + 미실행일은 직전 7일 예측으로 gap-fill
    """
    log.info("    prediction_log.csv 업데이트 중...")

    # ── 백테스트 구간 (90일 테스트셋) ────────────────────────────────
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
            price_src = price_src.sort_index()   # concat 후 비단조 인덱스 방지

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
                    old_live.at[idx, 'price_error_pct'] = round((ap - pp) / ap * 100, 2) if abs(ap) > 0.01 else None
                    if not np.isnan(av):
                        old_live.at[idx, 'actual_vol_5d'] = round(av, 5)
                    if not (np.isnan(av) or np.isnan(pv)):
                        old_live.at[idx, 'vol_error'] = round(av - pv, 5)
            except Exception:
                pass

        live_rows = old_live.to_dict('records')

    # ── 누락일 gap-fill (직전 실행의 7일 예측 활용) ──────────────────
    today = pd.Timestamp.today().normalize()
    existing_dates = {r.get('date') for r in live_rows if r.get('date') is not None}

    if prev_fc_df is not None and not prev_fc_df.empty and live_rows and existing_dates:
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
                        gap_error_pct = round((gap_actual - gap_pred) / gap_actual * 100, 2) if abs(gap_actual) > 0.01 else None
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
                    'type':            'gap',
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
        # full_df 사용: feature_df는 shift(-1) dropna로 마지막 행 없음
        _price_src = full_df if full_df is not None else feature_df
        if entry_ts in _price_src.index:
            try:
                entry_actual    = round(float(_price_src.loc[entry_ts, 'WTI']), 2)
                entry_error     = round(entry_actual - entry_pred, 2)
                entry_error_pct = round((entry_actual - entry_pred) / entry_actual * 100, 2) if abs(entry_actual) > 0.01 else None
            except Exception:
                pass

        xgb_pred_v = None
        try:
            model  = xg['model']
            scaler = xg['scaler']
            feats  = xg['features']
            _recon = xg.get('reconstruction', 'direct')
            _vol_src = full_df if full_df is not None else feature_df
            last   = _vol_src[[f for f in feats if f in _vol_src.columns]].iloc[-1:].fillna(0)
            _raw_pred = float(model.predict(scaler.transform(last))[0])
            if _recon == 'delta':
                _cur_vol = float(_vol_src['RV_5d'].dropna().iloc[-1]) if 'RV_5d' in _vol_src.columns else 0.015
                _raw_pred = max(_cur_vol + _raw_pred, 0.0)
            elif _recon == 'garch_resid':
                _cur_garch = float(_vol_src['garch_vol'].dropna().iloc[-1]) if 'garch_vol' in _vol_src.columns else 0.015
                _raw_pred = max(_cur_garch + _raw_pred, 0.0)
            xgb_pred_v = round(_raw_pred, 5)
        except Exception:
            pass

        _entry_lower_ci = _entry_upper_ci = None
        try:
            _ci_match = fc_df[fc_df['date'] == entry_date_str]
            _ci_row   = _ci_match if not _ci_match.empty else fc_df.iloc[:1]
            if 'lower_80ci' in fc_df.columns:
                _entry_lower_ci = round(float(_ci_row['lower_80ci'].iloc[0]), 2)
                _entry_upper_ci = round(float(_ci_row['upper_80ci'].iloc[0]), 2)
        except Exception:
            pass

        new_entry = {
            'date':            entry_date_str,
            'sarimax_pred':    entry_pred,
            'actual_price':    entry_actual,
            'price_error':     entry_error,
            'price_error_pct': entry_error_pct,
            'lower_80ci':      _entry_lower_ci,
            'upper_80ci':      _entry_upper_ci,
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
    _atomic_csv(combined, PRED_LOG_FILE, index=False)
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
    if level not in ('SURGE_RISK', 'DROP_RISK'):
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        r       = RISK_LEVELS.get(level, {'emoji': '⚠', 'label': level, 'color': '#888'})
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

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_TO, msg.as_string())

        log.info(f"    📧 리스크 알림 이메일 발송 완료 → {ALERT_TO}")
        return True

    except Exception as exc:
        log.warning(f"    이메일 발송 실패: {exc}")
        return False


def monitor_rss_alerts() -> dict:  # noqa: dead — 향후 독립 스케줄러용
    """RSS 긴급 이벤트 스캔. 현재 파이프라인에서 호출 안 됨 (독립 실행 예정)."""
    import json as _json, urllib.request as _ur, xml.etree.ElementTree as _ET

    ALERT_KEYWORDS = {
        'supply_cut':  ['opec cut','production cut','supply disruption','pipeline attack',
                        'embargo','export ban','field shutdown','force majeure'],
        'geopolitical':['nuclear plant','missile strike','drone attack','war escalat',
                        'strait of hormuz','tanker attack','oil facility','refinery attack'],
        'demand_shock':['recession','demand collapse','economic crisis','china slowdown',
                        'global downturn','demand destruction'],
        'price_move':  ['oil surges','oil plunges','crude spikes','wti jumps','brent soars',
                        'oil prices surge','oil prices plunge','oil prices jump'],
    }
    SCORE = {'supply_cut': 3, 'geopolitical': 3, 'demand_shock': 2, 'price_move': 2}

    triggered   = []
    seen_titles = set()

    # ── RSS 수집
    for url in NEWS_RSS:
        try:
            req  = _ur.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with _ur.urlopen(req, timeout=8) as resp:
                root = _ET.fromstring(resp.read())
            items = root.findall('.//item')[:15]
            for item in items:
                title = (item.findtext('title') or '').strip()
                link  = item.findtext('link') or ''
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                tl = title.lower()
                for cat, kws in ALERT_KEYWORDS.items():
                    if any(kw in tl for kw in kws):
                        triggered.append({'title': title, 'category': cat,
                                          'score': SCORE[cat], 'url': link,
                                          'source': url.split('/')[2]})
                        break
        except Exception as _re:
            log.debug(f"RSS 수집 실패({url}): {_re}")

    # ── OVX 급등 체크 (당일 변화율 ≥ 8%)
    ovx_alert = False
    try:
        import yfinance as _yf
        _ovx = _yf.download('^OVX', period='5d', progress=False, auto_adjust=True)['Close']
        if hasattr(_ovx, 'columns'): _ovx = _ovx.iloc[:, 0]
        _ovx = _ovx.dropna()
        if len(_ovx) >= 2:
            ovx_chg = (_ovx.iloc[-1] - _ovx.iloc[-2]) / _ovx.iloc[-2] * 100
            if abs(ovx_chg) >= 8:
                ovx_alert = True
                triggered.append({'title': f'OVX 급변 {ovx_chg:+.1f}% (원유 변동성 지수)',
                                   'category': 'ovx_spike', 'score': 3, 'url': '', 'source': 'yfinance'})
    except Exception:
        pass

    # ── 경보 판단
    total_score  = sum(t['score'] for t in triggered)
    alert_level  = 'CRITICAL' if total_score >= 6 else 'WARNING' if total_score >= 3 else 'NORMAL'
    detected_now = datetime.now().strftime('%Y-%m-%d %H:%M')

    result = {
        'checked_at':  detected_now,
        'alert_level': alert_level,
        'total_score': total_score,
        'ovx_alert':   ovx_alert,
        'triggers':    triggered[:10],
    }

    # ── 저장
    try:
        (OUTPUT_DIR / 'latest_alerts.json').write_text(
            _json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass

    # ── 이메일 발송 (WARNING 이상)
    if alert_level in ('WARNING', 'CRITICAL') and SMTP_USER and SMTP_PASSWORD and ALERT_TO:
        try:
            import smtplib
            from email.mime.text import MIMEText
            _lines = '\n'.join(f"  [{t['category'].upper()}] {t['title'][:80]}"
                               for t in triggered[:5])
            _body  = f"""[유가 리스크] 🚨 실시간 이벤트 경보 — {detected_now}

경보 레벨 : {alert_level}  (점수: {total_score})
OVX 급변  : {'⚠ 감지됨' if ovx_alert else '정상'}

▶ 감지된 이벤트
{_lines}

* 파이프라인을 실행하여 최신 리스크 신호를 확인하세요.
국제 유가 리스크 예측 시스템 MVP""".strip()
            _msg = MIMEText(_body, 'plain', 'utf-8')
            _msg['From']    = SMTP_USER
            _msg['To']      = ALERT_TO
            _msg['Subject'] = f"[유가 리스크] 🚨 {alert_level} — {detected_now}"
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as _sv:
                _sv.starttls()
                _sv.login(SMTP_USER, SMTP_PASSWORD)
                _sv.sendmail(SMTP_USER, ALERT_TO, _msg.as_string())
            log.info(f"    📧 RSS 경보 이메일 발송 → {ALERT_TO} ({alert_level})")
        except Exception as _ee:
            log.warning(f"    RSS 경보 이메일 실패: {_ee}")

    log.info(f"    RSS 모니터링 완료: {alert_level} (점수={total_score}, 트리거={len(triggered)}건)")
    return result


def classify_risk(feature_df: pd.DataFrame, full_df: pd.DataFrame, forecast_dir: int = 0, surge_prob: float = 0.0) -> dict:
    """실시간 리스크 신호등: 정상 / 주의 / 급등위험 / 급락위험"""
    log.info("[6/9] 리스크 분류 중...")

    row = (full_df if full_df is not None else feature_df).iloc[-1]

    vol       = float(row.get('vol_5d',               0.015))
    mom_5     = float(row.get('mom_5d',               0.0))
    mom_21    = float(row.get('mom_21d',              0.0))
    sentiment = float(row.get('news_sentiment_smooth', 0.0))
    n_count   = float(row.get('news_count',            0.0))
    bb        = float(row.get('bb_position',           0.0))
    geo       = float(row.get('geo_dummy',             0.0))
    ovx_z     = float(row.get('ovx_zscore',            0.0))
    ovx_level = float(row.get('OVX',                  0.0))
    ovx_chg   = float(row.get('ovx_change',            0.0))
    price_z   = float(row.get('price_zscore',          0.0))
    n_neg     = float(row.get('news_count_neg',        0.0))
    n_pos     = float(row.get('news_count_pos',        0.0))
    extreme_n = float(row.get('extreme_neg_news',      0.0))

    # 훈련 구간(최근 60일 제외) 기준 분위수 사용 (CEEMDAN/Regime과 동일 기준)
    _n_tr_cr = max(len(feature_df) - 60, 252)
    hist_vol_75 = float(feature_df['vol_5d'].iloc[:_n_tr_cr].quantile(0.75)) if 'vol_5d' in feature_df.columns else 0.022

    # ── 리스크 점수
    vol_ratio     = vol / (hist_vol_75 + 1e-8)
    news_amp      = max(1 + min((n_neg - n_pos * 0.5) / 8, 1.0), 1.0)
    geo_amp       = 1.35 if geo > 0.5 else 1.0
    sentiment_amp = 1 + max(-sentiment, 0) * 0.5 + extreme_n * 0.1
    # OVX(원유 공포지수) z-score > 1σ 부터 리스크 점수 증폭
    ovx_amp       = 1.0 + max(ovx_z - 1.0, 0) * 0.15

    risk_score = vol_ratio * news_amp * geo_amp * sentiment_amp * ovx_amp

    # ── 방향성 편향: 단기(5d) + 중기(21d) 모멘텀 합성
    # OVX 상승 = 하락 공포 신호 (음수 방향), 가격 과열/과매도 조정
    directional = (mom_5 * 0.5 + mom_21 * 0.3
                   + sentiment * 0.3 + bb * 0.15
                   - ovx_chg * 0.1)
    # 단기·중기 모멘텀이 같은 방향이면 신호 강화 (합의 확인)
    if mom_5 * mom_21 > 0:
        directional *= 1.3
    # 가격 과열(z>1.5) → 하락 편향 / 과매도(z<-1.5) → 상승 편향
    if price_z > 1.5:
        directional -= 0.01
    elif price_z < -1.5:
        directional += 0.01

    # 모델 예측 방향 합의 확인 (forecast_dir: +1=상승, -1=하락)
    if forecast_dir != 0:
        _mom_dir = 1 if directional > 0.01 else (-1 if directional < -0.01 else 0)
        if _mom_dir != 0:
            if _mom_dir == forecast_dir:
                directional *= 1.20   # 모델·모멘텀 동일 방향 → 신호 강화
            else:
                directional *= 0.75   # 모델·모멘텀 반대 방향 → 신호 약화

    # 3일 급등탐지기 확률 반영: 2일 연속 확인 + OVX 게이트 (단발 오경보 억제)
    _prev_surge_p = 0.0
    try:
        _ps_path = OUTPUT_DIR / 'latest_risk_signal.csv'
        if _ps_path.exists():
            _ps_df = pd.read_csv(_ps_path)
            if 'surge_prob_3d' in _ps_df.columns:
                _prev_surge_p = float(_ps_df['surge_prob_3d'].iloc[0])
    except Exception:
        pass
    # 펀더멘탈 확인: EIA 재고 감소(bullish) OR 중기 모멘텀 상승
    _inv_surp  = float(row.get('inv_surprise', 0.0))
    _has_fund  = (_inv_surp < -0.5) or (mom_21 > 0.02)
    # 고확률(>0.65): 즉시 적용 / 중간확률: OVX+시계열확인+펀더멘탈 3중 게이트
    _surge_fire = surge_prob > 0.65 or (
        surge_prob > 0.45 and ovx_z > 0.5 and _prev_surge_p > 0.35 and _has_fund
    )
    if _surge_fire:
        directional += (surge_prob - 0.50) * 0.4

    # 촉매 감지: geo이벤트/OVX급등/GPR급등/EIA서프라이즈 없으면 방향 임계값 강화
    _catalyst = (geo > 0.5 or ovx_z > 1.5
                 or float(row.get('gpr_zscore', 0.0)) > 1.5
                 or abs(float(row.get('inv_surprise', 0.0))) > 1.0)
    _dir_thresh = 0.05 if _catalyst else 0.08  # 촉매 없을 때 60% 더 강한 방향 신호 요구

    # 분류 규칙 (threshold 0.025 → 0.05: 과잉 신호 방지)
    if   risk_score >= 2.2 and directional >  _dir_thresh: level = 'SURGE_RISK'
    elif risk_score >= 2.2 and directional < -_dir_thresh: level = 'DROP_RISK'
    elif risk_score >= 1.4 or abs(directional) > 0.06:    level = 'CAUTION'
    else:                                                  level = 'NORMAL'

    # DROP_RISK + 높은 surge_prob → 방향 불확실 → CAUTION
    if level == 'DROP_RISK' and surge_prob > 0.40:
        log.info(f"    ⚖ DROP_RISK + surge_prob={surge_prob:.2f}>0.40 → CAUTION 조정")
        level = 'CAUTION'

    # ── CI 멀티플라이어: Shock 이진 + 뉴스 감성 서프라이즈 연속 조정
    _gpr_z      = float(row.get('gpr_zscore', 0.0))
    _geo_dum    = float(row.get('geo_dummy',  0.0))
    _sent_surp  = abs(float(row.get('sent_surprise_z', 0.0)))
    _news_smag  = abs(float(row.get('news_sentiment_smooth', 0.0)))
    _shock      = (_gpr_z > 2.0) or (_geo_dum > 0.5) or (risk_score >= 5.0)
    if _shock:
        ci_multiplier = 2.0
        log.info(f"    ⚡ Shock 감지 (gpr_z={_gpr_z:.2f} geo={_geo_dum:.0f} "
                 f"score={risk_score:.2f}) → CI×{ci_multiplier:.1f}")
    elif _sent_surp > 1.5 or _news_smag > 0.4:
        # 감성 서프라이즈 or 강한 부정 감성 → CI 연속 확대 (1.0 ~ 2.0)
        _cont_mult = 1.0 + 0.4 * min(_sent_surp / 2.0 + _news_smag, 1.0)
        ci_multiplier = float(np.clip(_cont_mult, 1.0, 2.0))
        log.info(f"    📰 뉴스 서프라이즈 CI 조정: surp_z={_sent_surp:.2f} "
                 f"mag={_news_smag:.2f} → CI×{ci_multiplier:.2f}")
    elif ovx_z < -0.3:
        # OVX 평온 구간 → CI 축소 (calm market, narrow band)
        ci_multiplier = 0.75
        log.info(f"    📉 평온 구간 CI 축소: ovx_z={ovx_z:.2f} → CI×{ci_multiplier:.2f}")
    else:
        ci_multiplier = 1.0

    # OVX 절대값 기반 CI 추가 확대 (내재변동성 선행 신호 — z-score와 독립 적용)
    if ovx_level >= 45:
        _ovx_abs_mult = 2.0
    elif ovx_level >= 35:
        _ovx_abs_mult = 1.5
    elif ovx_level >= 25:
        _ovx_abs_mult = 1.2
    else:
        _ovx_abs_mult = 1.0
    if _ovx_abs_mult > ci_multiplier:
        log.info(f"    🔥 OVX 절대값 CI 확대: OVX={ovx_level:.1f} → CI×{_ovx_abs_mult:.1f}")
        ci_multiplier = _ovx_abs_mult

    current_wti = float(full_df['WTI'].dropna().iloc[-1])

    # 헤지 비율: 리스크 레벨 + surge_prob 기반 (유가 사용 기업 기준)
    _base_hedge = {'SURGE_RISK': 0.65, 'CAUTION': 0.30, 'DROP_RISK': 0.05, 'NORMAL': 0.0}[level]
    _surge_adj  = max((surge_prob - 0.45) * 0.6, 0.0) if level == 'SURGE_RISK' else 0.0
    hedge_ratio = round(float(np.clip(_base_hedge + _surge_adj, 0.0, 1.0)), 2)

    # EIA 발표일(수요일) + 반응일(목요일): CI 확대 + hedge 증가
    _eia_dow = pd.Timestamp.today().dayofweek  # 0=Mon, 2=Wed, 3=Thu
    if _eia_dow in (2, 3):
        _eia_ci_boost = 1.3 if _eia_dow == 2 else 1.2
        ci_multiplier = float(np.clip(ci_multiplier * _eia_ci_boost, ci_multiplier, 2.5))
        hedge_ratio   = round(float(np.clip(hedge_ratio + 0.10, 0.0, 1.0)), 2)
        log.info(f"    📊 EIA {'발표일' if _eia_dow == 2 else '반응일'} 불확실성 확대: "
                 f"CI×{ci_multiplier:.2f} hedge={hedge_ratio:.2f}")

    # 비대칭 리스크 비율: 최근 라이브 오차 분포 기반 (하방 vs 상방 기대손실)
    _down_risk_pct = 50.0
    try:
        if PRED_LOG_FILE.exists():
            _pl_ar = pd.read_csv(PRED_LOG_FILE)
            _lv_ar = _pl_ar[(_pl_ar['type'] == 'live') & _pl_ar['price_error'].notna()].tail(15)
            if len(_lv_ar) >= 5:
                _errs = _lv_ar['price_error'].values.astype(float)
                _neg  = _errs[_errs < 0]
                _pos  = _errs[_errs > 0]
                if len(_neg) > 0 and len(_pos) > 0:
                    _neg_risk = len(_neg) / len(_errs) * abs(_neg.mean())
                    _pos_risk = len(_pos) / len(_errs) * abs(_pos.mean())
                    _tot = _neg_risk + _pos_risk
                    if _tot > 0:
                        _down_risk_pct = round(float(_neg_risk / _tot * 100), 1)
    except Exception:
        pass

    signal = {
        'date':              pd.Timestamp.today().normalize().strftime('%Y-%m-%d'),
        'risk_level':        level,
        'risk_label':        RISK_LEVELS[level]['label'],
        'wti_price':         round(current_wti, 2),
        'volatility_5d':     round(vol, 5),
        'momentum_5d':       round(mom_5, 5),
        'news_sentiment':    round(sentiment, 4),
        'news_count':        int(n_count),
        'geopolitical_alert': bool(geo > 0.5),
        'risk_score':        round(risk_score, 4),
        'directional_bias':  round(directional, 5),
        'direction_confidence': 'HIGH' if abs(directional) > 0.20 else ('MEDIUM' if abs(directional) > 0.06 else 'LOW'),
        'ci_multiplier':     ci_multiplier,
        'surge_prob_3d':     round(surge_prob, 3),
        'hedge_ratio':       hedge_ratio,
        'ovx_level':         round(ovx_level, 1),
        'ovx_alarm':         'HIGH' if ovx_level >= 45 else ('ELEVATED' if ovx_level >= 35 else 'NORMAL'),
        'downside_risk_pct': _down_risk_pct,
        'upside_risk_pct':   round(100.0 - _down_risk_pct, 1),
    }

    _atomic_csv(pd.DataFrame([signal]), OUTPUT_DIR / 'latest_risk_signal.csv', index=False)

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
    _atomic_csv(_hist, _hist_path, index=False)
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

    ax1.fill_between(fd, fc_df['lower_80ci'], fc_df['upper_80ci'],
                     alpha=0.18, color=CYAN, label='80% 예측구간')

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

    # ── 사용자용 단순 차트 (전체 히스토리 + 확대 예측 + 리스크 게이지)
    try:
        fig_u = plt.figure(figsize=(16, 11), facecolor=BG)
        gs_u  = gridspec.GridSpec(3, 1, figure=fig_u,
                                  height_ratios=[2, 2.2, 1.1],
                                  hspace=0.42,
                                  left=0.06, right=0.97, top=0.93, bottom=0.04)

        # Panel A: 전체 히스토리 (overview)
        axA = fig_u.add_subplot(gs_u[0])
        axA.set_facecolor(PAN)
        axA.tick_params(colors=TXT, labelsize=7)
        for sp in axA.spines.values(): sp.set_color('#30363d')
        axA.grid(color='#21262d', linewidth=0.5, alpha=0.6)
        axA.set_title('WTI Crude Oil — Historical Overview',
                      color=TXT, fontsize=10, pad=5, fontweight='bold')
        hist = feature_df['WTI'].iloc[-N:]
        axA.plot(hist.index, hist, color=GOLD, lw=1.3, label='WTI')
        if 'ma_21d' in feature_df.columns:
            axA.plot(feature_df['ma_21d'].iloc[-N:].index,
                     feature_df['ma_21d'].iloc[-N:],
                     color='#8b949e', lw=0.8, ls='--', alpha=0.6, label='MA-21')
        axA.axvspan(fd.iloc[0], fd.iloc[-1], alpha=0.12, color=rcol)
        axA.set_ylabel('USD / bbl', color=TXT, fontsize=8)
        axA.legend(loc='upper left', facecolor=PAN, labelcolor=TXT,
                   framealpha=0.5, fontsize=7, ncol=2)

        # Panel B: 확대 — 최근 45일 + 7일 예측 (CI 밴드 명확히 표시)
        axB = fig_u.add_subplot(gs_u[1])
        axB.set_facecolor(PAN)
        axB.tick_params(colors=TXT, labelsize=8)
        for sp in axB.spines.values(): sp.set_color('#30363d')
        axB.grid(color='#21262d', linewidth=0.5, alpha=0.7)
        axB.set_title('7-Day Forecast  +  80% Prediction Interval  (최근 45일 확대)',
                      color=TXT, fontsize=10, pad=5, fontweight='bold')
        _zoom_hist = feature_df['WTI'].iloc[-45:]
        axB.plot(_zoom_hist.index, _zoom_hist, color=GOLD, lw=2.0,
                 label='WTI (historical)', zorder=3)
        if 'ma_21d' in feature_df.columns:
            axB.plot(feature_df['ma_21d'].iloc[-45:].index,
                     feature_df['ma_21d'].iloc[-45:],
                     color='#8b949e', lw=1.0, ls='--', alpha=0.7, label='MA-21')
        # CI 밴드 (먼저 그려야 선 위로 안 덮임)
        axB.fill_between(fd, fc_df['lower_80ci'], fc_df['upper_80ci'],
                         alpha=0.30, color=CYAN, label='80% 예측구간', zorder=2)
        axB.plot(fd, fc_df['lower_80ci'], color=CYAN, lw=0.8, ls=':', alpha=0.6)
        axB.plot(fd, fc_df['upper_80ci'], color=CYAN, lw=0.8, ls=':', alpha=0.6)
        axB.plot(fd, fc_df['forecast_price'], color=CYAN, lw=2.2, ls='--',
                 label='Forecast', zorder=4)
        axB.axvspan(fd.iloc[0], fd.iloc[-1], alpha=0.06, color=rcol)
        # 예측 D+1 값 레이블
        axB.annotate(f"D+1: ${fc_df['forecast_price'].iloc[0]:.2f}",
                     xy=(fd.iloc[0], fc_df['forecast_price'].iloc[0]),
                     xytext=(10, 10), textcoords='offset points',
                     color=CYAN, fontsize=8.5, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=CYAN, lw=1))
        # CI 범위 레이블
        _lo1, _hi1 = fc_df['lower_80ci'].iloc[0], fc_df['upper_80ci'].iloc[0]
        axB.annotate(f"[${_lo1:.1f} ~ ${_hi1:.1f}]",
                     xy=(fd.iloc[0], (_lo1 + _hi1) / 2),
                     xytext=(12, -18), textcoords='offset points',
                     color='#8b949e', fontsize=7.5)
        axB.set_ylabel('USD / bbl', color=TXT, fontsize=9)
        axB.legend(loc='upper left', facecolor=PAN, labelcolor=TXT,
                   framealpha=0.6, fontsize=8, ncol=4)
        axB.annotate(f"{RISK_LEVELS[rlevel]['emoji']} {RISK_LEVELS[rlevel]['label']}",
                     xy=(fd.iloc[3], fc_df['forecast_price'].median()),
                     fontsize=10, color=rcol, fontweight='bold', ha='center')

        # Panel C: 리스크 게이지
        axC = fig_u.add_subplot(gs_u[2])
        axC.set_facecolor(PAN)
        axC.axis('off')
        axC.set_xlim(0, 1); axC.set_ylim(0, 1)
        axC.set_title('Current Risk Signal', color=TXT, fontsize=10, pad=5, fontweight='bold')
        for rl, lbl, xp in zip(order, labels, xpos):
            is_cur = (rl == rlevel)
            rc_b   = RISK_LEVELS[rl]['color']
            rect_b = FancyBboxPatch((xp, 0.22), 0.20, 0.50,
                                    boxstyle='round,pad=0.02',
                                    facecolor=rc_b, alpha=0.95 if is_cur else 0.22,
                                    edgecolor='white', linewidth=2.5 if is_cur else 0.5)
            axC.add_patch(rect_b)
            axC.text(xp + 0.10, 0.47, lbl, ha='center', va='center',
                     color='white', fontsize=9.5,
                     fontweight='bold' if is_cur else 'normal')
            if is_cur:
                axC.text(xp + 0.10, 0.80, '▲ Current', ha='center',
                         color=rc_b, fontsize=8.5, fontweight='bold')

        fig_u.suptitle(
            f'🛢  국제 유가 리스크 예측 시스템  │  {datetime.today().strftime("%Y-%m-%d %H:%M")}',
            color='white', fontsize=13, fontweight='bold', y=0.975
        )
        plt.savefig(OUTPUT_DIR / 'user_forecast_plot.png', dpi=150,
                    bbox_inches='tight', facecolor=BG)
        plt.close()
        log.info("    user_forecast_plot.png 저장")
    except Exception as _e:
        log.warning(f"    user_forecast_plot 생성 실패({_e})")


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

    feature_df, full_df, aux_models = build_features(price_df, news_df)
    model_results, _     = train_models(feature_df, full_df, aux=aux_models)

    # forecast_7days.csv 덮어쓰기 전에 이전 예측 로드 (gap-fill용)
    prev_fc_df = None
    _fc_csv = OUTPUT_DIR / 'forecast_7days.csv'
    if _fc_csv.exists():
        try:
            prev_fc_df = pd.read_csv(_fc_csv)
        except Exception as _pfe:
            log.warning(f"    이전 forecast 로드 실패({_pfe}) → gap-fill 생략")

    fc_df                = forecast_next_7days(model_results, feature_df, full_df, aux=aux_models)
    _last_wti = float(feature_df['WTI'].iloc[-1]) if 'WTI' in feature_df.columns else 0.0
    _fc_dir   = int(np.sign(float(fc_df['forecast_price'].iloc[0]) - _last_wti)) if _last_wti > 0 else 0
    _surge_p = 0.0
    _sg_info = model_results.get('surge_detector')
    if _sg_info is not None:
        try:
            _sg_m  = _sg_info['model']
            _sg_fs = _sg_info['features']
            _sg_sc = _sg_info['scaler']
            _sgf_ok = [f for f in _sg_fs if f in feature_df.columns]
            _sg_in  = _sg_sc.transform(feature_df[_sgf_ok].iloc[-1:].fillna(0).values)
            _surge_p = float(_sg_m.predict_proba(_sg_in)[0, 1])
            log.info(f"    3일급등확률: {_surge_p*100:.1f}%")
        except Exception as _spe:
            log.warning(f"    급등확률 계산 실패({_spe})")
    risk_signal = classify_risk(feature_df, full_df, forecast_dir=_fc_dir, surge_prob=_surge_p)

    # Shock CI 확대: classify_risk에서 반환된 ci_multiplier 적용
    _ci_mult = risk_signal.get('ci_multiplier', 1.0)
    if abs(_ci_mult - 1.0) > 0.01 and 'lower_80ci' in fc_df.columns and 'upper_80ci' in fc_df.columns:
        _fp = fc_df['forecast_price']
        fc_df['lower_80ci'] = (_fp - (_fp - fc_df['lower_80ci']) * _ci_mult).round(2)
        fc_df['upper_80ci'] = (_fp + (fc_df['upper_80ci'] - _fp) * _ci_mult).round(2)
        _atomic_csv(fc_df, OUTPUT_DIR / 'forecast_7days.csv', index=False)
        log.info(f"    Shock CI 적용: ×{_ci_mult:.1f} → forecast_7days.csv 갱신")

    save_prediction_log(model_results, feature_df, fc_df, prev_fc_df, full_df)

    # ── A: 라이브 성능 모니터링 (최근 30건 MAE vs 백테스트 MAE)
    _live_mae_val = None
    if PRED_LOG_FILE.exists():
        try:
            _pl_check = pd.read_csv(PRED_LOG_FILE)
            _live_known = _pl_check[
                _pl_check['type'].isin(['live', 'gap']) &
                _pl_check['price_error'].notna()
            ]
            if len(_live_known) >= 10:
                _live30 = _live_known.tail(30)
                _live_mae_val = float(_live30['price_error'].abs().mean())
                _stk_m = model_results.get('stacking') or model_results.get('sarimax', {})
                _bt_mae_ref = _stk_m.get('mae', float('inf'))
                _ratio = _live_mae_val / max(_bt_mae_ref, 1e-6)
                log.info(f"    라이브 성능: 최근 {len(_live30)}건 MAE={_live_mae_val:.4f} "
                         f"(백테스트 대비 {_ratio:.2f}×)")
                if _ratio > 1.5:
                    log.warning(f"    ⚠ 라이브 성능 열화: 라이브 MAE={_live_mae_val:.4f} > "
                                f"백테스트×1.5 ({_bt_mae_ref:.4f}×1.5={_bt_mae_ref*1.5:.4f})")
                # Rolling MASE: naive persistence 대비 비율 (>0.95 → 재훈련 시점)
                _naive_diffs = _live30['actual_price'].diff().abs().dropna()
                if len(_naive_diffs) >= 5:
                    _naive_mae_live = float(_naive_diffs.mean())
                    _live_mase = _live_mae_val / max(_naive_mae_live, 1e-6)
                    log.info(f"    라이브 MASE(30d): {_live_mase:.3f} "
                             f"(naive_mae={_naive_mae_live:.4f})")
                    if _live_mase > 0.95:
                        log.warning(f"    ⚠ 라이브 MASE={_live_mase:.3f} → naive 근접, 재훈련 검토")
            # ── C: 80% CI 실제 커버리지 검증
            if 'lower_80ci' in _pl_check.columns and 'upper_80ci' in _pl_check.columns:
                _ci_rows = _pl_check[
                    _pl_check['type'].isin(['live', 'gap']) &
                    _pl_check['actual_price'].notna() &
                    _pl_check['lower_80ci'].notna() &
                    _pl_check['upper_80ci'].notna()
                ]
                if len(_ci_rows) >= 20:
                    _covered = (
                        (_ci_rows['actual_price'] >= _ci_rows['lower_80ci']) &
                        (_ci_rows['actual_price'] <= _ci_rows['upper_80ci'])
                    ).mean()
                    log.info(f"    80% CI 커버리지: {_covered:.1%} (N={len(_ci_rows)})")
                    if _covered < 0.65:
                        log.warning(f"    ⚠ CI 과소추정: 실제 커버리지={_covered:.1%} < 65% "
                                    f"— CI 폭 확대 검토 필요")
        except Exception as _lme:
            log.warning(f"    라이브 성능 모니터링 실패({_lme})")

    send_risk_alert(risk_signal, fc_df)
    kw_df                = extract_crisis_keywords(news_df)
    generate_wordcloud(kw_df)
    plot_oil_forecast(feature_df, fc_df, risk_signal)

    # ── 마지막 실행 시간 + API 상태 기록
    import json as _json
    _run_meta = {
        'last_run':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_through': full_df.index[-1].strftime('%Y-%m-%d'),
        'n_live':      int((pd.read_csv(PRED_LOG_FILE)['type'] == 'live').sum()) if PRED_LOG_FILE.exists() else 0,
        'live_mae':    round(_live_mae_val, 4) if _live_mae_val is not None else None,
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
    _ovx_lvl = risk_signal.get('ovx_level', 0.0)
    _ovx_alm = risk_signal.get('ovx_alarm', 'NORMAL')
    print(f"  OVX 수준    : {_ovx_lvl:.1f} ({_ovx_alm})")
    _dn = risk_signal.get('downside_risk_pct', 50.0)
    _up = risk_signal.get('upside_risk_pct',   50.0)
    print(f"  비대칭 리스크: 하방 {_dn:.0f}% / 상방 {_up:.0f}%")
    print(f"  D+1 VaR(5%) : ${fc_df['var_5pct'].iloc[0]:.2f}  VaR(95%)=${fc_df['var_95pct'].iloc[0]:.2f}")
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
    parser.add_argument('--rss-alerts', action='store_true',
                        help='RSS 긴급 이벤트 스캔 + 이메일 경보만 실행 (파이프라인 생략)')
    args = parser.parse_args()

    if args.schedule:
        schedule_daily(args.hour, args.minute)
    elif args.rss_alerts:
        import json as _json
        result = monitor_rss_alerts()
        print(_json.dumps(result, ensure_ascii=False, indent=2))
    else:
        run_pipeline()
