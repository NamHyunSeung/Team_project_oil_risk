#!/usr/bin/env python3
"""
국제 유가 리스크 예측 시스템 MVP
International Oil Price Risk Prediction System

Models  : XGBoost-HAR (volatility) + SARIMAX (price forecast)
Features: WTI/Brent, DXY, demand/supply shock, news sentiment, news count, geo-dummy
Output  : model_performance.csv, forecast_7days.csv, latest_risk_signal.csv,
          crisis_keywords.csv, oil_forecast_plot.png, wordcloud.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  파일 구조 (Ctrl+F로 섹션 이동)
  File Structure — search section tags to jump:

  §CFG  CONFIG & CONSTANTS        api keys, paths, thresholds, OPEC dates
  §RSS  NEWS RSS SOURCES          NEWS_RSS list, OIL_FILTER — 소스 추가/삭제
  §SEN  SENTIMENT DICTIONARIES    SENTIMENT_MAP, PHRASE_SENTIMENT, INTENSIFIERS
  §DAT  DATA COLLECTION           fetch_data(), fetch_cot(), _attach_*
  §NWS  NEWS & NLP                fetch_news(), finbert, embeddings
  §FEA  FEATURE ENGINEERING       build_features()
  §MDL  MODEL TRAINING            train_models(), SARIMAX/XGB/VAR/ETS
  §ENS  ENSEMBLE & FORECAST       forecast_next_7days(), compute_ensemble_weights()
  §LOG  PREDICTION LOGGING        save_prediction_log()
  §RSK  RISK CLASSIFICATION       classify_risk()
  §ALT  ALERTS & MONITORING       send_risk_alert(), monitor_rss_alerts()
  §VIZ  VISUALIZATION             plot_oil_forecast(), generate_wordcloud()
  §PIP  PIPELINE ORCHESTRATION    run_pipeline(), schedule_daily()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
# Windows 콘솔 UTF-8 설정 (emoji/한글 출력 안전화)
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
import os, re, logging, json, yaml
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
        kwargs.setdefault('encoding', 'utf-8')
        df.to_csv(_tmp, **kwargs)
        _os.replace(_tmp, path)
    except Exception:
        try:
            _os.unlink(_tmp)
        except Exception:
            pass
        raise

def _atomic_json(obj, path, **kwargs) -> None:
    """Write JSON atomically via temp file + os.replace to avoid partial reads."""
    import os as _os, tempfile as _tf, json as _json
    _dir = Path(path).parent
    with _tf.NamedTemporaryFile(mode='w', dir=_dir, suffix='.tmp', delete=False,
                                encoding='utf-8') as _fh:
        _tmp = _fh.name
    try:
        with open(_tmp, 'w', encoding='utf-8') as _f:
            _json.dump(obj, _f, **kwargs)
        _os.replace(_tmp, path)
    except Exception:
        try:
            _os.unlink(_tmp)
        except Exception:
            pass
        raise

def _atomic_pkl(obj, path, **kwargs) -> None:
    """Write pickle atomically via temp file + os.replace to avoid partial reads."""
    import os as _os, tempfile as _tf, pickle as _pickle
    _dir = Path(path).parent
    with _tf.NamedTemporaryFile(mode='wb', dir=_dir, suffix='.tmp', delete=False) as _fh:
        _tmp = _fh.name
    try:
        with open(_tmp, 'wb') as _f:
            _pickle.dump(obj, _f, **kwargs)
        _os.replace(_tmp, path)
    except Exception:
        try:
            _os.unlink(_tmp)
        except Exception:
            pass
        raise

EMBED_CACHE_FILE = OUTPUT_DIR / 'news_embed_cache.pkl'
COT_CACHE_FILE   = OUTPUT_DIR / 'cot_cache.csv'
COT_RAW_FILE     = Path('data/annual.txt')   # CFTC 연간 전체 데이터
XGB_OPTUNA_CACHE    = OUTPUT_DIR / 'xgb_optuna_cache.json'
STACK_WEIGHTS_EMA   = OUTPUT_DIR / 'stacking_weights_ema.json'
FEAT_TRAIN_STATS    = OUTPUT_DIR / 'feature_train_stats.json'
LAST_RETRAIN_FILE   = OUTPUT_DIR / 'last_retrain.json'
SVM_CACHE           = OUTPUT_DIR / 'svm_cache.json'
SVM_MODEL_CACHE     = OUTPUT_DIR / 'svm_model_cache.pkl'
GARCH_CACHE         = OUTPUT_DIR / 'garch_cache.pkl'
INTRADAY_CACHE      = OUTPUT_DIR / 'intraday_rv_cache.pkl'
DEGRADATION_FILE    = OUTPUT_DIR / 'degradation_tracker.json'
REFIT_STALE_DAYS    = 7   # 이 일수 초과 시 Optuna 캐시 초기화 → 하이퍼파라미터 재탐색
MASE_RETRAIN_THRESH = 1.5  # MASE 이 값 초과 시 열화 카운트
MASE_RETRAIN_DAYS   = 3    # 연속 N일 열화 → 강제 재훈련 트리거
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_TOP_K      = 15   # WTI 수익률 상관관계 상위 차원 수

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def _get_model_version() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return datetime.now().strftime('%Y%m%d')

MODEL_VERSION = _get_model_version()
log = logging.getLogger(__name__)

def _check_refit_staleness() -> int:
    """마지막 재훈련 경과일 반환. REFIT_STALE_DAYS 초과 시 Optuna 캐시 삭제."""
    import json as _jr, time as _tr
    age_days = 0
    if LAST_RETRAIN_FILE.exists():
        try:
            _ts = _jr.loads(LAST_RETRAIN_FILE.read_text()).get('timestamp', 0)
            age_days = int((_tr.time() - _ts) / 86400)
        except Exception:
            pass
    if age_days >= REFIT_STALE_DAYS and XGB_OPTUNA_CACHE.exists():
        try:
            XGB_OPTUNA_CACHE.unlink()
            log.info(f"[Refit] Optuna 캐시 초기화 (마지막 재훈련 {age_days}일 전) → 하이퍼파라미터 재탐색")
        except Exception:
            pass
    return age_days

def _save_retrain_record() -> None:
    """재훈련 완료 시각 + 모델 버전 기록."""
    import json as _jw, time as _tw
    try:
        _atomic_json({
            'timestamp': _tw.time(),
            'model_version': MODEL_VERSION,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }, LAST_RETRAIN_FILE)
    except Exception:
        pass

def _check_degradation_trigger(mase: float, ovx_level: float = 0.0) -> bool:
    """MASE 열화 일수 추적. 연속 MASE_RETRAIN_DAYS 초과 시 True 반환 + 캐시 초기화.
    OVX≥60 고변동기는 임계값 완화 (시장 구조적 예측 난이도 상승 반영).
    """
    import json as _jd
    try:
        rec = _jd.loads(DEGRADATION_FILE.read_text()) if DEGRADATION_FILE.exists() else {}
    except Exception:
        rec = {}

    # M5: 고변동기 임계값 완화 — OVX≥60 시 재훈련 루프 방지
    _thresh = 2.0 if ovx_level >= 60 else (1.7 if ovx_level >= 40 else MASE_RETRAIN_THRESH)
    today = datetime.now().strftime('%Y-%m-%d')
    if mase > _thresh:
        rec['consecutive_days'] = rec.get('consecutive_days', 0) + 1
        rec['last_bad_date'] = today
    else:
        rec['consecutive_days'] = 0

    try:
        _atomic_json(rec, DEGRADATION_FILE)
    except Exception:
        pass

    if rec.get('consecutive_days', 0) >= MASE_RETRAIN_DAYS:
        # 강제 재훈련: 모든 모델 캐시 초기화
        for _cache in [XGB_OPTUNA_CACHE, SVM_CACHE, SVM_MODEL_CACHE, GARCH_CACHE]:
            try:
                if Path(_cache).exists():
                    Path(_cache).unlink()
            except Exception:
                pass
        rec['consecutive_days'] = 0
        try:
            _atomic_json(rec, DEGRADATION_FILE)
        except Exception:
            pass
        log.warning(f"⚠ 강제 재훈련 트리거: MASE>{_thresh:.1f} {MASE_RETRAIN_DAYS}일 연속 "
                    f"(ovx={ovx_level:.0f}) → Optuna/SVM/GARCH 캐시 초기화 완료")
        return True
    return False

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

# ─────────────────────────────────────────────────────────────────────────────
# §CFG  CONFIG & CONSTANTS  —  API keys, paths, thresholds, OPEC dates
# ─────────────────────────────────────────────────────────────────────────────
# ── API 키 & 파일 경로 설정 (.env 파일 또는 환경변수에서 로드)
FRED_API_KEY     = os.getenv("FRED_API_KEY",     "0a1d6c8b56c44eff8716c204f0aa49bf")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "3a287cda-6e49-49f0-8998-3092657e209e")
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY",      "daa97319786e4b4f8d96f99cec8f682c")
EIA_API_KEY      = os.getenv("EIA_API_KEY",      "")
GPR_FILE         = "data/data_gpr_daily_recent.xls"   # 프로젝트 폴더에 위치
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

# ─────────────────────────────────────────────────────────────────────────────
# §SEN  SENTIMENT DICTIONARIES  —  SENTIMENT_MAP, PHRASE_SENTIMENT, INTENSIFIERS
#        → 감성어 추가/수정 시 이 섹션만 편집
# ─────────────────────────────────────────────────────────────────────────────
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
    'tariff', 'tariffs', 'trade', 'tradewar', 'reciprocal', 'retaliation',
    'retaliatory', 'protectionism', 'import', 'levy', 'duties',
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

# 소스별 신뢰도 가중치 (EIA 공식 > 에너지 전문 > 일반; 일반 소스 희석 방지)
SOURCE_WEIGHTS = {
    'EIA':        4.0,
    'OilPrice':   3.0,
    'Rigzone':    2.5,
    'Reuters':    1.5,
    'MarketWatch':0.6,
    'Guardian':   0.5,
    'RSS':        0.7,
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
    'tariff':1.5,'tariffs':1.5,'tradewar':2.0,'trade_war':2.0,
    'reciprocal':1.5,'retaliation':1.5,'retaliatory':1.5,
    'protectionism':1.0,'levy':1.0,'duties':1.0,
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

# ─────────────────────────────────────────────────────────────────────────────
# §RSS  NEWS RSS SOURCES  —  소스 추가: 리스트에 URL 추가, 제거: 해당 줄 삭제
#        → 도메인→소스명 매핑은 §NWS 내 _RSS_DOMAIN_MAP 딕셔너리에 함께 추가
# ─────────────────────────────────────────────────────────────────────────────
NEWS_RSS = [
    # 종합 경제/비즈니스
    "https://www.marketwatch.com/rss/topstories",
    # 에너지 전문
    "https://www.oilprice.com/rss/main",
    "https://www.eia.gov/rss/todayinenergy.xml",                      # EIA Today in Energy (URL 변경)
    "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    # 추가 에너지/원유 소스
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",           # CNBC Energy
    "https://www.hellenicshippingnews.com/feed/",                     # Hellenic Shipping (탱커/원유)
    "https://www.offshore-technology.com/feed/",                      # Offshore Technology
    "https://www.naturalgasintel.com/feed/",                          # Natural Gas Intel
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
# §DAT  DATA COLLECTION  —  fetch_data(), _attach_gpr(), _attach_fred_data()
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
        gpr_z = (gpr_raw - mu) / sigma if sigma > 0 else (gpr_raw - mu)

        # df 인덱스에 맞춰 정렬 (없는 날은 앞날 값으로 채움)
        gpr_aligned = gpr_z.reindex(df.index).ffill().bfill().fillna(0)

        df['gpr_zscore'] = gpr_aligned
        df['geo_dummy']  = (gpr_aligned > 0.5).astype(float)  # 상위 ~30% 이상 = 지정학 위기 (완화된 임계값)

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

        # CL2=F Yahoo Finance 미지원 → 63일 이평 proxy (term structure 근사)
        futures_spread = (wti.rolling(63, min_periods=20).mean() - wti).rename("futures_spread")

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
        cutoff    = today - pd.Timedelta(days=90)
        check_idx = df.index.intersection(fred_data.index)
        check_idx = check_idx[check_idx < cutoff]

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
# §NWS  NEWS & NLP  —  fetch_news(), Guardian/NewsAPI, finbert, embeddings
#        → RSS 소스 추가/삭제: §RSS  (Ctrl+F "§RSS")
#        → 감성 사전 수정:    §SEN  (Ctrl+F "§SEN")
# ─────────────────────────────────────────────────────────────────────────────

GUARDIAN_QUERY = (
    'oil OR "crude oil" OR "brent crude" OR WTI OR OPEC OR "OPEC+" OR petroleum '
    'OR "energy crisis" OR "oil price" OR "oil supply" OR "oil demand" '
    'OR "natural gas" OR LNG OR "oil production" OR "oil sanction" '
    'OR "crude inventory" OR "oil inventory" OR "oil sanctions"'
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


def _newsapi_fetch(api_key: str, from_dt: str, to_dt: str) -> list:
    """NewsAPI.org에서 원유 뉴스 수집 (무료 플랜: 최근 30일, 100req/day)"""
    import urllib.request, json as _json, urllib.parse
    NEWSAPI_QUERY = (
        'oil OR "crude oil" OR WTI OR OPEC OR "OPEC+" OR petroleum '
        'OR "brent crude" OR "oil price" OR "oil supply" OR "crude inventory"'
    )
    articles = []
    page = 1
    while page <= 3:  # 최대 3페이지(300건)
        params = urllib.parse.urlencode({
            'apiKey':   api_key,
            'q':        NEWSAPI_QUERY,
            'from':     from_dt,
            'to':       to_dt,
            'language': 'en',
            'sortBy':   'publishedAt',
            'pageSize': 100,
            'page':     page,
        })
        url = f"https://newsapi.org/v2/everything?{params}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = _json.loads(r.read())
            if data.get('status') != 'ok':
                log.warning(f"NewsAPI 응답 오류: {data.get('message','')}")
                break
            results = data.get('articles', [])
            log.info(f"    NewsAPI p{page}: {len(results)}건, total={data.get('totalResults',0)}")
            for item in results:
                headline = item.get('title', '') or ''
                desc     = (item.get('description') or '')[:200]
                full_text = headline + (' ' + desc if desc and desc != headline else '')
                pub_date  = (item.get('publishedAt') or '')[:10]
                src_name  = (item.get('source') or {}).get('name', 'NewsAPI')
                if pub_date:
                    articles.append({'date': pub_date, 'title': full_text, 'source': src_name})
            total = data.get('totalResults', 0)
            if len(results) < 100 or page * 100 >= min(total, 300):
                break
            page += 1
        except Exception as exc:
            log.warning(f"NewsAPI 수집 실패: {exc}")
            break
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
                    _atomic_csv(cache.reset_index().rename(columns={'index': 'date'}),
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
                            _atomic_csv(cache.reset_index().rename(columns={'index': 'date'}),
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
            if last_cached > end_dt:
                log.info("    캐시 최신 → API 호출 생략")
                return cache_df.sort_values('date').reset_index(drop=True)
            start_dt = last_cached   # 당일 포함 재수집 (당일 기사 추가분 반영)
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
            _OIL_KW = {'oil','crude','brent','wti','opec','barrel','petroleum','refinery','shale','tanker','sanctions','inventory'}
            new_articles = [
                a for a in new_articles
                if any(kw in a.get('title','').lower() for kw in _OIL_KW)
            ]
            log.info(f"    Guardian 신규 수집: {len(new_articles)}건 (oil 필터 후)")
        except Exception as exc:
            log.warning(f"    Guardian API 오류({exc}) → RSS 폴백")

    # ── RSS 수집 (항상 실행 — 오늘 날짜 기사 최대 수집) ──────────────────────
    if _FEED:
        cutoff = datetime.today() - timedelta(days=60)
        OIL_FILTER = {'oil','crude','brent','wti','opec','energy','barrel','petroleum','pipeline','sanctions','inventory','lng','refin','tanker','shale','upstream','downstream'}
        _RSS_DOMAIN_MAP = {
            'eia.gov':               'EIA',
            'oilprice.com':          'OilPrice',
            'rigzone.com':           'Rigzone',
            'marketwatch.com':       'MarketWatch',
            'cnbc.com':              'CNBC',
            'hellenicshippingnews':  'HellenicShipping',
            'offshore-technology':   'OffshoreTech',
            'naturalgasintel':       'NGIntel',
        }
        for url in NEWS_RSS:
            src_name = next((v for k, v in _RSS_DOMAIN_MAP.items() if k in url), 'RSS')
            try:
                feed = _feedparser_mod.parse(url)
                if len(feed.entries) == 0:
                    import time as _t; _t.sleep(3)
                    feed = _feedparser_mod.parse(url)
                _rss_before = len(new_articles)
                for entry in feed.entries[:30]:
                    try:
                        pub = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else datetime.today()
                    except Exception:
                        pub = datetime.today()
                    title = entry.get('title', '')
                    _body = (getattr(entry, 'summary', '') or
                             getattr(entry, 'description', '') or '')
                    _body = str(_body)[:300].strip()
                    full_text = title + (' ' + _body if _body and _body != title else '')
                    if pub >= cutoff and any(kw in full_text.lower() for kw in OIL_FILTER):
                        new_articles.append({'date': pub.strftime('%Y-%m-%d'),
                                             'title': full_text, 'source': src_name})
                _rss_added = len(new_articles) - _rss_before
                if _rss_added > 0:
                    log.info(f"      RSS {src_name}: +{_rss_added}건")
                else:
                    log.warning(f"      RSS {src_name}: 0건 (항목={len(feed.entries)})")
            except Exception as _rss_exc:
                log.warning(f"      RSS {src_name}: 실패({type(_rss_exc).__name__})")

    # ── NewsAPI 보충 (항상 최근 7일치 — 중복은 drop_duplicates 처리) ──────────
    if NEWSAPI_KEY:
        try:
            _na_from = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            _na_arts_raw = _newsapi_fetch(NEWSAPI_KEY, _na_from, end_dt)
            log.info(f"    NewsAPI raw 수집: {len(_na_arts_raw)}건 (필터 전)")
            _OIL_KW_N = {'oil','crude','brent','wti','opec','barrel','petroleum',
                         'refinery','shale','tanker','sanctions','inventory'}
            _na_arts = [a for a in _na_arts_raw
                        if any(kw in a.get('title','').lower() for kw in _OIL_KW_N)]
            new_articles.extend(_na_arts)
            log.info(f"    NewsAPI 수집: {len(_na_arts)}건 (oil 필터 후)")
        except Exception as exc:
            log.warning(f"    NewsAPI 오류({exc}) → 생략")

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

    # 2년치만 유지 (오래된 캐시 정리)
    _cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=2)
    _before_trim = len(combined)
    combined = combined[combined['date'] >= _cutoff].reset_index(drop=True)
    if _before_trim > len(combined):
        log.info(f"    캐시 트리밍: {_before_trim}건 → {len(combined)}건 ({_cutoff.date()} 이후만 유지)")

    try:
        _atomic_csv(combined, NEWS_CACHE_FILE, index=False)
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

        # ── 캐시 업데이트 (기존 행 갱신 + 신규 행 추가)
        try:
            _c = pd.read_csv(NEWS_CACHE_FILE) if NEWS_CACHE_FILE.exists() else pd.DataFrame(columns=['date','title','source','finbert_score'])
            if 'finbert_score' not in _c.columns:
                _c['finbert_score'] = np.nan
            _new_rows = news_df.loc[mask, ['date','title','source','finbert_score']].copy()
            # 기존 행 갱신
            _upd = dict(zip(_new_rows['title'].astype(str), _new_rows['finbert_score']))
            _idx = _c['title'].astype(str).isin(_upd)
            _c.loc[_idx, 'finbert_score'] = _c.loc[_idx, 'title'].astype(str).map(_upd)
            # 캐시에 없는 신규 행 추가
            _existing_titles = set(_c['title'].astype(str))
            _append = _new_rows[~_new_rows['title'].astype(str).isin(_existing_titles)]
            if len(_append) > 0:
                _c = pd.concat([_c, _append], ignore_index=True)
            _atomic_csv(_c, NEWS_CACHE_FILE, index=False)
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
_OPEC_EVENTS = {
    "OPEC production cut two million barrels per day deep",
    "OPEC production cut agreement extended deeper barrels per day",
    "OPEC surprise voluntary output cut announced",
    "Saudi Arabia announces unilateral production cut one million",
    "Saudi Arabia voluntary cut extended deeper barrel reduction",
    "OPEC+ ministerial meeting agrees production cut quota",
    "OPEC+ compliance exceeds quota output below target",
    "OPEC agrees increase output production quota barrels",
    "OPEC+ eases cuts production increase members",
    "Saudi Arabia increases output production boost barrels",
    "weak oil demand forecast IEA OPEC lowers outlook",
}

def _event_category(text):
    if text in _GEO_EVENTS:  return 'geo'
    if text in _OPEC_EVENTS: return 'opec'
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
        _EMBED_TOK = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, local_files_only=True)
        _EMBED_MDL = AutoModel.from_pretrained(EMBED_MODEL_NAME, local_files_only=True)
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
            _atomic_pkl(cache, EMBED_CACHE_FILE, protocol=4)
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
    'garch_vol',
    # ewma_vol_10/21/63 제거 — 서로 VIF 수백, garch_vol로 대체됨
    # 변동성 모멘텀 (rv_mom_5_21 = RV_5d - RV_21d 선형결합 — 완전공선성 제거)
    'rv_term_slope', 'rv_5d_chg',
    # 레버리지 효과
    'leverage_effect', 'neg_return',
    # EIA 발표일 효과
    'dow_wednesday', 'dow_thursday', 'eia_vol_signal',
    # 장중 RV (정확한 측정; _5d/_21d/_vs_close는 rv_intraday 고VIF 파생 — 제거)
    'rv_intraday',
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
    'news_sentiment_lag1',
    'news_count', 'news_count_neg',
    'oil_event_score', 'oil_event_score_smooth',
    'sentiment_magnitude', 'extreme_neg_news',
    'jump_flag', 'jump_magnitude', 'compound_shock_flag',
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
    'opec_event_score', 'opec_event_score_smooth',
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
    'news_sentiment_lag1',
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
    # return_neg 제거 (return_1d = return_pos + return_neg 선형결합)
    'RV_63d', 'neg_return', 'return_pos',
    'leverage_effect', 'dow_wednesday', 'dow_thursday', 'dow_monday',
    'eia_vol_signal',
    # A: GARCH 조건부 분산
    'garch_vol',
    # B: Parkinson 장중 범위 추정 (rv_intraday와 |r|=1 완전공선성 — 제거)
    # 'parkinson_vol',  # rv_intraday와 동일, 제거됨
    # C: EWMA 변동성 (ewma_vol_10/21/63 VIF 수백 — 다중공선성 제거)
    # D: 변동성 모멘텀 (rv_mom_5_21 제거 — HAR_FEATURE_COLS와 동일하게 통일)
    'rv_term_slope', 'rv_5d_chg',
    # 장중 고빈도 실현분산 (1h; _5d/_21d/_vs_close 고VIF 파생 제거)
    'rv_intraday',
    # VIX 기간구조 + SKEW
    'vix_term_slope', 'vix_ts_zscore', 'skew_zscore', 'skew_chg',
    # 5번: 시장 국면(Regime) 피처
    'regime', 'regime_x_sent', 'regime_x_gpr',
    # 가격 점프 감지 (블랙스완 이벤트)
    'jump_flag', 'jump_magnitude',
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
    'supply_demand_gap', 'demand_event_score', 'inv_draw_x_macd', 'sentiment_chg3',
    'opec_event_score', 'opec_event_score_smooth',
]


# ─────────────────────────────────────────────────────────────────────────────
# §FEA  FEATURE ENGINEERING  —  build_features(): 174개 피처 생성
# ─────────────────────────────────────────────────────────────────────────────
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
        _n_tr_cem = _n_total - 90 if _n_total > 90 else _n_total
        _wti_full = df['WTI'].ffill().values.astype(float)
        _wti_arr  = _wti_full[:_n_tr_cem]          # 훈련 구간만 사용
        _snap_n   = (_n_tr_cem // 30) * 30   # 30행 단위 스냅 → 월 1회만 재계산
        _cem_hash = _hl.md5(_wti_arr[:_snap_n].tobytes()).hexdigest()[:16]
        _cem_cache = OUTPUT_DIR / f'ceemdan_{_cem_hash}.npy'
        _next_miss = ((_n_tr_cem // 30) + 1) * 30 - _n_tr_cem
        if _cem_cache.exists():
            _imfs = np.load(str(_cem_cache), allow_pickle=True)
            log.info("    CEEMDAN 캐시 로드 (다음 재계산까지 ~%d 거래일)", _next_miss)
        else:
            _cem = _CEEMDAN(trials=20, epsilon=0.005, parallel=False)
            _imfs = _cem(_wti_arr)
            np.save(str(_cem_cache), _imfs)
            for _old_c in OUTPUT_DIR.glob('ceemdan_*.npy'):
                if _old_c != _cem_cache:
                    try:
                        _old_c.unlink(missing_ok=True)
                    except Exception as _ul_e:
                        log.debug(f"    CEEMDAN 구 캐시 삭제 실패({_old_c.name}): {_ul_e}")
        # 캐시 IMF 길이 기준으로 _n_cem 결정 (해시 충돌 시 길이 불일치 방지)
        _imf_len = _imfs.shape[1] if _imfs.ndim == 2 else len(_imfs[-1])
        _n_cem = min(len(_wti_arr), _imf_len)
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
        import hashlib as _ghl, pickle as _gpk
        _ret_pct = df['log_return'].dropna() * 100   # % 스케일
        _ghash = _ghl.md5(_ret_pct.values.tobytes()[-2000:]).hexdigest()[:12]
        _garch_loaded = False
        if GARCH_CACHE.exists():
            try:
                with open(GARCH_CACHE, 'rb') as _gf:
                    _gc = _gpk.load(_gf)
                if _gc.get('hash') == _ghash:
                    _res = _gc['result']
                    _garch_loaded = True
                    log.info("    GARCH 캐시 로드")
            except Exception:
                pass
        if not _garch_loaded:
            _garch = _arch_model(_ret_pct, vol='Garch', p=1, q=1,
                                 dist='Normal', rescale=False)
            _res = _garch.fit(disp='off', show_warning=False)
            try:
                _atomic_pkl({'hash': _ghash, 'result': _res}, GARCH_CACHE)
            except Exception:
                pass
        _cond_vol = _res.conditional_volatility / 100   # 소수점 스케일 복원
        df['garch_vol'] = _cond_vol.reindex(df.index).ffill().bfill().fillna(0)
        _garch_result = _res   # 외부 전달용 저장
        log.info(f"    GARCH(1,1) 조건부 분산 생성 (μ={df['garch_vol'].mean():.5f})")
    except Exception as _ge:
        log.warning(f"    GARCH 실패({_ge}) → garch_vol=0")
        df['garch_vol'] = 0.0

    # ── 장중 고빈도 실현분산 (1h 데이터, 최근 365일 — 더 정확한 RV 추정)
    try:
        _intra_today = pd.Timestamp.today().normalize()
        _intra_stale = True
        if INTRADAY_CACHE.exists():
            try:
                import pickle as _pk
                with open(INTRADAY_CACHE, 'rb') as _pf:
                    _ic = _pk.load(_pf)
                if _ic.get('date') == str(_intra_today.date()):
                    _raw_1h = _ic['data']
                    _intra_stale = False
                    log.info("    장중 RV 캐시 로드")
            except Exception:
                pass
        if _intra_stale:
            _raw_1h = yf.download("CL=F", period="365d", interval="1h",
                                  progress=False, auto_adjust=True)
            try:
                _atomic_pkl({'date': str(_intra_today.date()), 'data': _raw_1h}, INTRADAY_CACHE)
            except Exception:
                pass
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
        # 하이브리드: 유가 특화 규칙 75% + FinBERT 25% (FinBERT 중립 편향 희석 방지)
        news_df['sentiment'] = 0.25 * news_df['finbert_score'].values + 0.75 * _rule_scores

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
                for _cat, _col in [('supply','_supply_score'),('demand','_demand_score'),('geo','_geo_score'),('opec','_opec_score')]:
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
                    '_opec':   _wavg_col(g, '_opec_score'),
                })
            )
            _oil_daily.index = pd.to_datetime(_oil_daily.index)
            _oil_s = _oil_daily['_oil'].reindex(df.index).ffill().fillna(0)
            df['oil_event_score']        = _oil_s.values
            df['oil_event_score_smooth'] = _oil_s.ewm(span=3, min_periods=1).mean().values
            for _cat, _col in [('_supply','supply_event_score'),
                                ('_demand','demand_event_score'),
                                ('_geo',   'geo_event_score'),
                                ('_opec',  'opec_event_score')]:
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
    # 가격 점프 감지: |일별수익률| > 3σ(60일) → 블랙스완 이벤트 플래그
    _ret = df['WTI'].pct_change().fillna(0)
    _ret_std60 = _ret.rolling(60, min_periods=10).std()
    df['jump_flag'] = ((_ret.abs() > 3 * _ret_std60) & (_ret_std60 > 0)).astype(float)
    df['jump_magnitude'] = (_ret.abs() / (_ret_std60 + 1e-8)).clip(upper=10).fillna(0)
    # 복합충격 플래그: 독립 이벤트 타입(지정학/OPEC/극단부정뉴스) 2개 이상 5일 내 동시 발생
    _geo_f  = (df['geo_event_score_smooth'].fillna(0)  > 0.3).astype(float) if 'geo_event_score_smooth'  in df.columns else pd.Series(0.0, index=df.index)
    _opec_f = (df['opec_event_score_smooth'].fillna(0) > 0.3).astype(float) if 'opec_event_score_smooth' in df.columns else pd.Series(0.0, index=df.index)
    _neg_f  = df['extreme_neg_news']
    df['compound_shock_flag'] = (
        (_geo_f.rolling(5, min_periods=1).max() +
         _opec_f.rolling(5, min_periods=1).max() +
         _neg_f.rolling(5, min_periods=1).max()) >= 2.0
    ).astype(float)
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
        raise ValueError(f"피처 행 부족: {len(df)}행 (최소 92 필요). 데이터 수집 실패 또는 dropna 과다.")

    log.info(f"    피처 완성: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, df_full, {'garch_model': _garch_result}


# ─────────────────────────────────────────────────────────────────────────────
# §MDL  MODEL TRAINING  —  train_models(): XGBoost-HAR, SARIMAX, VAR, ETS
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
        _atomic_json(_fd_stats, FEAT_TRAIN_STATS)
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

        # ── 과적합 감지 (훈련 R² vs WF CV R²: 표시되는 r2와 일치, r2_ho는 별도 진단용)
        r2_train = float(r2_score(y_rv_tr, modelA.predict(X_tr_s)))
        overfit_gap = r2_train - r2_cv  # train vs WF-CV (r2 컬럼과 동일 기준)
        if overfit_gap > 0.20:
            log.warning(f"    ⚠️ 과적합 의심: 훈련R²={r2_train:.4f} vs CV R²={r2_cv:.4f} "
                        f"(gap={overfit_gap:.3f})  hold-out R²={r2_ho:.4f}")
        elif overfit_gap < -0.10:
            log.info(f"        테스트>훈련 (정상): 훈련R²={r2_train:.4f} vs CV R²={r2_cv:.4f} (gap={overfit_gap:.3f})  hold-out R²={r2_ho:.4f}")
        else:
            log.info(f"        훈련R²={r2_train:.4f}  CV R²={r2_cv:.4f}  hold-out R²={r2_ho:.4f}  gap={overfit_gap:.3f} (정상)")

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
    # Exog: 거시(DXY/수급충격) + VIX — 4개 핵심 변수만 (과적합 억제)
    exog_cols = [c for c in [
        'dxy_change', 'demand_shock', 'supply_shock', 'vix_change',
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
            _adp_src = full_df if full_df is not None else feature_df
            _adp_ovx = float(_adp_src['OVX'].iloc[-1]) if 'OVX' in _adp_src.columns and _adp_src['OVX'].notna().any() else 0.0
            _adp_mom = float(_adp_src['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _adp_src.columns and _adp_src['mom_5d'].notna().any() else 0.0
            # 평가 기간: main test_df와 동일(n_test=90)으로 고정 → 스태킹 정렬 일치
            # 훈련 창: test 시작점 기준 역산 (today 기준 아님 → test 데이터 오염 방지)
            n_test_sx = n_test  # = 90
            _test_start = feature_df.index[-n_test]
            if _adp_ovx >= 50 or abs(_adp_mom) >= 0.06:
                cutoff = _test_start - pd.tseries.offsets.BDay(120)
                log.info(f"    SARIMAX 적응형 윈도우 120일 (train, ovx={_adp_ovx:.0f} mom5={_adp_mom:.1%})")
            else:
                cutoff = _test_start - pd.DateOffset(years=SARIMAX_WINDOW_YEARS)
                log.info(f"    SARIMAX {SARIMAX_WINDOW_YEARS}년 윈도우 (train)")
            sx_df  = _sx_base[_sx_base.index >= cutoff]
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
                        d=1, seasonal=False,
                        start_p=0, max_p=4, start_q=0, max_q=3,
                        information_criterion='oob',
                        out_of_sample_size=30,
                        stepwise=True,
                        error_action='ignore', suppress_warnings=True,
                        n_jobs=1,
                    )
                    sarimax_order    = aa.order
                    sarimax_seasonal = aa.seasonal_order
                    log.info(f"    auto_arima 최적: {sarimax_order} × {sarimax_seasonal}")
                    _atomic_json({
                        'key': _sx_cache_key,
                        'order': list(sarimax_order),
                        'seasonal': list(sarimax_seasonal),
                    }, _sx_cache_file)
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
            _pred_all = pred_obj.predicted_mean.values[-n_test_sx:]
            # 라이브 등가 D+1 평가: pred[k]=E[WTI_k|WTI_{k-1},exog_k], 실제 return 보정
            # → WTI_k + β*exog_k (라이브와 동일: 오늘 exog로 내일 예측)
            # vs target_price[k] = WTI_{k+1}
            # full_wti는 _to_bday로 확장될 수 있으므로 마지막 n_test_sx 포지션 사용
            _wti_returns = full_wti.values[-n_test_sx:] - full_wti.values[-(n_test_sx+1):-1]
            pred_price   = _pred_all + _wti_returns
            y_px_te_sx   = sx_test['target_price'].values
            _eval_mask   = sx_test.index.isin(feature_df.index)
            _pred_eval   = pred_price[_eval_mask]
            _actual_eval = y_px_te_sx[_eval_mask]
            _dates_eval  = sx_test.index[_eval_mask]
            rmse_b = float(np.sqrt(mean_squared_error(_actual_eval, _pred_eval)))
            mae_b  = float(mean_absolute_error(_actual_eval, _pred_eval))
            r2_b   = float(r2_score(_actual_eval, _pred_eval))

            # 라이브 예측용: 전체 데이터로 재학습 (훈련셋만 학습한 fit은 60일 전 상태)
            log.info("    [B1] 전체 데이터 SARIMAX 재학습 (라이브 예측용)...")
            mdl_live = SARIMAX(
                full_wti, exog=full_exog,
                order=sarimax_order, seasonal_order=sarimax_seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit_live = mdl_live.fit(disp=False, maxiter=100)

            _ci_q75_sx = float(np.percentile(np.abs(_actual_eval - _pred_eval), 75))
            results['sarimax'] = {
                'model': fit_live, 'features': exog_cols, 'type': 'price',
                'rmse': rmse_b, 'mae': mae_b, 'r2': r2_b,
                'ci_calib_q75': _ci_q75_sx,
                'name': f'SARIMAX{sarimax_order} 1-step',
                'pred_price_test':   _pred_eval,
                'actual_price_test': _actual_eval,
                'test_dates':        _dates_eval,
                'order': sarimax_order, 'seasonal_order': sarimax_seasonal,
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

    # Prophet: WTI는 불연속 충격 기반이라 트렌드+계절성 모델 부적합 (R²=-4.3 확인)
    # → 학습 생략, 2모델 앙상블(SARIMAX+XGBoost) 유지

    # ─────────────────────────────────────────────────────────────────────
    # Model B2: VAR (WTI + Brent + DXY + VIX 4변수 벡터 자기회귀)
    # ─────────────────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR as _VAR
        _var_cols = [c for c in ['WTI', 'Brent', 'DXY', 'VIX', 'OVX'] if c in feature_df.columns]
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
            _ci_q75_var = float(np.percentile(np.abs(_y_var_te - _var_pred_arr), 75))
            _sx_mae = results.get('sarimax', {}).get('mae', 999)
            results['var'] = {
                'pred_price_test': _var_pred_arr,
                'actual_price_test': _y_var_te,
                'rmse': _rmse_var, 'mae': _mae_var, 'r2': _r2_var,
                'ci_calib_q75': _ci_q75_var,
                'model': _var_fit, 'cols': _var_cols, 'lag': _lag_ord,
                'n_test': _n_te_v, 'type': 'price',
                'name': f'VAR({_lag_ord}) WTI+Brent+DXY+VIX+OVX',
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
                    # 캐시 키: 날짜 + 피처명 (데이터 해시는 매 실행마다 달라져 캐시 미작동)
                    _feat_key = '|'.join(sorted(X_tr_all.columns.tolist())).encode()
                    _date_key = datetime.now().strftime('%Y-%m-%d').encode()
                    _opt_key = _hl_xgb.md5(_feat_key + _date_key).hexdigest()[:16]
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
                            _atomic_json({'key': _opt_key, 'params': _best_xgb_params},
                                         XGB_OPTUNA_CACHE)
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
            _atomic_csv(_imp_df, OUTPUT_DIR / 'xgb_return_importance.csv', index=False)

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
                _pr = model.predict(Xte)   # log return 예측
                _px = test_df['WTI'].values * np.exp(np.clip(_pr, -0.5, 0.5))
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
                    _pp   = _mw.predict(_Xw_val)   # log return 예측
                    _y_px = train_df['WTI'].values[_vi] * np.exp(np.clip(_pp, -0.5, 0.5))
                    maes.append(mean_absolute_error(train_df['target_price'].values[_vi], _y_px))
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
                if _dz_mask_tr.sum() >= 100:
                    _mD_cls_dir.fit(_Xtr_cls[_dz_mask_tr], _y_cls_tr[_dz_mask_tr],
                                    sample_weight=w_ret[_dz_mask_tr])
                else:
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

                    # C 그리드 서치 (7일 이내 캐시 사용으로 skip 가능)
                    import json as _jsc
                    _best_c, _best_c_dir = 1.0, -1.0
                    _svm_c_cached = False
                    if SVM_CACHE.exists():
                        try:
                            _sc_d = _jsc.loads(SVM_CACHE.read_text())
                            _sc_age = (pd.Timestamp.today() - pd.Timestamp(_sc_d['date'])).days
                            if _sc_age < REFIT_STALE_DAYS:
                                _best_c = float(_sc_d['best_c'])
                                _svm_c_cached = True
                                log.info(f"        SVM C 캐시 로드: C={_best_c} (age={_sc_age}d)")
                        except Exception:
                            pass
                    if not _svm_c_cached:
                        _c_grid = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
                        _n_cval = max(30, len(_Xtr_svm) // 4)
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
                        try:
                            _atomic_json({'best_c': _best_c,
                                          'date': str(pd.Timestamp.today().date())}, SVM_CACHE)
                        except Exception:
                            pass
                        log.info(f"        SVM C 그리드: best_C={_best_c} (검증_dir={_best_c_dir*100:.1f}%)")

                    _sm = _SVC(kernel='rbf', C=_best_c, gamma='scale',
                               probability=True, class_weight='balanced', random_state=42)
                    # SVM 모델 캐시 로드 시도 (전체 fit skip으로 ~7s 절감)
                    _svm_model_loaded = False
                    import pickle as _pkl_svm
                    if SVM_MODEL_CACHE.exists() and _svm_c_cached:
                        try:
                            with open(SVM_MODEL_CACHE, 'rb') as _f_sm:
                                _sm_cached = _pkl_svm.load(_f_sm)
                            _n_feat_cached = getattr(_sm_cached, 'n_features_in_', None)
                            if (hasattr(_sm_cached, 'predict_proba') and
                                    (_n_feat_cached is None or _n_feat_cached == _Xtr_svm.shape[1])):
                                _sm = _sm_cached
                                _svm_model_loaded = True
                                log.info(f"        SVM 모델 캐시 로드 (fit skip, feat={_Xtr_svm.shape[1]})")
                            elif _n_feat_cached is not None and _n_feat_cached != _Xtr_svm.shape[1]:
                                log.info(f"        SVM 캐시 무효화: 피처 수 불일치 ({_n_feat_cached}→{_Xtr_svm.shape[1]}), 재훈련")
                        except Exception:
                            pass
                    if not _svm_model_loaded:
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
                        try:
                            _atomic_pkl(_sm, SVM_MODEL_CACHE, protocol=4)
                        except Exception:
                            pass
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

                    # 주간 refit 시 5 seeds, 일간은 3 seeds (앙상블 분산 ±0.3% 이내)
                    try:
                        _lr_age = (pd.Timestamp.today() - pd.Timestamp(
                            __import__('json').loads(LAST_RETRAIN_FILE.read_text()).get(
                                'last_retrain', '2000-01-01'))).days
                    except Exception:
                        _lr_age = 0
                    _n_lstm_seeds = 5 if _lr_age >= REFIT_STALE_DAYS else 3
                    _lprobs = [_ltrain(_Xls_tr, _yls_tr, _Xls_te, _yls_te,
                                       _sw_ls, _dz_ls, seed=s) for s in range(_n_lstm_seeds)]
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

                # ── Walk-forward 5폴드 방향성 평가 (신뢰 지표, 모델 선택 영향 없음)
                # 실제 예측 파이프라인과 동일한 피처셋(_cem_cols 전체) + 최적 C 사용
                _wf_dir_acc = 0.0
                _wf_opt_th  = 0.5
                try:
                    from sklearn.svm import SVC as _SVC_wf
                    _wf_dirs, _wf_ths = [], []
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
                        # WF fold별 최적 threshold 탐색
                        _fold_th, _fold_best = 0.5, 0.0
                        for _th in np.arange(0.35, 0.66, 0.05):
                            _d = float(((_wpv > _th).astype(int) == _y_cls_tr[_wvi]).mean())
                            if _d > _fold_best:
                                _fold_best, _fold_th = _d, float(_th)
                        _wf_dirs.append(_fold_best)
                        _wf_ths.append(_fold_th)
                    _wf_dir_acc = float(np.mean(_wf_dirs)) if _wf_dirs else 0.0
                    _wf_opt_th  = float(np.median(_wf_ths)) if _wf_ths else 0.5
                    log.info(f"        SVM WF 최적 threshold={_wf_opt_th:.2f}")
                except Exception as _wfe:
                    log.warning(f"    WF 평가 실패({_wfe})")
                _wf_svm_acc = _wf_dir_acc  # SVM WF acc 별도 보존

                # XGB Classifier WF 평가 (SVM WF와 최댓값 채택)
                _wf_xgb_acc = 0.0
                try:
                    _wf_xgb_dirs = []
                    _wf_xgb_p = {**_xgb_cls_p, 'n_estimators': 100}  # WF fold용 경량화
                    for _wti, _wvi in TimeSeriesSplit(n_splits=3).split(_Xtr_cls):
                        if len(_wti) < 100 or len(_wvi) < 15:
                            continue
                        _wf_sw2 = np.exp(0.002 * np.arange(len(_wti)))
                        _dz_wf = _dz_mask_tr[_wti]
                        _wxgb = xgb.XGBClassifier(**_wf_xgb_p)
                        if _dz_wf.sum() >= 30:
                            _wxgb.fit(_Xtr_cls[_wti][_dz_wf], _y_cls_tr[_wti][_dz_wf],
                                      sample_weight=_wf_sw2[_dz_wf])
                        else:
                            _wxgb.fit(_Xtr_cls[_wti], _y_cls_tr[_wti], sample_weight=_wf_sw2)
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

                # WF 평가 완료 후 threshold 적용 (WF fold threshold는 참고용; test엔 0.5 고정)
                _best_th    = 0.5
                _best_dir_th = float(((_prob_up_te > _best_th).astype(int) == _y_cls_te).mean())
                log.info(f"        threshold={_best_th:.2f} (WF최적) → dir_acc={_best_dir_th*100:.1f}%")
                _dir_cls = _best_dir_th
                # 분류기 방향 + 회귀 크기 결합 (Classifier-adj)
                _sign_cls = np.where(_prob_up_te > _best_th, 1.0, -1.0)
                _pr_cls_adj = np.clip(np.abs(_pr_s) * _sign_cls, -0.5, 0.5)
                _px_cls_adj = test_df['WTI'].values * np.exp(_pr_cls_adj)
                _mae_cls    = float(mean_absolute_error(y_px_te, _px_cls_adj))
                _r2_cls     = float(r2_score(y_px_te, _px_cls_adj))
                _rmse_cls   = float(np.sqrt(mean_squared_error(y_px_te, _px_cls_adj)))

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

            # ── HAR-XGB Enhanced — MI+HAR 결합, dead-zone, inv-vol, SVM, WF-blend
            _har_cls_cand = None
            try:
                _har_y_tr = (y_ret_tr.values > 0).astype(int)
                _har_y_te = (y_ret_te.values > 0).astype(int)
                _dz_har   = np.abs(y_ret_tr.values) > 0.003

                # ① MI+HAR 하이브리드 피처
                _har_base = [c for c in har_feats if c in train_df.columns]
                try:
                    from sklearn.feature_selection import mutual_info_classif as _mic_h
                    _mi_h_sc = _mic_h(train_df[available_feats].fillna(0).values,
                                      _har_y_tr, random_state=42)
                    _mi_h_ranked = sorted(zip(available_feats, _mi_h_sc),
                                          key=lambda x: x[1], reverse=True)
                    _mi_h_extra = [f for f, _ in _mi_h_ranked[:30]
                                   if f not in _har_base and f in train_df.columns][:10]
                    _har_cls_feats = _har_base + _mi_h_extra
                    log.info(f"    HAR+MI: {len(_har_base)} HAR + {len(_mi_h_extra)} MI = {len(_har_cls_feats)}")
                except Exception as _mi_h_e:
                    _har_cls_feats = _har_base
                    log.warning(f"    HAR MI 실패({_mi_h_e}), HAR만 사용")

                # ② inv-vol 가중치
                _sw_har_exp = np.exp(0.002 * np.arange(len(_har_y_tr)))
                try:
                    _vol_har    = train_df['vol_5d'].fillna(train_df['vol_5d'].median()).values.astype(np.float32)
                    _inv_vol_h  = 1.0 / (_vol_har + 1e-6)
                    _inv_vol_h /= _inv_vol_h.mean()
                    _sw_har     = _sw_har_exp * _inv_vol_h
                    _sw_har    /= _sw_har.mean()
                except Exception:
                    _sw_har = _sw_har_exp

                _sc_har_cls = StandardScaler()
                _Xtr_har    = _sc_har_cls.fit_transform(train_df[_har_cls_feats].fillna(0).values)
                _Xte_har    = _sc_har_cls.transform(test_df[_har_cls_feats].fillna(0).values)

                _har_cls_p = dict(
                    n_estimators=300, max_depth=3, learning_rate=0.02,
                    subsample=0.75, colsample_bytree=0.6, colsample_bynode=0.8,
                    min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0,
                    n_jobs=-1, random_state=42, verbosity=0
                )
                _har_mdl = xgb.XGBClassifier(**_har_cls_p)
                # ③ dead-zone 마스크 적용
                if _dz_har.sum() >= 100:
                    _har_mdl.fit(_Xtr_har[_dz_har], _har_y_tr[_dz_har],
                                 sample_weight=_sw_har[_dz_har])
                else:
                    _har_mdl.fit(_Xtr_har, _har_y_tr, sample_weight=_sw_har)
                _har_prob    = _har_mdl.predict_proba(_Xte_har)[:, 1]
                _har_dir     = float(((_har_prob > 0.5).astype(int) == _har_y_te).mean())
                _prob_har_up = _har_prob.copy()

                # ④ SVM on HAR+MI features (C grid)
                _har_svm_prob  = None
                _best_c_har    = 1.0
                _wf_har_svm_acc = 0.0
                try:
                    from sklearn.svm import SVC as _SVC_har
                    _best_c_har_dir = -1.0
                    _n_cv_h = max(30, len(_Xtr_har) // 4)
                    _sw_cv_h = np.exp(0.002 * np.arange(len(_Xtr_har) - _n_cv_h))
                    _dz_cv_h = _dz_har[:len(_Xtr_har) - _n_cv_h]
                    for _ci_h in [0.3, 0.5, 1.0, 2.0, 3.0]:
                        _smh_c = _SVC_har(kernel='rbf', C=_ci_h, gamma='scale',
                                          probability=True, class_weight='balanced', random_state=42)
                        if _dz_cv_h.sum() >= 30:
                            _smh_c.fit(_Xtr_har[:-_n_cv_h][_dz_cv_h],
                                       _har_y_tr[:-_n_cv_h][_dz_cv_h],
                                       sample_weight=_sw_cv_h[_dz_cv_h])
                        else:
                            _smh_c.fit(_Xtr_har[:-_n_cv_h], _har_y_tr[:-_n_cv_h],
                                       sample_weight=_sw_cv_h)
                        _cp_h = _smh_c.predict_proba(_Xtr_har[-_n_cv_h:])[:, 1]
                        _cd_h = float(((_cp_h > 0.5).astype(int) == _har_y_tr[-_n_cv_h:]).mean())
                        if _cd_h > _best_c_har_dir:
                            _best_c_har_dir, _best_c_har = _cd_h, _ci_h
                    _sm_har = _SVC_har(kernel='rbf', C=_best_c_har, gamma='scale',
                                       probability=True, class_weight='balanced', random_state=42)
                    if _dz_har.sum() >= 100:
                        _sm_har.fit(_Xtr_har[_dz_har], _har_y_tr[_dz_har],
                                    sample_weight=_sw_har[_dz_har])
                    else:
                        _sm_har.fit(_Xtr_har, _har_y_tr, sample_weight=_sw_har)
                    _har_svm_prob = _sm_har.predict_proba(_Xte_har)[:, 1]
                    _har_svm_dir  = float(((_har_svm_prob > 0.5).astype(int) == _har_y_te).mean())
                    log.info(f"    [HAR-SVM] dir={_har_svm_dir*100:.1f}%  C={_best_c_har}")
                    if _har_svm_dir > _har_dir:
                        _prob_har_up = _har_svm_prob
                        log.info(f"    → HAR-SVM 채택 ({_har_dir*100:.1f}% → {_har_svm_dir*100:.1f}%)")
                except Exception as _svm_h_e:
                    log.warning(f"    HAR-SVM 실패({_svm_h_e})")

                # ⑤ WF-weighted blend (XGB WF + SVM WF)
                _wf_har_xgb_acc = 0.0
                try:
                    _wf_har_xgb_dirs = []
                    for _wti_h, _wvi_h in TimeSeriesSplit(n_splits=3).split(_Xtr_har):
                        if len(_wti_h) < 100 or len(_wvi_h) < 15:
                            continue
                        _wh   = xgb.XGBClassifier(**_har_cls_p)
                        _dz_wh = _dz_har[_wti_h]
                        _sw_wh = np.exp(0.002 * np.arange(len(_wti_h)))
                        if _dz_wh.sum() >= 30:
                            _wh.fit(_Xtr_har[_wti_h][_dz_wh], _har_y_tr[_wti_h][_dz_wh],
                                    sample_weight=_sw_wh[_dz_wh])
                        else:
                            _wh.fit(_Xtr_har[_wti_h], _har_y_tr[_wti_h], sample_weight=_sw_wh)
                        _whp = _wh.predict_proba(_Xtr_har[_wvi_h])[:, 1]
                        _wf_har_xgb_dirs.append(
                            float(((_whp > 0.5).astype(int) == _har_y_tr[_wvi_h]).mean()))
                    _wf_har_xgb_acc = float(np.mean(_wf_har_xgb_dirs)) if _wf_har_xgb_dirs else 0.0

                    if _har_svm_prob is not None:
                        _wf_har_svm_dirs = []
                        for _wti_h, _wvi_h in TimeSeriesSplit(n_splits=3).split(_Xtr_har):
                            if len(_wti_h) < 100 or len(_wvi_h) < 15:
                                continue
                            _sw_wh_s = np.exp(0.002 * np.arange(len(_wti_h)))
                            _dz_wh_s = _dz_har[_wti_h]
                            _smh_wf  = _SVC_har(kernel='rbf', C=_best_c_har, gamma='scale',
                                                probability=True, class_weight='balanced', random_state=42)
                            if _dz_wh_s.sum() >= 30:
                                _smh_wf.fit(_Xtr_har[_wti_h][_dz_wh_s],
                                            _har_y_tr[_wti_h][_dz_wh_s],
                                            sample_weight=_sw_wh_s[_dz_wh_s])
                            else:
                                _smh_wf.fit(_Xtr_har[_wti_h], _har_y_tr[_wti_h],
                                            sample_weight=_sw_wh_s)
                            _whps = _smh_wf.predict_proba(_Xtr_har[_wvi_h])[:, 1]
                            _wf_har_svm_dirs.append(
                                float(((_whps > 0.5).astype(int) == _har_y_tr[_wvi_h]).mean()))
                        _wf_har_svm_acc = float(np.mean(_wf_har_svm_dirs)) if _wf_har_svm_dirs else 0.0
                        log.info(f"    HAR WF: XGB={_wf_har_xgb_acc*100:.1f}%  SVM={_wf_har_svm_acc*100:.1f}%")

                        _tot_har_wf = _wf_har_xgb_acc + _wf_har_svm_acc
                        if _tot_har_wf > 0:
                            _wx_h = _wf_har_xgb_acc / _tot_har_wf
                            _ws_h = _wf_har_svm_acc / _tot_har_wf
                            _pb_har = _wx_h * _har_prob + _ws_h * _har_svm_prob
                            _db_har = float(((_pb_har > 0.5).astype(int) == _har_y_te).mean())
                            _cur_d_har = float(((_prob_har_up > 0.5).astype(int) == _har_y_te).mean())
                            if _db_har > _cur_d_har:
                                _prob_har_up = _pb_har
                                log.info(f"    [HAR WF-blend] dir={_db_har*100:.1f}% "
                                         f"(xgb_w={_wx_h:.2f} svm_w={_ws_h:.2f})")
                        _wf_har_acc = max(_wf_har_xgb_acc, _wf_har_svm_acc)
                    else:
                        _wf_har_acc = _wf_har_xgb_acc
                except Exception as _wf_h_e:
                    _wf_har_acc = _wf_har_xgb_acc
                    log.warning(f"    HAR WF-blend 실패({_wf_h_e})")

                _har_final_dir = float(((_prob_har_up > 0.5).astype(int) == _har_y_te).mean())
                _sign_har      = np.where(_prob_har_up > 0.5, 1.0, -1.0)
                _pr_har_adj    = np.clip(np.abs(_pr_s) * _sign_har, -0.5, 0.5)
                _px_har_adj    = test_df['WTI'].values * np.exp(_pr_har_adj)
                _mae_har       = float(mean_absolute_error(y_px_te, _px_har_adj))
                _r2_har        = float(r2_score(y_px_te, _px_har_adj))
                _rmse_har      = float(np.sqrt(mean_squared_error(y_px_te, _px_har_adj)))
                log.info(f"    [HAR-XGB Enhanced] test_dir={_har_final_dir*100:.1f}%  "
                         f"wf_dir={_wf_har_acc*100:.1f}%  MAE={_mae_har:.4f}  "
                         f"feats={len(_har_cls_feats)}")
                results['har_classifier'] = {
                    'model': _har_mdl, 'scaler': _sc_har_cls, 'features': _har_cls_feats,
                    'dir_acc': _har_final_dir, 'wf_dir_acc': _wf_har_acc, 'type': 'price',
                    'rmse': _rmse_har, 'mae': _mae_har, 'r2': _r2_har,
                    'name': f'HAR-XGB-Enhanced (방향성={_har_final_dir*100:.1f}%)',
                }
                if _wf_har_acc >= 0.51:
                    _har_cls_cand = (_mae_har, _har_final_dir, _har_mdl, _sc_har_cls,
                                     _pr_har_adj, _px_har_adj, _r2_har,
                                     _har_cls_feats, 'HAR-Enhanced')
                    log.info(f"    HAR-Enhanced 후보 추가 (wf_dir={_wf_har_acc*100:.1f}%)")
            except Exception as _hce:
                log.warning(f"    HAR-XGB Enhanced 실패({_hce})")

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

            # ── 회귀 후보 WF dir 계산 (분류기와 동일 기준으로 공정 비교)
            _wf_dir_f, _wf_dir_s = _dir_f, _dir_s  # fallback: test-set
            try:
                _y_ret_arr = y_ret_tr.values   # 훈련 타겟 + 방향 평가용
                _wf_dirs_f, _wf_dirs_s = [], []
                for _wti_r, _wvi_r in TimeSeriesSplit(n_splits=3).split(_Xtr_full):
                    if len(_wti_r) < 100 or len(_wvi_r) < 15:
                        continue
                    _wr_f = xgb.XGBRegressor(**_xgb_p)
                    _wr_f.fit(_Xtr_full[_wti_r], _y_ret_arr[_wti_r],
                              sample_weight=w_ret[_wti_r])
                    _pr_wf_f = _wr_f.predict(_Xtr_full[_wvi_r])  # log return 예측
                    _wf_dirs_f.append(
                        float((np.sign(_pr_wf_f) == np.sign(_y_ret_arr[_wvi_r])).mean()))

                    _wr_s = xgb.XGBRegressor(**_xgb_p)
                    _wr_s.fit(_Xtr_sel[_wti_r], _y_ret_arr[_wti_r],
                              sample_weight=w_ret[_wti_r])
                    _pr_wf_s = _wr_s.predict(_Xtr_sel[_wvi_r])  # log return 예측
                    _wf_dirs_s.append(
                        float((np.sign(_pr_wf_s) == np.sign(_y_ret_arr[_wvi_r])).mean()))

                _wf_dir_f = float(np.mean(_wf_dirs_f)) if _wf_dirs_f else _dir_f
                _wf_dir_s = float(np.mean(_wf_dirs_s)) if _wf_dirs_s else _dir_s
                log.info(f"    회귀 WF dir: 전체={_wf_dir_f*100:.1f}%  선택={_wf_dir_s*100:.1f}%")
            except Exception as _wf_reg_e:
                log.warning(f"    회귀 WF dir 실패({_wf_reg_e}), test-set dir fallback")

            # 후보 모델: MAE 기준 선택피처 vs 전체피처, + Quantile
            _candidates = [
                (_mae_f, _wf_dir_f, _mD_full, _sc_full, _pr_f, _px_f, _r2_f, available_feats, 'MSE-전체'),
                (_mae_s, _wf_dir_s, _mD_sel,  _sc_sel,  _pr_s, _px_s, _r2_s, _sel_feats,      'MSE-선택'),
            ]
            if _mD_q is not None:
                _candidates.append((_mae_q, _dir_q, _mD_q, _sc_sel, _pr_q, _px_q, _r2_q, _sel_feats, 'Quantile'))
            if _mD_cls_dir is not None and _wf_dir_acc >= 0.51:
                # 모델 선택 기준은 신뢰도 높은 wf_dir_acc 사용 (test-set 단일 평가 과대평가 방지)
                _sel_dir_cls = _wf_dir_acc if _wf_dir_acc > 0 else _dir_cls
                _candidates.append((_mae_cls, _sel_dir_cls, _mD_cls_dir, _sc_sel, _pr_cls_adj, _px_cls_adj, _r2_cls, _sel_feats, 'Classifier-adj'))
            if _har_cls_cand is not None:
                _candidates.append(_har_cls_cand)

            # MAE ≤ 최선 + 5% 범위 내에서 dir_acc 최고 모델 채택
            _best_mae = min(c[0] for c in _candidates)
            _filtered = [c for c in _candidates if c[0] <= _best_mae * 1.05]
            _chosen = max(_filtered, key=lambda c: c[1])
            mae_d, dir_acc, modelD, ret_scaler, pred_ret, pred_px_d, r2_d, _use_feats, _cname = _chosen
            log.info(f"    ✅ 채택: {_cname} (MAE={mae_d:.4f} dir={dir_acc*100:.1f}%)")

            rmse_d = float(np.sqrt(mean_squared_error(y_px_te, pred_px_d)))

            # 신뢰구간용 Quantile 모델 (α=0.125, α=0.875) — 75% 예측 구간
            _Xtr_chosen = ret_scaler.transform(train_df[_use_feats])
            _mD_q10, _mD_q90 = None, None
            try:
                _qbase = {k: v for k, v in _xgb_p.items() if k not in ('objective', 'huber_slope')}
                _q10p  = dict(**_qbase, objective='reg:quantileerror', quantile_alpha=0.125)
                _q90p  = dict(**_qbase, objective='reg:quantileerror', quantile_alpha=0.875)
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
            # 분류기 채택 시: 회귀 모델(크기) + 분류기(방향) 분리 보관 (라이브 예측 반영)
            if _cname == 'Classifier-adj' and _mD_sel is not None:
                _xr_entry['_reg_model']      = _mD_sel
                _xr_entry['_reg_scaler']     = _sc_sel
                _xr_entry['_reg_features']   = _sel_feats
                _xr_entry['xgb_cls_model']   = locals().get('_xgb_cls_saved', None)
                _xr_entry['xgb_cls_blend_w'] = locals().get('_xgb_blend_w_best', 0.0)
            elif _cname == 'HAR-Enhanced' and _mD_sel is not None:
                _xr_entry['_reg_model']    = _mD_sel
                _xr_entry['_reg_scaler']   = _sc_sel
                _xr_entry['_reg_features'] = _sel_feats
            results['xgb_return'] = {**_xr_entry,
            }

        except Exception as exc:
            log.warning(f"    XGBoost 수익률 예측 실패({exc})")
    # ─────────────────────────────────────────────────────────────────────
    # Model F: 직접 다단계 XGBoost (D+2-7 horizon-specific)
    # ─────────────────────────────────────────────────────────────────────
    if _XGB and _SKL and 'xgb_return' in results:
        try:
            log.info("    [F] 직접 다단계 XGBoost (D+2-7) 학습 중...")
            _xr = results['xgb_return']
            _feats_ms = [f for f in (_xr.get('_reg_features') or _xr['features']) if f in feature_df.columns]
            _ms_models, _ms_mase = {}, {}
            _xgb_ms_p = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.7, colsample_bytree=0.6,
                             min_child_weight=8, reg_alpha=0.2, reg_lambda=2.0,
                             n_jobs=1, random_state=42, verbosity=0)
            for h in range(2, 8):
                _wti_s    = feature_df['WTI']
                _log_ret_h = np.log(_wti_s.shift(-h) / _wti_s)
                _y_tr_h   = _log_ret_h.iloc[:-n_test].values
                _y_te_h   = _log_ret_h.iloc[-n_test:].values
                _n_valid  = n_test - h
                _mask_tr  = ~np.isnan(_y_tr_h) & np.all(np.isfinite(train_df[_feats_ms].values), axis=1)
                _y_te_v   = _y_te_h[:_n_valid]
                _Xte_v    = test_df[_feats_ms].values[:_n_valid]
                _mask_te  = ~np.isnan(_y_te_v) & np.all(np.isfinite(_Xte_v), axis=1)
                if _mask_tr.sum() < 120 or _mask_te.sum() < 10:
                    continue
                _sc_h  = StandardScaler()
                _Xtr_sc = _sc_h.fit_transform(train_df[_feats_ms].values[_mask_tr])
                _Xte_sc = _sc_h.transform(_Xte_v[_mask_te])
                _m_h   = xgb.XGBRegressor(**_xgb_ms_p)
                _m_h.fit(_Xtr_sc, _y_tr_h[_mask_tr])
                _pred_ret = np.clip(_m_h.predict(_Xte_sc), -0.3, 0.3)
                _wti_cur  = test_df['WTI'].values[:_n_valid][_mask_te]
                _pred_px  = _wti_cur * np.exp(_pred_ret)
                _act_px   = _wti_cur * np.exp(_y_te_v[_mask_te])
                _mae_h    = float(mean_absolute_error(_act_px, _pred_px))
                _naive_h  = float(mean_absolute_error(_act_px, _wti_cur))
                _mase_h   = _mae_h / max(_naive_h, 1e-6)
                _ms_models[h] = {'model': _m_h, 'scaler': _sc_h, 'features': _feats_ms}
                _ms_mase[h]   = _mase_h
                log.info(f"        D+{h} XGB: MAE={_mae_h:.4f}, MASE={_mase_h:.4f}")
            if _ms_models:
                results['xgb_ms'] = {'models': _ms_models, 'mase': _ms_mase}
        except Exception as _ms_e:
            log.warning(f"    직접 다단계 XGBoost 실패({_ms_e})")
    # ─────────────────────────────────────────────────────────────────────
    # Model E: Stacking 앙상블 (SARIMAX + XGBoost → Ridge 메타러너)
    # ─────────────────────────────────────────────────────────────────────
    if (_SKL and 'sarimax' in results and 'xgb_return' in results):
        try:
            sx_info  = results['sarimax']
            xr_info  = results['xgb_return']
            sx_pred  = sx_info.get('pred_price_test')
            xr_pred  = None

            # XGBoost-Return 테스트 예측값 재계산 (분류기 채택 시 회귀 백업 사용)
            if xr_info:
                _md_s  = xr_info.get('_reg_model') or xr_info['model']
                _sc_s  = xr_info.get('_reg_scaler') or xr_info['scaler']
                _fs_s  = [f for f in (xr_info.get('_reg_features') or xr_info['features'])
                          if f in test_df.columns]
                _Xte_s = _sc_s.transform(test_df[_fs_s])
                _pr_s  = _md_s.predict(_Xte_s)
                if hasattr(_md_s, 'predict_proba'):
                    log.warning("    [Stacking eval] XGB 회귀모델 없음 — 분류기 감지, Persistence fallback 적용")
                    _pr_s = np.zeros(len(_pr_s))
                xr_pred = test_df['WTI'].values * np.exp(np.clip(_pr_s, -0.5, 0.5))

            if sx_pred is not None and xr_pred is not None:
                # 적응형 윈도우 시 SARIMAX test < XGB test → 짧은 쪽으로 정렬
                _n_sx = len(sx_pred)
                _n_xr = len(xr_pred)
                if _n_sx != _n_xr:
                    _n_stk = min(_n_sx, _n_xr)
                    sx_pred = sx_pred[-_n_stk:]
                    xr_pred = xr_pred[-_n_stk:]
                    log.info(f"    스태킹 test 길이 정렬: SARIMAX {_n_sx} ≠ XGB {_n_xr} → {_n_stk}행")

                _stack_parts = [sx_pred, xr_pred]
                _stack_names = ['SARIMAX', 'XGB']

                # ③ VAR 예측이 있으면 스택에 추가 (테스트 길이 맞춰 정렬)
                if 'var' in results:
                    _var_pred_raw = results['var']['pred_price_test']
                    _n_align = min(len(sx_pred), len(_var_pred_raw))
                    if _n_align == len(sx_pred):
                        _stack_parts.append(_var_pred_raw[-len(sx_pred):])
                        _stack_names.append('VAR')

                # ETS는 stacking에서 제외 — SARIMAX가 앙상블 다양성 기여
                # (ETS 단독 MAE 우수하더라도 스태킹 성능 저하 확인됨)

                _stack_X = np.column_stack(_stack_parts)
                _n_stk_actual = len(_stack_X)
                _stack_y = y_px_te.iloc[-_n_stk_actual:] if hasattr(y_px_te, 'iloc') else y_px_te[-_n_stk_actual:]

                # 역MAE 가중평균 (초기값)
                _mae_lookup = {
                    'SARIMAX': sx_info.get('mae', 999.0),
                    'ETS':     results.get('ets', {}).get('mae', 999.0),
                    'XGB':     xr_info.get('mae', 999.0),
                    'VAR':     results.get('var', {}).get('mae', 999.0),
                }
                # MASE 필터: naive 대비 개선된 모델만(MASE<1.0) 풀 진입
                _y_stk_naive   = test_df['WTI'].values[-_n_stk_actual:]
                _naive_mae_stk = float(mean_absolute_error(_stack_y, _y_stk_naive))
                _stk_thresh    = _naive_mae_stk
                _mase_mask = np.array([_mae_lookup.get(n, 999.0) < _stk_thresh
                                       for n in _stack_names])
                if not _mase_mask.any():
                    _mase_mask[:] = True  # 폴백: 전 모델 기준 미달이면 원본 InvMAE 사용
                    log.info("    MASE 필터 무력화: 전 모델 MASE≥0.85 → 원본 InvMAE 사용")
                elif _mase_mask.sum() <= 1:
                    _sole = [n for n, m in zip(_stack_names, _mase_mask) if m]
                    log.info(f"    스태킹 건너뜀: 필터 통과 모델 1개({_sole}) — 단독 모델 사용")
                    raise ValueError(f"단일 모델 풀({_sole[0]}) — 스태킹 무효")
                else:
                    _excl = [n for n, m in zip(_stack_names, _mase_mask) if not m]
                    if _excl:
                        log.info(f"    MASE 필터 제외: {_excl} "
                                 f"(기준={_stk_thresh:.4f}, naive×0.85)")
                _inv_mae = np.array([
                    1.0 / max(_mae_lookup.get(n, 999.0), 1e-6) if _mase_mask[i] else 0.0
                    for i, n in enumerate(_stack_names)
                ])
                _stack_weights = _inv_mae / _inv_mae.sum()
                _stack_pred    = _stack_X @ _stack_weights
                _meta_type     = 'InvMAE-WA'
                _stk_oos_slice = slice(None)  # OOS 평가 구간: Roll30 채택 시 holdout으로 제한

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
                        _naive_mae_fh = float(mean_absolute_error(
                            _stack_y[:_split_m], _y_stk_naive[:_split_m]))
                        _mase_ok_fh = np.array([_mae_fh[n] < _naive_mae_fh
                                                for n in _stack_names])
                        if not _mase_ok_fh.any():
                            _mase_ok_fh[:] = True
                        _inv_fh = np.array([
                            1.0 / max(_mae_fh[n], 1e-6) if _mase_ok_fh[i] else 0.0
                            for i, n in enumerate(_stack_names)
                        ])
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
                            # holdout 검증된 _rm.coef_ 그대로 사용 (전체셋 재훈련 시 in-sample leakage 발생)
                            _coef = np.maximum(_rm.coef_, 0)
                            if _coef.sum() > 0:
                                _stack_weights = _coef / _coef.sum()
                            else:
                                _stack_weights = (np.abs(_rm.coef_)
                                                  / np.abs(_rm.coef_).sum())
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
                        # 가중치 보정 구간(_n30)은 미래 데이터 — 성능 메트릭은 holdout 앞부분으로만 측정
                        _stk_oos_slice = slice(None, _eval_end)
                        log.info(f"    ✅ Rolling-30d 재보정 채택")

                # ── 동적 블랙스완 가중치 (jump_flag/extreme_neg_news → SARIMAX per-row 페널티)
                if 'SARIMAX' in _stack_names and 'jump_flag' in test_df.columns:
                    _sx_i   = _stack_names.index('SARIMAX')
                    _n_s    = len(_stack_X)
                    _jf_t   = test_df['jump_flag'].fillna(0).values[-_n_s:]
                    _en_t   = (test_df['extreme_neg_news'].fillna(0).values[-_n_s:]
                               if 'extreme_neg_news' in test_df.columns else np.zeros(_n_s))
                    _gz_t   = (test_df['gpr_zscore'].fillna(0).values[-_n_s:]
                               if 'gpr_zscore' in test_df.columns else np.zeros(_n_s))
                    _bpen_t = np.where((_en_t >= 1) & (_jf_t >= 1), 0.45,
                              np.where((_en_t >= 1) | (_jf_t >= 1) | (_gz_t > 2.0), 0.60, 1.0))
                    _pm     = np.ones((_n_s, len(_stack_names)))
                    _pm[:, _sx_i] = _bpen_t
                    _wr     = _stack_weights[np.newaxis, :] * _pm
                    _wr    /= _wr.sum(axis=1, keepdims=True)
                    _dyn_p  = (_stack_X * _wr).sum(axis=1)
                    _s_ev   = _stk_oos_slice
                    _mae_pre  = float(mean_absolute_error(_stack_y[_s_ev], _stack_pred[_s_ev]))
                    _mae_dyn  = float(mean_absolute_error(_stack_y[_s_ev], _dyn_p[_s_ev]))
                    _maxe_pre = float(np.abs(_stack_y - _stack_pred).max())
                    _maxe_dyn = float(np.abs(_stack_y - _dyn_p).max())
                    _n_stress = int((_bpen_t < 1.0).sum())
                    if _mae_dyn <= _mae_pre + 0.05:
                        _stack_pred = _dyn_p
                        log.info(f"    ✅ 동적BS가중치 채택 (stress={_n_stress}일): "
                                 f"MAE {_mae_pre:.4f}→{_mae_dyn:.4f}  "
                                 f"maxErr {_maxe_pre:.2f}→{_maxe_dyn:.2f}")
                    else:
                        log.info(f"    동적BS가중치 미채택: MAE {_mae_pre:.4f}→{_mae_dyn:.4f} "
                                 f"(stress={_n_stress}일)")

                _s = _stk_oos_slice
                _mae_stack  = float(mean_absolute_error(_stack_y[_s], _stack_pred[_s]))
                _r2_stack   = float(r2_score(_stack_y[_s], _stack_pred[_s]))
                _rmse_stack = float(np.sqrt(mean_squared_error(_stack_y[_s], _stack_pred[_s])))
                _ci_q75_stk = float(np.percentile(np.abs(_stack_y - _stack_pred), 75))
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
                    elif 0 < _wv < 0.05:
                        log.warning(f"    ⚠ 스태킹 가중치 미미: {_wn}={_wv:.3f} < 0.05 "
                                    f"(해당 모델 기여 없음)")

                # Stacking 채택 기준: 모든 컴포넌트 모델 중 최고 성능 기준
                # MAE < min(SARIMAX,XGB,VAR MAE)  OR  (R² > max R² AND MAE < min_mae × 1.05)
                _mae_best = min(
                    sx_info.get('mae', float('inf')),
                    xr_info.get('mae', float('inf')),
                    results.get('var', {}).get('mae', float('inf')),
                )
                _r2_best = max(
                    sx_info.get('r2', -float('inf')),
                    xr_info.get('r2', -float('inf')),
                    results.get('var', {}).get('r2', -float('inf')),
                )
                _stk_mae_ok    = _mae_stack < _mae_best
                _stk_r2_relax  = (_r2_stack > _r2_best) and (_mae_stack < _mae_best * 1.05)
                if _stk_mae_ok or _stk_r2_relax:
                    _base_name = '+'.join(_stack_names)
                    _stk_info = {
                        'stack_weights': _stack_weights, 'type': 'price',
                        'rmse': _rmse_stack, 'mae': _mae_stack, 'r2': _r2_stack,
                        'ci_calib_q75': _ci_q75_stk,
                        'name': f'Stacking ({_base_name},InvMAE-WA)',
                        'sx_feats':    sx_info.get('features', []),
                        'xr_feats':    xr_info['features'],
                        'xr_scaler':   xr_info['scaler'],
                        'stack_names': _stack_names,
                        'meta_type':   _meta_type,
                        'pred_price_test':   _stack_pred.copy(),
                        'actual_price_test': _stack_y.copy(),
                    }
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
                        _atomic_json({
                            'names':   list(_stack_names),
                            'weights': list(_stk_info['stack_weights'].round(6).tolist()),
                        }, STACK_WEIGHTS_EMA)
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
                    log.info(f"    Stacking 미채택 (MAE={_mae_stack:.4f} ≥ best_component {_mae_best:.4f}×1.05={_mae_best*1.05:.4f})")
                    results['stacking_rejected'] = {
                        'name': f'Stacking (+{"+".join(_stack_names)},미채택)',
                        'type': 'price',
                        'rmse': round(_rmse_stack, 5),
                        'mae':  round(_mae_stack,  5),
                        'r2':   round(_r2_stack,   4),
                    }
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
    # M6-ABS 게이트용: 나이브 MAE를 results에 저장
    results['naive_px_mae'] = _naive_mae_px

    # ── 성능 저장
    perf_rows = []
    for v in results.values():
        if not isinstance(v, dict):
            continue
        if 'name' not in v or 'type' not in v:
            continue
        if v.get('benchmark_only', False):
            continue
        row = {'model': v['name'], 'target': v['type'],
               'rmse': round(v['rmse'], 5), 'mae': round(v['mae'], 5), 'r2': round(v['r2'], 4)}
        if 'dir_acc'     in v: row['dir_acc']     = round(v['dir_acc'], 4)
        if 'wf_dir_acc'  in v: row['wf_dir_acc']  = round(v['wf_dir_acc'], 4)
        if 'r2_ho'       in v: row['r2_ho']       = round(v['r2_ho'], 4)
        if 'train_r2'    in v: row['train_r2']    = round(v['train_r2'], 4)
        if 'overfit_gap' in v: row['overfit_gap'] = round(v['overfit_gap'], 4)
        # MASE (1.0보다 작아야 기준선 대비 개선)
        try:
            _nm = (_naive_mae_px if v['type'] == 'price' else _naive_mae_rv)
            if _nm and _nm > 0:
                row['mase'] = round(v['mae'] / _nm, 4)
        except Exception:
            pass
        # MaxError (꼬리 오차 관리) + CI Q75
        if 'pred_price_test' in v and 'actual_price_test' in v:
            try:
                _errs = np.abs(np.array(v['actual_price_test']) - np.array(v['pred_price_test']))
                row['max_error'] = round(float(_errs.max()), 4)
                row['ci_q75']    = round(float(np.percentile(_errs, 75)), 4)
            except Exception:
                pass
        elif 'ci_calib_q75' in v:
            row['ci_q75'] = round(float(v['ci_calib_q75']), 4)
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
            ('inv_chg_zscore',        f'EIA 재고 (fix: +{EIA_SHIFT}영업일 발표지연 적용)'),
            ('inv_surprise',          f'EIA 서프라이즈 (fix: +{EIA_SHIFT}영업일)'),
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




def compute_ensemble_weights(window: int = 30, ovx_level: float = 0.0, price_mom: float = 0.0):
    """R² 기반 초기 가중치 + MAPE 미세조정 + OVX/모멘텀 레짐 패널티로 SARIMAX/XGBoost 가중치 산출.

    1단계: model_performance.csv의 역MAE로 비례 가중치 계산
    2단계: 최근 backtest/live MAPE로 미세조정 (OVX HIGH 시 범위 확장)
    3단계: OVX 레짐 페널티 (OVX≥60 → -0.12) + M5 가격 모멘텀 충격 페널티
    R² 정보 없으면 기본값(0.65/0.35), 최종 클램프 [0.10, 0.70].
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
        if len(lv) >= 5:
            lv_mape = (lv['price_error'].abs() / lv['actual_price'].replace(0, np.nan) * 100).mean()
            sarimax_mape = 0.3 * bt_mape + 0.7 * lv_mape
        else:
            sarimax_mape = bt_mape

        # M4: OVX HIGH 시 MAPE 보정 범위 확장 + OVX 레짐 페널티
        _mape_lo = -0.20 if ovx_level >= 60 else (-0.15 if ovx_level >= 40 else -0.10)
        mape_adj = float(np.clip(np.interp(sarimax_mape, [3.0, 8.0], [0.05, _mape_lo]), _mape_lo, 0.05))
        _ovx_pen = -0.12 if ovx_level >= 60 else (-0.06 if ovx_level >= 40 else 0.0)

        # M5: 가격 모멘텀 충격 페널티 (OVX/뉴스보다 빠른 선행 신호)
        _mom_abs = abs(price_mom)
        _mom_pen = (-0.25 if _mom_abs >= 0.10 else
                    -0.15 if _mom_abs >= 0.06 else 0.0)

        _w_lo = 0.10 if _mom_abs >= 0.06 or ovx_level >= 60 else 0.30
        w_s = float(np.clip(w_s_base + mape_adj + _ovx_pen + _mom_pen, _w_lo, 0.70))
        log.info(f"    최종 앙상블 가중치(2모델 fallback): SARIMAX={w_s:.2f} XGB={1-w_s:.2f} "
                 f"(base={w_s_base:.2f} MAPE={mape_adj:+.2f} OVX={_ovx_pen:+.2f} "
                 f"MOM={_mom_pen:+.2f} ovx={ovx_level:.0f} mom={price_mom:.1%})")
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
# §ENS  ENSEMBLE & FORECAST  —  forecast_next_7days(), compute_ensemble_weights()
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
            last_exog = _exog_src[ecols].iloc[-1]
            fut_exog  = pd.DataFrame(np.zeros((7, len(ecols))), columns=ecols)  # 변화량 피처 미래 기대값=0
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

            # 분류기 채택 시: 회귀 크기 + 분류기 방향 결합
            if hasattr(model, 'predict_proba'):
                _reg_m  = info.get('_reg_model')
                _reg_sc = info.get('_reg_scaler')
                _reg_fs = info.get('_reg_features')
                if _reg_m is not None:
                    if _reg_sc is not None and _reg_fs is not None:
                        # HAR-Enhanced: 회귀 모델 전용 피처/스케일러로 magnitude 계산
                        _rf_avail = [f for f in _reg_fs if f in _feat_src.columns]
                        _reg_ls   = _reg_sc.transform(
                            _feat_src[_rf_avail].iloc[-1:].fillna(0).values)
                        _mag_ret  = float(_reg_m.predict(_reg_ls)[0])
                    else:
                        _mag_ret = float(_reg_m.predict(last_s)[0])
                else:
                    _mag_ret = 0.001
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
                pred_ret_d1 = float(model.predict(last_s)[0])  # log return 예측
            # D+1~7 가격 경로
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

    # ── 4번: 롤링 bias 교정 (live 예측 오차 기반 SARIMAX 체계적 오차 보정)
    if 'sarimax' in forecasts:
        try:
            _log_path = OUTPUT_DIR / 'prediction_log.csv'
            if _log_path.exists():
                _log_df = pd.read_csv(_log_path)
                if 'price_error' in _log_df.columns and 'type' in _log_df.columns:
                    _live_df = _log_df[_log_df['type'] == 'live']
                    _recent_err = _live_df.sort_values('date').tail(5)['price_error'].dropna()
                    if len(_recent_err) >= 3:
                        _bias = float(np.clip(_recent_err.mean(), -10.0, 10.0))
                        if abs(_bias) > 1.0:
                            forecasts['sarimax'] = forecasts['sarimax'] + _bias
                            log.info(f"    롤링 bias 교정: {_bias:+.2f}$ (최근 {len(_recent_err)}일 live 평균오차 기반)")
        except Exception as _be:
            log.debug(f"    롤링 bias 교정 실패: {_be}")

    # ── 동적 앙상블 가중치 (최근 backtest 오차 기반)
    _feat_ovx_src = full_df if full_df is not None else feature_df
    _live_ovx = float(_feat_ovx_src['OVX'].iloc[-1]) if 'OVX' in _feat_ovx_src.columns and _feat_ovx_src['OVX'].notna().any() else 0.0
    _live_mom5 = float(_feat_ovx_src['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _feat_ovx_src.columns and _feat_ovx_src['mom_5d'].notna().any() else 0.0
    w_sarimax, w_xgb = compute_ensemble_weights(ovx_level=_live_ovx, price_mom=_live_mom5)

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
            _reg_adj.update({'VAR': 1.6, 'XGB': 1.0, 'SARIMAX': 0.5})
            log.info(f"    레짐: 위기/고공포 (ovx_z={_cur_ovx_z:.2f} gpr_z={_cur_gpr_z:.2f} "
                     f"vol_pct={_vol_pct:.0%}) → SARIMAX↓↓ VAR↑↑")
        elif _cur_ovx_z > 1.0 or _vol_pct > 0.80:
            # 상승변동성: 중간 강도 SARIMAX 축소
            _reg_adj.update({'VAR': 1.3, 'XGB': 1.0, 'SARIMAX': 0.7})
            log.info(f"    레짐: 상승변동성 (ovx_z={_cur_ovx_z:.2f} vol_pct={_vol_pct:.0%}) → SARIMAX↓ VAR↑")
        # M5(stacking): 가격 모멘텀 충격 — OVX/GPR보다 빠른 선행 신호로 SARIMAX 추가 억제
        _stk_mom5 = float(_feat_src_r['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _feat_src_r.columns and _feat_src_r['mom_5d'].notna().any() else 0.0
        _stk_mom_abs = abs(_stk_mom5)
        if _stk_mom_abs >= 0.10:
            _reg_adj['SARIMAX'] = min(_reg_adj.get('SARIMAX', 1.0), 0.20)
            log.info(f"    M5(stk) 모멘텀충격 (mom5={_stk_mom5:.1%}) → SARIMAX ×0.20")
        elif _stk_mom_abs >= 0.06:
            _reg_adj['SARIMAX'] = min(_reg_adj.get('SARIMAX', 1.0), 0.40)
            log.info(f"    M5(stk) 모멘텀 (mom5={_stk_mom5:.1%}) → SARIMAX ×0.40")
        _adj_arr = np.array([_reg_adj.get(n, 1.0) for n in _stk_names]) * _stk_wts
        if _adj_arr.sum() > 0:
            _stk_wts = _adj_arr / _adj_arr.sum()

        # VAR 예측 생성
        _var_fc = None
        if 'VAR' in _stk_names and 'var' in results:
            try:
                _vr    = results['var']
                _vfit  = _vr['model']
                _vcols = _vr['cols']
                _vlag  = _vr['lag']
                _vhist = feature_df[_vcols].dropna().asfreq('B', method='ffill').dropna()
                _var_fc = _vfit.forecast(_vhist.values[-_vlag:], steps=7)[:, 0]
                forecasts['var_fc'] = _var_fc
            except Exception as _ve:
                log.warning(f"VAR 예측 실패: {_ve}")

        # D+2-7: SARIMAX 가중치 제거 (flat forecast 영향 배제), SARIMAX가 스택에 있으면 항상 적용
        _sx_idx = None
        if 'SARIMAX' in list(_stk_names):
            _sx_idx = list(_stk_names).index('SARIMAX')
            _wts_multistep = _stk_wts.copy()
            _wts_multistep[_sx_idx] = 0.0
            _rest = _wts_multistep.sum()
            if _rest > 0:
                _wts_multistep = _wts_multistep / _rest
            else:
                # SARIMAX 100% 케이스: D+2-7 = 나머지 모델 균등 분배
                _non_sx = [j for j in range(len(_stk_names)) if j != _sx_idx]
                _wts_multistep = np.zeros(len(_stk_names))
                for _nj in _non_sx:
                    _wts_multistep[_nj] = 1.0 / len(_non_sx)

        # VAR 예측 실패 시 VAR 가중치 제거 후 재정규화
        _stk_wts_live = _stk_wts.copy()
        if _var_fc is None and 'VAR' in list(_stk_names):
            _var_live_idx = list(_stk_names).index('VAR')
            _stk_wts_live[_var_live_idx] = 0.0
            if _stk_wts_live.sum() > 0:
                _stk_wts_live /= _stk_wts_live.sum()
            if _sx_idx is not None:
                _wts_multistep[_var_live_idx] = 0.0
                if _wts_multistep.sum() > 0:
                    _wts_multistep /= _wts_multistep.sum()
                else:
                    _xgb_live_idx = list(_stk_names).index('XGB') if 'XGB' in list(_stk_names) else 0
                    _wts_multistep = np.zeros(len(_stk_names))
                    _wts_multistep[_xgb_live_idx] = 1.0
        ensemble = np.zeros(7)
        for i in range(7):
            _pts = [_sx_fc[i], _xb_fc[i]]
            if _var_fc is not None:
                _pts.append(float(_var_fc[i]))
            # stack_names 순서에 맞게 패딩 (None 베이스는 center로 대체)
            while len(_pts) < len(_stk_names):
                _pts.append(_sx_fc[i])
            # D+1: 원래 stacking 가중치, D+2-7: SARIMAX 제외 후 재정규화
            _w = _wts_multistep if (i >= 1 and _sx_idx is not None) else _stk_wts_live
            ensemble[i] = float(np.dot(_pts[:len(_stk_names)], _w))
        log.info(f"    ✅ Stacking 앙상블 적용: D+1={ensemble[0]:.2f} "
                 f"(D+2-7: {'SARIMAX 제외' if _sx_idx is not None else '전체 가중치'})")
    elif 'sarimax' in forecasts and 'xgb' in forecasts:
        # stacking이 SARIMAX 대비 명시적으로 미채택된 경우 → SARIMAX 단독 사용
        # (fallback 역MAE 앙상블은 XGB/VAR 노이즈로 오히려 악화)
        _stk_rej = results.get('stacking_rejected', {})
        _sx_mae_fb = results.get('sarimax', {}).get('mae', 999.0)
        _naive_px_fb = results.get('naive_px_mae')
        _sx_reliable = (not _naive_px_fb) or (_sx_mae_fb <= _naive_px_fb)
        if _stk_rej and _stk_rej.get('mae', 999.0) > _sx_mae_fb:
            if _sx_reliable:
                # SARIMAX가 stacking과 M6-ABS 모두 통과 → SARIMAX 단독 사용
                ensemble = forecasts['sarimax'].copy()
                log.info(f"    stacking 미채택 fallback: SARIMAX 단독 (최적, M6-ABS pass) → D+1={ensemble[0]:.2f}")
            elif 'var' in results:
                # SARIMAX M6-ABS FAIL → VAR fallback
                try:
                    _vr_fb2 = results['var']
                    _vhfb2  = feature_df[_vr_fb2['cols']].dropna().asfreq('B', method='ffill').dropna()
                    ensemble = _vr_fb2['model'].forecast(_vhfb2.values[-_vr_fb2['lag']:], steps=7)[:, 0]
                    log.info(f"    stacking 미채택 fallback: VAR 단독 (SARIMAX M6-ABS FAIL) → D+1={ensemble[0]:.2f}")
                except Exception as _vfe2:
                    ensemble = forecasts['xgb'].copy()
                    log.info(f"    stacking 미채택 fallback: VAR 실패({_vfe2}) → XGBoost 단독 → D+1={ensemble[0]:.2f}")
            else:
                ensemble = forecasts['xgb'].copy()
                log.info(f"    stacking 미채택 fallback: XGBoost 단독 (SARIMAX M6-ABS FAIL) → D+1={ensemble[0]:.2f}")
        # VAR가 있으면 3모델 역MAE 가중 앙상블, 없으면 2모델
        elif 'var' in results:
            try:
                _vr2    = results['var']
                _vhist2 = feature_df[_vr2['cols']].dropna().asfreq('B', method='ffill').dropna()
                _var_fc2 = _vr2['model'].forecast(_vhist2.values[-_vr2['lag']:], steps=7)[:, 0]
                forecasts['var_fc'] = _var_fc2
                _mae_sx  = results['sarimax'].get('mae', 999.0)
                _mae_xb  = results['xgb_return'].get('mae', 999.0)
                _mae_vr  = _vr2.get('mae', 999.0)
                # M2: OVX 레짐 페널티 — 고변동기 SARIMAX 과신 방지
                _feat_r2 = full_df if full_df is not None else feature_df
                _ovx_abs = float(_feat_r2['OVX'].iloc[-1]) if 'OVX' in _feat_r2.columns and _feat_r2['OVX'].notna().any() else 0.0
                _sx_pen  = 0.50 if _ovx_abs >= 60 else (0.72 if _ovx_abs >= 40 else 1.0)
                # M3: 블랙스완 국면 페널티 — extreme_neg_news + jump_flag 동시 발생 시 SARIMAX 추가 억제
                _enews = float(_feat_r2['extreme_neg_news'].iloc[-1]) if 'extreme_neg_news' in _feat_r2.columns and _feat_r2['extreme_neg_news'].notna().any() else 0.0
                _jflag = float(_feat_r2['jump_flag'].iloc[-1]) if 'jump_flag' in _feat_r2.columns and _feat_r2['jump_flag'].notna().any() else 0.0
                _gpr_z = float(_feat_r2['gpr_zscore'].iloc[-1]) if 'gpr_zscore' in _feat_r2.columns and _feat_r2['gpr_zscore'].notna().any() else 0.0
                _cshock = float(_feat_r2['compound_shock_flag'].iloc[-1]) if 'compound_shock_flag' in _feat_r2.columns and _feat_r2['compound_shock_flag'].notna().any() else 0.0
                _bs_pen = (0.25 if _cshock >= 1.0 else
                           0.45 if (_enews >= 1.0 and _jflag >= 1.0) else
                           0.60 if (_enews >= 1.0 or _jflag >= 1.0 or _gpr_z > 2.0) else 1.0)
                _sx_pen = min(_sx_pen, _bs_pen)
                # M5: 가격 모멘텀 충격 페널티 — OVX/뉴스보다 빠른 선행 신호
                _mom5_val = float(_feat_r2['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _feat_r2.columns and _feat_r2['mom_5d'].notna().any() else 0.0
                _mom3_val = float(_feat_r2['WTI'].pct_change(3).dropna().iloc[-1]) if 'WTI' in _feat_r2.columns else _mom5_val
                _max_mom_abs = max(abs(_mom5_val), abs(_mom3_val))
                _mom_pen2 = (0.20 if _max_mom_abs >= 0.10 else
                             0.40 if _max_mom_abs >= 0.06 else 1.0)
                _sx_pen = min(_sx_pen, _mom_pen2)
                # M6: 최근 SARIMAX 실제 오차 기반 동적 페널티
                try:
                    _pl_rc = pd.read_csv(PRED_LOG_FILE)
                    _bt_rc = _pl_rc[(_pl_rc['type'] == 'backtest') & _pl_rc['sarimax_pred'].notna() & _pl_rc['actual_price'].notna()].tail(7)
                    if len(_bt_rc) >= 3:
                        _rc_sx_mae = (_bt_rc['sarimax_pred'] - _bt_rc['actual_price']).abs().mean()
                        _ov_sx_mae = max(_mae_sx, 0.1)
                        _rc_ratio = _rc_sx_mae / _ov_sx_mae
                        _rc_pen = (0.0 if _rc_ratio > 1.5 else
                                   0.70 if _rc_ratio > 1.2 else 1.0)
                        _sx_pen = min(_sx_pen, _rc_pen)
                        log.info(f"    M6 최근SARIMAX MAE: {_rc_sx_mae:.2f} / 전체: {_ov_sx_mae:.2f} "
                                 f"= {_rc_ratio:.2f}x → rc_pen={_rc_pen:.2f}"
                                 f"{'  [SARIMAX GATE-OUT]' if _rc_pen == 0.0 else ''}")
                except Exception:
                    pass
                # M6-절대 게이트: SARIMAX가 나이브 퍼시스턴스보다 나쁘면 즉시 배제
                _naive_px = results.get('naive_px_mae')
                if _naive_px and _mae_sx > _naive_px:
                    _sx_pen = 0.0
                    log.info(f"    M6-ABS GATE-OUT: SARIMAX MAE {_mae_sx:.2f} > Naive {_naive_px:.2f} → weight=0")
                _inv = np.array([1/_mae_sx * _sx_pen, 1/_mae_xb, 1/_mae_vr])
                _wts = _inv / _inv.sum()
                ensemble = (_wts[0] * forecasts['sarimax']
                            + _wts[1] * forecasts['xgb']
                            + _wts[2] * _var_fc2)
                log.info(f"    3모델 앙상블(InvMAE+OVX+BS+MOM+RC): SARIMAX×{_wts[0]:.2f} "
                         f"XGB×{_wts[1]:.2f} VAR×{_wts[2]:.2f} "
                         f"(ovx={_ovx_abs:.0f} sx_pen={_sx_pen:.2f} mom5={_mom5_val:.1%} "
                         f"enews={_enews:.0f} jump={_jflag:.0f} gpr_z={_gpr_z:.1f}) → D+1={ensemble[0]:.2f}")
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
    # 모멘텀 조건부 추가 보정: 강한 하락 추세 구간 과대 예측 패턴 완화
    _mom_src_bias = full_df if full_df is not None else feature_df
    _mom_bias = float(_mom_src_bias['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _mom_src_bias.columns else 0.0
    # 1일 수익률 기반 모멘텀 감속 감지: 누적 하락 대비 직전일 변화가 미미하면 보정 축소
    _mom_1d = float(_mom_src_bias['WTI'].pct_change(1).dropna().iloc[-1]) if 'WTI' in _mom_src_bias.columns else _mom_bias
    _mom_decel = abs(_mom_1d) / max(abs(_mom_bias), 1e-6)
    _mom_scale = float(np.clip(_mom_decel / 0.3, 0.3, 1.0)) if _mom_decel < 0.3 else 1.0
    if _mom_bias < -0.08:
        _mom_adj = -3.0 * _mom_scale
        bias += _mom_adj
        log.info(f"    모멘텀 하락({_mom_bias*100:.1f}%) 추가 보정: {_mom_adj:.2f}$ (감속비={_mom_decel:.3f} scale={_mom_scale:.2f})")
    elif _mom_bias < -0.05:
        _mom_adj = -2.0 * _mom_scale
        bias += _mom_adj
        log.info(f"    모멘텀 하락({_mom_bias*100:.1f}%) 추가 보정: {_mom_adj:.2f}$ (감속비={_mom_decel:.3f} scale={_mom_scale:.2f})")
    # live bias(±5$) + momentum(±3$) 합산 캡 — 설계 상한 이상 누적 방지
    _bias_before_cap = bias
    bias = float(np.clip(bias, -8.0, 8.0))
    if bias != _bias_before_cap:
        log.info(f"    bias 합산 캡: {_bias_before_cap:.2f} → {bias:.2f}$")
    # 모멘텀 방향 vs bias 방향 불일치 시 반감 (방향 전환 후 과소/과대 예측 방지)
    if bias != 0.0 and _mom_bias != 0.0:
        if (bias < 0 and _mom_bias > 0.02) or (bias > 0 and _mom_bias < -0.02):
            log.info(f"    bias-모멘텀 방향 불일치: bias={bias:.2f}x0.5 (mom={_mom_bias*100:.1f}%)")
            bias *= 0.5
    if bias != 0.0:
        _bias_decay = np.array([1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15])[:len(ensemble)]
        ensemble = ensemble + bias * _bias_decay
        _bias_per_day = np.round(bias * _bias_decay, 3)
    else:
        _bias_per_day = np.zeros(len(ensemble))

    # ── B: D+2-7 모멘텀 드리프트 (추세 국면 예측 경로 방향성 반영, D+1 제외)
    if abs(_mom_bias) > 0.03 and len(ensemble) > 1:
        _drift_w = np.array([0.0, 0.30, 0.25, 0.18, 0.12, 0.08, 0.05])[:len(ensemble)]
        _drift = np.clip(_mom_bias * last_price * _drift_w, -4.0, 4.0)
        ensemble = ensemble + _drift
        log.info(f"    모멘텀 드리프트(D+2-7): mom={_mom_bias*100:.1f}% "
                 f"→ D+2 {_drift[1]:+.2f}, D+7 {_drift[-1]:+.2f}")

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

    # ── A3.5: 직접 D+2-7 XGBoost 예측 주입 (MASE < 1.0인 호라이즌)
    if 'xgb_ms' in results:
        _ms_info  = results['xgb_ms']
        _feat_src = full_df if full_df is not None else feature_df
        _last_row = _feat_src.iloc[-1]
        for _h in range(2, 8):
            _h_idx = _h - 1
            if _h in _ms_info['models'] and _h_idx < len(ensemble) and _ms_info['mase'].get(_h, 1.0) < 1.0:
                _md = _ms_info['models'][_h]
                _flist = [f for f in _md['features'] if f in _feat_src.columns]
                _X_live = _md['scaler'].transform(_last_row[_flist].values.reshape(1, -1))
                _ret_h  = float(np.clip(_md['model'].predict(_X_live)[0], -0.3, 0.3))
                ensemble[_h_idx] = last_price * np.exp(_ret_h)

    # ── A4: D+2-7 구조적 persistence 블렌딩 (MASE 기반 동적 조정)
    _ms_blend = np.array([0.0, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98])[:len(ensemble)]
    if 'xgb_ms' in results:
        _ms_mase_d = results['xgb_ms']['mase']
        for _bi in range(1, len(_ms_blend)):
            _h_b = _bi + 1
            if _h_b in _ms_mase_d and _ms_mase_d[_h_b] < 1.0:
                _ms_blend[_bi] = float(min(_ms_blend[_bi], np.clip(_ms_mase_d[_h_b], 0.0, 0.98)))
    # 모멘텀 바이어스: persistence 기준점을 최근 추세로 보정 (리스크 방향성 정보 제공)
    _mom_src      = full_df if full_df is not None else feature_df
    _mom_val      = float(_mom_src['mom_5d'].dropna().iloc[-1]) if 'mom_5d' in _mom_src.columns else 0.0
    _daily_mom    = np.clip(_mom_val / 5, -0.02, 0.02)
    _mom_days     = np.arange(len(ensemble))
    _persist_base = last_price * (1 + _daily_mom * _mom_days * 0.3)
    ensemble = ensemble * (1 - _ms_blend) + _persist_base * _ms_blend
    _d7_adj = float(_persist_base[-1] - last_price) if len(_persist_base) > 1 else 0.0
    log.info(f"    D+2-7 Persistence 블렌딩: D+2={_ms_blend[1]:.0%} ~ D+7={_ms_blend[min(6,len(_ms_blend)-1)]:.0%} "
             f"(모멘텀: {_daily_mom*100:+.2f}%/일, D+7기준점 {_d7_adj:+.2f}$)")

    # ── 5번: 감성 충격 보정 (앙상블 최종 오버레이 — 모든 모델 가중치 희석 없이 완전 반영)
    # 원시 단순평균 사용 — _wavg_sent weighted avg는 극단 기사 희석시킴
    try:
        _nc_file_sh = OUTPUT_DIR / 'guardian_news_cache.csv'
        if _nc_file_sh.exists():
            _nc_sh = pd.read_csv(_nc_file_sh)
            _nc_sh['date'] = pd.to_datetime(_nc_sh['date'])
            _raw_d_sh  = _nc_sh.groupby('date')['finbert_score'].mean().sort_index()
            _raw_c3_sh = _raw_d_sh.diff(3)
            _today_key = _raw_d_sh.index[-1]
            _chg3_sh   = float(_raw_c3_sh.loc[_today_key]) if _today_key in _raw_c3_sh.index and pd.notna(_raw_c3_sh.loc[_today_key]) else 0.0
            _sent_sh   = float(_raw_d_sh.loc[_today_key])  if _today_key in _raw_d_sh.index  else 0.0
            _ens_chg_pct = (ensemble[0] - last_price) / last_price if last_price != 0 else 0.0
            if _chg3_sh < -0.15 and _sent_sh < 0.10 and _ens_chg_pct > -0.02:
                # ±3% 캡: FinBERT 극단값이 예측 전체를 왜곡하지 않도록 제한
                _shock_adj = float(np.clip(_chg3_sh * 20.0, -last_price * 0.03, last_price * 0.03))
                ensemble = ensemble + _shock_adj
                log.info(f"    ⚡ 감성 충격 보정: raw_chg3={_chg3_sh:.3f} raw_sent={_sent_sh:.3f} → {_shock_adj:+.2f}$")
    except Exception as _se:
        log.debug(f"    감성 충격 보정 실패: {_se}")

    # ── A5: 충격 후 레짐 감지 — 최근 5영업일 내 단일 일간 변동 ±8% 이상 시 D+1 변동폭 ±8% 캡
    # 충격 직후 극단 피처값에 모델이 과반응하는 오차 완충 (4/8 충격 → 4/13 오차 +$9 사례 기반)
    try:
        _wti_regime = full_df['WTI'].dropna()
        _regime_chg = _wti_regime.pct_change().abs()
        _shock_in_window = float(_regime_chg.iloc[-5:].max()) if len(_regime_chg) >= 5 else 0.0
        _SHOCK_REGIME_THRESHOLD = 0.14  # 단일 일간 변동 14% 이상 (8%로 했을 때 정확한 예측까지 캡 막음)
        _SHOCK_REGIME_CAP = 0.08        # 충격 레짐에서 D+1 변동폭 ±8%
        if _shock_in_window >= _SHOCK_REGIME_THRESHOLD:
            _cap_lo = last_price * (1 - _SHOCK_REGIME_CAP)
            _cap_hi = last_price * (1 + _SHOCK_REGIME_CAP)
            _d1_before = ensemble[0]
            ensemble[0] = float(np.clip(ensemble[0], _cap_lo, _cap_hi))
            if ensemble[0] != _d1_before:
                log.info(f"    🛡 충격 레짐 캡: D+1 {_d1_before:.2f}$ → {ensemble[0]:.2f}$ "
                         f"(최근5일 최대변동 {_shock_in_window*100:.1f}%, 캡±{_SHOCK_REGIME_CAP*100:.0f}%)")
            else:
                log.info(f"    충격 레짐 감지(최근5일 최대변동 {_shock_in_window*100:.1f}%) — D+1 캡 미발동")
    except Exception as _re:
        log.debug(f"    충격 레짐 감지 실패: {_re}")

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

    lower_75ci = upper_75ci = None
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
            lower_75ci = ensemble + _spread_lo * np.power(t, 0.4)
            upper_75ci = ensemble + _spread_hi * np.power(t, 0.4)
            lower_75ci = np.clip(lower_75ci, last_price * 0.80, last_price * 1.20)
            upper_75ci = np.clip(upper_75ci, last_price * 0.80, last_price * 1.20)
            log.info(f"    신뢰구간 Q10/Q90 적용: D+1 [{lower_75ci[0]:.2f}, {upper_75ci[0]:.2f}]")
        except Exception as _qce:
            log.warning(f"    분위 신뢰구간 계산 실패({_qce}) → 변동성 폴백")

    if lower_75ci is None:
        # GARCH 조건부 변동성 우선 (더 정확한 변동성 클러스터링 반영)
        # GARCH(1,1) 다단계 조건부 분산 예측 (최우선) — h_{t+k} 각 스텝별 독립 분산
        _garch_res_fc = (aux or {}).get('garch_model') or results.get('garch_vol', {}).get('model')
        _used_garch_fc = False
        if _garch_res_fc is not None:
            try:
                _fc7 = _garch_res_fc.forecast(horizon=7, reindex=False)
                _h7  = np.sqrt(_fc7.variance.values[-1, :]) / 100   # % → 소수 변환
                ci_half = last_price * _h7 * 1.15
                lower_75ci = ensemble - ci_half
                upper_75ci = ensemble + ci_half
                _used_garch_fc = True
                log.info(f"    GARCH(1,1) 다단계 CI: D+1 ±{ci_half[0]:.2f}$ → D+7 ±{ci_half[6]:.2f}$")
            except Exception as _gfe:
                log.warning(f"    GARCH 다단계 예측 실패({_gfe}) → vol 폴백")
        if not _used_garch_fc:
            _src_vol = full_df if full_df is not None else feature_df
            if 'garch_vol' in _src_vol.columns:
                _gv_ser = _src_vol['garch_vol'].dropna()
                _garch_v = float(_gv_ser.iloc[-1]) if len(_gv_ser) > 0 else 1e-4
                ci_half  = last_price * max(_garch_v, 1e-4) * 1.15 * np.power(t, 0.4)
            else:
                _v5_ser = _src_vol['vol_5d'].dropna() if 'vol_5d' in _src_vol.columns else pd.Series(dtype=float)
                recent_vol = float(_v5_ser.iloc[-1]) if len(_v5_ser) > 0 else 0.015
                ci_half    = ensemble * recent_vol * 1.15 * np.power(t, 0.4)
            lower_75ci = ensemble - ci_half
            upper_75ci = ensemble + ci_half

    # CI 경험적 보정: live Q75 → stacking 백테스트 Q75 → SARIMAX 백테스트 Q75 순 폴백
    _ci_q75 = (
        (aux or {}).get('live_ci_q75')
        or (results.get('stacking') or {}).get('ci_calib_q75')
        or (results.get('sarimax') or {}).get('ci_calib_q75')
    )
    if _ci_q75 is not None and lower_75ci is not None:
        _d1_half = (float(upper_75ci[0]) - float(lower_75ci[0])) / 2
        if _d1_half > 0.1:
            _calib = float(np.clip(_ci_q75 / _d1_half, 0.5, 2.5))
            if abs(_calib - 1.0) > 0.05:
                _mid     = (lower_75ci + upper_75ci) / 2
                _half_ci = (upper_75ci - lower_75ci) / 2
                lower_75ci = _mid - _half_ci * _calib
                upper_75ci = _mid + _half_ci * _calib
                log.info(f"    CI 경험적 보정 ×{_calib:.2f} (Q75={_ci_q75:.2f}$)")

    fc_df = pd.DataFrame({
        'date':            fc_dates.strftime('%Y-%m-%d'),
        'forecast_price':  np.round(ensemble,    2),
        'lower_75ci':      np.round(lower_75ci,  2),
        'upper_75ci':      np.round(upper_75ci,  2),
        'bias_correction': _bias_per_day,
    })
    if 'sarimax' in forecasts:
        fc_df['sarimax_forecast'] = np.round(forecasts['sarimax'], 2)
    if 'xgb' in forecasts:
        fc_df['xgb_forecast'] = np.round(forecasts['xgb'], 2)
    if 'var_fc' in forecasts and forecasts['var_fc'] is not None:
        fc_df['var_forecast'] = np.round(forecasts['var_fc'], 2)
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
            fc_df['lower_75ci'] = (_fp - (_fp - fc_df['lower_75ci']) * _ci_expand).round(2)
            fc_df['upper_75ci'] = (_fp + (fc_df['upper_75ci'] - _fp) * _ci_expand).round(2)
            log.warning(f"    ⚠ 모델 불일치 과다(D+1 std={_d1_std:.2f}$) → CI ×{_ci_expand}")

    # VaR: 정규분포 5%/95% 분위수 (헤지 기준선 — 단방향 꼬리 리스크)
    _var_src_v = full_df if full_df is not None else feature_df
    _vol5d_now = float(_var_src_v['vol_5d'].dropna().iloc[-1]) if 'vol_5d' in _var_src_v.columns else 0.015
    _t_var = np.arange(1, 8)
    fc_df['var_5pct']  = (fc_df['forecast_price'] - last_price * _vol5d_now * 1.645 * np.sqrt(_t_var)).round(2)
    fc_df['var_95pct'] = (fc_df['forecast_price'] + last_price * _vol5d_now * 1.645 * np.sqrt(_t_var)).round(2)
    log.info(f"    VaR(5%) D+1={fc_df['var_5pct'].iloc[0]:.2f}$  D+7={fc_df['var_5pct'].iloc[min(6, len(fc_df)-1)]:.2f}$")

    _atomic_csv(fc_df, OUTPUT_DIR / 'forecast_7days.csv', index=False)
    log.info("    forecast_7days.csv 저장")

    # ── D+1 예측 스냅샷 누적 저장 (run_date당 1행 append-only)
    try:
        _run_date_str = pd.Timestamp.today().strftime('%Y-%m-%d')
        _snap_path = OUTPUT_DIR / 'forecast_snapshots.csv'
        _d1 = fc_df.iloc[0]
        _d27_cols = {}
        for _di in range(1, min(7, len(fc_df))):
            _row_i = fc_df.iloc[_di]
            _d27_cols[f'd{_di+1}_date']     = str(_row_i['date'])
            _d27_cols[f'd{_di+1}_forecast'] = round(float(_row_i['forecast_price']), 2)
        _snap_row = pd.DataFrame([{
            'run_date':       _run_date_str,
            'forecast_date':  str(_d1['date']),
            'forecast_price': round(float(_d1['forecast_price']), 2),
            'lower_75ci':     round(float(_d1['lower_75ci']), 2) if pd.notna(_d1.get('lower_75ci')) else None,
            'upper_75ci':     round(float(_d1['upper_75ci']), 2) if pd.notna(_d1.get('upper_75ci')) else None,
            'sarimax_pred':   round(float(_d1['sarimax_forecast']), 2) if pd.notna(_d1.get('sarimax_forecast')) else None,
            **_d27_cols,
        }])
        if _snap_path.exists():
            _snap_existing = pd.read_csv(_snap_path)
            _snap_existing = _snap_existing[_snap_existing['run_date'] != _run_date_str]
            _snap_row = pd.concat([_snap_existing, _snap_row], ignore_index=True)
        # 실제가 채우기: full_df WTI + prediction_log 두 소스로 각 행의 actual 컬럼 업데이트
        try:
            _price_str_map = {}
            if full_df is not None:
                _price_str_map.update({
                    str(k.date()): round(float(v), 2)
                    for k, v in full_df['WTI'].dropna().items()
                })
            # prediction_log의 actual_price도 역채움 소스에 포함 (full_df 없는 날짜 보완)
            if PRED_LOG_FILE.exists():
                try:
                    _pl_src = pd.read_csv(PRED_LOG_FILE)[['date', 'actual_price']].dropna()
                    _price_str_map.update({
                        str(r['date']): round(float(r['actual_price']), 2)
                        for _, r in _pl_src.iterrows()
                    })
                except Exception:
                    pass
            if _price_str_map:
                for _cd, _ca in [('forecast_date', 'actual_price')] + [
                    (f'd{_i}_date', f'd{_i}_actual') for _i in range(2, 8)
                ]:
                    if _cd in _snap_row.columns:
                        _mapped = _snap_row[_cd].map(_price_str_map)
                        if _ca in _snap_row.columns:
                            # fillna로 기존 actual 보존 — 덮어쓰기 금지
                            _snap_row[_ca] = _snap_row[_ca].fillna(_mapped)
                        else:
                            _snap_row[_ca] = _mapped
        except Exception as _ae:
            log.warning(f"    스냅샷 실제가 채우기 실패: {_ae}")
        _atomic_csv(_snap_row, _snap_path, index=False)
        log.info(f"    forecast_snapshots.csv 저장 (run={_run_date_str} → {str(_d1['date'])} 예측={round(float(_d1['forecast_price']),2)})")
    except Exception as _se:
        log.warning(f"    스냅샷 저장 실패: {_se}")

    return fc_df


# ─────────────────────────────────────────────────────────────────────────────
# §LOG  PREDICTION LOGGING  —  save_prediction_log(), live performance monitoring
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
    sx  = results.get('sarimax', {})
    xg  = results.get('xgb_har', {})
    stk = results.get('stacking', {})

    # 감성 충격 보정용 원시 감성 조회 (weighted avg 아닌 단순 mean — 희석 방지)
    _raw_sent_lkp = {}
    _raw_chg3_lkp = {}
    try:
        _nc_file = OUTPUT_DIR / 'guardian_news_cache.csv'
        if _nc_file.exists():
            _nc_df = pd.read_csv(_nc_file)
            _nc_df['date'] = pd.to_datetime(_nc_df['date'])
            _raw_daily = _nc_df.groupby('date')['finbert_score'].mean().sort_index()
            _raw_chg3s = _raw_daily.diff(3)
            _raw_sent_lkp = {k: float(v) for k, v in _raw_daily.items() if pd.notna(v)}
            _raw_chg3_lkp = {k: float(v) for k, v in _raw_chg3s.items() if pd.notna(v)}
    except Exception:
        pass

    if 'pred_price_test' in sx and 'test_dates' in sx:
        dates         = sx['test_dates']
        pred_prices   = sx['pred_price_test']
        actual_prices = sx['actual_price_test']
        pred_vols     = xg.get('pred_rv_test',   np.full(len(dates), np.nan))
        actual_vols   = xg.get('actual_rv_test', np.full(len(dates), np.nan))
        stk_preds     = stk.get('pred_price_test')  # None when Stacking not adopted

        for i, dt in enumerate(dates):
            pp = float(pred_prices[i])
            ap = float(actual_prices[i])
            pv = float(pred_vols[i])   if i < len(pred_vols)   else np.nan
            av = float(actual_vols[i]) if i < len(actual_vols) else np.nan
            sp = round(float(stk_preds[i]), 2) if stk_preds is not None and i < len(stk_preds) else None

            # 감성 충격 보정 (원시 단순평균 사용 — _wavg_sent 희석 방지)
            if i > 0 and _raw_chg3_lkp:
                try:
                    _prev_dt_bt = pd.Timestamp(dates[i - 1])
                    _chg3_bt    = _raw_chg3_lkp.get(_prev_dt_bt, 0.0)
                    _sent_bt    = _raw_sent_lkp.get(_prev_dt_bt, 0.0)
                    _prev_ap    = float(actual_prices[i - 1])
                    _sx_chg     = (pp - _prev_ap) / _prev_ap if _prev_ap != 0 else 0.0
                    if _chg3_bt < -0.15 and _sent_bt < 0.10 and _sx_chg > -0.02:
                        pp = pp + _chg3_bt * 20.0
                except Exception:
                    pass

            # 충격 후 레짐 캡 — 최근 5영업일 내 단일 일간 변동 ≥14% 시 D+1 변동폭 ±8% 캡
            if i > 0:
                try:
                    _bt_window = np.array(actual_prices[max(0, i - 6):i], dtype=float)
                    if len(_bt_window) >= 2:
                        _bt_chgs = np.abs(np.diff(_bt_window) / _bt_window[:-1])
                        _bt_shock = float(_bt_chgs.max())
                    else:
                        _bt_shock = 0.0
                    if _bt_shock >= 0.14:
                        _bt_last = float(actual_prices[i - 1])
                        pp = float(np.clip(pp, _bt_last * 0.92, _bt_last * 1.08))
                except Exception:
                    pass

            p_err     = round(ap - pp, 2)
            p_err_pct = round((ap - pp) / ap * 100, 2) if ap != 0 else None
            v_err     = round(av - pv, 5) if not (np.isnan(av) or np.isnan(pv)) else None
            stk_err   = round(ap - sp, 2) if sp is not None else None

            pred_dir = actual_dir = dir_correct = None
            if i > 0:
                prev_ap = float(actual_prices[i - 1])
                _pred_p = sp if sp is not None else pp
                pred_dir   = 'UP' if _pred_p > prev_ap else ('DOWN' if _pred_p < prev_ap else 'FLAT')
                actual_dir = 'UP' if ap > prev_ap else ('DOWN' if ap < prev_ap else 'FLAT')
                dir_correct = 1 if pred_dir == actual_dir else 0

            bt_rows.append({
                'date':            dt.strftime('%Y-%m-%d'),
                'sarimax_pred':    round(pp, 2),
                'stacking_pred':   sp,
                'actual_price':    round(ap, 2),
                'price_error':     p_err,
                'stacking_error':  stk_err,
                'price_error_pct': p_err_pct,
                'xgb_pred_vol':    round(pv, 5) if not np.isnan(pv) else None,
                'actual_vol_5d':   round(av, 5) if not np.isnan(av) else None,
                'vol_error':       v_err,
                'type':            'backtest',
                'pred_direction':  pred_dir,
                'actual_direction': actual_dir,
                'dir_correct':     dir_correct,
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
                _needs_actual   = pd.isna(row.get('actual_price')) and dt in price_src.index
                _needs_stk_err  = (pd.notna(row.get('actual_price'))
                                   and pd.isna(row.get('stacking_error'))
                                   and pd.notna(row.get('stacking_pred')))
                _needs_direction = (pd.notna(row.get('actual_price'))
                                    and pd.isna(row.get('actual_direction')))
                if (_needs_actual or _needs_stk_err or _needs_direction) and dt in price_src.index:
                    ap = float(price_src.loc[dt, 'WTI'])
                    if not _needs_actual and pd.notna(row.get('actual_price')):
                        ap = float(row['actual_price'])  # confirmed actual 우선 (rollover 오염 방지)
                    pp = float(row['sarimax_pred'])
                    av = float(feature_df.loc[dt, 'RV_5d']) if (dt in feature_df.index and 'RV_5d' in feature_df.columns) else np.nan
                    pv = float(row['xgb_pred_vol']) if pd.notna(row.get('xgb_pred_vol')) else np.nan

                    if _needs_actual:
                        old_live.at[idx, 'actual_price']    = round(ap, 2)
                        old_live.at[idx, 'price_error']     = round(ap - pp, 2)
                        old_live.at[idx, 'price_error_pct'] = round((ap - pp) / ap * 100, 2) if abs(ap) > 0.01 else None
                    _sp_live = row.get('stacking_pred')
                    # stacking_error: 이미 확정된 actual_price 우선 사용 (price_src는 데이터 갱신으로 값 달라질 수 있음)
                    _ap_for_err = float(row['actual_price']) if pd.notna(row.get('actual_price')) else ap
                    if pd.notna(_sp_live) and pd.isna(row.get('stacking_error')):
                        old_live.at[idx, 'stacking_error'] = round(_ap_for_err - float(_sp_live), 2)
                    if not np.isnan(av):
                        old_live.at[idx, 'actual_vol_5d'] = round(av, 5)
                    if not (np.isnan(av) or np.isnan(pv)):
                        old_live.at[idx, 'vol_error'] = round(av - pv, 5)
                    # 방향성 추적 (actual 확정 시 계산)
                    _prev_bday_dir = dt - pd.tseries.offsets.BDay(1)
                    if _prev_bday_dir in price_src.index:
                        try:
                            _pc = float(price_src.loc[_prev_bday_dir, 'WTI'])
                            _pd_dir = 'UP' if pp > _pc else ('DOWN' if pp < _pc else 'FLAT')
                            _ad_dir = 'UP' if ap > _pc else ('DOWN' if ap < _pc else 'FLAT')
                            if pd.isna(row.get('pred_direction', np.nan)):
                                old_live.at[idx, 'pred_direction'] = _pd_dir
                            old_live.at[idx, 'actual_direction'] = _ad_dir
                            old_live.at[idx, 'dir_correct'] = 1 if _pd_dir == _ad_dir else 0
                        except Exception:
                            pass
            except Exception:
                pass

        live_rows = old_live.to_dict('records')

    # ── 누락일 gap-fill (직전 실행의 7일 예측 활용) ──────────────────
    today = pd.Timestamp.today().normalize()
    existing_dates = {r.get('date') for r in live_rows if r.get('date') is not None}

    if prev_fc_df is not None and not prev_fc_df.empty:
        try:
            prev_lookup = {str(row['date']): float(row['forecast_price'])
                           for _, row in prev_fc_df.iterrows()}
            # 마지막 actual 확정일 기준 (미래 dated entry 무시)
            _confirmed = [pd.Timestamp(r['date']) for r in live_rows
                          if r.get('date') and pd.notna(r.get('actual_price'))
                          and pd.Timestamp(r['date']) <= today]
            last_confirmed = max(_confirmed) if _confirmed else today - pd.tseries.offsets.BDay(10)
            gap_dates = pd.bdate_range(
                start=last_confirmed + timedelta(days=1),
                end=today,
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
                    'stacking_pred':   gap_pred,
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

    # ── 예측 대상일 live entry 추가 / 덮어쓰기 ──────────────────────
    # entry_date = 예측 대상일 (fc_df 첫 번째 날짜 = 다음 영업일)
    # 같은 날짜로 여러 번 실행 시 마지막 값으로 덮어쓰기
    if fc_df is not None and len(fc_df) > 0:
        entry_date_str = str(fc_df['date'].iloc[0])
        entry_pred = round(float(fc_df['forecast_price'].iloc[0]), 2)

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
            if 'lower_75ci' in fc_df.columns:
                _entry_lower_ci = round(float(_ci_row['lower_75ci'].iloc[0]), 2)
                _entry_upper_ci = round(float(_ci_row['upper_75ci'].iloc[0]), 2)
        except Exception:
            pass

        # 방향성 예측 (pred - 전일종가 부호)
        _pred_dir_e = _actual_dir_e = _dir_correct_e = None
        try:
            _entry_ts_dt = pd.Timestamp(entry_date_str)
            _wti_hist = price_src['WTI'].dropna() if 'WTI' in price_src.columns else pd.Series(dtype=float)
            _wti_hist = _wti_hist[_wti_hist.index < _entry_ts_dt]
            if len(_wti_hist) > 0:
                _prev_c = float(_wti_hist.iloc[-1])
                _pred_dir_e = 'UP' if entry_pred > _prev_c else ('DOWN' if entry_pred < _prev_c else 'FLAT')
                if entry_actual is not None:
                    _actual_dir_e = 'UP' if entry_actual > _prev_c else ('DOWN' if entry_actual < _prev_c else 'FLAT')
                    _dir_correct_e = 1 if _pred_dir_e == _actual_dir_e else 0
        except Exception:
            pass

        _stk_err_entry = round(float(entry_actual) - float(entry_pred), 2) if entry_actual is not None else None
        new_entry = {
            'date':            entry_date_str,
            'sarimax_pred':    entry_pred,
            'stacking_pred':   entry_pred,
            'actual_price':    entry_actual,
            'price_error':     entry_error,
            'stacking_error':  _stk_err_entry,
            'price_error_pct': entry_error_pct,
            'lower_75ci':      _entry_lower_ci,
            'upper_75ci':      _entry_upper_ci,
            'xgb_pred_vol':    xgb_pred_v,
            'actual_vol_5d':   None,
            'vol_error':       None,
            'type':            'live',
            'model_version':   MODEL_VERSION,
            'pred_direction':  _pred_dir_e,
            'actual_direction': _actual_dir_e,
            'dir_correct':     _dir_correct_e,
        }

        # 같은 날짜 entry가 있으면 최신 실행값으로 덮어쓰기
        today_idx = next((i for i, r in enumerate(live_rows) if r.get('date') == entry_date_str), None)
        if today_idx is not None:
            live_rows[today_idx] = new_entry
        else:
            live_rows.append(new_entry)

    live_df = pd.DataFrame(live_rows) if live_rows else pd.DataFrame()

    # ── 합치기 & 저장 ────────────────────────────────────────────────
    if not live_df.empty:
        combined = pd.concat([bt_df, live_df], ignore_index=True)
        # live 날짜와 겹치는 backtest 행 제거 — live 행 우선 (look-ahead 편향 방지)
        _live_dates = set(live_df['date'].astype(str))
        combined = combined[~((combined['type'] == 'backtest') & combined['date'].astype(str).isin(_live_dates))]
        combined = combined.reset_index(drop=True)
    else:
        combined = bt_df
    _atomic_csv(combined, PRED_LOG_FILE, index=False)
    n_live = len(live_df)
    n_filled = int(live_df['actual_price'].notna().sum()) if not live_df.empty else 0
    log.info(f"    prediction_log.csv 저장 "
             f"(백테스트 {len(bt_df)}일 | 실시간 {n_live}건 중 {n_filled}건 확인 완료)")


def analyze_live_gap() -> dict:
    """Backtest-Live 괴리 원인 분석.

    출력 파일:
      output/live_gap_monthly.csv  — 월별 MAE 추이 (backtest vs live)
      output/live_gap_spikes.csv   — 오차 급등 날짜 top-N (원인 파악용)
    반환: {'monthly': DataFrame, 'spikes': DataFrame, 'summary': dict}
    """
    if not PRED_LOG_FILE.exists():
        return {}
    try:
        df = pd.read_csv(PRED_LOG_FILE, parse_dates=['date'])
        # backtest: stacking_error 우선(동일 모델 기준 비교); live: price_error(=stacking 예측 오차)
        if 'stacking_error' in df.columns:
            _is_bt  = df['type'] == 'backtest'
            _has_se = df['stacking_error'].notna()
            df.loc[_is_bt & _has_se, 'price_error'] = df.loc[_is_bt & _has_se, 'stacking_error']
        df = df[df['price_error'].notna()].copy()
        df['abs_error'] = df['price_error'].abs()

        bt   = df[df['type'] == 'backtest']
        live = df[df['type'].isin(['live', 'gap'])]

        bt_mae   = float(bt['abs_error'].mean())   if len(bt)   > 0 else float('nan')
        live_mae = float(live['abs_error'].mean()) if len(live) > 0 else float('nan')

        # ── 월별 MAE 추이
        monthly_rows = []
        for _type, _grp in [('backtest', bt), ('live', live)]:
            if len(_grp) == 0:
                continue
            for _m, _mg in _grp.groupby(_grp['date'].dt.to_period('M')):
                monthly_rows.append({
                    'month': str(_m), 'type': _type,
                    'mae':  round(float(_mg['abs_error'].mean()), 4),
                    'bias': round(float(_mg['price_error'].mean()), 4),
                    'n':    len(_mg),
                    'max_error': round(float(_mg['abs_error'].max()), 4),
                })
        monthly_df = pd.DataFrame(monthly_rows)
        _atomic_csv(monthly_df, OUTPUT_DIR / 'live_gap_monthly.csv', index=False)

        # ── 오차 급등 날짜 (|error| > 2 × backtest_mae)
        thresh = bt_mae * 2.0 if bt_mae > 0 else 5.0
        spikes = live[live['abs_error'] > thresh].copy()
        spikes = spikes.sort_values('abs_error', ascending=False).head(20)
        spike_cols = ['date', 'actual_price', 'stacking_pred', 'sarimax_pred', 'price_error',
                      'abs_error', 'xgb_pred_vol', 'actual_vol_5d']
        spike_out  = spikes[[c for c in spike_cols if c in spikes.columns]]
        _atomic_csv(spike_out, OUTPUT_DIR / 'live_gap_spikes.csv', index=False)

        _ratio_val = round(live_mae / max(bt_mae, 1e-6), 3) if len(live) >= 5 else float('nan')
        summary = {
            'bt_mae': round(bt_mae, 4), 'live_mae': round(live_mae, 4),
            'ratio':  _ratio_val,
            'n_live': len(live),
            'n_spikes': len(spikes),
            'spike_thresh': round(thresh, 4),
        }
        _ratio_str = f"{_ratio_val:.2f}×" if not np.isnan(_ratio_val) else f"n/a(n={len(live)}<5)"
        log.info(f"    [Gap Analysis] bt_MAE={bt_mae:.4f}  live_MAE={live_mae:.4f}  "
                 f"ratio={_ratio_str}  spikes={len(spikes)}건 (>{thresh:.2f}$)")
        log.info(f"    → output/live_gap_monthly.csv, live_gap_spikes.csv 저장")
        return {'monthly': monthly_df, 'spikes': spike_out, 'summary': summary}
    except Exception as _ge:
        log.warning(f"    Gap Analysis 실패({_ge})")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# §ALT  ALERTS & MONITORING  —  send_risk_alert(), monitor_rss_alerts()
# ─────────────────────────────────────────────────────────────────────────────

def _fire_and_forget(fn, *args, **kwargs):
    """이메일 전송을 비동기 실행 — 파이프라인 블로킹 방지. non-daemon: 프로세스 종료 전 완료 보장."""
    import threading
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=False)
    t.start()


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
[유가 리스크 시스템] {r['emoji']} {r['label']} 경보 — {today}

━━━━━━━━━━━━━━━━━━━━━━━━━
리스크 레벨  : {r['emoji']} {r['label']}
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
        msg['Subject'] = f"[유가 리스크] {r['emoji']} {r['label']} 경보 — WTI ${wti:.2f}"
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


def _send_single_alert(to_email: str, risk_signal: dict, fc_df) -> bool:
    """단일 수신자에게 리스크 알람 이메일 발송."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        level   = risk_signal.get('risk_level', '')
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
[유가 리스크 시스템] {r['emoji']} {r['label']} 경보 — {today}

━━━━━━━━━━━━━━━━━━━━━━━━━
리스크 레벨  : {r['emoji']} {r['label']}
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
        msg['To']      = to_email
        msg['Subject'] = f"[유가 리스크] {r['emoji']} {r['label']} 경보 — WTI ${wti:.2f}"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())

        log.info(f"    📧 알림 발송 → {to_email}")
        return True
    except Exception as exc:
        log.warning(f"    이메일 발송 실패 ({to_email}): {exc}")
        return False


def send_alerts_to_subscribers(risk_signal: dict, fc_df) -> None:
    """auth_config.yaml의 구독자 목록을 읽어 플랜별 알람 발송."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return
    level = risk_signal.get('risk_level', '')
    if level not in ('SURGE_RISK', 'DROP_RISK', 'CRITICAL'):
        return
    try:
        _cfg_path = Path(__file__).parent / 'config' / 'auth_config.yaml'
        with open(_cfg_path, encoding='utf-8') as _f:
            _cfg = yaml.safe_load(_f) or {}
        users = _cfg.get('credentials', {}).get('usernames', {})
        score = float(risk_signal.get('risk_score', 0))
        for _uname, _udata in users.items():
            if not isinstance(_udata, dict):
                continue
            plan  = _udata.get('plan', 'free')
            email = _udata.get('email', '')
            if plan not in ('standard', 'pro') or not email:
                continue
            threshold = float(_udata.get('alert_threshold', 0.7))
            if score < threshold:
                continue
            _send_single_alert(email, risk_signal, fc_df)
    except Exception as exc:
        log.warning(f"    구독자 알람 발송 오류: {exc}")


def monitor_rss_alerts() -> dict:  # noqa: dead — 향후 독립 스케줄러용
    """RSS 긴급 이벤트 스캔. 현재 파이프라인에서 호출 안 됨 (독립 실행 예정)."""
    import json as _json, urllib.request as _ur, xml.etree.ElementTree as _ET, time as _time

    ALERT_KEYWORDS = {
        'supply_cut':  ['opec cut','production cut','supply disruption','pipeline attack',
                        'embargo','export ban','field shutdown','force majeure'],
        'geopolitical':['nuclear plant','missile strike','drone attack','war escalat',
                        'strait of hormuz','tanker attack','oil facility','refinery attack',
                        'iran sanction','iran attack','iran war','hormuz block','hormuz clos',
                        'oil blockade','ceasefire fail','military strike','naval blockade',
                        'supply disruption','oil embargo','pipeline sabotage','houthi attack'],
        'demand_shock':['recession','demand collapse','economic crisis','china slowdown',
                        'global downturn','demand destruction'],
        'price_move':  ['oil surges','oil plunges','crude spikes','wti jumps','brent soars',
                        'oil prices surge','oil prices plunge','oil prices jump'],
    }
    SCORE = {'supply_cut': 3, 'geopolitical': 3, 'demand_shock': 2, 'price_move': 2}

    triggered   = []
    seen_titles = set()
    _rss_deadline = _time.time() + 30  # 총 RSS 수집 최대 30초

    # ── RSS 수집
    for url in NEWS_RSS:
        if _time.time() > _rss_deadline:
            log.debug("RSS 수집 총 타임아웃(30s) 초과 — 나머지 소스 건너뜀")
            break
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
        if len(_ovx) >= 2 and _ovx.iloc[-2] != 0:
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
        _atomic_json(result, OUTPUT_DIR / 'latest_alerts.json', ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ── 이메일 발송 (WARNING 이상)
    if alert_level in ('WARNING', 'CRITICAL') and SMTP_USER and SMTP_PASSWORD and ALERT_TO:
        try:
            import smtplib
            from email.mime.text import MIMEText
            _cat_em = {'supply_cut': '공급감산', 'geopolitical': '지정학',
                       'demand_shock': '수요충격', 'price_move': '가격변동', 'ovx_spike': 'OVX급등'}
            _lines = '\n'.join(
                f"  [{_cat_em.get(t['category'], t['category'])}] {t['title'][:80]}"
                for t in triggered[:5])
            _lvl_ko = {'CRITICAL': '위험', 'WARNING': '경고', 'NORMAL': '정상'}
            _al_ko  = _lvl_ko.get(alert_level, alert_level)
            _body  = f"""[유가 리스크] 🚨 실시간 이벤트 경보 — {detected_now}

경보 레벨 : {_al_ko}  (점수: {total_score})
OVX 급변  : {'⚠ 감지됨' if ovx_alert else '정상'}

▶ 감지된 이벤트
{_lines}

* 파이프라인을 실행하여 최신 리스크 신호를 확인하세요.
국제 유가 리스크 예측 시스템 MVP""".strip()
            _msg = MIMEText(_body, 'plain', 'utf-8')
            _msg['From']    = SMTP_USER
            _msg['To']      = ALERT_TO
            _msg['Subject'] = f"[유가 리스크] 🚨 {_al_ko} — {detected_now}"
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as _sv:
                _sv.starttls()
                _sv.login(SMTP_USER, SMTP_PASSWORD)
                _sv.sendmail(SMTP_USER, ALERT_TO, _msg.as_string())
            log.info(f"    📧 RSS 경보 이메일 발송 → {ALERT_TO} ({alert_level})")
        except Exception as _ee:
            log.warning(f"    RSS 경보 이메일 실패: {_ee}")

    log.info(f"    RSS 모니터링 완료: {alert_level} (점수={total_score}, 트리거={len(triggered)}건)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# §RSK  RISK CLASSIFICATION  —  classify_risk(): 리스크 레벨/점수/VaR 계산
# ─────────────────────────────────────────────────────────────────────────────
def classify_risk(feature_df: pd.DataFrame, full_df: pd.DataFrame, forecast_dir: int = 0, surge_prob: float = 0.0) -> dict:
    """실시간 리스크 신호등: 정상 / 주의 / 급등위험 / 급락위험"""
    log.info("[6/9] 리스크 분류 중...")

    row = (full_df if full_df is not None else feature_df).iloc[-1]

    vol       = float(row.get('vol_5d',               0.015))
    mom_5     = float(row.get('mom_5d',               0.0))
    mom_21    = float(row.get('mom_21d',              0.0))
    sentiment = float(row.get('news_sentiment_smooth', 0.0))
    n_count   = float(feature_df['news_count'].iloc[-7:].sum()) if 'news_count' in feature_df.columns else float(row.get('news_count', 0.0))
    bb        = float(row.get('bb_position',           0.0))
    geo       = float(row.get('geo_dummy',             0.0))
    # 뉴스 기반 지정학 감지로 GPR 데이터 지연 보완
    try:
        import json as _json
        _alerts = _json.loads((OUTPUT_DIR / 'latest_alerts.json').read_text(encoding='utf-8'))
        if any(t.get('category') == 'geopolitical' for t in _alerts.get('triggers', [])):
            geo = max(geo, 1.0)
    except Exception:
        pass
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
    hist_mom_75 = float(feature_df['mom_5d'].abs().iloc[:_n_tr_cr].quantile(0.75)) if 'mom_5d' in feature_df.columns else 0.047

    # ── 리스크 점수
    # mom_ratio: 5일 모멘텀 절대값 기반 (트렌드형 이동 포착, cap=3.5로 과도 증폭 방지)
    mom_ratio     = min(abs(mom_5) / (hist_mom_75 + 1e-8), 3.5)
    vol_ratio     = max(vol / (hist_vol_75 + 1e-8), mom_ratio)
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

    # 모멘텀 하락 시 surge_prob 할인 (하락 국면 급등 확률 과대평가 억제)
    _surge_p_adj = surge_prob * max(0.0, 1.0 + mom_5 * 3.0) if mom_5 < -0.03 else surge_prob
    if _surge_p_adj != surge_prob:
        log.info(f"    📉 surge_prob 모멘텀 할인: {surge_prob:.3f}→{_surge_p_adj:.3f} (mom={mom_5*100:.1f}%)")
    # DROP_RISK + 높은 surge_prob → 방향 불확실 → CAUTION (모멘텀 할인 후 기준)
    if level == 'DROP_RISK' and _surge_p_adj > 0.50:
        log.info(f"    ⚖ DROP_RISK + surge_prob_adj={_surge_p_adj:.2f}>0.50 → CAUTION 조정")
        level = 'CAUTION'

    # ── CI 멀티플라이어: Shock 이진 + 뉴스 감성 서프라이즈 연속 조정
    _gpr_z      = float(row.get('gpr_zscore', 0.0))
    _geo_dum    = float(row.get('geo_dummy',  0.0))
    _sent_surp  = abs(float(row.get('sent_surprise_z', 0.0)))
    _news_smag  = abs(float(row.get('news_sentiment_smooth', 0.0)))
    _shock      = (_gpr_z > 2.0) or (_geo_dum > 0.5) or (risk_score >= 5.0)
    if _shock:
        ci_multiplier = 1.5
        log.info(f"    ⚡ Shock 감지 (gpr_z={_gpr_z:.2f} geo={_geo_dum:.0f} "
                 f"score={risk_score:.2f}) → CI×{ci_multiplier:.1f}")
    elif _sent_surp > 1.5 or _news_smag > 0.4:
        # 감성 서프라이즈 or 강한 부정 감성 → CI 연속 확대 (1.0 ~ 1.5)
        _cont_mult = 1.0 + 0.4 * min(_sent_surp / 2.0 + _news_smag, 1.0)
        ci_multiplier = float(np.clip(_cont_mult, 1.0, 1.5))
        log.info(f"    📰 뉴스 서프라이즈 CI 조정: surp_z={_sent_surp:.2f} "
                 f"mag={_news_smag:.2f} → CI×{ci_multiplier:.2f}")
    elif ovx_z < -0.3:
        # OVX 평온 구간 → CI 축소 (calm market, narrow band)
        ci_multiplier = 0.75
        log.info(f"    📉 평온 구간 CI 축소: ovx_z={ovx_z:.2f} → CI×{ci_multiplier:.2f}")
    else:
        ci_multiplier = 1.0

    # OVX 절대값 기반 CI 추가 확대 (내재변동성 선행 신호 — z-score와 독립 적용, max로 중복 누적 방지)
    if ovx_level >= 60:
        _ovx_abs_mult = 1.5
    elif ovx_level >= 45:
        _ovx_abs_mult = 1.3
    elif ovx_level >= 35:
        _ovx_abs_mult = 1.2
    elif ovx_level >= 25:
        _ovx_abs_mult = 1.1
    else:
        _ovx_abs_mult = 1.0
    if _ovx_abs_mult > ci_multiplier:
        log.info(f"    🔥 OVX 절대값 CI 확대: OVX={ovx_level:.1f} → CI×{_ovx_abs_mult:.1f}")
        ci_multiplier = _ovx_abs_mult

    current_wti = float(full_df['WTI'].dropna().iloc[-1])

    # 헤지 비율: 리스크 레벨 + surge_prob 기반 (유가 사용 기업 기준)
    _base_hedge = {'SURGE_RISK': 0.65, 'CAUTION': 0.30, 'DROP_RISK': 0.20, 'NORMAL': 0.0}[level]
    _surge_adj  = max((surge_prob - 0.45) * 0.6, 0.0) if level == 'SURGE_RISK' else 0.0
    _geo_adj    = 0.10 if geo > 0.5 else 0.0
    _mom_adj    = 0.10 if mom_5 < -0.05 else 0.0
    _ovx_adj    = 0.05 if ovx_level >= 60 else 0.0
    hedge_ratio = round(float(np.clip(_base_hedge + _surge_adj + _geo_adj + _mom_adj + _ovx_adj, 0.0, 0.75)), 2)

    # EIA 발표일(수요일) + 반응일(목요일): CI max 적용 (곱셈 누적 방지) + hedge 증가
    _eia_dow = pd.Timestamp.now('Asia/Seoul').dayofweek  # 0=Mon, 2=Wed, 3=Thu
    if _eia_dow in (2, 3):
        _eia_ci_floor = 1.2 if _eia_dow == 2 else 1.1
        ci_multiplier = max(ci_multiplier, _eia_ci_floor)
        hedge_ratio   = round(float(np.clip(hedge_ratio + 0.10, 0.0, 1.0)), 2)
        log.info(f"    📊 EIA {'발표일' if _eia_dow == 2 else '반응일'} 불확실성 확대: "
                 f"CI×{ci_multiplier:.2f} hedge={hedge_ratio:.2f}")

    # 복합충격(geo+opec+극단뉴스 동시 발생) → CI 하드캡 우회 + 극단 확대
    _cshock_r = float(feature_df.iloc[-1].get('compound_shock_flag', 0.0)) if 'compound_shock_flag' in feature_df.columns else 0.0
    if _cshock_r >= 1.0:
        ci_multiplier = max(ci_multiplier, 2.5)
        if level == 'NORMAL':
            level = 'CAUTION'
        log.info(f"    🚨 복합충격 감지: CI×{ci_multiplier:.1f} (지정학+OPEC+극단뉴스 동시)")
    else:
        ci_multiplier = float(min(ci_multiplier, 1.5))

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
        'surge_prob_3d':     round(_surge_p_adj, 3),
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
        _hist['date'] = pd.to_datetime(_hist['date'])
        _hist = _hist[_hist['date'] >= pd.Timestamp.now() - pd.DateOffset(years=1)]
        _hist['date'] = _hist['date'].dt.strftime('%Y-%m-%d')
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
# §VIZ  VISUALIZATION  —  extract_crisis_keywords(), generate_wordcloud(), plot_oil_forecast()
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

    _atomic_csv(kw_df, OUTPUT_DIR / 'crisis_keywords.csv', index=False)
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
    # ── CRISIS_SEED 누락 보완
    'shortage':'공급부족', 'shortages':'공급부족',
    'embargo':'금수조치', 'embargoes':'금수조치',
    'explosion':'폭발', 'explosions':'폭발',
    'disruption':'공급차질', 'disruptions':'공급차질',
    'houthi':'후티', 'houthis':'후티',
    'hurricane':'허리케인', 'hurricanes':'허리케인',
    'strategic':'전략적',
    'spr':'전략비축유',
    'crash':'폭락',
    'fear':'공포', 'concern':'우려', 'concerns':'우려',
    # ── 지정학/분쟁 추가
    'blockade':'봉쇄', 'blockades':'봉쇄',
    'seizure':'나포', 'seizures':'나포',
    'sabotage':'사보타주',
    'airstrike':'공습', 'airstrikes':'공습',
    'nuclear':'핵/원자력', 'weapons':'무기',
    'rebel':'반군', 'rebels':'반군',
    'militia':'민병대',
    'tribunal':'재판소',
    # ── 수급/시장 추가
    'talks':'협상', 'deals':'거래협상',
    'negotiation':'협상', 'negotiations':'협상',
    'meeting':'회의', 'meetings':'회의',
    'expected':'예상', 'expect':'예상', 'expects':'예상',
    'slowdown':'경기둔화', 'slowing':'둔화',
    'boost':'부양', 'boosts':'부양', 'boosted':'부양',
    'pressure':'압박', 'pressures':'압박',
    'signal':'신호', 'signals':'신호',
    'halt':'중단', 'halts':'중단', 'halted':'중단',
    'jump':'급등', 'jumped':'급등', 'jumps':'급등',
    'plunge':'급락', 'plunged':'급락', 'plunges':'급락',
    'rallied':'반등',
    'reduce':'감소', 'reduced':'감소', 'reduction':'감소',
    'raise':'인상', 'lower':'인하',
    'trader':'트레이더', 'traders':'트레이더', 'trading':'거래',
    # ── 에너지/상품 추가
    'futures':'선물', 'spot':'현물',
    'tanker':'유조선', 'tankers':'유조선',
    'shipping':'해운', 'vessel':'선박', 'vessels':'선박',
    'cargo':'화물',
    'drilling':'시추', 'drill':'시추',
    'fracking':'수압파쇄',
    'upstream':'상류', 'downstream':'하류', 'midstream':'중류',
    'gasoline':'가솔린', 'diesel':'디젤',
    'heating':'난방유', 'kerosene':'등유',
    'petrochemical':'석유화학',
    'hydrogen':'수소', 'methane':'메탄',
    'heavy':'중질유', 'light':'경질유',
    'sour':'고유황', 'sweet':'저유황',
    'blend':'혼합', 'blends':'혼합',
    'nymex':'NYMEX',
    'contango':'콘탱고', 'backwardation':'백워데이션',
    'spread':'스프레드', 'premium':'프리미엄',
    'benchmark':'기준가',
    # ── 거시/금융 추가
    'tariff':'관세', 'tariffs':'관세',
    'ban':'금지', 'bans':'금지', 'banned':'금지',
    'regulation':'규제', 'regulations':'규제', 'regulatory':'규제',
    'approval':'승인', 'approved':'승인',
    'merger':'합병', 'acquisition':'인수',
    'capex':'설비투자', 'spending':'지출',
    'revenue':'수익', 'revenues':'수익',
    'cost':'비용', 'costs':'비용',
    'profit':'이익', 'profits':'이익',
    'loss':'손실', 'losses':'손실',
    'capacity':'생산능력',
    'major':'주요', 'key':'핵심',
    'news':'뉴스',
    # ── 실제 데이터 기반 추가 (2026-05-25)
    'president':'대통령',
    'zelenskyy':'젤렌스키',
    'hormuz':'호르무즈', 'strait':'해협',
    'donald':'도널드',
    'peace':'평화',
    'end':'종료', 'ends':'종료',
    'claims':'주장',
    'strikes':'파업/공습',
    'easy':'완화',
}

_KW_AUTO_CACHE = OUTPUT_DIR / 'kw_translate_auto.json'


def _auto_translate_words(words: list[str]) -> dict[str, str]:
    """미매핑 영어 단어 → Google 자동 번역 후 _KW_AUTO_CACHE에 저장."""
    # 캐시 로드
    cache: dict = {}
    if _KW_AUTO_CACHE.exists():
        try:
            with open(_KW_AUTO_CACHE, encoding='utf-8') as _f:
                cache = json.load(_f)
        except Exception:
            cache = {}

    to_translate = [w for w in words if w not in cache]

    if to_translate:
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='en', target='ko')
            for w in to_translate:
                try:
                    result = translator.translate(w)
                    cache[w] = result if result else w
                except Exception:
                    cache[w] = w
            log.info(f"    [자동번역] {len(to_translate)}개 번역: {to_translate}")
        except ImportError:
            log.warning("    [자동번역] deep-translator 없음 → pip install deep-translator")
            for w in to_translate:
                cache[w] = w

        try:
            with open(_KW_AUTO_CACHE, 'w', encoding='utf-8') as _f:
                json.dump(cache, _f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as _e:
            log.warning(f"    [자동번역] 캐시 저장 실패: {_e}")

    return {w: cache.get(w, w) for w in words}


def generate_wordcloud(kw_df: pd.DataFrame):
    """위기 키워드 워드클라우드 — 영어 키워드를 한글로 변환하여 표시"""
    log.info("[8/9] 워드클라우드 생성 중...")
    crisis_set = set(kw_df[kw_df['is_crisis_word']]['keyword'])

    # 영어 → 한글 변환
    kw_ko = kw_df.copy()
    kw_ko['keyword_display'] = kw_ko['keyword'].apply(
        lambda w: _KW_TRANSLATE.get(w.lower(), w)
    )

    # 변환 후에도 소문자 ASCII → 미매핑 영어: 자동 번역 후 적용
    def _is_unmapped(s: str) -> bool:
        return s == s.lower() and s.isascii()

    unmapped_mask = kw_ko['keyword_display'].apply(_is_unmapped)
    unmapped_words = kw_ko.loc[unmapped_mask, 'keyword_display'].tolist()
    if unmapped_words:
        auto = _auto_translate_words(unmapped_words)
        kw_ko.loc[unmapped_mask, 'keyword_display'] = kw_ko.loc[unmapped_mask, 'keyword_display'].map(auto)

    # 자동 번역 후에도 여전히 소문자 ASCII(번역 실패)이면 최종 제외
    still_unmapped = kw_ko['keyword_display'].apply(_is_unmapped)
    if still_unmapped.any():
        log.warning(f"    [워드클라우드] 번역 실패 최종 제외: {kw_ko.loc[still_unmapped, 'keyword_display'].tolist()}")
        kw_ko = kw_ko[~still_unmapped].copy()

    if kw_ko.empty:
        log.warning("    [워드클라우드] 번역된 키워드 없음 — 생성 건너뜀")
        return

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
    if 'var_forecast' in fc_df.columns:
        ax1.plot(fd, fc_df['var_forecast'], color='#3fb950', lw=1, ls=':', alpha=0.8, label='VAR')

    ax1.fill_between(fd, fc_df['lower_75ci'], fc_df['upper_75ci'],
                     alpha=0.18, color=CYAN, label='75% 예측구간')

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
        axB.fill_between(fd, fc_df['lower_75ci'], fc_df['upper_75ci'],
                         alpha=0.30, color=CYAN, label='75% 예측구간', zorder=2)
        axB.plot(fd, fc_df['lower_75ci'], color=CYAN, lw=0.8, ls=':', alpha=0.6)
        axB.plot(fd, fc_df['upper_75ci'], color=CYAN, lw=0.8, ls=':', alpha=0.6)
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
        _lo1, _hi1 = fc_df['lower_75ci'].iloc[0], fc_df['upper_75ci'].iloc[0]
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
# §PIP  PIPELINE ORCHESTRATION  —  run_pipeline(), schedule_daily()
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(start_date=None, end_date=None) -> dict:
    """전체 분석 파이프라인 실행 후 결과 dict 반환"""
    # 파일 로그 핸들러 (관리자 대시보드용) — 중복 추가 방지
    _log_file = OUTPUT_DIR / 'pipeline_run.log'
    _root_log = logging.getLogger()
    _has_fh = any(isinstance(_h, logging.FileHandler) and
                  getattr(_h, 'baseFilename', '').endswith('pipeline_run.log')
                  for _h in _root_log.handlers)
    _root_log.setLevel(logging.INFO)
    if not _has_fh:
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            _fh = logging.FileHandler(_log_file, encoding='utf-8')
            _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s",
                                               datefmt='%H:%M:%S'))
            _root_log.addHandler(_fh)
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  🛢  국제 유가 리스크 예측 시스템  MVP")
    print("=" * 65)

    api_status = {}  # API별 수집 성공 여부 추적

    # yfinance + Guardian 병렬 fetch
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=2) as _ex:
        _f_price = _ex.submit(fetch_data, start_date, end_date)
        _f_news  = _ex.submit(fetch_news)
        try:
            price_df = _f_price.result()
            _is_dummy = (len(price_df) < 100 and price_df.index[0].year < 2010)
            api_status['yfinance'] = '❌ 더미' if _is_dummy else '✅ 정상'
        except Exception:
            price_df = fetch_data(start_date, end_date)
            api_status['yfinance'] = '❌ 오류'
        try:
            news_df = _f_news.result()
            api_status['Guardian'] = '✅ 정상' if len(news_df) > 10 else '⚠️ 부족'
        except Exception:
            news_df = fetch_news()
            api_status['Guardian'] = '❌ 오류'

    # FRED
    api_status['FRED'] = '✅ 정상' if (_FRED and FRED_API_KEY) else '❌ 미설정'

    feature_df, full_df, aux_models = build_features(price_df, news_df)
    # EIA: build_features 이후 실제 데이터 존재 여부로 검증
    _eia_ok = ('inv_surprise' in feature_df.columns
               and feature_df['inv_surprise'].abs().sum() > 0)
    api_status['EIA'] = ('✅ 정상' if _eia_ok
                         else '⚠️ 수집 실패' if EIA_API_KEY
                         else '❌ 미설정')
    _model_age_days = _check_refit_staleness()
    model_results, _     = train_models(feature_df, full_df, aux=aux_models)
    _save_retrain_record()

    # forecast_7days.csv 덮어쓰기 전에 이전 예측 로드 (gap-fill용)
    prev_fc_df = None
    _fc_csv = OUTPUT_DIR / 'forecast_7days.csv'
    if _fc_csv.exists():
        try:
            prev_fc_df = pd.read_csv(_fc_csv)
        except Exception as _pfe:
            log.warning(f"    이전 forecast 로드 실패({_pfe}) → gap-fill 생략")

    # Live CI recalibration: live type 실제 예측 오차 Q75로 훈련시점 Q75 override
    # backtest 오차는 제외 — outlier가 많아 CI 과도 확대 초래
    _pl_path_ci = OUTPUT_DIR / 'prediction_log.csv'
    if _pl_path_ci.exists():
        try:
            _pl_ci = pd.read_csv(_pl_path_ci)
            _live_rows = _pl_ci[
                (_pl_ci.get('type', pd.Series(dtype=str)) == 'live') &
                _pl_ci['stacking_error'].notna()
            ] if 'type' in _pl_ci.columns else pd.DataFrame()
            _se_ci = _live_rows['stacking_error'] if not _live_rows.empty else pd.Series(dtype=float)
            if len(_se_ci) >= 20:
                _live_q75 = float(np.percentile(np.abs(_se_ci.iloc[-60:]), 75))
                aux_models['live_ci_q75'] = _live_q75
                log.info(f"    Live CI Q75 재보정(live): {_live_q75:.2f}$ (N={min(len(_se_ci), 60)})")
            else:
                log.info(f"    Live CI 재보정 스킵 (live 오차 N={len(_se_ci)} < 20, backtest Q75 유지)")
        except Exception as _lce:
            log.warning(f"    Live CI recalibration 실패({_lce})")

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

    # ── risk_history.csv 백필: feature_df 기준 미기록 과거 날짜 보완
    try:
        _bfill_path = OUTPUT_DIR / 'risk_history.csv'
        _existing_dates: set = set()
        if _bfill_path.exists():
            _bfill_df = pd.read_csv(_bfill_path)
            _existing_dates = set(pd.to_datetime(_bfill_df['date']).dt.strftime('%Y-%m-%d'))
        _bfill_cutoff = pd.Timestamp.now() - pd.DateOffset(years=1)
        _bfill_fd = feature_df[feature_df.index >= _bfill_cutoff] if hasattr(feature_df.index, 'min') and pd.api.types.is_datetime64_any_dtype(feature_df.index) else feature_df.iloc[-252:]
        # 공휴일(WTI 없는 영업일) 포함: 영업일 기준 reindex 후 전일 값 forward fill
        if pd.api.types.is_datetime64_any_dtype(_bfill_fd.index) and len(_bfill_fd) > 0:
            _bfill_end = min(_bfill_fd.index.max(), pd.Timestamp.now().normalize())
            _bfill_fd = _bfill_fd.reindex(
                pd.date_range(_bfill_fd.index.min(), _bfill_end, freq='D'),
                method='ffill'
            )
        # hist_vol_75: classify_risk와 동일 방법 (고정 기준선)
        _n_tr_cr_bf = max(len(feature_df) - 60, 252)
        _h_vol75_bf = float(feature_df['vol_5d'].iloc[:_n_tr_cr_bf].quantile(0.75)) if 'vol_5d' in feature_df.columns else 0.022
        _h_mom75_bf = float(feature_df['mom_5d'].abs().iloc[:_n_tr_cr_bf].quantile(0.75)) if 'mom_5d' in feature_df.columns else 0.047
        _bfill_rows = []
        for _bdate, _brow in _bfill_fd.iterrows():
            _ds = str(_bdate)[:10]
            if _ds in _existing_dates:
                continue
            _bvol     = float(_brow.get('vol_5d',                  0.015))
            _bmom5    = float(_brow.get('mom_5d',                  0.0))
            _bmom21   = float(_brow.get('mom_21d',                 0.0))
            _bsent    = float(_brow.get('news_sentiment_smooth',   0.0))
            _bneg     = float(_brow.get('news_count_neg',          0.0))
            _bpos     = float(_brow.get('news_count_pos',          0.0))
            _bgeo     = float(_brow.get('geo_dummy',               0.0))
            _bovx_z   = float(_brow.get('ovx_zscore',             0.0))
            _bovx_chg = float(_brow.get('ovx_change',             0.0))
            _bbb      = float(_brow.get('bb_position',            0.0))
            _bpz      = float(_brow.get('price_zscore',           0.0))
            _bexn     = float(_brow.get('extreme_neg_news',       0.0))
            _bwti     = float(_brow.get('WTI',                    0.0))
            _bmom_ratio  = min(abs(_bmom5) / (_h_mom75_bf + 1e-8), 3.5)
            _bvol_ratio  = max(_bvol / (_h_vol75_bf + 1e-8), _bmom_ratio)
            _bnews_amp   = max(1 + min((_bneg - _bpos * 0.5) / 8, 1.0), 1.0)
            _bgeo_amp    = 1.35 if _bgeo > 0.5 else 1.0
            _bsent_amp   = 1 + max(-_bsent, 0) * 0.5 + _bexn * 0.1
            _bovx_amp    = 1.0 + max(_bovx_z - 1.0, 0) * 0.15
            _bscore      = _bvol_ratio * _bnews_amp * _bgeo_amp * _bsent_amp * _bovx_amp
            _bdir  = (_bmom5 * 0.5 + _bmom21 * 0.3 + _bsent * 0.3
                      + _bbb * 0.15 - _bovx_chg * 0.1)
            if _bmom5 * _bmom21 > 0:
                _bdir *= 1.3
            if _bpz > 1.5:
                _bdir -= 0.01
            elif _bpz < -1.5:
                _bdir += 0.01
            _bdth = 0.05 if (_bgeo > 0.5 or _bovx_z > 1.5) else 0.08
            if   _bscore >= 2.2 and _bdir >  _bdth: _blevel = 'SURGE_RISK'
            elif _bscore >= 2.2 and _bdir < -_bdth: _blevel = 'DROP_RISK'
            elif _bscore >= 1.4 or abs(_bdir) > 0.06: _blevel = 'CAUTION'
            else:                                      _blevel = 'NORMAL'
            _bfill_rows.append({'date': _ds, 'risk_level': _blevel,
                                'risk_score': round(_bscore, 4),
                                'wti_price': round(_bwti, 2),
                                'volatility': round(_bvol, 5)})
        if _bfill_rows:
            _bfill_new = pd.DataFrame(_bfill_rows)
            if _bfill_path.exists():
                _bfill_old = pd.read_csv(_bfill_path)
                _bfill_combined = (pd.concat([_bfill_old, _bfill_new], ignore_index=True)
                                   .drop_duplicates(subset=['date'], keep='last')
                                   .sort_values('date').tail(365))
            else:
                _bfill_combined = _bfill_new.sort_values('date').tail(365)
            _atomic_csv(_bfill_combined, _bfill_path, index=False)
            log.info(f"    risk_history.csv 백필: {len(_bfill_rows)}행 추가 (총 {len(_bfill_combined)}행)")
    except Exception as _bfe:
        log.warning(f"    risk_history 백필 실패({_bfe})")

    # Shock CI 확대: classify_risk에서 반환된 ci_multiplier 적용
    _ci_mult = risk_signal.get('ci_multiplier', 1.0)
    if abs(_ci_mult - 1.0) > 0.01 and 'lower_75ci' in fc_df.columns and 'upper_75ci' in fc_df.columns:
        _fp = fc_df['forecast_price']
        # D+1: x(1+(mult-1)*0.5) ~ D+7: x mult — 단기 CI 과도 확대 방지
        _mult_arr = 1.0 + (_ci_mult - 1.0) * np.linspace(0.5, 1.0, len(fc_df))
        fc_df['lower_75ci'] = (_fp - (_fp - fc_df['lower_75ci']) * _mult_arr).round(2)
        fc_df['upper_75ci'] = (_fp + (fc_df['upper_75ci'] - _fp) * _mult_arr).round(2)
        _atomic_csv(fc_df, OUTPUT_DIR / 'forecast_7days.csv', index=False)
        log.info(f"    Shock CI 적용: ×{_ci_mult:.1f} → forecast_7days.csv 갱신")

    save_prediction_log(model_results, feature_df, fc_df, prev_fc_df, full_df)
    analyze_live_gap()

    # ── A: 예측 신뢰도 — forecast_snapshots.csv 전체 기준 MASE
    _live_mae_val = None
    _live_mase_val = None
    _live_drift_flag = False
    _forecast_reliability = 'UNKNOWN'
    _snap_mase = None
    _snap_n = 0
    _snap_path_chk = OUTPUT_DIR / 'forecast_snapshots.csv'
    if _snap_path_chk.exists():
        try:
            _snap_df = pd.read_csv(_snap_path_chk)
            if PRED_LOG_FILE.exists():
                _pl_act = (pd.read_csv(PRED_LOG_FILE)[['date', 'actual_price']]
                           .rename(columns={'date': 'forecast_date', 'actual_price': '_pl_actual'})
                           .drop_duplicates('forecast_date'))  # backtest/live 중복 방지
                _snap_df = _snap_df.merge(_pl_act, on='forecast_date', how='left')
                # snap에 actual_price 컬럼이 이미 있으면 merge 시 _x/_y로 분리됨 → coalesce
                if 'actual_price_x' in _snap_df.columns:
                    _snap_df['actual_price'] = _snap_df['actual_price_x'].fillna(_snap_df.get('actual_price_y'))
                    _snap_df = _snap_df.drop(columns=[c for c in ['actual_price_x', 'actual_price_y'] if c in _snap_df.columns])
                elif '_pl_actual' in _snap_df.columns:
                    _snap_df['actual_price'] = _snap_df['actual_price'].fillna(_snap_df['_pl_actual'])
                _snap_df = _snap_df.drop(columns=[c for c in ['_pl_actual'] if c in _snap_df.columns])
            _snap_known = (_snap_df[_snap_df['actual_price'].notna()]
                           .sort_values('forecast_date')  # diff() 날짜 순서 보장
                           .copy())
            _snap_known['abs_err'] = (_snap_known['actual_price'] - _snap_known['forecast_price']).abs()
            # d2~d7 actual 포함: 각 horizon의 (actual, forecast) 쌍을 모아 MASE에 반영
            _extra_pairs = []
            for _di in range(2, 8):
                _fcol, _acol = f'd{_di}_forecast', f'd{_di}_actual'
                if _fcol in _snap_df.columns and _acol in _snap_df.columns:
                    _di_df = _snap_df[_snap_df[_acol].notna()][[_acol, _fcol]].copy()
                    if len(_di_df) > 0:
                        _di_df = _di_df.rename(columns={_acol: 'actual_price', _fcol: 'forecast_price'})
                        _di_df['abs_err'] = (_di_df['actual_price'] - _di_df['forecast_price']).abs()
                        _extra_pairs.append(_di_df[['actual_price', 'forecast_price', 'abs_err']])
            _snap_known_all = (pd.concat([_snap_known[['actual_price', 'forecast_price', 'abs_err']], *_extra_pairs], ignore_index=True)
                               if _extra_pairs else _snap_known[['actual_price', 'forecast_price', 'abs_err']].copy())
            _snap_n = len(_snap_known)  # D+1 기준 행 수 유지
            _snap_n_all = len(_snap_known_all)
            if _snap_n >= 5:
                _snap_mae = float(_snap_known_all['abs_err'].mean())
                _snap_naive = float(_snap_known['actual_price'].diff().abs().dropna().mean())
                _snap_mase = _snap_mae / max(_snap_naive, 1e-6)
                _forecast_reliability = (
                    'HIGH' if _snap_mase < 1.0 else
                    'MEDIUM' if _snap_mase < 1.5 else 'LOW'
                )
                log.info(f"    예측 신뢰도: MASE={_snap_mase:.3f} (n={_snap_n}, multi-horizon={_snap_n_all}) → {_forecast_reliability}")
        except Exception as _sne:
            log.warning(f"    스냅샷 MASE 계산 실패({_sne})")

    if PRED_LOG_FILE.exists():
        try:
            _pl_check = pd.read_csv(PRED_LOG_FILE)
            _live_known = _pl_check[
                _pl_check['type'].isin(['live', 'gap']) &
                _pl_check['price_error'].notna()
            ]
            if len(_live_known) >= 5:
                _live30 = _live_known.tail(30)
                # σ-clip: 이상치(±2σ 초과) 제외 후 MAE 계산 (단발 급락/급등 오염 방지)
                _abs_err = _live30['price_error'].abs()
                _err_mu, _err_sd = _abs_err.mean(), _abs_err.std()
                _clip_mask = _abs_err <= (_err_mu + 2 * _err_sd)
                _live30_clean = _live30[_clip_mask]
                _n_clipped = int((~_clip_mask).sum())
                _live_mae_val = float(_live30_clean['price_error'].abs().mean()) if len(_live30_clean) >= 5 else float(_abs_err.mean())
                if _n_clipped > 0:
                    log.info(f"    σ-clip: {_n_clipped}건 이상치 제외 (raw_mae={_abs_err.mean():.4f} → clean_mae={_live_mae_val:.4f})")
                _stk_m = model_results.get('stacking') or model_results.get('sarimax', {})
                _bt_mae_ref = _stk_m.get('mae', float('inf'))
                _ratio = _live_mae_val / max(_bt_mae_ref, 1e-6)
                log.info(f"    라이브 성능: 최근 {len(_live30)}건 MAE={_live_mae_val:.4f} "
                         f"(백테스트 대비 {_ratio:.2f}×)")
                if _ratio > 1.5:
                    log.warning(f"    ⚠ 라이브 성능 열화: 라이브 MAE={_live_mae_val:.4f} > "
                                f"백테스트×1.5 ({_bt_mae_ref:.4f}×1.5={_bt_mae_ref*1.5:.4f})")
                # 열화 추적용 live MASE (재훈련 트리거 기준)
                _naive_diffs = _live30_clean['actual_price'].diff().abs().dropna()
                if len(_naive_diffs) >= 5:
                    _naive_mae_live = float(_naive_diffs.mean())
                    _live_mase = _live_mae_val / max(_naive_mae_live, 1e-6)
                    _live_mase_val = round(_live_mase, 4)
                    log.info(f"    라이브 MASE(30d, σ-clip): {_live_mase:.3f} "
                             f"(naive_mae={_naive_mae_live:.4f})")
                    if _live_mase > 0.95:
                        log.warning(f"    ⚠ 라이브 MASE={_live_mase:.3f} → naive 근접, 재훈련 검토")
                    # n_live>=20 시 drift 경고 — run_meta.json에 기록
                    if len(_live_known) >= 20 and _live_mase > 0.85:
                        _live_drift_flag = True
                        log.warning(f"    ⚠ Live MASE 경고: {_live_mase:.3f} > 0.85 "
                                    f"(n={len(_live_known)}) → 성능 검토 필요")
                    # 열화 추적 → 연속 N일 초과 시 캐시 강제 초기화
                    _ovx_deg = 0.0
                    try:
                        _sig_tmp = pd.read_csv(OUTPUT_DIR / 'latest_risk_signal.csv')
                        _ovx_deg = float(_sig_tmp['ovx_level'].iloc[-1]) if 'ovx_level' in _sig_tmp.columns else 0.0
                    except Exception:
                        pass
                    _force_retrain = _check_degradation_trigger(_live_mase, ovx_level=_ovx_deg)
                    if _force_retrain and SMTP_USER and SMTP_PASSWORD and ALERT_TO:
                        def _send_retrain_email(_mase=_live_mase):
                            try:
                                import smtplib
                                from email.mime.text import MIMEText as _MT
                                _body = (
                                    f"[유가 리스크] 자동 재훈련 트리거\n\n"
                                    f"라이브 MASE={_mase:.3f} ({MASE_RETRAIN_THRESH} 초과) "
                                    f"{MASE_RETRAIN_DAYS}일 연속 감지\n"
                                    f"→ Optuna/SVM/GARCH 캐시 초기화 완료\n"
                                    f"→ 다음 실행 시 전체 하이퍼파라미터 재탐색\n\n"
                                    f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                )
                                _em = _MT(_body, 'plain', 'utf-8')
                                _em['Subject'] = f"[OilRisk] 자동 재훈련 트리거 (MASE={_mase:.2f})"
                                _em['From']    = SMTP_USER
                                _em['To']      = ALERT_TO
                                with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as _sv:
                                    _sv.ehlo(); _sv.starttls(); _sv.ehlo()
                                    _sv.login(SMTP_USER, SMTP_PASSWORD)
                                    _sv.sendmail(SMTP_USER, ALERT_TO, _em.as_string())
                                log.info(f"    📧 재훈련 알림 이메일 발송 → {ALERT_TO}")
                            except Exception as _me:
                                log.warning(f"    재훈련 알림 이메일 실패({_me})")
                        _fire_and_forget(_send_retrain_email)
            # ── C: 75% CI 실제 커버리지 검증
            if 'lower_75ci' in _pl_check.columns and 'upper_75ci' in _pl_check.columns:
                _ci_rows = _pl_check[
                    _pl_check['type'].isin(['live', 'gap']) &
                    _pl_check['actual_price'].notna() &
                    _pl_check['lower_75ci'].notna() &
                    _pl_check['upper_75ci'].notna()
                ]
                if len(_ci_rows) >= 20:
                    _covered = (
                        (_ci_rows['actual_price'] >= _ci_rows['lower_75ci']) &
                        (_ci_rows['actual_price'] <= _ci_rows['upper_75ci'])
                    ).mean()
                    log.info(f"    75% CI 커버리지: {_covered:.1%} (N={len(_ci_rows)})")
                    if _covered < 0.65:
                        log.warning(f"    ⚠ CI 과소추정: 실제 커버리지={_covered:.1%} < 65% "
                                    f"— CI 폭 확대 검토 필요")
        except Exception as _lme:
            log.warning(f"    라이브 성능 모니터링 실패({_lme})")

    # 신뢰도 플래그 → latest_risk_signal.csv 패치
    _sig_path = OUTPUT_DIR / 'latest_risk_signal.csv'
    if _sig_path.exists():
        try:
            _sig_df = pd.read_csv(_sig_path)
            _sig_df['forecast_reliability'] = _forecast_reliability
            _sig_df['model_age_days'] = _model_age_days
            _atomic_csv(_sig_df, _sig_path, index=False)
            log.info(f"    forecast_reliability: {_forecast_reliability}  model_age_days: {_model_age_days}")
        except Exception:
            pass

    _fire_and_forget(send_risk_alert, risk_signal, fc_df)
    _fire_and_forget(send_alerts_to_subscribers, risk_signal, fc_df)
    kw_df                = extract_crisis_keywords(news_df)
    generate_wordcloud(kw_df)
    plot_oil_forecast(feature_df, fc_df, risk_signal)

    # ── 마지막 실행 시간 + API 상태 기록
    import json as _json
    _run_meta = {
        'last_run':           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_through':       full_df.index[-1].strftime('%Y-%m-%d'),
        'n_live':             int((pd.read_csv(PRED_LOG_FILE)['type'] == 'live').sum()) if PRED_LOG_FILE.exists() else 0,
        'live_mae':           round(_live_mae_val, 4) if _live_mae_val is not None else None,
        'live_mase':          _live_mase_val,
        'live_drift_warning': _live_drift_flag,
        'api_status':         api_status,
    }
    _atomic_json(_run_meta, OUTPUT_DIR / 'run_meta.json')

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
    _ovx_alm_ko = {'HIGH': '높음', 'ELEVATED': '상승', 'NORMAL': '정상'}.get(_ovx_alm, _ovx_alm)
    print(f"  OVX 수준    : {_ovx_lvl:.1f} ({_ovx_alm_ko})")
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

    monitor_rss_alerts()

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
