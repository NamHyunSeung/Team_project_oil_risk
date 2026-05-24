"""
국제 유가 리스크 예측 시스템 — Streamlit 대시보드
실행 방법: streamlit run dashboard.py
"""

import streamlit as st
import yaml
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import importlib.util as _ilu
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 설정 (Windows 맑은 고딕 우선, 없으면 AppleGothic, NanumGothic)
def _set_korean_font():
    candidates = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'NanumBarunGothic']
    available  = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
    # 직접 경로 시도 (Windows)
    path = Path('C:/Windows/Fonts/malgun.ttf')
    if path.exists():
        fm.fontManager.addfont(str(path))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

_set_korean_font()

OUTPUT_DIR = Path("output")

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="국제 유가 리스크 예측 시스템",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 로그인 ────────────────────────────────────────────────────────────────────
_AUTH_CFG = Path(__file__).parent / "auth_config.yaml"

def _load_authenticator():
    with open(_AUTH_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return stauth.Authenticate(
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
        auto_hash=True,
    )

_authenticator = _load_authenticator()

# 미인증 상태일 때만 로그인 폼 렌더링
if not st.session_state.get("authentication_status"):
    # ── 구독 플랜 안내
    st.markdown("""
<div style='text-align:center; padding: 32px 0 8px 0;'>
  <span style='font-size:2.4rem;'>🛢</span>
  <h2 style='color:#e6edf3; margin:8px 0 4px 0; font-size:1.6rem;'>국제 유가 리스크 예측 시스템</h2>
  <p style='color:#8b949e; margin:0 0 28px 0; font-size:0.9rem;'>AI 기반 WTI 유가 예측 · 리스크 신호 · 뉴스 감성 분석</p>
</div>
""", unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown("""
<div style='border-radius:12px; padding:24px 24px; text-align:center;
            border:1px solid #1f6feb; background:#0d2137;'>
  <p style='color:#58a6ff; font-size:0.8rem; margin:0 0 4px 0; letter-spacing:1px;'>PRO</p>
  <p style='color:#e6edf3; font-size:2.2rem; font-weight:700; margin:0;'>$79<span style='font-size:0.95rem; font-weight:400; color:#8b949e;'>/월</span></p>
  <hr style='border-color:#1f6feb; margin:14px 0;'>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 7일 WTI 가격 예측</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 리스크 신호 (NORMAL ~ SURGE)</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 뉴스 감성 분석</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 변동성 · VIX 상세 분석</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 이메일 알림 (급등/급락)</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 리스크 히스토리 차트</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ CSV 데이터 다운로드</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    _authenticator.login(
        location="main",
        fields={"Form name": "🔐  로그인",
                "Username": "아이디", "Password": "비밀번호", "Login": "로그인"},
    )
    _auth_status = st.session_state.get("authentication_status")
    if _auth_status is True:
        st.rerun()
    if _auth_status is False:
        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

_name     = st.session_state.get("name", "")
_username = st.session_state.get("username", "")
_is_admin = (_username == "admin")

# ── 구독 만료일 체크
try:
    with open(Path(__file__).parent / 'auth_config.yaml', encoding='utf-8') as _f2:
        _cfg2 = yaml.safe_load(_f2)
    _expiry_str = (_cfg2.get('credentials', {}).get('usernames', {})
                       .get(_username, {}).get('subscription_expiry', ''))
    if _expiry_str:
        _expiry    = datetime.datetime.strptime(_expiry_str, '%Y-%m-%d').date()
        _days_left = (_expiry - datetime.date.today()).days
        if _days_left < 0:
            st.error(f'구독이 만료됐습니다 ({_expiry_str}). 관리자에게 문의하세요.')
            _authenticator.logout('로그아웃', location='sidebar')
            st.stop()
        elif _days_left <= 30:
            st.warning(f'구독 만료 {_days_left}일 전 ({_expiry_str}). 갱신이 필요합니다.')
except Exception:
    pass

# ── 인증 성공: 사이드바에 사용자 정보 + 로그아웃 버튼
with st.sidebar:
    st.markdown(f"**{_name}** 님 환영합니다")
    _authenticator.logout("로그아웃", location="sidebar")

# ── 자동 새로고침 (10분 주기, st_autorefresh 없을 시 meta refresh 폴백) ────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")
except ImportError:
    st.markdown(
        '<meta http-equiv="refresh" content="600">',
        unsafe_allow_html=True,
    )

# ── 스타일 ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }
.block-container                   { padding-top: 1.2rem; }
.metric-box {
    background: linear-gradient(135deg, #1c2433, #161b22);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
}
h1, h2, h3, p, label, .stMarkdown { color: #e6edf3 !important; }

/* ── 메트릭 카드 ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1c2433, #161b22);
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #30363d;
    transition: border-color 0.2s;
}

/* ── 탭 스타일 ── */
button[data-baseweb="tab"] {
    color: #8b949e !important;
    border-radius: 6px 6px 0 0 !important;
    transition: color 0.15s, background 0.15s !important;
    font-size: 0.88rem !important;
}
button[data-baseweb="tab"]:hover {
    color: #e6edf3 !important;
    background: rgba(255,255,255,0.05) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: rgba(88,166,255,0.07) !important;
    font-weight: 600 !important;
}

/* ── 모바일 반응형 ── */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.8rem !important; }
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    [data-testid="column"] { min-width: 100% !important; }
    [data-testid="stTab"] { font-size: 0.75rem !important; padding: 4px 6px !important; }
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
</style>
""", unsafe_allow_html=True)

RISK_COLOR = {
    'NORMAL': '#2ecc71', 'CAUTION': '#f39c12',
    'SURGE_RISK': '#e74c3c', 'DROP_RISK': '#3498db',
}
RISK_LABEL = {
    'NORMAL': ('🟢', '정상'),
    'CAUTION': ('🟡', '주의'),
    'SURGE_RISK': ('🔴', '급등위험'),
    'DROP_RISK': ('🔵', '급락위험'),
}


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
_LOG_LIMIT = 1000

@st.cache_data(ttl=600)
def load_all():
    out = {}
    for name in ['model_performance', 'forecast_7days', 'latest_risk_signal', 'crisis_keywords', 'feature_importance', 'risk_history']:
        p = OUTPUT_DIR / f'{name}.csv'
        if p.exists():
            out[name] = pd.read_csv(p)
    p_log = OUTPUT_DIR / 'prediction_log.csv'
    if p_log.exists():
        out['prediction_log'] = pd.read_csv(p_log).tail(_LOG_LIMIT).reset_index(drop=True)
    return out


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛢 Oil Risk System")
    st.markdown("---")

    if st.button("🔄 파이프라인 실행 / 새로고침", use_container_width=True, type="primary"):
        with st.spinner("분석 중... (약 1-2분)"):
            try:
                import importlib, oil_risk_mvp
                importlib.reload(oil_risk_mvp)
                oil_risk_mvp.run_pipeline()
                st.cache_data.clear()
                st.success("✅ 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"오류: {e}")
                st.info("pip install -r requirements.txt 후 재시도하세요.")

    st.markdown("---")

    # 마지막 실행 시간
    _meta_path = OUTPUT_DIR / 'run_meta.json'
    if _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding='utf-8'))
            st.markdown("**마지막 파이프라인 실행**")
            st.markdown(f"🕐 `{_meta.get('last_run', '—')}`")
            st.markdown(f"📅 데이터: `~ {_meta.get('data_through', '—')}`")
            st.markdown(f"📊 실시간 예측: `{_meta.get('n_live', 0)}건`")

            _api = _meta.get('api_status', {})
            if _api:
                st.markdown("**API 수집 상태**")
                for _api_name, _stat in _api.items():
                    st.markdown(f"{_stat} `{_api_name}`")
        except Exception:
            pass
    else:
        st.caption("파이프라인 실행 후 갱신 시간이 표시됩니다.")

    st.markdown("---")
    st.markdown("**출력 파일 상태**")
    for fname in ['model_performance.csv', 'forecast_7days.csv',
                  'latest_risk_signal.csv', 'crisis_keywords.csv',
                  'oil_forecast_plot.png', 'wordcloud.png']:
        exists = (OUTPUT_DIR / fname).exists()
        st.markdown(f"{'✅' if exists else '❌'} `{fname}`")

    st.markdown("---")
    st.caption(f"대시보드 로드: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛢 국제 유가 리스크 예측 시스템")

if not (OUTPUT_DIR / 'latest_risk_signal.csv').exists():
    st.warning("⚠️ 분석 결과가 없습니다. 사이드바에서 **파이프라인 실행** 버튼을 클릭하세요.")
    st.stop()

data = load_all()

# ── 실시간 RSS 이벤트 경보 배너 ───────────────────────────────────────────────
_alert_path = OUTPUT_DIR / 'latest_alerts.json'
if _alert_path.exists():
    try:
        _al = json.loads(_alert_path.read_text(encoding='utf-8'))
        if _al.get('alert_level') in ('WARNING', 'CRITICAL'):
            _ac = '#e74c3c' if _al['alert_level'] == 'CRITICAL' else '#f39c12'
            _trigs = _al.get('triggers', [])[:3]
            _trig_html = ''.join(f"<li style='margin:2px 0'>{t['title'][:80]}</li>" for t in _trigs)
            st.markdown(f"""
            <div style="background:{_ac}18;border:2px solid {_ac};border-radius:12px;
                        padding:14px 20px;margin-bottom:12px;">
              <b style="color:{_ac}">🚨 실시간 이벤트 경보: {_al['alert_level']}</b>
              &nbsp;<span style="color:#8b949e;font-size:0.8rem">({_al['checked_at']} 기준)</span>
              <ul style="color:#c9d1d9;margin:6px 0 0;padding-left:18px;font-size:0.85rem">
              {_trig_html}
              </ul>
            </div>""", unsafe_allow_html=True)
    except Exception:
        pass

# ── 리스크 신호 배너 ───────────────────────────────────────────────────────────
if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
    sig   = data['latest_risk_signal'].iloc[0]
    level = sig['risk_level']
    col   = RISK_COLOR.get(level, '#888')
    em, lbl = RISK_LABEL.get(level, ('⚪', '알 수 없음'))

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{col}18,{col}0a);
                border:2px solid {col};border-radius:14px;
                padding:18px 24px;margin-bottom:18px;">
      <h2 style="color:{col};margin:0;font-size:1.6rem">{em} 현재 리스크: {lbl}</h2>
      <p style="color:#8b949e;margin:4px 0 0">기준일: {sig['date']}</p>
    </div>
    """, unsafe_allow_html=True)

    _surge_p = float(sig.get('surge_prob_3d', 0.0))
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("WTI 가격",   f"${sig['wti_price']:.2f}",
              f"{sig['momentum_5d']*100:+.1f}% (5일)")
    c2.metric("5일 변동성", f"{sig['volatility_5d']*100:.2f}%")
    c3.metric("뉴스 감성",  f"{sig['news_sentiment']:+.3f}",
              "부정" if sig['news_sentiment'] < -0.05 else
              ("긍정" if sig['news_sentiment'] > 0.05 else "중립"))
    c4.metric("뉴스 건수",  f"{int(sig['news_count'])}건")
    c5.metric("지정학 경보", "🔴 활성" if str(sig['geopolitical_alert']).lower() in ('true', '1') else "🟢 없음")
    c6.metric("리스크 점수", f"{sig['risk_score']:.3f}")
    c7.metric("급등확률(3일)", f"{_surge_p*100:.0f}%",
              "🔴 높음" if _surge_p > 0.6 else ("🟡 보통" if _surge_p > 0.3 else "🟢 낮음"))
    _ci_mult = sig.get('ci_multiplier', 1.0)
    if _ci_mult and float(_ci_mult) > 1.0:
        st.warning(f"⚡ **Shock 감지** — CI 구간 ×{float(_ci_mult):.1f} 확대 적용 중 (GPR 급등 또는 지정학 이벤트)")

    _hedge = float(sig.get('hedge_ratio', 0.0))
    if _hedge > 0.0:
        _hedge_clr = {'SURGE_RISK': '🔴', 'CAUTION': '🟡', 'DROP_RISK': '🟢', 'NORMAL': '⚪'}.get(level, '⚪')
        st.info(f"{_hedge_clr} **권장 헤지 비율: {_hedge*100:.0f}%** — "
                f"3일 급등확률 {_surge_p*100:.0f}% · {RISK_LABEL.get(level, ('',''))[1]} 신호 기반 "
                f"(유가 사용 기업: 항공·해운·제조)")

st.markdown("---")

# ── 탭 ───────────────────────────────────────────────────────────────────────
_tab_labels = ["📈 예측", "🌡 리스크", "☁ 키워드", "📊 성능", "📋 오차"]
if _is_admin:
    _tab_labels.append("🔧 관리자")
_tabs = st.tabs(_tab_labels)
tab1, tab2, tab3, tab4, tab5 = _tabs[:5]
tab_admin = _tabs[5] if _is_admin else None

# ── Tab 1: 가격 예측 ──────────────────────────────────────────────────────────
with tab1:
    col_chart, col_table = st.columns([2.2, 1])

    with col_chart:
        st.subheader("유가 예측 차트")
        if 'prediction_log' in data and 'forecast_7days' in data:
            _pl_fc = data['prediction_log'].copy()
            _fc_df = data['forecast_7days'].copy()
            _bt_fc = _pl_fc[_pl_fc['type'] == 'backtest'].copy()
            _lv_fc = _pl_fc[_pl_fc['type'].isin(['live', 'gap'])].copy()
            _bt_fc['date'] = pd.to_datetime(_bt_fc['date'])
            _lv_fc['date'] = pd.to_datetime(_lv_fc['date'])
            _fc_df['date'] = pd.to_datetime(_fc_df['date'])
            _bt30 = _bt_fc.dropna(subset=['actual_price']).tail(30)
            _lv_actual = _lv_fc[_lv_fc['actual_price'].notna()]

            fig_fc = go.Figure()
            if not _bt30.empty:
                fig_fc.add_trace(go.Scatter(
                    x=_bt30['date'], y=_bt30['actual_price'],
                    mode='lines', name='실제 WTI',
                    line=dict(color='#58a6ff', width=2),
                    hovertemplate='%{x|%m/%d}<br>실제: $%{y:.2f}<extra></extra>',
                ))
            if not _lv_actual.empty:
                fig_fc.add_trace(go.Scatter(
                    x=_lv_actual['date'], y=_lv_actual['actual_price'],
                    mode='lines+markers', name='실제 (실시간)',
                    line=dict(color='#58a6ff', width=2),
                    marker=dict(size=5),
                    hovertemplate='%{x|%m/%d}<br>실제: $%{y:.2f}<extra></extra>',
                    showlegend=False,
                ))
            if _is_admin and not _bt30.empty:
                _bt30_p = _bt30.dropna(subset=['sarimax_pred'])
                fig_fc.add_trace(go.Scatter(
                    x=_bt30_p['date'], y=_bt30_p['sarimax_pred'],
                    mode='lines', name='SARIMAX 예측',
                    line=dict(color='#f0c040', width=1.5, dash='dash'),
                    hovertemplate='%{x|%m/%d}<br>SARIMAX: $%{y:.2f}<extra></extra>',
                ))
            if 'lower_80ci' in _fc_df.columns and 'upper_80ci' in _fc_df.columns:
                _ci_x = pd.concat([_fc_df['date'], _fc_df['date'][::-1]])
                _ci_y = pd.concat([_fc_df['upper_80ci'], _fc_df['lower_80ci'][::-1]])
                fig_fc.add_trace(go.Scatter(
                    x=_ci_x, y=_ci_y,
                    fill='toself', fillcolor='rgba(240,192,64,0.10)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='80% 예측구간', hoverinfo='skip',
                ))
            fig_fc.add_trace(go.Scatter(
                x=_fc_df['date'], y=_fc_df['forecast_price'],
                mode='lines+markers', name='앙상블 예측',
                line=dict(color='#f0c040', width=2, dash='dot'),
                marker=dict(size=7, symbol='circle-open'),
                hovertemplate='%{x|%m/%d}<br>예측: $%{y:.2f}<extra></extra>',
            ))
            if not _bt30.empty:
                fig_fc.add_vline(x=str(_bt30['date'].max()),
                                 line=dict(color='#8b949e', dash='dash', width=1), opacity=0.5)
            fig_fc.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                            borderwidth=1, font=dict(size=11)),
                xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='날짜'),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='WTI ($)'),
                margin=dict(l=60, r=20, t=20, b=50),
                height=390, hovermode='x unified',
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            _plot_file = OUTPUT_DIR / ('oil_forecast_plot.png' if _is_admin else 'user_forecast_plot.png')
            if not _plot_file.exists():
                _plot_file = OUTPUT_DIR / 'oil_forecast_plot.png'
            if _plot_file.exists():
                st.image(str(_plot_file), use_container_width=True)
            else:
                st.info("파이프라인 실행 후 차트가 표시됩니다.")

    with col_table:
        st.subheader("📅 7일 예측")
        if 'forecast_7days' in data:
            fc = data['forecast_7days'].copy()
            _DAY_KO = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}
            fc['날짜'] = pd.to_datetime(fc['date']).apply(
                lambda d: f"{d.strftime('%m/%d')}({_DAY_KO[d.weekday()]})"
            )

            # 합의도 표시 (model_std 기반)
            if 'model_std' in fc.columns:
                def _consensus(std):
                    if std < 2:   return '🟢 높음'
                    if std < 5:   return '🟡 보통'
                    return          '🔴 낮음'
                fc['합의도'] = fc['model_std'].apply(_consensus)

            # 80% 예측구간 컬럼 생성
            if 'lower_80ci' in fc.columns and 'upper_80ci' in fc.columns:
                fc['80% 구간'] = fc.apply(
                    lambda r: f"${r['lower_80ci']:.1f} ~ ${r['upper_80ci']:.1f}", axis=1
                )

            if _is_admin:
                display_cols = {
                    '날짜': '날짜', 'forecast_price': '앙상블($)',
                    '80% 구간': '80% 구간',
                    'sarimax_forecast': 'SARIMAX($)', 'xgb_forecast': 'XGB($)',
                    '합의도': '합의도', 'bias_correction': 'Bias($)',
                }
            else:
                display_cols = {
                    '날짜': '날짜', 'forecast_price': '앙상블($)',
                    '80% 구간': '80% 구간', '합의도': '합의도',
                }
            show_fc = fc[[c for c in display_cols if c in fc.columns]].rename(columns=display_cols)
            st.dataframe(show_fc, hide_index=True, use_container_width=True)

            # 합의도 낮으면 경고
            if 'model_std' in fc.columns and not fc.empty and float(fc['model_std'].iloc[0]) >= 5:
                st.warning(f"⚠️ 모델 간 예측 편차 ${fc['model_std'].iloc[0]:.1f} — 불확실성 높음")
            if 'reliable_forecast' in fc.columns and not fc['reliable_forecast'].astype(str).eq('True').all():
                st.warning("⚠️ 일부 예측일의 신뢰도가 낮습니다. 예측 결과 해석 시 주의하세요.")

            if _is_admin:
                csv_bytes = fc.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 CSV 다운로드", csv_bytes,
                    "forecast_7days.csv", "text/csv",
                    use_container_width=True,
                )

# ── Tab 2: 리스크 상세 ────────────────────────────────────────────────────────
with tab2:
    st.subheader("🌡 리스크 구성 요소 분석")

    if 'latest_risk_signal' in data:
        sig = data['latest_risk_signal'].iloc[0]
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**리스크 팩터 강도**")
            vol_n  = min(sig['volatility_5d'] * 22, 1.0)
            mom_n  = min(abs(sig['momentum_5d']) * 8, 1.0)
            sent_n = min(max(-sig['news_sentiment'], 0), 1.0)
            geo_n  = 1.0 if str(sig['geopolitical_alert']).lower() in ('true', '1') else 0.0
            rs_n   = min(sig['risk_score'] / 3.0, 1.0)

            factors = {
                '📊 실현변동성':    vol_n,
                '📐 가격 방향성':  mom_n,
                '📰 뉴스 부정감성': sent_n,
                '🌍 지정학 리스크': geo_n,
                '⚡ 종합 리스크':   rs_n,
            }

            _bar_clrs = ['#ff6b6b', '#ffd93d', '#ff4757', '#ff6348', '#e74c3c']
            _y = list(factors.keys())
            _x = list(factors.values())
            fig_rf = go.Figure(go.Bar(
                x=_x, y=_y, orientation='h',
                marker_color=_bar_clrs, opacity=0.85,
                text=[f'{v:.2f}' for v in _x],
                textposition='outside',
                textfont=dict(color='#c9d1d9', size=11),
                hovertemplate='%{y}<br>강도: %{x:.2f}<extra></extra>',
            ))
            fig_rf.add_vline(x=0.5, line=dict(color='rgba(255,255,255,0.35)', dash='dash', width=1))
            fig_rf.add_vline(x=0.8, line=dict(color='rgba(231,76,60,0.7)', dash='dash', width=1))
            fig_rf.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                xaxis=dict(range=[0, 1.25], title='Relative Intensity (0–1)',
                           gridcolor='#21262d', tickfont=dict(size=10)),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=11)),
                margin=dict(l=10, r=60, t=10, b=40),
                height=270, showlegend=False,
            )
            st.plotly_chart(fig_rf, use_container_width=True)

        with col_b:
            st.markdown("**주요 지표**")
            st.metric("WTI 현재가", f"${sig['wti_price']:.2f}")
            st.metric("5일 변동성", f"{sig['volatility_5d']*100:.2f}%")
            st.metric("뉴스 감성", f"{float(sig['news_sentiment']):.3f}")
            if _is_admin:
                st.markdown("---")
                st.markdown("**상세 수치 (관리자)**")
                st.json({
                    "날짜":        str(sig['date']),
                    "리스크 레벨": sig['risk_level'],
                    "리스크 점수": float(sig['risk_score']),
                    "모멘텀(5일)": f"{sig['momentum_5d']*100:+.2f}%",
                    "뉴스 기사 수": int(sig['news_count']),
                    "지정학 경보":  str(sig['geopolitical_alert']).lower() in ('true', '1'),
                    "방향성 편향":  float(sig['directional_bias']),
                    "방향 신뢰도":  sig.get('direction_confidence', 'N/A'),
                })

    if _is_admin:
        st.markdown("---")
        sig_csv = (OUTPUT_DIR / 'latest_risk_signal.csv')
        if sig_csv.exists():
            st.download_button("💾 리스크 신호 CSV 다운로드",
                               sig_csv.read_bytes(),
                               "latest_risk_signal.csv", "text/csv")

    st.markdown("---")
    st.markdown("**📅 리스크 레벨 히스토리**" + ("" if _is_admin else " (최근 30일)"))
    if 'risk_history' in data:
        rh = data['risk_history'].copy()
        rh['date'] = pd.to_datetime(rh['date'])
        rh = rh.sort_values('date')
        if not _is_admin:
            rh = rh.tail(30)
        _LEVEL_CLR = {'NORMAL': '#2ecc71', 'CAUTION': '#f39c12',
                      'SURGE_RISK': '#e74c3c', 'DROP_RISK': '#3498db', 'CRITICAL': '#8e44ad'}
        fig_rh = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.05, row_heights=[0.6, 0.4])
        fig_rh.add_trace(go.Scatter(
            x=rh['date'], y=rh['wti_price'],
            mode='lines', name='WTI ($)',
            line=dict(color='#58a6ff', width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>WTI: $%{y:.2f}<extra></extra>',
        ), row=1, col=1)
        for _lv_name, _lv_clr in _LEVEL_CLR.items():
            _lv_mask = rh['risk_level'] == _lv_name
            if _lv_mask.any():
                fig_rh.add_trace(go.Scatter(
                    x=rh[_lv_mask]['date'], y=rh[_lv_mask]['wti_price'],
                    mode='markers',
                    marker=dict(color=_lv_clr, size=10, symbol='circle',
                                line=dict(color='white', width=1)),
                    name=_lv_name,
                    hovertemplate=f'%{{x|%Y-%m-%d}}<br>{_lv_name}<br>$%{{y:.2f}}<extra></extra>',
                ), row=1, col=1)
        _rh_bar_clrs = [_LEVEL_CLR.get(l, '#888') for l in rh['risk_level']]
        fig_rh.add_trace(go.Bar(
            x=rh['date'], y=rh['risk_score'],
            marker_color=_rh_bar_clrs, opacity=0.75, name='리스크 점수',
            hovertemplate='%{x|%Y-%m-%d}<br>점수: %{y:.2f}<extra></extra>',
        ), row=2, col=1)
        fig_rh.add_hline(y=0.3, line=dict(color='#f39c12', dash='dash', width=1),
                         opacity=0.7, row=2, col=1)
        fig_rh.add_hline(y=0.6, line=dict(color='#e74c3c', dash='dash', width=1),
                         opacity=0.7, row=2, col=1)
        fig_rh.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
            font=dict(color='#c9d1d9'),
            height=430, showlegend=True,
            legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                        font=dict(size=9), orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        fig_rh.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
        fig_rh.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
        fig_rh.update_yaxes(title_text='WTI ($)', title_font=dict(size=9), row=1, col=1)
        fig_rh.update_yaxes(title_text='리스크 점수', title_font=dict(size=9), row=2, col=1)
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("파이프라인 실행 후 리스크 히스토리가 표시됩니다.")

# ── Tab 3: 키워드 분석 ───────────────────────────────────────────────────────
with tab3:
    st.subheader("☁ 뉴스 위기 키워드")
    col_wc, col_kw = st.columns([1.3, 1])

    with col_wc:
        wc_img = OUTPUT_DIR / 'wordcloud.png'
        if wc_img.exists():
            st.image(str(wc_img), use_container_width=True,
                     caption='🔴 위기 키워드  |  🔵 일반 키워드')
        else:
            st.info("파이프라인 실행 후 워드클라우드가 표시됩니다.")

    with col_kw:
        if 'crisis_keywords' in data:
            kw = data['crisis_keywords'].head(25).copy()
            kw['분류'] = kw['is_crisis_word'].astype(str).map({'True': '🔴', 'False': '🔵'})
            # 키워드 한글 변환 (oil_risk_mvp의 _KW_TRANSLATE 재사용)
            _KW_KO = {
                'oil':'원유','crude':'원유','brent':'브렌트','wti':'WTI','gas':'가스',
                'energy':'에너지','fuel':'연료','fossil':'화석연료','pipeline':'파이프라인',
                'coal':'석탄','nuclear':'원자력','lng':'LNG','shale':'셰일',
                'opec':'OPEC','iea':'IEA','eia':'EIA',
                'russia':'러시아','russian':'러시아','saudi':'사우디','iran':'이란',
                'iraq':'이라크','china':'중국','ukraine':'우크라이나','israel':'이스라엘',
                'usa':'미국','america':'미국','uae':'UAE','venezuela':'베네수엘라',
                'nigeria':'나이지리아','kuwait':'쿠웨이트','libya':'리비아',
                'australia':'호주','australian':'호주','britain':'영국','england':'영국',
                'europe':'유럽','india':'인도','norway':'노르웨이','canada':'캐나다',
                'war':'전쟁','sanctions':'제재','crisis':'위기','attack':'공격',
                'conflict':'분쟁','ceasefire':'휴전','invasion':'침공',
                'tension':'긴장','tensions':'긴장','military':'군사',
                'supply':'공급','demand':'수요','production':'생산','output':'생산량',
                'inventory':'재고','reserve':'매장량','market':'시장','markets':'시장',
                'price':'가격','prices':'가격','cut':'감산','cuts':'감산',
                'rise':'상승','fall':'하락','surge':'급등','slump':'급락',
                'deal':'협상','trade':'무역','export':'수출','import':'수입',
                'record':'기록','risk':'리스크','threat':'위협','fears':'우려',
                'inflation':'인플레이션','recession':'경기침체','growth':'성장',
                'economy':'경제','dollar':'달러','rate':'금리','bank':'은행',
                'climate':'기후','renewable':'재생에너지','solar':'태양광',
                'carbon':'탄소','emission':'탄소배출','emissions':'탄소배출',
                'green':'친환경','warming':'온난화',
                'trump':'트럼프','biden':'바이든','putin':'푸틴','zelensky':'젤렌스키',
                'global':'세계','world':'세계','international':'국제',
                'government':'정부','election':'선거','policy':'정책',
                'industry':'산업','power':'전력','sea':'해상','north':'북쪽',
                'labour':'노동당','tax':'세금','plan':'계획','change':'변화',
                'hit':'타격','hits':'타격','warns':'경고','food':'식품',
                'covid':'코로나19','pandemic':'팬데믹','brexit':'브렉시트',
                'london':'런던','security':'안보',
            }
            kw['키워드'] = kw['keyword'].apply(
                lambda w: _KW_KO.get(w.lower(), w)
            )
            _kw_cols = ['분류', '키워드', 'count', 'weight'] if _is_admin else ['분류', '키워드', 'count']
            _kw_rename = {'count': '빈도', 'weight': '가중치'}
            st.dataframe(
                kw[[c for c in _kw_cols if c in kw.columns]].rename(columns=_kw_rename),
                hide_index=True, use_container_width=True
            )
            if _is_admin:
                kw_csv = kw.to_csv(index=False).encode('utf-8')
                st.download_button("💾 키워드 CSV 다운로드", kw_csv,
                                   "crisis_keywords.csv", "text/csv")

# ── Tab 4: 모델 성능 ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("📊 모델 성능")

    if 'model_performance' not in data:
        st.info("파이프라인 실행 후 성능 데이터가 표시됩니다.")
    elif not _is_admin:
        pf = data['model_performance']
        _cls = pf[pf['model'].str.contains('Classifier', na=False)]
        c1, c2, c3 = st.columns(3)
        if not _cls.empty:
            _dir   = float(_cls['dir_acc'].iloc[0])   if 'dir_acc'    in _cls.columns else None
            _wfdir = float(_cls['wf_dir_acc'].iloc[0]) if 'wf_dir_acc' in _cls.columns and _cls['wf_dir_acc'].notna().any() else None
            _mae   = float(_cls['mae'].iloc[0])        if 'mae'        in _cls.columns else None
            if _dir is not None:
                c1.metric(
                    '상승/하락 예측 적중률',
                    f"{_dir*100:.0f}%",
                    help=f"10번 예측 중 약 {_dir*10:.0f}번 방향(상승/하락)이 일치"
                )
            if _mae is not None:
                c2.metric(
                    '가격 예측 오차',
                    f"${_mae:.2f}",
                    help="예측 가격과 실제 가격의 평균 차이 (낮을수록 좋음)"
                )
            if _wfdir is not None:
                _wf_help = "과적합 없이 실전 데이터에서 측정한 방향 적중률"
                c3.metric('검증 적중률 (실전)', f"{_wfdir*100:.0f}%", help=_wf_help)
    else:
        pf = data['model_performance']

        # 역할 컬럼 동적 추가
        def _assign_role(m):
            if 'Stacking' in m:     return '앙상블 채택'
            if 'HAR' in m or 'GARCH' in m: return '변동성 진단'
            if 'Prophet' in m:      return '모니터링 전용'
            return '모니터링/비교'
        pf = pf.copy()
        pf.insert(0, '역할', pf['model'].apply(_assign_role))

        # 가격 vs 변동성 분리
        _pf_price = pf[pf['target'] == 'price'].copy()
        _pf_vol   = pf[pf['target'] == 'vol_5d'].copy()
        # 차트용: price 모델 중 비정상 MAE 제외 (Prophet 등)
        _pf_chart = _pf_price[_pf_price['mae'] < 20].copy()

        col_t, col_g = st.columns([1, 1.2])

        with col_t:
            st.caption("**가격 예측 모델** (RMSE·MAE: ↓좋음 | R²: ↑좋음)")
            st.dataframe(_pf_price, hide_index=True, use_container_width=True)
            _mon = _pf_price[_pf_price['mae'] >= 20]
            if not _mon.empty:
                st.caption(f"※ {', '.join(_mon['model'].tolist())} — 모니터링 전용 (차트 제외)")
            st.caption("**변동성 예측 모델**")
            st.dataframe(_pf_vol[[c for c in ['역할','model','rmse','mae','r2','train_r2','overfit_gap'] if c in _pf_vol.columns]],
                         hide_index=True, use_container_width=True)
            if 'overfit_gap' in pf.columns and 'train_r2' in pf.columns:
                for _, _row in pf.dropna(subset=['overfit_gap']).iterrows():
                    _gap = float(_row['overfit_gap'])
                    _tr  = float(_row['train_r2'])
                    _cv  = float(_row['r2'])
                    if _gap > 0.20:
                        st.warning(f"⚠️ **{_row['model']} 과적합 의심** — "
                                   f"훈련 R²={_tr:.3f} vs CV R²={_cv:.3f} (gap={_gap:.3f})")
                    else:
                        st.success(f"✅ **{_row['model']} 과적합 정상** — "
                                   f"훈련 R²={_tr:.3f} / CV R²={_cv:.3f} (gap={_gap:.3f})")

        with col_g:
            _perf_labels = [m.split('(')[0].strip()[:20] for m in _pf_chart['model']]
            fig_perf = make_subplots(rows=1, cols=3,
                                     subplot_titles=['RMSE (↓)', 'MAE (↓)', 'R² (↑)'])
            for _col_i, (_met, _clr) in enumerate(
                    zip(['rmse', 'mae', 'r2'], ['#58a6ff', '#3fb950', '#f0c040']), 1):
                fig_perf.add_trace(go.Bar(
                    x=_perf_labels, y=_pf_chart[_met],
                    marker_color=_clr, opacity=0.85,
                    text=[f'{v:.3f}' for v in _pf_chart[_met]],
                    textposition='outside', textfont=dict(size=9, color='#c9d1d9'),
                    showlegend=False,
                    hovertemplate='%{x}<br>' + _met.upper() + ': %{y:.3f}<extra></extra>',
                ), row=1, col=_col_i)
            fig_perf.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9', size=10),
                height=320, margin=dict(l=30, r=20, t=50, b=60),
            )
            fig_perf.update_xaxes(gridcolor='#21262d', tickfont=dict(size=8),
                                   tickangle=25)
            fig_perf.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig_perf.update_annotations(font=dict(color='#e6edf3', size=10))
            st.plotly_chart(fig_perf, use_container_width=True)
            st.caption("차트: 가격 예측 모델 (Prophet 등 모니터링 전용 제외)")

        # ── 피처 중요도 (관리자)
        st.markdown("---")
        _FEAT_KO = {
            'RV_5d':'5일 실현변동성', 'RV_22d':'22일 실현변동성', 'RV_1d':'1일 실현변동성',
            'return_1d':'1일 수익률', 'vol_5d':'5일 변동성', 'vol_21d':'21일 변동성',
            'price_vs_ma5':'MA5 대비 가격', 'price_vs_ma21':'MA21 대비 가격',
            'bb_position':'볼린저밴드 위치', 'Brent':'브렌트 가격',
            'DXY':'달러인덱스', 'WTI':'WTI 가격', 'VIX':'VIX 공포지수',
            'OVX':'OVX 원유변동성', 'dxy_change':'달러인덱스 변화', 'vix_change':'VIX 변화',
            'vix_zscore':'VIX Z-스코어', 'ovx_change':'OVX 변화', 'ovx_zscore':'OVX Z-스코어',
            'demand_shock':'수요 충격', 'supply_shock':'공급 충격',
            'inv_chg_zscore':'재고 변화 Z-스코어', 'inv_lvl_zscore':'재고 수준 Z-스코어',
            'news_sentiment':'뉴스 감성', 'news_sentiment_smooth':'뉴스 감성(평활)',
            'news_count':'뉴스 건수', 'geo_dummy':'지정학 위기 더미',
            'gpr_zscore':'GPR Z-스코어', 'fear_composite':'공포 복합지수',
            'regime':'시장 국면', 'futures_spread':'선물 커브 스프레드',
            'contango_dummy':'콘탱고 더미', 'covid_dummy':'COVID 더미',
        }
        if 'feature_importance' in data:
            fi = data['feature_importance'].head(15).copy()
            fi['feature_ko'] = fi['feature'].map(_FEAT_KO).fillna(fi['feature'])
            fi = fi.sort_values('importance')
            fig_fi = go.Figure(go.Bar(
                x=fi['importance'], y=fi['feature_ko'], orientation='h',
                marker_color='#58a6ff', opacity=0.85,
                text=[f'{v:.3f}' for v in fi['importance']],
                textposition='outside', textfont=dict(color='#c9d1d9', size=10),
                hovertemplate='%{y}<br>중요도: %{x:.4f}<extra></extra>',
            ))
            fig_fi.update_layout(
                title=dict(text='XGBoost-HAR 피처 중요도 (Top 15)',
                           font=dict(color='#e6edf3', size=11)),
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                xaxis=dict(title='Importance', gridcolor='#21262d', tickfont=dict(size=9)),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
                margin=dict(l=10, r=70, t=50, b=40),
                height=430, showlegend=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("파이프라인 실행 후 피처 중요도가 표시됩니다.")

        st.markdown("---")
        st.markdown("""
        **모델 설명**
        | 모델 | 역할 | 특징 |
        |------|------|------|
        | **XGBoost-HAR** | 변동성(리스크) 예측 | HAR 구성요소 + 뉴스/지정학 외생변수 |
        | **SARIMAX** | 7일 가격 예측 | AR(2,1,2) × 주간 계절성 + 외생변수 |
        | **XGBoost-Classifier** | 최종 방향 판단 | SVM+CEEMDAN 앙상블 → 상승/하락 확률 |
        """)

# ── Tab 5: 예측 오차 로그 ─────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 예측 정확도")

    if 'prediction_log' not in data:
        st.info("파이프라인 실행 후 데이터가 표시됩니다.")
    else:
        pl = data['prediction_log'].copy()
        bt = pl[pl['type'] == 'backtest'].copy()
        lv = pl[pl['type'].isin(['live', 'gap'])].copy()   # gap-fill 항목 포함
        live_confirmed = lv[lv['actual_price'].notna()]

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        if not bt.empty and bt['price_error'].notna().any():
            col_s1.metric("백테스트 MAE", f"${bt['price_error'].abs().mean():.2f}")
            if 'price_error_pct' in bt.columns and bt['price_error_pct'].notna().any():
                col_s2.metric("백테스트 MAPE", f"{bt['price_error_pct'].abs().mean():.2f}%")
        if not live_confirmed.empty:
            col_s3.metric("실시간 MAE", f"${live_confirmed['price_error'].abs().mean():.2f}")
            if 'price_error_pct' in live_confirmed.columns and live_confirmed['price_error_pct'].notna().any():
                col_s4.metric("실시간 MAPE", f"{live_confirmed['price_error_pct'].abs().mean():.2f}%")

        if not _is_admin:
            # 실시간 예측 기록 테이블
            if not live_confirmed.empty:
                st.markdown("---")
                st.markdown("**실시간 예측 기록**")
                _lv_cols = [c for c in ['date', 'type', 'forecast_price', 'actual_price', 'price_error_pct']
                            if c in live_confirmed.columns]
                _lv_show = live_confirmed[_lv_cols].tail(30).copy()
                if 'type' in _lv_show.columns:
                    _lv_show['type'] = _lv_show['type'].map({'live': '실측', 'gap': '갭채움'})
                _lv_show.columns = [{'date':'날짜','type':'구분','forecast_price':'예측가($)',
                                     'actual_price':'실제가($)',
                                     'price_error_pct':'오차(%)'}.get(c, c)
                                    for c in _lv_show.columns]
                st.dataframe(_lv_show, hide_index=True, use_container_width=True)
            st.info("ℹ️ 상세 분석 (드리프트 감지, 오차 추이, 방향성 정확도)은 관리자 전용입니다.")
            st.stop()

        # ── 드리프트 경고 (live MAPE > backtest MAPE × 2)
        if (not bt.empty and 'price_error_pct' in bt.columns and bt['price_error_pct'].notna().any()
                and not live_confirmed.empty and 'price_error_pct' in live_confirmed.columns
                and live_confirmed['price_error_pct'].notna().any()):
            bt_mape  = bt['price_error_pct'].abs().mean()
            lv_mape  = live_confirmed['price_error_pct'].abs().mean()
            if lv_mape > bt_mape * 2:
                st.warning(
                    f"⚠️ **모델 드리프트 감지** — 실시간 MAPE({lv_mape:.1f}%)가 "
                    f"백테스트 MAPE({bt_mape:.1f}%)의 2배 초과. 재학습 또는 피처 점검 필요."
                )

        # ── Bias correction 경고
        if 'forecast_7days' in data:
            _bc = data['forecast_7days'].get('bias_correction', pd.Series([0])).iloc[0]
            try:
                _bc = float(_bc)
                if abs(_bc) > 3.0:
                    _dir = "과소예측" if _bc > 0 else "과대예측"
                    st.info(
                        f"ℹ️ **Bias 보정 활성** — 현재 예측에 {_bc:+.2f}$ 보정 적용 중 "
                        f"({_dir} 패턴). live 데이터 누적 시 자동 조정됩니다."
                    )
            except (TypeError, ValueError):
                pass

        st.markdown("---")

        # 가격 오차 추이 차트
        if not bt.empty and bt['price_error'].notna().any():
            st.markdown("**SARIMAX 가격 예측 오차 추이 (백테스트 60일)**")
            bt_plot = bt.dropna(subset=['price_error']).copy()
            bt_plot['date'] = pd.to_datetime(bt_plot['date'])
            fig_err = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.05, row_heights=[0.55, 0.45])
            fig_err.add_trace(go.Scatter(
                x=bt_plot['date'], y=bt_plot['actual_price'],
                mode='lines', name='실제 WTI',
                line=dict(color='#58a6ff', width=1.5),
                hovertemplate='%{x|%m/%d}<br>실제: $%{y:.2f}<extra></extra>',
            ), row=1, col=1)
            if 'sarimax_pred' in bt_plot.columns:
                fig_err.add_trace(go.Scatter(
                    x=bt_plot['date'], y=bt_plot['sarimax_pred'],
                    mode='lines', name='SARIMAX 예측',
                    line=dict(color='#f0c040', width=1.5, dash='dash'),
                    hovertemplate='%{x|%m/%d}<br>예측: $%{y:.2f}<extra></extra>',
                ), row=1, col=1)
            _err_clrs = ['#3fb950' if e >= 0 else '#f85149' for e in bt_plot['price_error']]
            fig_err.add_trace(go.Bar(
                x=bt_plot['date'], y=bt_plot['price_error'],
                marker_color=_err_clrs, opacity=0.85, name='오차 ($)',
                hovertemplate='%{x|%m/%d}<br>오차: $%{y:.2f}<extra></extra>',
            ), row=2, col=1)
            fig_err.add_hline(y=0, line=dict(color='rgba(255,255,255,0.4)', width=0.8),
                              row=2, col=1)
            fig_err.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                height=380, hovermode='x unified',
                legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                            font=dict(size=10)),
                margin=dict(l=60, r=20, t=20, b=40),
            )
            fig_err.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig_err.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig_err.update_yaxes(title_text='WTI ($)', title_font=dict(size=9), row=1, col=1)
            fig_err.update_yaxes(title_text='오차 ($)', title_font=dict(size=9), row=2, col=1)
            st.plotly_chart(fig_err, use_container_width=True)

        # ── 방향성 정확도 / 누적 오차 / 롤링 MAPE ────────────────────────────
        if not bt.empty and bt['price_error'].notna().any():
            bt_a = bt.dropna(subset=['price_error', 'actual_price', 'sarimax_pred']).copy()
            bt_a['date'] = pd.to_datetime(bt_a['date'])
            bt_a = bt_a.sort_values('date').reset_index(drop=True)

            # 방향성 정확도
            bt_a['actual_dir']  = bt_a['actual_price'].diff().apply(lambda x: np.nan if pd.isna(x) else (1 if x > 0 else -1))
            bt_a['pred_dir']    = bt_a['sarimax_pred'].diff().apply(lambda x: np.nan if pd.isna(x) else (1 if x > 0 else -1))
            bt_a['dir_correct'] = (bt_a['actual_dir'] == bt_a['pred_dir']).astype(float).where(bt_a['actual_dir'].notna())
            dir_acc = bt_a['dir_correct'].dropna().mean() * 100

            bt_a['abs_pct_err'] = bt_a['price_error_pct'].abs() if 'price_error_pct' in bt_a.columns else np.nan
            rolling_mape = bt_a['abs_pct_err'].rolling(10).mean() if 'price_error_pct' in bt_a.columns else pd.Series(dtype=float)
            bt_a['cum_abs_err'] = bt_a['price_error'].abs().cumsum()

            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("방향성 정확도 (상승/하락)", f"{dir_acc:.1f}%",
                          help="예측 방향(상승/하락)이 실제와 일치한 비율")
            col_d2.metric("최근 10일 롤링 MAPE",
                          f"{rolling_mape.dropna().iloc[-1]:.2f}%" if rolling_mape.notna().any() else "—")
            col_d3.metric("누적 절대 오차", f"${bt_a['cum_abs_err'].iloc[-1]:.2f}")

            mean_mape2 = bt_a['abs_pct_err'].mean()
            fig2 = make_subplots(rows=1, cols=2,
                                  subplot_titles=['누적 절대 오차 ($)', '롤링 MAPE — 10일 윈도우 (%)'])
            fig2.add_trace(go.Scatter(
                x=bt_a['date'], y=bt_a['cum_abs_err'],
                mode='lines', name='누적 오차',
                fill='tozeroy', fillcolor='rgba(240,192,64,0.12)',
                line=dict(color='#f0c040', width=1.5),
                hovertemplate='%{x|%m/%d}<br>누적: $%{y:.2f}<extra></extra>',
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=bt_a['date'], y=rolling_mape,
                mode='lines', name='10일 롤링 MAPE',
                line=dict(color='#3fb950', width=1.5),
                hovertemplate='%{x|%m/%d}<br>MAPE: %{y:.2f}%<extra></extra>',
            ), row=1, col=2)
            fig2.add_hline(y=mean_mape2, line=dict(color='#ff6b6b', dash='dash', width=1),
                           annotation_text=f'평균 {mean_mape2:.2f}%',
                           annotation_font=dict(color='#ff6b6b', size=9),
                           row=1, col=2)
            fig2.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                height=300, margin=dict(l=50, r=20, t=50, b=40),
                showlegend=False,
            )
            fig2.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig2.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig2.update_annotations(font=dict(color='#e6edf3', size=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # 테이블
        col_bt, col_lv = st.columns(2)
        with col_bt:
            st.markdown("**백테스트 (최근 10일)**")
            _bt_show_cols = [c for c in ['date', 'sarimax_pred', 'actual_price', 'price_error', 'price_error_pct'] if c in bt.columns]
            show_bt = bt.tail(10)[_bt_show_cols].rename(columns={
                'date': '날짜', 'sarimax_pred': '예측가($)',
                'actual_price': '실제가($)', 'price_error': '오차($)',
                'price_error_pct': '오차(%)'
            })
            st.dataframe(show_bt, hide_index=True, use_container_width=True)

        with col_lv:
            st.markdown("**실시간 예측 기록**")
            if lv.empty:
                st.info("실시간 예측 기록이 없습니다.")
            else:
                _lv_show_cols = [c for c in ['date', 'sarimax_pred', 'actual_price', 'price_error', 'price_error_pct'] if c in lv.columns]
                show_lv = lv[_lv_show_cols].rename(columns={
                    'date': '날짜', 'sarimax_pred': '예측가($)',
                    'actual_price': '실제가($)', 'price_error': '오차($)',
                    'price_error_pct': '오차(%)'
                })
                st.dataframe(show_lv, hide_index=True, use_container_width=True)

        # ── 예측 정확도 트렌드 차트 (backtest 30일 rolling MAPE)
        st.markdown("---")
        st.markdown("**📈 예측 정확도 트렌드 (Rolling 10일 MAPE)**")
        bt_trend = bt.dropna(subset=['price_error_pct', 'date']).copy()
        bt_trend['date'] = pd.to_datetime(bt_trend['date'])
        bt_trend = bt_trend.sort_values('date').tail(60)

        if len(bt_trend) >= 10:
            bt_trend['abs_pct'] = bt_trend['price_error_pct'].abs()
            bt_trend['rolling_mape'] = bt_trend['abs_pct'].rolling(10).mean()
            bt_trend['rolling_mae']  = bt_trend['price_error'].abs().rolling(10).mean()

            _avg_mape = bt_trend['abs_pct'].mean()
            _avg_mae  = bt_trend['price_error'].abs().mean()
            fig_t = make_subplots(rows=1, cols=2,
                                   subplot_titles=['MAPE 트렌드 (%)', 'MAE 트렌드 ($)'])
            fig_t.add_trace(go.Scatter(
                x=bt_trend['date'], y=bt_trend['rolling_mape'],
                mode='lines', name='10일 Rolling MAPE',
                line=dict(color='#3fb950', width=1.5),
                hovertemplate='%{x|%m/%d}<br>MAPE: %{y:.2f}%<extra></extra>',
            ), row=1, col=1)
            fig_t.add_hline(y=_avg_mape, line=dict(color='#f0c040', dash='dash', width=1),
                            annotation_text=f'평균 {_avg_mape:.1f}%',
                            annotation_font=dict(color='#f0c040', size=9), row=1, col=1)
            fig_t.add_trace(go.Scatter(
                x=bt_trend['date'], y=bt_trend['rolling_mae'],
                mode='lines', name='10일 Rolling MAE',
                line=dict(color='#58a6ff', width=1.5),
                hovertemplate='%{x|%m/%d}<br>MAE: $%{y:.2f}<extra></extra>',
            ), row=1, col=2)
            fig_t.add_hline(y=_avg_mae, line=dict(color='#f0c040', dash='dash', width=1),
                            annotation_text=f'평균 ${_avg_mae:.2f}',
                            annotation_font=dict(color='#f0c040', size=9), row=1, col=2)
            if not live_confirmed.empty:
                _lcp = live_confirmed.copy()
                _lcp['date'] = pd.to_datetime(_lcp['date'])
                if 'price_error_pct' in _lcp.columns:
                    fig_t.add_trace(go.Scatter(
                        x=_lcp['date'], y=_lcp['price_error_pct'].abs(),
                        mode='markers', name='Live 실측',
                        marker=dict(color='#f85149', size=7, symbol='circle'),
                        hovertemplate='%{x|%m/%d}<br>Live MAPE: %{y:.2f}%<extra></extra>',
                    ), row=1, col=1)
                if 'price_error' in _lcp.columns:
                    fig_t.add_trace(go.Scatter(
                        x=_lcp['date'], y=_lcp['price_error'].abs(),
                        mode='markers', name='Live 실측',
                        marker=dict(color='#f85149', size=7, symbol='circle'),
                        hovertemplate='%{x|%m/%d}<br>Live MAE: $%{y:.2f}<extra></extra>',
                        showlegend=False,
                    ), row=1, col=2)
            fig_t.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'),
                height=300, margin=dict(l=50, r=20, t=50, b=40),
                legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                            font=dict(size=9)),
            )
            fig_t.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig_t.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
            fig_t.update_annotations(font=dict(color='#e6edf3', size=10))
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("트렌드 차트는 백테스트 10일 이상 데이터 필요")

        # 다운로드
        csv_bytes = pl.to_csv(index=False).encode('utf-8')
        st.download_button("💾 전체 로그 CSV 다운로드", csv_bytes,
                           "prediction_log.csv", "text/csv")


# ── Tab 관리자 ───────────────────────────────────────────────────────────────
if _is_admin and tab_admin is not None:
    with tab_admin:
        st.subheader('🔧 관리자 패널')
        adm1, adm2, adm3, adm4 = st.tabs(['👤 사용자 관리', '⚙️ 시스템 모니터링', '🚀 파이프라인 실행', '📧 이메일 알림'])

        # ── 사용자 관리 ──────────────────────────────────────────────────────
        with adm1:
            _cfg_path = Path(__file__).parent / 'auth_config.yaml'
            with open(_cfg_path, encoding='utf-8') as _f:
                _cfg = yaml.safe_load(_f)
            _users = _cfg['credentials']['usernames']

            st.markdown('#### 계정 목록')
            _user_rows = [{'아이디': k, '이름': v.get('name', ''), '이메일': v.get('email', '')}
                          for k, v in _users.items()]
            st.dataframe(_user_rows, use_container_width=True)

            st.markdown('---')
            st.markdown('#### 계정 추가')
            with st.form('add_user_form', clear_on_submit=True):
                _new_id = st.text_input('아이디')
                _new_nm = st.text_input('이름')
                _new_em = st.text_input('이메일')
                _new_pw = st.text_input('비밀번호', type='password')
                if st.form_submit_button('추가'):
                    if _new_id and _new_pw:
                        if _new_id in _users:
                            st.error('이미 존재하는 아이디입니다.')
                        else:
                            _users[_new_id] = {'name': _new_nm, 'email': _new_em, 'password': _new_pw}
                            with open(_cfg_path, 'w', encoding='utf-8') as _f:
                                yaml.dump(_cfg, _f, allow_unicode=True)
                            st.success(f'{_new_id} 계정이 추가됐습니다. 페이지를 새로고침하세요.')
                    else:
                        st.warning('아이디와 비밀번호를 입력하세요.')

            st.markdown('---')
            st.markdown('#### 구독 만료일 설정')
            _exp_target = st.selectbox('계정', list(_users.keys()), key='exp_sel')
            _cur_exp = _users[_exp_target].get('subscription_expiry', '2026-12-31')
            try:
                _cur_exp_date = datetime.datetime.strptime(_cur_exp, '%Y-%m-%d').date()
            except Exception:
                _cur_exp_date = datetime.date.today()
            _new_exp = st.date_input('만료일', value=_cur_exp_date, key='exp_date')
            if st.button('만료일 저장'):
                _users[_exp_target]['subscription_expiry'] = str(_new_exp)
                with open(_cfg_path, 'w', encoding='utf-8') as _f:
                    yaml.dump(_cfg, _f, allow_unicode=True)
                st.success(f'{_exp_target} 만료일 → {_new_exp}')

            st.markdown('---')
            st.markdown('#### 비밀번호 초기화')
            _all_users = list(_users.keys())
            if _all_users:
                _pw_target = st.selectbox('계정 선택', _all_users, key='pw_sel')
                with st.form('pw_reset_form', clear_on_submit=True):
                    _new_pw_r = st.text_input('새 비밀번호', type='password')
                    _confirm  = st.text_input('비밀번호 확인', type='password')
                    if st.form_submit_button('변경'):
                        if not _new_pw_r:
                            st.warning('비밀번호를 입력하세요.')
                        elif _new_pw_r != _confirm:
                            st.error('비밀번호가 일치하지 않습니다.')
                        else:
                            _users[_pw_target]['password'] = _new_pw_r
                            with open(_cfg_path, 'w', encoding='utf-8') as _f:
                                yaml.dump(_cfg, _f, allow_unicode=True)
                            st.success(f'{_pw_target} 비밀번호가 변경됐습니다.')

            st.markdown('---')
            st.markdown('#### 계정 삭제')
            _deletable = [k for k in _users if k != 'admin']
            if _deletable:
                _del_target = st.selectbox('삭제할 계정', _deletable)
                if st.button('삭제', type='secondary'):
                    del _users[_del_target]
                    with open(_cfg_path, 'w', encoding='utf-8') as _f:
                        yaml.dump(_cfg, _f, allow_unicode=True)
                    st.success(f'{_del_target} 계정이 삭제됐습니다. 페이지를 새로고침하세요.')
            else:
                st.info('삭제 가능한 계정이 없습니다.')

        # ── 시스템 모니터링 ──────────────────────────────────────────────────
        with adm2:
            _log_path = OUTPUT_DIR / 'pipeline_run.log'
            _fc_path  = OUTPUT_DIR / 'forecast_7days.csv'

            st.markdown('#### 시스템 상태')
            m1, m2, m3 = st.columns(3)

            if _log_path.exists():
                _last_run = datetime.datetime.fromtimestamp(os.path.getmtime(_log_path)).strftime('%Y-%m-%d %H:%M')
                m1.metric('마지막 파이프라인 실행', _last_run)
            else:
                m1.metric('마지막 파이프라인 실행', '기록 없음')

            if _fc_path.exists():
                _fc_tmp = pd.read_csv(_fc_path)
                m2.metric('예측 기준일 (D+1)', _fc_tmp['date'].iloc[0] if 'date' in _fc_tmp.columns else '—')

            if 'model_performance' in data:
                _mp = data['model_performance']
                _cls_m = _mp[_mp['model'].str.contains('Classifier', na=False)]
                if not _cls_m.empty and 'dir_acc' in _cls_m.columns:
                    _dir = float(_cls_m['dir_acc'].iloc[0])
                    m3.metric('방향성 정확도', f'{_dir*100:.1f}%',
                              delta='정상' if _dir >= 0.52 else '롤백 기준 미달',
                              delta_color='normal' if _dir >= 0.52 else 'inverse')

            st.markdown('---')
            st.markdown('#### 모델 성능 상세')
            if 'model_performance' in data:
                st.dataframe(data['model_performance'], use_container_width=True)
                _mp = data['model_performance']
                _har = _mp[_mp['model'].str.contains('HAR', na=False)]
                _stk = _mp[_mp['model'].str.contains('Stacking', na=False)]
                _xgb = _mp[_mp['model'].str.contains('XGBoost-Return', na=False)]
                _warns = []
                if not _har.empty and float(_har['r2'].iloc[0]) < 0.48:
                    _warns.append(f"HAR R²={float(_har['r2'].iloc[0]):.3f} < 0.48")
                if not _stk.empty and float(_stk['r2'].iloc[0]) < 0.68:
                    _warns.append(f"Stacking R²={float(_stk['r2'].iloc[0]):.3f} < 0.68")
                if not _xgb.empty and 'dir_acc' in _xgb.columns and float(_xgb['dir_acc'].iloc[0]) < 0.52:
                    _warns.append(f"dir_acc={float(_xgb['dir_acc'].iloc[0])*100:.1f}% < 52%")
                for w in _warns:
                    st.warning(f'⚠️ {w} (롤백 기준)')
                if not _warns:
                    st.success('✅ 모든 모델 롤백 기준 충족')


            st.markdown('---')
            st.markdown('#### 자동 실행 스케줄러')
            _sch_name = 'OilRiskPipeline'

            def _sch_status():
                r = subprocess.run(['schtasks', '/Query', '/TN', _sch_name, '/FO', 'LIST'],
                             capture_output=True, encoding='cp949', errors='replace')
                return r.returncode == 0, r.stdout

            _sch_ok, _sch_info = _sch_status()
            if _sch_ok:
                st.success('✅ 스케줄 등록됨')
                for line in _sch_info.splitlines():
                    if any(k in line for k in ('다음 실행', '상태', 'Next Run', 'Status', '작업 이름', 'TaskName')):
                        st.caption(line.strip())
            else:
                st.warning('⚠️ 스케줄 미등록')

            col_h, col_m, col_btn = st.columns([1, 1, 2])
            _h = col_h.number_input('시', 0, 23, 7, key='sch_h')
            _m = col_m.number_input('분', 0, 59, 0, key='sch_m')
            with col_btn:
                st.markdown('<br>', unsafe_allow_html=True)
                if st.button('등록/갱신'):
                    _sched_py = Path(__file__).parent / 'setup_scheduler.py'
                    _r = subprocess.run([sys.executable, str(_sched_py), 'install', str(_h), str(_m)],
                                  capture_output=True, text=True, encoding='utf-8', errors='replace')
                    st.success(_r.stdout.strip() or '등록 완료') if _r.returncode == 0 else st.error(_r.stderr)

            if st.button('스케줄 해제', type='secondary'):
                _sched_py = Path(__file__).parent / 'setup_scheduler.py'
                _r = subprocess.run([sys.executable, str(_sched_py), 'remove'],
                              capture_output=True, text=True, encoding='utf-8', errors='replace')
                st.info(_r.stdout.strip() or '해제 완료')

            st.markdown('---')
            st.markdown('#### 파이프라인 로그')
            if _log_path.exists():
                with open(_log_path, encoding='utf-8', errors='replace') as _f:
                    st.text_area('pipeline_run.log', _f.read()[-3000:], height=250)
            else:
                st.info('로그 파일 없음')

        # ── 파이프라인 실행 ──────────────────────────────────────────────────
        with adm3:
            st.markdown('#### 수동 파이프라인 실행')
            st.caption('oil_risk_mvp.py 를 즉시 실행합니다. 완료까지 2~5분 소요됩니다.')

            if 'pipeline_running' not in st.session_state:
                st.session_state['pipeline_running'] = False
            if 'pipeline_output' not in st.session_state:
                st.session_state['pipeline_output'] = ''

            if st.button('▶ 파이프라인 실행', disabled=st.session_state['pipeline_running']):
                st.session_state['pipeline_running'] = True
                st.session_state['pipeline_output'] = '실행 중...'
                _pipe_path = Path(__file__).parent / 'oil_risk_mvp.py'
                try:
                    _result = subprocess.run(
                        [sys.executable, str(_pipe_path)],
                        cwd=str(Path(__file__).parent),
                        capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=600,
                    )
                    _out = (_result.stdout or '') + (_result.stderr or '')
                    st.session_state['pipeline_output'] = _out[-4000:] if len(_out) > 4000 else _out
                except subprocess.TimeoutExpired:
                    st.session_state['pipeline_output'] = '타임아웃 (10분 초과)'
                except Exception as _e:
                    st.session_state['pipeline_output'] = f'오류: {_e}'
                finally:
                    st.session_state['pipeline_running'] = False
                st.rerun()

            if st.session_state['pipeline_output']:
                st.markdown(f"**{'실행 중...' if st.session_state['pipeline_running'] else '실행 완료'}**")
                st.text_area('출력 로그', st.session_state['pipeline_output'], height=350)


        # ── 이메일 알림 설정 ─────────────────────────────────────────────
        with adm4:
            _env_path = Path(__file__).parent / '.env'

            def _read_env():
                env = {}
                if _env_path.exists():
                    for line in _env_path.read_text(encoding='utf-8').splitlines():
                        if '=' in line and not line.startswith('#'):
                            k, v = line.split('=', 1)
                            env[k.strip()] = v.strip()
                return env

            def _write_env_key(key, val):
                lines = _env_path.read_text(encoding='utf-8').splitlines() if _env_path.exists() else []
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith(key + '='):
                        lines[i] = f'{key}={val}'
                        updated = True
                        break
                if not updated:
                    lines.append(f'{key}={val}')
                _env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

            _env = _read_env()
            st.markdown('#### 현재 설정')
            st.markdown(f"발신 계정: `{_env.get('SMTP_USER', '미설정')}`")
            st.markdown(f"수신 주소: `{_env.get('ALERT_TO', '미설정')}`")
            st.caption('SURGE_RISK / DROP_RISK 감지 시 자동 발송')

            st.markdown('---')
            st.markdown('#### 수신 주소 변경')
            with st.form('email_cfg_form'):
                _new_to = st.text_input('수신 이메일', value=_env.get('ALERT_TO', ''))
                _new_pw = st.text_input('Gmail 앱 비밀번호 (변경 시만)', type='password')
                if st.form_submit_button('저장'):
                    _write_env_key('ALERT_TO', _new_to)
                    if _new_pw:
                        _write_env_key('SMTP_PASSWORD', _new_pw)
                    st.success('저장 완료. 다음 파이프라인 실행부터 적용됩니다.')

            st.markdown('---')
            st.markdown('#### 테스트 발송')
            if st.button('📧 테스트 이메일 발송'):
                _spec = _ilu.spec_from_file_location('mvp', str(Path(__file__).parent / 'oil_risk_mvp.py'))
                _mvp  = _ilu.module_from_spec(_spec)
                try:
                    _spec.loader.exec_module(_mvp)
                    _test_sig = {'risk_level': 'SURGE_RISK', 'wti_price': 0.0,
                                 'risk_score': 0.0, 'volatility_5d': 0.0,
                                 'news_sentiment': 0.0, 'geopolitical_alert': False}
                    ok = _mvp.send_risk_alert(_test_sig, None)
                    if ok:
                        _to = _read_env().get('ALERT_TO', '')
                        st.success(f'발송 완료 → {_to}')
                    else:
                        st.error('발송 실패. SMTP 설정(.env)을 확인하세요.')
                except Exception as _e:
                    st.error(f'오류: {_e}')

st.markdown("---")
st.caption("국제 유가 리스크 예측 시스템 MVP  |  XGBoost-HAR + SARIMAX Ensemble  |  News Sentiment + Geopolitical Risk")
