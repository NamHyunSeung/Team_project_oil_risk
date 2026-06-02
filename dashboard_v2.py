"""
국제 유가 리스크 예측 시스템 — Streamlit 대시보드 v2
실행 방법: streamlit run dashboard_v2.py

사용자: 단일 스크롤 페이지 (탭 없음)
관리자: 5탭 (리스크현황 / 모델모니터링 / 예측오차 / 파이프라인 / 사용자관리)
"""

import streamlit as st
import yaml
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ── 한글 폰트
def _set_korean_font():
    candidates = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'NanumBarunGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
    path = Path('C:/Windows/Fonts/malgun.ttf')
    if path.exists():
        fm.fontManager.addfont(str(path))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

_set_korean_font()

OUTPUT_DIR = Path("output")

st.set_page_config(
    page_title="국제 유가 리스크 예측 시스템",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 로그인
_AUTH_CFG = Path(__file__).parent / "config/auth_config.yaml"

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

if not st.session_state.get("authentication_status"):
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
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 리스크 신호 (정상 ~ 급등위험)</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 뉴스 감성 분석</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ OVX 원유변동성 · 지정학 알람</p>
  <p style='color:#c9d1d9; font-size:0.85rem; margin:6px 0;'>✓ 이메일 알림 (급등/급락)</p>
</div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    _authenticator.login(
        location="main",
        fields={"Form name": "🔐  로그인", "Username": "아이디", "Password": "비밀번호", "Login": "로그인"},
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

# ── 구독 만료 체크
try:
    with open(Path(__file__).parent / 'config/auth_config.yaml', encoding='utf-8') as _f2:
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

# ── 사이드바 (최소)
with st.sidebar:
    st.markdown(f"**{_name}** 님")
    _authenticator.logout("로그아웃", location="sidebar")

# ── 자동 새로고침 (5분 간격으로 run_meta.json 변경 감지)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh_v2")
except ImportError:
    st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

_meta_path = OUTPUT_DIR / 'run_meta.json'
if _meta_path.exists():
    _mtime = os.path.getmtime(_meta_path)
    if 'last_meta_mtime' not in st.session_state:
        st.session_state['last_meta_mtime'] = _mtime
    elif st.session_state['last_meta_mtime'] != _mtime:
        st.session_state['last_meta_mtime'] = _mtime
        st.rerun()

# ── 스타일
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }
.block-container                   { padding-top: 1.2rem; }
h1, h2, h3, p, label, .stMarkdown { color: #e6edf3 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1c2433, #161b22);
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #30363d;
}
button[data-baseweb="tab"] {
    color: #8b949e !important;
    border-radius: 6px 6px 0 0 !important;
    font-size: 0.88rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: rgba(88,166,255,0.07) !important;
    font-weight: 600 !important;
}
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.8rem !important; }
    [data-testid="column"] { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ── 상수
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
_KW_KO = {
    'oil':'원유','crude':'원유','brent':'브렌트','wti':'WTI','gas':'가스',
    'energy':'에너지','fuel':'연료','opec':'OPEC','russia':'러시아',
    'saudi':'사우디','iran':'이란','iraq':'이라크','china':'중국',
    'war':'전쟁','sanctions':'제재','crisis':'위기','attack':'공격',
    'supply':'공급','demand':'수요','production':'생산','inventory':'재고',
    'price':'가격','prices':'가격','cut':'감산','rise':'상승','fall':'하락',
    'surge':'급등','slump':'급락','trump':'트럼프','putin':'푸틴',
    'inflation':'인플레이션','recession':'경기침체','growth':'성장',
    'ukraine':'우크라이나','israel':'이스라엘','nigeria':'나이지리아',
}
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

# ── 데이터 로드 (캐시 없음 — 파이프라인 실행 즉시 반영)
def load_all():
    out = {}
    for name in ['model_performance', 'forecast_7days', 'latest_risk_signal',
                 'crisis_keywords', 'feature_importance', 'risk_history',
                 'live_gap_monthly', 'live_gap_spikes']:
        p = OUTPUT_DIR / f'{name}.csv'
        if p.exists():
            out[name] = pd.read_csv(p)
    p_log = OUTPUT_DIR / 'prediction_log.csv'
    if p_log.exists():
        out['prediction_log'] = pd.read_csv(p_log).tail(1000).reset_index(drop=True)
    p_snap = OUTPUT_DIR / 'forecast_snapshots.csv'
    if p_snap.exists():
        out['forecast_snapshots'] = pd.read_csv(p_snap)
    return out

if not (OUTPUT_DIR / 'latest_risk_signal.csv').exists():
    st.markdown("# 🛢 국제 유가 리스크 예측 시스템")
    st.warning("⚠️ 분석 결과가 없습니다.")
    if _is_admin:
        st.info("관리자: 파이프라인 탭에서 실행하세요. (로그인 후 관리자 페이지 → 파이프라인 탭)")
    st.stop()

data = load_all()


# ─────────────────────────────────────────────────────────────────────────────
# 공통 렌더링 함수
# ─────────────────────────────────────────────────────────────────────────────

def _render_risk_hero(sig):
    level = sig.get('risk_level', 'UNKNOWN')
    col   = RISK_COLOR.get(level, '#888')
    em, lbl = RISK_LABEL.get(level, ('⚪', '알 수 없음'))
    _fc_rel = str(sig.get('forecast_reliability', '') or '').upper()
    _rel_clr = '#f85149' if _fc_rel == 'LOW' else ('#ffa657' if _fc_rel == 'MEDIUM' else '#3fb950')
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{col}18,{col}0a);
                border:2px solid {col};border-radius:14px;
                padding:18px 24px;margin-bottom:18px;">
      <h2 style="color:{col};margin:0;font-size:1.8rem">{em} 현재 리스크: {lbl}</h2>
      <p style="color:#8b949e;margin:6px 0 0">
        기준일: {sig.get('date', '—')} &nbsp;|&nbsp;
        예측신뢰도: <b style="color:{_rel_clr}">{_fc_rel or '—'}</b>
      </p>
    </div>
    """, unsafe_allow_html=True)


def _render_key_metrics(sig):
    _surge_p  = float(sig.get('surge_prob_3d', 0.0))
    _ovx_val  = float(sig.get('ovx_level', 0) or 0)
    _ovx_al   = str(sig.get('ovx_alarm', 'NORMAL')).upper()  # 'HIGH'/'ELEVATED'/'NORMAL'
    _hedge    = float(sig.get('hedge_ratio', 0.0))
    _down     = float(sig.get('downside_risk_pct', 0.0))
    _up       = float(sig.get('upside_risk_pct', 0.0))
    _dir_bias = float(sig.get('directional_bias', 0.0))
    _dir_conf = str(sig.get('direction_confidence', '—'))
    _geo      = str(sig.get('geopolitical_alert', '')).lower() in ('true', '1')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WTI 현재가", f"${float(sig.get('wti_price', 0)):.2f}",
              f"{float(sig.get('momentum_5d', 0))*100:+.1f}% (5일)")
    c2.metric("OVX",
              f"{_ovx_val:.0f}",
              "🔴 높음" if _ovx_al == 'HIGH' else ("🟡 상승" if _ovx_al == 'ELEVATED' else "🟢 정상"),
              help="원유 공포지수(CBOE OVX). 높을수록 시장 불안 → 예측 신뢰구간 자동 확대.\n🟢 정상 < 35 / 🟡 상승 35~45 / 🔴 높음 ≥ 45")
    c3.metric("헤지 비율 권고", f"{_hedge*100:.0f}%")
    c4.metric("하방 / 상방 리스크", f"{_down:.0f}% / {_up:.0f}%")
    c5.metric("급등확률 3일", f"{_surge_p*100:.0f}%",
              "🔴 높음" if _surge_p > 0.6 else ("🟡 보통" if _surge_p > 0.3 else "🟢 낮음"))
    _arrow = "↑" if _dir_bias > 0.02 else ("↓" if _dir_bias < -0.02 else "→")
    c6.metric("방향성", f"{_arrow} {_dir_conf}", f"bias {_dir_bias:+.3f}")

    if _geo:
        st.warning("🌍 **지정학 알람 활성** — 지정학 위기 신호 감지됨")
    _ci_mult = sig.get('ci_multiplier', 1.0)
    if _ci_mult and float(_ci_mult) > 1.0:
        st.warning(f"⚡ **Shock 감지** — 예측 불확실성 구간 ×{float(_ci_mult):.1f} 확대")


def _render_price_charts(data):
    col_hist, col_fc = st.columns([1.4, 1])

    with col_hist:
        _hist_label = "전체 기간" if _is_admin else "최근 30일"
        st.markdown(f"**📈 WTI 가격 + 리스크 레벨 ({_hist_label})**")
        if 'risk_history' in data and not data['risk_history'].empty:
            rh = data['risk_history'].copy()
            rh['date'] = pd.to_datetime(rh['date'])
            _cutoff = pd.Timestamp.now() - pd.DateOffset(days=90)
            rh = rh[rh['date'] >= _cutoff].sort_values('date')
            if not _is_admin:
                rh = rh.tail(30)
            _LEVEL_CLR = {
                'NORMAL': '#2ecc71', 'CAUTION': '#f39c12',
                'SURGE_RISK': '#e74c3c', 'DROP_RISK': '#3498db', 'CRITICAL': '#8e44ad',
            }
            fig_rh = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rh.add_trace(go.Scatter(
                x=rh['date'], y=rh['wti_price'],
                mode='lines', name='WTI',
                line=dict(color='#58a6ff', width=2),
                hovertemplate='%{x|%m/%d}<br>$%{y:.2f}<extra></extra>',
            ), secondary_y=False)
            for _lv, _clr in _LEVEL_CLR.items():
                _mask = rh['risk_level'] == _lv
                if _mask.any():
                    fig_rh.add_trace(go.Scatter(
                        x=rh[_mask]['date'], y=rh[_mask]['wti_price'],
                        mode='markers', name=_lv,
                        marker=dict(color=_clr, size=10, line=dict(color='white', width=1)),
                        hovertemplate=f'%{{x|%m/%d}}<br>{_lv}<br>$%{{y:.2f}}<extra></extra>',
                    ), secondary_y=False)
            if 'risk_score' in rh.columns and rh['risk_score'].notna().any():
                fig_rh.add_trace(go.Bar(
                    x=rh['date'], y=rh['risk_score'],
                    name='리스크 스코어', marker_color='rgba(231,76,60,0.25)',
                    hovertemplate='%{x|%m/%d}<br>Risk: %{y:.3f}<extra></extra>',
                ), secondary_y=True)
                fig_rh.add_shape(type='line', yref='y2', xref='paper',
                                 x0=0, x1=1, y0=0.3, y1=0.3,
                                 line=dict(color='#f39c12', dash='dash', width=1), opacity=0.7)
                fig_rh.add_shape(type='line', yref='y2', xref='paper',
                                 x0=0, x1=1, y0=0.6, y1=0.6,
                                 line=dict(color='#e74c3c', dash='dash', width=1), opacity=0.7)
            fig_rh.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'), height=320,
                legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                            font=dict(size=9), orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=50, r=20, t=30, b=40),
                xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9), title='날짜'),
                hovermode='x unified',
            )
            fig_rh.update_yaxes(title_text='WTI ($)', secondary_y=False,
                                 gridcolor='#21262d', tickfont=dict(size=9))
            fig_rh.update_yaxes(title_text='Risk Score', secondary_y=True,
                                 gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9), showgrid=False)
            st.plotly_chart(fig_rh, use_container_width=True)
        else:
            st.info("파이프라인 실행 후 표시됩니다.")

    with col_fc:
        st.markdown("**📅 D+1~7 가격 예측**")
        if 'forecast_7days' in data and not data['forecast_7days'].empty:
            fc = data['forecast_7days'].copy()
            fc['date'] = pd.to_datetime(fc['date'])
            _DAY_KO = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'}

            fig_fc = go.Figure()
            if 'lower_75ci' in fc.columns and 'upper_75ci' in fc.columns:
                _ci_x = pd.concat([fc['date'], fc['date'][::-1]])
                _ci_y = pd.concat([fc['upper_75ci'], fc['lower_75ci'][::-1]])
                fig_fc.add_trace(go.Scatter(
                    x=_ci_x, y=_ci_y, fill='toself',
                    fillcolor='rgba(240,192,64,0.12)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='75% 구간', hoverinfo='skip',
                ))
            fig_fc.add_trace(go.Scatter(
                x=fc['date'], y=fc['forecast_price'],
                mode='lines+markers', name='예측',
                line=dict(color='#f0c040', width=2, dash='dot'),
                marker=dict(size=8, symbol='circle-open'),
                hovertemplate='%{x|%m/%d}<br>$%{y:.2f}<extra></extra>',
            ))
            if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
                _cur_px = float(data['latest_risk_signal'].iloc[0]['wti_price'])
                fig_fc.add_hline(
                    y=_cur_px,
                    line=dict(color='#58a6ff', dash='dash', width=1),
                    annotation_text=f'현재 ${_cur_px:.1f}',
                    annotation_font=dict(color='#58a6ff', size=9),
                )
            fig_fc.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'), height=200,
                margin=dict(l=50, r=20, t=20, b=30),
                xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9)),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=9), title='WTI ($)'),
                showlegend=False,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # 예측 테이블 (compact)
            fc['날짜'] = fc['date'].apply(
                lambda d: f"{d.strftime('%m/%d')}({_DAY_KO[d.weekday()]})"
            )
            if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
                _last_px = float(data['latest_risk_signal'].iloc[0]['wti_price'])
                # 변화 = D+1은 현재가 대비, D+2~D+7은 전일 예측가 대비 (일별 방향 표시)
                _prev_prices = [_last_px] + list(fc['forecast_price'].iloc[:-1])
                fc['변화'] = [
                    (f"{'↑' if p > b else ('↓' if p < b else '→')} {(p-b)/b*100:+.1f}%") if b != 0 else "→ —"
                    for p, b in zip(fc['forecast_price'], _prev_prices)
                ]
            _show_cols = ['날짜', 'forecast_price', '변화']
            if 'model_std' in fc.columns:
                fc['합의도'] = fc['model_std'].apply(
                    lambda s: '🟢 높음' if s < 2 else ('🟡 보통' if s < 5 else '🔴 낮음')
                )
                _show_cols.append('합의도')
            if 'lower_75ci' in fc.columns:
                fc['75% 구간'] = fc.apply(
                    lambda r: f"${r['lower_75ci']:.0f}~${r['upper_75ci']:.0f}", axis=1
                )
                _show_cols.append('75% 구간')
            if 'var_5pct' in fc.columns:
                fc['VaR하한($)'] = fc['var_5pct']
                _show_cols.append('VaR하한($)')
            if _is_admin:
                for _ac in ['sarimax_forecast', 'xgb_forecast', 'var_forecast', 'bias_correction']:
                    if _ac in fc.columns and _ac not in _show_cols:
                        _show_cols.append(_ac)
            show_fc = fc[[c for c in _show_cols if c in fc.columns]].rename(
                columns={
                    'forecast_price': '예측가($)',
                    'sarimax_forecast': 'SARIMAX($)',
                    'xgb_forecast': 'XGB($)',
                    'var_forecast': 'VAR($)',
                    'bias_correction': 'Bias($)',
                }
            )
            st.dataframe(show_fc, hide_index=True, use_container_width=True)
            if 'lower_75ci' in fc.columns and len(fc) > 0:
                _ci_d1 = float(fc['upper_75ci'].iloc[0]) - float(fc['lower_75ci'].iloc[0])
                _ci_d7 = float(fc['upper_75ci'].iloc[-1]) - float(fc['lower_75ci'].iloc[-1])
                st.caption(f"75% 신뢰구간: D+1 ±${_ci_d1/2:.1f}  →  D+7 ±${_ci_d7/2:.1f}")
            if 'var_5pct' in fc.columns and len(fc) > 0:
                _var_loss_d1 = float(fc['forecast_price'].iloc[0]) - float(fc['var_5pct'].iloc[0])
                _var_loss_d7 = float(fc['forecast_price'].iloc[-1]) - float(fc['var_5pct'].iloc[-1])
                st.caption(f"📉 VaR(95%): D+1 최대 ${_var_loss_d1:.1f} 하락 위험  →  D+7 최대 ${_var_loss_d7:.1f}")
            if 'model_std' in fc.columns and not fc.empty and float(fc['model_std'].iloc[0]) >= 5:
                st.warning(f"⚠️ 모델 간 예측 편차 ${fc['model_std'].iloc[0]:.1f} — 불확실성 높음")
            if 'reliable_forecast' in fc.columns and not fc['reliable_forecast'].astype(str).eq('True').all():
                st.warning("⚠️ 일부 예측일의 신뢰도가 낮습니다. 예측 결과 해석 시 주의하세요.")

            # ② D+1 방향 합의 신호
            try:
                if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
                    _wti   = float(data['latest_risk_signal'].iloc[0]['wti_price'])
                    _d1    = fc.iloc[0]
                    _dirs  = []
                    if 'sarimax_forecast' in fc.columns:
                        _dirs.append('UP' if float(_d1['sarimax_forecast']) > _wti else 'DOWN')
                    if 'xgb_forecast' in fc.columns:
                        _dirs.append('UP' if float(_d1['xgb_forecast'])    > _wti else 'DOWN')
                    _dirs.append('UP' if float(_d1['forecast_price']) > _wti else 'DOWN')
                    if 'prediction_log' in data and not data['prediction_log'].empty:
                        _pl_live = data['prediction_log'][data['prediction_log']['type'] == 'live']
                        _pl_dir  = _pl_live[_pl_live['pred_direction'].isin(['UP','DOWN'])]
                        if not _pl_dir.empty:
                            _dirs.append(str(_pl_dir.iloc[-1]['pred_direction']))
                    _n   = len(_dirs)
                    _up  = _dirs.count('UP')
                    _dn  = _dirs.count('DOWN')
                    _lbl = '↑ 상승' if _up > _dn else '↓ 하락'
                    _maj = max(_up, _dn)
                    if _maj == _n:
                        st.success(f"✅ D+1 방향 합의: {_lbl} ({_maj}/{_n} 모델 일치)")
                    elif _maj >= _n - 1:
                        st.info(f"🔶 D+1 방향: {_lbl} 우세 ({_maj}/{_n}) — 일부 불일치")
                    else:
                        st.warning(f"⚠️ D+1 방향 불일치 (↑{_up} vs ↓{_dn}) — 신호 불확실")
            except Exception:
                pass

            # ④ 단일 예측 이상치 경고
            try:
                if 'prediction_log' in data and not data['prediction_log'].empty:
                    _pl_conf = data['prediction_log'][
                        (data['prediction_log']['type'] == 'live') &
                        data['prediction_log']['price_error'].notna()
                    ].tail(20)
                    if len(_pl_conf) >= 5:
                        _errs   = _pl_conf['price_error'].abs()
                        _mu     = _errs.mean()
                        _thresh = _mu + 2 * _errs.std()
                        _last   = _pl_conf.iloc[-1]
                        _last_e = abs(float(_last['price_error']))
                        if _last_e > _thresh:
                            st.warning(
                                f"⚠️ {_last['date']} 예측 오차 ${_last_e:.2f} — "
                                f"최근 평균 ${_mu:.2f} + 2σ(기준 ${_thresh:.2f}) 초과"
                            )
            except Exception:
                pass

            if _is_admin:
                _fc_dl = fc.to_csv(index=False).encode('utf-8')
                st.download_button("💾 예측 CSV", _fc_dl, "forecast_7days.csv", "text/csv",
                                   key="fc_dl_admin")
        else:
            st.info("파이프라인 실행 후 표시됩니다.")


def _render_snapshot_analysis(data):
    """예측 vs 실제 차트 + 방향성 정확도 + 누적 MASE 추이"""
    if 'forecast_snapshots' not in data or 'prediction_log' not in data:
        return
    try:
        snap = data['forecast_snapshots'].copy()
        pl   = data['prediction_log'][['date', 'actual_price']].rename(columns={'date': 'forecast_date'})
        snap = snap.merge(pl, on='forecast_date', how='left')
        known = snap[snap['actual_price'].notna()].copy().sort_values('forecast_date').reset_index(drop=True)
    except Exception:
        return
    if len(known) < 5:
        return

    known['abs_err'] = (known['actual_price'] - known['forecast_price']).abs()
    _mae   = known['abs_err'].mean()
    _naive = known['actual_price'].diff().abs().dropna().mean()
    _mase  = _mae / max(_naive, 1e-6)

    known['actual_dir']   = known['actual_price'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    known['forecast_dir'] = (known['forecast_price'] - known['actual_price'].shift(1)).apply(
                                lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    _dmask   = known['actual_dir'] != 0
    _dir_acc = (known.loc[_dmask, 'actual_dir'] == known.loc[_dmask, 'forecast_dir']).mean() if _dmask.sum() > 0 else 0.0

    st.markdown("**📊 예측 정확도 — 누적 분석**")
    _sa1, _sa2, _sa3 = st.columns(3)
    _sa1.metric("방향성 정확도", f"{_dir_acc*100:.0f}%", "상승/하락 방향")
    _sa2.metric("MASE (전체)", f"{_mase:.3f}",
                "HIGH" if _mase < 1.0 else ("MEDIUM" if _mase < 1.5 else "LOW"))
    _sa3.metric("평균 오차 (MAE)", f"${_mae:.2f}", f"N={len(known)}")

    # 예측 vs 실제 차트
    fig_sa = go.Figure()
    fig_sa.add_trace(go.Scatter(
        x=known['forecast_date'], y=known['actual_price'],
        mode='lines', name='실제가', line=dict(color='#58a6ff', width=2),
        hovertemplate='%{x}<br>실제: $%{y:.2f}<extra></extra>',
    ))
    fig_sa.add_trace(go.Scatter(
        x=known['forecast_date'], y=known['forecast_price'],
        mode='lines+markers', name='예측가',
        line=dict(color='#f0c040', width=1.5, dash='dot'),
        marker=dict(size=4),
        hovertemplate='%{x}<br>예측: $%{y:.2f}<extra></extra>',
    ))
    fig_sa.update_layout(
        paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
        font=dict(color='#c9d1d9'), height=250,
        legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', font=dict(size=9),
                    orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=9), title='WTI ($)'),
        hovermode='x unified',
    )
    st.plotly_chart(fig_sa, use_container_width=True)

    # 누적 MASE 추이 (30일 롤링)
    if len(known) >= 30:
        _rmase_vals, _rmase_dates = [], []
        for _i in range(29, len(known)):
            _w = known.iloc[_i-29:_i+1]
            _wm = _w['abs_err'].mean()
            _wn = _w['actual_price'].diff().abs().dropna().mean()
            _rmase_vals.append(_wm / max(_wn, 1e-6))
            _rmase_dates.append(known.iloc[_i]['forecast_date'])
        fig_mase = go.Figure()
        fig_mase.add_trace(go.Scatter(
            x=_rmase_dates, y=_rmase_vals,
            mode='lines', name='MASE (30일 롤링)',
            line=dict(color='#79c0ff', width=2),
            hovertemplate='%{x}<br>MASE: %{y:.3f}<extra></extra>',
        ))
        fig_mase.add_shape(type='line', xref='paper', x0=0, x1=1, y0=1.0, y1=1.0,
                           line=dict(color='#e74c3c', dash='dash', width=1), opacity=0.7)
        fig_mase.add_shape(type='line', xref='paper', x0=0, x1=1, y0=1.5, y1=1.5,
                           line=dict(color='#f39c12', dash='dash', width=1), opacity=0.5)
        fig_mase.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
            font=dict(color='#c9d1d9'), height=200,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9)),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(size=9), title='MASE'),
            showlegend=False,
        )
        st.plotly_chart(fig_mase, use_container_width=True)
        st.caption("빨간 점선: MASE=1.0 (naive 기준선) | 주황 점선: MASE=1.5 (LOW 임계)")


def _render_alerts_news(data):
    col_al, col_news = st.columns([1, 1.5])

    with col_al:
        st.markdown("**📡 리스크 알람**")
        _alert_path = OUTPUT_DIR / 'latest_alerts.json'
        if _alert_path.exists():
            try:
                _al   = json.loads(_alert_path.read_text(encoding='utf-8'))
                _trigs = _al.get('triggers', [])
                _lvl_ko = {'WARNING': '⚠️ 경고', 'CRITICAL': '🔴 위험', 'NORMAL': '🟢 정상'}
                _cat_ko = {
                    'price_move': '가격변동', 'geopolitical': '지정학',
                    'supply': '공급', 'supply_cut': '공급감산',
                    'demand': '수요', 'demand_shock': '수요충격',
                    'inventory': '재고', 'sanctions': '제재',
                    'opec': 'OPEC', 'ovx_spike': 'OVX 급등',
                }
                _lvl_txt = _lvl_ko.get(_al.get('alert_level', ''), _al.get('alert_level', ''))
                st.caption(f"{_lvl_txt} · {_al.get('checked_at', '—')}")
                if _trigs:
                    for _tr in _trigs[:3]:
                        _cat = _cat_ko.get(_tr.get('category', ''), _tr.get('category', ''))
                        st.markdown(f"- **[{_cat}]** {_tr['title'][:70]}")
                else:
                    st.success("현재 주요 알람 없음")
            except Exception:
                st.info("알람 데이터 없음")
        else:
            st.info("알람 데이터 없음")

        if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
            st.markdown("**⚡ 리스크 팩터 강도**")
            sig2 = data['latest_risk_signal'].iloc[0]
            factors = {
                '변동성':    min(float(sig2['volatility_5d']) * 22, 1.0),
                '방향성':    min(abs(float(sig2['momentum_5d'])) * 8, 1.0),
                '부정감성':  min(max(-float(sig2['news_sentiment']), 0), 1.0),
                '지정학':    1.0 if str(sig2['geopolitical_alert']).lower() in ('true','1') else 0.0,
                '종합위험':  min(float(sig2['risk_score']) / 3.0, 1.0),
            }
            fig_rf = go.Figure(go.Bar(
                x=list(factors.values()), y=list(factors.keys()),
                orientation='h',
                marker_color=['#ff6b6b','#ffd93d','#ff4757','#ff6348','#e74c3c'],
                opacity=0.85,
                text=[f'{v:.2f}' for v in factors.values()],
                textposition='outside', textfont=dict(color='#c9d1d9', size=10),
            ))
            fig_rf.add_vline(x=0.5, line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1))
            fig_rf.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                font=dict(color='#c9d1d9'), height=190, showlegend=False,
                margin=dict(l=10, r=55, t=10, b=20),
                xaxis=dict(range=[0, 1.35], gridcolor='#21262d', tickfont=dict(size=9)),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_rf, use_container_width=True)

    with col_news:
        st.markdown("**📰 주요 뉴스 키워드**")
        if 'crisis_keywords' in data and not data['crisis_keywords'].empty:
            kw = data['crisis_keywords'].head(10).copy()
            kw['키워드'] = kw['keyword'].apply(lambda w: _KW_KO.get(w.lower(), w))
            kw['분류']   = kw['is_crisis_word'].astype(str).map({'True': '🔴', 'False': '🔵'})
            st.dataframe(
                kw[['분류', '키워드', 'count']].rename(columns={'count': '빈도'}),
                hide_index=True, use_container_width=True, height=200,
            )
        wc_img = OUTPUT_DIR / 'wordcloud.png'
        if wc_img.exists():
            st.image(str(wc_img), use_container_width=True,
                     caption='🔴 위기 키워드  |  🔵 일반 키워드')


def _render_risk_drivers(sig):
    _drivers = []
    _gpr_z = float(sig.get('gpr_zscore', 0) or 0)
    _ovx   = float(sig.get('ovx_level', 0) or 0)
    _geo   = str(sig.get('geopolitical_alert', '')).lower() in ('true', '1', 'yes')
    _sent  = float(sig.get('news_sentiment', 0) or 0)
    _mom   = float(sig.get('momentum_5d', 0) or 0)
    if _geo:             _drivers.append("지정학 위기")
    if _gpr_z > 1.5:     _drivers.append(f"GPR 급등(z={_gpr_z:.1f})")
    if _ovx >= 60:       _drivers.append(f"OVX 급등({_ovx:.0f})")
    elif _ovx >= 45:     _drivers.append(f"OVX 상승({_ovx:.0f})")
    if _sent < -0.3:     _drivers.append("부정 뉴스 감성")
    if _mom < -0.05:     _drivers.append("하락 모멘텀")
    elif _mom > 0.05:    _drivers.append("상승 모멘텀")
    if _drivers:
        st.caption(f"**주요 리스크 드라이버:** {' · '.join(_drivers)}")


# ─────────────────────────────────────────────────────────────────────────────
# 사용자 페이지 — 단일 스크롤
# ─────────────────────────────────────────────────────────────────────────────

def render_user_page():
    st.markdown("# 🛢 국제 유가 리스크 예측 시스템")

    if 'latest_risk_signal' not in data or data['latest_risk_signal'].empty:
        st.warning("데이터 없음. 파이프라인 실행이 필요합니다.")
        return

    sig = data['latest_risk_signal'].iloc[0]

    # 1. 리스크 히어로 카드
    _render_risk_hero(sig)

    # 예측 신뢰도 경고 + 리스크 드라이버
    _fc_rel_u = str(sig.get('forecast_reliability', '') or '').upper()
    _m_age_u  = int(sig.get('model_age_days', 0) or 0)
    if _fc_rel_u == 'LOW':
        st.error("⚠️ **예측 신뢰도 낮음** — 라이브 오차가 백테스트 대비 1.5× 초과. 예측값 참고 수준으로만 활용하세요.")
    elif _m_age_u > 30:
        st.error(f"🚨 **파이프라인 미실행 {_m_age_u}일** — 데이터·모델 미갱신 상태. 즉시 재실행 필요.")
    elif _m_age_u > 7:
        st.warning(f"⚠️ 파이프라인 미실행 **{_m_age_u}일** 경과 — 재실행 권장.")
    _render_risk_drivers(sig)

    # 2. 핵심 지표 행
    _render_key_metrics(sig)

    st.markdown("---")

    # 3. 차트 (가격 히스토리 + D+1~7 예측)
    _render_price_charts(data)

    st.markdown("---")

    # 4. 알람 + 뉴스 키워드
    _render_alerts_news(data)

    st.markdown("---")

    # 5. 예측 정확도 누적 분석
    _render_snapshot_analysis(data)

    # 하단 업데이트 시각
    _meta_path = OUTPUT_DIR / 'run_meta.json'
    if _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text(encoding='utf-8'))
            st.caption(
                f"마지막 업데이트: {_meta.get('last_run', '—')}  |  "
                f"데이터 기준: {_meta.get('data_through', '—')}"
            )
        except Exception:
            pass

    st.markdown("---")
    st.caption("국제 유가 리스크 예측 시스템  |  XGBoost-HAR + SARIMAX + VAR Ensemble  |  News Sentiment + Geopolitical Risk")


# ─────────────────────────────────────────────────────────────────────────────
# 관리자 페이지 — 5탭
# ─────────────────────────────────────────────────────────────────────────────

def render_admin_page():
    st.markdown("# 🛢 국제 유가 리스크 예측 시스템 — 관리자")

    tab_risk, tab_monitor, tab_error, tab_pipeline, tab_users = st.tabs([
        "🌡 리스크 현황", "📊 모델 모니터링", "📋 예측 오차", "🚀 파이프라인", "👤 사용자 관리"
    ])

    # ── Tab1: 리스크 현황 (사용자 뷰와 동일)
    with tab_risk:
        if 'latest_risk_signal' not in data or data['latest_risk_signal'].empty:
            st.warning("데이터 없음.")
        else:
            sig = data['latest_risk_signal'].iloc[0]
            _render_risk_hero(sig)
            _render_risk_drivers(sig)
            _render_key_metrics(sig)
            st.markdown("---")
            _render_price_charts(data)
            st.markdown("---")
            _render_alerts_news(data)
            # 관리자 추가: 리스크 신호 상세 JSON
            st.markdown("---")
            st.markdown("**📋 리스크 신호 상세 (관리자)**")
            _sig_csv = OUTPUT_DIR / 'latest_risk_signal.csv'
            if _sig_csv.exists():
                st.download_button("💾 latest_risk_signal.csv", _sig_csv.read_bytes(),
                                   "latest_risk_signal.csv", "text/csv")
            st.json({k: str(v) for k, v in sig.items()})

    # ── Tab2: 모델 모니터링
    with tab_monitor:
        st.subheader("📊 모델 상태")

        if 'latest_risk_signal' in data and not data['latest_risk_signal'].empty:
            sig = data['latest_risk_signal'].iloc[0]
            _fc_rel = str(sig.get('forecast_reliability', '') or '').upper()
            _m_age  = int(sig.get('model_age_days', 0) or 0)
            _ovx    = float(sig.get('ovx_level', 0) or 0)
            m1, m2, m3, m4 = st.columns(4)
            _rel_clr = 'inverse' if _fc_rel == 'LOW' else ('off' if _fc_rel == 'MEDIUM' else 'normal')
            m1.metric("forecast_reliability", _fc_rel,
                      help="D+1 예측 vs 실제 가격 정확도(MASE 기준).\nHIGH = 나이브 예측보다 잘 맞음 / LOW = 나이브보다 못 맞음")
            _age_label = "🚨 즉시 재실행" if _m_age > 30 else ("⚠️ 재실행 권장" if _m_age > 7 else ("정상" if _m_age <= 1 else "주의"))
            _age_clr   = "inverse" if _m_age > 7 else "off"
            m2.metric("모델 나이", f"{_m_age}일",
                      delta=_age_label,
                      delta_color=_age_clr,
                      help="파이프라인 마지막 실행 경과일 — 0~1일: 정상 / 2~7일: 주의 / 8~30일: 재실행 권장 / 30일+: 즉시 재실행")
            m3.metric("OVX 레짐", f"{_ovx:.0f}",
                      "높음" if _ovx >= 60 else ("보통" if _ovx >= 40 else "낮음"),
                      help="원유 공포지수(CBOE OVX). 높을수록 시장 불안 → 예측 신뢰구간 자동 확대.\n🟢 정상 < 35 / 🟡 상승 35~45 / 🔴 높음 ≥ 45")
            m4.metric("리스크 스코어", f"{float(sig.get('risk_score', 0)):.3f}",
                      help="변동성·뉴스·지정학·감성·OVX 5개 요소를 곱해 산출한 종합 위험도.\n2.2 이상이면 급등/급락 위험 단계 진입.")

        # 앙상블 정보 (최근 로그에서 파싱)
        _log_path = OUTPUT_DIR / 'pipeline_run.log'
        if _log_path.exists():
            st.markdown("---")
            st.markdown("**앙상블 / Bias 교정 (최근 실행)**")
            with open(_log_path, encoding='utf-8', errors='replace') as _lf:
                _log_lines = _lf.readlines()
            _keywords = ['3모델 앙상블', '롤링 bias', '잔차 교정', 'σ-clip', '열화 트리거',
                         'Optuna 캐시', 'SVM 모델 캐시', 'forecast_reliability']
            for line in _log_lines:
                if any(k in line for k in _keywords):
                    st.caption(f"🔍 {line.strip()}")

        st.markdown("---")
        st.markdown("**모델 성능 테이블**")
        if 'model_performance' in data and not data['model_performance'].empty:
            pf = data['model_performance']
            _price = pf[pf['target'] == 'price'].copy() if 'target' in pf.columns else pf
            _vol   = pf[pf['target'] == 'vol_5d'].copy() if 'target' in pf.columns else pd.DataFrame()

            _price_disp = _price.copy()
            _stk_row_m = _price_disp[_price_disp['model'].str.contains('Stacking', na=False)]
            _stk_adopted_m = not (_stk_row_m.empty or _stk_row_m['model'].str.contains('미채택', na=False).any())
            # HAR-Enhanced 채택 여부: dir_acc 일치로 판단 (MAE는 우연 일치 가능 — 취약)
            _har_enh_adopted = False
            try:
                _xr_row = _price[_price['model'].str.contains('XGBoost-Return', na=False)]
                _he_row = _price[_price['model'].str.contains('HAR-XGB-Enhanced', na=False)]
                if not _xr_row.empty and not _he_row.empty:
                    if 'dir_acc' in _price.columns:
                        _har_enh_adopted = (
                            abs(float(_xr_row['dir_acc'].values[0]) -
                                float(_he_row['dir_acc'].values[0])) < 0.001
                        )
                    elif 'mae' in _price.columns:
                        _har_enh_adopted = (
                            abs(float(_xr_row['mae'].values[0]) -
                                float(_he_row['mae'].values[0])) < 0.01
                        )
            except Exception:
                pass
            def _assign_role_tab(m):
                if 'Stacking' in m and '미채택' in m: return '앙상블 (미채택)'
                if 'Stacking' in m:      return '앙상블 채택 ✅'
                if 'HAR-XGB-Enhanced' in m:
                    if _har_enh_adopted:
                        return '가격예측' if _stk_adopted_m else '가격예측 (라이브 ✅)'
                    return '모니터링/비교'
                if 'HAR' in m or 'GARCH' in m: return '변동성 진단'
                if 'Prophet' in m:       return '모니터링 전용'
                if 'XGBoost-Return' in m: return '가격예측' if _stk_adopted_m else ('모니터링/비교' if _har_enh_adopted else '가격예측 (라이브 ✅)')
                if 'SARIMAX' in m:       return '가격예측' if _stk_adopted_m else '앙상블 컴포넌트'
                return '모니터링/비교'
            _price_disp.insert(0, '역할', _price_disp['model'].apply(_assign_role_tab))
            _p_cols = ['역할'] + [c for c in ['model','rmse','mae','r2','mase','dir_acc','wf_dir_acc'] if c in _price.columns]
            st.caption("가격 예측 모델 (RMSE·MAE: ↓ | R²·dir_acc: ↑)")
            st.dataframe(_price_disp[_p_cols], hide_index=True, use_container_width=True)
            _mon_only = _price[_price['mae'] >= 20] if 'mae' in _price.columns else pd.DataFrame()
            if not _mon_only.empty:
                st.caption(f"※ {', '.join(_mon_only['model'].tolist())} — 모니터링 전용 (차트 제외)")

            if not _vol.empty:
                _v_cols = [c for c in ['model','rmse','mae','r2','train_r2','overfit_gap'] if c in _vol.columns]
                st.caption("변동성 예측 모델")
                st.dataframe(_vol[_v_cols], hide_index=True, use_container_width=True)

            # 롤백 기준 경고
            _warns = []
            _har = pf[pf['model'].str.contains('HAR', na=False)] if 'model' in pf.columns else pd.DataFrame()
            _stk = pf[pf['model'].str.contains('Stacking', na=False)] if 'model' in pf.columns else pd.DataFrame()
            _xgb = pf[pf['model'].str.contains('XGBoost-Return', na=False)] if 'model' in pf.columns else pd.DataFrame()
            if not _har.empty and 'r2' in _har.columns and float(_har['r2'].iloc[0]) < 0.48:
                _warns.append(f"HAR R²={float(_har['r2'].iloc[0]):.3f} < 0.48")
            if not _stk.empty and 'r2' in _stk.columns and float(_stk['r2'].iloc[0]) < 0.83:
                _warns.append(f"Stacking R²={float(_stk['r2'].iloc[0]):.3f} < 0.83")
            if not _xgb.empty and 'dir_acc' in _xgb.columns and float(_xgb['dir_acc'].iloc[0]) < 0.52:
                _warns.append(f"dir_acc={float(_xgb['dir_acc'].iloc[0])*100:.1f}% < 52%")
            for w in _warns:
                st.warning(f"⚠️ {w} (롤백 기준)")
            if not _warns:
                st.success("✅ 모든 모델 롤백 기준 충족")

            # overfit gap 경고
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

            # 성능 바 차트 (가격 예측 모델 RMSE/MAE/R²)
            _pf_chart = _price[_price['mae'] < 20].copy() if 'mae' in _price.columns else _price
            if not _pf_chart.empty and all(c in _pf_chart.columns for c in ['rmse', 'mae', 'r2']):
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
                fig_perf.update_xaxes(gridcolor='#21262d', tickfont=dict(size=8), tickangle=25)
                fig_perf.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
                fig_perf.update_annotations(font=dict(color='#e6edf3', size=10))
                st.plotly_chart(fig_perf, use_container_width=True)

            # 모델 설명
            st.markdown("---")
            _stk_role_desc = "✅ 앙상블 가격 예측 (채택)" if _stk_adopted_m else "앙상블 (미채택)"
            _dir_model_name = "HAR-XGB-Enhanced" if _har_enh_adopted else "XGBoost-Classifier"
            # 방향성 정확도: test-set과 WF 모두 표시 (test-set 단독 과대평가 방지)
            _dir_acc_note = ""
            if _har_enh_adopted:
                try:
                    _he_r = _price[_price['model'].str.contains('HAR-XGB-Enhanced', na=False)]
                    if not _he_r.empty:
                        _ts_a = float(_he_r['dir_acc'].values[0]) * 100 if 'dir_acc' in _he_r.columns else None
                        _wf_a = (float(_he_r['wf_dir_acc'].values[0]) * 100
                                 if 'wf_dir_acc' in _he_r.columns
                                 and pd.notna(_he_r['wf_dir_acc'].values[0]) else None)
                        if _ts_a and _wf_a:
                            _dir_acc_note = f" (**test={_ts_a:.1f}%** / WF={_wf_a:.1f}%)"
                        elif _ts_a:
                            _dir_acc_note = f" (test={_ts_a:.1f}%)"
                except Exception:
                    pass
            _dir_model_desc = (
                f"HAR+MI 하이브리드 피처, dead-zone, inv-vol, SVM 블렌드{_dir_acc_note}"
                if _har_enh_adopted else "SVM+CEEMDAN 앙상블 → 상승/하락 확률"
            )
            st.markdown(f"""
**모델 설명**
| 모델 | 역할 | 특징 |
|------|------|------|
| **Stacking (Ridge)** | {_stk_role_desc} | SARIMAX + XGB + VAR → Ridge 메타러너 |
| **XGBoost-HAR** | 변동성(리스크) 예측 | HAR 구성요소 + 뉴스/지정학 외생변수 |
| **SARIMAX** | 가격 예측 컴포넌트 | AR(2,1,2) × 주간 계절성 + 외생변수 |
| **{_dir_model_name}** | 최종 방향 판단 | {_dir_model_desc} |
""")
        else:
            st.info("파이프라인 실행 후 표시됩니다.")

        # 피처 중요도
        if 'feature_importance' in data and not data['feature_importance'].empty:
            st.markdown("---")
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
                height=420, showlegend=False,
                margin=dict(l=10, r=70, t=45, b=20),
                xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9)),
                yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")
        _render_snapshot_analysis(data)

    # ── Tab3: 예측 오차
    with tab_error:
        st.subheader("📋 예측 오차 분석")

        # 오차 스파이크 테이블
        if 'live_gap_spikes' in data and not data['live_gap_spikes'].empty:
            st.markdown("**⚡ 오차 스파이크 (최근)**")
            _gs = data['live_gap_spikes'].copy()
            _gs.columns = [{
                'date': '날짜', 'actual_price': '실제가($)', 'sarimax_pred': 'SARIMAX예측($)',
                'price_error': '오차($)', 'abs_error': '절대오차($)',
                'xgb_pred_vol': 'XGB예측변동성', 'actual_vol_5d': '실제변동성(5d)',
            }.get(c, c) for c in _gs.columns]
            st.dataframe(_gs, hide_index=True, use_container_width=True)

        if 'prediction_log' not in data:
            st.info("파이프라인 실행 후 데이터가 표시됩니다.")
        else:
            pl = data['prediction_log'].copy()
            bt = pl[pl['type'] == 'backtest'].copy()
            lv = pl[pl['type'].isin(['live', 'gap'])].copy()
            live_confirmed = lv[lv['actual_price'].notna()]

            c1, c2, c3, c4 = st.columns(4)
            _bt_err_col = ('stacking_error'
                           if 'stacking_error' in bt.columns and bt['stacking_error'].notna().any()
                           else 'price_error')
            if not bt.empty and bt[_bt_err_col].notna().any():
                c1.metric("백테스트 MAE (Stacking)" if _bt_err_col == 'stacking_error' else "백테스트 MAE",
                          f"${bt[_bt_err_col].abs().mean():.2f}")
                if 'price_error_pct' in bt.columns and bt['price_error_pct'].notna().any():
                    c2.metric("백테스트 MAPE", f"{bt['price_error_pct'].abs().mean():.2f}%")
            if not live_confirmed.empty:
                _bt_mae = bt[_bt_err_col].abs().mean() if not bt.empty and bt[_bt_err_col].notna().any() else None
                _lv_mae = live_confirmed['price_error'].abs().mean()
                _delta  = f"+${_lv_mae - _bt_mae:.2f}" if _bt_mae else None
                c3.metric("라이브 MAE", f"${_lv_mae:.2f}", delta=_delta, delta_color="inverse")
                if 'price_error_pct' in live_confirmed.columns and live_confirmed['price_error_pct'].notna().any():
                    c4.metric("라이브 MAPE", f"{live_confirmed['price_error_pct'].abs().mean():.2f}%")

            # 드리프트 경고
            if 'live_gap_monthly' in data and not data['live_gap_monthly'].empty:
                _gm = data['live_gap_monthly']
                _bt_r = _gm[_gm['type'] == 'backtest']
                _lv_r = _gm[_gm['type'] == 'live']
                _bt_g = _bt_r['mae'].mean() if not _bt_r.empty else None
                _lv_g = _lv_r['mae'].mean() if not _lv_r.empty else None
                _lv_n = int(_lv_r['n'].sum()) if not _lv_r.empty and 'n' in _lv_r.columns else 0
                if _bt_g and _lv_g and _lv_n >= 5 and _lv_g / _bt_g > 2.0:
                    st.warning(f"⚠️ 모델 드리프트 — 라이브/백테스트 MAE 비율 {_lv_g/_bt_g:.2f}×")

            # Bias 보정 경고
            if 'forecast_7days' in data and not data['forecast_7days'].empty:
                _bc_val = data['forecast_7days']['bias_correction'].iloc[0] if 'bias_correction' in data['forecast_7days'].columns else 0
                try:
                    _bc = float(_bc_val)
                    if abs(_bc) > 3.0:
                        _dir = "과소예측" if _bc > 0 else "과대예측"
                        st.info(f"ℹ️ Rolling Bias 교정 {_bc:+.2f}$ 적용 중 ({_dir} 패턴)")
                except (TypeError, ValueError):
                    pass

            st.markdown("---")

            # 오차 추이 차트
            if not bt.empty and bt['price_error'].notna().any():
                bt_p = bt.dropna(subset=['price_error']).copy()
                bt_p['date'] = pd.to_datetime(bt_p['date'])
                fig_err = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.05, row_heights=[0.55, 0.45])
                fig_err.add_trace(go.Scatter(
                    x=bt_p['date'], y=bt_p['actual_price'],
                    mode='lines', name='실제 WTI', line=dict(color='#58a6ff', width=1.5),
                    hovertemplate='%{x|%m/%d}<br>실제: $%{y:.2f}<extra></extra>',
                ), row=1, col=1)
                if 'stacking_pred' in bt_p.columns and bt_p['stacking_pred'].notna().any():
                    fig_err.add_trace(go.Scatter(
                        x=bt_p['date'], y=bt_p['stacking_pred'],
                        mode='lines', name='Stacking 예측',
                        line=dict(color='#3fb950', width=1.5, dash='dash'),
                        hovertemplate='%{x|%m/%d}<br>Stacking: $%{y:.2f}<extra></extra>',
                    ), row=1, col=1)
                if 'sarimax_pred' in bt_p.columns:
                    fig_err.add_trace(go.Scatter(
                        x=bt_p['date'], y=bt_p['sarimax_pred'],
                        mode='lines', name='SARIMAX 예측',
                        line=dict(color='#f0c040', width=1.5, dash='dash'),
                        hovertemplate='%{x|%m/%d}<br>SARIMAX: $%{y:.2f}<extra></extra>',
                    ), row=1, col=1)
                _err_clrs = ['#3fb950' if e >= 0 else '#f85149' for e in bt_p['price_error']]
                fig_err.add_trace(go.Bar(
                    x=bt_p['date'], y=bt_p['price_error'],
                    marker_color=_err_clrs, opacity=0.85, name='오차($)',
                    hovertemplate='%{x|%m/%d}<br>오차: $%{y:.2f}<extra></extra>',
                ), row=2, col=1)
                fig_err.add_hline(y=0, line=dict(color='rgba(255,255,255,0.4)', width=0.8), row=2, col=1)
                fig_err.update_layout(
                    paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                    font=dict(color='#c9d1d9'), height=380, hovermode='x unified',
                    legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', font=dict(size=10)),
                    margin=dict(l=60, r=20, t=20, b=40),
                )
                fig_err.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
                fig_err.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
                fig_err.update_yaxes(title_text='WTI ($)', title_font=dict(size=9), row=1, col=1)
                fig_err.update_yaxes(title_text='오차 ($)', title_font=dict(size=9), row=2, col=1)
                st.plotly_chart(fig_err, use_container_width=True)

            # ── 방향성 정확도 / 누적 오차 / 롤링 MAPE
            if not bt.empty and bt['price_error'].notna().any():
                _dir_pred_col = ('stacking_pred'
                                 if 'stacking_pred' in bt.columns and bt['stacking_pred'].notna().any()
                                 else 'sarimax_pred')
                bt_a = bt.dropna(subset=['price_error', 'actual_price', _dir_pred_col]).copy()
                bt_a['date'] = pd.to_datetime(bt_a['date'])
                bt_a = bt_a.sort_values('date').reset_index(drop=True)
                # dir_correct: pred vs prev_close (prediction_log에 사전 계산됨)
                if 'dir_correct' in bt_a.columns and bt_a['dir_correct'].notna().any():
                    dir_acc = bt_a['dir_correct'].dropna().mean() * 100
                else:
                    bt_a['actual_dir'] = bt_a['actual_price'].diff().apply(
                        lambda x: np.nan if pd.isna(x) else (1 if x > 0 else -1))
                    bt_a['pred_dir'] = bt_a[_dir_pred_col].diff().apply(
                        lambda x: np.nan if pd.isna(x) else (1 if x > 0 else -1))
                    _dc = (bt_a['actual_dir'] == bt_a['pred_dir']).astype(float).where(bt_a['actual_dir'].notna())
                    dir_acc = _dc.dropna().mean() * 100
                bt_a['abs_pct_err'] = bt_a['price_error_pct'].abs() if 'price_error_pct' in bt_a.columns else pd.Series(np.nan, index=bt_a.index)
                rolling_mape = (bt_a['abs_pct_err'].rolling(10).mean()
                                if 'price_error_pct' in bt_a.columns else pd.Series(dtype=float))
                bt_a['cum_abs_err'] = bt_a['price_error'].abs().cumsum()
                _lv_total_n = len(lv)

                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                col_d1.metric("방향성 정확도 (백테스트)", f"{dir_acc:.1f}%",
                              help="예측가 vs 전일 실제 종가 방향 일치율 (prediction_log dir_correct 기준)")
                col_d2.metric("라이브 예측 누적", f"{_lv_total_n}건",
                              help="운영 시작 후 누적 라이브 예측 수 (n<10 구간은 방향 정확도 통계적 무의미)")
                col_d3.metric("최근 10일 롤링 MAPE",
                              f"{rolling_mape.dropna().iloc[-1]:.2f}%" if len(rolling_mape.dropna()) > 0 else "—")
                col_d4.metric("누적 절대 오차", f"${bt_a['cum_abs_err'].iloc[-1]:.2f}" if len(bt_a) > 0 else "—")
                mean_mape2 = bt_a['abs_pct_err'].mean() if bt_a['abs_pct_err'].notna().any() else 0
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
                if mean_mape2:
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

            # 백테스트 vs 라이브 테이블
            st.markdown("---")
            col_bt, col_lv = st.columns(2)
            with col_bt:
                st.caption("**백테스트 (최근 10일)**")
                _bt_cols = [c for c in ['date','sarimax_pred','actual_price','price_error','price_error_pct'] if c in bt.columns]
                st.dataframe(bt.tail(10)[_bt_cols].rename(columns={
                    'date':'날짜','sarimax_pred':'SARIMAX예측($)','actual_price':'실제가($)',
                    'price_error':'오차($)','price_error_pct':'오차(%)',
                }), hide_index=True, use_container_width=True)
            with col_lv:
                st.caption("**라이브 예측 기록**")
                if live_confirmed.empty:
                    st.info("라이브 기록 없음")
                else:
                    _lv_cols = [c for c in ['date','sarimax_pred','actual_price','price_error','price_error_pct'] if c in lv.columns]
                    st.dataframe(lv[_lv_cols].rename(columns={
                        'date':'날짜','sarimax_pred':'예측가($)','actual_price':'실제가($)',
                        'price_error':'오차($)','price_error_pct':'오차(%)',
                    }), hide_index=True, use_container_width=True)

            # ── 예측 정확도 트렌드 차트 (fig_t)
            st.markdown("---")
            st.markdown("**📈 예측 정확도 트렌드 (Rolling 10일 MAPE)**")
            if not bt.empty and 'price_error_pct' in bt.columns:
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

            # ── Gap 분석 (4 summary metrics + fig_gap)
            st.markdown("---")
            st.markdown("**🔍 백테스트 vs 라이브 Gap 분석**")
            _gm_full = data.get('live_gap_monthly')
            _gs_full = data.get('live_gap_spikes')
            if _gm_full is not None and not _gm_full.empty:
                _bt_rows_g = _gm_full[_gm_full['type'] == 'backtest']
                _lv_rows_g = _gm_full[_gm_full['type'] == 'live']
                _bt_mae_g  = _bt_rows_g['mae'].mean() if not _bt_rows_g.empty else None
                _lv_mae_g  = _lv_rows_g['mae'].mean() if not _lv_rows_g.empty else None
                _ratio_g   = (_lv_mae_g / _bt_mae_g) if (_bt_mae_g and _lv_mae_g and _bt_mae_g > 0) else None
                _n_spk     = len(_gs_full) if _gs_full is not None else 0
                _gc1, _gc2, _gc3, _gc4 = st.columns(4)
                _gc1.metric("백테스트 MAE", f"${_bt_mae_g:.2f}" if _bt_mae_g else "—")
                _gc2.metric("라이브 MAE", f"${_lv_mae_g:.2f}" if _lv_mae_g else "—")
                _gc3.metric("Gap 배율", f"{_ratio_g:.2f}×" if _ratio_g else "—",
                            delta=f"{_ratio_g-1:+.2f}×" if _ratio_g else None,
                            delta_color="inverse")
                _gc4.metric("오차 스파이크 건수", str(_n_spk))
                _gm2 = _gm_full.copy()
                _gm2['month'] = _gm2['month'].astype(str)
                _months = sorted(_gm2['month'].unique())
                _bt_vals = [_gm2[(_gm2['month']==m) & (_gm2['type']=='backtest')]['mae'].values[0]
                            if len(_gm2[(_gm2['month']==m) & (_gm2['type']=='backtest')]) > 0 else None
                            for m in _months]
                _lv_vals = [_gm2[(_gm2['month']==m) & (_gm2['type']=='live')]['mae'].values[0]
                            if len(_gm2[(_gm2['month']==m) & (_gm2['type']=='live')]) > 0 else None
                            for m in _months]
                _bias_vals = [_gm2[(_gm2['month']==m) & (_gm2['type']=='live')]['bias'].values[0]
                              if (len(_gm2[(_gm2['month']==m) & (_gm2['type']=='live')]) > 0
                                  and 'bias' in _gm2.columns) else None
                              for m in _months]
                fig_gap = make_subplots(rows=1, cols=2,
                                        subplot_titles=['월별 MAE (백테스트 vs 라이브)', '라이브 편향 (Bias)'])
                fig_gap.add_trace(go.Bar(x=_months, y=_bt_vals, name='백테스트',
                                         marker_color='#3fb950', opacity=0.8,
                                         hovertemplate='%{x}<br>BT MAE: $%{y:.2f}<extra></extra>'),
                                  row=1, col=1)
                fig_gap.add_trace(go.Bar(x=_months, y=_lv_vals, name='라이브',
                                         marker_color='#f85149', opacity=0.8,
                                         hovertemplate='%{x}<br>Live MAE: $%{y:.2f}<extra></extra>'),
                                  row=1, col=1)
                _bias_colors = ['#f85149' if (v is not None and v < 0) else '#3fb950' for v in _bias_vals]
                fig_gap.add_trace(go.Bar(x=_months, y=_bias_vals, name='라이브 편향',
                                         marker_color=_bias_colors, opacity=0.85,
                                         hovertemplate='%{x}<br>Bias: $%{y:.3f}<extra></extra>'),
                                  row=1, col=2)
                fig_gap.add_hline(y=0, line=dict(color='rgba(255,255,255,0.4)', width=0.8), row=1, col=2)
                fig_gap.update_layout(
                    paper_bgcolor='#161b22', plot_bgcolor='#1c2433',
                    font=dict(color='#c9d1d9'),
                    height=300, margin=dict(l=50, r=20, t=50, b=40),
                    barmode='group',
                    legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', font=dict(size=9)),
                )
                fig_gap.update_xaxes(gridcolor='#21262d', tickfont=dict(size=9))
                fig_gap.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
                fig_gap.update_annotations(font=dict(color='#e6edf3', size=10))
                st.plotly_chart(fig_gap, use_container_width=True)
            else:
                st.info("Gap 분석 데이터 없음 (파이프라인 실행 후 생성)")

            csv_bytes = pl.to_csv(index=False).encode('utf-8')
            st.download_button("💾 전체 로그 CSV", csv_bytes, "prediction_log.csv", "text/csv")

    # ── Tab4: 파이프라인
    with tab_pipeline:
        st.subheader("🚀 파이프라인 관리")

        # 상태
        _meta_path = OUTPUT_DIR / 'run_meta.json'
        _log_path  = OUTPUT_DIR / 'pipeline_run.log'
        if _meta_path.exists():
            try:
                _meta = json.loads(_meta_path.read_text(encoding='utf-8'))
                c1, c2, c3 = st.columns(3)
                c1.metric("마지막 실행", _meta.get('last_run', '—'))
                c2.metric("데이터 기준", _meta.get('data_through', '—'))
                c3.metric("라이브 예측", f"{_meta.get('n_live', 0)}건")
                _api = _meta.get('api_status', {})
                if _api:
                    st.markdown("**API 수집 상태**")
                    for _api_name, _stat in _api.items():
                        st.caption(f"{_stat} `{_api_name}`")
            except Exception:
                pass

        # 캐시 상태
        st.markdown("---")
        st.markdown("**캐시 상태**")
        _cache_files = {
            'Optuna XGB':      OUTPUT_DIR / 'xgb_optuna_cache.json',
            'SVM 모델':        OUTPUT_DIR / 'svm_model_cache.pkl',
            'SVM C 파라미터': OUTPUT_DIR / 'svm_cache.json',
            'GARCH':           OUTPUT_DIR / 'garch_cache.pkl',
        }
        _cc = st.columns(len(_cache_files))
        for (name, path), col in zip(_cache_files.items(), _cc):
            if path.exists():
                _mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%m/%d %H:%M')
                col.metric(name, "✅", _mtime)
            else:
                col.metric(name, "❌", "없음")

        _news_path = OUTPUT_DIR / 'guardian_news_cache.csv'
        if _news_path.exists():
            try:
                _n_news = sum(1 for _ in open(_news_path, encoding='utf-8')) - 1
                st.caption(f"뉴스 캐시: {_n_news:,}건")
            except Exception:
                pass

        # 출력 파일 상태
        st.markdown("**출력 파일 상태**")
        _out_files = ['model_performance.csv', 'forecast_7days.csv', 'latest_risk_signal.csv',
                      'crisis_keywords.csv', 'oil_forecast_plot.png', 'wordcloud.png']
        _of_cols = st.columns(len(_out_files))
        for fname, col in zip(_out_files, _of_cols):
            _exists = (OUTPUT_DIR / fname).exists()
            col.caption(f"{'✅' if _exists else '❌'} `{fname}`")

        # 수동 실행
        st.markdown("---")
        st.markdown("**수동 실행**")
        if 'pipeline_running' not in st.session_state:
            st.session_state['pipeline_running'] = False

        if st.button("▶ 파이프라인 실행",
                     disabled=st.session_state['pipeline_running'],
                     type="primary"):
            st.session_state['pipeline_running'] = True
            _log_box  = st.empty()
            _pipe_path = Path(__file__).parent / 'oil_risk_mvp.py'
            try:
                import time as _time
                _log_f = open(_log_path, 'w', encoding='utf-8', buffering=1)
                _proc = subprocess.Popen(
                    [sys.executable, str(_pipe_path)],
                    cwd=str(Path(__file__).parent),
                    stdout=_log_f, stderr=subprocess.STDOUT,
                )
                while _proc.poll() is None:
                    if _log_path.exists():
                        with open(_log_path, encoding='utf-8', errors='replace') as _lf:
                            _tail = ''.join(_lf.readlines()[-8:])
                        _log_box.code(_tail, language=None)
                    _time.sleep(2)
                _log_f.close()
                st.session_state['pipeline_running'] = False
                if _proc.returncode == 0:
                    st.cache_data.clear()
                    _log_box.empty()
                    st.success("✅ 완료!")
                    st.rerun()
                else:
                    st.error(f"파이프라인 오류 (exit={_proc.returncode})")
            except Exception as _e:
                st.session_state['pipeline_running'] = False
                st.error(f"오류: {_e}")

        # 자동 스케줄러
        st.markdown("---")
        st.markdown("**자동 실행 스케줄러**")
        _sch_name = 'OilRiskPipeline'
        try:
            _r = subprocess.run(
                ['schtasks', '/Query', '/TN', _sch_name, '/FO', 'LIST'],
                capture_output=True, encoding='cp949', errors='replace',
            )
            if _r.returncode == 0:
                st.success("✅ 스케줄 등록됨")
                for line in _r.stdout.splitlines():
                    if any(k in line for k in ('다음 실행', '상태', 'Next Run', 'Status')):
                        st.caption(line.strip())
            else:
                st.warning("⚠️ 스케줄 미등록")
        except Exception:
            st.caption("스케줄러 상태 확인 불가")

        col_h, col_m, col_btn = st.columns([1, 1, 2])
        _h = col_h.number_input('시', 0, 23, 7, key='sch_h_v2')
        _m = col_m.number_input('분', 0, 59, 0, key='sch_m_v2')
        with col_btn:
            st.markdown('<br>', unsafe_allow_html=True)
            if st.button('등록/갱신', key='sch_reg_v2'):
                _sched_py = Path(__file__).parent / 'setup_scheduler.py'
                _r2 = subprocess.run(
                    [sys.executable, str(_sched_py), 'install', str(_h), str(_m)],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                )
                (st.success(_r2.stdout.strip() or '등록 완료')
                 if _r2.returncode == 0 else st.error(_r2.stderr))
        if st.button('스케줄 해제', type='secondary', key='sch_del_v2'):
            _sched_py = Path(__file__).parent / 'setup_scheduler.py'
            _r3 = subprocess.run(
                [sys.executable, str(_sched_py), 'remove'],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
            )
            st.info(_r3.stdout.strip() or '해제 완료')

        # 파이프라인 로그
        st.markdown("---")
        st.markdown("**파이프라인 로그 (최근)**")
        if _log_path.exists():
            with open(_log_path, encoding='utf-8', errors='replace') as _lf:
                st.text_area('pipeline_run.log', _lf.read()[-3000:], height=250)

        # 이메일 설정
        st.markdown("---")
        st.markdown("**📧 이메일 알림 설정**")
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
        st.markdown(f"발신: `{_env.get('SMTP_USER', '미설정')}` | 수신: `{_env.get('ALERT_TO', '미설정')}`")
        st.caption("급등위험 / 급락위험 감지 시 자동 발송")

        with st.form('email_cfg_v2'):
            _new_to = st.text_input('수신 이메일', value=_env.get('ALERT_TO', ''))
            _new_pw = st.text_input('Gmail 앱 비밀번호 (변경 시만)', type='password')
            if st.form_submit_button('저장'):
                _write_env_key('ALERT_TO', _new_to)
                if _new_pw:
                    _write_env_key('SMTP_PASSWORD', _new_pw)
                st.success('저장됨. 다음 파이프라인 실행부터 적용됩니다.')

        st.markdown("---")
        st.markdown("**테스트 발송**")
        if st.button('📧 테스트 이메일 발송', key='test_email_v2'):
            _spec = _ilu.spec_from_file_location('mvp', str(Path(__file__).parent / 'oil_risk_mvp.py'))
            _mvp  = _ilu.module_from_spec(_spec)
            try:
                _spec.loader.exec_module(_mvp)
                _test_sig = {'risk_level': 'SURGE_RISK', 'wti_price': 0.0,
                             'risk_score': 0.0, 'volatility_5d': 0.0,
                             'news_sentiment': 0.0, 'geopolitical_alert': False}
                ok = _mvp.send_risk_alert(_test_sig, None)
                if ok:
                    st.success(f"발송 완료 → {_read_env().get('ALERT_TO', '')}")
                else:
                    st.error('발송 실패. SMTP 설정(.env)을 확인하세요.')
            except Exception as _e:
                st.error(f'오류: {_e}')

    # ── Tab5: 사용자 관리
    with tab_users:
        st.subheader("👤 사용자 관리")
        _cfg_path = Path(__file__).parent / 'config/auth_config.yaml'
        with open(_cfg_path, encoding='utf-8') as _f:
            _cfg = yaml.safe_load(_f)
        _users = _cfg['credentials']['usernames']

        st.markdown("#### 계정 목록")
        _user_rows = [
            {'아이디': k, '이름': v.get('name', ''), '이메일': v.get('email', ''),
             '구독만료': v.get('subscription_expiry', '—')}
            for k, v in _users.items()
        ]
        st.dataframe(_user_rows, use_container_width=True)

        st.markdown("---")
        col_add, col_exp = st.columns(2)

        with col_add:
            st.markdown("#### 계정 추가")
            with st.form('add_user_v2', clear_on_submit=True):
                _new_id = st.text_input('아이디')
                _new_nm = st.text_input('이름')
                _new_em = st.text_input('이메일')
                _new_pw = st.text_input('비밀번호', type='password')
                if st.form_submit_button('추가'):
                    if _new_id and _new_pw:
                        if _new_id in _users:
                            st.error('이미 존재하는 아이디')
                        else:
                            _users[_new_id] = {'name': _new_nm, 'email': _new_em, 'password': _new_pw}
                            with open(_cfg_path, 'w', encoding='utf-8') as _fw:
                                yaml.dump(_cfg, _fw, allow_unicode=True)
                            st.success(f'{_new_id} 추가됨. 페이지를 새로고침하세요.')
                    else:
                        st.warning('아이디/비밀번호 필수')

        with col_exp:
            st.markdown("#### 구독 만료일")
            _exp_target = st.selectbox('계정', list(_users.keys()), key='exp_v2')
            _cur_exp = _users[_exp_target].get('subscription_expiry', '2026-12-31')
            try:
                _cur_exp_date = datetime.datetime.strptime(_cur_exp, '%Y-%m-%d').date()
            except Exception:
                _cur_exp_date = datetime.date.today()
            _new_exp = st.date_input('만료일', value=_cur_exp_date, key='exp_date_v2')
            if st.button('저장', key='exp_save_v2'):
                _users[_exp_target]['subscription_expiry'] = str(_new_exp)
                with open(_cfg_path, 'w', encoding='utf-8') as _fw:
                    yaml.dump(_cfg, _fw, allow_unicode=True)
                st.success(f'{_exp_target} 만료일 → {_new_exp}')

        st.markdown("---")
        col_pw, col_del = st.columns(2)

        with col_pw:
            st.markdown("#### 비밀번호 초기화")
            _pw_target = st.selectbox('계정', list(_users.keys()), key='pw_v2')
            with st.form('pw_reset_v2', clear_on_submit=True):
                _new_pw_r = st.text_input('새 비밀번호', type='password')
                _confirm  = st.text_input('비밀번호 확인', type='password')
                if st.form_submit_button('변경'):
                    if not _new_pw_r:
                        st.warning('비밀번호 입력 필요')
                    elif _new_pw_r != _confirm:
                        st.error('비밀번호 불일치')
                    else:
                        _users[_pw_target]['password'] = _new_pw_r
                        with open(_cfg_path, 'w', encoding='utf-8') as _fw:
                            yaml.dump(_cfg, _fw, allow_unicode=True)
                        st.success(f'{_pw_target} 비밀번호 변경됨')

        with col_del:
            st.markdown("#### 계정 삭제")
            _deletable = [k for k in _users if k != 'admin']
            if _deletable:
                _del_target = st.selectbox('삭제할 계정', _deletable, key='del_v2')
                if st.button('삭제', type='secondary', key='del_btn_v2'):
                    del _users[_del_target]
                    with open(_cfg_path, 'w', encoding='utf-8') as _fw:
                        yaml.dump(_cfg, _fw, allow_unicode=True)
                    st.success(f'{_del_target} 삭제됨. 페이지를 새로고침하세요.')
            else:
                st.info('삭제 가능한 계정 없음')

    st.markdown("---")
    st.caption("국제 유가 리스크 예측 시스템  |  XGBoost-HAR + SARIMAX + VAR Ensemble  |  News Sentiment + Geopolitical Risk")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

if _is_admin:
    render_admin_page()
else:
    render_user_page()
