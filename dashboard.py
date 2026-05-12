"""
국제 유가 리스크 예측 시스템 — Streamlit 대시보드
실행 방법: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime

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
@st.cache_data(ttl=180)
def load_all():
    out = {}
    for name in ['model_performance', 'forecast_7days', 'latest_risk_signal', 'crisis_keywords', 'prediction_log', 'feature_importance', 'risk_history']:
        p = OUTPUT_DIR / f'{name}.csv'
        if p.exists():
            out[name] = pd.read_csv(p)
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
    import json as _json
    _meta_path = OUTPUT_DIR / 'run_meta.json'
    if _meta_path.exists():
        try:
            _meta = _json.loads(_meta_path.read_text())
            st.markdown("**마지막 파이프라인 실행**")
            st.markdown(f"🕐 `{_meta.get('last_run', '—')}`")
            st.markdown(f"📅 데이터: `~ {_meta.get('data_through', '—')}`")
            st.markdown(f"📊 실시간 예측: `{_meta.get('n_live', 0)}건`")

            _api = _meta.get('api_status', {})
            if _api:
                st.markdown("**API 수집 상태**")
                for _name, _stat in _api.items():
                    st.markdown(f"{_stat} `{_name}`")
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
    st.caption(f"대시보드 로드: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛢 국제 유가 리스크 예측 시스템")

if not (OUTPUT_DIR / 'latest_risk_signal.csv').exists():
    st.warning("⚠️ 분석 결과가 없습니다. 사이드바에서 **파이프라인 실행** 버튼을 클릭하세요.")
    st.stop()

data = load_all()

# ── 리스크 신호 배너 ───────────────────────────────────────────────────────────
if 'latest_risk_signal' in data:
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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WTI 가격",   f"${sig['wti_price']:.2f}",
              f"{sig['momentum_5d']*100:+.1f}% (5일)")
    c2.metric("5일 변동성", f"{sig['volatility_5d']*100:.2f}%")
    c3.metric("뉴스 감성",  f"{sig['news_sentiment']:+.3f}",
              "부정" if sig['news_sentiment'] < -0.05 else
              ("긍정" if sig['news_sentiment'] > 0.05 else "중립"))
    c4.metric("뉴스 건수",  f"{int(sig['news_count'])}건")
    c5.metric("지정학 경보", "🔴 활성" if sig['geopolitical_alert'] else "🟢 없음")
    c6.metric("리스크 점수", f"{sig['risk_score']:.3f}")

st.markdown("---")

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 가격 예측", "🌡 리스크 상세", "☁ 키워드 분석", "📊 모델 성능", "📋 예측 오차 로그"]
)

# ── Tab 1: 가격 예측 ──────────────────────────────────────────────────────────
with tab1:
    col_chart, col_table = st.columns([2.2, 1])

    with col_chart:
        st.subheader("유가 예측 차트")
        img = OUTPUT_DIR / 'oil_forecast_plot.png'
        if img.exists():
            st.image(str(img), use_container_width=True)
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

            display_cols = {
                '날짜': '날짜',
                'forecast_price': '앙상블($)',
                'sarimax_forecast': 'SARIMAX($)',
                'xgb_forecast': 'XGB($)',
                'prophet_forecast': 'Prophet($)',
                '합의도': '합의도',
                'bias_correction': 'Bias($)',
            }
            show_fc = fc[[c for c in display_cols if c in fc.columns]].rename(columns=display_cols)
            st.dataframe(show_fc, hide_index=True, use_container_width=True)

            # 합의도 낮으면 경고
            if 'model_std' in fc.columns and float(fc['model_std'].iloc[0]) >= 5:
                st.warning(f"⚠️ 모델 간 예측 편차 ${fc['model_std'].iloc[0]:.1f} — 불확실성 높음")

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
            geo_n  = 1.0 if sig['geopolitical_alert'] else 0.0
            rs_n   = min(sig['risk_score'] / 3.0, 1.0)

            factors = {
                '📊 실현변동성':    vol_n,
                '📐 가격 방향성':  mom_n,
                '📰 뉴스 부정감성': sent_n,
                '🌍 지정학 리스크': geo_n,
                '⚡ 종합 리스크':   rs_n,
            }

            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#161b22')
            ax.set_facecolor('#1c2433')
            bar_colors = ['#ff6b6b', '#ffd93d', '#ff4757', '#ff6348', '#e74c3c']
            y = list(factors.keys())
            x = list(factors.values())
            bars = ax.barh(y, x, color=bar_colors, alpha=0.85, height=0.55)
            ax.set_xlim(0, 1.05)
            ax.axvline(0.5, color='white', lw=0.6, ls='--', alpha=0.4)
            ax.axvline(0.8, color='#e74c3c', lw=0.6, ls='--', alpha=0.4)
            for bar, val in zip(bars, x):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', color='white', fontsize=8)
            ax.tick_params(colors='#ccc', labelsize=8)
            for sp in ax.spines.values(): sp.set_color('#30363d')
            ax.set_xlabel('Relative Intensity (0–1)', color='#ccc', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("**상세 수치**")
            st.json({
                "날짜":        str(sig['date']),
                "리스크 레벨": sig['risk_level'],
                "리스크 점수": float(sig['risk_score']),
                "WTI 가격":    f"${sig['wti_price']}",
                "변동성(5일)": f"{sig['volatility_5d']*100:.2f}%",
                "모멘텀(5일)": f"{sig['momentum_5d']*100:+.2f}%",
                "뉴스 감성":   float(sig['news_sentiment']),
                "뉴스 기사 수": int(sig['news_count']),
                "지정학 경보":  bool(sig['geopolitical_alert']),
                "방향성 편향":  float(sig['directional_bias']),
            })

    st.markdown("---")
    sig_csv = (OUTPUT_DIR / 'latest_risk_signal.csv')
    if sig_csv.exists():
        st.download_button("💾 리스크 신호 CSV 다운로드",
                           sig_csv.read_bytes(),
                           "latest_risk_signal.csv", "text/csv")

    # ── 리스크 히스토리 타임라인
    st.markdown("---")
    st.markdown("**📅 리스크 레벨 히스토리**")
    _rh_path = OUTPUT_DIR / 'risk_history.csv'
    if _rh_path.exists():
        rh = pd.read_csv(_rh_path)
        rh['date'] = pd.to_datetime(rh['date'])
        rh = rh.sort_values('date')

        _LEVEL_CLR = {'NORMAL': '#2ecc71', 'CAUTION': '#f39c12',
                      'SURGE_RISK': '#e74c3c', 'DROP_RISK': '#3498db', 'CRITICAL': '#8e44ad'}

        fig_rh, axes_rh = plt.subplots(2, 1, figsize=(10, 4.5),
                                        facecolor='#161b22', sharex=True)
        fig_rh.subplots_adjust(hspace=0.08)

        ax_p = axes_rh[0]
        ax_p.set_facecolor('#1c2433')
        ax_p.plot(rh['date'], rh['wti_price'], color='#58a6ff', lw=1.5)
        ax_p.set_ylabel('WTI ($)', color='#ccc', fontsize=8)
        ax_p.tick_params(colors='#ccc', labelsize=7)
        for sp in ax_p.spines.values(): sp.set_color('#30363d')
        ax_p.grid(color='#21262d', lw=0.5)
        for _, row in rh.iterrows():
            ax_p.axvline(row['date'], color=_LEVEL_CLR.get(row['risk_level'], '#888'),
                         alpha=0.15, lw=3)

        ax_r = axes_rh[1]
        ax_r.set_facecolor('#1c2433')
        ax_r.fill_between(rh['date'], 0, rh['risk_score'], color='#f0c040', alpha=0.4)
        ax_r.plot(rh['date'], rh['risk_score'], color='#f0c040', lw=1.2)
        for val, clr in [(0.3, '#f39c12'), (0.6, '#e74c3c')]:
            ax_r.axhline(val, color=clr, lw=0.8, ls='--', alpha=0.7)
        ax_r.set_ylabel('리스크 점수', color='#ccc', fontsize=8)
        ax_r.tick_params(colors='#ccc', labelsize=7)
        for sp in ax_r.spines.values(): sp.set_color('#30363d')
        ax_r.grid(color='#21262d', lw=0.5)

        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=l) for l, c in _LEVEL_CLR.items()]
        ax_p.legend(handles=handles, fontsize=7, facecolor='#1c2433',
                    labelcolor='white', loc='upper left', ncol=5)
        plt.tight_layout()
        st.pyplot(fig_rh)
        plt.close()

        if len(rh) < 2:
            st.info("파이프라인을 2회 이상 실행하면 히스토리가 표시됩니다.")
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
            kw['분류'] = kw['is_crisis_word'].map({True: '🔴', False: '🔵'})
            st.dataframe(
                kw[['분류', 'keyword', 'count', 'weight']].rename(columns={
                    'keyword': '키워드', 'count': '빈도', 'weight': '가중치'
                }),
                hide_index=True, use_container_width=True
            )
            kw_csv = kw.to_csv(index=False).encode('utf-8')
            st.download_button("💾 키워드 CSV 다운로드", kw_csv,
                               "crisis_keywords.csv", "text/csv")

# ── Tab 4: 모델 성능 ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("📊 모델 성능 비교 (테스트셋 기준)")

    if 'model_performance' in data:
        pf = data['model_performance']
        col_t, col_g = st.columns([1, 1.2])

        with col_t:
            st.dataframe(pf, hide_index=True, use_container_width=True)
            st.caption("RMSE·MAE: 낮을수록 좋음  |  R²: 높을수록 좋음 (최대 1.0)")

        with col_g:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3.5), facecolor='#161b22')
            metrics = ['rmse', 'mae', 'r2']
            ylabels = ['RMSE (↓)', 'MAE (↓)', 'R² (↑)']
            colors  = ['#58a6ff', '#3fb950', '#f0c040']

            for ax, met, ylbl, clr in zip(axes, metrics, ylabels, colors):
                ax.set_facecolor('#1c2433')
                bars = ax.bar(pf['model'], pf[met], color=clr, alpha=0.85)
                ax.set_title(ylbl, color='#e6edf3', fontsize=9)
                ax.tick_params(colors='#ccc', labelsize=7)
                plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=7)
                for sp in ax.spines.values(): sp.set_color('#30363d')
                ax.grid(axis='y', color='#21262d', lw=0.5)
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h * 1.03,
                            f'{h:.3f}', ha='center', va='bottom',
                            color='white', fontsize=7)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("파이프라인 실행 후 성능 데이터가 표시됩니다.")

    # ── XGBoost 피처 중요도 차트
    st.markdown("---")
    _FEAT_KO = {
        'RV_5d':'5일 실현변동성', 'RV_22d':'22일 실현변동성', 'RV_1d':'1일 실현변동성',
        'return_1d':'1일 수익률', 'log_return':'로그 수익률',
        'vol_5d':'5일 변동성', 'vol_21d':'21일 변동성',
        'momentum_5d':'5일 모멘텀', 'momentum_21d':'21일 모멘텀',
        'price_vs_ma5':'MA5 대비 가격', 'price_vs_ma21':'MA21 대비 가격',
        'bb_position':'볼린저밴드 위치',
        'Brent':'브렌트 가격', 'DXY':'달러인덱스', 'WTI':'WTI 가격',
        'VIX':'VIX 공포지수', 'OVX':'OVX 원유변동성',
        'dxy_change':'달러인덱스 변화', 'vix_change':'VIX 변화',
        'vix_zscore':'VIX Z-스코어', 'ovx_change':'OVX 변화',
        'ovx_zscore':'OVX Z-스코어', 'ovx_rv_spread':'내재-실현변동성 스프레드',
        'demand_shock':'수요 충격', 'supply_shock':'공급 충격',
        'inv_chg_zscore':'재고 변화 Z-스코어', 'inv_lvl_zscore':'재고 수준 Z-스코어',
        'news_sentiment':'뉴스 감성', 'news_sentiment_smooth':'뉴스 감성(평활)',
        'news_count':'뉴스 건수', 'news_sentiment_lag1':'뉴스 감성(1일 전)',
        'news_sentiment_lag2':'뉴스 감성(2일 전)',
        'news_count_lag1':'뉴스 건수(1일 전)', 'news_count_lag2':'뉴스 건수(2일 전)',
        'geo_dummy':'지정학 위기 더미', 'gpr_zscore':'GPR Z-스코어',
        'fear_composite':'공포 복합지수', 'vix_amplified':'VIX 증폭 감성',
        'vix_sent_diverge':'VIX-감성 괴리',
        'regime':'시장 국면', 'regime_x_mom':'국면×모멘텀',
        'regime_x_sent':'국면×감성', 'regime_x_gpr':'국면×지정학',
        'futures_spread':'선물 커브 스프레드', 'futures_spread_chg':'선물 스프레드 변화',
        'contango_dummy':'콘탱고 더미', 'covid_dummy':'COVID 더미',
    }

    _fi_path = OUTPUT_DIR / 'feature_importance.csv'
    if _fi_path.exists():
        fi = pd.read_csv(_fi_path).head(15)
        fi['feature_ko'] = fi['feature'].map(_FEAT_KO).fillna(fi['feature'])
        fi = fi.sort_values('importance')

        fig_fi, ax_fi = plt.subplots(figsize=(8, 4.5), facecolor='#161b22')
        ax_fi.set_facecolor('#1c2433')
        bars = ax_fi.barh(fi['feature_ko'], fi['importance'], color='#58a6ff', alpha=0.85)
        ax_fi.set_title('XGBoost-HAR 피처 중요도 (Top 15)', color='#e6edf3', fontsize=10)
        ax_fi.tick_params(colors='#ccc', labelsize=8)
        ax_fi.set_xlabel('Importance', color='#ccc', fontsize=8)
        for sp in ax_fi.spines.values(): sp.set_color('#30363d')
        ax_fi.grid(axis='x', color='#21262d', lw=0.5)
        for bar in bars:
            w = bar.get_width()
            ax_fi.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{w:.3f}', va='center', color='#ccc', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()
    else:
        st.info("파이프라인 실행 후 피처 중요도가 표시됩니다.")

    st.markdown("---")
    st.markdown("""
    **모델 설명**
    | 모델 | 역할 | 특징 |
    |------|------|------|
    | **XGBoost-HAR** | 변동성(리스크) 예측 | HAR 구성요소 + 뉴스/지정학 외생변수 |
    | **SARIMAX** | 7일 가격 예측 | AR(2,1,2) × 주간 계절성 + 외생변수 |
    | **Ensemble** | 최종 예측 | SARIMAX 65% + XGBoost 35% 가중 평균 |
    """)

# ── Tab 5: 예측 오차 로그 ─────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 예측 vs 실제 오차 로그")

    if 'prediction_log' not in data:
        st.info("파이프라인 실행 후 오차 로그가 표시됩니다.")
    else:
        pl = data['prediction_log'].copy()

        # 요약 지표
        bt = pl[pl['type'] == 'backtest'].copy()
        lv = pl[pl['type'] == 'live'].copy()

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        if not bt.empty and bt['price_error'].notna().any():
            col_s1.metric("백테스트 MAE (가격)",
                          f"${bt['price_error'].abs().mean():.2f}")
            col_s2.metric("백테스트 MAPE",
                          f"{bt['price_error_pct'].abs().mean():.2f}%")
        live_confirmed = lv[lv['actual_price'].notna()]
        if not live_confirmed.empty:
            col_s3.metric("실시간 MAE (가격)",
                          f"${live_confirmed['price_error'].abs().mean():.2f}")
            col_s4.metric("실시간 MAPE",
                          f"{live_confirmed['price_error_pct'].abs().mean():.2f}%")

        # ── 드리프트 경고 (live MAPE > backtest MAPE × 2)
        if (not bt.empty and bt['price_error_pct'].notna().any()
                and not live_confirmed.empty):
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
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), facecolor='#161b22')

            bt_plot = bt.dropna(subset=['price_error']).copy()
            bt_plot['date'] = pd.to_datetime(bt_plot['date'])

            ax1 = axes[0]
            ax1.set_facecolor('#1c2433')
            ax1.plot(bt_plot['date'], bt_plot['actual_price'],
                     color='#58a6ff', lw=1.5, label='실제 WTI')
            ax1.plot(bt_plot['date'], bt_plot['sarimax_pred'],
                     color='#f0c040', lw=1.5, ls='--', label='SARIMAX 예측')
            ax1.set_ylabel('WTI ($)', color='#ccc', fontsize=9)
            ax1.tick_params(colors='#ccc', labelsize=8)
            ax1.legend(fontsize=8, facecolor='#1c2433', labelcolor='white')
            for sp in ax1.spines.values(): sp.set_color('#30363d')

            ax2 = axes[1]
            ax2.set_facecolor('#1c2433')
            colors = ['#3fb950' if e >= 0 else '#f85149'
                      for e in bt_plot['price_error']]
            ax2.bar(bt_plot['date'], bt_plot['price_error'],
                    color=colors, alpha=0.8, width=0.8)
            ax2.axhline(0, color='white', lw=0.6)
            ax2.set_ylabel('오차 (실제-예측, $)', color='#ccc', fontsize=9)
            ax2.tick_params(colors='#ccc', labelsize=8)
            for sp in ax2.spines.values(): sp.set_color('#30363d')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── 방향성 정확도 / 누적 오차 / 롤링 MAPE ────────────────────────────
        if not bt.empty and bt['price_error'].notna().any():
            bt_a = bt.dropna(subset=['price_error', 'actual_price', 'sarimax_pred']).copy()
            bt_a['date'] = pd.to_datetime(bt_a['date'])
            bt_a = bt_a.sort_values('date').reset_index(drop=True)

            # 방향성 정확도
            bt_a['actual_dir']  = bt_a['actual_price'].diff().apply(lambda x: 1 if x > 0 else -1)
            bt_a['pred_dir']    = bt_a['sarimax_pred'].diff().apply(lambda x: 1 if x > 0 else -1)
            bt_a['dir_correct'] = (bt_a['actual_dir'] == bt_a['pred_dir']).astype(float)
            dir_acc = bt_a['dir_correct'].dropna().mean() * 100

            bt_a['abs_pct_err'] = bt_a['price_error_pct'].abs()
            rolling_mape = bt_a['abs_pct_err'].rolling(10).mean()
            bt_a['cum_abs_err'] = bt_a['price_error'].abs().cumsum()

            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("방향성 정확도 (상승/하락)", f"{dir_acc:.1f}%",
                          help="예측 방향(상승/하락)이 실제와 일치한 비율")
            col_d2.metric("최근 10일 롤링 MAPE",
                          f"{rolling_mape.dropna().iloc[-1]:.2f}%" if rolling_mape.notna().any() else "—")
            col_d3.metric("누적 절대 오차", f"${bt_a['cum_abs_err'].iloc[-1]:.2f}")

            # 누적 오차 & 롤링 MAPE 차트
            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.2), facecolor='#161b22')

            ax_c = axes2[0]
            ax_c.set_facecolor('#1c2433')
            ax_c.plot(bt_a['date'], bt_a['cum_abs_err'], color='#f0c040', lw=1.5)
            ax_c.fill_between(bt_a['date'], 0, bt_a['cum_abs_err'], alpha=0.15, color='#f0c040')
            ax_c.set_title('누적 절대 오차 ($)', color='#e6edf3', fontsize=9)
            ax_c.set_ylabel('누적 오차 ($)', color='#ccc', fontsize=8)
            ax_c.tick_params(colors='#ccc', labelsize=7)
            for sp in ax_c.spines.values(): sp.set_color('#30363d')
            ax_c.grid(color='#21262d', lw=0.5)

            ax_r = axes2[1]
            ax_r.set_facecolor('#1c2433')
            ax_r.plot(bt_a['date'], rolling_mape, color='#3fb950', lw=1.5, label='10일 롤링 MAPE')
            mean_mape = bt_a['abs_pct_err'].mean()
            ax_r.axhline(mean_mape, color='#ff6b6b', lw=0.9, ls='--',
                         label=f'평균 {mean_mape:.2f}%')
            ax_r.set_title('롤링 MAPE — 10일 윈도우 (%)', color='#e6edf3', fontsize=9)
            ax_r.set_ylabel('MAPE (%)', color='#ccc', fontsize=8)
            ax_r.tick_params(colors='#ccc', labelsize=7)
            ax_r.legend(fontsize=7, facecolor='#1c2433', labelcolor='white')
            for sp in ax_r.spines.values(): sp.set_color('#30363d')
            ax_r.grid(color='#21262d', lw=0.5)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")

        # 테이블
        col_bt, col_lv = st.columns(2)
        with col_bt:
            st.markdown("**백테스트 (최근 10일)**")
            show_bt = bt.tail(10)[['date', 'sarimax_pred', 'actual_price',
                                   'price_error', 'price_error_pct']].rename(columns={
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
                show_lv = lv[['date', 'sarimax_pred', 'actual_price',
                              'price_error', 'price_error_pct']].rename(columns={
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

            fig_t, axes_t = plt.subplots(1, 2, figsize=(10, 2.8), facecolor='#161b22')

            for ax in axes_t:
                ax.set_facecolor('#1c2433')
                for sp in ax.spines.values(): sp.set_color('#30363d')
                ax.tick_params(colors='#ccc', labelsize=7)
                ax.grid(color='#21262d', lw=0.5)

            # MAPE 트렌드
            ax_m = axes_t[0]
            ax_m.plot(bt_trend['date'], bt_trend['rolling_mape'],
                      color='#3fb950', lw=1.5, label='10일 Rolling MAPE')
            ax_m.axhline(bt_trend['abs_pct'].mean(), color='#f0c040',
                         lw=0.9, ls='--', label=f"전체 평균 {bt_trend['abs_pct'].mean():.1f}%")
            ax_m.set_title('MAPE 트렌드 (%)', color='#e6edf3', fontsize=9)
            ax_m.set_ylabel('MAPE (%)', color='#ccc', fontsize=8)
            ax_m.legend(fontsize=7, facecolor='#1c2433', labelcolor='white')

            # MAE 트렌드
            ax_e = axes_t[1]
            ax_e.plot(bt_trend['date'], bt_trend['rolling_mae'],
                      color='#58a6ff', lw=1.5, label='10일 Rolling MAE')
            ax_e.axhline(bt_trend['price_error'].abs().mean(), color='#f0c040',
                         lw=0.9, ls='--', label=f"전체 평균 ${bt_trend['price_error'].abs().mean():.2f}")
            ax_e.set_title('MAE 트렌드 ($)', color='#e6edf3', fontsize=9)
            ax_e.set_ylabel('MAE ($)', color='#ccc', fontsize=8)
            ax_e.legend(fontsize=7, facecolor='#1c2433', labelcolor='white')

            # live 확인 건 표시
            if not live_confirmed.empty:
                live_confirmed_plot = live_confirmed.copy()
                live_confirmed_plot['date'] = pd.to_datetime(live_confirmed_plot['date'])
                ax_m.scatter(live_confirmed_plot['date'],
                             live_confirmed_plot['price_error_pct'].abs(),
                             color='#f85149', s=40, zorder=5, label='Live 실측')
                ax_e.scatter(live_confirmed_plot['date'],
                             live_confirmed_plot['price_error'].abs(),
                             color='#f85149', s=40, zorder=5, label='Live 실측')

            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close()
        else:
            st.info("트렌드 차트는 백테스트 10일 이상 데이터 필요")

        # 다운로드
        csv_bytes = pl.to_csv(index=False).encode('utf-8')
        st.download_button("💾 전체 로그 CSV 다운로드", csv_bytes,
                           "prediction_log.csv", "text/csv")

st.markdown("---")
st.caption("국제 유가 리스크 예측 시스템 MVP  |  XGBoost-HAR + SARIMAX Ensemble  |  News Sentiment + Geopolitical Risk")
