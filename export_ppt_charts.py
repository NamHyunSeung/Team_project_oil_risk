"""PPT용 차트 PNG 일괄 생성 스크립트
출력: output/ppt/ 폴더에 PNG 저장
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = Path("output")
PPT_DIR = OUT_DIR / "ppt"
PPT_DIR.mkdir(exist_ok=True)

_FEAT_KO = {
    'RV_5d':'5일 실현변동성', 'RV_21d':'22일 실현변동성', 'RV_22d':'22일 실현변동성',
    'RV_1d':'1일 실현변동성', 'return_1d':'1일 수익률', 'vol_5d':'5일 변동성',
    'price_vs_ma5':'MA5 대비 가격', 'price_vs_ma21':'MA21 대비 가격',
    'bb_position':'볼린저밴드 위치', 'dxy_change':'달러인덱스 변화',
    'demand_shock':'수요 충격', 'supply_shock':'공급 충격',
    'news_sentiment':'뉴스 감성', 'news_sentiment_smooth':'뉴스 감성(평활)',
    'news_count':'뉴스 건수', 'geo_dummy':'지정학 위기 더미',
    'gpr_zscore':'GPR Z-스코어', 'mom_5d':'5일 모멘텀', 'mom_21d':'21일 모멘텀',
    'macd':'MACD', 'rsi_14':'RSI(14)', 'brent_wti_spread':'브렌트-WTI 스프레드',
    'return_lag1':'수익률 래그1', 'return_lag2':'수익률 래그2',
    'vol_10d':'10일 변동성', 'vol_21d':'21일 변동성', 'RV_lag1':'RV 래그1',
    'news_sentiment_lag1':'뉴스감성 래그1', 'news_count_lag1':'뉴스건수 래그1',
}

BG   = '#161b22'
PLOT = '#1c2433'
FONT = '#c9d1d9'
HEAD = '#e6edf3'

COMMON = dict(
    paper_bgcolor=BG, plot_bgcolor=PLOT,
    font=dict(color=FONT, family='Noto Sans KR, sans-serif'),
)

def _save(fig, name, width=1200, height=None):
    h = height or fig.layout.height or 500
    path = PPT_DIR / f"{name}.png"
    fig.write_image(str(path), width=width, height=h, scale=2)
    print(f"  saved → {path}")


# ── 1. WTI 가격 추이 + 리스크 레벨 ──────────────────────────────────────
def chart_wti_risk_history():
    rh = pd.read_csv(OUT_DIR / "risk_history.csv")
    rh['date'] = pd.to_datetime(rh['date'])
    rh = rh.sort_values('date')

    _LEVEL_CLR = {
        'NORMAL': '#2ecc71', 'CAUTION': '#f39c12',
        'SURGE_RISK': '#e74c3c', 'DROP_RISK': '#3498db',
    }
    _LEVEL_KO = {
        'NORMAL': '정상', 'CAUTION': '주의',
        'SURGE_RISK': '급등위험', 'DROP_RISK': '급락위험',
    }

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=rh['date'], y=rh['wti_price'],
        mode='lines', name='WTI 가격',
        line=dict(color='#58a6ff', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>',
    ), secondary_y=False)

    for lv, clr in _LEVEL_CLR.items():
        mask = rh['risk_level'] == lv
        if mask.any():
            fig.add_trace(go.Scatter(
                x=rh[mask]['date'], y=rh[mask]['wti_price'],
                mode='markers', name=_LEVEL_KO[lv],
                marker=dict(color=clr, size=9, line=dict(color='white', width=1)),
                hovertemplate=f'%{{x|%m/%d}} {_LEVEL_KO[lv]}<br>$%{{y:.2f}}<extra></extra>',
            ), secondary_y=False)

    if 'risk_score' in rh.columns and rh['risk_score'].notna().any():
        fig.add_trace(go.Bar(
            x=rh['date'], y=rh['risk_score'],
            name='리스크 스코어', marker_color='rgba(231,76,60,0.20)',
            hovertemplate='%{x|%m/%d}<br>Risk: %{y:.3f}<extra></extra>',
        ), secondary_y=True)
        for y0, clr, dash in [(0.3, '#f39c12', 'dash'), (0.6, '#e74c3c', 'dash')]:
            fig.add_shape(type='line', yref='y2', xref='paper',
                          x0=0, x1=1, y0=y0, y1=y0,
                          line=dict(color=clr, dash=dash, width=1), opacity=0.6)

    fig.update_layout(
        **COMMON, height=420,
        title=dict(text='WTI 가격 + 리스크 레벨 이력', font=dict(color=HEAD, size=14)),
        legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                    font=dict(size=10), orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='날짜'),
        hovermode='x unified',
    )
    fig.update_yaxes(title_text='WTI (USD/bbl)', secondary_y=False,
                     gridcolor='#21262d', tickfont=dict(size=10))
    fig.update_yaxes(title_text='Risk Score', secondary_y=True,
                     gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10), showgrid=False)
    _save(fig, "01_wti_risk_history", height=420)


# ── 2. D+1~D+7 예측 차트 ──────────────────────────────────────────────────
def chart_forecast_7days():
    fc = pd.read_csv(OUT_DIR / "forecast_7days.csv")
    fc['date'] = pd.to_datetime(fc['date'])
    sig = pd.read_csv(OUT_DIR / "latest_risk_signal.csv")
    cur_px = float(sig.iloc[0]['wti_price'])

    fig = go.Figure()
    if 'lower_75ci' in fc.columns:
        ci_x = pd.concat([fc['date'], fc['date'][::-1]])
        ci_y = pd.concat([fc['upper_75ci'], fc['lower_75ci'][::-1]])
        fig.add_trace(go.Scatter(
            x=ci_x, y=ci_y, fill='toself',
            fillcolor='rgba(240,192,64,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='75% 신뢰구간', hoverinfo='skip',
        ))
    if 'var_5pct' in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc['date'], y=fc['var_5pct'],
            mode='lines', name='VaR 5%',
            line=dict(color='#e74c3c', width=1, dash='dot'),
            hovertemplate='%{x|%m/%d}<br>VaR(5%): $%{y:.2f}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=fc['date'], y=fc['forecast_price'],
        mode='lines+markers', name='예측가',
        line=dict(color='#f0c040', width=2.5, dash='dot'),
        marker=dict(size=9, symbol='circle-open', line=dict(color='#f0c040', width=2)),
        hovertemplate='%{x|%m/%d}<br>예측: $%{y:.2f}<extra></extra>',
    ))
    fig.add_hline(y=cur_px,
                  line=dict(color='#58a6ff', dash='dash', width=1.5),
                  annotation_text=f'현재 ${cur_px:.2f}',
                  annotation_font=dict(color='#58a6ff', size=11))

    fig.update_layout(
        **COMMON, height=380,
        title=dict(text='D+1 ~ D+7 가격 예측 (75% 신뢰구간)', font=dict(color=HEAD, size=14)),
        legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                    font=dict(size=10), orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='날짜'),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='WTI (USD/bbl)'),
        hovermode='x unified',
    )
    _save(fig, "02_forecast_7days", height=380)


# ── 3. 성능 비교 차트 (RMSE / MAE / MASE) ────────────────────────────────
def chart_model_performance():
    mp = pd.read_csv(OUT_DIR / "model_performance.csv")
    price = mp[mp['target'] == 'price'].copy()
    price = price[price['mae'] < 20].copy()
    price['label'] = price['model'].str.split('(').str[0].str.strip().str[:22]

    # MASE 비교 메인
    fig_mase = go.Figure()
    colors = []
    for lbl in price['label']:
        if 'SARIMAX' in lbl:
            colors.append('#f0c040')
        elif 'Stacking' in lbl:
            colors.append('#3fb950')
        elif 'Persistence' in lbl:
            colors.append('#8b949e')
        elif 'VAR' in lbl or 'ETS' in lbl:
            colors.append('#6e7681')
        else:
            colors.append('#58a6ff')

    fig_mase.add_trace(go.Bar(
        x=price['label'], y=price['mase'],
        marker_color=colors, opacity=0.88,
        text=[f'{v:.3f}' for v in price['mase']],
        textposition='outside', textfont=dict(color=FONT, size=10),
        hovertemplate='%{x}<br>MASE: %{y:.3f}<extra></extra>',
    ))
    fig_mase.add_hline(y=1.0, line=dict(color='#e74c3c', dash='dash', width=1.5),
                       annotation_text='MASE=1.0 (Naive 기준)',
                       annotation_font=dict(color='#e74c3c', size=10))
    fig_mase.update_layout(
        **COMMON, height=400,
        title=dict(text='모델별 MASE 비교 (↓ 낮을수록 우수)', font=dict(color=HEAD, size=14)),
        margin=dict(l=50, r=20, t=60, b=80),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=9), tickangle=20),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='MASE'),
        showlegend=False,
    )
    _save(fig_mase, "03_model_mase_comparison", height=400)

    # RMSE / MAE / R² 3패널
    fig3 = make_subplots(rows=1, cols=3,
                         subplot_titles=['RMSE (↓)', 'MAE (↓)', 'R² (↑)'])
    for ci, (met, clr) in enumerate(
            zip(['rmse', 'mae', 'r2'], ['#58a6ff', '#3fb950', '#f0c040']), 1):
        if met not in price.columns:
            continue
        fig3.add_trace(go.Bar(
            x=price['label'], y=price[met],
            marker_color=clr, opacity=0.85,
            text=[f'{v:.3f}' for v in price[met]],
            textposition='outside', textfont=dict(size=8, color=FONT),
            showlegend=False,
            hovertemplate='%{x}<br>' + met.upper() + ': %{y:.3f}<extra></extra>',
        ), row=1, col=ci)
    fig3.update_layout(
        **COMMON, height=400,
        title=dict(text='모델 성능 상세 비교', font=dict(color=HEAD, size=14)),
        margin=dict(l=40, r=20, t=70, b=80),
    )
    fig3.update_xaxes(gridcolor='#21262d', tickfont=dict(size=7), tickangle=25)
    fig3.update_yaxes(gridcolor='#21262d', tickfont=dict(size=9))
    fig3.update_annotations(font=dict(color=HEAD, size=11))
    _save(fig3, "04_model_performance_3panel", height=400)


# ── 4. 예측 vs 실제 차트 ──────────────────────────────────────────────────
def chart_pred_vs_actual():
    log = pd.read_csv(OUT_DIR / "prediction_log.csv")
    known = log[log['actual_price'].notna()].copy()
    if known.empty:
        print("  [SKIP] prediction_log: 실제값 없음")
        return
    known['forecast_date'] = pd.to_datetime(known['date'])
    known = known.sort_values('forecast_date').tail(90)
    if 'stacking_pred' not in known.columns:
        print("  [SKIP] stacking_pred 컬럼 없음")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=known['forecast_date'], y=known['actual_price'],
        mode='lines', name='실제가',
        line=dict(color='#58a6ff', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>실제: $%{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=known['forecast_date'], y=known['stacking_pred'],
        mode='lines+markers', name='예측가 (Stacking)',
        line=dict(color='#f0c040', width=1.5, dash='dot'),
        marker=dict(size=4),
        hovertemplate='%{x|%Y-%m-%d}<br>예측: $%{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        **COMMON, height=380,
        title=dict(text='예측 vs 실제 가격 (최근 90일)', font=dict(color=HEAD, size=14)),
        legend=dict(bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                    font=dict(size=10), orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='WTI (USD/bbl)'),
        hovermode='x unified',
    )
    _save(fig, "05_pred_vs_actual", height=380)


# ── 5. 피처 중요도 ─────────────────────────────────────────────────────────
def chart_feature_importance():
    fi = pd.read_csv(OUT_DIR / "feature_importance.csv").head(15).copy()
    fi['label'] = fi['feature'].map(_FEAT_KO).fillna(fi['feature'])
    fi = fi.sort_values('importance')

    fig = go.Figure(go.Bar(
        x=fi['importance'], y=fi['label'], orientation='h',
        marker_color='#58a6ff', opacity=0.85,
        text=[f'{v:.3f}' for v in fi['importance']],
        textposition='outside', textfont=dict(color=FONT, size=10),
        hovertemplate='%{y}<br>중요도: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        **COMMON, height=480,
        title=dict(text='XGBoost-HAR 피처 중요도 Top 15', font=dict(color=HEAD, size=14)),
        margin=dict(l=10, r=80, t=60, b=20),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='중요도'),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10)),
        showlegend=False,
    )
    _save(fig, "06_feature_importance", height=480)


# ── 6. 리스크 팩터 강도 ────────────────────────────────────────────────────
def chart_risk_factors():
    sig = pd.read_csv(OUT_DIR / "latest_risk_signal.csv").iloc[0]
    factors = {
        '5일 변동성':   min(float(sig['volatility_5d']) * 22, 1.0),
        '모멘텀 강도':  min(abs(float(sig['momentum_5d'])) * 8, 1.0),
        '부정 감성':    min(max(-float(sig['news_sentiment']), 0), 1.0),
        '지정학 위기':  1.0 if str(sig['geopolitical_alert']).lower() in ('true','1') else 0.0,
        '종합 위험':    min(float(sig['risk_score']) / 3.0, 1.0),
    }

    fig = go.Figure(go.Bar(
        x=list(factors.values()), y=list(factors.keys()),
        orientation='h',
        marker_color=['#ff6b6b','#ffd93d','#ff4757','#ff6348','#e74c3c'],
        opacity=0.88,
        text=[f'{v:.2f}' for v in factors.values()],
        textposition='outside', textfont=dict(color=FONT, size=12),
    ))
    fig.add_vline(x=0.5, line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1.5))
    risk_lv = str(sig['risk_level'])
    risk_ko = {'NORMAL':'정상','CAUTION':'주의','SURGE_RISK':'급등위험','DROP_RISK':'급락위험'}.get(risk_lv, risk_lv)
    fig.update_layout(
        **COMMON, height=300,
        title=dict(text=f'리스크 팩터 강도 — 현재 등급: {risk_ko} | WTI ${float(sig["wti_price"]):.2f}',
                   font=dict(color=HEAD, size=14)),
        margin=dict(l=10, r=80, t=60, b=30),
        xaxis=dict(range=[0, 1.4], gridcolor='#21262d', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=12)),
        showlegend=False,
    )
    _save(fig, "07_risk_factors", height=300)


# ── 7. 앙상블 가중치 ────────────────────────────────────────────────────────
def chart_ensemble_weights():
    weights_path = OUT_DIR / "stacking_weights_ema.json"
    if not weights_path.exists():
        print("  [SKIP] stacking_weights_ema.json 없음")
        return
    w = json.loads(weights_path.read_text())
    names = w['names']
    vals  = w['weights']

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=['#f0c040','#3fb950','#58a6ff'],
        opacity=0.88,
        text=[f'{v:.3f}' for v in vals],
        textposition='outside', textfont=dict(color=FONT, size=12),
        hovertemplate='%{x}<br>가중치: %{y:.4f}<extra></extra>',
    ))
    fig.update_layout(
        **COMMON, height=320,
        title=dict(text='Stacking 앙상블 가중치 (EMA 기반 동적 조정)',
                   font=dict(color=HEAD, size=14)),
        margin=dict(l=50, r=20, t=60, b=50),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=11)),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10), title='가중치'),
        showlegend=False,
    )
    _save(fig, "08_ensemble_weights", height=320)


# ── 8. 파이프라인 아키텍처 텍스트 카드 ────────────────────────────────────
def chart_data_pipeline():
    stages = [
        ('yfinance\nWTI/Brent\nDXY/VIX/OVX', '#58a6ff'),
        ('FRED API\n거시경제\nWTI fallback', '#3fb950'),
        ('EIA API\n원유 재고\n서프라이즈', '#f0c040'),
        ('RSS 50+\n실시간 헤드라인\nFinBERT 감성', '#e67e22'),
        ('CFTC CoT\n롱/숏 포지션\nManaged Money', '#9b59b6'),
        ('GPR 지수\n지정학 리스크\n월별', '#e74c3c'),
    ]
    fig = go.Figure()
    for i, (label, color) in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=[i], y=[0.5],
            mode='markers+text',
            marker=dict(size=80, color=color, opacity=0.3,
                        line=dict(color=color, width=2)),
            text=[label], textposition='middle center',
            textfont=dict(color='white', size=10),
            hoverinfo='skip', showlegend=False,
        ))
        if i < len(stages) - 1:
            fig.add_annotation(
                x=i + 0.5, y=0.5, ax=i + 0.3, ay=0.5,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowcolor='#8b949e', arrowwidth=2,
            )
    fig.update_layout(
        **COMMON, height=250,
        title=dict(text='데이터 수집 파이프라인 (6개 소스)', font=dict(color=HEAD, size=14)),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False, range=[-0.6, len(stages) - 0.4]),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    _save(fig, "09_data_pipeline", height=250)


if __name__ == '__main__':
    import sys
    print("PPT 차트 생성 중...")
    funcs = [
        chart_wti_risk_history,
        chart_forecast_7days,
        chart_model_performance,
        chart_pred_vs_actual,
        chart_feature_importance,
        chart_risk_factors,
        chart_ensemble_weights,
        chart_data_pipeline,
    ]
    errors = []
    for fn in funcs:
        try:
            print(f"[{fn.__name__}]")
            fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append((fn.__name__, str(e)))
    print(f"\n완료: {len(funcs)-len(errors)}/{len(funcs)} 성공")
    if errors:
        for name, err in errors:
            print(f"  FAILED: {name} → {err}")
    sys.exit(0)
