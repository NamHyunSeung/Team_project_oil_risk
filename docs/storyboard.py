"""
스토리보드 이미지 생성기
output: docs/storyboard.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as _fm

_korean_font = next(
    (f.name for f in _fm.fontManager.ttflist if 'Malgun' in f.name),
    'DejaVu Sans'
)
plt.rcParams['font.family'] = _korean_font
plt.rcParams['axes.unicode_minus'] = False

FIG_W, FIG_H = 20, 26
BG = '#F5F7FA'
HEADER_C = '#1A3A5C'
SCREEN_C = '#FFFFFF'
SCREEN_BORDER = '#CBD5E0'
TAB_ACTIVE = '#2B6CB0'
TAB_INACTIVE = '#718096'
ARROW_C = '#4A5568'
SECTION_BG = '#EBF4FF'
LABEL_C = '#2D3748'
SUB_C = '#4A5568'
ACC_RED = '#E53E3E'
ACC_YELLOW = '#D69E2E'
ACC_GREEN = '#38A169'
ACC_BLUE = '#3182CE'

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)


def screen(ax, x, y, w, h, title, title_bg=HEADER_C, radius=0.3):
    outer = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           linewidth=1.5, edgecolor=SCREEN_BORDER,
                           facecolor=SCREEN_C, zorder=2)
    ax.add_patch(outer)
    header = FancyBboxPatch((x, y + h - 0.55), w, 0.55,
                            boxstyle=f"round,pad=0.02,rounding_size={radius}",
                            linewidth=0, edgecolor='none',
                            facecolor=title_bg, zorder=3)
    ax.add_patch(header)
    ax.text(x + w/2, y + h - 0.275, title,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='white', zorder=4)


def section_box(ax, x, y, w, h, label, bg=SECTION_BG, fs=8):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.1",
                       linewidth=0.8, edgecolor='#BEE3F8',
                       facecolor=bg, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center', fontsize=fs, color=LABEL_C, zorder=4,
            wrap=True)


def tab_bar(ax, x, y, w, tabs, active_idx=0):
    tw = w / len(tabs)
    for i, t in enumerate(tabs):
        c = TAB_ACTIVE if i == active_idx else '#EDF2F7'
        tc = 'white' if i == active_idx else TAB_INACTIVE
        p = FancyBboxPatch((x + i*tw, y), tw, 0.35,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           linewidth=0.5, edgecolor=SCREEN_BORDER,
                           facecolor=c, zorder=4)
        ax.add_patch(p)
        ax.text(x + i*tw + tw/2, y + 0.175, t,
                ha='center', va='center', fontsize=6.5,
                color=tc, fontweight='bold' if i == active_idx else 'normal', zorder=5)


def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW_C,
                                lw=1.8, connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color=ARROW_C,
                ha='left', va='center', style='italic')


def badge(ax, x, y, txt, color):
    p = FancyBboxPatch((x, y), 1.0, 0.28,
                       boxstyle="round,pad=0.03,rounding_size=0.1",
                       linewidth=0, facecolor=color, zorder=5)
    ax.add_patch(p)
    ax.text(x + 0.5, y + 0.14, txt, ha='center', va='center',
            fontsize=7, color='white', fontweight='bold', zorder=6)


# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
ax.text(FIG_W/2, 25.5, '국제 유가 리스크 예측 시스템 — 스토리보드',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=HEADER_C)
ax.text(FIG_W/2, 25.1, 'XGBoost-HAR · SARIMAX · VAR Ensemble  |  News Sentiment · Geopolitical Risk · Black Swan Detection',
        ha='center', va='center', fontsize=8.5, color=SUB_C)
ax.axhline(24.8, color=HEADER_C, lw=1.5, alpha=0.3)

# ──────────────────────────────────────────────
# SCREEN 1 — 로그인
# ──────────────────────────────────────────────
sx1, sy1, sw1, sh1 = 0.8, 20.8, 4.5, 3.6
screen(ax, sx1, sy1, sw1, sh1, '① 로그인 화면')
section_box(ax, sx1+0.3, sy1+2.3, sw1-0.6, 0.65, '🛢  국제 유가 리스크 예측 시스템', bg='#EBF4FF', fs=8)
section_box(ax, sx1+0.3, sy1+1.55, sw1-0.6, 0.6, '아이디  _______________', bg='#F7FAFC', fs=8)
section_box(ax, sx1+0.3, sy1+0.85, sw1-0.6, 0.6, '비밀번호  _______________', bg='#F7FAFC', fs=8)
section_box(ax, sx1+1.2, sy1+0.2, sw1-2.4, 0.5, '로그인', bg=TAB_ACTIVE, fs=8.5)
ax.text(sx1+sw1/2, sy1+0.35, '', ha='center', va='center', fontsize=8, color='white')

# ──────────────────────────────────────────────
# SCREEN 2 — 사용자 뷰
# ──────────────────────────────────────────────
sx2, sy2, sw2, sh2 = 0.8, 12.5, 8.8, 7.8
screen(ax, sx2, sy2, sw2, sh2, '② 사용자 대시보드  (일반 회원)')

# Risk Hero
section_box(ax, sx2+0.3, sy2+6.5, sw2-0.6, 0.85,
            '🟡  CAUTION (주의)  |  WTI $91.88/bbl  |  리스크 스코어 0.894', bg='#FEFCBF', fs=8.5)

# Key metrics row
for i, (lbl, val) in enumerate([('5일 변동성','1.38%'),('뉴스 감성','-0.11'),('OVX','65.1 HIGH'),('헤지 비율','0.65')]):
    section_box(ax, sx2+0.3+i*2.05, sy2+5.55, 1.9, 0.8, f'{lbl}\n{val}', bg='#EDF2F7', fs=7.5)

# Price chart
section_box(ax, sx2+0.3, sy2+3.2, sw2-0.6, 2.15,
            '📈 가격 예측 차트  (실제 vs 예측 / 7일 예측 / 신뢰구간)\n'
            '─────────────────────────────────────────\n'
            '       [WTI 가격 라인]  [예측 라인]  [CI 밴드]', bg='#F0FFF4', fs=8)

# Risk drivers
section_box(ax, sx2+0.3, sy2+1.8, (sw2-0.75)/2, 1.2,
            '⚡ 리스크 동인\n지정학 활성 ▲\nOVX 고공 유지', bg='#FFF5F5', fs=7.5)
section_box(ax, sx2+0.3+(sw2-0.75)/2+0.15, sy2+1.8, (sw2-0.75)/2, 1.2,
            '📰 뉴스 경보\n위기 키워드 분석\n감성 스코어 트렌드', bg='#FFFAF0', fs=7.5)

# Forecast
section_box(ax, sx2+0.3, sy2+0.2, sw2-0.6, 1.4,
            '🔮 D+1~D+7 예측  |  VaR(5%) $89.31  |  Q10: $90.12  Q90: $93.76', bg='#EBF4FF', fs=8)

badge(ax, sx2+0.3, sy2+7.42, '일반 접근', ACC_BLUE)

# ──────────────────────────────────────────────
# SCREEN 3 — 관리자 탭 전체
# ──────────────────────────────────────────────
sx3, sy3, sw3, sh3 = 10.4, 12.5, 9.0, 7.8
screen(ax, sx3, sy3, sw3, sh3, '③ 관리자 대시보드  (admin 전용)', title_bg='#742A2A')
badge(ax, sx3+0.3, sy3+7.42, '관리자 전용', ACC_RED)

tabs3 = ['🌡 리스크\n현황', '📊 모델\n모니터링', '📋 예측\n오차', '🚀 파이프\n라인', '👤 사용자\n관리']
tab_bar(ax, sx3+0.3, sy3+6.55, sw3-0.6, tabs3, active_idx=0)

# Tab content panels (stacked vertically as examples)
panel_contents = [
    ('Tab ① 리스크 현황',
     '사용자 뷰와 동일\n+ 리스크 신호 JSON 다운로드',
     '#F0FFF4'),
    ('Tab ② 모델 모니터링',
     'forecast_reliability  /  모델 나이  /  OVX 레짐\n앙상블 가중치  (SARIMAX 0.780 · XGB 0.131 · VAR 0.089)\n모델 성능  R²=0.9711  MAE=$2.03\n블랙스완 페널티 상태  (compound_shock / jump_flag)',
     '#EBF8FF'),
    ('Tab ③ 예측 오차',
     '백테스트 90일 실제 vs 예측 비교\n오차 분포 히스토그램  /  최대 오차 이벤트\n라이브-백테스트 gap 추이  /  스파이크 분석',
     '#FFFAF0'),
    ('Tab ④ 파이프라인',
     '수동 실행 버튼  /  실행 로그 스트리밍\n마지막 실행 시간  /  모델 나이 경고',
     '#FFF5F5'),
    ('Tab ⑤ 사용자 관리',
     '계정 목록  /  계정 추가  /  비밀번호 초기화\n구독 만료일 설정  /  계정 삭제',
     '#FAF5FF'),
]

py = sy3 + 0.15
for i, (title, body, bg) in enumerate(panel_contents):
    ph = 1.15
    section_box(ax, sx3+0.3, py + i*ph, sw3-0.6, ph-0.08,
                f'[{title}]\n{body}', bg=bg, fs=7.2)

# ──────────────────────────────────────────────
# FLOW ARROWS
# ──────────────────────────────────────────────
# Login → 분기
arrow(ax, sx1 + sw1/2, sy1,      sx1 + sw1/2, sy1 - 0.5)  # 아래로
ax.plot([sx1+sw1/2, 5.25], [sy1-0.5, sy1-0.5], color=ARROW_C, lw=1.8)
ax.plot([5.25, 5.25],       [sy1-0.5, sy2+sh2], color=ARROW_C, lw=1.8)
ax.plot([5.25, 14.9],       [sy1-0.5, sy1-0.5], color=ARROW_C, lw=1.8)
ax.plot([14.9, 14.9],       [sy1-0.5, sy2+sh2], color=ARROW_C, lw=1.8)

# 화살촉
ax.annotate('', xy=(sx2+sw2/2, sy2+sh2), xytext=(sx2+sw2/2, sy2+sh2+0.01),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.8))
ax.annotate('', xy=(sx3+sw3/2, sy2+sh2), xytext=(sx3+sw3/2, sy2+sh2+0.01),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.8))

ax.text(4.3, sy1-0.65, '일반 로그인', fontsize=8, color=ACC_BLUE, style='italic')
ax.text(13.0, sy1-0.65, 'admin 로그인', fontsize=8, color=ACC_RED, style='italic')

# ──────────────────────────────────────────────
# DATA FLOW (하단)
# ──────────────────────────────────────────────
dy = 10.8
ax.axhline(dy + 1.3, color=HEADER_C, lw=1, alpha=0.2)
ax.text(FIG_W/2, dy+1.05, '데이터 파이프라인  (oil_risk_mvp.py)',
        ha='center', fontsize=11, fontweight='bold', color=HEADER_C)

data_boxes = [
    (0.5,  'yfinance\nWTI / Brent\n/ VIX / OVX', '#E9F5E9'),
    (3.3,  'RSS / 뉴스\nFinBERT 감성\n관세·지정학', '#FFF9E6'),
    (6.1,  'EIA 재고\nCOT 포지션\nGPR 지수', '#E6F0FF'),
    (8.9,  'CEEMDAN\n신호 분해\n피처 생성', '#F0E6FF'),
    (11.7, 'SARIMAX\nXGBoost\nVAR', '#FFE6E6'),
    (14.5, '앙상블\n(InvMAE+Roll30\n+BS페널티)', '#E6FFEE'),
    (17.3, '리스크\n분류·경보\nCI 확대', '#FFF0E6'),
]

for bx, (x, lbl, bg) in enumerate(data_boxes):
    section_box(ax, x, dy, 2.5, 0.95, lbl, bg=bg, fs=7.5)
    if bx < len(data_boxes)-1:
        ax.annotate('', xy=(x+2.6, dy+0.475), xytext=(x+2.5, dy+0.475),
                    arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.5))

# ──────────────────────────────────────────────
# MODEL PERFORMANCE SUMMARY (하단)
# ──────────────────────────────────────────────
py2 = 8.5
ax.text(FIG_W/2, py2+1.2, '핵심 성능 지표',
        ha='center', fontsize=10, fontweight='bold', color=HEADER_C)

perf = [
    ('Stacking R²',  '0.9711',  ACC_GREEN),
    ('Stacking MAE', '$2.03',   ACC_BLUE),
    ('Max Error',    '$9.61',   ACC_YELLOW),
    ('Dir Accuracy', '58.9%',   ACC_BLUE),
    ('HAR Vol R²',   '0.657',   ACC_GREEN),
    ('SARIMAX',      'w=0.780', ACC_BLUE),
]
pw = (FIG_W - 1.0) / len(perf)
for i, (lbl, val, c) in enumerate(perf):
    bx2 = 0.5 + i*pw
    p = FancyBboxPatch((bx2, py2-0.05), pw-0.2, 1.1,
                       boxstyle="round,pad=0.05,rounding_size=0.15",
                       linewidth=1, edgecolor=c, facecolor=SCREEN_C, zorder=3)
    ax.add_patch(p)
    ax.text(bx2+(pw-0.2)/2, py2+0.65, lbl, ha='center', va='center',
            fontsize=7.5, color=SUB_C, zorder=4)
    ax.text(bx2+(pw-0.2)/2, py2+0.2, val, ha='center', va='center',
            fontsize=11, fontweight='bold', color=c, zorder=4)

# ──────────────────────────────────────────────
# BLACK SWAN SECTION
# ──────────────────────────────────────────────
by = 6.2
ax.text(FIG_W/2, by+1.9, '블랙스완 대응 로직',
        ha='center', fontsize=10, fontweight='bold', color=HEADER_C)

bs_items = [
    ('tariff / 관세\n키워드 감지', ACC_YELLOW),
    ('jump_flag\n|ret|>3σ(60d)', ACC_RED),
    ('OPEC 이벤트\n별도 스코어', '#805AD5'),
    ('compound_shock\n지정학+OPEC+뉴스', ACC_RED),
    ('_bs_pen\nSARIMAX ×0.25~0.60', ACC_ORANGE := '#DD6B20'),
    ('CI 확대\n×1.5 → ×2.5', ACC_RED),
]
bw = (FIG_W - 1.0) / len(bs_items)
for i, (lbl, c) in enumerate(bs_items):
    bx3 = 0.5 + i*bw
    section_box(ax, bx3, by, bw-0.2, 1.65, lbl, bg=SCREEN_C, fs=8)
    p2 = ax.patches[-1]
    p2.set_edgecolor(c)
    p2.set_linewidth(2)
    if i < len(bs_items)-1:
        ax.annotate('', xy=(bx3+bw+0.0, by+0.825), xytext=(bx3+bw-0.2, by+0.825),
                    arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.2))

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
ax.axhline(0.6, color=HEADER_C, lw=1, alpha=0.2)
ax.text(FIG_W/2, 0.3,
        'XGBoost-HAR (변동성)  +  SARIMAX (가격 1-step)  +  VAR (다변량)  →  Stacking Ensemble  |  '
        'Sentence Embedding  ·  FinBERT  ·  Oil Event Library 65개  |  '
        'Black Swan: jump_flag · compound_shock · _bs_pen',
        ha='center', va='center', fontsize=7, color=SUB_C)

plt.tight_layout(pad=0)
out = r'c:\Users\namzx\Team_project_oil_risk\docs\storyboard.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'saved: {out}')
