import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CAT_COLOR = {
    '예측 정확도': '#2E75B6',
    '리스크 감지': '#3A9A48',
    '운영 안정화': '#C07020',
    '비즈니스':    '#7B3FA0',
}

# (phase label, date, category, title, 2줄 desc)
phases = [
    ('Ph.0~1',   '05-07',       '예측 정확도', 'MVP 런칭',              '7일 가격 예측 파이프라인\n핵심 기술 스택 확립'),
    ('Ph.3~4',   '05-11~12',    '예측 정확도', '불필요 모델 도태',      '실전 부적합 모델 제거\n예측 신뢰성 기반 정비'),
    ('Ph.5~6',   '05-12~13',    '예측 정확도', '감성 분석 통합',        '과적합 84% 감소\n뉴스 신호 500+ 사전'),
    ('Ph.7~8',   '05-14~15',    '예측 정확도', '자동 최적화 도입',      '하이퍼파라미터 자동 탐색\n수동 튜닝 완전 대체'),
    ('Ph.9~11',  '05-16~21',    '예측 정확도', '3중 예측 앙상블',       '방향성 정확도 61%\n상승 · 하락 신호 추가'),
    ('Ph.12',    '05-21~25',    '비즈니스',    '운영 시스템 구축',      '사용자 인증 + 관리자\n대시보드 초기 버전'),
    ('Ph.13~14', '05-25~06-01', '운영 안정화', '예측 오차 40% 개선 ★', 'MASE=0.602 확정\n데이터 품질 4건 수정'),
    ('Ph.15',    '06-01~02',    '리스크 감지', '리스크 신호 고도화',    'VaR · OVX · 헤지 비율\n블랙스완 조기 경보'),
    ('Ph.16',    '06-02',       '운영 안정화', '모델 신뢰도 확보 ★',   '예측 변동성 R² +27%\n불필요 변수 9개 제거'),
    ('Ph.17~19', '06-04',       '운영 안정화', '극단 오차 방어',        '충격 후 오차 $9→$2\n3중 보정 레이어'),
    ('Ph.20~21', '06-04',       '비즈니스',    '분석 대시보드 완성',    '오차 원인 전수 분석\n가격 · 뉴스 · 성과 3탭'),
    ('Ph.22~23', '06-04',       '비즈니스',    '수익화 모델 구축',      '3단계 구독 플랜\n₩0 · ₩49만 · ₩129만'),
    ('Ph.24~25', '06-05',       '비즈니스',    '계정 자동화',           '자동 가입 · 만료 관리\n관리자 개입 최소화'),
    ('Ph.26',    '06-05',       '비즈니스',    'SaaS 완성 ★',          '업그레이드 인앱 승인\n완전한 구독 플로우'),
]

MILESTONES = {'Ph.13~14', 'Ph.16', 'Ph.26'}

N       = len(phases)   # 14
SPACING = 2.9
FIG_W   = N * SPACING + 4.5
FIG_H   = 17.0
BOX_W   = 2.55
BOX_H   = 3.0
ABOVE_Y = 3.9
BELOW_Y = -3.9

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('#F8F8F8')
ax.set_facecolor('#F8F8F8')
ax.set_xlim(0, FIG_W)
ax.set_ylim(-8.8, 9.2)
ax.axis('off')

# 제목
ax.text(FIG_W / 2, 8.5,
        '유가 리스크 예측 SaaS  —  개발 여정',
        ha='center', va='center', fontsize=18, fontweight='bold', color='#111111')
ax.text(FIG_W / 2, 7.7,
        '2026-05-07 ~ 2026-06-05  |  4주 스프린트  |  ★ = 핵심 비즈니스 마일스톤',
        ha='center', va='center', fontsize=11, color='#555555')

# 중앙 화살표
ax.annotate('', xy=(FIG_W - 1.3, 0), xytext=(0.8, 0),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=3.0, mutation_scale=22))

# 날짜 구간 배경 밴드
date_bands = [
    (0.5,   5.9,  '#E8EEF8', '1주차  MVP 구축'),
    (5.9,  14.3,  '#EAF4EA', '2주차  예측 고도화'),
    (14.3, 23.2,  '#FFF5E0', '3주차  운영 안정화'),
    (23.2, FIG_W - 1.0, '#F5E8FF', '4주차  비즈니스 완성'),
]
for bx0, bx1, bc, blabel in date_bands:
    ax.add_patch(mpatches.Rectangle(
        (bx0, -0.48), bx1 - bx0, 0.96,
        facecolor=bc, edgecolor='none', alpha=0.85, zorder=1
    ))
    ax.text((bx0 + bx1) / 2, -0.92, blabel,
            ha='center', va='center', fontsize=8.5, color='#555', zorder=2,
            fontweight='bold')

# Phase 박스
for i, (pnum, date, cat, title, desc) in enumerate(phases):
    x = 1.5 + i * SPACING
    fc = CAT_COLOR[cat]
    is_key = pnum in MILESTONES
    cy = ABOVE_Y if i % 2 == 0 else BELOW_Y

    # 점선
    if cy > 0:
        ax.plot([x, x], [0.49, cy - BOX_H / 2 + 0.05],
                color='#BBBBBB', ls='--', lw=0.9, zorder=2)
    else:
        ax.plot([x, x], [-0.49, cy + BOX_H / 2 - 0.05],
                color='#BBBBBB', ls='--', lw=0.9, zorder=2)

    # 중앙 점
    dot_c = '#D4A017' if is_key else '#888888'
    ax.add_patch(mpatches.Circle((x, 0), 0.18, color=dot_c, zorder=5))
    ax.add_patch(mpatches.Circle((x, 0), 0.18, fill=False,
                                  ec='#333333', lw=1.3, zorder=6))

    # 마일스톤 글로우 효과
    if is_key:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - BOX_W / 2 - 0.08, cy - BOX_H / 2 - 0.08), BOX_W + 0.16, BOX_H + 0.16,
            boxstyle='round,pad=0.10',
            facecolor='#D4A017', edgecolor='none',
            alpha=0.25, zorder=3
        ))

    # 박스
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0.07',
        facecolor=fc,
        edgecolor='#D4A017' if is_key else '#CCCCCC',
        linewidth=2.8 if is_key else 0.8,
        alpha=0.93, zorder=4
    ))

    # 텍스트 4줄
    ax.text(x, cy + 1.08, pnum,
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color='white', zorder=7)
    ax.text(x, cy + 0.53, date,
            ha='center', va='center', fontsize=7.5,
            color='#EEE', zorder=7, multialignment='center')
    ax.text(x, cy - 0.05, title,
            ha='center', va='center', fontsize=8.8, fontweight='bold',
            color='white', zorder=7, multialignment='center')
    ax.text(x, cy - 0.90, desc,
            ha='center', va='center', fontsize=7.5,
            color='#EEE', zorder=7,
            multialignment='center', linespacing=1.45)

# 범례
leg_y = -7.5
ax.text(1.5, leg_y + 0.85, '카테고리', fontsize=11, fontweight='bold', color='#333')
leg_step = (FIG_W - 3.0) / len(CAT_COLOR)
for j, (cat, col) in enumerate(CAT_COLOR.items()):
    lx = 1.5 + j * leg_step
    ax.add_patch(mpatches.Rectangle(
        (lx, leg_y - 0.28), 1.5, 0.78,
        facecolor=col, edgecolor='#666', lw=0.6, zorder=3, alpha=0.93
    ))
    ax.text(lx + 1.7, leg_y + 0.11, cat,
            fontsize=10.5, va='center', color='#222')

plt.tight_layout(pad=0.4)
out = 'docs/timeline_biz.png'
plt.savefig(out, dpi=130, bbox_inches='tight',
            facecolor='#F8F8F8', edgecolor='none')
print(f'Saved: {out}')
