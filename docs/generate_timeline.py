import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

CAT_COLOR = {
    '모델':      '#4472C4',
    '피처':      '#3A9A48',
    '인프라':    '#D08820',
    '버그수정':  '#C03030',
    '대시보드':  '#A05820',
    '계정/구독': '#7B3FA0',
}

# (phase label, date, category, title, 2줄 desc)  — 28 → 14 그룹
phases = [
    ('Ph.0~1',   '05-07',       '인프라',    '기획 + 초기 MVP',   'SARIMAX(2,1,1)\n28피처 1419줄'),
    ('Ph.3',     '05-11',       '모델',      'Prophet 제거',      'R²=-4.3\n충격시계열 부적합'),
    ('Ph.5~6',   '05-12~13',    '피처',      'HAR + 뉴스감성',    'overfit_gap 0.44→0.08\nSENTIMENT_MAP 500+'),
    ('Ph.7~8',   '05-14~15',    '모델',      'SARIMAX + FinBERT', 'Optuna 도입\nFinBERT 3종 롤백'),
    ('Ph.9~11',  '05-16~21',    '모델',      '방향성 + 앙상블',   'dir_acc 60%  thr=0.51\n스태킹 MASE=0.7135 미채택'),
    ('Ph.12',    '05-21~25',    '인프라',    '운영 기능 확장',    '로그인 / EXE\n관리자탭 구축'),
    ('Ph.13+15.5','05-25~28',   '모델',      'VAR+ETS + 안정화',  'MASE=1.0 미채택\nconfig/ 분리·버그 정리'),
    ('Ph.14',    '05-29~06-01', '인프라',    'SARIMAX 확정 ★',   'MASE=0.602\n데이터누출 4건 수정'),
    ('Ph.15',    '06-01~02',    '인프라',    '리스크 신호',       'VaR / OVX / hedge\nBlack Swan 감지'),
    ('Ph.16',    '06-02',       '버그수정',  '다중공선성 제거 ★', 'FEATURE 114→105 / HAR 27→22\nHAR hold-out R²  +27%'),
    ('Ph.17~19', '06-04',       '버그수정',  '오차 보정 3종',     '감성±3% · bias±$8 캡\n충격 레짐캡  $9→$2'),
    ('Ph.20~21', '06-04',       '대시보드',  '분석 + 탭 분리',   '$6+ 오차 5건 전수 분석\n가격/뉴스/성과 3탭'),
    ('Ph.22~23', '06-04',       '계정/구독', '플랜 잠금 + 카드',  'free/standard/pro 잠금\n3-플랜 전환 UX'),
    ('Ph.24~26', '06-05',       '계정/구독', '계정 시스템 완성',  '가입·만료 다운그레이드\n업그레이드 인앱 승인'),
]

N       = len(phases)   # 14
SPACING = 2.9
FIG_W   = N * SPACING + 4.5   # ~45.1
FIG_H   = 17.0
BOX_W   = 2.55
BOX_H   = 3.0
ABOVE_Y = 3.9
BELOW_Y = -3.9
MILESTONES = {'Ph.14', 'Ph.16'}

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('#F4F4F4')
ax.set_facecolor('#F4F4F4')
ax.set_xlim(0, FIG_W)
ax.set_ylim(-8.8, 9.0)
ax.axis('off')

# 제목
ax.text(FIG_W / 2, 8.3,
        '유가 리스크 예측 시스템  개발 타임라인  Phase 0 ~ 26',
        ha='center', va='center', fontsize=17, fontweight='bold', color='#111111')
ax.text(FIG_W / 2, 7.55,
        '2026-05-07 ~ 2026-06-05  |  총 28단계  |  ★ = 주요 확정 마일스톤',
        ha='center', va='center', fontsize=10.5, color='#555555')

# 중앙 화살표
ax.annotate('', xy=(FIG_W - 1.3, 0), xytext=(0.8, 0),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=3.0, mutation_scale=22))

# 날짜 구간 배경 밴드 (선택적 가독성 향상)
date_bands = [
    (0.5,   5.9,  '#E8EEF5', '2026-05-07~11'),
    (5.9,  14.3,  '#EAF2E8', '05-12~21'),
    (14.3, 23.2,  '#FFF3E0', '05-21~06-02'),
    (23.2, FIG_W-1.0, '#FFE8E8', '06-04~05'),
]
for bx0, bx1, bc, blabel in date_bands:
    ax.add_patch(mpatches.Rectangle(
        (bx0, -0.45), bx1 - bx0, 0.90,
        facecolor=bc, edgecolor='none', alpha=0.7, zorder=1
    ))
    ax.text((bx0 + bx1) / 2, -0.85, blabel,
            ha='center', va='center', fontsize=8.0, color='#666', zorder=2)

# Phase 박스
for i, (pnum, date, cat, title, desc) in enumerate(phases):
    x = 1.5 + i * SPACING
    fc = CAT_COLOR[cat]
    is_key = pnum in MILESTONES
    cy = ABOVE_Y if i % 2 == 0 else BELOW_Y

    # 점선
    if cy > 0:
        ax.plot([x, x], [0.46, cy - BOX_H / 2 + 0.05],
                color='#BBBBBB', ls='--', lw=0.9, zorder=2)
    else:
        ax.plot([x, x], [-0.46, cy + BOX_H / 2 - 0.05],
                color='#BBBBBB', ls='--', lw=0.9, zorder=2)

    # 중앙 점
    dot_c = '#D4A017' if is_key else '#777777'
    ax.add_patch(mpatches.Circle((x, 0), 0.17, color=dot_c, zorder=5))
    ax.add_patch(mpatches.Circle((x, 0), 0.17, fill=False,
                                  ec='#333333', lw=1.3, zorder=6))

    # 박스
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0.07',
        facecolor=fc,
        edgecolor='#D4A017' if is_key else '#CCCCCC',
        linewidth=3.0 if is_key else 0.8,
        alpha=0.92, zorder=4
    ))

    # 텍스트 4줄
    ax.text(x, cy + 1.08, pnum,
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color='white', zorder=7)
    ax.text(x, cy + 0.53, date,
            ha='center', va='center', fontsize=7.5,
            color='#EEE', zorder=7, multialignment='center')
    ax.text(x, cy - 0.05, title,
            ha='center', va='center', fontsize=9.0, fontweight='bold',
            color='white', zorder=7, multialignment='center')
    ax.text(x, cy - 0.90, desc,
            ha='center', va='center', fontsize=7.5,
            color='#EEE', zorder=7,
            multialignment='center', linespacing=1.45)

# 범례
leg_y = -7.6
ax.text(1.5, leg_y + 0.78, '카테고리', fontsize=10.5, fontweight='bold', color='#333')
leg_step = (FIG_W - 3.0) / len(CAT_COLOR)
for j, (cat, col) in enumerate(CAT_COLOR.items()):
    lx = 1.5 + j * leg_step
    ax.add_patch(mpatches.Rectangle(
        (lx, leg_y - 0.28), 1.5, 0.78,
        facecolor=col, edgecolor='#666', lw=0.6, zorder=3, alpha=0.92
    ))
    ax.text(lx + 1.7, leg_y + 0.11, cat,
            fontsize=10.0, va='center', color='#222')

plt.tight_layout(pad=0.4)
out = 'docs/timeline.png'
plt.savefig(out, dpi=130, bbox_inches='tight',
            facecolor='#F4F4F4', edgecolor='none')
print(f'Saved: {out}')
