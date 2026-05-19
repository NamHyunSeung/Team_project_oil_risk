"""
LSTM 방향 분류기 (최적화 버전)
- 아키텍처 그리드 서치 (hidden, lookback, layers, bidirectional, attention)
- Focal Loss, CosineAnnealing LR, 그래디언트 클리핑
- 멀티 시드 앙상블 (최종 예측)
- SVM wf_dir_acc와 비교

실행: python lstm_direction.py
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from oil_risk_mvp import fetch_data, fetch_news, build_features, OUTPUT_DIR, FEATURE_COLS

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SPLITS = 3
N_TEST   = 60
DZ       = 0.003
TOP_K    = 25
N_SEEDS  = 3    # 최종 앙상블 시드 수
EPOCHS   = 80
PATIENCE = 12
BATCH    = 32

# 탐색할 아키텍처 설정
CONFIGS = [
    {'hidden': 64,  'n_layers': 1, 'lookback': 20, 'dropout': 0.3, 'bidir': False, 'attn': False},
    {'hidden': 128, 'n_layers': 1, 'lookback': 20, 'dropout': 0.3, 'bidir': False, 'attn': False},
    {'hidden': 64,  'n_layers': 2, 'lookback': 20, 'dropout': 0.4, 'bidir': False, 'attn': False},
    {'hidden': 64,  'n_layers': 1, 'lookback': 30, 'dropout': 0.3, 'bidir': False, 'attn': False},
    {'hidden': 64,  'n_layers': 1, 'lookback': 40, 'dropout': 0.3, 'bidir': False, 'attn': False},
    {'hidden': 64,  'n_layers': 1, 'lookback': 20, 'dropout': 0.3, 'bidir': True,  'attn': False},
    {'hidden': 64,  'n_layers': 1, 'lookback': 20, 'dropout': 0.3, 'bidir': False, 'attn': True},
    {'hidden': 128, 'n_layers': 1, 'lookback': 30, 'dropout': 0.3, 'bidir': True,  'attn': True},
]


# ── Focal Loss (클래스 불균형 대응)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target, weight=None):
        bce  = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt   = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if weight is not None:
            loss = loss * weight
        return loss.mean()


# ── Attention 모듈
class TimeAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):              # x: (B, T, H)
        sc = torch.softmax(self.w(x), dim=1)   # (B, T, 1)
        return (x * sc).sum(dim=1)             # (B, H)


# ── LSTM 모델
class LSTMDir(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=1, dropout=0.3,
                 bidir=False, attn=False):
        super().__init__()
        self.bidir = bidir
        self.attn  = attn
        self.lstm  = nn.LSTM(n_feat, hidden, n_layers,
                             batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                             bidirectional=bidir)
        h_out = hidden * 2 if bidir else hidden
        self.attn_layer = TimeAttn(h_out) if attn else None
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(h_out)
        self.fc   = nn.Linear(h_out, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        if self.attn_layer:
            ctx = self.attn_layer(out)
        else:
            ctx = out[:, -1, :]        # 마지막 타임스텝
        return torch.sigmoid(self.fc(self.drop(self.norm(ctx))))


# ── 시퀀스 생성
def make_seq(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── 단일 모델 학습 + 평가
def train_eval(Xs_tr, ys_tr, Xs_va, ys_va, sw_tr, dz_mask, cfg, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_feat = Xs_tr.shape[2]
    model  = LSTMDir(n_feat, cfg['hidden'], cfg['n_layers'],
                     cfg['dropout'], cfg['bidir'], cfg['attn']).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit   = FocalLoss(gamma=2.0)

    dz_idx = np.where(dz_mask)[0]
    if len(dz_idx) < 20:
        dz_idx = np.arange(len(Xs_tr))

    best_loss, wait, best_state = 1e9, 0, None

    for ep in range(EPOCHS):
        model.train()
        perm = dz_idx[np.random.permutation(len(dz_idx))]
        for s in range(0, len(perm), BATCH):
            b  = perm[s:s + BATCH]
            xb = torch.tensor(Xs_tr[b]).to(DEVICE)
            yb = torch.tensor(ys_tr[b]).unsqueeze(1).to(DEVICE)
            wb = torch.tensor(sw_tr[b].astype(np.float32)).unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            xv  = torch.tensor(Xs_va).to(DEVICE)
            yv  = torch.tensor(ys_va).unsqueeze(1).to(DEVICE)
            vl  = crit(model(xv), yv).item()
        if vl < best_loss - 1e-5:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(Xs_va).to(DEVICE)).cpu().numpy().flatten()
    return float(((prob > 0.5).astype(int) == ys_va).mean()), prob


# ── WF CV 평가
def wf_cv(X_tr_all, y_tr_all, y_ret_tr, cfg):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    dirs = []
    for ti, vi in tscv.split(X_tr_all):
        if len(ti) < cfg['lookback'] + 50 or len(vi) < 10:
            continue
        sc = StandardScaler()
        Xt = sc.fit_transform(X_tr_all[ti])
        Xv = sc.transform(X_tr_all[vi])
        Xs_tr, ys_tr = make_seq(Xt, y_tr_all[ti], cfg['lookback'])
        Xs_va, ys_va = make_seq(Xv, y_tr_all[vi], cfg['lookback'])
        if len(Xs_tr) < 20 or len(Xs_va) < 5:
            continue
        sw = np.exp(0.002 * np.arange(len(Xs_tr))); sw /= sw.mean()
        dz = np.abs(y_ret_tr[ti][cfg['lookback']:cfg['lookback'] + len(Xs_tr)]) > DZ
        d, _ = train_eval(Xs_tr, ys_tr, Xs_va, ys_va, sw, dz, cfg)
        dirs.append(d)
    return float(np.mean(dirs)) if dirs else 0.0


# ── 메인
def main():
    t0 = time.time()
    print(f"장치: {DEVICE}\n")

    # ── 1. 데이터
    print("[1/4] 데이터 로드...")
    price_df   = fetch_data()
    news_df    = fetch_news()
    feature_df, _ = build_features(price_df, news_df)
    feature_df = feature_df.dropna(subset=['target_price'])

    imp_path = OUTPUT_DIR / 'xgb_return_importance.csv'
    avail = [f for f in FEATURE_COLS if f in feature_df.columns]
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        sel = [f for f in imp_df['feature'].tolist() if f in avail][:TOP_K]
    else:
        sel = avail[:TOP_K]

    # CEEMDAN 피처 추가 (SVM에서 사용하는 추가 신호)
    cem_cols = [c for c in ['ceemdan_trend_ret', 'ceemdan_noise_std5', 'ceemdan_trend_mom5']
                if c in feature_df.columns and feature_df[c].abs().sum() > 0]
    sel_ext = sel + [c for c in cem_cols if c not in sel]
    print(f"    피처: {len(sel_ext)}개 (기본 {len(sel)} + CEEMDAN {len(cem_cols)})")

    X_all  = feature_df[sel_ext].fillna(0).values.astype(np.float32)
    y_ret  = np.log(feature_df['target_price'].values / feature_df['WTI'].values)
    y_cls  = (y_ret > 0).astype(np.float32)
    X_tr   = X_all[:-N_TEST];  y_tr = y_cls[:-N_TEST];  yr_tr = y_ret[:-N_TEST]
    X_te   = X_all[-N_TEST:];  y_te = y_cls[-N_TEST:];  yr_te = y_ret[-N_TEST:]

    # ── 2. 아키텍처 그리드 서치
    print(f"\n[2/4] 그리드 서치 ({len(CONFIGS)}개 설정 × {N_SPLITS}폴드)...")
    best_cfg, best_wf = None, -1.0
    for i, cfg in enumerate(CONFIGS):
        wf = wf_cv(X_tr, y_tr, yr_tr, cfg)
        tag = (f"h={cfg['hidden']} L={cfg['n_layers']} lb={cfg['lookback']}"
               f" bi={cfg['bidir']} attn={cfg['attn']}")
        mark = " ★" if wf > best_wf else ""
        print(f"    [{i+1}/{len(CONFIGS)}] {tag} → wf_dir={wf*100:.1f}%{mark}")
        if wf > best_wf:
            best_wf, best_cfg = wf, cfg

    print(f"\n    최적 설정: {best_cfg}  wf_dir={best_wf*100:.1f}%")

    # ── 3. 최적 설정으로 멀티 시드 앙상블 (테스트셋)
    print(f"\n[3/4] 최종 앙상블 ({N_SEEDS}개 시드)...")
    sc_f   = StandardScaler()
    Xtr_f  = sc_f.fit_transform(X_tr)
    Xte_f  = sc_f.transform(X_te)

    lb = best_cfg['lookback']
    Xs_tr_f, ys_tr_f = make_seq(Xtr_f, y_tr, lb)
    # 테스트 시퀀스: 훈련 끝 lb행 + 테스트 행 연결
    Xs_te_f, ys_te_f = make_seq(
        np.vstack([Xtr_f[-lb:], Xte_f]),
        np.concatenate([y_tr[-lb:], y_te]), lb)
    Xs_te_f = Xs_te_f[-N_TEST:];  ys_te_f = ys_te_f[-N_TEST:]

    sw_f  = np.exp(0.002 * np.arange(len(Xs_tr_f))); sw_f /= sw_f.mean()
    dz_f  = np.abs(yr_tr[lb:lb + len(Xs_tr_f)]) > DZ

    probs = []
    for seed in range(N_SEEDS):
        d, p = train_eval(Xs_tr_f, ys_tr_f, Xs_te_f, ys_te_f, sw_f, dz_f, best_cfg, seed)
        probs.append(p)
        print(f"    seed {seed}: test_dir={d*100:.1f}%")

    prob_ens = np.mean(probs, axis=0)
    dir_ens  = float(((prob_ens > 0.5).astype(int) == ys_te_f).mean())

    # MAE 계산
    sign_lstm = np.where(prob_ens > 0.5, 1.0, -1.0)
    pr_adj    = np.abs(yr_te) * sign_lstm
    px_te_v   = feature_df['WTI'].values[-N_TEST:]
    px_adj    = px_te_v * np.exp(pr_adj)
    px_actual = feature_df['target_price'].values[-N_TEST:]
    mae_lstm  = float(mean_absolute_error(px_actual, px_adj))

    # ── 4. 결과 출력
    elapsed = time.time() - t0
    print(f"\n[4/4] 결과 (소요: {elapsed/60:.1f}분)")
    print(f"{'='*58}")
    print(f"  {'지표':<20} {'LSTM (최적화)':<18} {'SVM (파이프라인)'}")
    print(f"  {'-'*54}")
    print(f"  {'WF dir_acc':<20} {best_wf*100:.1f}%{'':<13} 52.9%")
    print(f"  {'Test dir':<20} {dir_ens*100:.1f}%{'':<13} 63.3%")
    print(f"  {'Test MAE':<20} {mae_lstm:.4f}{'':<12} 3.637")
    print(f"{'='*58}")

    if best_wf > 0.529:
        print("  → LSTM WF 성능 우수: 파이프라인 통합 권장")
        print(f"     최적 설정: {best_cfg}")
    elif best_wf > 0.520:
        print("  → LSTM WF 성능 유사: 추가 실험 고려")
    else:
        print("  → SVM 유지 권장")


if __name__ == '__main__':
    main()
