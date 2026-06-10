"""개선안: 일별 단일 감성값 -> 5/10/20일 이동평균 감성지수로 스무딩
+ 예측 호라이즌을 D+1 외 D+3/D+5/D+10/주간으로 확장하여 상관관계 재검증
입력: output/finbert_training_data_2010.csv (헤드라인, 2010-2026)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

# ── 1. 헤드라인 감성 점수 (pretrained FinBERT) ─────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device).eval()

df = pd.read_csv('output/finbert_training_data_2010.csv')
titles = df['title'].tolist()
pos, neg = [], []
bs = 64
with torch.no_grad():
    for i in range(0, len(titles), bs):
        batch = [t[:512] for t in titles[i:i+bs]]
        enc = tok(batch, truncation=True, padding=True, max_length=64, return_tensors='pt').to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        pos.extend(probs[:, 0]); neg.extend(probs[:, 1])
df['net_sent'] = np.array(pos) - np.array(neg)

daily = df.groupby('date')['net_sent'].mean().reset_index()
daily['date'] = pd.to_datetime(daily['date'])
print(f'뉴스 있는 날: {len(daily)}일')

# ── 2. Brent 가격 (전체 영업일 캘린더) ─────────────────────────────────────
brent = yf.download('BZ=F', start='2009-12-01', end='2026-06-10', progress=False)['Close']
brent = brent.dropna()
if isinstance(brent, pd.DataFrame):
    brent = brent.iloc[:, 0]
brent.index = pd.to_datetime(brent.index).tz_localize(None)
print(f'Brent 영업일: {len(brent)}일 ({brent.index.min().date()} ~ {brent.index.max().date()})')

# ── 3. 영업일 캘린더에 감성 reindex + ffill(limit=5) ───────────────────────
cal = pd.DataFrame(index=brent.index)
cal = cal.join(daily.set_index('date'))
cal['net_sent_ff'] = cal['net_sent'].ffill(limit=5)
cal['price'] = brent

for w in [5, 10, 20]:
    cal[f'sent_ma{w}'] = cal['net_sent_ff'].rolling(w, min_periods=max(2, w//2)).mean()

# ── 4. 다양한 호라이즌 미래수익률 ──────────────────────────────────────────
for h in [1, 3, 5, 10]:
    cal[f'fwd_ret_{h}'] = cal['price'].shift(-h) / cal['price'] - 1

cal = cal.dropna(subset=['sent_ma20'])
print(f'\n분석 가능 행: {len(cal)}')

# ── 5. 상관관계 매트릭스: 스무딩 윈도우 x 호라이즌 ─────────────────────────
print('\n[스무딩 감성지수 vs 미래수익률] Pearson r (p-value)')
print(f'{"":>10}' + ''.join(f'{"D+"+str(h):>16}' for h in [1, 3, 5, 10]))
for w in [5, 10, 20]:
    row = f'sent_ma{w:<3}'
    for h in [1, 3, 5, 10]:
        sub = cal[[f'sent_ma{w}', f'fwd_ret_{h}']].dropna()
        r, p = stats.pearsonr(sub[f'sent_ma{w}'], sub[f'fwd_ret_{h}'])
        row += f'{f"{r:+.3f}({p:.3f})":>16}'
    print(row)

# ── 6. 가장 유망한 조합으로 방향예측(로지스틱회귀, 시계열분할) ─────────────
best = None
for w in [5, 10, 20]:
    for h in [1, 3, 5, 10]:
        sub = cal[[f'sent_ma{w}', f'fwd_ret_{h}']].dropna()
        r, p = stats.pearsonr(sub[f'sent_ma{w}'], sub[f'fwd_ret_{h}'])
        if best is None or abs(r) > abs(best[2]):
            best = (w, h, r, p)

w, h, r, p = best
print(f'\n가장 강한 상관: sent_ma{w} vs D+{h} (r={r:+.3f}, p={p:.3f})')

sub = cal[[f'sent_ma{w}', f'fwd_ret_{h}']].dropna().copy()
sub['dir'] = (sub[f'fwd_ret_{h}'] > 0).astype(int)
n = len(sub)
train_end, valid_end = int(n*0.8), int(n*0.9)
train, test = sub.iloc[:train_end], sub.iloc[valid_end:]

clf = LogisticRegression(max_iter=1000)
clf.fit(train[[f'sent_ma{w}']], train['dir'])
pred = clf.predict(test[[f'sent_ma{w}']])
proba = clf.predict_proba(test[[f'sent_ma{w}']])[:, 1]

print(f'\n로지스틱회귀 sent_ma{w} -> D+{h} 방향 (train={len(train)}, test={len(test)})')
print(f'  Test Accuracy: {accuracy_score(test["dir"], pred):.4f}')
print(f'  Test AUC: {roc_auc_score(test["dir"], proba):.4f}')
print(f'  Majority (test): {test["dir"].value_counts(normalize=True).max():.4f}')
