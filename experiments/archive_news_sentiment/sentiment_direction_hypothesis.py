"""가설: 감성(FinBERT pretrained) -> 기대(일별 집계) -> D+1 가격방향
- 파인튜닝 없이 ProsusAI/finbert의 pretrained 감성(pos/neg/neutral)을 사용
- 일별 평균 감성과 D+1 수익률(부호 포함)의 상관관계 + 방향 예측력 검증
입력: output/finbert_training_data_2010.csv (2010-2026, 2464행, 1033일)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device).eval()
print('id2label:', model.config.id2label)

df = pd.read_csv('output/finbert_training_data_2010.csv')
df['label_id'] = (df['label'] == 'up').astype(int)

titles = df['title'].tolist()
pos, neg, neu = [], [], []
bs = 64
with torch.no_grad():
    for i in range(0, len(titles), bs):
        batch = [t[:512] for t in titles[i:i+bs]]
        enc = tok(batch, truncation=True, padding=True, max_length=64, return_tensors='pt').to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        pos.extend(probs[:, 0]); neg.extend(probs[:, 1]); neu.extend(probs[:, 2])
df['pos'] = pos; df['neg'] = neg; df['neu'] = neu
df['net_sent'] = df['pos'] - df['neg']

# ── 일별 집계 ────────────────────────────────────────────────────────────
agg = df.groupby('date').agg(
    label_id=('label_id', 'first'),
    ret_pct=('ret_pct', 'first'),
    net_sent=('net_sent', 'mean'),
    pos=('pos', 'mean'), neg=('neg', 'mean'), neu=('neu', 'mean'),
    n=('net_sent', 'size'),
).reset_index().sort_values('date').reset_index(drop=True)

print(f'\n총 {len(agg)}일')

# ── 상관관계: 일별 평균 감성 vs D+1 수익률(부호 포함) ───────────────────────
for col in ['net_sent', 'pos', 'neg']:
    r, p = stats.pearsonr(agg[col], agg['ret_pct'])
    rs, ps = stats.spearmanr(agg[col], agg['ret_pct'])
    print(f'{col:>10} vs ret_pct: Pearson r={r:+.3f}(p={p:.3f})  Spearman r={rs:+.3f}(p={ps:.3f})')

# ── 단순 부호일치: net_sent>0 -> up 예측 ───────────────────────────────────
agg['pred_sign'] = (agg['net_sent'] > 0).astype(int)
acc_sign = (agg['pred_sign'] == agg['label_id']).mean()
print(f'\n단순 부호일치 정확도 (net_sent>0 -> up): {acc_sign:.4f}')
print(f'Majority baseline (전체): {agg["label_id"].value_counts(normalize=True).max():.4f}')

# ── 시계열 분할 로지스틱회귀 ─────────────────────────────────────────────
n = len(agg)
train_end, valid_end = int(n * 0.8), int(n * 0.9)
train, test = agg.iloc[:train_end], agg.iloc[valid_end:]

feat_cols = ['net_sent', 'pos', 'neg', 'neu', 'n']
clf = LogisticRegression(max_iter=1000)
clf.fit(train[feat_cols], train['label_id'])
pred = clf.predict(test[feat_cols])
proba = clf.predict_proba(test[feat_cols])[:, 1]

print(f'\n로지스틱회귀 (train={len(train)}, test={len(test)})')
print(f'  Test period: {test["date"].min()} ~ {test["date"].max()}')
print(f'  Test Accuracy: {accuracy_score(test["label_id"], pred):.4f}')
print(f'  Test AUC: {roc_auc_score(test["label_id"], proba):.4f}')
print(f'  Majority (test): {test["label_id"].value_counts(normalize=True).max():.4f}')
