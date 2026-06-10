"""FinBERT pretrained 감성점수의 '극단성/분산'과 |D+1 수익률| 상관 + 지정학 키워드 분석"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device).eval()
print('labels:', model.config.id2label)

df = pd.read_csv('output/finbert_training_data.csv')
df['abs_ret'] = df['ret_pct'].abs()
titles = df['title'].tolist()

scores = []
bs = 64
with torch.no_grad():
    for i in range(0, len(titles), bs):
        batch = [t[:512] for t in titles[i:i+bs]]
        enc = tok(batch, truncation=True, padding=True, max_length=64, return_tensors='pt').to(device)
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        # id2label: 0 positive, 1 negative, 2 neutral (확인 후 인덱스 사용)
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())  # pos - neg
df['sent'] = scores

# 키워드 플래그
KW = ['attack','strike','war','sanction','crisis','conflict','invasion',
      'drone','missile','explosion','ceasefire','seize','seizure','blockade']
df['has_kw'] = df['title'].str.lower().apply(lambda t: any(k in t for k in KW))

agg = df.groupby('date').agg(
    abs_ret=('abs_ret','first'),
    n=('sent','size'),
    sent_mean=('sent','mean'),
    sent_abs_mean=('sent', lambda x: x.abs().mean()),
    sent_std=('sent','std'),
    kw_ratio=('has_kw','mean'),
).reset_index()
agg['sent_std'] = agg['sent_std'].fillna(0)

print(f'\n전체 {len(agg)}일')
for col in ['sent_abs_mean','sent_std','kw_ratio','sent_mean']:
    r, p = stats.pearsonr(agg[col], agg['abs_ret'])
    rs, ps = stats.spearmanr(agg[col], agg['abs_ret'])
    print(f'{col:>14}: Pearson r={r:+.3f}(p={p:.3f})  Spearman r={rs:+.3f}(p={ps:.3f})')

# n>=2 인 날만
agg2 = agg[agg['n']>=2]
print(f'\nn>=2인 {len(agg2)}일')
for col in ['sent_abs_mean','sent_std','kw_ratio','sent_mean']:
    r, p = stats.pearsonr(agg2[col], agg2['abs_ret'])
    rs, ps = stats.spearmanr(agg2[col], agg2['abs_ret'])
    print(f'{col:>14}: Pearson r={r:+.3f}(p={p:.3f})  Spearman r={rs:+.3f}(p={ps:.3f})')

agg.to_csv('output/sentiment_extremity_daily.csv', index=False)
