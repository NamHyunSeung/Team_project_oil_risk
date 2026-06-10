"""FinBERT 임베딩(맥락 기반) 헤드라인 유사도 -> 의미적 중복도 vs |D+1 수익률| 상관분석"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert').to(device).eval()

df = pd.read_csv('output/finbert_training_data.csv')
df['abs_ret'] = df['ret_pct'].abs()

titles = df['title'].tolist()
embs = []
bs = 64
with torch.no_grad():
    for i in range(0, len(titles), bs):
        batch = titles[i:i+bs]
        enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        out = model(**enc).last_hidden_state[:, 0, :]  # [CLS]
        embs.append(out.cpu().numpy())
embs = np.concatenate(embs, axis=0)
df['emb_idx'] = range(len(df))

results = []
for date, g in df.groupby('date'):
    n = len(g)
    if n < 2:
        continue
    idx = g['emb_idx'].values
    sim = cosine_similarity(embs[idx])
    np.fill_diagonal(sim, 0)
    avg_redundancy = sim.max(axis=1).mean()
    th = 0.9
    n_dup = (sim.max(axis=1) > th).sum()
    results.append({'date': date, 'n': n, 'avg_redundancy': avg_redundancy,
                     'n_dup': n_dup, 'abs_ret': g['abs_ret'].iloc[0]})

res = pd.DataFrame(results)
print('days with n>=2:', len(res))
print('avg_redundancy range:', res['avg_redundancy'].min().round(3), '~', res['avg_redundancy'].max().round(3))
print('corr(avg_redundancy, |ret|):', res['avg_redundancy'].corr(res['abs_ret']).round(3))
print('corr(n_dup, |ret|):', res['n_dup'].corr(res['abs_ret']).round(3))
print('corr(n_dup_ratio, |ret|):', (res['n_dup']/res['n']).corr(res['abs_ret']).round(3))
print('corr(n, |ret|) [n>=2]:', res['n'].corr(res['abs_ret']).round(3))
