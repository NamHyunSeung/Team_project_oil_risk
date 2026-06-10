"""2010-2026 데이터셋: 의미적 중복제거(cosine>0.9) + 날짜당 cap=10
입력: output/finbert_training_data_2010.csv
출력: output/finbert_training_data_2010_final.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.9
CAP = 10
SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert').to(device).eval()

df = pd.read_csv('output/finbert_training_data_2010.csv')
print(f'원본: {len(df)}행, {df["date"].nunique()}개 날짜')

titles = df['title'].tolist()
embs = []
bs = 64
with torch.no_grad():
    for i in range(0, len(titles), bs):
        batch = titles[i:i+bs]
        enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        out = model(**enc).last_hidden_state[:, 0, :]
        embs.append(out.cpu().numpy())
embs = np.concatenate(embs, axis=0)

# ── 1. 의미적 중복제거 ────────────────────────────────────────────────────
keep_idx = []
for date, g in df.groupby('date', sort=False):
    idx = g.index.tolist()
    if len(idx) == 1:
        keep_idx.extend(idx)
        continue
    sim = cosine_similarity(embs[idx])
    kept = []
    for i in range(len(idx)):
        if not any(sim[i, j] > SIM_THRESHOLD for j in kept):
            kept.append(i)
    keep_idx.extend([idx[i] for i in kept])

dedup = df.loc[sorted(keep_idx)].reset_index(drop=True)
print(f'중복제거 후: {len(dedup)}행 (제거 {len(df)-len(dedup)})')

# ── 2. 날짜당 cap ─────────────────────────────────────────────────────────
parts = []
for date, g in dedup.groupby('date', sort=False):
    if len(g) > CAP:
        g = g.sample(n=CAP, random_state=SEED)
    parts.append(g)
final = pd.concat(parts, ignore_index=True)
final = final.sort_values('date').reset_index(drop=True)

OUT = 'output/finbert_training_data_2010_final.csv'
final.to_csv(OUT, index=False)
print(f'cap 적용 후: {len(final)}행 -> {OUT}')

final['date_dt'] = pd.to_datetime(final['date'])
final['year'] = final['date_dt'].dt.year
print(f'\n연도별:\n{final.groupby("year").size().to_string()}')
print(f'\n날짜당 최대 기사수: {final.groupby("date").size().max()}')
print(f'label 분포: {final["label"].value_counts().to_dict()}')
