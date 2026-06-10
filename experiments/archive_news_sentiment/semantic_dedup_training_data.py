"""날짜별 의미적 중복 헤드라인 제거 (FinBERT [CLS] 임베딩, cosine sim > 0.9 그룹은 1개만 유지)
입력: output/finbert_training_data.csv
출력: output/finbert_training_data_dedup.csv
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.9

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert').to(device).eval()

df = pd.read_csv('output/finbert_training_data.csv')
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

keep_idx = []
for date, g in df.groupby('date', sort=False):
    idx = g.index.tolist()
    if len(idx) == 1:
        keep_idx.extend(idx)
        continue
    sub_embs = embs[idx]
    sim = cosine_similarity(sub_embs)
    kept = []
    for i in range(len(idx)):
        is_dup = any(sim[i, j] > SIM_THRESHOLD for j in kept)
        if not is_dup:
            kept.append(i)
    keep_idx.extend([idx[i] for i in kept])

dedup = df.loc[sorted(keep_idx)].reset_index(drop=True)
dedup.to_csv('output/finbert_training_data_dedup.csv', index=False)

print(f'중복제거 후: {len(dedup)}행, {dedup["date"].nunique()}개 날짜')
print(f'제거된 행: {len(df) - len(dedup)}')

dedup['date_dt'] = pd.to_datetime(dedup['date'])
dedup['year'] = dedup['date_dt'].dt.year
print('\n연도별 행 수 (dedup 후):')
print(dedup.groupby('year').size().to_string())

cnt = dedup.groupby('date').size()
print(f'\n날짜당 최대 기사수: {cnt.max()}')
print('상위 10개 날짜:')
print(cnt.sort_values(ascending=False).head(10).to_string())

print('\nlabel 분포:', dedup['label'].value_counts().to_dict())
