"""
파인튜닝 FinBERT 백테스트 평가
- 뉴스: output/guardian_news_cache.csv (백테스트 90 영업일 구간)
- 모델: output/finbert_oil/ (LoRA fine-tuned)
- 비교: 기존 pretrained finbert_score vs fine-tuned up_prob
- 대시보드 무관 독립 스크립트
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
NEWS_CACHE  = Path("output/guardian_news_cache.csv")
MODEL_DIR   = Path("output/finbert_oil")
BASE_MODEL  = "ProsusAI/finbert"
N_TEST      = 90   # 영업일
BATCH_SIZE  = 32
MAX_LEN     = 128

OIL_KW = [
    'crude oil','brent crude','brent','wti','opec','barrel','petroleum',
    'refinery','shale','tanker','crude inventory','oil inventory',
    'oil price','oil supply','oil demand','oil production','oil tanker',
    'oil facility','oil sanctions','oil embargo','hormuz','lng','natural gas',
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, str(MODEL_DIR))
    model.eval()
    if device == "cuda":
        model = model.cuda()
    print("Model loaded.")
    return tokenizer, model


# ── 배치 추론 → up 확률 ────────────────────────────────────────────────────────
def predict_batch(texts, tokenizer, model):
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            [t[:512] for t in batch],
            truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt',
        )
        if device == "cuda":
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
    return scores


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    # 뉴스 로드 + OIL_KW 필터
    news = pd.read_csv(NEWS_CACHE)
    news['date'] = pd.to_datetime(news['date'])
    news = news[news['title'].str.lower().apply(lambda t: any(kw in str(t) for kw in OIL_KW))]
    print(f"Total news after OIL_KW filter: {len(news)}")

    # 백테스트 기간: 최근 N_TEST 영업일
    brent = yf.download('BZ=F', start='2024-01-01', progress=False)['Close'].squeeze()
    brent.index = pd.to_datetime(brent.index).tz_localize(None)
    bt_dates = brent.index[-N_TEST:]
    bt_start, bt_end = bt_dates[0], bt_dates[-1]
    print(f"\nBacktest period: {bt_start.date()} ~ {bt_end.date()} ({N_TEST} days)")

    # 백테스트 기간 뉴스 필터
    bt_news = news[(news['date'] >= bt_start) & (news['date'] <= bt_end)].copy()
    bt_news = bt_news.reset_index(drop=True)
    print(f"News in backtest period: {len(bt_news)}")

    if len(bt_news) == 0:
        print("No news in backtest period.")
        return

    # 파인튜닝 모델 추론
    tokenizer, model = load_model()
    print(f"\nRunning fine-tuned inference on {len(bt_news)} articles...")
    bt_news['up_prob'] = predict_batch(bt_news['title'].tolist(), tokenizer, model)
    bt_news['ft_score'] = (bt_news['up_prob'] - 0.5) * 2  # -1~+1

    # 날짜별 집계
    daily = bt_news.groupby('date').agg(
        n_articles   = ('title', 'count'),
        ft_score_avg = ('ft_score', 'mean'),
        up_prob_avg  = ('up_prob', 'mean'),
        pretrained_avg = ('finbert_score', 'mean'),
    ).reset_index()

    # Brent D+1 수익률 병합
    ret_next = brent.pct_change().shift(-1) * 100
    price_df = pd.DataFrame({'price': brent, 'ret_next': ret_next})
    price_df.index.name = 'date'
    price_df = price_df.reset_index()

    merged = daily.merge(price_df, on='date', how='inner')
    merged = merged.dropna(subset=['ret_next'])

    # 방향 일치 여부
    merged['ft_direction']  = merged['ft_score_avg'].apply(lambda x: 'up' if x > 0 else 'down')
    merged['actual_dir']    = merged['ret_next'].apply(lambda x: 'up' if x > 0 else 'down')
    merged['ft_correct']    = merged['ft_direction'] == merged['actual_dir']

    merged['pre_direction'] = merged['pretrained_avg'].apply(lambda x: 'up' if x > 0 else 'down')
    merged['pre_correct']   = merged['pre_direction'] == merged['actual_dir']

    acc_ft  = merged['ft_correct'].mean()
    acc_pre = merged['pre_correct'].mean()
    n_days  = len(merged)

    print(f"\n{'='*55}")
    print(f"Backtest period: {bt_start.date()} ~ {bt_end.date()}")
    print(f"Days with news + price: {n_days}")
    print(f"{'─'*55}")
    print(f"Fine-tuned  direction accuracy : {acc_ft:.1%}  ({merged['ft_correct'].sum()}/{n_days})")
    print(f"Pretrained  direction accuracy : {acc_pre:.1%}  ({merged['pre_correct'].sum()}/{n_days})")
    print(f"{'─'*55}")

    # 월별 breakdown
    merged['month'] = merged['date'].dt.to_period('M')
    monthly = merged.groupby('month').agg(
        days        = ('ft_correct', 'count'),
        ft_acc      = ('ft_correct', 'mean'),
        pre_acc     = ('pre_correct', 'mean'),
        avg_up_prob = ('up_prob_avg', 'mean'),
    )
    print("\n[ Monthly breakdown ]")
    print(f"{'Month':<10} {'Days':>5} {'FT Acc':>8} {'Pre Acc':>8} {'Avg UpProb':>11}")
    print("─" * 48)
    for m, row in monthly.iterrows():
        print(f"{str(m):<10} {int(row['days']):>5} {row['ft_acc']:>8.1%} {row['pre_acc']:>8.1%} {row['avg_up_prob']:>11.3f}")

    # 고확신 예측 (|ft_score| > 0.3) 정확도
    high_conf = merged[merged['ft_score_avg'].abs() > 0.3]
    if len(high_conf) > 0:
        print(f"\n[ High confidence (|ft_score|>0.3): {len(high_conf)}일 ]")
        print(f"  FT accuracy: {high_conf['ft_correct'].mean():.1%}")

    # 결과 저장
    out_path = Path("output/eval_finetuned_backtest.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nDetailed results saved -> {out_path}")


if __name__ == '__main__':
    main()
