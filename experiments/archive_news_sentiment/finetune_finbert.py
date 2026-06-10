"""
FinBERT LoRA Fine-tuning for Oil Price Direction
- Base: ProsusAI/finbert
- LoRA: rank=8, alpha=16, target=[query, value]
- Binary classification: up(1) / down(0)
- Split: train 2018-2025 (80%), valid 2018-2025 (20%), test 2026
- Output: output/finbert_oil/
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
LABELED_FILE = Path("output/finbert_training_data_2010_final.csv")
MODEL_DIR    = Path("output/finbert_oil")
BASE_MODEL   = "ProsusAI/finbert"
MAX_LEN      = 256
BATCH_SIZE   = 16
MAX_EPOCHS   = 20
LR           = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTH = 0.1
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()} | {'labels': self.labels[idx]}


# ── Load & split ──────────────────────────────────────────────────────────────
def load_splits():
    df = pd.read_csv(LABELED_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df['label_id'] = (df['label'] == 'up').astype(int)
    df = df.sort_values('date').reset_index(drop=True)

    # 시계열 순서 80/10/10 분할 (미래 누출 방지)
    n = len(df)
    train_end = int(n * 0.80)
    valid_end = int(n * 0.90)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test  = df.iloc[valid_end:]

    print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    print(f"  Train period: {train['date'].min().date()} ~ {train['date'].max().date()}")
    print(f"  Valid period: {valid['date'].min().date()} ~ {valid['date'].max().date()}")
    print(f"  Test  period: {test['date'].min().date()} ~ {test['date'].max().date()}")
    print(f"  Train up ratio: {train['label_id'].mean():.3f}")

    return train, valid, test


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(tokenizer):
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1':       f1_score(labels, preds, average='binary', zero_division=0),
        'auc':      roc_auc_score(labels, probs),
    }


# ── Label smoothing loss ───────────────────────────────────────────────────────
class SmoothedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        n_cls = logits.size(-1)
        smooth = LABEL_SMOOTH / n_cls
        one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        soft_labels = one_hot * (1 - LABEL_SMOOTH) + smooth
        loss = -(soft_labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    train_df, valid_df, test_df = load_splits()

    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = NewsDataset(train_df['title'], train_df['label_id'], tokenizer)
    valid_ds = NewsDataset(valid_df['title'], valid_df['label_id'], tokenizer)
    test_ds  = NewsDataset(test_df['title'],  test_df['label_id'],  tokenizer)

    model = build_model(tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SmoothedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n=== Training ===")
    trainer.train()

    # ── Evaluate on test (2026) ───────────────────────────────────────────────
    print("\n=== Test set evaluation (2026) ===")
    result = trainer.evaluate(test_ds)
    print(f"  Accuracy : {result.get('eval_accuracy', 0):.4f}")
    print(f"  F1       : {result.get('eval_f1', 0):.4f}")
    print(f"  AUC      : {result.get('eval_auc', 0):.4f}")

    # ── Baseline: majority class on test ─────────────────────────────────────
    majority = test_df['label_id'].mode()[0]
    majority_acc = (test_df['label_id'] == majority).mean()
    print(f"\nMajority baseline accuracy: {majority_acc:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModel saved -> {MODEL_DIR}")

    # ── Per-year breakdown (valid+test) ────────────────────────────────────────
    print("\n=== Prediction detail on test ===")
    preds_out = trainer.predict(test_ds)
    pred_labels = np.argmax(preds_out.predictions, axis=1)
    test_df = test_df.copy()
    test_df['pred'] = pred_labels
    test_df['correct'] = test_df['pred'] == test_df['label_id']
    monthly = test_df.set_index('date').resample('ME')['correct'].mean()
    print(monthly.round(3).to_string())


if __name__ == '__main__':
    train()
