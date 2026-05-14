#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABSA (Aspect-Based Sentiment Analysis) trainer with MEDICATION tagging.

- Replaces the target medication in each text with a standard token "<MEDICATION>"
  so the model learns context *around* the target independent of its surface form.
- Trains and evaluates multiple Transformer backbones and picks the best one.
- Saves the best model and provides a 'predict' mode to run on new TRD data later.

Input CSVs (train / val / test) should have at least these columns (configurable):
    text   : the post / sentence
    target : the drug/medication name for this training instance
    label  : sentiment label, one of {negative, neutral, positive} (case-insensitive)
"""
import argparse
import json
import os
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MED_TOKEN = "<MEDICATION>"


def normalize_label(x: str) -> str:
    if x is None:
        raise ValueError("Missing label value.")
    x = str(x).strip().lower()
    mapping = {"neg": "negative", "pos": "positive", "neu": "neutral", "-1": "negative", "0": "neutral", "1": "positive"}
    return mapping.get(x, x)


def read_csv(path: str, text_col: str, target_col: str, label_col: str = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must have columns '{text_col}' and '{target_col}'. Found: {list(df.columns)}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"CSV must have label column '{label_col}'. Found: {list(df.columns)}")
    return df


def compile_drug_regex(drug: str) -> re.Pattern:
    esc = re.escape(drug.strip())
    pattern = rf'(?i)(?<!\w){esc}(?!\w)'
    return re.compile(pattern)


def tag_medication(text: str, target: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    if not isinstance(target, str):
        target = "" if pd.isna(target) else str(target)
    if not target:
        return text
    rx = compile_drug_regex(target)
    return rx.sub(MED_TOKEN, text)


def smart_window_around_med(text: str, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    idx = text.lower().find(MED_TOKEN.lower())
    if idx == -1:
        return text[:max_chars]
    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(text), idx + half)
    return text[start:end]


def prepare_dataframe(df: pd.DataFrame, text_col: str, target_col: str, label_col: str = None) -> pd.DataFrame:
    out = df.copy()
    out["tagged_text"] = [
        smart_window_around_med(tag_medication(t, a))
        for t, a in zip(out[text_col].astype(str), out[target_col].astype(str))
    ]
    if label_col:
        out["label_norm"] = [normalize_label(x) for x in out[label_col]]
        bad = set(out["label_norm"]) - set(LABELS)
        if bad:
            raise ValueError(f"Found unknown labels {bad}. Allowed: {LABELS}")
        out["label_id"] = [LABEL2ID[x] for x in out["label_norm"]]
    return out


def bootstrap_ci(y_true, y_pred, metric_fn, B: int = 2000, seed: int = 13, alpha: float = 0.05):
    """
    Nonparametric percentile bootstrap CI for a scalar metric.
    Returns (lo, hi) for the given alpha (default 95% CI).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(B, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(lo), float(hi)



@dataclass
class SimpleDataset:
    texts: List[str]
    labels: List[int]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx], "tagged_text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


class BatchTokenizeCollator:
    """Robust collator that accepts list[dict] or already-tokenized items."""
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # features is almost always a list[dict]
        if isinstance(features, dict):
            features = [features]

        # If already tokenized (rare in this script), just stack tensors
        if "input_ids" in features[0]:
            keys = [k for k in features[0].keys() if k != "labels"]
            batch = {k: torch.stack([f[k] for f in features]) for k in keys}
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([f["labels"] for f in features])
            return batch

        # Text key may be 'text' (from our Dataset) or 'tagged_text' fallback
        text_key = "text" if "text" in features[0] else ("tagged_text" if "tagged_text" in features[0] else None)
        if text_key is None:
            raise KeyError(f"Expected 'text' in features but got keys: {list(features[0].keys())}")

        texts = [f[text_key] for f in features]
        labels = [f.get("labels") for f in features] if "labels" in features[0] else None

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        if labels is not None:
            enc["labels"] = torch.tensor(labels)
        return enc


def set_all_seeds(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def train_and_eval_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str,
    max_length: int = 256,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
    seed: int = 13,
    use_fast_tokenizer: bool = True,
    ci_bootstrap: int = 2000, 
    ci_seed: int = 42   
) -> Dict[str, Any]:
    print(f"\n=== Training {model_name} ===")
    set_all_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    added = tokenizer.add_tokens([MED_TOKEN], special_tokens=False)
    print(f"Tokenizer added {added} tokens (should be 1 if not present).")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Datasets
    train_ds = SafeSimpleDataset(train_df["tagged_text"].tolist(), train_df["label_id"].tolist())
    val_ds = SafeSimpleDataset(val_df["tagged_text"].tolist(), val_df["label_id"].tolist())
    test_ds = SafeSimpleDataset(test_df["tagged_text"].tolist(), test_df["label_id"].tolist())

    collator = BatchTokenizeCollator(tokenizer=tokenizer, max_length=max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_micro = f1_score(labels, preds, average="micro")
        return {"accuracy": acc, "f1_micro": f1_micro}

    save_path = os.path.join(out_dir, model_name.replace("/", "_"))
    os.makedirs(save_path, exist_ok=True)

    args = TrainingArguments(
        output_dir=save_path,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        report_to=[],
        seed=seed,
        logging_steps=50,
        remove_unused_columns=False,   # keep raw 'text' for custom collator
        dataloader_pin_memory=False    # MPS doesn't support pin_memory
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Evaluate on val & test
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    # Save final reports
    trainer.args.remove_unused_columns = False
    preds_test = np.argmax(trainer.predict(test_ds).predictions, axis=-1)


    y_true = np.asarray(test_df["label_id"].tolist())
    acc_point = accuracy_score(y_true, preds_test)
    f1m_point = f1_score(y_true, preds_test, average="micro")

    test_ci = {}
    if ci_bootstrap and ci_bootstrap > 0:
        acc_lo, acc_hi = bootstrap_ci(
            y_true, preds_test,
            metric_fn=lambda yt, yp: accuracy_score(yt, yp),
            B=ci_bootstrap, seed=ci_seed
        )
        f1_lo, f1_hi = bootstrap_ci(
            y_true, preds_test,
            metric_fn=lambda yt, yp: f1_score(yt, yp, average="micro"),
            B=ci_bootstrap, seed=ci_seed
        )
        test_ci = {
            "accuracy_95ci": [acc_lo, acc_hi],
            "f1_micro_95ci": [f1_lo, f1_hi]
        }





    cls_report = classification_report(test_df["label_id"].tolist(), preds_test, target_names=LABELS, digits=4)
    with open(os.path.join(save_path, "test_classification_report.txt"), "w") as f:
        f.write(
            "# Test set metrics\n"
            f"# N={len(y_true)}  |  accuracy={acc_point:.4f}  |  f1_micro={f1m_point:.4f}\n"
        )
        if test_ci:
            f.write(
                f"# 95% CI: accuracy=({test_ci['accuracy_95ci'][0]:.4f}, {test_ci['accuracy_95ci'][1]:.4f})"
                f", f1_micro=({test_ci['f1_micro_95ci'][0]:.4f}, {test_ci['f1_micro_95ci'][1]:.4f})\n"
            )
        f.write("\n")
        f.write(cls_report)

    if test_ci:
       with open(os.path.join(save_path, "test_ci.json"), "w") as fci:
        json.dump(test_ci, fci, indent=2)


    # Persist the best model
    trainer.save_model(os.path.join(save_path, "best_model"))
    tokenizer.save_pretrained(os.path.join(save_path, "best_model"))

    result = {
        "model_name": model_name,
        "save_path": save_path,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_point": {"accuracy": acc_point, "f1_micro": f1m_point},
        "test_ci": test_ci                                    
    }
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{model_name}] val: {val_metrics} | test: {test_metrics}")
    print(f"Saved best model to {os.path.join(save_path, 'best_model')}")
    return result


def run_experiment(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_dir: str,
    models: List[str],
    text_col: str,
    target_col: str,
    label_col: str,
    max_length: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    use_fast_tokenizer: bool,
    ci_bootstrap: int, 
    ci_seed: int 
):
    os.makedirs(out_dir, exist_ok=True)

    train_raw = read_csv(train_csv, text_col, target_col, label_col)
    val_raw = read_csv(val_csv, text_col, target_col, label_col)
    test_raw = read_csv(test_csv, text_col, target_col, label_col)

    train_df = prepare_dataframe(train_raw, text_col, target_col, label_col)
    val_df = prepare_dataframe(val_raw, text_col, target_col, label_col)
    test_df = prepare_dataframe(test_raw, text_col, target_col, label_col)

    results = []
    for m in models:
        res = train_and_eval_one_model(
            model_name=m,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            out_dir=out_dir,
            max_length=max_length,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            use_fast_tokenizer=use_fast_tokenizer,
            ci_bootstrap=ci_bootstrap, 
            ci_seed=ci_seed  
        )
        results.append(res)

    best = max(results, key=lambda r: r["val_metrics"]["eval_f1_micro"])
    best_dir = os.path.join(best["save_path"], "best_model")
    summary = {
        "best_model": best["model_name"],
        "best_model_dir": best_dir,
        "best_val_f1_micro": best["val_metrics"]["eval_f1_micro"],
        "all_results": [
            {
                "model": r["model_name"],
                "val_f1_micro": r["val_metrics"]["eval_f1_micro"],
                "val_acc": r["val_metrics"]["eval_accuracy"],
                "test_f1_micro": r["test_metrics"]["eval_f1_micro"],
                "test_acc": r["test_metrics"]["eval_accuracy"],
                "dir": os.path.join(r["save_path"], "best_model")
            }
            for r in results
        ]
    }
    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Experiment summary ===")
    print(json.dumps(summary, indent=2))


def predict_file(model_dir: str, input_csv: str, out_csv: str, text_col: str, target_col: str, max_length: int = 256, label_col: str = None, use_fast_tokenizer: bool = True):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    df = read_csv(input_csv, text_col, target_col, label_col=None)
    df = prepare_dataframe(df, text_col, target_col, label_col=None)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer.add_tokens([MED_TOKEN], special_tokens=False)

    collator = BatchTokenizeCollator(tokenizer=tokenizer, max_length=max_length)
    ds = SafeSimpleDataset(df["tagged_text"].tolist(), labels=[0] * len(df))  # dummy labels

    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=collator)
    trainer.args.remove_unused_columns = False
    preds = trainer.predict(ds).predictions
    probs = torch.softmax(torch.tensor(preds), dim=-1).numpy()
    label_ids = probs.argmax(axis=1)
    labels = [ID2LABEL[i] for i in label_ids]

    out = df.copy()
    out["pred_label"] = labels
    out["prob_negative"] = probs[:, LABEL2ID["negative"]]
    out["prob_neutral"] = probs[:, LABEL2ID["neutral"]]
    out["prob_positive"] = probs[:, LABEL2ID["positive"]]

    if label_col and label_col in pd.read_csv(input_csv).columns:
        gt = [normalize_label(x) for x in pd.read_csv(input_csv)[label_col].tolist()]
        mask = [g in LABELS for g in gt]
        if any(mask):
            gt_ids = [LABEL2ID[g] for g, m in zip(gt, mask) if m]
            pr_ids = [LABEL2ID[l] for l, m in zip(labels, mask) if m]
            acc = accuracy_score(gt_ids, pr_ids)
            f1m = f1_score(gt_ids, pr_ids, average="micro")
            print(f"Metrics on rows with valid labels -> accuracy: {acc:.4f}, f1_micro: {f1m:.4f}")

    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="ABSA trainer with <MEDICATION> tagging")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train & evaluate multiple models, pick best")
    p_train.add_argument("--train_csv", required=True, type=str)
    p_train.add_argument("--val_csv", required=True, type=str)
    p_train.add_argument("--test_csv", required=True, type=str)
    p_train.add_argument("--out_dir", required=True, type=str)
    p_train.add_argument("--models", nargs="+", default=["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "microsoft/deberta-v3-base"])
    p_train.add_argument("--text_col", type=str, default="text")
    p_train.add_argument("--target_col", type=str, default="target")
    p_train.add_argument("--label_col", type=str, default="label")
    p_train.add_argument("--max_length", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=13)
    p_train.add_argument("--use_fast_tokenizer", type=lambda x: str(x).lower()!='false', default=True)
    p_train.add_argument("--ci_bootstrap", type=int, default=2000,
                        help="Bootstrap samples for 95% CI on test metrics (0 disables)")
    p_train.add_argument("--ci_seed", type=int, default=42,
                         help="Random seed for CI bootstrap resampling")


    p_pred = sub.add_parser("predict", help="Run inference with a trained model")
    p_pred.add_argument("--model_dir", required=True, type=str)
    p_pred.add_argument("--input_csv", required=True, type=str)
    p_pred.add_argument("--out_csv", required=True, type=str)
    p_pred.add_argument("--text_col", type=str, default="text")
    p_pred.add_argument("--target_col", type=str, default="target")
    p_pred.add_argument("--max_length", type=int, default=256)
    p_pred.add_argument("--label_col", type=str, default=None)
    p_pred.add_argument("--use_fast_tokenizer", type=lambda x: str(x).lower()!='false', default=True)

    args = parser.parse_args()

    if args.mode == "train":
        run_experiment(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            out_dir=args.out_dir,
            models=args.models,
            text_col=args.text_col,
            target_col=args.target_col,
            label_col=args.label_col,
            max_length=args.max_length,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            use_fast_tokenizer=args.use_fast_tokenizer,
            ci_bootstrap=args.ci_bootstrap,
            ci_seed=args.ci_seed
        )
    elif args.mode == "predict":
        predict_file(
            model_dir=args.model_dir,
            input_csv=args.input_csv,
            out_csv=args.out_csv,
            text_col=args.text_col,
            target_col=args.target_col,
            max_length=args.max_length,
            label_col=args.label_col,
            use_fast_tokenizer=args.use_fast_tokenizer
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")



# --- SafeSimpleDataset injected to guarantee 'text' is present for collator ---
class SafeSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = {"text": self.texts[idx], "tagged_text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item

if __name__ == "__main__":
    main()
