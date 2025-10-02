#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch
from datasets import load_dataset

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = float(((y_pred==1)&(y_true==1)).sum())
    tn = float(((y_pred==0)&(y_true==0)).sum())
    fp = float(((y_pred==1)&(y_true==0)).sum())
    fn = float(((y_pred==0)&(y_true==1)).sum())
    acc = (tp+tn)/max(tp+tn+fp+fn,1.0)
    prec = tp/max(tp+fp,1.0); rec = tp/max(tp+fn,1.0)
    f1 = 2*prec*rec/max(prec+rec,1e-12)
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}

def train_distilbert(args):
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
    )

    print("="*70); print(f"DEVICE: {DEVICE} | MODEL: {args.hf_model_name}"); print("="*70)

    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_ds, valid_ds = split["train"], split["test"]
    test_ds = raw["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name, use_fast=True)
    def tok(b): return tokenizer(b["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    valid_tok = valid_ds.map(tok, batched=True, remove_columns=["text"])
    test_tok  = test_ds.map(tok,  batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_name, num_labels=2)

    out_dir = args.output_dir; os.makedirs(out_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        use_mps_device=True,
    )

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
        labels = labels.astype(np.float32)
        m = compute_binary_metrics(labels, probs, threshold=0.5)
        return {k: float(v) for k, v in m.items()}

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    eval_valid = trainer.evaluate(eval_dataset=valid_tok)
    eval_test  = trainer.evaluate(eval_dataset=test_tok)
    print("\n=== Validation ===", eval_valid)
    print("\n=== Test ===", eval_test)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "model_name": args.hf_model_name,
            "task": "sentiment",
            "labels_map": {"neg":0,"pos":1},
            "best_valid": {k: float(v) for k,v in eval_valid.items()},
            "test": {k: float(v) for k,v in eval_test.items()},
            "max_length": args.max_length,
        }, f, indent=2)

    print(f"\nSaved model + tokenizer to: {out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="model_distilbert")
    return p.parse_args()

def set_seeds(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seeds(args.seed)
    train_distilbert(args)

if __name__ == "__main__":
    main()
