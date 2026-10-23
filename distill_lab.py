"""
Knowledge Distillation Lab (Stages 1-4)

Modes:
  teacher          -> fine-tune the big model (bert-base-uncased) on labels
  student_plain    -> fine-tune the small model (distilbert) on labels only (BASELINE)
  student_distill  -> train the small model with teacher guidance

Examples:
  python distill_lab.py --mode teacher
  python distill_lab.py --mode student_plain
  python distill_lab.py --mode student_distill --temperature 4 --alpha 0.5
  python distill_lab.py --mode teacher --train_limit 500 --epochs 1   # quick smoke test
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

NUM_LABELS = 6
RESULTS_CSV = "distill_results.csv"
# dair-ai/emotion label order
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["teacher", "student_plain", "student_distill"])
    p.add_argument("--teacher_name", default="bert-base-uncased")
    p.add_argument("--student_name", default="distilbert-base-uncased")
    p.add_argument("--teacher_dir", default="runs/teacher",
                   help="saved teacher (used in student_distill mode)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of label loss; (1-alpha) goes to teacher loss")
    p.add_argument("--train_limit", type=int, default=0,
                   help="use only N train examples (0 = all)")
    p.add_argument("--patience", type=int, default=0,
                   help="early stopping patience (0 = disabled)")
    p.add_argument("--run_name", default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# stage 1: data
def build_data(tokenizer, max_len, batch_size, train_limit, seed):
    ds = load_dataset("dair-ai/emotion")  # train / validation / test
    if train_limit:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(train_limit))

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)  # dynamic padding per batch
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                          collate_fn=collator, generator=g)
    val_dl = DataLoader(ds["validation"], batch_size=64, collate_fn=collator)
    test_dl = DataLoader(ds["test"], batch_size=64, collate_fn=collator)
    return train_dl, val_dl, test_dl


def unpack(batch, device):
    """Only input_ids + attention_mask go to the model.
    (BERT has token_type_ids, DistilBERT doesn't. Dropping it keeps both consistent.)"""
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    return ids, mask, labels


# helpers
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1024 / 1024


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        ids, mask, labels = unpack(batch, device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        preds += logits.argmax(-1).cpu().tolist()
        gold += labels.cpu().tolist()
    return accuracy_score(gold, preds), f1_score(gold, preds, average="macro")


# stage 4: loss
def distill_loss(s_logits, t_logits, labels, T, alpha):
    """alpha * CE(student, labels) + (1-alpha) * T^2 * KL(teacher_soft || student_soft)"""
    ce = F.cross_entropy(s_logits, labels)
    kd = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),   # student: log-probs
        F.softmax(t_logits / T, dim=-1),       # teacher: probs
        reduction="batchmean",
    ) * (T * T)                                # T^2 scaling
    return alpha * ce + (1 - alpha) * kd, ce, kd


# training loop
def train(model, train_dl, val_dl, args, device, out_dir, tokenizer, teacher=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, start = 0.0, time.time()
    no_improve = 0  # early stopping counter
    for epoch in range(args.epochs):
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_dl:
            ids, mask, labels = unpack(batch, device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                s_logits = model(input_ids=ids, attention_mask=mask).logits
                if teacher is None:
                    loss = F.cross_entropy(s_logits, labels)
                else:
                    with torch.no_grad():  # teacher is frozen
                        t_logits = teacher(input_ids=ids, attention_mask=mask).logits
                    loss, _, _ = distill_loss(
                        s_logits.float(), t_logits.float(), labels,
                        args.temperature, args.alpha)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            n += 1

        val_acc, val_f1 = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}/{args.epochs} | loss {run_loss / n:.4f} "
              f"| val acc {val_acc:.4f} | val f1 {val_f1:.4f}")

        if val_acc > best_val:  # keep best epoch by VALIDATION accuracy
            best_val = val_acc
            no_improve = 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"Early stopping triggered (patience={args.patience}).")
                break

    return best_val, time.time() - start


# main
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.teacher_name if args.mode == "teacher" else args.student_name
    short = model_name.split("/")[-1]
    n_tag = f"n{args.train_limit}" if args.train_limit else "full"
    if args.mode == "teacher":
        default_name = "teacher"
    elif args.mode == "student_plain":
        default_name = f"plain_{short}_{n_tag}_s{args.seed}"
    else:
        default_name = (f"distill_{short}_{n_tag}_"
                        f"T{args.temperature:g}_a{args.alpha:g}_s{args.seed}")
    run_name = args.run_name or default_name
    if args.mode == "teacher":
        out_dir = args.teacher_dir
    else:
        out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run: {run_name} | model: {model_name} | device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dl, val_dl, test_dl = build_data(
        tokenizer, args.max_len, args.batch_size, args.train_limit, args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS).to(device)

    teacher = None
    if args.mode == "student_distill":
        teacher_tok = AutoTokenizer.from_pretrained(args.teacher_dir)
        assert teacher_tok.get_vocab() == tokenizer.get_vocab(), \
            "Teacher and student tokenizers differ - logits would not match!"
        teacher = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_dir).to(device)
        teacher.eval()                      # no dropout
        for p in teacher.parameters():
            p.requires_grad = False         # frozen
        t_acc, _ = evaluate(teacher, val_dl, device)
        print(f"Teacher val acc (sanity check): {t_acc:.4f}")

    print(f"Trainable params: {count_params(model) / 1e6:.1f}M")
    best_val, train_time = train(
        model, train_dl, val_dl, args, device, out_dir, tokenizer, teacher)

    # Final: reload BEST checkpoint, evaluate on TEST exactly once
    best = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    test_acc, test_f1 = evaluate(best, test_dl, device)
    print(f"\nTEST acc {test_acc:.4f} | TEST macro-F1 {test_f1:.4f}")

    # per-class report
    from sklearn.metrics import classification_report
    best.eval()
    all_preds, all_gold = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids, mask, labels = unpack(batch, device)
            logits = best(input_ids=ids, attention_mask=mask).logits
            all_preds += logits.argmax(-1).cpu().tolist()
            all_gold += labels.cpu().tolist()
    print(classification_report(all_gold, all_preds, target_names=EMOTION_LABELS, digits=4))

    row = {
        "run": run_name, "mode": args.mode, "model": model_name,
        "params_M": round(count_params(best) / 1e6, 2),
        "size_mb": round(dir_size_mb(out_dir), 1),
        "train_limit": args.train_limit if args.train_limit else "full",
        "temperature": args.temperature if args.mode == "student_distill" else "",
        "alpha": args.alpha if args.mode == "student_distill" else "",
        "seed": args.seed, "epochs": args.epochs,
        "best_val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
        "train_time_s": round(train_time, 1),
    }
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"Saved results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
