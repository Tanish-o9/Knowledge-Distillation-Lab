"""
Benchmark latency and throughput of saved models (teacher / students).

Examples:
  python benchmark.py --runs runs/teacher runs/plain_distilbert-base-uncased_full_s42 --device both
  python benchmark.py --runs runs/teacher --device cpu --threads 4

Notes:
  - Models run in FP32 so the comparison is fair.
  - Tokenization is done OUTSIDE the timed region (we measure the model only).
  - Latency: batch size 1, warm-up first, then mean and p95 over many runs.
  - Throughput: batch size 32 (samples/sec), median of a few repetitions.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

OUT_CSV = "benchmark_results.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="model directories, e.g. runs/teacher runs/plain_...")
    p.add_argument("--device", default="both", choices=["cuda", "cpu", "both"])
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--n_texts", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lat_warmup", type=int, default=20)
    p.add_argument("--lat_iters", type=int, default=100)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--threads", type=int, default=4, help="CPU threads (fixed for fairness)")
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(model, tok, texts, device, warmup, iters, max_len):
    """Batch size 1. Returns (mean_ms, p95_ms)."""
    times = []
    for i in range(warmup + iters):
        enc = tok(texts[i % len(texts)], return_tensors="pt",
                  truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        sync(device)
        t0 = time.perf_counter()
        model(input_ids=ids, attention_mask=mask)
        sync(device)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(dt)
    return float(np.mean(times)), float(np.percentile(times, 95))


@torch.no_grad()
def measure_throughput(model, tok, texts, device, bs, reps, max_len):
    """Samples/sec at batch size bs (median of reps)."""
    batches = []
    for i in range(0, len(texts) - bs + 1, bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        batches.append((enc["input_ids"].to(device), enc["attention_mask"].to(device)))

    for ids, mask in batches[:3]:          # warm-up
        model(input_ids=ids, attention_mask=mask)

    results = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        for ids, mask in batches:
            model(input_ids=ids, attention_mask=mask)
        sync(device)
        elapsed = time.perf_counter() - t0
        results.append(bs * len(batches) / elapsed)
    return float(np.median(results))


def main():
    args = parse_args()

    if args.device == "both":
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = [args.device]
    if "cuda" in devices and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    texts = load_dataset("dair-ai/emotion", split="test")["text"][:args.n_texts]

    rows = []
    for dev_name in devices:
        device = torch.device(dev_name)
        if dev_name == "cpu":
            torch.set_num_threads(args.threads)
        for path in args.runs:
            run = os.path.basename(path.rstrip("/"))
            print(f"[{dev_name}] benchmarking {run} ...")
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

            lat_mean, lat_p95 = measure_latency(
                model, tok, texts, device, args.lat_warmup, args.lat_iters, args.max_len)
            thr = measure_throughput(
                model, tok, texts, device, args.batch_size, args.reps, args.max_len)

            print(f"    latency {lat_mean:.2f} ms (p95 {lat_p95:.2f}) | "
                  f"throughput {thr:.0f} samples/s")
            rows.append({
                "run": run, "device": dev_name,
                "lat_ms_mean": round(lat_mean, 3), "lat_ms_p95": round(lat_p95, 3),
                "throughput_sps": round(thr, 1), "batch_size": args.batch_size,
                "threads": args.threads if dev_name == "cpu" else "",
            })

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    new = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        keys = set(zip(new["run"], new["device"]))
        old = old[[(r, d) not in keys for r, d in zip(old["run"], old["device"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")


if __name__ == "__main__":
    main()
