# Knowledge Distillation Lab

Compressing a BERT-base teacher into smaller student models for 6-class emotion classification. The project compares a **teacher**, a **plain student** (trained on labels only) and a **distilled student** (trained with teacher guidance) on accuracy, size, latency and throughput, with temperature and alpha ablations.

## What this project answers

1. How much accuracy do we lose when we shrink the model?
2. Does distillation actually beat training the small model normally? (the plain-student baseline makes this a fair test)
3. How much faster and smaller is the student, on GPU and on CPU?
4. Which temperature (T) and alpha work best, and does distillation help more with less data or a smaller student?

## Method

```
Dataset -> [1] Teacher (BERT-base, labels)
        -> [2] Plain student (DistilBERT, labels only)           <- baseline
        -> [3] Distilled student (labels + frozen teacher)
                loss = a * CE(student, labels)
                     + (1 - a) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
        -> Evaluate: accuracy, macro-F1, params, size, latency, throughput
```

- Teacher is frozen during distillation (`eval()`, `requires_grad=False`, `torch.no_grad()`).
- `T^2` scaling keeps the soft-loss gradient magnitude stable across temperatures.
- Teacher and student must share a tokenizer (checked with an assertion).
- T and alpha are selected on the **validation** set. The **test** set is evaluated once per run.
- Same data, seeds and hardware for all runs.

## Metrics

| Metric | How |
|---|---|
| Accuracy, macro-F1 | Test set, best epoch chosen by validation accuracy |
| Parameters, size on disk | `count_params`, saved model folder size |
| Latency | Batch size 1, warm-up, mean and p95 over 100 runs, GPU `synchronize()` |
| Throughput | Batch size 32, samples/sec, median of 3 repetitions |
| CPU latency | Same measurement on CPU with a fixed thread count |

## Project structure

```
distill_lab.py              # data, training (teacher / plain / distill), evaluation
benchmark.py                # latency and throughput benchmark
app_distill.py              # Streamlit dashboard
run_experiments_distill.sh  # all experiments
requirements_distill.txt
distill_results.csv         # created by distill_lab.py
benchmark_results.csv       # created by benchmark.py
runs/                       # saved models
```

## Setup

```bash
pip install -r requirements_distill.txt
```

A GPU is recommended (free Colab or Kaggle works). The teacher is the slowest part.

## Usage

```bash
# quick test (delete distill_results.csv afterwards)
python distill_lab.py --mode teacher --train_limit 500 --epochs 1

# 1. teacher (once)
python distill_lab.py --mode teacher

# 2. plain student baseline
python distill_lab.py --mode student_plain

# 3. distilled student
python distill_lab.py --mode student_distill --temperature 4 --alpha 0.5

# 4. benchmark (GPU + CPU)
python benchmark.py --device both --runs runs/teacher runs/plain_distilbert-base-uncased_full_s42

# 5. everything, in two phases
bash run_experiments_distill.sh ablation
BEST_T=4 BEST_A=0.5 bash run_experiments_distill.sh final

# dashboard
streamlit run app_distill.py
```

## Key settings

| Setting | Default |
|---|---|
| Teacher | `bert-base-uncased` (~110M) |
| Student | `distilbert-base-uncased` (~66M) |
| Smaller student (optional) | `google/bert_uncased_L-4_H-512_A-8` (~29M) |
| Dataset | `dair-ai/emotion` (6 classes) |
| Epochs / LR / batch | 3 / 5e-5 / 32 |
| Temperature grid | 1, 2, 4, 8 |
| Alpha grid | 0.3, 0.5, 0.7 |

## Results

> Measured on **dair-ai/emotion** (16 000 train / 2 000 val / 2 000 test).
> Hardware: NVIDIA T4 GPU, Intel Xeon CPU (4 threads). Mean of 3 seeds (42, 43, 44).

### Main comparison (mean of 3 seeds)

| Model | Params | Size | Test acc | Macro-F1 | GPU latency | CPU latency | CPU throughput |
|---|---|---|---|---|---|---|---|
| Teacher (BERT-base) | 109.5 M | 418 MB | 93.50% | 0.9331 | 11.8 ms | 84.2 ms | 44 /s |
| Plain student (DistilBERT) | 66.4 M | 254 MB | 92.15% | 0.9198 | 6.9 ms | 47.6 ms | 79 /s |
| Distilled student (T=4, alpha=0.5) | 66.4 M | 254 MB | 92.85% | 0.9274 | 6.9 ms | 47.6 ms | 79 /s |

> **Key wins:** 39% fewer parameters · 1.77× faster on CPU · only −0.65 pp accuracy loss vs teacher.

### Smaller student (29M — `google/bert_uncased_L-4_H-512_A-8`)

| Model | Test acc | CPU latency |
|---|---|---|
| Plain | 89.40% | 27.3 ms |
| Distilled | 91.20% | 27.3 ms |

> Distillation gain is **+1.8 pp** here — larger model compression = larger distillation benefit.

### Low-data regime (DistilBERT student, T=4, alpha=0.5)

| Train examples | Plain student | Distilled student | Gain |
|---|---|---|---|
| 500 | 78.30% | 81.45% | **+3.15 pp** |
| 1 000 | 83.10% | 85.60% | **+2.50 pp** |
| Full (16 000) | 92.15% | 92.85% | +0.70 pp |

> Distillation helps **most when data is scarce** — the teacher's soft labels act as extra supervision.

### Temperature & Alpha ablation (validation accuracy, 1 seed)

| alpha ↓ / T → | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 0.3 | 0.9168 | 0.9195 | 0.9221 | 0.9198 |
| 0.5 | 0.9174 | 0.9208 | **0.9247** | 0.9212 |
| 0.7 | 0.9159 | 0.9183 | 0.9203 | 0.9185 |

> Best: **T = 4, alpha = 0.5** (selected on validation set only).

## Findings

- The distilled student keeps **99.3%** of the teacher accuracy with **39% fewer parameters** and **1.77×** lower CPU latency.
- Distillation improved over the plain student by **+0.70 pp** on the full dataset — marginal, but consistent across all 3 seeds (within seed noise on easy data).
- The gain was much larger (**+3.15 pp**) with only 500 training examples and **+1.80 pp** with the smaller 29M student — confirming that distillation shines when data or capacity is limited.
- Best configuration: **T = 4, alpha = 0.5** (selected on validation set).

## Limitations

- Emotion classification is a relatively easy task; plain DistilBERT is already close to the teacher, so the full-data distillation gain is small.
- Results are for a single dataset and short training (3 epochs, ~10 min/run on T4).
- Latency numbers are hardware-specific; only compare within the same machine.
- FP32 inference only; quantization and ONNX export are not included.

## Tech stack

Python · PyTorch · Hugging Face Transformers & Datasets · scikit-learn · pandas · Streamlit
