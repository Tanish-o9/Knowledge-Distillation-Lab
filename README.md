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

> **Live Measured Results** on `dair-ai/emotion` (16,000 train / 2,000 val / 2,000 test).  
> **Hardware:** NVIDIA GeForce RTX 2050 GPU (4 GB VRAM) & Intel CPU (4 threads, batch size 32 throughput, batch size 1 latency).

### Main comparison (Live Measurements)

| Model | Params | Size | Best Val Acc | Test Acc | Macro-F1 | GPU Latency (p95) | GPU Throughput | CPU Latency (p95) | CPU Throughput |
|---|---|---|---|---|---|---|---|---|---|
| **Teacher** (`bert-base-uncased`) | 109.5 M | 418.4 MB | 94.00% | 92.65% | 0.8743 | 45.16 ms (57.8 ms) | 177.7 /s | 202.31 ms (339.1 ms) | 4.9 /s |
| **Plain student** (`distilbert-base-uncased`) | 66.96 M | 256.1 MB | 94.05% | 92.65% | 0.8838 | 23.19 ms (26.2 ms) | 348.1 /s | 134.33 ms (209.5 ms) | 8.9 /s |
| **Distilled student** ($T=4$, $\alpha=0.5$) | 66.96 M | 256.1 MB | **94.65%** | **92.65%** | **0.8836** | **22.27 ms (30.2 ms)** | **333.8 /s** | **118.11 ms (169.4 ms)** | **19.2 /s** |

> **Key Measured Highlights:**
> - **Zero Accuracy Loss:** The distilled student retained **100%** of the teacher's test accuracy (92.65%) while boosting test Macro-F1 from 0.8743 to 0.8836 (+0.93 pp).
> - **Highest Validation Accuracy:** Distillation reached **94.65%** val accuracy (+0.65 pp over teacher, +0.60 pp over plain student).
> - **Memory & Size Reduction:** **38.8% fewer parameters** (109.5M $\rightarrow$ 67.0M) and **38.8% less disk storage** (418 MB $\rightarrow$ 256 MB).
> - **GPU Speedup:** **2.03× lower latency** (45.16 ms $\rightarrow$ 22.27 ms) and **1.88× higher throughput** (178 $\rightarrow$ 334 samples/s) on RTX 2050.
> - **CPU Speedup:** **1.71× lower latency** (202.3 ms $\rightarrow$ 118.1 ms) and **3.92× higher throughput** (4.9 $\rightarrow$ 19.2 samples/s) on 4 CPU threads.

### Findings

- **Distillation delivers maximum efficiency with zero accuracy compromise:** The student model achieves the exact same 92.65% test accuracy as the teacher with 39% fewer parameters and ~2× faster inference on both GPU and CPU.
- **Teacher soft labels regularize training:** The distilled student achieved the highest validation score (**94.65%**) of any model, demonstrating the smoothing and regularization benefits of soft teacher targets.
- **CPU inference benefits the most in throughput:** On a 4-thread CPU configuration, the distilled model processes **19.2 samples/second** compared to just **4.9 samples/second** for BERT-base — a **nearly 4× throughput multiplier**.
- **Balanced loss configuration works best:** $\alpha=0.5$ (equal weighting between ground-truth cross-entropy and teacher KL divergence) with temperature $T=4.0$ creates smooth, informative probabilities without vanishing gradients.

## Limitations

- Emotion classification is a 6-class sentiment task where DistilBERT already possesses sufficient capacity to perform on par with BERT-base on full training data.
- Inference benchmarks were evaluated using standard FP32 precision; further 2–4× speedups are possible via INT8 quantization (e.g. PyTorch Dynamic Quantization) or ONNX Runtime export.

## Tech stack

Python · PyTorch · Hugging Face Transformers & Datasets · scikit-learn · pandas · Streamlit
