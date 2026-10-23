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

> Fill these in after running the experiments. Replace X with your measured values.

### Main comparison (mean of 3 seeds)

| Model | Params | Size | Test acc | Macro-F1 | GPU latency | CPU latency | CPU throughput |
|---|---|---|---|---|---|---|---|
| Teacher (BERT-base) | X M | X MB | X% | X | X ms | X ms | X /s |
| Plain student | X M | X MB | X% | X | X ms | X ms | X /s |
| Distilled student (T=X, alpha=X) | X M | X MB | X% | X | X ms | X ms | X /s |

### Smaller student (29M)

| Model | Test acc | CPU latency |
|---|---|---|
| Plain | X% | X ms |
| Distilled | X% | X ms |

### Low-data regime

| Train examples | Plain student | Distilled student | Gain |
|---|---|---|---|
| 500 | X% | X% | +X |
| 1000 | X% | X% | +X |
| Full | X% | X% | +X |

## Findings

> Replace with your own observations from the dashboard. Be honest if the gain is small.

- The distilled student keeps X% of the teacher accuracy with X% fewer parameters and X times lower CPU latency.
- Distillation improved over the plain student by X points on the full data (within / beyond seed noise).
- The gain was larger (X points) with only 1000 training examples and with the smaller student.
- Best configuration: T = X, alpha = X (selected on validation).

## Limitations

- Emotion classification is an easy task, so a plain DistilBERT is already close to the teacher and the distillation gain is small on full data.
- Results are for a single dataset and short training (3 epochs).
- Latency depends on hardware; numbers are only comparable within the same machine.
- FP32 inference only; quantization and ONNX are not included.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, pandas, Streamlit.
