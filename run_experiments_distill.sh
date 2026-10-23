#!/usr/bin/env bash
# Knowledge Distillation Lab - experiment runner
#
# Phase 1 (ablation): teacher + plain student + T x alpha grid (seed 42)
#     bash run_experiments_distill.sh ablation
#
# -> open the dashboard / distill_results.csv, pick the best T and alpha by VALIDATION accuracy
#
# Phase 2 (final): extra seeds, tiny student, low-data runs, benchmarks
#     BEST_T=4 BEST_A=0.5 bash run_experiments_distill.sh final
#
# Tip: to save time, reduce the grids below (for example T in "2 4" and A in "0.5").

set -e

STUDENT=distilbert-base-uncased
STUDENT_SHORT=distilbert-base-uncased
TINY=google/bert_uncased_L-4_H-512_A-8
TINY_SHORT=bert_uncased_L-4_H-512_A-8
BEST_T=${BEST_T:-4}
BEST_A=${BEST_A:-0.5}

case "$1" in
  ablation)
    # Stage 2: teacher (trained once, reused by every distillation run)
    python distill_lab.py --mode teacher

    # Stage 3: plain student baseline (labels only)
    python distill_lab.py --mode student_plain --student_name $STUDENT --seed 42

    # Stage 4 + 6: distillation grid
    for T in 1 2 4 8; do
      for A in 0.3 0.5 0.7; do
        python distill_lab.py --mode student_distill --student_name $STUDENT \
          --temperature $T --alpha $A --seed 42
      done
    done
    ;;

  final)
    # Extra seeds for the main comparison (mean over 3 seeds)
    for S in 43 44; do
      python distill_lab.py --mode student_plain --student_name $STUDENT --seed $S
      python distill_lab.py --mode student_distill --student_name $STUDENT \
        --temperature $BEST_T --alpha $BEST_A --seed $S
    done

    # Smaller student (~29M): distillation gap is usually clearer here
    python distill_lab.py --mode student_plain --student_name $TINY --lr 1e-4
    python distill_lab.py --mode student_distill --student_name $TINY --lr 1e-4 \
      --temperature $BEST_T --alpha $BEST_A

    # Low-data regimes (teacher still trained on full data)
    for N in 500 1000; do
      python distill_lab.py --mode student_plain --student_name $STUDENT \
        --train_limit $N --epochs 6
      python distill_lab.py --mode student_distill --student_name $STUDENT \
        --train_limit $N --epochs 6 --temperature $BEST_T --alpha $BEST_A
    done

    # Stage 5: latency / throughput on GPU and CPU
    python benchmark.py --device both --runs \
      runs/teacher \
      runs/plain_${STUDENT_SHORT}_full_s42 \
      runs/distill_${STUDENT_SHORT}_full_T${BEST_T}_a${BEST_A}_s42 \
      runs/plain_${TINY_SHORT}_full_s42 \
      runs/distill_${TINY_SHORT}_full_T${BEST_T}_a${BEST_A}_s42
    ;;

  *)
    echo "Usage:"
    echo "  bash run_experiments_distill.sh ablation"
    echo "  BEST_T=4 BEST_A=0.5 bash run_experiments_distill.sh final"
    exit 1
    ;;
esac
