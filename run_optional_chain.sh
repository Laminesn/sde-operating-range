#!/bin/bash
# Optional experiments, run in order of value. Each writes its own results file.
set -x
cd "$(dirname "$0")"
# 1. the reviewers' top ask: match Zhang's Sec 4.3 protocol of 100 subsets
python3 -u run_unlearn_proper.py --R 05 --m 100 --draws 3   2>&1
# 2. ResNet-18 operating range, two more seeds
for s in 1 2; do
  python3 -u run_c100.py --dataset cifar10 --seed $s --save-ckpt 2>&1
  python3 -u run_r18_range.py --dataset cifar10 --seed $s     2>&1
done
# 3. the other forgetting ratios on the unlearning experiment
python3 -u run_unlearn_proper.py --R 10 2>&1
python3 -u run_unlearn_proper.py --R 20 2>&1
echo "CHAIN COMPLETE"
