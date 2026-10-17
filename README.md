# When Does Unlearning Verification Work?

Code and results for *When Does Unlearning Verification Work? Measuring the
Operating Range of a Split-Half Dependence Metric on CIFAR-10*, NeurIPS 2026
workshop on Privacy in the Era of Large Opaque Models (PriLOM), poster.

This is an **independent reimplementation** of the Split-half Dependence
Evaluation (SDE) of Zhang et al., ICLR 2026, built from the published text
because the repository linked in that paper is empty. It is not their code, and
any difference from their numbers may be ours. Everything that had to be
recovered rather than read off is listed in the paper's appendix; the two that
matter most are the biased HSIC estimator (their Sec. 2.2) and the All-CNN-C
variant recovered from the activation dimensionalities in their Table 6.

## What is here

| file | what it is |
|---|---|
| `hsic.py` | the SDE core: biased HSIC, the permutation null, the JSD decision rule, and the Sec. 3.1 Mann-Whitney applicability test |
| `protocol.py` | the Sec. 4.1 evaluation protocol and the activation bank |
| `arch.py` | All-CNN-C and ResNet-18, with all six tap dimensionalities asserted against their Table 6 |
| `pipeline.py` | data, forget/retain splits, the training recipe |
| `unlearn.py` | five approximate unlearning methods: Gradient Ascent, NegGrad+, Fine-Tuning and SCRUB from RULER's Sec. 4.2, plus Random-label, which is one of the four Zhang et al. actually validate SDE against |
| `analyse.py` | aggregation for the replication tables |
| `results/` | every result JSON behind the paper's tables and figures |
| `run_optional_chain.sh` | the follow-up runs: m=100 to match Zhang's Sec. 4.3, two more ResNet-18 seeds, and the other forgetting ratios |

## Reproducing each result

Each script writes a JSON into `results/`. Runtimes are for an Apple M2 Max
(MPS, no CUDA).

```bash
# Table 1, the replication. Trains 15 All-CNN models, then evaluates the |S| grid.
python3 run_replication.py --dataset cifar10 --arch allcnn --seeds 0 1 2 3 4
python3 run_table1_saved.py                 # re-scores saved models, ~1 min each

# Figure 1a and the operating-range bands: 525 checkpoints, from the same run above.

# Figure 1 (single panel with the inset used in the paper), and the
# generalisation-gap figure in the appendix. No training needed, seconds.
python3 make_figure_range.py --single
python3 make_figure_gap.py

# ResNet-18 operating range. The snapshots come from run_c100.py --save-ckpt,
# so generate them first (~70 min), then evaluate (~4 min).
python3 run_c100.py --dataset cifar10 --seed 0 --save-ckpt
python3 run_r18_range.py --dataset cifar10 --seed 0

# SDE on models that actually memorised the forget set, then unlearned it.
# Trains theta_o on D_r + D_f, unlearns with five methods, evaluates over
# several draws, and reports the out-of-training rate alongside accuracy (~4 h).
python3 run_unlearn_proper.py --R 05

# Evaluation draw vs training seed: 5 models x 5 draws (~10 min)
python3 run_evalnoise.py --R 05

# Does dropping augmentation reproduce their trajectory? (~25 min)
python3 run_noaug.py --seed 0
```

## Install

```bash
pip install -r requirements.txt
```

The results here were produced on an Apple M2 Max with the MPS backend and no
CUDA. `torch==2.4.1` has no MPS autocast, so there is no mixed-precision path.

## What is not here

Model checkpoints. They are ~11 GB and reproducible from the scripts above, at
roughly 32 GPU-hours in total. `run_r18_range.py` reads epoch snapshots that
`run_c100.py --save-ckpt` produces; regenerate them before running it.

CIFAR-10 downloads on first use.

## Notes

- Decoding and training are seeded, but evaluating mid-training perturbs the
  training RNG unless the global torch and numpy state is snapshotted around
  every checkpoint. `pipeline.py` does this; the paper's appendix explains why.
  Any mid-training evaluation loop on a shared global RNG has this bug.
- A single SDE evaluation at `m=25` is noisy: the same weights give accuracies
  spanning 0.24 to 0.94 across reference and target draws. Average over draws.

## Citation

```
@inproceedings{deen2026operatingrange,
  title     = {When Does Unlearning Verification Work? Measuring the Operating
               Range of a Split-Half Dependence Metric on CIFAR-10},
  author    = {Deen, Lamine},
  booktitle = {NeurIPS 2026 Workshop on Privacy in the Era of Large Opaque Models},
  year      = {2026}
}
```

MIT licensed.
