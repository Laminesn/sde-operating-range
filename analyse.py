"""
Aggregate results/*.json into the five things that decide whether this is a paper.

  1. TABLE 1 replication    -- does their headline F1 reproduce at their exact
                               architecture and scale? If yes, the harness is
                               trusted. If no, the reproduction failure is ours
                               and everything else is on hold.
  2. TABLE 2 replication    -- their training-sufficiency numbers.
  3. FINDING A              -- does the Sec. 3.1 ordering H(IT) > H(OOT) reverse
                               where Eq. 2 succeeds? Reported as the joint
                               distribution of (sec3.1 p-value, Eq. 2 F1).
  4. FINDING B              -- the continuous memorisation sweep. F1 vs train
                               accuracy, with the paper's own Table 7
                               checkpoints marked, to show where their
                               sufficiency study actually sampled.
  5. FINDING C              -- sqrt(dim) vs median bandwidth, same models.

Usage:  python3 analyse.py [--results results] [--csv out.csv]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))

# Paper Table 1, C10-AllCNN and C10-ResNet18 rows: F1 by (R, |S|)
TABLE1 = {
    ("cifar10", "allcnn"): {
        (0.05, 400): 0.69, (0.05, 1000): 0.79, (0.05, 2000): 0.81,
        (0.10, 400): 0.61, (0.10, 1000): 0.66, (0.10, 2000): 0.82,
        (0.20, 400): 0.45, (0.20, 1000): 0.71, (0.20, 2000): 0.79,
    },
    ("cifar10", "resnet18"): {
        (0.05, 400): 0.87, (0.05, 1000): 0.97, (0.05, 2000): 0.99,
        (0.10, 400): 0.88, (0.10, 1000): 0.95, (0.10, 2000): 0.97,
        (0.20, 400): 0.86, (0.20, 1000): 0.96, (0.20, 2000): 1.00,
    },
    ("svhn", "allcnn"): {
        (0.05, 400): 0.62, (0.05, 1000): 0.74, (0.05, 2000): 0.83,
        (0.10, 400): 0.50, (0.10, 1000): 0.62, (0.10, 2000): 0.73,
        (0.20, 400): 0.72, (0.20, 1000): 0.82, (0.20, 2000): 0.90,
    },
    ("cifar100", "resnet18"): {
        (0.05, 400): 0.99, (0.05, 1000): 1.00, (0.05, 2000): 1.00,
        (0.10, 400): 0.97, (0.10, 1000): 1.00, (0.10, 2000): 1.00,
        (0.20, 400): 0.99, (0.20, 1000): 1.00, (0.20, 2000): 1.00,
    },
}

# Paper Table 2: F1 at 10/20/40/80% of training (R=10%, |S|=1000, hp)
TABLE2 = {
    ("svhn", "allcnn"): (0.49, 0.51, 0.58, 0.62),
    ("cifar10", "allcnn"): (0.48, 0.66, 0.54, 0.66),
    ("svhn", "resnet18"): (0.58, 0.77, 0.79, 0.78),
    ("cifar10", "resnet18"): (0.69, 0.61, 0.72, 0.95),
    ("cifar100", "resnet18"): (0.59, 0.60, 0.88, 1.00),
}


def load(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def agg(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), v.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = load(args.results)
    if not runs:
        print(f"no results in {args.results} -- run run_replication.py first")
        return
    key = (runs[0]["dataset"], runs[0]["arch"])
    print(f"{len(runs)} runs loaded: "
          f"{sorted(set((r['dataset'], r['arch']) for r in runs))}")

    rows = []
    for r in runs:
        for rec in r["records"]:
            rows.append({**rec, "dataset": r["dataset"], "arch": r["arch"],
                         "R": r["R"], "seed": r["seed"], "ep80": r["ep80"],
                         "epochs": r["epochs"]})

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} rows)")

    # ---------------------------------------------------------- 1. Table 1
    print("\n" + "=" * 78)
    print("1. TABLE 1 REPLICATION  (80% checkpoint, sigma=sqrt(dim), hp)")
    print("=" * 78)
    paper = TABLE1.get(key, {})
    print(f"{'R':>6}{'|S|':>7}{'ours F1':>18}{'paper F1':>10}{'delta':>9}{'n':>4}")
    t1 = defaultdict(list)
    for x in rows:
        if x["role"] == "table1" and x["sigma_mode"] == "sqrtdim":
            t1[(x["R"], x["n"])].append(x["f1"])
    for (R, n) in sorted(t1):
        mu, sd = agg(t1[(R, n)])
        pv = paper.get((round(R, 2), n))
        d = f"{mu - pv:+.2f}" if pv is not None else "-"
        print(f"{R:>6.0%}{n:>7}{f'{mu:.2f}+-{sd:.2f}':>18}"
              f"{(f'{pv:.2f}' if pv else '-'):>10}{d:>9}{len(t1[(R,n)]):>4}")

    # ---------------------------------------------------------- 2. Table 2
    print("\n" + "=" * 78)
    print("2. TABLE 2 REPLICATION  (R=10%, |S|=1000, checkpoints at 10/20/40/80%)")
    print("=" * 78)
    p2 = TABLE2.get(key)
    print(f"{'ckpt':>8}{'train_acc':>12}{'ours F1':>18}{'paper F1':>10}{'delta':>9}")
    for i, frac in enumerate(pipeline.CKPT_FRACS):
        sel = [x for x in rows if x["role"] == "sweep" and abs(x["R"] - 0.10) < 1e-9
               and x["epoch"] == max(1, int(round(frac * x["epochs"])))]
        if not sel:
            continue
        mu, sd = agg([x["f1"] for x in sel])
        ta, _ = agg([x["train_acc"] for x in sel])
        pv = p2[i] if p2 else None
        d = f"{mu - pv:+.2f}" if pv is not None else "-"
        print(f"{frac:>8.0%}{ta:>12.4f}{f'{mu:.2f}+-{sd:.2f}':>18}"
              f"{(f'{pv:.2f}' if pv else '-'):>10}{d:>9}")
    if key in pipeline.TABLE7:
        print(f"  paper Table 7 train acc at these checkpoints: "
              f"{pipeline.TABLE7[key]['train']}")

    # ---------------------------------------------------------- 3. Finding A
    print("\n" + "=" * 78)
    print("3. FINDING A -- does the Sec. 3.1 ordering hold where Eq. 2 works?")
    print("=" * 78)
    print("   Sec. 3.1 claims H(S_IT) > H(S_OOT). 'holds' = one-sided p < 0.01.")
    print(f"{'train_acc bin':>16}{'n':>5}{'Eq.2 F1':>11}{'sec3.1 holds':>15}"
          f"{'mean p':>10}{'mean effect':>13}")
    bins = [(0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
    for lo, hi in bins:
        sel = [x for x in rows if x["role"] == "sweep" and lo <= x["train_acc"] < hi]
        if not sel:
            continue
        f1m, _ = agg([x["f1"] for x in sel])
        hold = np.mean([x["sec31_detected"] for x in sel])
        pm, _ = agg([x["sec31_p"] for x in sel])
        em, _ = agg([x["sec31_effect"] for x in sel])
        print(f"{f'[{lo:.2f},{hi:.2f})':>16}{len(sel):>5}{f1m:>11.3f}"
              f"{hold:>14.0%}{pm:>10.3g}{em:>+13.3f}")
    print("   Finding A is confirmed if F1 is HIGH where 'sec3.1 holds' is LOW.")

    # ---------------------------------------------------------- 4. Finding B
    print("\n" + "=" * 78)
    print("4. FINDING B -- continuous memorisation sweep (R=10%, |S|=1000)")
    print("=" * 78)
    sel = [x for x in rows if x["role"] == "sweep" and abs(x["R"] - 0.10) < 1e-9]
    byep = defaultdict(list)
    for x in sel:
        byep[x["epoch"]].append(x)
    print(f"{'epoch':>7}{'frac':>7}{'train_acc':>12}{'test_acc':>10}"
          f"{'F1':>16}{'accuracy':>11}{'OTR':>8}{'IT-rate':>9}")
    for ep in sorted(byep):
        g = byep[ep]
        ta, _ = agg([x["train_acc"] for x in g])
        te_, _ = agg([x["test_acc"] for x in g])
        f1m, f1s = agg([x["f1"] for x in g])
        ac, _ = agg([x["accuracy"] for x in g])
        ot, _ = agg([x["otr"] for x in g])
        it, _ = agg([x["in_training_rate_on_Dr"] for x in g])
        mark = "  <- paper ckpt" if abs(g[0]["frac"] - 0.10) < .02 or \
            abs(g[0]["frac"] - 0.20) < .02 or abs(g[0]["frac"] - 0.40) < .02 or \
            abs(g[0]["frac"] - 0.80) < .02 else ""
        print(f"{ep:>7}{g[0]['frac']:>7.2f}{ta:>12.4f}{te_:>10.4f}"
              f"{f'{f1m:.3f}+-{f1s:.3f}':>16}{ac:>11.3f}{ot:>8.3f}{it:>9.3f}{mark}")
    lo_acc = [x for x in sel if x["train_acc"] < 0.90]
    if lo_acc:
        f1m, _ = agg([x["f1"] for x in lo_acc])
        ac, _ = agg([x["accuracy"] for x in lo_acc])
        ot, _ = agg([x["otr"] for x in lo_acc])
        print(f"\n   below 90% train acc (n={len(lo_acc)}): "
              f"F1={f1m:.3f}  accuracy={ac:.3f}  OTR={ot:.3f}")
        print(f"   -> the paper's OWN Table 7 shows its earliest checkpoint is")
        print(f"      already at {pipeline.TABLE7.get(key,{}).get('train',['?'])[0]}% "
              f"train accuracy, so this regime is untested by them.")

    # ---------------------------------------------------------- 5. Finding C
    print("\n" + "=" * 78)
    print("5. FINDING C -- bandwidth: sqrt(dim) (their default) vs median")
    print("=" * 78)
    print(f"{'R':>6}{'|S|':>7}{'F1 sqrtdim':>16}{'F1 median':>16}{'delta':>9}"
          f"{'sigma sqrtdim':>15}{'sigma median':>14}")
    bw = defaultdict(dict)
    for x in rows:
        if x["role"] == "table1":
            bw[(x["R"], x["n"])].setdefault(x["sigma_mode"], []).append(x)
    for k in sorted(bw):
        d = bw[k]
        if "sqrtdim" not in d or "median" not in d:
            continue
        a, sa = agg([z["f1"] for z in d["sqrtdim"]])
        b, sb = agg([z["f1"] for z in d["median"]])
        s1, _ = agg([z["sigma"] for z in d["sqrtdim"]])
        s2, _ = agg([z["sigma"] for z in d["median"]])
        print(f"{k[0]:>6.0%}{k[1]:>7}{f'{a:.2f}+-{sa:.2f}':>16}"
              f"{f'{b:.2f}+-{sb:.2f}':>16}{b-a:>+9.2f}{s1:>15.2f}{s2:>14.2f}")

    # ------------------------------------------------ 6. degenerate F1 floor
    print("\n" + "=" * 78)
    print("6. IS F1 EVEN INFORMATIVE HERE?  (the degenerate-classifier floor)")
    print("=" * 78)
    print("   The task is balanced: m in-training targets, m out-of-training.")
    print("   A classifier that answers 'in-training' to EVERYTHING scores")
    print("   precision 0.5, recall 1.0 -> F1 = 2(.5)(1)/1.5 = 0.667, accuracy 0.5.")
    print("   So any F1 <= 0.667 is consistent with zero discrimination, and the")
    print("   paper reports F1 without accuracy or base rate. Their C10-AllCNN")
    print("   Table 2 row is 0.48 / 0.66 / 0.54 / 0.66 -- none above the floor.")
    sw = [x for x in rows if x["role"] == "sweep"]
    print(f"\n{'train_acc bin':>16}{'n':>5}{'F1':>9}{'accuracy':>11}"
          f"{'IT-rate':>9}{'OTR':>8}{'degenerate?':>13}")
    for lo, hi in [(0, .7), (.7, .8), (.8, .85), (.85, .90), (.90, .93),
                   (.93, .96), (.96, .98), (.98, .995), (.995, 1.01)]:
        s = [x for x in sw if lo <= x["train_acc"] < hi]
        if not s:
            continue
        f1m, _ = agg([x["f1"] for x in s])
        ac, _ = agg([x["accuracy"] for x in s])
        it, _ = agg([x["in_training_rate_on_Dr"] for x in s])
        ot, _ = agg([x["otr"] for x in s])
        deg = "YES" if ac < 0.55 else ("marginal" if ac < 0.60 else "no")
        print(f"{f'[{lo:.3f},{hi:.3f})':>16}{len(s):>5}{f1m:>9.3f}{ac:>11.3f}"
              f"{it:>9.3f}{ot:>8.3f}{deg:>13}")

    # ------------------------------------------------ 7. threshold location
    print("\n" + "=" * 78)
    print("7. WHERE IS THE THRESHOLD?  (accuracy, not F1 -- see section 6)")
    print("=" * 78)
    pts = sorted([(x["train_acc"], x["accuracy"], x["f1"]) for x in sw])
    if pts:
        ta = np.array([p[0] for p in pts])
        ac = np.array([p[1] for p in pts])
        # sliding window over train accuracy
        print(f"{'window':>18}{'n':>5}{'mean accuracy':>15}{'mean F1':>10}")
        edges = np.arange(0.40, 1.005, 0.05)
        cross = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (ta >= lo) & (ta < hi)
            if k.sum() < 3:
                continue
            a = ac[k].mean()
            print(f"{f'[{lo:.2f},{hi:.2f})':>18}{int(k.sum()):>5}{a:>15.3f}"
                  f"{np.array([p[2] for p in pts])[k].mean():>10.3f}")
            if cross is None and a >= 0.60:
                cross = (lo, hi)
        if cross:
            print(f"\n   -> accuracy first exceeds 0.60 in train_acc window "
                  f"[{cross[0]:.2f},{cross[1]:.2f})")
        else:
            print("\n   -> accuracy never reaches 0.60: no working regime observed")

    # ------------------------------------------------ 8. JSD grid sensitivity
    print("\n" + "=" * 78)
    print("8. THE JSD GRID -- an implementation choice the paper never specifies")
    print("=" * 78)
    print("   Eq. 2 compares JSD(tar,IT) against JSD(tar,OOT). JSD on samples")
    print("   needs a discretisation and the paper gives none. 'shared' bins all")
    print("   three distributions on one grid (the only way the two divergences")
    print("   are comparable); 'per-pair' bins each comparison on its own support.")
    if "grid_disagreement" in sw[0]:
        print(f"\n{'train_acc bin':>16}{'n':>5}{'F1 shared':>12}{'F1 per-pair':>13}"
              f"{'OTR shared':>12}{'OTR per-pair':>14}{'verdicts differ':>17}")
        for lo, hi in [(0, .8), (.8, .90), (.90, .96), (.96, .99), (.99, 1.01)]:
            s = [x for x in sw if lo <= x["train_acc"] < hi]
            if not s:
                continue
            print(f"{f'[{lo:.2f},{hi:.2f})':>16}{len(s):>5}"
                  f"{agg([x['f1'] for x in s])[0]:>12.3f}"
                  f"{agg([x['f1_pairgrid'] for x in s])[0]:>13.3f}"
                  f"{agg([x['otr'] for x in s])[0]:>12.3f}"
                  f"{agg([x['otr_pairgrid'] for x in s])[0]:>14.3f}"
                  f"{agg([x['grid_disagreement'] for x in s])[0]:>16.1%}")
        print("\n   OTR is the paper's headline unlearning number (Alg. 3, Table 4).")
        print("   If it moves materially between the two, the metric's output")
        print("   depends on a choice the paper does not make.")

    print("\n" + "=" * 78)
    print("Wall time:", f"{sum(r['wall_seconds'] for r in runs)/3600:.1f} h across "
          f"{len(runs)} models")


if __name__ == "__main__":
    main()
