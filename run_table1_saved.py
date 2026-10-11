"""
Table 1 replication on the SAVED post-RNG-fix models.

WHY THIS EXISTS
  The |S| x sigma grid was only ever run in the first (pre-RNG-fix) sweep, where
  the checkpoint schedule leaked into the training RNG. Everything else in the
  paper comes from the post-fix dense run, which was launched with --no-table1.
  Mixing the two would mean the replication claim and the threshold claim rest
  on different models.

  Retraining 15 models to the 80% checkpoint costs ~5 h. The final models are
  already on disk, so the grid is recomputed on those instead: ~1 min each.

DIFFERENCE FROM THE PAPER, STATED
  Sec. 4.1.2 evaluates at the checkpoint at 80% of training (epoch 40 of 50).
  These are the FINAL models (epoch 50). Both are fully memorised -- train
  accuracy 0.997 at ep40 vs 0.9996 at ep50 -- and the dense sweep measured both,
  so the gap is checkable rather than assumed.

Usage:  python3 run_table1_saved.py
"""

import glob
import json
import os
import time

import numpy as np
import torch

import pipeline
import protocol
from arch import build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_table1_saved.json")


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "models", "*.pt")))
    print(f"{len(paths)} saved models")
    tr_aug, tr_clean, te = pipeline.get_data("cifar10")
    y_tr, y_te = pipeline.labels_of(tr_clean), pipeline.labels_of(te)

    rows = []
    for i, p in enumerate(paths, 1):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = build(ck["arch"], n_cls=10).to(pipeline.DEV)
        m.load_state_dict(ck["state"])
        m.eval()
        D_r, D_f = np.asarray(ck["D_r"]), np.asarray(ck["D_f"])
        t0 = time.time()
        bank = protocol.ActBank(m, tr_clean, te, tap="hp")
        p_tr, p_te = bank.predictions()
        acc = dict(train_acc=float((p_tr[D_r] == y_tr[D_r]).mean()),
                   eval_acc=float((p_tr[D_f] == y_tr[D_f]).mean()),
                   test_acc=float((p_te == y_te).mean()))
        print(f"[{i}/{len(paths)}] R={ck['R']:.0%} s={ck['seed']}  "
              f"train={acc['train_acc']:.4f} test={acc['test_acc']:.4f}")
        for n in (400, 1000, 2000):
            for sm in ("sqrtdim", "median"):
                r = protocol.evaluate(bank, D_r, D_f, len(te), n=n, m=25,
                                      sigma_mode=sm, seed=ck["seed"])
                r.update(R=ck["R"], seed=ck["seed"], role="table1_saved", **acc)
                rows.append(r)
                print(f"      |S|={n:>4} sigma={sm:<8}({r['sigma']:6.2f})  "
                      f"F1={r['f1']:.3f}  acc={r['accuracy']:.3f}  OTR={r['otr']:.3f}")
        print(f"      ...{time.time()-t0:.0f}s")

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nsaved -> {OUT}  ({len(rows)} rows)")

    # summary against the paper
    from analyse import TABLE1, agg
    paper = TABLE1[("cifar10", "allcnn")]
    print("\n" + "=" * 78)
    print("TABLE 1 REPLICATION (post-RNG-fix, final checkpoint, sigma=sqrt(dim))")
    print("=" * 78)
    print(f"{'R':>6}{'|S|':>7}{'ours F1':>18}{'paper F1':>10}{'delta':>9}"
          f"{'ours acc':>10}")
    deltas = []
    for R in (0.05, 0.10, 0.20):
        for n in (400, 1000, 2000):
            s = [x for x in rows if abs(x["R"] - R) < 1e-9 and x["n"] == n
                 and x["sigma_mode"] == "sqrtdim"]
            if not s:
                continue
            mu, sd = agg([x["f1"] for x in s])
            a, _ = agg([x["accuracy"] for x in s])
            pv = paper[(round(R, 2), n)]
            deltas.append(mu - pv)
            print(f"{R:>6.0%}{n:>7}{f'{mu:.2f}+-{sd:.2f}':>18}{pv:>10.2f}"
                  f"{mu-pv:>+9.2f}{a:>10.3f}")
    print(f"\n  mean delta vs paper: {np.mean(deltas):+.3f}   "
          f"max |delta|: {np.max(np.abs(deltas)):.3f}")
    print("  NOTE: 'ours acc' is the accuracy the paper never reports. A constant")
    print("  'in-training' classifier scores F1 0.667 / accuracy 0.500.")


if __name__ == "__main__":
    main()
