#!/usr/bin/env python3
"""SDE on ACTUALLY UNLEARNED models — the program chair's camera-ready requirement.

The paper currently uses training checkpoints as a stand-in for unlearning. That
stand-in is the thing all three reviews objected to: an early checkpoint has
memorised nothing, whereas an unlearned model has memorised D_r and is supposed
to have released D_f. This runs the real thing.

For each saved SDE model (models/*.pt, which carries its own D_r/D_f split):
  theta_o  -> unlearn(method) -> ActBank -> protocol.evaluate

Reported per (model, method): SDE accuracy and F1, the Sec. 3.1 gate, and the
damage axis (retain / forget / test accuracy), so SDE can be read against how
much of D_f actually survived rather than against epoch count.

theta_o itself is evaluated as the "none" baseline, so each row has its own
control.

Usage:
  python3 run_unlearned_sde.py --R 05                 # 5 seeds x 4 methods + baseline
  python3 run_unlearned_sde.py --R 05 --methods FineTuning --seeds 0
  python3 run_unlearned_sde.py --R 05 10 20           # everything (~5 h)
"""
import argparse, glob, json, os, re, time
import numpy as np
import torch

import arch, pipeline, protocol, unlearn as U

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["GradientAscent", "NegGradPlus", "FineTuning", "SCRUB"]


def accs(bank, tr_clean, te, D_r, D_f):
    """Retain / forget / test accuracy, free from the activation bank."""
    p_tr, p_te = bank.predictions()
    ytr = np.asarray(tr_clean.targets)
    yte = np.asarray(te.targets)
    return dict(retain_acc=float((p_tr[D_r] == ytr[D_r]).mean()),
                forget_acc=float((p_tr[D_f] == ytr[D_f]).mean()),
                test_acc=float((p_te == yte).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", nargs="+", default=["05"], choices=["05", "10", "20"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--n", type=int, default=1000, help="target subset size |S|")
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = []
    for R in args.R:
        for s in args.seeds:
            p = os.path.join(HERE, "models", f"cifar10-allcnn-R{R}-s{s}-m25-dense.pt")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  MISSING {os.path.basename(p)}")
    if not paths:
        raise SystemExit("no models found")

    tr_aug, tr_clean, te = pipeline.get_data("cifar10")

    print("=" * 78)
    print(f"SDE ON UNLEARNED MODELS   {len(paths)} models x "
          f"({len(args.methods)} methods + baseline)   |S|={args.n}  m={args.m}")
    print(f"  methods: {', '.join(args.methods)}   device={pipeline.DEV}")
    print("=" * 78)

    records, t0 = [], time.time()
    for p in paths:
        tag = os.path.basename(p).replace(".pt", "")
        b = torch.load(p, map_location="cpu", weights_only=False)
        D_r, D_f, seed = b["D_r"], b["D_f"], b["seed"]
        assert b["dataset"] == "cifar10" and b["arch"] == "allcnn", b
        assert not set(D_r) & set(D_f), "D_r and D_f overlap"

        rc = pipeline.RECIPE[("cifar10", "allcnn")]
        theta_o = arch.build("allcnn", n_cls=rc["n_cls"]).to(pipeline.DEV)
        theta_o.load_state_dict(b["state"])
        theta_o.eval()

        print(f"\n{tag}   R={b['R']:.0%}  seed={seed}  "
              f"|D_r|={len(D_r)} |D_f|={len(D_f)}")

        for method in ["none"] + list(args.methods):
            t1 = time.time()
            if method == "none":
                model = theta_o
            else:
                model = U.unlearn(method, theta_o, tr_aug, D_r, D_f,
                                  "cifar10", "allcnn", seed=seed)
                model.eval()

            bank = protocol.ActBank(model, tr_clean, te, tap="hp")
            a = accs(bank, tr_clean, te, D_r, D_f)
            r = protocol.evaluate(bank, D_r, D_f, len(te),
                                  n=args.n, m=args.m, seed=seed)
            r.update(method=method, model=tag, seed=seed, R=b["R"],
                     secs=round(time.time() - t1, 1), **a)
            records.append(r)

            print(f"  {method:<14} retain {a['retain_acc']:.3f}  "
                  f"forget {a['forget_acc']:.3f}  test {a['test_acc']:.3f}   "
                  f"SDE acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  "
                  f"gate {'FIRES ' if r.get('sec31_detected') else 'silent'}"
                  f"  [{r['secs']:.0f}s]")

            del bank
            if method != "none":
                del model

        del theta_o

    out = args.out or os.path.join(
        HERE, f"results_unlearned_sde_R{'-'.join(args.R)}.json")
    json.dump(dict(dataset="cifar10", arch="allcnn", n=args.n, m=args.m,
                   methods=args.methods, seeds=args.seeds, R=args.R,
                   wall_seconds=round(time.time() - t0, 1), records=records),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}   ({(time.time() - t0)/60:.1f} min)")

    # ---- summary: SDE by method, and against surviving forget-set fit ----
    floor = 2 / 3
    print(f"\n{'method':<16}{'n':>3}{'retain':>9}{'forget':>9}{'SDE acc':>10}"
          f"{'F1':>8}{'>floor':>8}{'gate':>7}")
    print("-" * 70)
    for method in ["none"] + list(args.methods):
        s = [r for r in records if r["method"] == method]
        if not s:
            continue
        f = lambda k: sum(r[k] for r in s) / len(s)
        print(f"{method:<16}{len(s):>3}{f('retain_acc'):>9.3f}{f('forget_acc'):>9.3f}"
              f"{f('accuracy'):>10.3f}{f('f1'):>8.3f}"
              f"{sum(r['f1'] > floor for r in s)/len(s):>8.0%}"
              f"{sum(bool(r.get('sec31_detected')) for r in s)/len(s):>7.0%}")


if __name__ == "__main__":
    main()
