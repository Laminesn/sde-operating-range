#!/usr/bin/env python3
"""Separate evaluation noise from training-seed variance at the operating point.

Section 6 of the paper attributes a 0.413-0.840 spread across 15 nominally
identical models to the training seed. But each evaluation draws only 2m = 50
target subsets and ONE pair of reference sets, so some of that spread is the
draw, not the model. This re-evaluates each saved model several times with
different evaluation seeds, holding the weights fixed.

within-model sd  = evaluation noise (same weights, different draw)
between-model sd = training-seed effect, once the noise is accounted for

Writes results_evalnoise_R<R>.json.
"""
import argparse, json, os, time
import statistics as st
import torch

import arch, pipeline, protocol

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", default="05", choices=["05", "10", "20"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--m", type=int, default=25)
    args = ap.parse_args()

    tr_aug, tr_clean, te = pipeline.get_data("cifar10")
    rc = pipeline.RECIPE[("cifar10", "allcnn")]

    print("=" * 78)
    print(f"EVALUATION NOISE vs TRAINING SEED   R={args.R}%   "
          f"{len(args.seeds)} models x {len(args.eval_seeds)} evaluation seeds")
    print("=" * 78)

    records, per_model, t0 = [], {}, time.time()
    for s in args.seeds:
        p = os.path.join(HERE, "models", f"cifar10-allcnn-R{args.R}-s{s}-m25-dense.pt")
        if not os.path.exists(p):
            print(f"  MISSING {os.path.basename(p)}")
            continue
        b = torch.load(p, map_location="cpu", weights_only=False)
        model = arch.build("allcnn", n_cls=rc["n_cls"]).to(pipeline.DEV)
        model.load_state_dict(b["state"])
        model.eval()
        bank = protocol.ActBank(model, tr_clean, te, tap="hp")

        accs = []
        for es in args.eval_seeds:
            r = protocol.evaluate(bank, b["D_r"], b["D_f"], len(te),
                                  n=args.n, m=args.m, seed=es)
            r.update(train_seed=s, eval_seed=es, R=b["R"])
            records.append(r)
            accs.append(r["accuracy"])
        per_model[s] = accs
        print(f"  model s{s}: " + " ".join(f"{a:.2f}" for a in accs) +
              f"   mean {st.mean(accs):.3f}  sd {st.pstdev(accs):.3f}")
        del bank, model

    within = [st.pstdev(v) for v in per_model.values() if len(v) > 1]
    means = [st.mean(v) for v in per_model.values()]
    w = st.mean(within) if within else float("nan")
    bt = st.pstdev(means) if len(means) > 1 else float("nan")
    # observed between-model sd inflated by noise/k; subtract it off
    k = len(args.eval_seeds)
    seed_var = max(bt ** 2 - (w ** 2) / k, 0.0)

    print("-" * 78)
    print(f"  mean within-model sd (evaluation noise) : {w:.3f}")
    print(f"  sd of per-model means (observed)        : {bt:.3f}")
    print(f"  training-seed sd after removing noise   : {seed_var ** 0.5:.3f}")
    print(f"  share of observed variance that is noise: "
          f"{(w**2/k)/bt**2:.0%}" if bt else "")
    print(f"  full range across all evaluations       : "
          f"{min(r['accuracy'] for r in records):.3f} to "
          f"{max(r['accuracy'] for r in records):.3f}")

    out = os.path.join(HERE, f"results_evalnoise_R{args.R}.json")
    json.dump(dict(R=args.R, n=args.n, m=args.m, seeds=args.seeds,
                   eval_seeds=args.eval_seeds,
                   within_sd=w, between_sd=bt, seed_sd=seed_var ** 0.5,
                   wall_seconds=round(time.time() - t0, 1), records=records),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
