#!/usr/bin/env python3
"""SDE on models that actually memorised the forget set, then unlearned it.

Supersedes run_unlearned_sde.py, which applied unlearning to RETRAIN ORACLES —
models trained on D_r only, which never saw D_f. There was nothing to unlearn
there: forget accuracy equalled test accuracy to within 0.01 on every seed.

This trains theta_o on D_r UNION D_f, so D_f is genuinely memorised, then
unlearns, then runs SDE. Three reference points per seed:

  theta_o   trained on everything        -> D_f memorised, SDE should say "in"
  unlearned theta_o minus D_f            -> the case under test
  theta_r   retrain oracle (from models/) -> perfect deletion, SDE should say "out"

SDE accuracy is reported with the out-of-training rate on D_f subsets (OTR),
which is the quantity Zhang et al. use for unlearned models, so the verdict can
be read on their terms as well as ours.

Writes results_unlearn_proper_R<R>.json.
"""
import argparse, json, os, time
import numpy as np
import torch

import arch, pipeline, protocol, unlearn as U

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "theta_o_models")
METHODS = ["RandLabel", "GradientAscent", "NegGradPlus", "FineTuning", "SCRUB"]


def accs(bank, tr_clean, te, D_r, D_f):
    p_tr, p_te = bank.predictions()
    ytr = np.asarray(tr_clean.targets); yte = np.asarray(te.targets)
    return dict(retain_acc=float((p_tr[D_r] == ytr[D_r]).mean()),
                forget_acc=float((p_tr[D_f] == ytr[D_f]).mean()),
                test_acc=float((p_te == yte).mean()))


def evaluate(model, tr_clean, te, D_r, D_f, n, m, seed, label, tag, extra):
    bank = protocol.ActBank(model, tr_clean, te, tap="hp")
    a = accs(bank, tr_clean, te, D_r, D_f)
    r = protocol.evaluate(bank, D_r, D_f, len(te), n=n, m=m, seed=seed)
    # OTR: the rate at which D_f subsets are called out-of-training. The paper's
    # own accuracy counts a D_f subset correct when called out, and there are m
    # of each class, so tn/m is exactly that rate.
    r["otr_Df"] = float(r["tn"]) / max(r["tn"] + r["fp"], 1)
    r.update(method=label, model=tag, seed=seed, **a, **extra)
    print(f"  {label:<16} retain {a['retain_acc']:.3f}  forget {a['forget_acc']:.3f}  "
          f"test {a['test_acc']:.3f}   SDE acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  "
          f"OTR {r['otr_Df']:.3f}  gate {'FIRES ' if r.get('sec31_detected') else 'silent'}")
    del bank
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", default="05", choices=["05", "10", "20"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--draws", type=int, default=5,
                    help="evaluation draws per model; reported as a spread, "
                         "because a single draw carries about +/-0.4")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    tr_aug, tr_clean, te = pipeline.get_data("cifar10")
    rc = pipeline.RECIPE[("cifar10", "allcnn")]
    assert U.selftest(), "unlearning selftest failed"

    print("=" * 78)
    print(f"SDE ON PROPERLY UNLEARNED MODELS   R={args.R}%   seeds={args.seeds}")
    print(f"  theta_o trained on D_r UNION D_f, then unlearned")
    print(f"  methods: {', '.join(args.methods)}   draws/model: {args.draws}")
    print("=" * 78)

    records, t0 = [], time.time()
    for s in args.seeds:
        oracle_path = os.path.join(HERE, "models",
                                   f"cifar10-allcnn-R{args.R}-s{s}-m25-dense.pt")
        b = torch.load(oracle_path, map_location="cpu", weights_only=False)
        D_r, D_f = b["D_r"], b["D_f"]
        assert not set(D_r) & set(D_f)
        print(f"\nseed {s}   |D_r|={len(D_r)} |D_f|={len(D_f)}")

        # ---- theta_o: trained on EVERYTHING, so D_f is memorised ----
        o_path = os.path.join(CACHE, f"cifar10-allcnn-R{args.R}-s{s}-theta_o.pt")
        if os.path.exists(o_path):
            theta_o = arch.build("allcnn", n_cls=rc["n_cls"]).to(pipeline.DEV)
            theta_o.load_state_dict(torch.load(o_path, map_location=pipeline.DEV))
            print("  theta_o loaded from cache")
        else:
            t1 = time.time()
            all_idx = np.concatenate([D_r, D_f])
            theta_o, _ = pipeline.train("cifar10", "allcnn", all_idx, s,
                                        tr_aug=tr_aug, tr_clean=tr_clean, te=te,
                                        eval_idx=D_f, log=lambda *a: None)
            torch.save({k: v.detach().cpu().clone()
                        for k, v in theta_o.state_dict().items()}, o_path)
            print(f"  theta_o trained in {(time.time()-t1)/60:.1f} min")
        theta_o.eval()

        for d in range(args.draws):
            records.append(evaluate(theta_o, tr_clean, te, D_r, D_f, args.n,
                                    args.m, 1000 * d + s, "theta_o",
                                    f"R{args.R}-s{s}", dict(draw=d, R=b["R"])))

        # ---- theta_r: the retrain oracle, perfect deletion ----
        theta_r = arch.build("allcnn", n_cls=rc["n_cls"]).to(pipeline.DEV)
        theta_r.load_state_dict(b["state"]); theta_r.eval()
        for d in range(args.draws):
            records.append(evaluate(theta_r, tr_clean, te, D_r, D_f, args.n,
                                    args.m, 1000 * d + s, "theta_r_oracle",
                                    f"R{args.R}-s{s}", dict(draw=d, R=b["R"])))

        # ---- the unlearned models ----
        for method in args.methods:
            mdl = U.unlearn(method, theta_o, tr_aug, D_r, D_f,
                            "cifar10", "allcnn", seed=s)
            mdl.eval()
            for d in range(args.draws):
                records.append(evaluate(mdl, tr_clean, te, D_r, D_f, args.n,
                                        args.m, 1000 * d + s, method,
                                        f"R{args.R}-s{s}", dict(draw=d, R=b["R"])))
            del mdl
        del theta_o, theta_r

        out = os.path.join(HERE, f"results_unlearn_proper_R{args.R}.json")
        json.dump(dict(dataset="cifar10", arch="allcnn", R=args.R, n=args.n,
                       m=args.m, draws=args.draws, seeds=args.seeds,
                       methods=args.methods, theta_o_trained_on="D_r + D_f",
                       wall_seconds=round(time.time() - t0, 1), records=records),
                  open(out, "w"), indent=1)
        print(f"  [checkpointed {len(records)} records]")

    # ---- summary, averaged over draws then over seeds ----
    import statistics as st
    print(f"\n{'method':<17}{'retain':>8}{'forget':>8}{'SDE acc':>10}{'OTR(Df)':>9}"
          f"{'gate':>7}")
    print("-" * 60)
    for label in ["theta_o"] + list(args.methods) + ["theta_r_oracle"]:
        s_ = [r for r in records if r["method"] == label]
        if not s_:
            continue
        per_seed = {}
        for r in s_:
            per_seed.setdefault(r["seed"], []).append(r)
        acc = [st.mean(x["accuracy"] for x in v) for v in per_seed.values()]
        f = lambda k: st.mean(r[k] for r in s_)
        print(f"{label:<17}{f('retain_acc'):>8.3f}{f('forget_acc'):>8.3f}"
              f"{st.mean(acc):>10.3f}{f('otr_Df'):>9.3f}"
              f"{sum(bool(r.get('sec31_detected')) for r in s_)/len(s_):>7.0%}")
    print(f"\nwrote results_unlearn_proper_R{args.R}.json  "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
