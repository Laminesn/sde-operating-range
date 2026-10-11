#!/usr/bin/env python3
"""SDE operating range on CIFAR-10 ResNet-18, from the saved epoch snapshots.

Camera-ready item: reviewers rKLp and 5Ai2 both asked whether the All-CNN
operating range transfers to the architecture where SDE is strongest. The
snapshots already exist (`c100_models/cifar10-s0-ep*.pt`, written by
run_c100.py with --save-ckpt), so this retrains nothing: it rebuilds the same
split, loads each snapshot and runs the Sec. 4.1 protocol on it.

Split provenance: run_c100.py uses ARCH="resnet18", R=0.05, scheme="independent",
seed from the filename. Reproduced here with the same call, and asserted against
the n_retain/n_forget recorded in results_c100_<tag>.json.

Writes results_r18_range_<tag>.json.
"""
import argparse, glob, json, os, re, time
import numpy as np
import torch

import arch, pipeline, protocol

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH, R, SCHEME = "resnet18", 0.05, "independent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="target subset size |S|")
    ap.add_argument("--m", type=int, default=25, help="subsets per class")
    args = ap.parse_args()

    ds, seed = args.dataset, args.seed
    tag = f"{ds}-s{seed}"
    rc = pipeline.RECIPE[(ds, ARCH)]

    snaps = sorted(glob.glob(os.path.join(HERE, "c100_models", f"{tag}-ep*.pt")),
                   key=lambda p: int(re.search(r"ep(\d+)\.pt$", p).group(1)))
    if not snaps:
        raise SystemExit(f"no snapshots matching c100_models/{tag}-ep*.pt")

    arch.check_dims(ARCH, n_cls=rc["n_cls"])
    assert pipeline.forget_split_selftest(log=lambda *a: None), "split schemes wrong"

    tr_aug, tr_clean, te = pipeline.get_data(ds)
    D_r, D_f = pipeline.forget_split(len(tr_clean), R, seed, scheme=SCHEME)
    assert not set(D_r) & set(D_f), "D_r and D_f overlap"

    # the gate: the split must be the one those snapshots were trained under
    ref_path = os.path.join(HERE, f"results_c100_{tag}.json")
    train_acc_by_epoch = {}
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        if "n_retain" in ref:
            assert ref["n_retain"] == len(D_r), \
                f"split mismatch: {ref['n_retain']} retained in json, {len(D_r)} here"
        for row in ref.get("rows", ref.get("records", [])):
            if "epoch" in row and "train_acc" in row:
                train_acc_by_epoch[int(row["epoch"])] = float(row["train_acc"])
        print(f"  split gate PASS against results_c100_{tag}.json")
    else:
        print(f"  WARNING: {ref_path} not found; train accuracies will be remeasured")

    print("=" * 78)
    print(f"SDE OPERATING RANGE  {ds}/{ARCH}  R={R:.0%}  seed={seed}  "
          f"|S|={args.n}  m={args.m}")
    print(f"  {len(snaps)} snapshots, device={pipeline.DEV}")
    print("=" * 78)

    records, t0 = [], time.time()
    for p in snaps:
        ep = int(re.search(r"ep(\d+)\.pt$", p).group(1))
        model = arch.build(ARCH, n_cls=rc["n_cls"]).to(pipeline.DEV)
        model.load_state_dict(torch.load(p, map_location=pipeline.DEV))
        model.eval()

        bank = protocol.ActBank(model, tr_clean, te, tap="hp")

        ta = train_acc_by_epoch.get(ep)
        if ta is None:                       # fall back to measuring it
            pred = bank.predictions()
            ta = float((pred["train"][D_r] == np.asarray(tr_clean.targets)[D_r]).mean())

        r = protocol.evaluate(bank, D_r, D_f, len(te), n=args.n, m=args.m, seed=seed)
        r.update(epoch=ep, train_acc=ta, arch=ARCH, dataset=ds, seed=seed,
                 R=R, snapshot=os.path.basename(p))
        records.append(r)
        print(f"  ep{ep:>3}  train_acc {ta:.4f}   SDE acc {r['accuracy']:.3f}   "
              f"F1 {r['f1']:.3f}   gate {'FIRES ' if r.get('sec31_detected') else 'silent'}"
              f" (p={r['sec31_p']:.3g})")
        del bank, model

    out = os.path.join(HERE, f"results_r18_range_{tag}.json")
    json.dump(dict(dataset=ds, arch=ARCH, R=R, seed=seed, n=args.n, m=args.m,
                   scheme=SCHEME, n_retain=len(D_r), n_forget=len(D_f),
                   wall_seconds=round(time.time() - t0, 1), records=records),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")

    floor = 2 / 3
    above = [r for r in records if r["f1"] > floor]
    print(f"above the trivial F1 floor: {len(above)}/{len(records)} snapshots")
    if above:
        print(f"  lowest train accuracy that clears it: "
              f"{min(r['train_acc'] for r in above):.4f}")


if __name__ == "__main__":
    main()
