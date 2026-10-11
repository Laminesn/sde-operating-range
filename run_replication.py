"""
The full replication. Handoff steps 1-4 in one resumable driver.

WHAT EACH CHECKPOINT PRODUCES

  every checkpoint epoch   -> Table 2 setting (|S|=1000, hp, sigma=sqrt(dim))
       This is BOTH their training-sufficiency replication AND the continuous
       memorisation sweep (handoff step 4). Dense over epochs 1-10 because
       Table 7 shows C10-AllCNN is already at 92.77% train accuracy by their
       first checkpoint -- their sufficiency study never observes a partially
       memorising model, so the regime Finding B is about lives in early epochs
       and nowhere else.

  the 80% checkpoint       -> Table 1 (|S| in {400,1000,2000}) and the
       bandwidth contrast (sqrt(dim) vs median). Sec. 4.1.2 states models are
       evaluated at "the checkpoint corresponding to 80% of the total training
       epochs", so that is where Table 1 comparisons are made.

  the Sec. 3.1 Mann-Whitney test rides along with every single evaluation,
  because Finding A is that its ordering reverses exactly where Eq. 2 succeeds.

RESUMABLE
  One JSON per (dataset, arch, R, seed) in results/. An existing file is
  skipped, so the run can be interrupted and restarted freely.

Usage
  python3 run_replication.py --dataset cifar10 --arch allcnn
  python3 run_replication.py --dataset cifar10 --arch resnet18 --seeds 0 1 2
  python3 run_replication.py --dataset cifar10 --arch allcnn --ratios 0.10 --quick
"""

import argparse
import json
import os
import time

import numpy as np

import pipeline
import protocol

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def run_model(dataset, arch, R, seed, sizes, m, tap, dense_first, data, log,
              ckpt_list=None, table1=True, save_model=None):
    tr_aug, tr_clean, te = data
    D_r, D_f = pipeline.forget_split(len(tr_clean), R, seed)
    epochs = pipeline.RECIPE[(dataset, arch)]["epochs"]
    ckpt_eps = ckpt_list or pipeline.checkpoint_epochs(epochs, dense_first)
    ep80 = max(1, int(round(0.80 * epochs)))
    y_tr, y_te = pipeline.labels_of(tr_clean), pipeline.labels_of(te)

    log(f"\n{'='*78}\n{dataset}-{arch}  R={R:.0%}  seed={seed}  "
        f"|D_r|={len(D_r)} |D_f|={len(D_f)}  epochs={epochs}")
    log(f"checkpoints ({len(ckpt_eps)}) at {ckpt_eps}  (80% ckpt = ep {ep80})\n{'='*78}")

    records = []

    def on_ckpt(model, info):
        t0 = time.time()
        bank = protocol.ActBank(model, tr_clean, te, tap=tap)

        # accuracies straight off the bank: logits = head(hp), so this is exact
        # and removes a separate 60k-image forward pass per checkpoint
        p_tr, p_te = bank.predictions()
        acc = dict(train_acc=float((p_tr[D_r] == y_tr[D_r]).mean()),
                   eval_acc=float((p_tr[D_f] == y_tr[D_f]).mean()),
                   test_acc=float((p_te == y_te).mean()))
        info.update(acc)

        # Table 2 / memorisation sweep: always |S|=1000, their default sigma
        r = protocol.evaluate(bank, D_r, D_f, len(te), n=1000, m=m,
                              sigma_mode="sqrtdim", seed=seed)
        r.update(epoch=info["epoch"], frac=info["frac"], role="sweep", **acc)
        records.append(r)
        log(f"      [sweep ep{info['epoch']:>3}] train_acc={acc['train_acc']:.4f}  F1={r['f1']:.3f}  "
            f"acc={r['accuracy']:.3f}  OTR={r['otr']:.3f}  "
            f"F1_pg={r['f1_pairgrid']:.3f} disagree={r['grid_disagreement']:.2f}  "
            f"sec3.1 p={r['sec31_p']:.3g}")

        # Table 1 + bandwidth contrast, only at the 80% checkpoint
        if table1 and info["epoch"] == ep80:
            for n in sizes:
                for sm in ("sqrtdim", "median"):
                    if n == 1000 and sm == "sqrtdim":
                        rr = dict(r)          # already computed above
                    else:
                        rr = protocol.evaluate(bank, D_r, D_f, len(te), n=n, m=m,
                                               sigma_mode=sm, seed=seed)
                        rr.update(epoch=info["epoch"], frac=info["frac"], **acc)
                    rr["role"] = "table1"
                    records.append(rr)
                    log(f"      [table1] |S|={n:>4} sigma={sm:<8} "
                        f"({rr['sigma']:7.3f})  F1={rr['f1']:.3f}  "
                        f"acc={rr['accuracy']:.3f}  OTR={rr['otr']:.3f}")
        log(f"      ...eval {time.time()-t0:.0f}s")
        return acc

    t0 = time.time()
    model, hist = pipeline.train(dataset, arch, D_r, seed, log=log,
                                 tr_aug=tr_aug, tr_clean=tr_clean, te=te,
                                 eval_idx=D_f, on_checkpoint=on_ckpt,
                                 dense_first=dense_first, ckpt_list=ckpt_eps,
                                 exact_acc=False)
    if save_model:
        import torch
        torch.save({"state": model.state_dict(), "D_r": D_r, "D_f": D_f,
                    "dataset": dataset, "arch": arch, "R": R, "seed": seed},
                   save_model)
        log(f"  saved model -> {os.path.basename(save_model)}")
    final = hist[-1]
    pipeline.compare_table5(dataset, arch, final["train_acc"],
                            final["eval_acc"], final["test_acc"], log=log)

    return dict(
        dataset=dataset, arch=arch, R=R, seed=seed, epochs=epochs, tap=tap,
        m=m, sizes=list(sizes), ep80=ep80,
        wall_seconds=time.time() - t0,
        history=[{k: v for k, v in h.items()} for h in hist],
        records=records,
        table5_paper=pipeline.TABLE5.get((dataset, arch)),
        table7_paper=pipeline.TABLE7.get((dataset, arch)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                    choices=["cifar10", "cifar100", "svhn"])
    ap.add_argument("--arch", default="allcnn", choices=["allcnn", "resnet18"])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sizes", type=int, nargs="+", default=[400, 1000, 2000])
    ap.add_argument("--m", type=int, default=25, help="subsets per class (2m total)")
    ap.add_argument("--tap", default="hp")
    ap.add_argument("--dense-first", type=int, default=10,
                    help="checkpoint every epoch up to this one (memorisation sweep)")
    ap.add_argument("--ckpt-list", type=int, nargs="+", default=None,
                    help="explicit checkpoint epochs, overrides --dense-first")
    ap.add_argument("--no-table1", action="store_true",
                    help="skip the |S| x sigma grid; sweep only")
    ap.add_argument("--save-models", action="store_true",
                    help="save the final model per run (needed for step 3)")
    ap.add_argument("--tag", default="", help="suffix for result filenames")
    ap.add_argument("--quick", action="store_true",
                    help="m=6, sizes=[1000], dense-first=4 -- wiring check only")
    args = ap.parse_args()

    if args.quick:
        args.m, args.sizes, args.dense_first = 6, [1000], 4

    os.makedirs(RESULTS, exist_ok=True)
    if args.save_models:
        os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    data = pipeline.get_data(args.dataset)
    print(f"device={pipeline.DEV}  results -> {RESULTS}")

    todo = [(R, s) for R in args.ratios for s in args.seeds]
    for i, (R, seed) in enumerate(todo, 1):
        tag = (f"{args.dataset}-{args.arch}-R{int(round(R*100)):02d}"
               f"-s{seed}-m{args.m}" + args.tag + ("-quick" if args.quick else ""))
        path = os.path.join(RESULTS, tag + ".json")
        if os.path.exists(path):
            print(f"[{i}/{len(todo)}] skip {tag} (exists)")
            continue
        print(f"[{i}/{len(todo)}] {tag}")
        lines = []

        def log(*a):
            s = " ".join(str(x) for x in a)
            print(s, flush=True)
            lines.append(s)

        mpath = (os.path.join(HERE, "models", tag + ".pt")
                 if args.save_models else None)
        out = run_model(args.dataset, args.arch, R, seed, args.sizes, args.m,
                        args.tap, args.dense_first, data, log,
                        ckpt_list=args.ckpt_list, table1=not args.no_table1,
                        save_model=mpath)
        out["log"] = lines
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> {path}  ({out['wall_seconds']/60:.1f} min)")


if __name__ == "__main__":
    main()
