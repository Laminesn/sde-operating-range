"""
CIFAR-100 vs CIFAR-10: does M4's membership contrast track the generalisation gap?

Pre-registered in phase4-family-scope/PREREGISTRATION-cifar100.md. Read that
first -- the predictions, the decision rules and the effective-n statement are
fixed there and must not be relitigated here.

WHAT THIS RUNS
  One theta_r per (dataset, seed), 100 epochs, checkpointed at 5/10/20/40/60/80/
  100%. At every checkpoint it measures the train-test gap and M4's membership
  contrast on the SAME model, so the gap is varied without changing dataset,
  class count, architecture, |D_r| or recipe.

  No unlearning anywhere. Every model is a retrain oracle.

WHY A TRAJECTORY AND NOT TWO DATASETS
  The gap is the causal variable under test and CIFAR-10 cannot reach the
  interesting range at any architecture (~6.5 pp ceiling). CIFAR-100 traverses
  6.0 -> 26.6 pp along a single run (SDE Table 7, both rows). Two comparisons in
  the design are NOT collinear -- corr(gap, train acc) is 0.934 elsewhere:
    * at 40%, CIFAR-100 is LESS memorised (81.99 vs 91.50) with a 4x LARGER gap
      (24.85 vs 6.34), so gap and memorisation predict opposite orderings;
    * within CIFAR-100, 40% -> 80% moves the gap +1.7 pp while train accuracy
      moves +17.9 pp.
  Everything else on the trajectory is a consistency check, not evidence.

THE FIVE LANDMINES THIS FILE AVOIDS ON PURPOSE
  1. forget_split defaults to `legacy`, which NESTS across ratios. Every call
     here passes scheme="independent" (Revision 11). run_pilot.py:137 does NOT,
     so do not copy it.
  2. run_pilot.ckpt_schedule returns [10,20,40,80,100] and would silently drop
     the 5% and 60% checkpoints. The schedule here is explicit.
  3. run_m2gate5.py hardcodes ckpt_list=[50] and "allcnn". Nothing here is
     hardcoded to an architecture or an epoch count.
  4. The reference/member halves are drawn ONCE, outside the checkpoint loop,
     from a fresh Generator, and asserted identical at every checkpoint. Drawing
     them inside would silently differ per checkpoint, and pipeline.train's RNG
     snapshot/restore would MASK the symptom.
  5. m4() at large caps materialises R@R.T; at cap 20000 that peaked at 7.6 GB
     in the audit. Nothing here exceeds cap 2000. The Run C geometry sweep keeps
     |R| <= 10000 for the same reason.

SAFE, VERIFIED, DO NOT RE-CHECK
  pipeline.train snapshots and restores the global torch/numpy RNG around the
  whole checkpoint block INCLUDING the on_checkpoint call, so computing M4 there
  cannot perturb training. CosineAnnealingLR is stepped before the block and is
  untouched by it. activations() sets eval() and train() restores train() after.

Usage:
  python3 run_c100.py --dataset cifar100 --seed 0
  python3 run_c100.py --dataset cifar10  --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import arch
import metrics_phase4 as M
import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "c100_models")
ARCH, R = "resnet18", 0.05
CKPTS = [5, 10, 20, 40, 60, 80, 100]
SCHEME = "independent"

# Pre-registered anchors. RECORDED, not enforced (Revision 9: our augmented
# recipe lands high, so a miss is informative rather than disqualifying).
# NOTE: SDE Table 5 and Table 7 are R=10% models; this design is R=5%.
NOISE_DRAWS = 5


def halves(D_r, seed):
    """The member/reference split. Drawn ONCE per model -- see landmine 4."""
    rng = np.random.default_rng(7000 + seed)
    p = rng.permutation(len(D_r))
    return D_r[p[:len(D_r) // 2]].copy(), D_r[p[len(D_r) // 2:]].copy()


def measure(model, tr_clean, te, ref_h, mem_h, D_f):
    """Every M4 quantity at one checkpoint. ~14 s, dominated by activations."""
    A_ref = pipeline.activations(model, tr_clean, ref_h, tap="hp")
    A_mem = pipeline.activations(model, tr_clean, mem_h, tap="hp")
    A_f = pipeline.activations(model, tr_clean, D_f, tap="hp")
    A_te = pipeline.activations(model, te, np.arange(len(te)), tap="hp")

    member = M.m4(A_mem, A_ref)
    non_te = M.m4(A_te, A_ref)
    non_df = M.m4(A_f, A_ref)

    # Noise floor of the CONTRAST, not of M4 alone. Both ends share the same
    # reference subsample, so most of the cap noise cancels -- the audit measured
    # 0.0011 for the contrast against 0.0038 for M4(D_f|D_r) on its own.
    draws = [M.m4(A_mem, A_ref, seed=M.M4_RANDOM_STATE + k)
             - M.m4(A_te, A_ref, seed=M.M4_RANDOM_STATE + k)
             for k in range(NOISE_DRAWS)]
    return dict(m4_member=member, m4_nonmember_test=non_te,
                m4_nonmember_df=non_df,
                contrast=member - non_te,
                contrast_noise=float(np.std(draws, ddof=1)),
                m4_dim=int(A_ref.shape[1]))


def anchor_row(dataset, ep, epochs):
    """SDE Table 7 train/test/gap at this checkpoint, if it is one of theirs."""
    ref = pipeline.TABLE7.get((dataset, ARCH))
    if not ref:
        return None
    for frac, tr, te in zip(pipeline.CKPT_FRACS, ref["train"], ref["test"]):
        if max(1, int(round(frac * epochs))) == ep:
            return dict(frac=frac, train=tr, test=te, gap=tr - te)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar100", "cifar10"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true",
                    help="persist per-checkpoint weights for the Run C "
                         "matched-geometry re-scoring (seed 0 only, ~315 MB)")
    ap.add_argument("--smoke", action="store_true",
                    help="2-epoch wiring check. Writes NOTHING and caches "
                         "NOTHING -- the shortened schedule changes every "
                         "learning rate, so its numbers are not comparable.")
    args = ap.parse_args()
    ds, seed = args.dataset, args.seed
    rc = pipeline.RECIPE[(ds, ARCH)]

    global CKPTS
    if args.smoke:
        rc = {**rc, "epochs": 2}
        pipeline.RECIPE[(ds, ARCH)] = rc
        CKPTS = [1, 2]
        args.save_ckpt = False
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{ds}-s{seed}"
    out_path = os.path.join(HERE, f"results_c100_{tag}.json")

    print("=" * 78)
    print(f"GAP AXIS  {ds}/{ARCH}  R={R:.0%}  seed={seed}  scheme={SCHEME}")
    print(f"recipe {rc}  device={pipeline.DEV}  checkpoints {CKPTS}")
    print("=" * 78)

    assert M.selftest(verbose=False), "metrics selftest failed"
    arch.check_dims(ARCH, n_cls=rc["n_cls"])
    assert pipeline.forget_split_selftest(log=lambda *a: None), "split schemes wrong"
    print("  gates: metrics PASS, arch dims PASS, split schemes PASS")

    tr_aug, tr_clean, te = pipeline.get_data(ds)
    D_r, D_f = pipeline.forget_split(len(tr_clean), R, seed, scheme=SCHEME)
    ref_h, mem_h = halves(D_r, seed)
    assert not set(ref_h) & set(mem_h), "halves overlap"
    assert not set(D_r) & set(D_f), "D_r and D_f overlap"
    halves_fingerprint = (int(ref_h.sum()), int(mem_h.sum()))
    print(f"  |D_r|={len(D_r)}  |D_f|={len(D_f)}  "
          f"halves {len(ref_h)}/{len(mem_h)}  fp={halves_fingerprint}")

    model_path = os.path.join(CACHE, tag + ".pt")
    rows, lines = [], []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    def on_checkpoint(model, info):
        # landmine 4: the halves must be the same object at every checkpoint
        assert (int(ref_h.sum()), int(mem_h.sum())) == halves_fingerprint, \
            "reference/member halves changed between checkpoints"
        t = time.time()
        m = measure(model, tr_clean, te, ref_h, mem_h, D_f)
        gap = 100 * (info["train_acc"] - info["test_acc"])
        anc = anchor_row(ds, info["epoch"], rc["epochs"])
        row = dict(epoch=info["epoch"], frac=info["epoch"] / rc["epochs"],
                   train_acc=info["train_acc"], test_acc=info["test_acc"],
                   forget_acc=info["eval_acc"], gap_pp=gap, anchor=anc, **m)
        rows.append(row)
        if args.save_ckpt:
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()},
                       os.path.join(CACHE, f"{tag}-ep{info['epoch']}.pt"))
        a_txt = (f"  [SDE T7 R=10%: tr {anc['train']:.2f} te {anc['test']:.2f} "
                 f"gap {anc['gap']:.2f}]" if anc else "")
        log(f"    ep{info['epoch']:>3}  train {100*info['train_acc']:6.2f}  "
            f"test {100*info['test_acc']:6.2f}  gap {gap:6.2f} pp   "
            f"M4 mem {m['m4_member']:.4f} non {m['m4_nonmember_test']:.4f}  "
            f"contrast {m['contrast']:+.4f} +-{m['contrast_noise']:.4f}"
            f"   [{time.time()-t:.0f}s]{a_txt}")
        return {}

    t0 = time.time()
    if os.path.exists(out_path) and not args.smoke:
        print(f"\n  {out_path} exists -- delete it to re-run. Exiting.")
        return

    log(f"\ntraining theta_r on D_r  ({rc['epochs']} epochs)")
    model, hist = pipeline.train(
        ds, ARCH, D_r, seed=seed, tr_aug=tr_aug, tr_clean=tr_clean, te=te,
        eval_idx=D_f, ckpt_list=CKPTS, exact_acc=True,
        on_checkpoint=on_checkpoint, log=lambda *a: None)
    torch.save({"state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
                "final": dict(hist[-1]), "tag": tag}, model_path)

    fin = rows[-1]
    log("\n" + "=" * 78)
    log(f"RESULT  {ds}  seed {seed}   ({(time.time()-t0)/60:.1f} min)")
    log("=" * 78)
    log(f"  final   train {100*fin['train_acc']:.2f}  test {100*fin['test_acc']:.2f}"
        f"  gap {fin['gap_pp']:.2f} pp")
    t5 = pipeline.TABLE5.get((ds, ARCH))
    if t5:
        log(f"  SDE Table 5 (R=10%): train {t5[0]:.2f}  test {t5[2]:.2f}  "
            f"gap {t5[0]-t5[2]:.2f} pp   [RECORDED, not enforced]")
    log(f"  M4 member {fin['m4_member']:.4f}  non-member(test) "
        f"{fin['m4_nonmember_test']:.4f}  non-member(D_f) "
        f"{fin['m4_nonmember_df']:.4f}")
    log(f"  CONTRAST  {fin['contrast']:+.4f}  +- {fin['contrast_noise']:.4f} "
        f"(cap noise)")
    log(f"  [C10-ResNet18 pilot baseline: +0.05859 at gap 5.43 pp]")
    log(f"  [P1 interval for CIFAR-100: [0.079, 0.24], point 0.13]")
    log("\n  gap -> contrast trajectory:")
    for r in rows:
        log(f"    {100*r['frac']:>5.0f}%  gap {r['gap_pp']:6.2f} pp   "
            f"contrast {r['contrast']:+.4f}   train {100*r['train_acc']:6.2f}")

    if args.smoke:
        print("\n  SMOKE -- nothing written, nothing cached. Wiring OK.")
        return

    # manifest LAST -- its presence means the run completed
    json.dump(dict(dataset=ds, arch=ARCH, R=R, seed=seed, scheme=SCHEME,
                   recipe=rc, checkpoints=CKPTS, n_retain=len(D_r),
                   n_forget=len(D_f), halves_fingerprint=halves_fingerprint,
                   rows=rows, minutes=(time.time() - t0) / 60,
                   prereg="PREREGISTRATION-cifar100.md", log=lines),
              open(out_path, "w"), indent=1)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
