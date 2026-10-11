#!/usr/bin/env python3
"""Does dropping augmentation reproduce their training trajectory?

Our models reach 0.64-0.76 train accuracy at epoch 5 and do not pass 99% until
epochs 20-40. Their Table 7 reports 92.77% at epoch 5 and 99.21% at epoch 10.
The paper has guessed that the difference is the augmentation we added, which
their Appendix D never mentions. This tests the guess directly instead: same
recipe, same split, augmentation off.

Their Table 5 final test accuracy is 91.63%. If a no-augmentation run matches
their early trajectory but lands well below their final test accuracy, the
augmentation hypothesis does not explain both at once, and we say so.

Writes results_noaug_s<seed>.json.
"""
import argparse, json, os, time
import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE7 = {5: 0.9277, 10: 0.9921, 20: 1.0000, 40: 1.0000}   # their CIFAR-10 All-CNN
TABLE5_TEST = 0.9163


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=float, default=0.05)
    args = ap.parse_args()

    tr_aug, tr_clean, te = pipeline.get_data("cifar10")
    D_r, D_f = pipeline.forget_split(len(tr_clean), args.R, args.seed)

    print("=" * 78)
    print(f"NO-AUGMENTATION TRAJECTORY TEST   seed={args.seed}  R={args.R:.0%}")
    print("  training on the clean (un-augmented) tensors, recipe otherwise identical")
    print("=" * 78)

    rows, t0 = [], time.time()

    def on_ckpt(model, info):
        ep = info.get("epoch")
        row = dict(epoch=ep, train_acc=info.get("train_acc"),
                   eval_acc=info.get("eval_acc"), test_acc=info.get("test_acc"))
        rows.append(row)
        theirs = TABLE7.get(ep)
        tail = f"   theirs {theirs:.4f}" if theirs else ""
        print(f"  ep{ep:>3}  train {row['train_acc']:.4f}  "
              f"test {row['test_acc']:.4f}{tail}")

    # train on the CLEAN tensors: no RandomCrop, no horizontal flip
    pipeline.train("cifar10", "allcnn", D_r, args.seed,
                   tr_aug=tr_clean, tr_clean=tr_clean, te=te, eval_idx=D_f,
                   ckpt_list=[5, 10, 20, 40, 50], exact_acc=True,
                   log=lambda *a: None, on_checkpoint=on_ckpt)

    out = os.path.join(HERE, f"results_noaug_s{args.seed}.json")
    json.dump(dict(seed=args.seed, R=args.R, augmentation=False,
                   their_table7=TABLE7, their_table5_test=TABLE5_TEST,
                   wall_seconds=round(time.time() - t0, 1), rows=rows),
              open(out, "w"), indent=1)

    print("-" * 78)
    ep5 = next((r for r in rows if r["epoch"] == 5), None)
    ep10 = next((r for r in rows if r["epoch"] == 10), None)
    fin = rows[-1] if rows else None
    if ep5:
        print(f"  epoch 5 : ours {ep5['train_acc']:.4f}  theirs {TABLE7[5]:.4f}  "
              f"gap {ep5['train_acc']-TABLE7[5]:+.4f}")
    if ep10:
        print(f"  epoch 10: ours {ep10['train_acc']:.4f}  theirs {TABLE7[10]:.4f}  "
              f"gap {ep10['train_acc']-TABLE7[10]:+.4f}")
    if fin:
        print(f"  final test: ours {fin['test_acc']:.4f}  "
              f"their Table 5 {TABLE5_TEST:.4f}  gap {fin['test_acc']-TABLE5_TEST:+.4f}")
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
