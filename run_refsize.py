#!/usr/bin/env python3
"""Decouple reference size from target size.

Section 9 notes that H(S,h) is computed on |S|/2 x |S|/2 kernel blocks, so its
location and spread depend on |S| -- and that matching the reference size to the
target, as we and their Sec. 4.3 do, confounds the headline |S| trend with a
reference-size effect. This runs the 2x2 that settles it on the saved models.
"""
import itertools, json, os, statistics as st, time
import torch
import arch, pipeline, protocol

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [400, 2000]

tr_aug, tr_clean, te = pipeline.get_data("cifar10")
rc = pipeline.RECIPE[("cifar10", "allcnn")]
rows, t0 = [], time.time()
print(f"{'|S|':>6}{'|S_ref|':>9}{'F1':>10}{'accuracy':>11}{'n models':>10}")
print("-" * 46)
cells = {}
for s in range(5):
    b = torch.load(os.path.join(HERE, "models",
                   f"cifar10-allcnn-R05-s{s}-m25-dense.pt"),
                   map_location="cpu", weights_only=False)
    model = arch.build("allcnn", n_cls=rc["n_cls"]).to(pipeline.DEV)
    model.load_state_dict(b["state"]); model.eval()
    bank = protocol.ActBank(model, tr_clean, te, tap="hp")
    for n, n_ref in itertools.product(SIZES, SIZES):
        r = protocol.evaluate(bank, b["D_r"], b["D_f"], len(te),
                              n=n, m=25, seed=s, n_ref=n_ref)
        r.update(n=n, n_ref=n_ref, seed=s)
        rows.append(r)
        cells.setdefault((n, n_ref), []).append(r)
    del bank, model

for (n, n_ref), v in sorted(cells.items()):
    print(f"{n:>6}{n_ref:>9}{st.mean(r['f1'] for r in v):>10.3f}"
          f"{st.mean(r['accuracy'] for r in v):>11.3f}{len(v):>10}")

f = lambda n, nr: st.mean(r["f1"] for r in cells[(n, nr)])
print(f"\ntarget effect at fixed reference 400 : {f(2000,400) - f(400,400):+.3f}")
print(f"target effect at fixed reference 2000: {f(2000,2000) - f(400,2000):+.3f}")
print(f"reference effect at fixed target 400 : {f(400,2000) - f(400,400):+.3f}")
print(f"reference effect at fixed target 2000: {f(2000,2000) - f(2000,400):+.3f}")
json.dump(dict(sizes=SIZES, m=25, R=0.05, wall_seconds=round(time.time()-t0,1),
               records=rows), open(os.path.join(HERE, "results_refsize.json"), "w"), indent=1)
print(f"\nwrote results_refsize.json ({(time.time()-t0)/60:.1f} min)")
