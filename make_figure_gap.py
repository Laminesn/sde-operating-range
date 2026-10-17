#!/usr/bin/env python3
"""Appendix figure: the same 525 verdicts against the generalisation gap."""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.family": "serif",
                            "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"]})
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "paper")
rows = [r for f in sorted(glob.glob(os.path.join(HERE, "results", "cifar10-allcnn-R*-s*-m25-dense.json")))
        for r in json.load(open(f))["records"] if r.get("role") == "sweep"]
gap = np.array([r["train_acc"] - r["test_acc"] for r in rows]); acc = np.array([r["accuracy"] for r in rows])
ta  = np.array([r["train_acc"] for r in rows])
fig, ax = plt.subplots(figsize=(5.0, 2.4))
ax.scatter(gap, acc, s=7, color="0.55", alpha=0.6, edgecolors="none")
ge = np.quantile(gap, np.linspace(0, 1, 9)); gx, gy = [], []
for lo, hi in zip(ge[:-1], ge[1:]):
    m = (gap >= lo) & (gap < hi)
    if m.sum(): gx.append(gap[m].mean()); gy.append(acc[m].mean())
ax.plot(gx, gy, color="#1f4e79", lw=2.0, marker="o", ms=4)
ax.axhline(0.5, color="k", ls=":", lw=1.0)
ax.set_xlabel("train $-$ test accuracy"); ax.set_ylabel("SDE accuracy"); ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"figure_gap.{e}"), dpi=200, bbox_inches="tight")
print("wrote figure_gap.pdf  r(gap)=%.3f r(train)=%.3f" % (np.corrcoef(gap,acc)[0,1], np.corrcoef(ta,acc)[0,1]))
