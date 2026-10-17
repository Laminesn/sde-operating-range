#!/usr/bin/env python3
"""Figure 1 for the workshop paper: the operating range, two ways.

Left  : SDE accuracy against train accuracy, one line per model (15 All-CNN),
        with the ResNet-18 run overlaid and their earliest reported checkpoint
        marked. Replaces the band table.
Right : the same verdicts against the train-test gap, which is the axis the
        paper argues is the better proxy for memorisation.

Writes figure_range.pdf and .png next to the paper.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.family": "serif",
                            "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"]})
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper")

rows = [r for f in sorted(glob.glob(os.path.join(HERE, "results",
        "cifar10-allcnn-R*-s*-m25-dense.json")))
        for r in json.load(open(f))["records"] if r.get("role") == "sweep"]
assert len(rows) == 525, len(rows)

# group by (R, seed) -> one training run
runs = {}
for f in sorted(glob.glob(os.path.join(HERE, "results",
                "cifar10-allcnn-R*-s*-m25-dense.json"))):
    d = json.load(open(f))
    rs = [r for r in d["records"] if r.get("role") == "sweep"]
    runs[(d["R"], d["seed"])] = sorted(rs, key=lambda r: r["train_acc"])

r18 = None
p18 = os.path.join(HERE, "results_r18_range_cifar10-s0.json")
if os.path.exists(p18):
    r18 = sorted(json.load(open(p18))["records"], key=lambda r: r["train_acc"])

import sys
SINGLE = "--single" in sys.argv
if SINGLE:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 2.5)); ax2 = None
else:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3))

for (R, s), rs in runs.items():
    ax.plot([r["train_acc"] for r in rs], [r["accuracy"] for r in rs],
            color="0.72", lw=0.8, alpha=0.8, zorder=1)
ax.plot([], [], color="0.72", lw=0.8, label="All-CNN, 15 runs")

# binned mean over all All-CNN checkpoints
edges = [0.4, 0.75, 0.85, 0.90, 0.93, 0.96, 0.98, 0.99, 0.995, 0.999, 1.001]
xs, ys = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    b = [r["accuracy"] for r in rows if lo <= r["train_acc"] < hi]
    if b:
        xs.append((lo + hi) / 2); ys.append(np.mean(b))
ax.plot(xs, ys, color="#1f4e79", lw=2.2, marker="o", ms=4,
        label="All-CNN, band mean", zorder=3)

if r18:
    ax.plot([r["train_acc"] for r in r18], [r["accuracy"] for r in r18],
            color="#c0392b", lw=2.0, marker="s", ms=4,
            label="ResNet-18, 1 run", zorder=4)

ax.axhline(0.5, color="k", ls=":", lw=1.0, zorder=2)
ax.text(0.408, 0.455, "chance", fontsize=7, color="k")
ax.axvline(0.9277, color="#2e7d32", ls="--", lw=1.1, zorder=2)
ax.text(0.9255, 0.055, "their earliest\nreported checkpoint", fontsize=6.5,
        color="#2e7d32", ha="right")

ax.set_xlabel("train accuracy")
ax.set_ylabel("SDE accuracy")
ax.set_xlim(0.40, 1.005)
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=7, loc="lower left", framealpha=0.9, borderpad=0.4)
ax.set_title("" if SINGLE else "a. against memorisation of $D_r$", fontsize=9, loc="left")

if ax2 is None:
    # inset over the regime the whole result lives in
    fig.tight_layout()
    axi = ax.inset_axes([0.10, 0.56, 0.32, 0.40])
    for (R, s_), rs in runs.items():
        axi.plot([r["train_acc"] for r in rs], [r["accuracy"] for r in rs],
                 color="0.72", lw=0.7, alpha=0.8)
    axi.plot(xs, ys, color="#1f4e79", lw=1.8, marker="o", ms=3)
    if r18:
        axi.plot([r["train_acc"] for r in r18], [r["accuracy"] for r in r18],
                 color="#c0392b", lw=1.6, marker="s", ms=3)
    axi.axhline(0.5, color="k", ls=":", lw=0.9)
    axi.set_xlim(0.985, 1.0008); axi.set_ylim(0.3, 1.05)
    axi.set_xticks([0.99, 1.0]); axi.set_yticks([0.5, 1.0])
    axi.tick_params(labelsize=6)
    axi.set_title("the working regime", fontsize=6.5, pad=1.5)
    for sp in axi.spines.values():
        sp.set_linewidth(0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figure_range_single.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote figure_range_single.pdf"); raise SystemExit

gap = np.array([r["train_acc"] - r["test_acc"] for r in rows])
acc = np.array([r["accuracy"] for r in rows])
ax2.scatter(gap, acc, s=7, color="0.55", alpha=0.6, edgecolors="none")
ge = np.quantile(gap, np.linspace(0, 1, 9))
gx, gy = [], []
for lo, hi in zip(ge[:-1], ge[1:]):
    m = (gap >= lo) & (gap < hi)
    if m.sum():
        gx.append(gap[m].mean()); gy.append(acc[m].mean())
ax2.plot(gx, gy, color="#1f4e79", lw=2.2, marker="o", ms=4)
ax2.axhline(0.5, color="k", ls=":", lw=1.0)
ax2.set_xlabel("train $-$ test accuracy")
ax2.set_ylabel("SDE accuracy")
ax2.set_ylim(0.0, 1.05)
ax2.set_title(f"b. against the generalisation gap "
              f"($r={np.corrcoef(gap, acc)[0,1]:.2f}$ vs "
              f"${np.corrcoef([r['train_acc'] for r in rows], acc)[0,1]:.2f}$)",
              fontsize=9, loc="left")

for a in [x for x in (ax, ax2) if x is not None]:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure_range.{ext}"), dpi=200,
                bbox_inches="tight")
print("wrote", os.path.join(OUT, "figure_range.pdf"))
