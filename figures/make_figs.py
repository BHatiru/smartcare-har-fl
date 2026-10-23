"""Generate HAR plots from the archived experiment artifacts in results/.

Rerun after refreshing results:  python figures/make_figs.py
Reads:  results/*.json / *.csv
Writes: figures/*.pdf  (sized for a 3.5-inch column)
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RES = ROOT / "results"

# Okabe-Ito subset — colorblind-safe
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
GRAY = "#666666"

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "figure.dpi": 150,
})


def confusion_2d():
    rows = list(csv.reader(open(RES / "cent_2d_15cls_confusion_matrix.csv")))
    labels = rows[0][1:]
    m = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    norm = m / m.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.grid(False)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(15), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(15), labels, fontsize=6)
    for i in range(15):
        for j in range(15):
            v = norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=4.5, color="white" if v > 0.5 else "#333333")
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("True", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "confusion_2d.pdf")
    plt.close(fig)


def fl_baselines():
    d = json.load(open(RES / "phase1_baselines.json"))
    series = [("Global averaging", "fedavg", BLUE),
              ("Partial-BN exclusion", "fedbn", ORANGE),
              ("Cluster-balanced averaging", "cluster_fedbn", GREEN)]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for name, key, c in series:
        h = d[key]["sampled_history"]
        r = [p["round"] for p in h]
        a = [100 * p["test_acc"] for p in h]
        ax.plot(r, a, color=c, lw=1.6, marker="o", ms=3, label=name)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Pooled test accuracy (%)")
    ax.set_title("10-class HAR; pooled BN calibration", fontsize=8)
    ax.set_xlim(0, 52)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fl_baselines.pdf")
    plt.close(fig)


def perclass_2d3d():
    accs = {}
    for tag in ("2d", "3d"):
        d = json.load(open(RES / f"cent_{tag}_15cls_results.json"))
        accs[tag] = [d["per_class_acc"][str(i)]["accuracy"] * 100 for i in range(15)]
        labels = [d["label_names"][i] for i in range(15)]
    y = np.arange(15)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.barh(y - 0.2, accs["2d"], 0.38, color=BLUE, label="2D (COCO-17)")
    ax.barh(y + 0.2, accs["3d"], 0.38, color=ORANGE, label="3D (NTU-25)")
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class accuracy (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7, ncols=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "perclass_2d3d.pdf")
    plt.close(fig)


if __name__ == "__main__":
    confusion_2d()
    fl_baselines()
    perclass_2d3d()
    print("wrote", *[p.name for p in sorted(OUT.glob("*.pdf"))])
