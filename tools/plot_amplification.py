"""
Plot LAMPrey amplification curves (cumulative nucleotides synthesized vs time).

Overlays one or more pickled results. Each result contributes a curve of
``total_nt`` against ``time``; passing several replicates shows the spread.

    python tools/plot_amplification.py demo_result.pkl --out amplification.png
    python tools/plot_amplification.py a.pkl b.pkl --labels noTemplate spikeIn --out cmp.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_results, save_fig


def plot_amplification(result_paths, labels=None, out="amplification.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, path in enumerate(result_paths):
        results = load_results(path)
        base = labels[i] if labels and i < len(labels) else path
        for j, res in enumerate(results):
            t = res.get("time", [])
            y = res.get("total_nt", [])
            if not len(t) or not len(y):
                continue
            lbl = base if len(results) == 1 else f"{base} #{j}"
            ax.plot(t, y, lw=1.5, alpha=0.85, label=lbl)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative nucleotides synthesized")
    ax.set_title("LAMPrey amplification curve")
    ax.spines[["top", "right"]].set_visible(False)
    if ax.has_data():
        ax.legend(fontsize=8, frameon=False)
    return save_fig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", nargs="+", help="one or more pickled result files")
    ap.add_argument("--labels", nargs="*", default=None, help="legend label per results file")
    ap.add_argument("--out", default="amplification.png")
    args = ap.parse_args()
    path = plot_amplification(args.results, args.labels, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
