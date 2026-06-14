"""
Plot a summary of the products from a LAMPrey simulation: how many were made by
each mechanism, and their length distribution.

    python tools/plot_product_summary.py demo_result.pkl --out product_summary.png
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_results, save_fig

_KIND_COLOR = {
    "inter": "#1b9e77",
    "intra": "#d95f02",
    "loop": "#7570b3",
    "product": "#e7298a",
}


def plot_product_summary(result_path, out="product_summary.png"):
    res = load_results(result_path)[0]
    reads = res.get("reads", [])
    if not reads:
        raise ValueError("result has no 'reads' to summarize")

    kinds = Counter(r.get("kind", "inter") for r in reads)
    lengths = [len(r.get("sequence", "")) for r in reads]
    total_nt = res.get("total_nt", [0])[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    order = [k for k in _KIND_COLOR if k in kinds] + [k for k in kinds if k not in _KIND_COLOR]
    ax1.bar(order, [kinds[k] for k in order],
            color=[_KIND_COLOR.get(k, "#888888") for k in order])
    ax1.set_ylabel("number of products")
    ax1.set_xlabel("mechanism")
    ax1.set_title("Products by mechanism")
    for i, k in enumerate(order):
        ax1.text(i, kinds[k], str(kinds[k]), ha="center", va="bottom", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(lengths, bins=min(40, max(8, len(set(lengths)))), color="#2b6777", alpha=0.85)
    ax2.set_xlabel("product length (nt)")
    ax2.set_ylabel("count")
    ax2.set_title("Product-length distribution")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{len(reads)} products  •  {total_nt:.0f} nt synthesized", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", help="pickled result file")
    ap.add_argument("--out", default="product_summary.png")
    args = ap.parse_args()
    print(f"wrote {plot_product_summary(args.results, args.out)}")


if __name__ == "__main__":
    main()
