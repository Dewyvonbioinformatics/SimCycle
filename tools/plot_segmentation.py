"""
Plot how representative product sequences are segmented into primer-family blocks.

The segmenter labels each region of a product with the primer family it derives
from (F3, BIP_RC, ...). This view shows several products as horizontal tracks of
colored, labeled segments along the nucleotide axis — the per-read counterpart to
the family-level transition network.

    python tools/plot_segmentation.py demo_result.pkl --out segmentation.png --n 6
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _common import load_results, save_fig

_PALETTE = plt.get_cmap("tab20").colors


def _color_map(labels):
    cyc = itertools.cycle(_PALETTE)
    return {lab: next(cyc) for lab in sorted(labels)}


def plot_segmentation(result_path, out="segmentation.png", n=6):
    res = load_results(result_path)[0]
    reads = res.get("reads", [])
    if not reads:
        raise ValueError("result has no 'reads' to segment")

    # Most informative products first: most segments, then longest.
    reads = sorted(
        reads,
        key=lambda r: (len(r.get("segments", [])), len(r.get("sequence", ""))),
        reverse=True,
    )[:n]

    all_labels = {s.label for r in reads for s in r.get("segments", []) if getattr(s, "label", None)}
    cmap = _color_map(all_labels)

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(reads) + 1.5))
    seen = {}
    for row, r in enumerate(reads):
        y = len(reads) - row - 1
        for s in r.get("segments", []):
            w = s.end - s.start
            color = cmap.get(s.label, "#cccccc")
            ax.add_patch(Rectangle((s.start, y - 0.4), w, 0.8,
                                   facecolor=color, edgecolor="white", linewidth=0.5))
            if w >= max(6, 0.04 * len(r.get("sequence", "") or "x")):
                ax.text(s.start + w / 2, y, s.label, ha="center", va="center",
                        fontsize=7, color="black")
            seen.setdefault(s.label, color)

    ax.set_xlim(0, max((len(r.get("sequence", "")) for r in reads), default=1) * 1.02)
    ax.set_ylim(-0.6, len(reads) - 0.4)
    ax.set_yticks(range(len(reads)))
    ax.set_yticklabels([f"{r.get('kind','?')}  ({len(r.get('sequence',''))} nt)"
                        for r in reversed(reads)], fontsize=8)
    ax.set_xlabel("nucleotide position")
    ax.set_title(f"Product segmentation (top {len(reads)} products by segment count)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in seen.values()]
    ax.legend(handles, list(seen.keys()), fontsize=7, ncol=min(6, len(seen)),
              loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    return save_fig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", help="pickled result file")
    ap.add_argument("--out", default="segmentation.png")
    ap.add_argument("--n", type=int, default=6, help="number of products to show")
    args = ap.parse_args()
    print(f"wrote {plot_segmentation(args.results, args.out, args.n)}")


if __name__ == "__main__":
    main()
