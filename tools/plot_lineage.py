"""
Plot the product genealogy (lineage) of a LAMPrey simulation.

Every emitted product carries its own ``id`` and the ``template_id`` it was
synthesized from, which together form a directed genealogy. Each product is
placed at (emission time, product length) and an edge is drawn from its parent
template to the product, colored by the binding mechanism that created it
(inter / intra / loop / product-as-primer).

    python tools/plot_lineage.py demo_result.pkl --out lineage.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import load_results, save_fig

_KIND_COLOR = {
    "inter": "#1b9e77",
    "intra": "#d95f02",
    "loop": "#7570b3",
    "product": "#e7298a",
}


def plot_lineage(result_path, out="lineage.png"):
    res = load_results(result_path)[0]
    reads = res.get("reads", [])
    if not reads:
        raise ValueError("result has no 'reads' to build a lineage from")

    # node id -> (x=time, y=length, kind)
    pos = {}
    for r in reads:
        nid = r.get("id")
        if nid is None:
            continue
        pos[nid] = (
            float(r.get("time_emitted", 0.0)),
            float(len(r.get("sequence", "")) or 0),
            r.get("kind", "inter"),
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    # edges: parent template_id -> child id (only when parent is a known node)
    for r in reads:
        child, parent = r.get("id"), r.get("template_id")
        if child in pos and parent in pos:
            x0, y0, _ = pos[parent]
            x1, y1, _ = pos[child]
            ax.plot([x0, x1], [y0, y1], color="0.8", lw=0.6, zorder=1)

    for kind, color in _KIND_COLOR.items():
        xs = [p[0] for p in pos.values() if p[2] == kind]
        ys = [p[1] for p in pos.values() if p[2] == kind]
        if xs:
            ax.scatter(xs, ys, s=18, c=color, label=kind, zorder=2, edgecolors="none")

    ax.set_xlabel("emission time (s)")
    ax.set_ylabel("product length (nt)")
    ax.set_title(f"Product lineage  ({len(pos)} products)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in _KIND_COLOR.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False, title="mechanism")
    return save_fig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", help="pickled result file")
    ap.add_argument("--out", default="lineage.png")
    args = ap.parse_args()
    print(f"wrote {plot_lineage(args.results, args.out)}")


if __name__ == "__main__":
    main()
