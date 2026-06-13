"""
Plot the autocatalytic segment-transition network of a LAMPrey simulation.

Each product read is segmented into primer-family blocks (F3, BIP_RC, ...). This
tool aggregates the directed transitions between consecutive segments across all
reads into a network: nodes are primer-family segments (sized by how often they
occur), edges are segment-to-segment transitions (width = frequency). Self-loops
(X -> X_RC) correspond to hairpin turnbacks; longer cycles indicate the
multi-primer autocatalytic cascades characteristic of true LAMP amplification.

    python tools/plot_autocat_network.py demo_result.pkl --out autocat_network.png
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from _common import load_results, save_fig


def build_transition_counts(reads):
    node_counts = Counter()
    edge_counts = Counter()
    for r in reads:
        labels = [s.label for s in r.get("segments", []) if getattr(s, "label", None)]
        for lab in labels:
            node_counts[lab] += 1
        for a, b in zip(labels, labels[1:]):
            edge_counts[(a, b)] += 1
    return node_counts, edge_counts


def plot_autocat_network(result_path, out="autocat_network.png", min_edge=1, seed=42):
    res = load_results(result_path)[0]
    reads = res.get("reads", [])
    if not reads:
        raise ValueError("result has no 'reads' to build a network from")

    node_counts, edge_counts = build_transition_counts(reads)
    if not edge_counts:
        raise ValueError("no segment transitions found (products may be single-segment)")

    G = nx.DiGraph()
    for (a, b), w in edge_counts.items():
        if w >= min_edge:
            G.add_edge(a, b, weight=w)
    if G.number_of_nodes() == 0:
        raise ValueError(f"no transitions with weight >= {min_edge}")

    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=seed, k=0.9)

    n_sizes = [120 + 40 * node_counts.get(n, 1) ** 0.5 for n in G.nodes()]
    e_widths = [0.4 + 2.2 * (G[u][v]["weight"] / max(edge_counts.values())) for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=n_sizes, node_color="#cfe8ef",
                           edgecolors="#2b6777", linewidths=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=e_widths, edge_color="#2b6777",
                           alpha=0.6, arrowsize=10, connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f"Autocatalytic segment-transition network\n"
                 f"({G.number_of_nodes()} families, {G.number_of_edges()} transitions)")
    ax.axis("off")
    return save_fig(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", help="pickled result file")
    ap.add_argument("--out", default="autocat_network.png")
    ap.add_argument("--min-edge", type=int, default=1, help="drop transitions below this count")
    args = ap.parse_args()
    print(f"wrote {plot_autocat_network(args.results, args.out, args.min_edge)}")


if __name__ == "__main__":
    main()
