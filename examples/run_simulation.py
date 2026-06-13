"""
Minimal end-to-end LAMPrey example.

Loads a primer set (the bundled synthetic demo set by default), runs one
template-free SSA replicate, prints a summary, and pickles the result so the
visualization tools in ``tools/`` can consume it.

    python examples/run_simulation.py
    python examples/run_simulation.py --primers mydb.csv --name MySet --t-end 120 --seed 7
"""
from __future__ import annotations

import os
import sys
import argparse
import pickle

# Run from the repo root without an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lamprey import simulate_replicate_unified, DEFAULT_PARAMETERS
from tools.load_primers import load_primers

_DEFAULT_DB = os.path.join(_REPO_ROOT, "examples", "data", "demo_primers.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one LAMPrey template-free replicate.")
    ap.add_argument("--primers", default=_DEFAULT_DB, help="CSV/TSV primer file")
    ap.add_argument("--name", default=None, help="primer-set name (for multi-row files)")
    ap.add_argument("--t-end", type=float, default=90.0, help="simulation end time (s)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "examples", "demo_result.pkl"))
    args = ap.parse_args()

    primer_seqs = load_primers(args.primers, args.name)
    print(f"Loaded {len([k for k in primer_seqs if not k.endswith('_RC')])} primers "
          f"from {os.path.basename(args.primers)}")

    result = simulate_replicate_unified(
        primer_seqs=primer_seqs,
        params=DEFAULT_PARAMETERS,
        t_end=args.t_end,
        rng_seed=args.seed,
    )

    print(f"  products synthesized : {result['n_products']}")
    print(f"  distinct species     : {result.get('n_species', 'n/a')}")
    print(f"  final nucleotides    : {result['total_nt'][-1]:.0f}")
    print(f"  stop reason          : {result.get('stop_reason', 'n/a')}")

    with open(args.out, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nResult pickled to {args.out}")
    print("Visualize with, e.g.:")
    print(f"  python tools/plot_amplification.py {args.out}")
    print(f"  python tools/plot_lineage.py {args.out}")
    print(f"  python tools/plot_autocat_network.py {args.out}")


if __name__ == "__main__":
    main()
