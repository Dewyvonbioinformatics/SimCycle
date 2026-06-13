"""Shared helpers for LAMPrey tools: repo bootstrap, result loading, figure saving."""
import os
import sys
import pickle
from typing import List, Dict, Any

# Allow tools to be run directly (`python tools/plot_x.py`) from the repo root
# without an editable install, by putting the repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load a pickled LAMPrey result and normalize to a flat list of result dicts.

    Accepts any of:
      - a single result dict (output of ``simulate_replicate_unified``)
      - a list of result dicts (replicates)
      - a dict mapping primer_name -> list-of-results (batch output)
    """
    with open(path, "rb") as fh:
        obj = pickle.load(fh)

    out: List[Dict[str, Any]] = []

    def _add(o: Any) -> None:
        if isinstance(o, dict) and ("total_nt" in o or "reads" in o):
            out.append(o)
        elif isinstance(o, list):
            for x in o:
                _add(x)
        elif isinstance(o, dict):
            for v in o.values():
                _add(v)

    _add(obj)
    if not out:
        raise ValueError(f"No LAMPrey result dicts found in {path!r}")
    return out


def save_fig(fig, out_path: str, dpi: int = 200) -> str:
    """Save a matplotlib figure, creating parent dirs as needed."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
