"""
Invariant-style smoke tests for the LAMPrey unified engine.

These favor robust properties (determinism, monotonicity, conservation, I/O
round-trip) over brittle recorded golden numbers, so they survive legitimate
parameter retuning.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lamprey import simulate_replicate_unified, DEFAULT_PARAMETERS
from tools.load_primers import load_primers, primers_to_engine_dict

DEMO_CSV = os.path.join(_REPO_ROOT, "examples", "data", "demo_primers.csv")


def _run(seed=42, t_end=8.0):
    primers = load_primers(DEMO_CSV)
    return simulate_replicate_unified(
        primer_seqs=primers, params=DEFAULT_PARAMETERS, t_end=t_end, rng_seed=seed
    )


def test_loader_roundtrip_adds_reverse_complements():
    d = load_primers(DEMO_CSV)
    for role in ("F3", "B3", "FIP", "BIP"):
        assert role in d and d[role].isupper()
        assert d[f"{role}_RC"] == _rc(d[role])


def _rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def test_loader_requires_core_primers():
    with pytest.raises(ValueError):
        primers_to_engine_dict({"F3": "ACGT"})  # missing B3/FIP/BIP


def test_simulation_runs_and_reports():
    res = _run()
    assert res["n_products"] >= 0
    assert len(res["time"]) == len(res["total_nt"])
    assert res["total_nt"][-1] >= 0


def test_amplification_trace_is_monotonic_nondecreasing():
    res = _run()
    y = res["total_nt"]
    assert all(b >= a for a, b in zip(y, y[1:])), "cumulative nt must not decrease"


def test_same_seed_is_reproducible():
    a, b = _run(seed=7), _run(seed=7)
    assert a["n_products"] == b["n_products"]
    assert a["total_nt"][-1] == b["total_nt"][-1]


def test_different_seeds_diverge():
    a, b = _run(seed=1), _run(seed=2)
    # Not a hard guarantee, but these demo seeds should differ in trajectory.
    assert (a["n_products"], a["total_nt"][-1]) != (b["n_products"], b["total_nt"][-1])
