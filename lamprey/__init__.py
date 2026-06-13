"""
LAMPrey — a mechanistic, template-free LAMP reaction kinetic simulator.

LAMPrey is a Gillespie Stochastic Simulation Algorithm (SSA) engine that predicts
spurious (no-target) amplification products and kinetics of Loop-mediated isothermal
AMPlification (LAMP) reactions directly from primer sequences.

Public API
----------
    from lamprey import simulate_replicate_unified, DEFAULT_PARAMETERS

    result = simulate_replicate_unified(
        primer_seqs=primer_dict,      # {name: seq, name_RC: rc, ...}
        params=DEFAULT_PARAMETERS,
        t_end=90.0,
        rng_seed=42,
    )
"""

from lamprey.parameters import ParameterSet, DEFAULT_PARAMETERS
from lamprey.model_unified import (
    UnifiedSSAEngine,
    simulate_replicate_unified,
    SimulationRunner,
)

__all__ = [
    "ParameterSet",
    "DEFAULT_PARAMETERS",
    "UnifiedSSAEngine",
    "simulate_replicate_unified",
    "SimulationRunner",
]

__version__ = "0.1.0"
