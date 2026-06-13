from typing import Dict
from functools import lru_cache

# Precompute translation table once at module load
_RC_TABLE: Dict[int, int] = str.maketrans('ATCGNatcgn', 'TAGCNtagcn')

def reverse_complement(seq: str) -> str:
    """Fast reverse complement using precomputed translation table."""
    return seq.translate(_RC_TABLE)[::-1]


@lru_cache(maxsize=256)
def reverse_complement_cached(seq: str) -> str:
    """Cached reverse complement for frequently used sequences."""
    return seq.translate(_RC_TABLE)[::-1]


def geometric_length(mean_nt: float, rng) -> int:
    """
    Draw an extension length from a geometric distribution with a given mean.
    Returns at least 1 nt.
    """
    import numpy as np
    if mean_nt <= 1:
        return 1
    p = 1.0 / mean_nt
    return int(rng.geometric(p))


def seed_rng(seed: int = 42):
    import numpy as np
    return np.random.default_rng(seed)


def mutate_sequence(seq: str, error_rate: float, rng) -> str:
    """
    Apply stochastic mutations (substitutions, insertions, deletions) to a sequence.
    """
    if error_rate <= 0:
        return seq
        
    bases = ['A', 'T', 'C', 'G']
    mutated = []
    
    p_sub = error_rate
    p_ins = error_rate * 0.1
    p_del = error_rate * 0.1
    
    i = 0
    while i < len(seq):
        # Independent check for insertion before this base
        if rng.random() < p_ins:
            mutated.append(bases[rng.integers(0, 4)])
            
        # Check for deletion or substitution of current base
        r2 = rng.random()
        if r2 < p_del:
            i += 1
            continue
        elif r2 < p_del + p_sub:
            original = seq[i]
            options = [b for b in bases if b != original]
            if options:
                mutated.append(options[rng.integers(0, len(options))])
            else:
                mutated.append(original)
            i += 1
        else:
            mutated.append(seq[i])
            i += 1
            
    if rng.random() < p_ins:
        mutated.append(bases[rng.integers(0, 4)])
        
    return "".join(mutated)
