# Nearest-neighbor DNA duplex thermodynamics (SantaLucia 1998) and optional NUPACK backend.
# This module provides ΔG calculators and potency weights. When available and requested,
# NUPACK is used to compute ΔG for specific hairpin/dimer structures; otherwise a heuristic
# nearest-neighbor model is used.

from typing import Dict, Optional

# ΔH (kcal/mol) and ΔS (cal/mol/K) per dinucleotide step on the forward strand (SantaLucia 1998)
NN_H: Dict[str, float] = {
    'AA': -7.9, 'TT': -7.9,
    'AT': -7.2, 'TA': -7.2,
    'CA': -8.5, 'TG': -8.5,
    'GT': -8.4, 'AC': -8.4,
    'CT': -7.8, 'AG': -7.8,
    'GA': -8.2, 'TC': -8.2,
    'CG': -10.6, 'GC': -9.8,
    'GG': -8.0, 'CC': -8.0,
}
NN_S: Dict[str, float] = {
    'AA': -22.2, 'TT': -22.2,
    'AT': -20.4, 'TA': -21.3,
    'CA': -22.7, 'TG': -22.7,
    'GT': -22.4, 'AC': -22.4,
    'CT': -21.0, 'AG': -21.0,
    'GA': -22.2, 'TC': -22.2,
    'CG': -27.2, 'GC': -24.4,
    'GG': -19.9, 'CC': -19.9,
}

R_KCAL = 1.98720425864083e-3  # kcal/mol/K


def _nupack_available() -> bool:
    try:
        import nupack  # noqa: F401
        return True
    except Exception:
        return False


def _build_nupack_model(temp_c: float, sodium_M: float, magnesium_M: float, material: str = 'dna'):
    from nupack import Model
    # NUPACK expects salt in molar units
    return Model(material=material, celsius=float(temp_c), sodium=float(sodium_M), magnesium=float(magnesium_M))


def _nupack_duplex_dG(seq: str, temp_c: float, sodium_M: float, magnesium_M: float, material: str = 'dna') -> Optional[float]:
    """Return MFE ΔG (kcal/mol) for duplex of seq with its perfect complement using NUPACK.
    Returns None if NUPACK is unavailable or an error occurs.
    """
    try:
        from nupack import Strand, Complex, mfe, complex_analysis
        from .utils import reverse_complement
        model = _build_nupack_model(temp_c, sodium_M, magnesium_M, material=material)
        s1 = Strand(seq.upper(), name='s1')
        s2 = Strand(reverse_complement(seq), name='s2')
        cplx = Complex([s1, s2], name='dimer')
        # Try mfe first
        try:
            res = mfe([cplx], model=model)
            # res[0] may have .energy or .structures[0].energy depending on version
            e = getattr(res[0], 'energy', None)
            if e is None:
                e = getattr(res[0].structures[0], 'energy', None)
            if e is not None:
                return float(e)
        except Exception:
            pass
        # Fallback to complex_analysis
        try:
            ana = complex_analysis([cplx], model=model)
            e = ana[0].mfe[0].energy
            return float(e)
        except Exception:
            return None
    except Exception:
        return None


def _nupack_hairpin_dG(stem: str, loop_seq: str, temp_c: float, sodium_M: float, magnesium_M: float, material: str = 'dna') -> Optional[float]:
    """Return MFE ΔG (kcal/mol) for a toy hairpin with given stem and loop using NUPACK.
    Construct sequence: stem + loop_seq + rc(stem) as a single strand and compute MFE.
    Returns None if NUPACK is unavailable or an error occurs.
    """
    try:
        from nupack import Strand, Complex, mfe, complex_analysis
        from .utils import reverse_complement
        seq = stem.upper() + loop_seq.upper() + reverse_complement(stem)
        model = _build_nupack_model(temp_c, sodium_M, magnesium_M, material=material)
        s = Strand(seq, name='hp')
        cplx = Complex([s], name='hairpin')
        # Try mfe first
        try:
            res = mfe([cplx], model=model)
            e = getattr(res[0], 'energy', None)
            if e is None:
                e = getattr(res[0].structures[0], 'energy', None)
            if e is not None:
                return float(e)
        except Exception:
            pass
        # Fallback to complex_analysis
        try:
            ana = complex_analysis([cplx], model=model)
            e = ana[0].mfe[0].energy
            return float(e)
        except Exception:
            return None
    except Exception:
        return None


# Heuristic SantaLucia ΔG for perfect WC duplex of 'seq' with its complement.

def dg_duplex(seq: str, temp_c: float = 37.0) -> float:
    s = seq.upper()
    if len(s) < 2:
        return 0.0
    dH = 0.0
    dS = 0.0
    for i in range(len(s) - 1):
        step = s[i:i+2]
        dH += NN_H.get(step, 0.0)
        dS += NN_S.get(step, 0.0)
    T = temp_c + 273.15
    dG = dH - T * (dS / 1000.0)
    return dG


# Public potency functions with optional NUPACK backend

def potency_from_duplex(
    seq: str,
    temp_c: float = 37.0,
    use_nupack: Optional[bool] = None,
    sodium_mM: float = 150.0e-3,
    magnesium_mM: float = 8.0e-2,
    material: str = 'dna',
) -> float:
    """Convert duplex ΔG(T) to a dimensionless potency weight via exp(-ΔG/RT).
    If use_nupack is True (and NUPACK is available), ΔG is computed by NUPACK for the
    duplex of seq with its perfect complement; otherwise the heuristic is used.
    """
    want_np = (use_nupack is True) or (use_nupack is None and _nupack_available())
    dG: Optional[float] = None
    if want_np and _nupack_available():
        dG = _nupack_duplex_dG(seq, temp_c=float(temp_c), sodium_M=float(sodium_mM), magnesium_M=float(magnesium_mM), material=material)
    if dG is None:
        dG = dg_duplex(seq, temp_c=temp_c)
    T = temp_c + 273.15
    from math import exp
    return float(exp(-dG / (R_KCAL * T)))


def dg_loop_penalty(loop_len: int) -> float:
    """Crude loop penalty ΔG_loop (kcal/mol) as a function of loop length (nt).
    Empirical form: a + b*ln(loop_len+1).
    """
    import math
    if loop_len <= 0:
        return 0.0
    a = 1.75
    b = 0.62
    return a + b * math.log(loop_len + 1.0)


def potency_hairpin(
    lmer: str,
    loop_len: int,
    temp_c: float = 37.0,
    use_nupack: Optional[bool] = None,
    sodium_mM: float = 150.0e-3,
    magnesium_mM: float = 8.0e-2,
    loop_seq: Optional[str] = None,
    material: str = 'dna',
) -> float:
    """Potency for hairpin closure via exp(-ΔG/RT).
    If use_nupack is True and available, compute hairpin MFE ΔG with NUPACK for
    the toy sequence (lmer + loop_seq + rc(lmer)). When loop_seq is None, use a
    poly-T loop of length loop_len. Otherwise use heuristic: ΔG = ΔG_stem + ΔG_loop.
    """
    want_np = (use_nupack is True) or (use_nupack is None and _nupack_available())
    dG: Optional[float] = None
    if want_np and _nupack_available():
        if loop_seq is None:
            loop_seq = 'T' * max(0, int(loop_len))
        dG = _nupack_hairpin_dG(lmer, loop_seq, temp_c=float(temp_c), sodium_M=float(sodium_mM), magnesium_M=float(magnesium_mM), material=material)
    if dG is None:
        dG_stem = dg_duplex(lmer, temp_c=temp_c)
        dG_loop = dg_loop_penalty(loop_len)
        dG = dG_stem + dG_loop
    T = temp_c + 273.15
    from math import exp
    return float(exp(-dG / (R_KCAL * T)))

