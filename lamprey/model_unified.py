"""
Unified SSA Engine for LAMP Kinetic Simulation

This module provides a clean, unified implementation of the Gillespie SSA
for LAMP amplification, featuring:

1. L=3 only site indexing with dynamic expansion (6x memory reduction)
2. OBJ-034: dsDNA segment exclusion + strand displacement regeneration
3. OBJ-035: Hairpin loop priming via inter-molecular strand displacement
4. Live-query capability for real-time progress monitoring
5. Unified MoleculePool architecture for products-as-primers (Path 2)

Key Classes:
- UnifiedSSAEngine: Core SSA implementation
- EngineStatus: Status snapshot for live-query
- SimulationRunner: Thread-safe wrapper for async simulation

Author: LAMPrey Team
Date: 2025-12-17
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import defaultdict
import numpy as np
import threading
import time as time_module
import math
import os
import zlib

# Local imports
from lamprey.parameters import ParameterSet, DEFAULT_PARAMETERS
from lamprey.utils import geometric_length, reverse_complement, mutate_sequence
from lamprey.molecule_pool import (
    MoleculePool, Molecule, BindingComplex, LineageEntry, LineageLog, StratifiedCaps
)
from lamprey.sites import (
    SiteIndex, InterSite, IntraSite, LoopSite, ProductInterSite,
    PositionDuplexState, HairpinStructure
)
from lamprey.thermo import dg_duplex
from lamprey.segmentation.segmenter import Segmenter

# Try to import TdT tailing
try:
    # TdT logic lives in emergent.py (shared with legacy/NRM engines)
    from lamprey.emergent import propensity_tdt_tailing, execute_tdt_tailing
    HAS_TDT = True
except ImportError:
    HAS_TDT = False
    def propensity_tdt_tailing(*args, **kwargs): return 0.0
    def execute_tdt_tailing(seq, *args, **kwargs): return seq


# =============================================================================
# Constants
# =============================================================================

R_KCAL = 0.001987  # Gas constant in kcal/(mol·K)
AVOGADRO = 6.02214076e23


# =============================================================================
# Engine Status for Live Query
# =============================================================================

@dataclass
class EngineStatus:
    """Snapshot of engine state for live-query."""
    current_time: float = 0.0
    target_time: float = 0.0
    n_products: int = 0
    n_complexes: int = 0
    n_species: int = 0
    total_nt: int = 0
    n_inter_events: int = 0
    n_intra_events: int = 0
    n_loop_events: int = 0
    n_product_bind_events: int = 0
    n_product_ext_events: int = 0
    is_running: bool = False
    is_complete: bool = False
    phase: str = 'init'
    stop_reason: str = 'init'
    primer_counts: Dict[str, float] = field(default_factory=dict)
    stratified_counts: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Thermodynamic Helper Functions
# =============================================================================

def _dG_duplex_quick(seq: str, temperature_c: float = 65.0) -> float:
    """Quick dG estimate for short sequences."""
    if not seq:
        return 0.0
    try:
        return dg_duplex(seq, temperature_c)
    except Exception:
        # Fallback: rough estimate
        gc_count = seq.upper().count('G') + seq.upper().count('C')
        at_count = len(seq) - gc_count
        return -(gc_count * 2.0 + at_count * 1.5)


def _f_occupancy_duplex(dG: float, dG0: float, slope_kcal: float,
                        temperature_c: float = 65.0) -> float:
    """
    Logistic occupancy function for duplex formation.

    f = 1 / (1 + exp((dG - dG0) / slope))

    Args:
        dG: Free energy of duplex formation (kcal/mol, negative = stable)
        dG0: Midpoint dG for 50% occupancy
        slope_kcal: Logistic slope parameter
        temperature_c: Temperature in Celsius

    Returns:
        Occupancy fraction [0, 1]
    """
    if slope_kcal <= 0:
        slope_kcal = 1.0
    x = (dG - dG0) / slope_kcal
    x = max(-20.0, min(20.0, x))  # Clamp to prevent overflow
    return 1.0 / (1.0 + math.exp(x))


# =============================================================================
# Kinetic Rate Functions
# =============================================================================

def _kon_counts(L: int, params: ParameterSet) -> float:
    """
    Second-order association rate constant in counts units.

    k_on (counts^-1 s^-1) = k_on_base (M^-1 s^-1) / count_scale
    """
    kon_base = (params.kon_base_by_L or {}).get(L, 1e6)
    return float(kon_base) / float(params.count_scale)


def _koff_from_dG(dG: float, L: int, params: ParameterSet) -> float:
    """
    Dissociation rate from thermodynamics.

    k_off = k_on * K_d, where K_d = exp(dG / RT)
    """
    T_K = params.temperature_c + 273.15
    Kd = math.exp(dG / (R_KCAL * T_K))
    kon = (params.kon_base_by_L or {}).get(L, 1e6)
    scale = float(getattr(params, 'koff_scale', 1.0))
    return float(kon) * float(Kd) * scale


def _v_ext_nt_per_s(D_counts: float, params: ParameterSet) -> float:
    """
    Michaelis-Menten extension velocity.

    Notes:
    - In this codebase, D_counts represents the total dNTP counts across all 4 nucleotides.
      We therefore use per-nucleotide concentration [dNTP_each] = D_counts / (4 * count_scale).

    v = kcat * [dNTP_each] / (Km + [dNTP_each])
    """
    kcat = float(params.kcat_nt_per_s)
    Km = float(params.Km_dNTP_M)
    denom = max(1e-30, 4.0 * float(params.count_scale))
    D_each_M = float(D_counts) / denom
    return kcat * D_each_M / (Km + D_each_M) if (Km + D_each_M) > 0 else 0.0


def _size_penalty(template_len: int, params: ParameterSet) -> float:
    """
    Size-dependent diffusion penalty for large templates.

    penalty = (n / ref)^(-exponent)
    """
    ref = float(getattr(params, 'diffusion_reference_nt', 150.0))
    exp = float(getattr(params, 'diffusion_exponent', 0.5))
    if template_len <= 0:
        return 1.0
    return (template_len / ref) ** (-exp)


def _calculate_3p_occlusion(seq: str, params: ParameterSet) -> float:
    """
    Calculate 3' end occlusion penalty based on terminal stability.

    Penalizes overly stable 3' ends that may bind non-specifically.
    """
    if not getattr(params, 'use_3p_stability_penalty', True):
        return 1.0

    L_max = getattr(params, 'stability_penalty_L_max', 4)
    threshold = getattr(params, 'dG_3p_penalty_threshold', -2.0)
    strength = getattr(params, 'stability_penalty_strength', 0.3)
    slope = getattr(params, 'stability_penalty_slope', 0.4)

    if len(seq) < L_max:
        return 1.0

    tail = seq[-L_max:]
    dG = _dG_duplex_quick(tail, params.temperature_c)

    if dG >= threshold:
        return 1.0

    excess = threshold - dG  # Positive when more stable
    penalty = max(strength, 1.0 - slope * excess)
    return penalty


# =============================================================================
# Unified SSA Engine
# =============================================================================

class UnifiedSSAEngine:
    """
    Core Gillespie SSA engine for LAMP simulation.

    Features:
    - L=3 only site indexing with dynamic L expansion
    - OBJ-034: dsDNA exclusion + strand displacement regeneration
    - OBJ-035: Hairpin loop priming
    - Unified molecule pool for products-as-primers
    - Live-query capability
    """

    def __init__(self, primer_seqs: Dict[str, str], params: ParameterSet,
                 Lmin: int = 3, Lmax: int = 8,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the SSA engine.

        Args:
            primer_seqs: Dict mapping primer names to sequences
            params: ParameterSet with all simulation parameters
            Lmin: Minimum toehold length (default 3)
            Lmax: Maximum toehold length for expansion (default 8)
            progress_callback: Optional callback for progress updates
        """
        self.primer_seqs = primer_seqs
        self.params = params
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.progress_callback = progress_callback

        # Initialize molecule pool
        self.pool = MoleculePool(primer_seqs, params)

        # Initialize site index with L=3 only (pass primer_seqs for family registration)
        self.site_index = SiteIndex(
            primer_seqs, Lmin=Lmin, Lmax_inter=Lmax,
            min_loop_size=getattr(params, 'min_loop_size_for_binding', 4)  # CHG-054
        )

        # Initialize segmenter for product classification
        self.segmenter = Segmenter(primer_seqs)

        # dNTP pool (counts)
        self.D_total_counts = 4.0 * params.conc_dNTP_each * params.count_scale

        # Simulation state
        self.t = 0.0
        self.total_nt = 0
        self.simulated_reads: List[Dict] = []
        self.time_points: List[float] = [0.0]
        self.nt_trace: List[int] = [0]

        # Event counters
        self.n_inter_events = 0
        self.n_intra_events = 0
        self.n_loop_events = 0
        self.n_product_bind_events = 0
        self.n_product_ext_events = 0

        # CHG-050: Count only non-intra products for hard stop.
        # Intra (self-folding) extends an existing molecule, not a new species.
        self.n_distinct_products = 0

        # Active template tracking (must be initialized before _seed_primers)
        self.active_template_ids: Set[int] = set()
        self.last_activity_time: Dict[int, float] = {}
        self.creation_times: Dict[int, float] = {}

        # Primer template bookkeeping: primer label -> template_id in SiteIndex
        self._primer_template_ids: Dict[str, int] = {}

        # Seed primers into site index (after template tracking init)
        self._seed_primers()

        # Caches for efficiency
        self._kon_cache: Dict[int, float] = {}
        self._occupancy_cache: Dict[str, float] = {}
        self._template_occlusion_cache: Dict[int, float] = {}
        self._template_size_cache: Dict[int, float] = {}

        # Status for live-query
        self._status = EngineStatus()
        self._status_lock = threading.Lock()
        self._stop_requested = False
        self._stop_reason: str = 'init'

    def _seed_primers(self) -> None:
        """Seed initial templates from primer sequences."""
        # Seed primer templates with physically consistent copy counts.
        counts_by_label: Dict[str, int] = {}
        for name in self.primer_seqs:
            if name.endswith('_RC'):
                continue
            counts_by_label[name] = int(max(0, round(float(self.pool.get_primer_count(name)))))

        # Seed from primers for inter/intra binding sites
        # This also adds each primer sequence as a template
        self.site_index.seed_from_primers(counts_by_label=counts_by_label)

        # Build mapping from primer label -> template_id (used to keep primer/template counts aligned)
        self._primer_template_ids = {}
        for tid, lbl in enumerate(self.site_index.templates_labels):
            if lbl is None:
                continue
            if lbl in counts_by_label:
                self._primer_template_ids[str(lbl)] = int(tid)

        # Register all seeded templates as active
        for tid in range(len(self.site_index.templates)):
            self.active_template_ids.add(tid)
            self.creation_times[tid] = 0.0
            self.last_activity_time[tid] = 0.0

    # -------------------------------------------------------------------------
    # Propensity Calculations
    # -------------------------------------------------------------------------

    def _primer_attenuation_factor(self) -> float:
        """CHG-043: Compute attenuation factor for primer-template propensities.

        Returns a factor in (0, 1] that decreases as products accumulate,
        modeling competitive polymerase inhibition during the primer→product
        kinetic transition.  factor = 1 / (1 + n_products * rate)
        """
        n_products = len(self.simulated_reads)
        rate = float(getattr(self.params, 'primer_attenuation_rate', 0.002))
        return 1.0 / (1.0 + n_products * rate)

    def _get_template(self, template_id: int) -> str:
        """Safely get template sequence by ID."""
        if 0 <= template_id < len(self.site_index.templates):
            return self.site_index.templates[template_id]
        return ''

    def _get_kon(self, L: int) -> float:
        """Get cached k_on for toehold length L."""
        if L not in self._kon_cache:
            self._kon_cache[L] = _kon_counts(L, self.params)
        return self._kon_cache[L]

    def _get_occupancy(self, seq: str, L: int) -> float:
        """Get occupancy fraction for a toehold sequence."""
        key = f"{seq}_{L}"
        if key not in self._occupancy_cache:
            dG = _dG_duplex_quick(seq[-L:], self.params.temperature_c)
            dG0 = (self.params.dG0_by_L or {}).get(L, -3.0)
            slope = self.params.dG_slope_kcal
            self._occupancy_cache[key] = _f_occupancy_duplex(dG, dG0, slope, self.params.temperature_c)
        return self._occupancy_cache[key]

    def _in_product_phase(self) -> bool:
        """Check if simulation has passed the product phase time (X_P)."""
        xp = getattr(self.params, 'product_phase_time', None)
        return xp is not None and self.t >= xp

    def _propensity_inter(self, family: str) -> float:
        """
        Propensity for inter-molecular binding.

        prop = k_on * n_primer * n_sites_effective * f_occupancy

        CHG-043: When primer attenuation is enabled, n_sites is split into
        primer-template and product-template contributions. The primer-template
        contribution is attenuated as products accumulate, while product-template
        sites retain full propensity. This promotes primer→product reactions
        over primer→primer reactions as amplification progresses.
        """
        primer_name = family
        n_primer = self.pool.get_primer_count(primer_name)
        if n_primer <= 0:
            return 0.0

        n_sites = float(self.site_index.inter_count_weighted(family))
        if n_sites <= 0:
            return 0.0

        # CHG-050: Product phase — zero primer→primer-template binding after X_P
        if self._in_product_phase():
            n_sites_primer = float(self.site_index.inter_count_weighted_primer(family))
            n_sites = n_sites - n_sites_primer  # Only product-template sites remain
            if n_sites <= 0:
                return 0.0
        # CHG-043: Attenuate primer-template sites, preserve product-template sites
        elif getattr(self.params, 'enable_primer_attenuation', False):
            n_sites_primer = float(self.site_index.inter_count_weighted_primer(family))
            n_sites_product = n_sites - n_sites_primer
            att = self._primer_attenuation_factor()
            n_sites = n_sites_primer * att + n_sites_product
            if n_sites <= 0:
                return 0.0

        # Use Lmin for base propensity (actual L determined at sampling)
        L = self.Lmin
        kon = self._get_kon(L)
        f_occ = self._get_occupancy(self.primer_seqs.get(family, ''), L)

        inter_scale = float(getattr(self.params, 'inter_rate_scale', 1.0))
        return kon * n_primer * n_sites * f_occ * inter_scale

    def _propensity_intra(self) -> float:
        """
        Propensity for intra-molecular binding (hairpin formation).

        prop = k_close * n_sites_effective * f_occupancy * intra_rate_scale

        CHG-043: When primer attenuation is enabled, primer-template intra sites
        are attenuated while product-template intra sites retain full propensity.
        This reduces futile primer self-folding as products accumulate.
        """
        n_sites = float(self.site_index.intra_count_weighted())
        if n_sites <= 0:
            return 0.0

        # CHG-050: Product phase — zero primer self-folding after X_P
        if self._in_product_phase():
            n_sites_primer = float(self.site_index.intra_count_weighted_primer())
            n_sites = n_sites - n_sites_primer  # Only product-template intra sites remain
            if n_sites <= 0:
                return 0.0
        # CHG-043: Attenuate primer-template intra sites
        elif getattr(self.params, 'enable_primer_attenuation', False):
            n_sites_primer = float(self.site_index.intra_count_weighted_primer())
            n_sites_product = n_sites - n_sites_primer
            att = self._primer_attenuation_factor()
            n_sites = n_sites_primer * att + n_sites_product
            if n_sites <= 0:
                return 0.0

        L = self.Lmin
        # Effective molarity for intra-molecular reactions
        # _get_kon(L) = kon_base / count_scale (already in counts^-1 s^-1)
        # k_close = _get_kon(L) * Meff -> first-order rate (s^-1)
        # Original: (kon_base * Meff / count_scale) * scale
        Meff = (self.params.Meff_intra_by_L or {}).get(L, 1.0)
        k_close = self._get_kon(L) * Meff

        # Get template sequence for occupancy (use generic estimate)
        f_occ = 0.5  # Conservative estimate

        intra_scale = float(getattr(self.params, 'intra_rate_scale', 896.2))

        return k_close * n_sites * f_occ * intra_scale

    def _propensity_loop(self, family: str) -> float:
        """
        Propensity for loop priming (OBJ-035).

        Primer binding to hairpin loop regions.
        """
        if not getattr(self.params, 'enable_loop_priming', True):
            return 0.0

        primer_name = family
        n_primer = self.pool.get_primer_count(primer_name)
        if n_primer <= 0:
            return 0.0

        n_sites = float(self.site_index.loop_count_weighted(family))
        if n_sites <= 0:
            return 0.0

        L = self.Lmin
        kon = self._get_kon(L)
        f_occ = self._get_occupancy(self.primer_seqs.get(family, ''), L)
        loop_factor = float(getattr(self.params, 'loop_accessibility_factor', 0.5))

        inter_scale = float(getattr(self.params, 'inter_rate_scale', 1.0))
        return kon * n_primer * n_sites * f_occ * loop_factor * inter_scale

    def _propensity_product_inter(self, kmer: str) -> float:
        """
        CHG-044: Propensity for product-as-primer binding.

        Products with 3' end matching `kmer` can anneal to templates
        where RC(kmer) appears. Propensity scales with the number of
        free product copies carrying that 3' end and the number of
        matching template sites.

        prop = k_on * n_products_with_kmer * n_template_sites * f_occupancy * scale
        """
        if not getattr(self.params, 'enable_product_as_primer', False):
            return 0.0

        n_products = self.pool.product_primer_counts.get(kmer, 0.0)
        if n_products <= 0:
            return 0.0

        n_sites = float(self.site_index.product_inter_count_weighted(kmer))
        if n_sites <= 0:
            return 0.0

        L = self.Lmin
        kon = self._get_kon(L)
        f_occ = self._get_occupancy(kmer, L)

        inter_scale = float(getattr(self.params, 'inter_rate_scale', 1.0))
        product_scale = float(getattr(self.params, 'product_inter_rate_scale', 1.0))

        return kon * n_products * n_sites * f_occ * inter_scale * product_scale

    def _propensity_extension(self, cplx: BindingComplex) -> float:
        """Propensity for complex extension."""
        v_ext = _v_ext_nt_per_s(self.D_total_counts, self.params)
        if v_ext <= 0:
            return 0.0

        ext_scale = float(getattr(self.params, 'ext_rate_scale', 36.2))

        # dNTP budget (discrete nt counts). If no full nucleotides remain, extension cannot proceed.
        budget = int(max(0.0, math.floor(self.D_total_counts)))
        if budget <= 0:
            return 0.0

        # Maximum templated extension length possible for this complex (toward 5' of template)
        if cplx.kind in ('inter', 'intra', 'product'):
            max_add_len = max(0, cplx.site_pos - cplx.L)
        elif cplx.kind == 'loop' and cplx.hairpin_info:
            h = cplx.hairpin_info
            loop_remaining = max(0, cplx.site_pos - cplx.L - int(h.get('loop_start', 0)))
            stem_displaceable = max(0, int(h.get('stem_5p_end', 0)) - int(h.get('stem_5p_start', 0)))
            max_add_len = max(0, loop_remaining + stem_displaceable)
        else:
            max_add_len = max(0, cplx.site_pos - cplx.L)

        if max_add_len <= 0:
            return 0.0

        # Convert polymerase velocity (nt/s) into a first-order completion rate (s^-1)
        # by dividing by an expected extension length per completion event.
        mean_ext = float(getattr(self.params, 'mean_processivity_nt', 250.0))
        L_char = max(1.0, min(mean_ext, float(max_add_len), float(budget)))

        # Polymerase availability
        E_total = self.params.conc_polymerase_M * self.params.count_scale
        E_bound = float(self.pool.get_bound_count())
        E_free = max(0.0, E_total - E_bound)
        Kp = max(1e-6, float(self.params.Kp_polymerase_counts))
        f_pol = E_free / (E_free + Kp)

        k_ext_base = (v_ext / L_char) * f_pol * ext_scale

        # Occupancy for binding stability
        if cplx.kind in ('inter', 'loop'):
            seq = self.primer_seqs.get(cplx.family, '')
        elif cplx.kind == 'product':
            # Product-as-primer: family stores full 3' seq for accurate dG
            seq = cplx.family
        else:  # intra
            seq = self._get_template(cplx.template_id)
        f_site = self._get_occupancy(seq, cplx.L) if seq else 0.5

        # Size penalty
        templ_len = len(self._get_template(cplx.template_id))
        # CHG-054: lazily memoize by the bound template id. The previous
        # .get(key, _size_penalty(...)) recomputed the penalty on every call
        # (Python evaluates default args eagerly) and keyed by a tid that was
        # never populated, so the cache provided zero benefit.
        size_factor = self._template_size_cache.get(cplx.template_id)
        if size_factor is None:
            size_factor = _size_penalty(templ_len, self.params)
            self._template_size_cache[cplx.template_id] = size_factor

        # 3' occlusion penalty
        occlusion = 1.0
        if getattr(self.params, 'use_3p_occlusion', True):
            if cplx.kind in ('inter', 'loop', 'product'):
                occlusion = self._template_occlusion_cache.get(cplx.template_id, 1.0)
            else:
                templ_seq = self._get_template(cplx.template_id)
                if templ_seq:
                    occlusion = _calculate_3p_occlusion(templ_seq, self.params)

        return k_ext_base * f_site * size_factor * occlusion

    def _propensity_dissociation(self, cplx: BindingComplex) -> float:
        """Propensity for complex dissociation."""
        if cplx.kind in ('inter', 'loop'):
            seq = self.primer_seqs.get(cplx.family, '')
        elif cplx.kind == 'product':
            seq = cplx.family  # family stores full 3' seq for accurate dG
        else:
            seq = self._get_template(cplx.template_id)

        if not seq:
            return 1.0  # Fallback

        lmer = seq[-cplx.L:] if len(seq) >= cplx.L else seq
        dG = _dG_duplex_quick(lmer, self.params.temperature_c)
        koff = _koff_from_dG(dG, cplx.L, self.params)

        # CHG-047: Product complexes represent ~20bp primer-derived duplexes
        # but only match via 3bp toehold. Scale koff to compensate for
        # unmodeled duplex stability beyond the toehold.
        if cplx.kind == 'product':
            koff *= getattr(self.params, 'product_koff_scale', 1.0)

        return koff

    # -------------------------------------------------------------------------
    # Event Building
    # -------------------------------------------------------------------------

    def _build_hazards(self) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Build hazard array and event list.

        Returns:
            (hazards, events): Hazard values and corresponding event descriptors
        """
        hazards = []
        events = []

        # Inter binding events (one per primer family)
        for family in self.primer_seqs:
            if family.endswith('_RC'):
                continue
            h = self._propensity_inter(family)
            if h > 0:
                hazards.append(h)
                events.append(('inter_bind', family, None))

        # Intra binding event (single channel)
        h_intra = self._propensity_intra()
        if h_intra > 0:
            hazards.append(h_intra)
            events.append(('intra_bind', None, None))

        # Loop binding events (OBJ-035)
        if getattr(self.params, 'enable_loop_priming', True):
            for family in self.primer_seqs:
                if family.endswith('_RC'):
                    continue
                h = self._propensity_loop(family)
                if h > 0:
                    hazards.append(h)
                    events.append(('loop_bind', family, None))

        # CHG-044: Product-as-primer binding events (one per unique 3' k-mer)
        if getattr(self.params, 'enable_product_as_primer', False):
            for kmer, count in self.pool.product_primer_counts.items():
                if count <= 0:
                    continue
                h = self._propensity_product_inter(kmer)
                if h > 0:
                    hazards.append(h)
                    events.append(('product_bind', kmer, None))

        # Complex extension/dissociation events
        for i, cplx in enumerate(self.pool.complexes_inter):
            h_ext = self._propensity_extension(cplx)
            h_off = self._propensity_dissociation(cplx)
            hazards.extend([h_ext, h_off])
            events.extend([('inter_ext', 'inter', i), ('inter_off', 'inter', i)])

        for i, cplx in enumerate(self.pool.complexes_intra):
            h_ext = self._propensity_extension(cplx)
            h_off = self._propensity_dissociation(cplx)
            hazards.extend([h_ext, h_off])
            events.extend([('intra_ext', 'intra', i), ('intra_off', 'intra', i)])

        for i, cplx in enumerate(self.pool.complexes_loop):
            h_ext = self._propensity_extension(cplx)
            h_off = self._propensity_dissociation(cplx)
            hazards.extend([h_ext, h_off])
            events.extend([('loop_ext', 'loop', i), ('loop_off', 'loop', i)])

        # CHG-044: Product-as-primer complex extension/dissociation
        for i, cplx in enumerate(self.pool.complexes_product):
            h_ext = self._propensity_extension(cplx)
            h_off = self._propensity_dissociation(cplx)
            hazards.extend([h_ext, h_off])
            events.extend([('product_ext', 'product', i), ('product_off', 'product', i)])

        return np.array(hazards, dtype=float), events

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _adjust_primer_template_free_count(self, family: str, delta: int) -> None:
        """Keep primer-template multiplicity consistent with free primer counts."""
        tid = self._primer_template_ids.get(family)
        if tid is None:
            return
        self.site_index.adjust_template_count(tid, int(delta))

    def _handle_inter_bind(self, family: str, rng: np.random.Generator) -> bool:
        """Handle inter-molecular binding event."""
        # CHG-043: Pass attenuation factor for consistent sampling
        att = self._primer_attenuation_factor() if getattr(self.params, 'enable_primer_attenuation', False) else 1.0
        result = self.site_index.sample_inter(family, rng, primer_attenuation=att)
        if result is None:
            return False

        site, L = result

        # Check primer availability
        if not self.pool.consume_primer(family):
            self.site_index.restore_inter(site)
            return False

        # Primer count changed; keep primer-template multiplicity aligned.
        self._adjust_primer_template_free_count(family, -1)

        # Create complex
        self.pool.create_complex(
            kind='inter',
            binder_id=-1,  # Primer (not tracked as molecule)
            template_id=site.template_id,
            site_pos=site.pos,
            L=L,
            family=family,
            bind_time=self.t
        )

        self.last_activity_time[site.template_id] = self.t
        self.n_inter_events += 1
        return True

    def _handle_intra_bind(self, rng: np.random.Generator) -> bool:
        """Handle intra-molecular binding (hairpin formation)."""
        # CHG-043: Pass attenuation factor for consistent sampling
        att = self._primer_attenuation_factor() if getattr(self.params, 'enable_primer_attenuation', False) else 1.0
        result = self.site_index.sample_intra(rng, primer_attenuation=att)
        if result is None:
            return False

        site, L = result
        template_label = self.site_index.get_template_label(site.template_id) or 'unknown'

        # Create complex
        self.pool.create_complex(
            kind='intra',
            binder_id=site.template_id,  # Template binding itself
            template_id=site.template_id,
            site_pos=site.pos,
            L=L,
            family=template_label,
            bind_time=self.t
        )

        self.last_activity_time[site.template_id] = self.t
        self.n_intra_events += 1
        return True

    def _handle_loop_bind(self, family: str, rng: np.random.Generator) -> bool:
        """Handle loop binding event (OBJ-035)."""
        result = self.site_index.sample_loop(family, rng)
        if result is None:
            return False

        site, L = result

        # Check primer availability
        if not self.pool.consume_primer(family):
            self.site_index.restore_loop(site)
            return False

        # Primer count changed; keep primer-template multiplicity aligned.
        self._adjust_primer_template_free_count(family, -1)

        # Store hairpin info for extension
        hairpin_info = {
            'stem_5p_start': site.hairpin.stem_5p_start,
            'stem_5p_end': site.hairpin.stem_5p_end,
            'loop_start': site.hairpin.loop_start,
            'loop_end': site.hairpin.loop_end,
            'stem_3p_start': site.hairpin.stem_3p_start,
            'stem_3p_end': site.hairpin.stem_3p_end,
        }

        self.pool.create_complex(
            kind='loop',
            binder_id=-1,
            template_id=site.template_id,
            site_pos=site.pos,
            L=L,
            family=family,
            bind_time=self.t,
            hairpin_info=hairpin_info
        )

        self.last_activity_time[site.template_id] = self.t
        self.n_loop_events += 1
        return True

    def _find_product_3p_seq(self, kmer: str) -> Optional[str]:
        """Find the 3' end (up to Lmax bases) of any product matching the given 3' k-mer."""
        Lmax = self.site_index.Lmax_inter
        for mol in self.pool._products.values():
            if mol.sequence and mol.sequence[-len(kmer):] == kmer:
                return mol.sequence[-Lmax:] if len(mol.sequence) >= Lmax else mol.sequence
        return None

    def _handle_product_bind(self, kmer: str, rng: np.random.Generator) -> bool:
        """CHG-044: Handle product-as-primer binding event."""
        # Look up a product's full 3' end for accurate L expansion
        product_3p_seq = self._find_product_3p_seq(kmer)

        result = self.site_index.sample_product_inter(kmer, rng, product_3p_seq=product_3p_seq)
        if result is None:
            return False

        site, L = result

        # Consume one product copy with this 3' k-mer
        current = self.pool.product_primer_counts.get(kmer, 0.0)
        if current <= 0:
            self.site_index.restore_product_inter(site)
            return False
        self.pool.product_primer_counts[kmer] = current - 1.0

        # Create complex — family stores the full product 3' sequence (up to Lmax)
        # so that dissociation and extension occupancy use the correct L-length
        # duplex thermodynamics, not just the 3-mer k-mer.
        product_3p_full = product_3p_seq if product_3p_seq else kmer
        self.pool.create_complex(
            kind='product',
            binder_id=-2,  # Product (distinct from primer=-1)
            template_id=site.template_id,
            site_pos=site.pos,
            L=L,
            family=product_3p_full,  # Full 3' seq for accurate dG in ext/off
            bind_time=self.t
        )

        self.last_activity_time[site.template_id] = self.t
        self.n_product_bind_events += 1
        return True

    def _handle_product_extension(self, idx: int, rng: np.random.Generator) -> bool:
        """CHG-044: Handle product-as-primer extension."""
        if idx >= len(self.pool.complexes_product):
            return False

        cplx = self.pool.remove_complex('product', idx)
        if cplx is None:
            return False

        template_seq = self._get_template(cplx.template_id)
        if not template_seq:
            return False

        budget = int(max(0.0, math.floor(self.D_total_counts)))
        ext_tail = geometric_length(self.params.mean_processivity_nt, rng)

        available_len = max(0, cplx.site_pos - cplx.L)
        add_len = int(min(ext_tail, available_len, budget))

        slice_end = cplx.site_pos - cplx.L
        slice_start = max(0, slice_end - add_len)
        templ_tail = reverse_complement(template_seq[slice_start:slice_end]) if add_len > 0 else ''

        if add_len > 0 and getattr(self.params, 'use_dsDNA_exclusion', True):
            self.site_index.handle_displacement_event(cplx.template_id, slice_start, cplx.site_pos)

        if self.params.dnap_error_rate > 0:
            templ_tail = mutate_sequence(templ_tail, self.params.dnap_error_rate, rng)

        # Product extension: the product itself IS the primer. The emitted
        # sequence is the product's 3' k-mer (binding anchor) + templated extension.
        # In reality the full product sequence would be included, but for the SSA
        # we only track the new template-derived portion plus the binding toehold.
        # family stores the full product 3' seq; k-mer is last Lmin chars
        product_3p = cplx.family
        kmer = product_3p[-self.Lmin:] if len(product_3p) >= self.Lmin else product_3p
        primer_portion = product_3p
        seq_emit = primer_portion + templ_tail
        ext_len = len(templ_tail)

        # TdT tailing (budget-limited)
        remaining_budget = budget - ext_len
        tdt_len = 0
        if getattr(self.params, 'enable_tdt_tailing', True) and HAS_TDT and remaining_budget > 0:
            p_tdt = propensity_tdt_tailing(seq_emit, 1.0, self.params, rng)
            if rng.random() < p_tdt:
                seq_tdt = execute_tdt_tailing(seq_emit, self.params, rng)
                tail = seq_tdt[len(seq_emit):]
                if len(tail) > remaining_budget:
                    tail = tail[:remaining_budget]
                seq_emit = seq_emit + tail
                tdt_len = len(tail)

        ext_len_total = ext_len + tdt_len
        self.D_total_counts = max(0.0, self.D_total_counts - float(ext_len_total))
        self.total_nt += ext_len_total

        # Emit as a product-primed product
        self._emit_product(
            seq_emit, f'P2_{kmer}', cplx.template_id, 'product',
            cplx.L, slice_start, slice_end, len(kmer), rng
        )

        # Restore site
        self.site_index.restore_product_inter(ProductInterSite(
            template_id=cplx.template_id,
            pos=cplx.site_pos,
            kmer=kmer
        ))

        self.n_product_ext_events += 1
        return True

    def _handle_inter_extension(self, idx: int, rng: np.random.Generator) -> bool:
        """Handle inter-molecular extension."""
        if idx >= len(self.pool.complexes_inter):
            return False

        cplx = self.pool.remove_complex('inter', idx)
        if cplx is None:
            return False

        template_seq = self._get_template(cplx.template_id)
        if not template_seq:
            return False

        # dNTP budget (counts) - prevent nucleotide creation after depletion
        budget = int(max(0.0, math.floor(self.D_total_counts)))

        # Calculate extension length
        ext_tail = geometric_length(self.params.mean_processivity_nt, rng)

        # Available length: from binding site toward 5' of template
        # pos is END of binding site, so available = pos - L
        available_len = max(0, cplx.site_pos - cplx.L)
        add_len = int(min(ext_tail, available_len, budget))

        # Slice template (extending toward 5')
        slice_end = cplx.site_pos - cplx.L
        slice_start = max(0, slice_end - add_len)
        templ_tail = reverse_complement(template_seq[slice_start:slice_end]) if add_len > 0 else ''

        # OBJ-034: Handle strand displacement through dsDNA
        if add_len > 0 and getattr(self.params, 'use_dsDNA_exclusion', True):
            self.site_index.handle_displacement_event(cplx.template_id, slice_start, cplx.site_pos)

        # Apply DNAP errors
        if self.params.dnap_error_rate > 0:
            templ_tail = mutate_sequence(templ_tail, self.params.dnap_error_rate, rng)

        # Build product sequence
        primer_seq = self.primer_seqs.get(cplx.family, '')
        seq_emit = primer_seq + templ_tail
        ext_len = len(templ_tail)

        # TdT tailing (budget-limited)
        remaining_budget = budget - ext_len
        tdt_len = 0
        if getattr(self.params, 'enable_tdt_tailing', True) and HAS_TDT and remaining_budget > 0:
            p_tdt = propensity_tdt_tailing(seq_emit, 1.0, self.params, rng)
            if rng.random() < p_tdt:
                seq_tdt = execute_tdt_tailing(seq_emit, self.params, rng)
                tail = seq_tdt[len(seq_emit):]
                if len(tail) > remaining_budget:
                    tail = tail[:remaining_budget]
                seq_emit = seq_emit + tail
                tdt_len = len(tail)

        ext_len_total = ext_len + tdt_len

        # Update dNTP pool
        self.D_total_counts = max(0.0, self.D_total_counts - float(ext_len_total))
        self.total_nt += ext_len_total

        # Add product to pool and site index
        self._emit_product(
            seq_emit, cplx.family, cplx.template_id, 'inter',
            cplx.L, slice_start, slice_end, len(primer_seq), rng
        )

        # Restore site
        self.site_index.restore_inter(InterSite(
            template_id=cplx.template_id,
            pos=cplx.site_pos,
            family=cplx.family
        ))

        return True

    def _handle_intra_extension(self, idx: int, rng: np.random.Generator) -> bool:
        """Handle intra-molecular extension (hairpin)."""
        if idx >= len(self.pool.complexes_intra):
            return False

        cplx = self.pool.remove_complex('intra', idx)
        if cplx is None:
            return False

        template_seq = self._get_template(cplx.template_id)
        if not template_seq:
            return False

        # dNTP budget (counts) - prevent nucleotide creation after depletion
        budget = int(max(0.0, math.floor(self.D_total_counts)))

        ext_tail = geometric_length(self.params.mean_processivity_nt, rng)

        # For intra, available = pos - L (binding site to 5' end)
        available_len = max(0, cplx.site_pos - cplx.L)
        add_len = int(min(ext_tail, available_len, budget))

        # CHG-052: Skip non-productive terminal hairpins (nothing upstream to copy)
        if add_len == 0:
            # Restore site and return — no product emitted
            self.site_index.restore_intra(IntraSite(
                template_id=cplx.template_id, pos=cplx.site_pos))
            return False

        slice_end = cplx.site_pos - cplx.L
        slice_start = max(0, slice_end - add_len)
        templ_tail = reverse_complement(template_seq[slice_start:slice_end])

        # OBJ-034: Handle strand displacement
        if getattr(self.params, 'use_dsDNA_exclusion', True):
            self.site_index.handle_displacement_event(cplx.template_id, slice_start, cplx.site_pos)

        if self.params.dnap_error_rate > 0:
            templ_tail = mutate_sequence(templ_tail, self.params.dnap_error_rate, rng)

        # For intra, base sequence is the template itself
        seq_emit = template_seq + templ_tail
        ext_len = len(templ_tail)

        # TdT tailing (budget-limited)
        remaining_budget = budget - ext_len
        tdt_len = 0
        if getattr(self.params, 'enable_tdt_tailing', True) and HAS_TDT and remaining_budget > 0:
            p_tdt = propensity_tdt_tailing(seq_emit, 1.0, self.params, rng)
            if rng.random() < p_tdt:
                seq_tdt = execute_tdt_tailing(seq_emit, self.params, rng)
                tail = seq_tdt[len(seq_emit):]
                if len(tail) > remaining_budget:
                    tail = tail[:remaining_budget]
                seq_emit = seq_emit + tail
                tdt_len = len(tail)

        ext_len_total = ext_len + tdt_len

        self.D_total_counts = max(0.0, self.D_total_counts - float(ext_len_total))
        self.total_nt += ext_len_total

        # Emit hairpin product
        self._emit_hairpin_product(
            seq_emit, cplx.family, cplx.template_id, cplx.L,
            slice_start, slice_end, len(template_seq), rng
        )

        # Restore intra site
        self.site_index.restore_intra(IntraSite(
            template_id=cplx.template_id,
            pos=cplx.site_pos
        ))

        return True

    def _handle_loop_extension(self, idx: int, rng: np.random.Generator) -> bool:
        """Handle loop extension with strand displacement (OBJ-035)."""
        if idx >= len(self.pool.complexes_loop):
            return False

        cplx = self.pool.remove_complex('loop', idx)
        if cplx is None:
            return False

        template_seq = self._get_template(cplx.template_id)
        if not template_seq or not cplx.hairpin_info:
            return False

        hairpin = cplx.hairpin_info

        # dNTP budget (counts) - prevent nucleotide creation after depletion
        budget = int(max(0.0, math.floor(self.D_total_counts)))

        ext_tail = geometric_length(self.params.mean_processivity_nt, rng)

        # Extension through loop + possible stem displacement
        loop_remaining = max(0, cplx.site_pos - cplx.L - hairpin['loop_start'])
        stem_displaceable = hairpin['stem_5p_end'] - hairpin['stem_5p_start']

        # Strand displacement rate penalty
        disp_rate_scale = float(getattr(self.params, 'strand_displacement_rate_scale', 0.3))

        if loop_remaining >= ext_tail:
            add_len = int(ext_tail)
        else:
            loop_ext = loop_remaining
            stem_ext = int((ext_tail - loop_remaining) * disp_rate_scale)
            stem_ext = min(stem_ext, stem_displaceable)
            add_len = int(loop_ext + stem_ext)

        available_len = loop_remaining + stem_displaceable
        add_len = min(add_len, available_len)
        add_len = min(add_len, budget)

        slice_end = cplx.site_pos - cplx.L
        slice_start = max(0, slice_end - add_len)
        templ_tail = reverse_complement(template_seq[slice_start:slice_end]) if add_len > 0 else ''

        # Handle stem displacement
        stem_displaced = max(0, add_len - loop_remaining)
        if stem_displaced > 0 and getattr(self.params, 'use_dsDNA_exclusion', True):
            self.site_index.handle_displacement_event(
                cplx.template_id,
                hairpin['stem_5p_start'],
                hairpin['stem_5p_start'] + stem_displaced
            )

        if self.params.dnap_error_rate > 0:
            templ_tail = mutate_sequence(templ_tail, self.params.dnap_error_rate, rng)

        primer_seq = self.primer_seqs.get(cplx.family, '')
        seq_emit = primer_seq + templ_tail
        ext_len = len(templ_tail)

        # TdT tailing (budget-limited)
        remaining_budget = budget - ext_len
        tdt_len = 0
        if getattr(self.params, 'enable_tdt_tailing', True) and HAS_TDT and remaining_budget > 0:
            p_tdt = propensity_tdt_tailing(seq_emit, 1.0, self.params, rng)
            if rng.random() < p_tdt:
                seq_tdt = execute_tdt_tailing(seq_emit, self.params, rng)
                tail = seq_tdt[len(seq_emit):]
                if len(tail) > remaining_budget:
                    tail = tail[:remaining_budget]
                seq_emit = seq_emit + tail
                tdt_len = len(tail)

        ext_len_total = ext_len + tdt_len

        self.D_total_counts = max(0.0, self.D_total_counts - float(ext_len_total))
        self.total_nt += ext_len_total

        # Emit loop product
        self._emit_product(
            seq_emit, cplx.family, cplx.template_id, 'loop',
            cplx.L, slice_start, slice_end, len(primer_seq), rng,
            extra_info={'stem_displaced': stem_displaced}
        )

        # Restore loop site
        loop_site = LoopSite(
            template_id=cplx.template_id,
            pos=cplx.site_pos,
            family=cplx.family,
            hairpin=HairpinStructure(
                stem_5p_start=hairpin['stem_5p_start'],
                stem_5p_end=hairpin['stem_5p_end'],
                loop_start=hairpin['loop_start'],
                loop_end=hairpin['loop_end'],
                stem_3p_start=hairpin['stem_3p_start'],
                stem_3p_end=hairpin['stem_3p_end'],
                stem_length=hairpin['stem_5p_end'] - hairpin['stem_5p_start']
            )
        )
        self.site_index.restore_loop(loop_site)

        return True

    def _handle_dissociation(self, kind: str, idx: int) -> bool:
        """Handle complex dissociation."""
        cplx = self.pool.remove_complex(kind, idx)
        if cplx is None:
            return False

        # Restore primer for inter/loop
        if kind in ('inter', 'loop'):
            self.pool.restore_primer(cplx.family)
            # Primer count changed; keep primer-template multiplicity aligned.
            self._adjust_primer_template_free_count(cplx.family, +1)

        # Restore site
        if kind == 'inter':
            self.site_index.restore_inter(InterSite(
                template_id=cplx.template_id,
                pos=cplx.site_pos,
                family=cplx.family
            ))
        elif kind == 'intra':
            self.site_index.restore_intra(IntraSite(
                template_id=cplx.template_id,
                pos=cplx.site_pos
            ))
        elif kind == 'loop':
            # Need hairpin info to restore
            if cplx.hairpin_info:
                h = cplx.hairpin_info
                loop_site = LoopSite(
                    template_id=cplx.template_id,
                    pos=cplx.site_pos,
                    family=cplx.family,
                    hairpin=HairpinStructure(
                        stem_5p_start=h['stem_5p_start'],
                        stem_5p_end=h['stem_5p_end'],
                        loop_start=h['loop_start'],
                        loop_end=h['loop_end'],
                        stem_3p_start=h['stem_3p_start'],
                        stem_3p_end=h['stem_3p_end'],
                        stem_length=h['stem_5p_end'] - h['stem_5p_start']
                    )
                )
                self.site_index.restore_loop(loop_site)
        elif kind == 'product':
            # CHG-044: Restore product copy count and site
            # family stores full 3' seq; k-mer is last Lmin chars
            product_3p = cplx.family
            kmer = product_3p[-self.Lmin:] if len(product_3p) >= self.Lmin else product_3p
            self.pool.product_primer_counts[kmer] = self.pool.product_primer_counts.get(kmer, 0.0) + 1.0
            self.site_index.restore_product_inter(ProductInterSite(
                template_id=cplx.template_id,
                pos=cplx.site_pos,
                kmer=kmer
            ))

        return True

    # -------------------------------------------------------------------------
    # Product Emission
    # -------------------------------------------------------------------------

    def _emit_product(self, seq: str, family: str, template_id: int,
                      kind: str, L: int, slice_start: int, slice_end: int,
                      primer_len: int, rng: np.random.Generator,
                      extra_info: Optional[Dict] = None) -> None:
        """Emit a product from inter or loop extension."""
        # Add template
        new_tid, is_new = self.site_index.add_template(seq, label=family, coalesce=True)

        # Stratified counting / ghosting: decide BEFORE indexing so ghosted products do not
        # contribute sites/hazards.
        ghosted = False
        if getattr(self.params, 'use_stratified_limits', True):
            self.pool.stratified_caps.classify_and_count(
                new_tid, seq, is_intra=False, primer_name=family
            )
            ghosted = self.pool.stratified_caps.is_ghosted(new_tid)

        if is_new and not ghosted:
            self.site_index.update_with_product(
                new_tid,
                blocked_intra_region=None,
                duplex_regions=None  # Products are ssDNA at emission
            )

            self.active_template_ids.add(new_tid)
            self.creation_times[new_tid] = self.t

            # Cache size/occlusion
            self._template_size_cache[new_tid] = _size_penalty(len(seq), self.params)
            if getattr(self.params, 'use_3p_occlusion', True):
                self._template_occlusion_cache[new_tid] = _calculate_3p_occlusion(seq, self.params)

            # CHG-044: Scan new template for existing product 3' k-mers
            if getattr(self.params, 'enable_product_as_primer', False):
                self.site_index.index_product_primer_on_template(new_tid)

        if is_new and ghosted:
            # Ghosted products are recorded but intentionally do not enter the active
            # site-index (soft cap to prevent kinetic blowup).
            self.active_template_ids.discard(new_tid)

        self.last_activity_time[new_tid] = self.t

        # CHG-044: Register product's 3' k-mer for product-as-primer
        if getattr(self.params, 'enable_product_as_primer', False):
            min_len = int(getattr(self.params, 'min_product_length_for_priming', 30))
            if len(seq) >= min_len:
                kmer = seq[-self.Lmin:]
                # Index k-mer across all templates (idempotent for seen k-mers)
                self.site_index.index_product_primer_kmer(kmer)
                # Increment product count for this k-mer
                self.pool.product_primer_counts[kmer] += 1.0

        # Record in pool
        self.pool.add_product(
            sequence=seq,
            label=f'{family}+{kind}*{L}-{new_tid}',
            family=family,
            template_id=template_id,
            kind=kind,
            binding_L=L,
            creation_time=self.t,
            is_autocatalytic=False,
            coalesce=True
        )

        # Build read record
        segs = self.segmenter.segment(seq)
        header = f'sim_{len(self.simulated_reads):07d}'
        read = {
            'header': header,
            'sequence': seq,
            'segments': segs,
            'time_emitted': self.t,
            'label': family,
            'product_label': f'{family}+{kind}*{L}-{new_tid}',
            'kind': kind,
            'L': L,
            'template_id': template_id,
            'id': new_tid,
        }
        if extra_info:
            read.update(extra_info)

        self.simulated_reads.append(read)
        # CHG-054: count only genuinely new species toward the hard stop. The
        # counter bounds the search space; a coalesced re-emission (is_new=False)
        # adds a copy to an existing species and does not enlarge it.
        if is_new:
            self.n_distinct_products += 1
        self.time_points.append(self.t)
        self.nt_trace.append(self.total_nt)

    def _emit_hairpin_product(self, seq: str, family: str, template_id: int,
                              L: int, slice_start: int, slice_end: int,
                              original_len: int, rng: np.random.Generator) -> None:
        """Emit a hairpin product from intra extension."""
        new_tid, is_new = self.site_index.add_template(seq, label=family, coalesce=True)

        # Stratified counting / ghosting: decide BEFORE indexing so ghosted products do not
        # contribute sites/hazards.
        ghosted = False
        if getattr(self.params, 'use_stratified_limits', True):
            self.pool.stratified_caps.classify_and_count(
                new_tid, seq, is_intra=True, primer_name=family
            )
            ghosted = self.pool.stratified_caps.is_ghosted(new_tid)

        if is_new and not ghosted:
            # Compute hairpin topology for ALL intra products (needed for loop priming OBJ-035)
            extended_len = len(seq)

            # 5' stem: from 0 to where the hairpin loop starts
            stem_5p_end = slice_start  # Approximate

            # 3' stem: from original 3' end to current 3' end
            stem_3p_start = original_len - L
            stem_3p_end = extended_len

            # Loop region is between stems
            loop_start = stem_5p_end
            loop_end = stem_3p_start

            hairpin = HairpinStructure(
                stem_5p_start=0,
                stem_5p_end=stem_5p_end,
                loop_start=loop_start,
                loop_end=loop_end,
                stem_3p_start=stem_3p_start,
                stem_3p_end=stem_3p_end,
                stem_length=stem_5p_end
            )

            # OBJ-034: dsDNA exclusion — only compute duplex regions when enabled
            duplex_regions = []
            if getattr(self.params, 'use_dsDNA_exclusion', True):
                # Add dsDNA regions for stems (linked to each other)
                if stem_5p_end > 0:
                    duplex_regions.append(PositionDuplexState(
                        start=0, end=stem_5p_end, p_dsDNA=1.0,
                        linked_template=new_tid,
                        linked_region=(stem_3p_start, stem_3p_end)
                    ))
                if stem_3p_end > stem_3p_start:
                    duplex_regions.append(PositionDuplexState(
                        start=stem_3p_start, end=stem_3p_end, p_dsDNA=1.0,
                        linked_template=new_tid,
                        linked_region=(0, stem_5p_end)
                    ))

            self.site_index.update_with_product(
                new_tid,
                blocked_intra_region=(slice_start, slice_end) if slice_end > slice_start else None,
                duplex_regions=duplex_regions if duplex_regions else None,
                hairpin=hairpin,
            )

            self.active_template_ids.add(new_tid)
            self.creation_times[new_tid] = self.t
            self._template_size_cache[new_tid] = _size_penalty(len(seq), self.params)
            if getattr(self.params, 'use_3p_occlusion', True):
                self._template_occlusion_cache[new_tid] = _calculate_3p_occlusion(seq, self.params)

            # CHG-044: Scan new template for existing product 3' k-mers
            if getattr(self.params, 'enable_product_as_primer', False):
                self.site_index.index_product_primer_on_template(new_tid)

        if is_new and ghosted:
            self.active_template_ids.discard(new_tid)

        self.last_activity_time[new_tid] = self.t

        # CHG-044: Register product's 3' k-mer for product-as-primer
        if getattr(self.params, 'enable_product_as_primer', False):
            min_len = int(getattr(self.params, 'min_product_length_for_priming', 30))
            if len(seq) >= min_len:
                kmer = seq[-self.Lmin:]
                self.site_index.index_product_primer_kmer(kmer)
                self.pool.product_primer_counts[kmer] += 1.0

        self.pool.add_product(
            sequence=seq,
            label=f'{family}+intra*{L}-{new_tid}',
            family=family,
            template_id=template_id,
            kind='intra',
            binding_L=L,
            creation_time=self.t,
            is_autocatalytic=True,
            coalesce=True
        )

        segs = self.segmenter.segment(seq)
        header = f'sim_{len(self.simulated_reads):07d}'
        self.simulated_reads.append({
            'header': header,
            'sequence': seq,
            'segments': segs,
            'time_emitted': self.t,
            'label': family,
            'product_label': f'{family}+intra*{L}-{new_tid}',
            'kind': 'intra',
            'L': L,
            'template_id': template_id,
            'id': new_tid,
            'is_autocatalytic': True,
        })
        self.time_points.append(self.t)
        self.nt_trace.append(self.total_nt)

    # -------------------------------------------------------------------------
    # Main SSA Loop
    # -------------------------------------------------------------------------

    def step(self, rng: np.random.Generator) -> float:
        """
        Execute one SSA step.

        Returns:
            Time increment (dt)
        """
        # Build hazards
        hazards, events = self._build_hazards()
        a0 = float(hazards.sum())

        if a0 <= 0:
            return float('inf')  # No reactions possible

        # Sample time increment
        dt = -math.log(rng.random()) / a0

        # Sample event
        r = rng.random() * a0
        cum = np.cumsum(hazards)
        idx = int(np.searchsorted(cum, r))
        if idx >= len(events):
            idx = len(events) - 1

        event_type, data1, data2 = events[idx]

        # Execute event
        if event_type == 'inter_bind':
            self._handle_inter_bind(data1, rng)
        elif event_type == 'intra_bind':
            self._handle_intra_bind(rng)
        elif event_type == 'loop_bind':
            self._handle_loop_bind(data1, rng)
        elif event_type == 'inter_ext':
            self._handle_inter_extension(data2, rng)
        elif event_type == 'intra_ext':
            self._handle_intra_extension(data2, rng)
        elif event_type == 'loop_ext':
            self._handle_loop_extension(data2, rng)
        elif event_type == 'inter_off':
            self._handle_dissociation('inter', data2)
        elif event_type == 'intra_off':
            self._handle_dissociation('intra', data2)
        elif event_type == 'loop_off':
            self._handle_dissociation('loop', data2)
        elif event_type == 'product_bind':
            self._handle_product_bind(data1, rng)
        elif event_type == 'product_ext':
            self._handle_product_extension(data2, rng)
        elif event_type == 'product_off':
            self._handle_dissociation('product', data2)

        return dt

    def run(self, t_end: float, rng_seed: int = 42,
            hard_stop_limit: int = 5000,
            max_species: Optional[int] = 500) -> Dict[str, Any]:
        """Run simulation until t_end or a configured hard-stop.

        Args:
            t_end: End time in seconds.
            rng_seed: Random seed.
            hard_stop_limit: Maximum emitted products (reads) before stopping.
            max_species: Maximum number of unique template species before stopping.
                Uses SiteIndex template count (coalesced by sequence). Set to None or
                <= 0 to disable.

        Returns:
            Result dictionary with time series and products.
        """
        rng = np.random.default_rng(rng_seed)
        self.t = 0.0
        self._stop_requested = False
        self._stop_reason = 'running'

        max_species_eff: Optional[int]
        if max_species is None or int(max_species) <= 0:
            max_species_eff = None
        else:
            max_species_eff = int(max_species)

        stop_reason = 'completed'

        # Update status
        with self._status_lock:
            self._status.is_running = True
            self._status.is_complete = False
            self._status.target_time = t_end
            self._status.phase = 'running'
            self._status.stop_reason = self._stop_reason

        last_callback_time = 0.0
        callback_interval = float(getattr(self.params, 'gui_update_interval', 0.5))
        
        step_count = 0
        
        # DIAGNOSTIC: Log propensity feedback every 10 sim-seconds
        _diag_enabled = bool(os.environ.get('SSA_DIAG', ''))
        _diag_last_t = -999.0
        _diag_interval = 10.0
        if _diag_enabled:
            import sys as _sys
            print(f"{'t':>8s} {'steps':>8s} {'a0':>10s} {'n_spec':>6s} {'n_prod':>6s} {'inter_s':>8s} {'intra_s':>8s} {'FIP_n':>8s} {'BIP_n':>8s} {'n_cplx':>6s} {'tot_nt':>8s}", file=_sys.stderr, flush=True)

        while self.t < t_end and not self._stop_requested:
            step_count += 1

            # Check hard stops BEFORE building hazards.
            # CHG-050: Use n_distinct_products (excludes intra self-folding) for hard stop.
            # Intra events extend existing molecules, not new species.
            if hard_stop_limit is not None and self.n_distinct_products >= hard_stop_limit:
                stop_reason = 'hard_stop_products'
                break

            if max_species_eff is not None and len(self.site_index.templates) >= max_species_eff:
                stop_reason = 'hard_stop_species'
                break

            # Safety valve: cap total reads (including intra) to prevent runaway
            if len(self.simulated_reads) >= hard_stop_limit * 10:
                stop_reason = 'hard_stop_total_reads'
                break

            dt = self.step(rng)

            if dt == float('inf'):
                # No reactions possible - advance time
                self.t = t_end
                stop_reason = 'no_reactions'
                break

            self.t += dt

            # Update status periodically
            if self.t - last_callback_time >= callback_interval:
                self._update_status()
                if self.progress_callback:
                    self.progress_callback(self._status)
                last_callback_time = self.t
            
            # DIAGNOSTIC: Lightweight propensity logging
            if _diag_enabled and self.t - _diag_last_t >= _diag_interval:
                _d_inter = sum(float(self.site_index.inter_count_weighted(f)) for f in self.primer_seqs if not f.endswith('_RC'))
                _d_intra = float(self.site_index.intra_count_weighted())
                _d_fip = self.pool.get_primer_count('FIP') if 'FIP' in self.primer_seqs else 0
                _d_bip = self.pool.get_primer_count('BIP') if 'BIP' in self.primer_seqs else 0
                _d_cplx = self.pool.get_bound_count()
                print(f"{self.t:8.1f} {step_count:8d} {0:10.4f} {len(self.site_index.templates):6d} {len(self.simulated_reads):6d} {_d_inter:8.1f} {_d_intra:8.1f} {_d_fip:8.1f} {_d_bip:8.1f} {_d_cplx:6d} {self.total_nt:8d}", file=_sys.stderr, flush=True)
                _diag_last_t = self.t

        if self._stop_requested and stop_reason == 'completed' and self.t < t_end:
            stop_reason = 'stop_requested'

        self._stop_reason = stop_reason

        # Final status update
        with self._status_lock:
            self._status.is_running = False
            self._status.is_complete = True
            self._status.current_time = self.t
            self._status.n_products = len(self.simulated_reads)
            self._status.n_species = len(self.site_index.templates)
            self._status.total_nt = self.total_nt
            self._status.phase = 'complete'
            self._status.stop_reason = self._stop_reason

        return self._build_result(target_time=t_end, actual_final_time=self.t, stop_reason=stop_reason)

    def _update_status(self) -> None:
        """Update status snapshot."""
        with self._status_lock:
            self._status.current_time = self.t
            self._status.n_products = len(self.simulated_reads)
            self._status.n_complexes = self.pool.get_bound_count()
            self._status.n_species = len(self.site_index.templates)
            self._status.total_nt = self.total_nt
            self._status.n_inter_events = self.n_inter_events
            self._status.n_intra_events = self.n_intra_events
            self._status.n_loop_events = self.n_loop_events
            self._status.n_product_bind_events = self.n_product_bind_events
            self._status.n_product_ext_events = self.n_product_ext_events
            self._status.stop_reason = self._stop_reason
            self._status.primer_counts = dict(self.pool.primer_counts)
            self._status.stratified_counts = self.pool.stratified_caps.get_counts()

    def get_status(self) -> EngineStatus:
        """Get current status snapshot (thread-safe)."""
        with self._status_lock:
            return replace(self._status)

    def request_stop(self) -> None:
        """Request graceful stop of simulation."""
        self._stop_requested = True

    def _build_result(
        self,
        target_time: Optional[float] = None,
        actual_final_time: Optional[float] = None,
        stop_reason: str = 'completed',
    ) -> Dict[str, Any]:
        """Build result dictionary."""
        time_points = list(self.time_points)
        nt_trace = list(self.nt_trace)

        # Ensure time series can be treated as complete at target_time.
        if target_time is not None:
            target_time = float(target_time)
            if len(time_points) == 0:
                time_points = [0.0, target_time]
                nt_trace = [0, 0]
            elif time_points[-1] < target_time:
                time_points.append(target_time)
                nt_trace.append(int(nt_trace[-1]))

        # For hard-stop cases, treat the simulation as if it reached target_time.
        final_time_effective = self.t
        if target_time is not None and stop_reason in ('hard_stop_products', 'hard_stop_species'):
            final_time_effective = target_time

        return {
            'time': time_points,
            'total_nt': nt_trace,
            'reads': self.simulated_reads,
            'target_time': float(target_time) if target_time is not None else None,
            'final_time': float(final_time_effective),
            'actual_final_time': float(actual_final_time) if actual_final_time is not None else float(self.t),
            'final_nt': int(self.total_nt),
            'n_products': len(self.simulated_reads),
            'n_species': len(self.site_index.templates),
            'n_molecules': len(self.pool.molecules),
            'n_complexes': self.pool.get_bound_count(),
            'stop_reason': str(stop_reason),
            'count_by_source': {
                'inter': self.n_inter_events,
                'intra': self.n_intra_events,
                'loop': self.n_loop_events,
                'product_bind': self.n_product_bind_events,
                'product_ext': self.n_product_ext_events,
            },
            'primer_counts': dict(self.pool.primer_counts),
            'stratified_counts': self.pool.stratified_caps.get_counts(),
            'lineage': self.pool.lineage.to_list(),
        }


# =============================================================================
# Simulation Runner (Live Query Interface)
# =============================================================================

class SimulationRunner:
    """
    Thread-safe wrapper for running simulations with live-query capability.

    Example usage:
        runner = SimulationRunner(primer_seqs, params)
        runner.start(t_end=3600, seed=42)
        while runner.is_running():
            status = runner.get_status()
            print(f"Progress: {status.current_time:.1f}s, Products: {status.n_products}")
            time.sleep(1.0)
        result = runner.get_result()
    """

    def __init__(self, primer_seqs: Dict[str, str], params: ParameterSet,
                 Lmin: int = 3, Lmax: int = 8):
        self.primer_seqs = primer_seqs
        self.params = params
        self.Lmin = Lmin
        self.Lmax = Lmax

        self._engine: Optional[UnifiedSSAEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict] = None
        self._lock = threading.Lock()

    def start(
        self,
        t_end: float,
        seed: int = 42,
        hard_stop_limit: int = 5000,
        max_species: Optional[int] = 500,
    ) -> None:
        """Start simulation in background thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Simulation already running")

            self._engine = UnifiedSSAEngine(
                self.primer_seqs, self.params, self.Lmin, self.Lmax
            )
            self._result = None

            def run_sim():
                result = self._engine.run(t_end, seed, hard_stop_limit, max_species=max_species)
                with self._lock:
                    self._result = result

            self._thread = threading.Thread(target=run_sim, daemon=True)
            self._thread.start()

    def is_running(self) -> bool:
        """Check if simulation is still running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> EngineStatus:
        """Get current simulation status."""
        with self._lock:
            if self._engine is None:
                return EngineStatus()
            return self._engine.get_status()

    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Get simulation result (blocks until complete if timeout=None)."""
        if self._thread is not None:
            self._thread.join(timeout)

        with self._lock:
            return self._result

    def stop(self) -> None:
        """Request simulation stop."""
        with self._lock:
            if self._engine is not None:
                self._engine.request_stop()


# =============================================================================
# Convenience Functions
# =============================================================================

def simulate_replicate_unified(
    primer_seqs: Dict[str, str],
    params: ParameterSet,
    Lmin: int = 3,
    t_end: float = 3600.0,
    rng_seed: int = 42,
    hard_stop_limit: int = 5000,
    max_species: Optional[int] = 500,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run a single simulation replicate.

    This is the main entry point for running simulations with the unified engine.

    Args:
        primer_seqs: Dict mapping primer names to sequences
        params: ParameterSet with simulation parameters
        Lmin: Minimum toehold length (default 3)
        t_end: Simulation end time in seconds
        rng_seed: Random seed for reproducibility
        hard_stop_limit: Maximum products before stopping
        max_species: Maximum number of unique template species before stopping
        progress_callback: Optional callback for progress updates

    Returns:
        Result dictionary with time series, products, and statistics
    """
    engine = UnifiedSSAEngine(
        primer_seqs, params, Lmin=Lmin, Lmax=8,
        progress_callback=progress_callback
    )
    return engine.run(t_end, rng_seed, hard_stop_limit, max_species=max_species)


# Global registry for running simulations (for query_running_simulations)
_running_simulations: Dict[str, SimulationRunner] = {}
_registry_lock = threading.Lock()


def query_running_simulations() -> Dict[str, EngineStatus]:
    """
    Query status of all running simulations.

    Returns:
        Dict mapping simulation IDs to their current status
    """
    with _registry_lock:
        return {
            sim_id: runner.get_status()
            for sim_id, runner in _running_simulations.items()
            if runner.is_running()
        }


def register_simulation(sim_id: str, runner: SimulationRunner) -> None:
    """Register a simulation runner for querying."""
    with _registry_lock:
        _running_simulations[sim_id] = runner


def unregister_simulation(sim_id: str) -> None:
    """Unregister a completed simulation."""
    with _registry_lock:
        _running_simulations.pop(sim_id, None)


# =============================================================================
# Multi-Core Batch Runner with Live Query (ProcessPoolExecutor)
# =============================================================================

# Standalone worker function for ProcessPoolExecutor (must be at module level for pickling)
def _batch_worker(args: Tuple) -> Dict[str, Any]:
    """
    Standalone worker function for ProcessPoolExecutor.

    This runs in a separate process for true parallelism (bypasses GIL).
    """
    primer_name, primer_seqs, params_dict, t_end, seed, time_grid = args

    try:
        # Reconstruct ParameterSet from dict
        from lamprey.parameters import ParameterSet
        params = ParameterSet(**params_dict)

        # Run simulation
        result = simulate_replicate_unified(
            primer_seqs=primer_seqs,
            params=params,
            Lmin=3,
            t_end=t_end,
            rng_seed=seed,
            hard_stop_limit=5000,
            max_species=getattr(params, 'max_species', 500)
        )

        # Interpolate to time grid if provided
        if time_grid is not None and len(result['time']) > 1:
            from scipy.interpolate import interp1d
            import numpy as np
            try:
                f = interp1d(result['time'], result['total_nt'],
                            kind='linear', bounds_error=False, fill_value='extrapolate')
                nt_interp = f(time_grid)
                result['nt_interp'] = nt_interp.tolist()
                result['time_grid'] = list(time_grid)
            except Exception:
                pass

        return {
            'primer_name': primer_name,
            'success': True,
            'result': result,
            'seed': seed,
            'error': None
        }

    except Exception as e:
        import traceback
        return {
            'primer_name': primer_name,
            'success': False,
            'result': None,
            'seed': seed,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@dataclass
class BatchStatus:
    """Aggregated status for batch of simulations."""
    total_jobs: int = 0
    completed_jobs: int = 0
    running_jobs: int = 0
    failed_jobs: int = 0
    total_products: int = 0
    total_nt: int = 0
    elapsed_seconds: float = 0.0
    jobs_per_second: float = 0.0
    eta_seconds: float = 0.0
    phase: str = 'init'
    primer_progress: Dict[str, int] = field(default_factory=dict)  # primer -> completed reps


class LiveQueryBatchRunner:
    """
    Multi-core batch runner for parallel simulation replicates.

    Uses ProcessPoolExecutor for true parallelism (bypasses Python GIL).
    Provides aggregate batch status for live query.

    Example usage:
        runner = LiveQueryBatchRunner(params, max_workers=27)
        runner.submit_batch(jobs)  # jobs = [(primer_name, primer_seqs, seed), ...]
        while not runner.is_complete():
            status = runner.get_batch_status()
            print(f"Progress: {status.completed_jobs}/{status.total_jobs}")
            time.sleep(5.0)
        results = runner.get_results()
    """

    def __init__(self, params: ParameterSet, max_workers: int = 27,
                 t_end: float = 1038.0, time_grid: Optional[np.ndarray] = None):
        """
        Initialize batch runner.

        Args:
            params: ParameterSet for all simulations
            max_workers: Number of parallel processes (default 27)
            t_end: Simulation end time
            time_grid: Optional time grid for interpolation
        """
        self.params = params
        self.max_workers = max_workers
        self.t_end = t_end
        self.time_grid = time_grid

        # Convert params to dict for pickling
        self._params_dict = {
            k: getattr(params, k) for k in params.__dataclass_fields__
            if not k.startswith('_') and getattr(params, k) is not None
        }
        # Handle special fields that might not be directly picklable
        for key in ['kon_base_by_L', 'Meff_intra_by_L', 'dG0_by_L', 'dG0_inter_by_L', 'toehold_rates']:
            if key in self._params_dict and self._params_dict[key] is not None:
                self._params_dict[key] = dict(self._params_dict[key])

        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures: Dict[Any, Tuple[str, int]] = {}  # future -> (primer_name, seed)
        self._results: List[Dict] = []
        self._lock = threading.Lock()

        self._start_time: Optional[float] = None
        self._total_jobs: int = 0
        self._completed_jobs: int = 0
        self._failed_jobs: int = 0
        self._primer_completed: Dict[str, int] = defaultdict(int)

    def submit_batch(self, jobs: List[Tuple[str, Dict[str, str], int]]) -> None:
        """
        Submit batch of simulation jobs.

        Args:
            jobs: List of (primer_name, primer_seqs, seed) tuples
        """
        from concurrent.futures import ProcessPoolExecutor

        with self._lock:
            self._total_jobs = len(jobs)
            self._completed_jobs = 0
            self._failed_jobs = 0
            self._results = []
            self._primer_completed.clear()
            self._start_time = time_module.time()

        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self._futures = {}

        for primer_name, primer_seqs, seed in jobs:
            args = (primer_name, primer_seqs, self._params_dict,
                    self.t_end, seed, self.time_grid)
            future = self._executor.submit(_batch_worker, args)
            self._futures[future] = (primer_name, seed)

    def poll_progress(self) -> BatchStatus:
        """
        Poll for completed jobs and update status.

        This should be called periodically to check progress.
        Returns current batch status.
        """
        from concurrent.futures import as_completed, FIRST_COMPLETED, wait

        if not self._futures:
            return self.get_batch_status()

        # Check for newly completed futures (non-blocking)
        done_futures = []
        for future in list(self._futures.keys()):
            if future.done():
                done_futures.append(future)

        # Process completed futures
        for future in done_futures:
            primer_name, seed = self._futures.pop(future)
            try:
                result = future.result(timeout=0.1)
                with self._lock:
                    self._results.append(result)
                    self._completed_jobs += 1
                    if result.get('success', False):
                        self._primer_completed[primer_name] += 1
                    else:
                        self._failed_jobs += 1
            except Exception as e:
                with self._lock:
                    self._completed_jobs += 1
                    self._failed_jobs += 1
                    self._results.append({
                        'primer_name': primer_name,
                        'success': False,
                        'error': str(e),
                        'seed': seed
                    })

        return self.get_batch_status()

    def get_batch_status(self) -> BatchStatus:
        """Get current batch status."""
        with self._lock:
            elapsed = time_module.time() - self._start_time if self._start_time else 0.0
            jobs_per_sec = self._completed_jobs / elapsed if elapsed > 0 else 0.0
            remaining = self._total_jobs - self._completed_jobs
            eta = remaining / jobs_per_sec if jobs_per_sec > 0 else float('inf')

            total_products = sum(
                r.get('result', {}).get('n_products', 0)
                for r in self._results if r.get('success')
            )
            total_nt = sum(
                r.get('result', {}).get('final_nt', 0)
                for r in self._results if r.get('success')
            )

            phase = 'init'
            if self._completed_jobs > 0:
                phase = 'running'
            if self._completed_jobs >= self._total_jobs:
                phase = 'complete'

            return BatchStatus(
                total_jobs=self._total_jobs,
                completed_jobs=self._completed_jobs,
                running_jobs=len(self._futures),
                failed_jobs=self._failed_jobs,
                total_products=total_products,
                total_nt=total_nt,
                elapsed_seconds=elapsed,
                jobs_per_second=jobs_per_sec,
                eta_seconds=eta,
                phase=phase,
                primer_progress=dict(self._primer_completed)
            )

    def is_complete(self) -> bool:
        """Check if all jobs are complete."""
        with self._lock:
            return self._completed_jobs >= self._total_jobs

    def get_results(self) -> List[Dict]:
        """Get all completed results."""
        with self._lock:
            return list(self._results)

    def wait_with_progress(self, callback: Optional[Callable[[BatchStatus], None]] = None,
                           poll_interval: float = 5.0) -> List[Dict]:
        """
        Wait for batch completion with periodic progress callbacks.

        Args:
            callback: Optional function to call with status updates
            poll_interval: Seconds between status checks

        Returns:
            List of all results
        """
        while not self.is_complete():
            status = self.poll_progress()
            if callback:
                callback(status)
            time_module.sleep(poll_interval)

        # Final poll
        status = self.poll_progress()
        if callback:
            callback(status)

        return self.get_results()

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        self.shutdown()


def run_batch_with_live_query(
    primer_jobs: List[Tuple[str, Dict[str, str]]],
    params: ParameterSet,
    n_reps: int = 10,
    t_end: float = 1038.0,
    max_workers: int = 27,
    time_grid: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[BatchStatus], None]] = None,
    poll_interval: float = 30.0
) -> Dict[str, List[Dict]]:
    """
    Run batch simulations for multiple primer sets with live query.

    This is the main entry point for comprehensive analysis workflows.

    Args:
        primer_jobs: List of (primer_name, primer_seqs) tuples
        params: ParameterSet for all simulations
        n_reps: Number of replicates per primer set
        t_end: Simulation end time
        max_workers: Number of parallel processes
        time_grid: Optional time grid for interpolation
        progress_callback: Optional callback for progress updates
        poll_interval: Seconds between status polls

    Returns:
        Dict mapping primer_name -> list of result dicts
    """
    # Build full job list with seeds
    all_jobs = []
    for primer_name, primer_seqs in primer_jobs:
        for rep in range(n_reps):
            # CHG-054: deterministic seed across processes. Python salts str
            # hashing per-process (PYTHONHASHSEED), so hash((name, rep)) broke
            # run-to-run reproducibility of "seeded" batches.
            seed = zlib.crc32(f"{primer_name}_{rep}".encode()) & 0x7FFFFFFF
            all_jobs.append((primer_name, primer_seqs, seed))

    print(f"Starting batch: {len(all_jobs)} jobs ({len(primer_jobs)} primers x {n_reps} reps)")
    print(f"  Workers: {max_workers}")
    print(f"  Simulation time: {t_end:.0f}s")

    # Run with progress
    runner = LiveQueryBatchRunner(params, max_workers, t_end, time_grid)
    runner.submit_batch(all_jobs)

    results = runner.wait_with_progress(
        callback=progress_callback or _default_progress_callback,
        poll_interval=poll_interval
    )

    runner.shutdown()

    # Organize results by primer
    by_primer: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if r.get('success'):
            by_primer[r['primer_name']].append(r['result'])

    return dict(by_primer)


def _default_progress_callback(status: BatchStatus) -> None:
    """Default progress callback that prints to stdout."""
    pct = 100.0 * status.completed_jobs / status.total_jobs if status.total_jobs > 0 else 0
    eta_min = status.eta_seconds / 60.0

    print(f"[{status.phase.upper()}] {status.completed_jobs}/{status.total_jobs} "
          f"({pct:.1f}%) | Products: {status.total_products:,} | "
          f"NT: {status.total_nt:,} | ETA: {eta_min:.1f}min")
