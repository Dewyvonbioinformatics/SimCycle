from dataclasses import dataclass, field
from typing import Dict, Optional

AVOGADRO = 6.02214076e23

@dataclass
class ParameterSet:
    # polymerase and temperature parameters
    temperature_c: float = 65.0
    magnesium_mM: float = 8.0e-2
    sodium_mM: float = 150.0e-3

    # polymerase kinetics
    kcat_nt_per_s: float = 100.0
    Km_dNTP_M: float = 5.0e-5  # 50 µM default Michaelis constant for dNTP
    # Set default polymerase concentration to match actual LAMP conditions (~810 molecules in 1.7 pL)
    conc_polymerase_M: float = 8.1e-7  # ~810 polymerases in ~1.7 pL (10X less than previous)
    Kp_polymerase_counts: float = 25.0  # half-saturation for extension hazard scaling in counts

    # Kinetic calibration factors (balanced for 10x less polymerase)
    # With lower polymerase, maintain reasonable rate scales to allow productive amplification
    # OPT-001: LHS-optimized for inflection time rank correlation (ρ=1.0, r=0.985, CV=0.40)
    koff_scale: float = 2.751  # LHS-F1: scale down k_off to reflect stabilization by polymerase
    ext_rate_scale: float = 14.52  # LHS-F1: extension rate scaling
    intra_rate_scale: float = 480.6  # LHS-F1: intra (hairpin) closure rate
    inter_rate_scale: float = 0.0092  # LHS-F1: inter-molecular binding propensity scaling
    
    # DNAP Error Rate (per base per extension)
    dnap_error_rate: float = 1.0e-5  # ~1 error per 100kb (substitution/indel combined)

    # processivity modeled as geometric distribution mean (in nt)
    mean_processivity_nt: float = 250.0

    # characteristic commit length for extension hazard (nt) - OPT-001: +47% from 17.0
    commit_length_nt: float = 12.9  # LHS-F1: commit length for extension hazard

    # control early-stop for stagnation window in SSA (False for GUI streaming)
    allow_stagnation_stop: bool = False
    
    # Primer depletion probability (Config #60: 0.146)
    primer_depletion_prob: float = 0.146

    # GUI update interval (seconds of simulation time between updates)
    gui_update_interval: float = 0.5

    # concentrations (M)
    conc_FIP: float = 1.6e-6  # 1.6 µM - standard LAMP concentration
    conc_BIP: float = 1.6e-6  # 1.6 µM - standard LAMP concentration
    conc_LF: float = 0.4e-6   # Updated to realistic experimental concentration
    conc_LB: float = 0.4e-6   # Updated to realistic experimental concentration
    conc_F3: float = 0.2e-6
    conc_B3: float = 0.2e-6
    conc_dNTP_each: float = 1.4e-3

    # Volumes
    volume_phys_L: float = 1.7e-15  # Physical reaction volume (~1.7 pL)
    volume_sim_L: float = 1.7e-15   # Simulation volume; may be < volume_phys_L for speed

    # Effective scaling from molar concentration to SSA counts (dimensionless)
    # Recomputed in __post_init__ as N_A * volume_sim_L
    count_scale: float = field(default=0.0)

    # Base second-order association rates in M^-1 s^-1 (per L)
    # Used to derive stochastic counts-units rates by dividing by count_scale.
    kon_base_by_L: Optional[Dict[int, float]] = None  # if None, use defaults

    # Intramolecular effective molarity (M) as a simple function of loop-bin or L
    # For now: per-L effective molarity; can be refined later
    Meff_intra_by_L: Optional[Dict[int, float]] = None

    # Toehold thermodynamics gate parameters (occupancy/logistic mapping)
    dG0_by_L: Optional[Dict[int, float]] = None  # kcal/mol threshold for 50% occupancy (default / intra)
    dG0_inter_by_L: Optional[Dict[int, float]] = None  # Specific threshold for intermolecular (dimer) events
    dG_slope_kcal: float = 3.87  # LHS-F1: logistic slope scale in kcal/mol

    # Legacy placeholder per-length rates retained for backwards compatibility in some paths
    # Keys: L -> { 'kon_inter':..., 'koff_inter':..., 'k_close_intra':..., 'k_open_intra':..., 'alpha':... }
    toehold_rates: Optional[Dict[int, Dict[str, float]]] = None

    # Thermodynamics backend toggle
    use_nupack: bool = False
    nupack_material: str = 'dna'

    # Diffusion and accessibility scaling for growing complexes
    diffusion_reference_nt: float = 150.0  # reference length where diffusion factor = 1
    diffusion_exponent: float = 0.5        # exponent for D ~ (n / ref)^(-exponent)
    accessibility_mid_nt: float = 400.0    # midpoint length for accessibility logistic
    accessibility_slope: float = 0.01      # slope for accessibility logistic
    accessibility_floor: float = 0.05      # minimum accessibility factor

    # Structure-based 3' occlusion parameters
    use_3p_occlusion: bool = True          # Enable penalty for 3' end sequestration
    occlusion_window_nt: int = 100         # Window size to check for hairpins at 3' end
    occlusion_penalty_scale: float = 1.0   # Scaling factor for the occlusion penalty

    # OBJ-024: 3' Terminal Stability Penalty (Goldilocks Hypothesis)
    # Penalizes overly stable (GC-rich) 3' ends that bind non-specifically
    # and cause false positives. Optimal 3' dG is -1.0 to -1.6 kcal/mol.
    use_3p_stability_penalty: bool = True       # Enable 3' terminal stability penalty
    stability_penalty_L_max: int = 4            # Apply penalty only to L <= this value
    dG_3p_penalty_threshold: float = -2.0       # dG threshold (kcal/mol); more negative = penalty
    stability_penalty_strength: float = 0.3     # Minimum penalty factor (floor)
    stability_penalty_slope: float = 0.4        # Penalty slope per kcal/mol excess stability

    # TdT-tailing (Terminal Transferase Activity) parameters
    enable_tdt_tailing: bool = True
    k_tdt_base: float = 0.01  # Base propensity for TdT event
    tdt_purine_bias: float = 0.8  # Probability of purine vs pyrimidine
    tdt_mean_length: float = 5.0  # Mean length of tail
    tdt_rate_per_mol_s: float = 0.1  # Rate of TdT tailing per free primer molecule (s^-1)
    tdt_conc_scale: float = 1.0  # Scaling factor for TdT concentration effect
    tdt_motif_prob: float = 0.7            # Probability of using purine motif  vs random base

    # Product fitness scoring parameters
    enable_fitness_scoring: bool = True    # Enable sequence-feature-based fitness
    fitness_weight_length: float = 0.4     # Weight for length fitness component
    fitness_weight_gc: float = 0.3         # Weight for GC content fitness
    fitness_weight_palindrome: float = 0.2 # Weight for palindrome score
    fitness_weight_base: float = 0.1       # Base fitness (constant)
    fitness_boost_chimeric: float = 1.2    # Multiplier for chimeric products
    fitness_boost_tdt: float = 1.3         # Multiplier for TdT-tailed products
    fitness_length_inflection: float = 75.0  # Length (bp) where sigmoid peaks
    fitness_gc_optimal: float = 0.525      # Optimal GC content (52.5%)
    fitness_gc_width: float = 0.15         # GC tolerance width

    # Stratified product limits (prevents early termination from non-autocatalytic products)
    use_stratified_limits: bool = False    # Disable ghosting to allow exponential growth
    limit_autocatalytic: int = 150000      # Higher cap if limits are re-enabled
    limit_active: int = 50000              # Higher cap if limits are re-enabled
    limit_dormant: int = 20000             # Higher cap if limits are re-enabled

    # Reaction temperature for Tm-gating of hairpin stem stability (CHG-053)
    reaction_temperature_C: float = 65.0

    # OBJ-034 / CHG-051: dsDNA Segment Exclusion from Site Indexing
    # Hairpin (intra) products have dsDNA stems — sites within stem regions
    # are inaccessible for new binding events.  When enabled, the stem duplex
    # regions computed during _emit_hairpin_product() are passed to
    # update_with_product() which filters out occluded inter/intra sites.
    use_dsDNA_exclusion: bool = True

    # OBJ-035: Hairpin Loop Priming via Inter-Molecular Strand Displacement
    enable_loop_priming: bool = True        # Master toggle for loop priming
    min_loop_size_for_binding: int = 4      # Minimum loop nucleotides for priming
    loop_accessibility_factor: float = 0.5  # Steric penalty for binding to loop regions
    strand_displacement_rate_scale: float = 0.3  # Rate penalty for extension through dsDNA

    # CHG-042: Strand Displacement Product Release
    # When extension displaces a dsDNA stem, emit the displaced strand as free ssDNA
    enable_displacement_release: bool = True      # Master toggle for displaced strand release
    min_displacement_length: int = 8              # Minimum displaced nt to emit as product

    # CHG-043: Time-Compression Primer Attenuation
    # As products accumulate, attenuate primer-template propensities (inter and intra)
    # while preserving product-template propensities. Models competitive polymerase
    # inhibition and accelerates the primer→product kinetic transition.
    enable_primer_attenuation: bool = True
    primer_attenuation_rate: float = 0.002  # Attenuation per product: factor = 1/(1 + n_products * rate)

    # CHG-044: Path 2 — Product-as-Primer
    # Products can act as primers by annealing their 3' end to complementary
    # template sites and priming extension. This is the canonical LAMP
    # autocatalytic mechanism that enables exponential amplification.
    enable_product_as_primer: bool = True
    product_inter_rate_scale: float = 1.0   # Rate scale for product-as-primer binding
    product_koff_scale: float = 0.0175  # LHS-F1: Product complex koff scaling
    min_product_length_for_priming: int = 30  # Minimum product length to act as primer

    # CHG-050: Product phase time (X_P) — after this time, primer→primer-template
    # binding (inter and intra) is zeroed; only primer→product-template and all
    # product-initiated reactions proceed. None = disabled.
    product_phase_time: float = 219.0  # LHS-F1: product phase time (seconds)

    # Gibson-Bruck Next Reaction Method (NRM) optimization
    use_nrm: bool = False                  # Use NRM scheduler (O(log n)) vs Direct Method (O(n))

    def __post_init__(self):
        # Ensure simulation count scale is always consistent with the configured volume
        object.__setattr__(self, 'count_scale', AVOGADRO * float(self.volume_sim_L))

    @property
    def count_scale_phys(self) -> float:
        return AVOGADRO * float(self.volume_phys_L)

    @property
    def volume_ratio_sim_to_phys(self) -> float:
        phys = max(1e-30, float(self.volume_phys_L))
        return float(self.volume_sim_L) / phys

    @property
    def volume_ratio_phys_to_sim(self) -> float:
        sim = max(1e-30, float(self.volume_sim_L))
        return float(self.volume_phys_L) / sim


DEFAULT_PARAMETERS = ParameterSet(
    kon_base_by_L={  # M^-1 s^-1; rough defaults that decrease with shorter L
        3: 1e5, 4: 2e5, 5: 5e5, 6: 1e6, 7: 1e6, 8: 1e6,
        9: 1e6, 10: 1e6, 11: 1e6, 12: 1e6, 13: 1e6, 14: 1e6,
        15: 1e6, 16: 1e6, 17: 1e6, 18: 1e6, 19: 1e6, 20: 1e6,
    },
    Meff_intra_by_L={
        3: 8.0, 4: 3.0, 5: 1.0, 6: 0.5, 7: 0.2, 8: 0.1,  # ×100 boost for competitive hairpin formation
    },
    dG0_by_L={  # CHG-049: 1.5× scale from dG gate sweep (J=+0.54, best discrimination)
        3: -2.25, 4: -3.0, 5: -3.75, 6: -5.25, 7: -6.75, 8: -8.25,
        9: -9.75, 10: -11.25, 11: -12.75, 12: -14.25, 13: -15.75, 14: -17.25,
        15: -18.75, 16: -20.25, 17: -21.75, 18: -23.25, 19: -24.75, 20: -26.25,
    },
    dG0_inter_by_L={
        4: -7.0,  # Stricter threshold for 4bp dimers to suppress P06 artifact
    },
    toehold_rates={
        # Retained for compatibility elsewhere; not used in the new mechanistic SSA
        3: {"kon_inter": 1e-6, "koff_inter": 5.0, "k_close_intra": 0.05, "k_open_intra": 5.0, "alpha": 0.02},  # k_open_intra /2
        4: {"kon_inter": 2e-6, "koff_inter": 3.0, "k_close_intra": 0.08, "k_open_intra": 3.0,  "alpha": 0.04},  # k_open_intra /2
        5: {"kon_inter": 4e-6, "koff_inter": 2.0, "k_close_intra": 0.12, "k_open_intra": 1.5,  "alpha": 0.06},  # k_open_intra /2
        6: {"kon_inter": 8e-6, "koff_inter": 1.2, "k_close_intra": 0.18, "k_open_intra": 0.9, "alpha": 0.09},   # k_open_intra /2
        7: {"kon_inter": 1.5e-5, "koff_inter": 0.8, "k_close_intra": 0.25, "k_open_intra": 0.55, "alpha": 0.12}, # k_open_intra /2
        8: {"kon_inter": 2.5e-5, "koff_inter": 0.5, "k_close_intra": 0.35, "k_open_intra": 0.35, "alpha": 0.15}, # k_open_intra /2
    }
)
