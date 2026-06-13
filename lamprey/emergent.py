"""
Emergent reaction mechanisms: TdT-tailing and product fitness scoring.

This module implements terminal transferase activity and sequence-feature-based 
fitness scoring for autocatalytic products.
"""

import numpy as np
from typing import Tuple
from .parameters import ParameterSet
from .utils import reverse_complement


def propensity_tdt_tailing(sequence: str, count: float, params: ParameterSet, rng: np.random.Generator) -> float:
    """
    Calculate propensity for TdT-tailing event.
    
    Universal mechanism - based on sequence features not primer identity.
    
    Args:
        sequence: Product sequence
        count: Number of molecules
        params: Parameter set
        rng: Random number generator
        
    Returns:
        Propensity (rate * count) for TdT event
    """
    if not params.enable_tdt_tailing or len(sequence) == 0:
        return 0.0
    
    # Base rate
    k_tdt = params.k_tdt_base
    
    # GC content modifier (lower GC = more breathing = more TdT)
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    gc_modifier = np.exp(-2.0 * (gc_content - 0.5))  # Peak at 50% GC
    
    # Length modifier (shorter products more prone to TdT)
    length_modifier = np.exp(-len(sequence) / 100.0)
    
    # 3' end purine content (purines favor TdT initiation)
    end_3p = sequence[-min(5, len(sequence)):]  # Last 5bp
    purine_count = end_3p.count('A') + end_3p.count('G')
    purine_3p = purine_count / len(end_3p) if len(end_3p) > 0 else 0.5
    purine_modifier = 1.0 + purine_3p  # 1.0-2.0x
    
    # Combined propensity
    k_tdt_effective = k_tdt * gc_modifier * length_modifier * purine_modifier
    
    return k_tdt_effective * count


def generate_purine_tail(length: int, params: ParameterSet, rng: np.random.Generator) -> str:
    """
    Generate purine-biased TdT tail sequence with empirical motif preferences.
    
    Args:
        length: Desired tail length (bp)
        params: Parameter set
        rng: Random number generator
        
    Returns:
        TdT tail sequence string
    """
    if length <= 0:
        return ""
    
    # Purine motifs from empirical analysis
    purine_motifs = ['AGA', 'GAG', 'AGG', 'GGA', 'AAG', 'GAA']
    
    seq = []
    i = 0
    
    while i < length:
        # Use motif or random base
        if rng.random() < params.tdt_motif_prob and i < length - 2:
            # Use a purine motif
            motif = rng.choice(purine_motifs)
            seq.extend(list(motif))
            i += 3
        else:
            # Single base with purine bias
            if rng.random() < params.tdt_purine_bias:
                # Purine (A or G, equal probability)
                seq.append(rng.choice(['A', 'G']))
            else:
                # Pyrimidine (rare)
                seq.append(rng.choice(['C', 'T']))
            i += 1
    
    # Truncate to exact length
    return ''.join(seq[:length])


def execute_tdt_tailing(sequence: str, params: ParameterSet, rng: np.random.Generator) -> str:
    """
    Execute TdT-tailing event by adding purine-biased tail to 3' end.
    
    Args:
        sequence: Current product sequence
        params: Parameter set
        rng: Random number generator
        
    Returns:
        New sequence with TdT tail appended
    """
    # Sample tail length from exponential distribution
    tail_length = max(1, int(rng.exponential(params.tdt_mean_length)))
    
    # Generate tail
    tail_seq = generate_purine_tail(tail_length, params, rng)
    
    # Append to sequence
    return sequence + tail_seq


def calculate_product_fitness(sequence: str, product_type: str, params: ParameterSet) -> float:
    """
    Calculate autocatalytic fitness score from sequence features.
    
    Higher score = more autocatalytic potential = faster amplification.
    
    Based on empirical observations:
    - Length: ≥75bp products are more effective
    - GC content: 50-60% optimal
    - Palindrome score: Self-complementarity aids folding
    - Type: Chimeric/TdT products have diverse binding sites
    
    Args:
        sequence: Product sequence
        product_type: Type classification ('canonical', 'chimeric', 'tdt_tailed', etc.)
        params: Parameter set
        
    Returns:
        Fitness score (typically 0-2 range)
    """
    if not params.enable_fitness_scoring or len(sequence) == 0:
        return 1.0  # Default neutral fitness
    
    length = len(sequence)
    gc_content = (sequence.count('G') + sequence.count('C')) / length
    
    # 1. Length fitness (sigmoid, peaks around inflection point)
    length_fitness = 1.0 / (1.0 + np.exp(-(length - params.fitness_length_inflection) / 20.0))
    
    # 2. GC fitness (Gaussian, peaked around optimal)
    gc_fitness = np.exp(-((gc_content - params.fitness_gc_optimal) / params.fitness_gc_width)**2)
    
    # 3. Palindrome score (self-complementarity)
    rc = reverse_complement(sequence)
    if len(rc) > 0:
        matches = sum(1 for i, c in enumerate(sequence) 
                      if i < len(rc) and c == rc[-(i+1)])
        palindrome_score = matches / length
    else:
        palindrome_score = 0.0
    
    # 4. Type modifier (chimeric/TdT have more diverse binding sites)
    type_fitness = 1.0
    if 'chimeric' in product_type.lower():
        type_fitness = params.fitness_boost_chimeric
    elif 'tdt' in product_type.lower():
        type_fitness = params.fitness_boost_tdt
    
    # Weighted combination
    base_fitness = (params.fitness_weight_length * length_fitness +
                   params.fitness_weight_gc * gc_fitness +
                   params.fitness_weight_palindrome * palindrome_score +
                   params.fitness_weight_base)
    
    # Apply type modifier
    return base_fitness * type_fitness


def classify_product_type(seq1: str, seq2: str, primer_names: list) -> str:
    """
    Classify product type based on component sequences.
    
    Args:
        seq1: First component sequence
        seq2: Second component sequence  
        primer_names: List of primer names involved
        
    Returns:
        Product type string ('canonical', 'chimeric', 'unknown')
    """
    # Simple heuristic: if primers are from expected pairs, canonical
    # This is a placeholder - actual classification would be more sophisticated
    canonical_pairs = {
        ('FIP', 'BIP'), ('BIP', 'FIP'),
        ('F3', 'FIP'), ('FIP', 'F3'),
        ('B3', 'BIP'), ('BIP', 'B3'),
    }
    
    if len(primer_names) >= 2:
        pair = (primer_names[0], primer_names[1])
        if pair in canonical_pairs:
            return 'canonical'
    
    return 'chimeric'  # Default to chimeric for unexpected combinations
