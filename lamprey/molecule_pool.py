"""
Molecule Pool Architecture for Unified SSA Engine

This module provides a unified representation for all molecular species
(primers and products) in the LAMP simulation, enabling cleaner
"products-as-primers" (Path 2) logic and live-query capabilities.

Key Classes:
- Molecule: Represents any molecular species (primer or product)
- BindingComplex: Represents a bound primer-template complex
- LineageEntry/LineageLog: Track product genealogy
- StratifiedCaps: Manage product limits by category
- MoleculePool: Unified pool for all molecules with propensity tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from collections import defaultdict
import numpy as np

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lamprey.sites import SiteIndex, InterSite, IntraSite, LoopSite


# =============================================================================
# Molecule Representation
# =============================================================================

@dataclass
class Molecule:
    """
    Represents a molecular species in the simulation.

    Can be a primer (is_primer=True) or a product (is_primer=False).
    Products track their origin and structural properties.
    """
    id: int                          # Unique molecule ID
    sequence: str                    # DNA sequence
    label: str                       # Human-readable label (e.g., 'FIP', 'FIP+BIP*4-123')
    family: str                      # Primer family for binding (e.g., 'FIP')
    is_primer: bool = True           # True if original primer, False if product
    count: float = 0.0               # Copy number (can be fractional for stochastic)

    # Product-specific fields
    template_id: Optional[int] = None       # Template this was synthesized from
    parent_id: Optional[int] = None         # Parent molecule ID
    generation: int = 0                     # Generation number (0 = primer)
    kind: str = 'primer'                    # 'primer', 'inter', 'intra', 'loop'
    binding_L: int = 0                      # Binding toehold length used

    # Structural properties
    length: int = 0                         # Sequence length
    is_autocatalytic: bool = False          # Has self-priming capability

    # Timestamps
    creation_time: float = 0.0              # Simulation time when created

    def __post_init__(self):
        if self.length == 0:
            self.length = len(self.sequence)


@dataclass
class BindingComplex:
    """
    Represents a primer/product bound to a template awaiting extension.

    Unified representation for inter, intra, and loop binding events.
    """
    id: int                          # Unique complex ID
    binder_id: int                   # ID of the binding molecule
    template_id: int                 # ID of the template being bound
    site_pos: int                    # Position on template where binding occurred
    L: int                           # Toehold length
    kind: str                        # 'inter', 'intra', or 'loop'
    family: str                      # Primer family of the binder

    # For loop complexes, store hairpin structure reference
    hairpin_info: Optional[Dict] = None

    # Binding time for kinetics
    bind_time: float = 0.0


# =============================================================================
# Lineage Tracking
# =============================================================================

@dataclass
class LineageEntry:
    """
    Records a single product emission event for lineage analysis.
    """
    time: float                      # Simulation time of emission
    product_id: int                  # ID of emitted product
    parent_id: int                   # ID of template/parent
    kind: str                        # 'inter', 'intra', 'loop'
    primer_family: str               # Which primer family initiated
    binding_L: int                   # Toehold length
    extension_length: int            # Nucleotides extended
    sequence: str                    # Product sequence
    is_autocatalytic: bool = False   # Product has self-priming capability
    binding_position: int = 0        # Position on template where binding occurred


class LineageLog:
    """
    Maintains complete lineage history for all products.

    Enables post-simulation analysis of cascade patterns and
    mechanistic pathway reconstruction.
    """

    def __init__(self):
        self.entries: List[LineageEntry] = []
        self._by_parent: Dict[int, List[LineageEntry]] = defaultdict(list)
        self._by_product: Dict[int, LineageEntry] = {}

    def record(self, entry: LineageEntry) -> None:
        """Record a new lineage entry."""
        self.entries.append(entry)
        self._by_parent[entry.parent_id].append(entry)
        self._by_product[entry.product_id] = entry

    def get_children(self, parent_id: int) -> List[LineageEntry]:
        """Get all products derived from a parent."""
        return self._by_parent.get(parent_id, [])

    def get_entry(self, product_id: int) -> Optional[LineageEntry]:
        """Get lineage entry for a specific product."""
        return self._by_product.get(product_id)

    def get_ancestry(self, product_id: int) -> List[LineageEntry]:
        """Trace ancestry back to original template."""
        ancestry = []
        current_id = product_id
        while current_id in self._by_product:
            entry = self._by_product[current_id]
            ancestry.append(entry)
            current_id = entry.parent_id
        return ancestry

    def count_by_kind(self) -> Dict[str, int]:
        """Count products by emission kind."""
        counts = defaultdict(int)
        for entry in self.entries:
            counts[entry.kind] += 1
        return dict(counts)

    def to_list(self) -> List[Tuple]:
        """Export as list of tuples for compatibility."""
        return [
            (e.time, {'type': e.kind, 'template_id': e.parent_id,
                      'primer': e.primer_family, 'L': e.binding_L},
             e.product_id, e.sequence)
            for e in self.entries
        ]


# =============================================================================
# Stratified Product Caps
# =============================================================================

@dataclass
class StratifiedCaps:
    """
    Manages stratified product limits to prevent simulation blowup
    while maintaining biological realism.

    Categories:
    - autocatalytic: Self-priming products (highest priority)
    - active: Products that can serve as templates
    - dormant: Dead-end products (lowest priority)
    """
    limit_autocatalytic: int = 1355
    limit_active: int = 500
    limit_dormant: int = 200

    count_autocatalytic: int = 0
    count_active: int = 0
    count_dormant: int = 0

    # Track which products are ghosted (over limit)
    ghosted_ids: Set[int] = field(default_factory=set)

    def classify_and_count(self, product_id: int, sequence: str,
                          is_intra: bool, primer_name: str) -> str:
        """
        Classify a product and update counts.

        Returns category and whether product should be ghosted.
        """
        category = self._classify(sequence, is_intra, primer_name)

        should_ghost = False
        if category == 'autocatalytic':
            self.count_autocatalytic += 1
            if self.count_autocatalytic > self.limit_autocatalytic:
                should_ghost = True
        elif category == 'active':
            self.count_active += 1
            if self.count_active > self.limit_active:
                should_ghost = True
        else:  # dormant
            self.count_dormant += 1
            if self.count_dormant > self.limit_dormant:
                should_ghost = True

        if should_ghost:
            self.ghosted_ids.add(product_id)

        return category

    def _classify(self, sequence: str, is_intra: bool, primer_name: str) -> str:
        """Classify product into autocatalytic, active, or dormant."""
        # Autocatalytic: intra products (can self-prime) or loop products
        if is_intra:
            return 'autocatalytic'

        # Active: inter products from inner primers (FIP, BIP)
        if primer_name in ('FIP', 'BIP', 'FIP_RC', 'BIP_RC'):
            return 'active'

        # Dormant: outer primer products (F3, B3) or loop primers
        return 'dormant'

    def is_ghosted(self, product_id: int) -> bool:
        """Check if a product is ghosted (over limit)."""
        return product_id in self.ghosted_ids

    def get_counts(self) -> Dict[str, int]:
        """Get current counts by category."""
        return {
            'autocatalytic': self.count_autocatalytic,
            'active': self.count_active,
            'dormant': self.count_dormant,
            'ghosted': len(self.ghosted_ids)
        }


# =============================================================================
# Unified Molecule Pool
# =============================================================================

class MoleculePool:
    """
    Unified pool for all molecular species in the simulation.

    Manages:
    - Primer and product molecules
    - Binding complexes (inter, intra, loop)
    - Copy number tracking
    - Propensity dirty flags for incremental updates

    This architecture enables "products-as-primers" (Path 2) naturally
    since all molecules are treated uniformly.
    """

    def __init__(self, primer_seqs: Dict[str, str], params: Any):
        """
        Initialize pool with primer sequences.

        Args:
            primer_seqs: Dict mapping primer names to sequences
            params: ParameterSet with simulation parameters
        """
        self.params = params
        self.primer_seqs = primer_seqs

        # Molecule storage
        self.molecules: Dict[int, Molecule] = {}
        self._next_mol_id: int = 0

        # Binding complexes
        self.complexes_inter: List[BindingComplex] = []
        self.complexes_intra: List[BindingComplex] = []
        self.complexes_loop: List[BindingComplex] = []
        self.complexes_product: List[BindingComplex] = []  # CHG-044: product-as-primer
        self._next_complex_id: int = 0

        # CHG-044: Product-as-primer counts by 3' k-mer
        self.product_primer_counts: Dict[str, float] = defaultdict(float)

        # Fast lookups
        self._primers: Dict[str, Molecule] = {}          # name -> primer molecule
        self._products: Dict[int, Molecule] = {}         # id -> product molecule
        self._by_sequence: Dict[str, int] = {}           # sequence -> mol_id (coalescence)
        self._by_family: Dict[str, List[int]] = defaultdict(list)  # family -> mol_ids

        # Counts for propensity calculations
        self.primer_counts: Dict[str, float] = {}        # primer_name -> count
        self.product_count: int = 0

        # Stratified caps
        self.stratified_caps = StratifiedCaps(
            limit_autocatalytic=getattr(params, 'limit_autocatalytic', 1355),
            limit_active=getattr(params, 'limit_active', 500),
            limit_dormant=getattr(params, 'limit_dormant', 200)
        )

        # Lineage tracking
        self.lineage = LineageLog()

        # Dirty flags for incremental propensity updates
        self._propensity_dirty: bool = True
        self._dirty_families: Set[str] = set()

        # Initialize primers
        self._init_primers()

    def _init_primers(self) -> None:
        """Initialize primer molecules from sequences."""
        count_scale = getattr(self.params, 'count_scale', 1e9)

        for name, seq in self.primer_seqs.items():
            # Skip RC sequences - they're handled as families
            if name.endswith('_RC'):
                continue

            # Determine initial count from concentration
            conc_attr = f'conc_{name}'
            conc = getattr(self.params, conc_attr, 0.4e-6)  # Default 0.4 µM
            # SSA counts are discrete; round to nearest integer count.
            initial_count = float(max(0, int(round(conc * count_scale))))

            # Create primer molecule
            mol = Molecule(
                id=self._next_mol_id,
                sequence=seq,
                label=name,
                family=name,
                is_primer=True,
                count=initial_count,
                generation=0,
                kind='primer',
                length=len(seq)
            )

            self.molecules[mol.id] = mol
            self._primers[name] = mol
            self.primer_counts[name] = initial_count
            self._by_family[name].append(mol.id)
            self._by_sequence[seq] = mol.id
            self._next_mol_id += 1

    def get_primer_count(self, name: str) -> float:
        """Get current count for a primer."""
        return self.primer_counts.get(name, 0.0)

    def consume_primer(self, name: str, amount: float = 1.0) -> bool:
        """
        Consume primer molecules (e.g., for binding).

        Returns True if successful, False if insufficient.
        """
        current = self.primer_counts.get(name, 0.0)
        if current < amount:
            return False
        self.primer_counts[name] = current - amount
        if name in self._primers:
            self._primers[name].count = self.primer_counts[name]
        self._propensity_dirty = True
        self._dirty_families.add(name)
        return True

    def restore_primer(self, name: str, amount: float = 1.0) -> None:
        """Restore primer molecules (e.g., on dissociation)."""
        self.primer_counts[name] = self.primer_counts.get(name, 0.0) + amount
        if name in self._primers:
            self._primers[name].count = self.primer_counts[name]
        self._propensity_dirty = True
        self._dirty_families.add(name)

    def add_product(self, sequence: str, label: str, family: str,
                   template_id: int, kind: str, binding_L: int,
                   creation_time: float, is_autocatalytic: bool = False,
                   coalesce: bool = True) -> Tuple[int, bool]:
        """
        Add a new product to the pool.

        Args:
            sequence: Product DNA sequence
            label: Human-readable label
            family: Primer family that initiated synthesis
            template_id: ID of template molecule
            kind: 'inter', 'intra', or 'loop'
            binding_L: Toehold length used for binding
            creation_time: Simulation time of creation
            is_autocatalytic: Whether product can self-prime
            coalesce: If True, merge with existing identical sequence

        Returns:
            (product_id, is_new): Product ID and whether it's newly created
        """
        # Check for coalescence
        if coalesce and sequence in self._by_sequence:
            existing_id = self._by_sequence[sequence]
            existing = self.molecules.get(existing_id)
            if existing:
                existing.count += 1.0
                return existing_id, False

        # Create new product
        mol = Molecule(
            id=self._next_mol_id,
            sequence=sequence,
            label=label,
            family=family,
            is_primer=False,
            count=1.0,
            template_id=template_id,
            generation=self._get_generation(template_id) + 1,
            kind=kind,
            binding_L=binding_L,
            length=len(sequence),
            is_autocatalytic=is_autocatalytic,
            creation_time=creation_time
        )

        self.molecules[mol.id] = mol
        self._products[mol.id] = mol
        self._by_sequence[sequence] = mol.id
        self._by_family[family].append(mol.id)
        self._next_mol_id += 1
        self.product_count += 1

        # Record lineage
        entry = LineageEntry(
            time=creation_time,
            product_id=mol.id,
            parent_id=template_id,
            kind=kind,
            primer_family=family,
            binding_L=binding_L,
            extension_length=len(sequence),  # Approximate
            sequence=sequence,
            is_autocatalytic=is_autocatalytic
        )
        self.lineage.record(entry)

        self._propensity_dirty = True
        return mol.id, True

    def _get_generation(self, template_id: int) -> int:
        """Get generation number of a template."""
        mol = self.molecules.get(template_id)
        return mol.generation if mol else 0

    # -------------------------------------------------------------------------
    # Complex Management
    # -------------------------------------------------------------------------

    def create_complex(self, kind: str, binder_id: int, template_id: int,
                       site_pos: int, L: int, family: str,
                       bind_time: float, hairpin_info: Optional[Dict] = None) -> BindingComplex:
        """Create a new binding complex."""
        cplx = BindingComplex(
            id=self._next_complex_id,
            binder_id=binder_id,
            template_id=template_id,
            site_pos=site_pos,
            L=L,
            kind=kind,
            family=family,
            hairpin_info=hairpin_info,
            bind_time=bind_time
        )
        self._next_complex_id += 1

        if kind == 'inter':
            self.complexes_inter.append(cplx)
        elif kind == 'intra':
            self.complexes_intra.append(cplx)
        elif kind == 'loop':
            self.complexes_loop.append(cplx)
        elif kind == 'product':
            self.complexes_product.append(cplx)

        self._propensity_dirty = True
        return cplx

    def remove_complex(self, kind: str, index: int) -> Optional[BindingComplex]:
        """Remove and return a complex by kind and index."""
        if kind == 'inter' and index < len(self.complexes_inter):
            cplx = self.complexes_inter.pop(index)
        elif kind == 'intra' and index < len(self.complexes_intra):
            cplx = self.complexes_intra.pop(index)
        elif kind == 'loop' and index < len(self.complexes_loop):
            cplx = self.complexes_loop.pop(index)
        elif kind == 'product' and index < len(self.complexes_product):
            cplx = self.complexes_product.pop(index)
        else:
            return None
        self._propensity_dirty = True
        return cplx

    def get_bound_count(self) -> int:
        """Get total number of bound complexes (for polymerase occupancy)."""
        return (len(self.complexes_inter) + len(self.complexes_intra)
                + len(self.complexes_loop) + len(self.complexes_product))

    # -------------------------------------------------------------------------
    # Propensity Helpers
    # -------------------------------------------------------------------------

    def is_dirty(self) -> bool:
        """Check if propensities need recalculation."""
        return self._propensity_dirty

    def clear_dirty(self) -> None:
        """Clear dirty flags after propensity recalculation."""
        self._propensity_dirty = False
        self._dirty_families.clear()

    def get_dirty_families(self) -> Set[str]:
        """Get families that have changed since last propensity calc."""
        return self._dirty_families

    # -------------------------------------------------------------------------
    # Snapshot and Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        return {
            'n_primers': len(self._primers),
            'n_products': self.product_count,
            'n_molecules': len(self.molecules),
            'n_complexes_inter': len(self.complexes_inter),
            'n_complexes_intra': len(self.complexes_intra),
            'n_complexes_loop': len(self.complexes_loop),
            'primer_counts': dict(self.primer_counts),
            'stratified_counts': self.stratified_caps.get_counts(),
            'lineage_counts': self.lineage.count_by_kind()
        }

    def get_product_sequences(self) -> List[str]:
        """Get all product sequences."""
        return [mol.sequence for mol in self._products.values()]
