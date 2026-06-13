#!/usr/bin/env python3
"""
Dynamic toehold site indexing for emergent autocatalysis.

This module tracks inter- and intra-molecular toehold sites as new products
are emitted. It supports sampling/removal upon binding and restoration upon
complex dissociation.

OPTIMIZATION (OBJ-034): Sites are indexed only at L=Lmin (3). When sampled,
the maximum complementary L is computed dynamically. This reduces storage
and indexing time by ~6x while ensuring each physical site contributes
exactly once to binding propensity.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from .utils import reverse_complement


@dataclass
class InterSite:
    """Inter-molecular binding site (primer/product -> template)."""
    template_id: int
    pos: int  # position in template where RC(3-mer) match starts
    family: str  # primer family label (e.g., 'FIP','BIP','F3','B3')
    free_multiplicity: int = 1  # number of currently free copies of this site across coalesced templates
    # L is computed dynamically via _expand_site_L(), not stored


@dataclass
class IntraSite:
    """Intra-molecular binding site (hairpin formation)."""
    template_id: int
    pos: int  # upstream match position where RC(3' 3-mer) starts
    free_multiplicity: int = 1  # number of currently free copies of this site across coalesced templates
    # L is computed dynamically via _expand_site_L(), not stored


@dataclass
class PositionDuplexState:
    """Track dsDNA state for a specific position range on a template (OBJ-034)."""
    start: int                    # Start of dsDNA region (inclusive)
    end: int                      # End of dsDNA region (exclusive)
    p_dsDNA: float = 1.0          # Probability this region is dsDNA (decremented on displacement)
    linked_template: Optional[int] = None  # Template ID of the linked complement
    linked_region: Optional[Tuple[int, int]] = None  # Corresponding region on linked template


@dataclass
class HairpinStructure:
    """Track hairpin topology for loop priming (OBJ-035)."""
    stem_5p_start: int    # 5' stem start position
    stem_5p_end: int      # 5' stem end (loop start)
    loop_start: int       # Loop start position (same as stem_5p_end)
    loop_end: int         # Loop end position (same as stem_3p_start)
    stem_3p_start: int    # 3' stem start (loop end)
    stem_3p_end: int      # 3' stem end
    stem_length: int      # Stem length in bp


@dataclass
class ProductInterSite:
    """CHG-044: Template site where a product's 3' end can anneal as primer."""
    template_id: int
    pos: int              # Position where RC(product_3p_kmer) match ends
    kmer: str             # The 3' k-mer of the product that matches here
    free_multiplicity: int = 1


@dataclass
class LoopSite:
    """Inter-molecular binding site within hairpin loop (OBJ-035)."""
    template_id: int
    pos: int              # Position in template where RC(lmer) starts
    family: str           # Primer family label
    hairpin: HairpinStructure  # Reference to hairpin structure
    free_multiplicity: int = 1  # number of currently free copies of this site across coalesced templates
    # L is computed dynamically


class SiteIndex:
    """
    Maintains dynamic lists of available toehold sites derived from current
    templates (initial primers plus emitted products).

    OPTIMIZATION: Sites are indexed only at L=Lmin (3). The maximum L is
    computed dynamically when sampling to determine actual binding strength.
    """
    def __init__(self, primer_seqs: Dict[str, str], Lmin: int = 3, Lmax_inter: int = 8,
                 min_loop_size: int = 4):
        # Template library: list of sequences and optional labels; index is template_id
        self.templates: List[str] = []
        self.templates_labels: List[Optional[str]] = []

        # Map from family -> list of InterSite (L=3 only indexing)
        self.inter_sites: Dict[str, List[InterSite]] = {}
        # Single list of IntraSite (L=3 only indexing)
        self.intra_sites: List[IntraSite] = []

        # Reverse mapping: template_id -> set of families/intra where this template has sites
        self.template_inter_keys: Dict[int, Set[str]] = defaultdict(set)
        self.template_has_intra: Set[int] = set()  # Templates with intra sites

        # Incremental stats tracking: key -> sum of template lengths
        self.inter_stats: Dict[str, float] = defaultdict(float)
        self.intra_stats: float = 0.0

        self.Lmin = int(Lmin)
        self.Lmax_inter = int(Lmax_inter)
        self.min_loop_size = int(min_loop_size)  # CHG-054: was hardcoded in _index_loop_sites
        # Save base family labels for scanning
        self.families = [l for l in primer_seqs.keys() if not l.endswith('_RC')]
        self.primer_seqs = primer_seqs

        # COALESCENCE: Sequence-to-tid lookup for O(1) duplicate detection
        self.seq_to_tid: Dict[str, int] = {}
        # COALESCENCE: Molecule counts per template (for exact lumping propensity scaling)
        self.template_counts: Dict[int, int] = {}

        # Incremental weighted site totals (O(1) lookup)
        self._weighted_inter_totals: Dict[str, float] = defaultdict(float)
        self._weighted_intra_total: float = 0.0

        # CHG-043: Primer template IDs for attenuation split
        self.primer_template_ids: Set[int] = set()

        # OBJ-034: Position-specific dsDNA regions per template
        self.template_duplex_positions: Dict[int, List[PositionDuplexState]] = defaultdict(list)

        # OBJ-035: Loop sites and hairpin structures
        self.loop_sites: Dict[str, List[LoopSite]] = {}
        self.template_loop_keys: Dict[int, Set[str]] = defaultdict(set)
        self._weighted_loop_totals: Dict[str, float] = defaultdict(float)
        self.template_hairpins: Dict[int, HairpinStructure] = {}

        # Site multiplicity bookkeeping (mass-action correct under coalescence)
        # Lookup: family -> (template_id, pos) -> site object
        self._inter_site_lookup: Dict[str, Dict[Tuple[int, int], InterSite]] = defaultdict(dict)
        self._loop_site_lookup: Dict[str, Dict[Tuple[int, int], LoopSite]] = defaultdict(dict)
        self._intra_site_lookup: Dict[Tuple[int, int], IntraSite] = {}

        # Per-template site references for O(n_sites_on_template) coalescence updates
        self._template_inter_sites: Dict[int, List[InterSite]] = defaultdict(list)
        self._template_intra_sites: Dict[int, List[IntraSite]] = defaultdict(list)
        self._template_loop_sites: Dict[int, List[LoopSite]] = defaultdict(list)

        # CHG-044: Product-as-primer (Path 2) site indexing
        # Sites are indexed by the product's 3' k-mer (Lmin bases)
        self._product_inter_sites: Dict[str, List[ProductInterSite]] = defaultdict(list)
        self._product_inter_site_lookup: Dict[str, Dict[Tuple[int, int], ProductInterSite]] = defaultdict(dict)
        self._weighted_product_inter_totals: Dict[str, float] = defaultdict(float)
        self._template_product_inter_sites: Dict[int, List[ProductInterSite]] = defaultdict(list)
        self._indexed_product_kmers: Set[str] = set()  # k-mers already scanned across all templates

    # ---------- Utilities ----------
    @staticmethod
    def _three_prime_lmer(seq: str, L: int) -> str:
        return seq[-L:] if L <= len(seq) else ''

    def _ensure_key(self, family: str) -> None:
        self.inter_sites.setdefault(family, [])
        self.loop_sites.setdefault(family, [])

    def _expand_site_L(self, template_id: int, pos: int, primer_3p_seq: str,
                        max_L: int = 8, template_seq: Optional[str] = None) -> int:
        """
        Expand from L=Lmin to find maximum matching L.

        Args:
            template_id: Template ID containing the site
            pos: Position where L=Lmin match ends (first nt AFTER the match)
            primer_3p_seq: The primer's 3' sequence to match against
            max_L: Maximum L to check (default 8)
            template_seq: Optional template sequence (avoids lookup if provided)

        Returns:
            Maximum L where RC(primer_3p_seq[-L:]) matches template[match_start:match_start+L]
        """
        templ = template_seq if template_seq is not None else self.templates[template_id]
        match_start = pos - self.Lmin  # Where the L=Lmin match starts on template

        for L in range(self.Lmin + 1, max_L + 1):
            # Check template bounds (match extends downstream from match_start)
            if match_start + L > len(templ):
                return L - 1
            if L > len(primer_3p_seq):
                return L - 1

            # Check if L-mer still matches (extending downstream on template)
            lmer = primer_3p_seq[-L:]
            rc = reverse_complement(lmer)
            if templ[match_start:match_start + L] != rc:
                return L - 1

        return max_L

    def _expand_intra_L(self, template_id: int, pos: int,
                         max_L: int = 8, template_seq: Optional[str] = None) -> int:
        """
        Expand from L=Lmin to find maximum matching L for intra-molecular site.

        For intra sites, we compare template's 3' end to the 5' stem position.
        The 5' stem starts at match_start and extends downstream.

        Args:
            template_id: Template ID
            pos: Position where L=Lmin match ends (first nt AFTER the 5' stem match)
            max_L: Maximum L to check
            template_seq: Optional template sequence

        Returns:
            Maximum L where stem can form
        """
        templ = template_seq if template_seq is not None else self.templates[template_id]
        n = len(templ)
        match_start = pos - self.Lmin  # Where the 5' stem L=Lmin match starts

        for L in range(self.Lmin + 1, min(max_L, n // 2) + 1):
            # Check if L-mer at 3' end matches RC at 5' stem (extending downstream)
            lmer_3p = templ[-L:]
            rc = reverse_complement(lmer_3p)

            # The 5' stem extends downstream from match_start
            if match_start + L > n:
                return L - 1

            if templ[match_start:match_start + L] != rc:
                return L - 1

            # Also check loop length constraint (minimum 3bp loop)
            stem_5p_end = match_start + L
            stem_3p_start = n - L
            loop_len = stem_3p_start - stem_5p_end
            if loop_len < 3:
                return L - 1

        return min(max_L, n // 2)

    # ---------- Template and site seeding ----------
    def add_template(self, sequence: str, label: Optional[str] = None, coalesce: bool = True) -> Tuple[int, bool]:
        """
        Add template or coalesce with existing identical sequence.

        Args:
            sequence: Template sequence
            label: Optional primer label
            coalesce: If True, merge with existing identical sequence

        Returns:
            (tid, is_new): Template ID and whether this is a new unique template.
        """
        # Check for existing identical sequence
        if coalesce and sequence in self.seq_to_tid:
            existing_tid = self.seq_to_tid[sequence]
            self.template_counts[existing_tid] = self.template_counts.get(existing_tid, 1) + 1
            templ_len = len(self.templates[existing_tid]) if existing_tid < len(self.templates) else 0

            # A new physical copy of the template contributes one additional free copy of each indexed site.
            # We model occupancy by decrementing/incrementing free_multiplicity per site.
            for s in self._template_inter_sites.get(existing_tid, []):
                s.free_multiplicity += 1
                self._weighted_inter_totals[s.family] += 1.0
                self.inter_stats[s.family] += templ_len

            for s in self._template_intra_sites.get(existing_tid, []):
                s.free_multiplicity += 1
                self._weighted_intra_total += 1.0
                self.intra_stats += templ_len

            for s in self._template_loop_sites.get(existing_tid, []):
                s.free_multiplicity += 1
                self._weighted_loop_totals[s.family] += 1.0

            # CHG-044: Increment product-primer site multiplicities
            for s in self._template_product_inter_sites.get(existing_tid, []):
                s.free_multiplicity += 1
                self._weighted_product_inter_totals[s.kmer] += 1.0

            return existing_tid, False

        # New unique sequence
        tid = len(self.templates)
        self.templates.append(sequence)
        self.templates_labels.append(label)
        self.seq_to_tid[sequence] = tid
        self.template_counts[tid] = 1
        return tid, True

    def get_count(self, tid: int) -> int:
        """Get molecule count for template."""
        return self.template_counts.get(tid, 1)

    def adjust_template_count(self, template_id: int, delta: int) -> None:
        """Adjust template copy-number and propagate to indexed site free multiplicities.

        This is used to keep template multiplicity consistent with an external count model
        (e.g., primer depletion or explicit removal/addition of template copies).

        Notes:
        - This adjusts *free* site-copies uniformly across all indexed sites on the template.
        - On negative deltas, free multiplicities are clamped at 0 (we do not model which
          occupied copies were removed).
        """
        if delta == 0:
            return
        if not (0 <= template_id < len(self.templates)):
            return

        old = int(self.template_counts.get(template_id, 1))
        new = int(max(0, old + int(delta)))
        actual_delta = new - old
        if actual_delta == 0:
            return

        templ = self.templates[template_id]
        templ_len = len(templ) if templ is not None else 0

        if actual_delta > 0:
            inc = int(actual_delta)
            self.template_counts[template_id] = new

            for s in self._template_inter_sites.get(template_id, []):
                s.free_multiplicity += inc
                self._weighted_inter_totals[s.family] += float(inc)
                self.inter_stats[s.family] += templ_len * inc

            for s in self._template_intra_sites.get(template_id, []):
                s.free_multiplicity += inc
                self._weighted_intra_total += float(inc)
                self.intra_stats += templ_len * inc

            for s in self._template_loop_sites.get(template_id, []):
                s.free_multiplicity += inc
                self._weighted_loop_totals[s.family] += float(inc)

            # CHG-044: product-primer sites
            for s in self._template_product_inter_sites.get(template_id, []):
                s.free_multiplicity += inc
                self._weighted_product_inter_totals[s.kmer] += float(inc)

            return

        # actual_delta < 0
        dec = int(-actual_delta)
        self.template_counts[template_id] = new

        for s in self._template_inter_sites.get(template_id, []):
            take = min(int(getattr(s, 'free_multiplicity', 0)), dec)
            if take <= 0:
                continue
            s.free_multiplicity -= take
            self._weighted_inter_totals[s.family] -= float(take)
            self.inter_stats[s.family] -= templ_len * take

        for s in self._template_intra_sites.get(template_id, []):
            take = min(int(getattr(s, 'free_multiplicity', 0)), dec)
            if take <= 0:
                continue
            s.free_multiplicity -= take
            self._weighted_intra_total -= float(take)
            self.intra_stats -= templ_len * take

        for s in self._template_loop_sites.get(template_id, []):
            take = min(int(getattr(s, 'free_multiplicity', 0)), dec)
            if take <= 0:
                continue
            s.free_multiplicity -= take
            self._weighted_loop_totals[s.family] -= float(take)

        # CHG-044: product-primer sites
        for s in self._template_product_inter_sites.get(template_id, []):
            take = min(int(getattr(s, 'free_multiplicity', 0)), dec)
            if take <= 0:
                continue
            s.free_multiplicity -= take
            self._weighted_product_inter_totals[s.kmer] -= float(take)

    def seed_from_primers(self, counts_by_label: Optional[Dict[str, int]] = None) -> None:
        """
        Seed inter- and intra-molecular sites using the initial primer sequences.
        Only indexes at L=Lmin for efficiency.

        Args:
            counts_by_label: Optional mapping primer label -> initial copy count to use for
                the corresponding primer template sequence.
        """
        # Add each real primer sequence as a template (skip *_RC helper sequences)
        for lbl, seq in self.primer_seqs.items():
            if lbl.endswith('_RC'):
                continue
            tid, _is_new = self.add_template(seq, label=lbl)
            self.primer_template_ids.add(tid)  # CHG-043: mark as primer template
            if counts_by_label is not None and lbl in counts_by_label:
                self.template_counts[tid] = int(max(0, int(counts_by_label[lbl])))

        L = self.Lmin  # Only scan for L=3

        # Build inter sites: for each family, find RC(3' L-mer) in every template
        for fam in self.families:
            fam_seq = self.primer_seqs.get(fam, '')
            if len(fam_seq) < L:
                continue
            lmer = self._three_prime_lmer(fam_seq, L)
            rc = reverse_complement(lmer)
            self._ensure_key(fam)

            for tid, templ in enumerate(self.templates):
                start = 0
                while True:
                    pos = templ.find(rc, start)
                    if pos == -1:
                        break
                    # pos is where RC starts; pos+L is where RC ends (first nt after match)
                    site_pos = pos + L
                    key = (tid, site_pos)
                    if key not in self._inter_site_lookup[fam]:
                        m = int(self.get_count(tid))
                        site = InterSite(template_id=tid, pos=site_pos, family=fam, free_multiplicity=m)
                        self.inter_sites[fam].append(site)
                        self._inter_site_lookup[fam][key] = site
                        self._template_inter_sites[tid].append(site)
                        self.template_inter_keys[tid].add(fam)
                        templ_len = len(templ)
                        self.inter_stats[fam] += templ_len * m
                        self._weighted_inter_totals[fam] += float(m)
                    start = pos + 1

        # Build intra sites: for each template, search upstream RC of 3' L-mer
        for tid, templ in enumerate(self.templates):
            n = len(templ)
            if n < L * 2 + 3:  # Need room for stem + loop
                continue
            lmer = self._three_prime_lmer(templ, L)
            rc = reverse_complement(lmer)
            # Search in region that leaves room for 3' stem and min loop
            region = templ[:-L]  # exclude terminal L-mer
            start = 0
            while True:
                pos = region.find(rc, start)
                if pos == -1:
                    break

                # Calculate loop length and skip if < 3bp
                stem_5p_end = pos + L
                stem_3p_start = n - L
                loop_len = stem_3p_start - stem_5p_end

                if loop_len < 3:
                    start = pos + 1
                    continue

                site_pos = pos + L
                key = (tid, site_pos)
                if key not in self._intra_site_lookup:
                    m = int(self.get_count(tid))
                    site = IntraSite(template_id=tid, pos=site_pos, free_multiplicity=m)
                    self.intra_sites.append(site)
                    self._intra_site_lookup[key] = site
                    self._template_intra_sites[tid].append(site)
                    self.template_has_intra.add(tid)
                    templ_len = len(templ)
                    self.intra_stats += templ_len * m
                    self._weighted_intra_total += float(m)
                start = pos + 1

    # ---------- Update with new product ----------
    def update_with_product(self, template_id: int,
                            blocked_intra_region: Optional[Tuple[int, int]] = None,
                            duplex_regions: Optional[List[PositionDuplexState]] = None,
                            hairpin: Optional[HairpinStructure] = None,
                            skip_intra: bool = False) -> None:
        """
        Scan the newly added template and insert sites accordingly.

        Args:
            template_id: The ID of the newly added template
            blocked_intra_region: Optional (start, end) tuple for regions to skip for intra sites
            duplex_regions: Optional list of dsDNA regions to exclude from site indexing (OBJ-034)
            hairpin: Optional hairpin structure for loop site indexing (OBJ-035)
            skip_intra: CHG-053 — if True, skip intra site indexing (dead-end hairpin)
        """
        templ = self.templates[template_id]
        templ_len = len(templ)
        L = self.Lmin

        # Store duplex regions if provided
        if duplex_regions:
            self.template_duplex_positions[template_id].extend(duplex_regions)

        # Store hairpin structure if provided
        if hairpin:
            self.template_hairpins[template_id] = hairpin
            # Index loop sites
            self._index_loop_sites(template_id, hairpin)

        # Inter: for each family, scan for RC(3' L-mer)
        for fam in self.families:
            fam_seq = self.primer_seqs.get(fam, '')
            if len(fam_seq) < L:
                continue
            lmer = self._three_prime_lmer(fam_seq, L)
            rc = reverse_complement(lmer)
            self._ensure_key(fam)

            start = 0
            while True:
                pos = templ.find(rc, start)
                if pos == -1:
                    break

                site_pos = pos + L  # Position where match ends

                # Check dsDNA exclusion (OBJ-034)
                if not self._is_position_accessible(template_id, pos, L, duplex_regions):
                    start = pos + 1
                    continue

                key = (template_id, site_pos)
                if key not in self._inter_site_lookup[fam]:
                    m = int(self.get_count(template_id))
                    site = InterSite(template_id=template_id, pos=site_pos, family=fam, free_multiplicity=m)
                    self.inter_sites[fam].append(site)
                    self._inter_site_lookup[fam][key] = site
                    self._template_inter_sites[template_id].append(site)
                    self.template_inter_keys[template_id].add(fam)
                    self.inter_stats[fam] += templ_len * m
                    self._weighted_inter_totals[fam] += float(m)
                start = pos + 1

        # Intra: hairpin sites anchored at 3' end
        # CHG-053: skip if stem-stability gate determined product is a dead-end hairpin
        n = len(templ)
        if skip_intra:
            pass  # No intra sites for dead-end hairpins
        elif n >= L * 2 + 3:
            lmer = self._three_prime_lmer(templ, L)
            rc = reverse_complement(lmer)
            region = templ[:-L]
            start = 0
            while True:
                pos = region.find(rc, start)
                if pos == -1:
                    break

                # Calculate loop length
                stem_5p_end = pos + L
                stem_3p_start = n - L
                loop_len = stem_3p_start - stem_5p_end

                if loop_len < 3:
                    start = pos + 1
                    continue

                # Check blocked region
                if blocked_intra_region is not None:
                    block_start, block_end = blocked_intra_region
                    site_start = pos
                    site_end = pos + L
                    if not (site_end <= block_start or site_start >= block_end):
                        start = pos + 1
                        continue

                # Check dsDNA exclusion for intra sites
                if not self._is_position_accessible(template_id, pos, L, duplex_regions):
                    start = pos + 1
                    continue

                site_pos = pos + L
                key = (template_id, site_pos)
                if key not in self._intra_site_lookup:
                    m = int(self.get_count(template_id))
                    site = IntraSite(template_id=template_id, pos=site_pos, free_multiplicity=m)
                    self.intra_sites.append(site)
                    self._intra_site_lookup[key] = site
                    self._template_intra_sites[template_id].append(site)
                    self.template_has_intra.add(template_id)
                    self.intra_stats += templ_len * m
                    self._weighted_intra_total += float(m)
                start = pos + 1

    def _is_position_accessible(self, template_id: int, pos: int, L: int,
                                 duplex_regions: Optional[List[PositionDuplexState]] = None) -> bool:
        """
        Check if site at pos with length L overlaps any dsDNA region (OBJ-034).

        Args:
            template_id: Template ID
            pos: Start position of the site
            L: Length of the site
            duplex_regions: Optional list of duplex regions to check (uses stored if None)

        Returns:
            True if site is accessible (not in dsDNA), False otherwise
        """
        regions = duplex_regions if duplex_regions is not None else self.template_duplex_positions.get(template_id, [])
        site_start = pos
        site_end = pos + L

        for dr in regions:
            # Check for overlap
            if not (site_end <= dr.start or site_start >= dr.end):
                # Site overlaps dsDNA region - check p_dsDNA
                if dr.p_dsDNA > 0.5:  # Consider blocked if mostly dsDNA
                    return False
        return True

    def get_site_accessibility(self, template_id: int, site_pos: int, site_L: int) -> float:
        """
        Return accessibility factor (0-1) for a site at given position (OBJ-034).

        Returns 1.0 if site is fully in ssDNA region.
        Returns (1 - p_dsDNA) if site overlaps a dsDNA region.
        """
        site_start = site_pos - site_L  # Convert from end position to start
        site_end = site_pos

        for pds in self.template_duplex_positions.get(template_id, []):
            if site_end <= pds.start or site_start >= pds.end:
                continue
            return 1.0 - pds.p_dsDNA
        return 1.0

    def handle_displacement_event(self, template_id: int,
                                  displacement_start: int,
                                  displacement_end: int) -> None:
        """
        Handle strand displacement through dsDNA region (OBJ-034).

        When template is extended through, the LINKED COMPLEMENT's corresponding
        positions have their P_dsDNA decreased.
        """
        for pds in self.template_duplex_positions.get(template_id, []):
            if displacement_end <= pds.start or displacement_start >= pds.end:
                continue

            overlap_start = max(displacement_start, pds.start)
            overlap_end = min(displacement_end, pds.end)

            if pds.linked_template is not None and pds.linked_region is not None:
                complement_tid = pds.linked_template
                link_start, link_end = pds.linked_region

                rel_start = overlap_start - pds.start
                rel_end = overlap_end - pds.start
                comp_displaced_start = link_start + rel_start
                comp_displaced_end = link_start + rel_end

                self._reduce_position_p_dsDNA(complement_tid, comp_displaced_start, comp_displaced_end)
                # Newly displaced regions can become accessible ssDNA; regenerate any sites that were
                # previously excluded due to dsDNA (OBJ-034).
                self.reindex_region(complement_tid, comp_displaced_start, comp_displaced_end)

    def _reduce_position_p_dsDNA(self, template_id: int, start: int, end: int,
                                  reduction: float = 1.0) -> None:
        """Reduce P_dsDNA for positions in the given range on template.

        When p_dsDNA reaches 0, clear the linkage to prevent further decrement attempts
        which could create negative probabilities (bug fix).
        """
        for pds in self.template_duplex_positions.get(template_id, []):
            if end <= pds.start or start >= pds.end:
                continue

            # Only decrement if there's something to decrement
            if pds.p_dsDNA > 0:
                pds.p_dsDNA = max(0.0, pds.p_dsDNA - reduction)

                # When p_dsDNA reaches 0, clear the link to prevent future decrement attempts
                if pds.p_dsDNA <= 0:
                    pds.linked_template = None
                    pds.linked_region = None

    def reindex_region(self, template_id: int, start: int, end: int,
                       blocked_intra_region: Optional[Tuple[int, int]] = None) -> None:
        """Re-scan a region for sites after dsDNA becomes accessible (OBJ-034).

        This is used after strand displacement reduces p_dsDNA for a region, which may
        unblock previously excluded sites.
        """
        if not (0 <= template_id < len(self.templates)):
            return

        templ = self.templates[template_id]
        if not templ:
            return

        n = len(templ)
        L = self.Lmin

        start = max(0, int(start))
        end = min(n, int(end))
        if end <= start:
            return

        # Match region [pos, pos+L) overlaps [start, end) iff pos < end and pos+L > start.
        scan_start = max(0, start - (L - 1))
        scan_end = min(n, end + (L - 1))
        if scan_end <= scan_start:
            return

        templ_len = n
        m = int(self.get_count(template_id))

        # Inter sites: scan for each primer family's RC(3' L-mer)
        for fam in self.families:
            fam_seq = self.primer_seqs.get(fam, '')
            if len(fam_seq) < L:
                continue

            lmer = self._three_prime_lmer(fam_seq, L)
            rc = reverse_complement(lmer)
            self._ensure_key(fam)

            sub = templ[scan_start:scan_end]
            idx = 0
            while True:
                pos = sub.find(rc, idx)
                if pos == -1:
                    break

                abs_pos = scan_start + pos
                if abs_pos < end and (abs_pos + L) > start:
                    if self._is_position_accessible(template_id, abs_pos, L, None):
                        site_pos = abs_pos + L
                        key = (template_id, site_pos)
                        if key not in self._inter_site_lookup[fam]:
                            site = InterSite(template_id=template_id, pos=site_pos, family=fam, free_multiplicity=m)
                            self.inter_sites[fam].append(site)
                            self._inter_site_lookup[fam][key] = site
                            self._template_inter_sites[template_id].append(site)
                            self.template_inter_keys[template_id].add(fam)
                            self.inter_stats[fam] += templ_len * m
                            self._weighted_inter_totals[fam] += float(m)

                idx = pos + 1

        # Intra sites: stem anchored at template 3' end
        if n >= L * 2 + 3:
            lmer_3p = self._three_prime_lmer(templ, L)
            rc_3p = reverse_complement(lmer_3p)
            region = templ[:-L]
            region_n = len(region)

            intra_scan_start = max(0, min(region_n, scan_start))
            intra_scan_end = max(0, min(region_n, scan_end))
            if intra_scan_end > intra_scan_start:
                sub = region[intra_scan_start:intra_scan_end]
                idx = 0
                while True:
                    pos = sub.find(rc_3p, idx)
                    if pos == -1:
                        break

                    abs_pos = intra_scan_start + pos
                    if abs_pos < end and (abs_pos + L) > start:
                        stem_5p_end = abs_pos + L
                        stem_3p_start = n - L
                        loop_len = stem_3p_start - stem_5p_end
                        if loop_len >= 3:
                            if blocked_intra_region is not None:
                                block_start, block_end = blocked_intra_region
                                site_start = abs_pos
                                site_end = abs_pos + L
                                if not (site_end <= block_start or site_start >= block_end):
                                    idx = pos + 1
                                    continue

                            if self._is_position_accessible(template_id, abs_pos, L, None):
                                site_pos = abs_pos + L
                                key = (template_id, site_pos)
                                if key not in self._intra_site_lookup:
                                    site = IntraSite(template_id=template_id, pos=site_pos, free_multiplicity=m)
                                    self.intra_sites.append(site)
                                    self._intra_site_lookup[key] = site
                                    self._template_intra_sites[template_id].append(site)
                                    self.template_has_intra.add(template_id)
                                    self.intra_stats += templ_len * m
                                    self._weighted_intra_total += float(m)

                    idx = pos + 1

    def _index_loop_sites(self, template_id: int, hairpin: HairpinStructure) -> None:
        """Index inter-molecular binding sites within hairpin loop (OBJ-035)."""
        if hairpin.loop_end - hairpin.loop_start < self.min_loop_size:
            return  # Loop too small

        templ = self.templates[template_id]
        L = self.Lmin

        for fam in self.families:
            fam_seq = self.primer_seqs.get(fam, '')
            if len(fam_seq) < L:
                continue
            lmer = self._three_prime_lmer(fam_seq, L)
            rc = reverse_complement(lmer)
            self._ensure_key(fam)

            # Search only within loop region
            loop_region = templ[hairpin.loop_start:hairpin.loop_end]
            start = 0
            while True:
                pos = loop_region.find(rc, start)
                if pos == -1:
                    break

                # Convert to absolute position
                abs_pos = hairpin.loop_start + pos + L

                key = (template_id, abs_pos)
                if key not in self._loop_site_lookup[fam]:
                    m = int(self.get_count(template_id))
                    site = LoopSite(
                        template_id=template_id,
                        pos=abs_pos,
                        family=fam,
                        hairpin=hairpin,
                        free_multiplicity=m
                    )
                    self.loop_sites[fam].append(site)
                    self._loop_site_lookup[fam][key] = site
                    self._template_loop_sites[template_id].append(site)
                    self.template_loop_keys[template_id].add(fam)
                    self._weighted_loop_totals[fam] += float(m)
                start = pos + 1

    # ---------- Counting methods ----------
    def inter_count(self, family: str) -> int:
        """Count active inter sites (unweighted)."""
        return len(self.inter_sites.get(family, []))

    def inter_count_weighted(self, family: str) -> float:
        """Count-weighted inter sites for exact lumping."""
        return self._weighted_inter_totals.get(family, 0.0)

    def inter_count_weighted_primer(self, family: str) -> float:
        """CHG-043: Count-weighted inter sites on primer templates only."""
        total = 0.0
        for s in self.inter_sites.get(family, []):
            if s.template_id in self.primer_template_ids:
                total += max(0, s.free_multiplicity)
        return total

    def intra_count(self) -> int:
        """Count active intra sites (unweighted)."""
        return len(self.intra_sites)

    def intra_count_weighted(self) -> float:
        """Count-weighted intra sites for exact lumping."""
        return self._weighted_intra_total

    def intra_count_weighted_primer(self) -> float:
        """CHG-043: Count-weighted intra sites on primer templates only."""
        total = 0.0
        for s in self.intra_sites:
            if s.template_id in self.primer_template_ids:
                total += max(0, s.free_multiplicity)
        return total

    def loop_count(self, family: str) -> int:
        """Count active loop sites (unweighted)."""
        return len(self.loop_sites.get(family, []))

    def loop_count_weighted(self, family: str) -> float:
        """Count-weighted loop sites for exact lumping."""
        return self._weighted_loop_totals.get(family, 0.0)

    def get_mean_length_inter(self, family: str) -> float:
        """Get mean length of templates contributing free inter-site copies (weighted)."""
        total_free = float(self.inter_count_weighted(family))
        if total_free <= 0:
            return 0.0
        return float(self.inter_stats.get(family, 0.0)) / total_free

    def get_mean_length_intra(self) -> float:
        """Get mean length of templates contributing free intra-site copies (weighted)."""
        total_free = float(self.intra_count_weighted())
        if total_free <= 0:
            return 0.0
        return float(self.intra_stats) / total_free

    def get_template_label(self, template_id: int) -> Optional[str]:
        if 0 <= template_id < len(self.templates_labels):
            return self.templates_labels[template_id]
        return None

    # ---------- Backward compatibility methods (deprecated, use new API) ----------
    def inter_count_old(self, family: str, L: int) -> int:
        """DEPRECATED: Count active inter sites for (family, L). Use inter_count(family) instead."""
        return len(self.inter_sites.get(family, []))

    def inter_count_weighted_old(self, family: str, L: int) -> float:
        """DEPRECATED: Count-weighted inter sites for (family, L). Use inter_count_weighted(family) instead."""
        return self._weighted_inter_totals.get(family, 0.0)

    def intra_count_old(self, L: int) -> int:
        """DEPRECATED: Count active intra sites for L. Use intra_count() instead."""
        return len(self.intra_sites)

    def intra_count_weighted_old(self, L: int) -> float:
        """DEPRECATED: Count-weighted intra sites for L. Use intra_count_weighted() instead."""
        return self._weighted_intra_total

    def get_mean_length_inter_old(self, family: str, L: int) -> float:
        """DEPRECATED: Get mean length for (family, L). Use get_mean_length_inter(family) instead."""
        return self.get_mean_length_inter(family)

    def get_mean_length_intra_old(self, L: int) -> float:
        """DEPRECATED: Get mean length for L. Use get_mean_length_intra() instead."""
        return self.get_mean_length_intra()

    def sample_inter_old(self, family: str, L: int, rng) -> Optional[InterSite]:
        """DEPRECATED: Sample inter site with L. Use sample_inter(family, rng) instead."""
        result = self.sample_inter(family, rng)
        if result is None:
            return None
        site, _ = result  # Ignore the computed L
        return site

    def sample_intra_old(self, L: int, rng) -> Optional[IntraSite]:
        """DEPRECATED: Sample intra site with L. Use sample_intra(rng) instead."""
        result = self.sample_intra(rng)
        if result is None:
            return None
        site, _ = result  # Ignore the computed L
        return site

    # ---------- Sampling methods (return site + computed L) ----------
    def sample_inter(self, family: str, rng,
                     primer_attenuation: float = 1.0) -> Optional[Tuple[InterSite, int]]:
        """
        Sample an inter-molecular site and compute its max L.

        This consumes exactly ONE free copy of the selected site (mass-action correct
        under template coalescence).

        Args:
            primer_attenuation: CHG-043 factor in (0,1] applied to primer-template
                site weights during sampling, so product-template sites are
                relatively more likely to be selected.

        Returns:
            (site, max_L) tuple or None if no sites available
        """
        lst = self.inter_sites.get(family, [])
        if not lst:
            return None

        weights = np.array([max(0, int(getattr(s, 'free_multiplicity', 0))) for s in lst], dtype=float)

        # CHG-043: attenuate primer-template site weights for sampling consistency
        if primer_attenuation < 1.0:
            for i, s in enumerate(lst):
                if s.template_id in self.primer_template_ids:
                    weights[i] *= primer_attenuation

        total = weights.sum()
        if total <= 0:
            return None

        p = weights / total
        idx = int(rng.choice(len(lst), p=p))
        site = lst[idx]
        if site.free_multiplicity <= 0:
            return None

        # Consume one free site-copy
        site.free_multiplicity -= 1

        # Update stats (one site-copy consumed)
        templ = self.templates[site.template_id]
        templ_len = len(templ)
        self.inter_stats[family] -= templ_len
        self._weighted_inter_totals[family] -= 1.0

        # Compute max L dynamically
        primer_seq = self.primer_seqs.get(family, '')
        max_L = self._expand_site_L(site.template_id, site.pos, primer_seq,
                                     self.Lmax_inter, templ)

        return site, max_L

    def restore_inter(self, site: InterSite) -> None:
        """Restore one free copy of an inter site after dissociation/extension."""
        self._ensure_key(site.family)

        key = (site.template_id, site.pos)
        existing = self._inter_site_lookup[site.family].get(key)
        if existing is None:
            existing = InterSite(template_id=site.template_id, pos=site.pos, family=site.family, free_multiplicity=0)
            self.inter_sites[site.family].append(existing)
            self._inter_site_lookup[site.family][key] = existing
            self._template_inter_sites[site.template_id].append(existing)

        self.template_inter_keys[site.template_id].add(site.family)

        max_free = int(self.get_count(site.template_id))
        if max_free <= 0:
            return
        if existing.free_multiplicity >= max_free:
            return

        existing.free_multiplicity += 1
        templ_len = len(self.templates[site.template_id])
        self.inter_stats[site.family] += templ_len
        self._weighted_inter_totals[site.family] += 1.0

    def sample_intra(self, rng,
                     primer_attenuation: float = 1.0) -> Optional[Tuple[IntraSite, int]]:
        """
        Sample an intra-molecular site and compute its max L.

        This consumes exactly ONE free copy of the selected site (mass-action correct
        under template coalescence).

        Args:
            primer_attenuation: CHG-043 factor in (0,1] applied to primer-template
                site weights during sampling.

        Returns:
            (site, max_L) tuple or None if no sites available
        """
        if not self.intra_sites:
            return None

        weights = np.array([max(0, int(getattr(s, 'free_multiplicity', 0))) for s in self.intra_sites], dtype=float)

        # CHG-043: attenuate primer-template intra site weights
        if primer_attenuation < 1.0:
            for i, s in enumerate(self.intra_sites):
                if s.template_id in self.primer_template_ids:
                    weights[i] *= primer_attenuation

        total = weights.sum()
        if total <= 0:
            return None

        p = weights / total
        idx = int(rng.choice(len(self.intra_sites), p=p))
        site = self.intra_sites[idx]
        if site.free_multiplicity <= 0:
            return None

        site.free_multiplicity -= 1

        # Update stats (one site-copy consumed)
        templ = self.templates[site.template_id]
        templ_len = len(templ)
        self.intra_stats -= templ_len
        self._weighted_intra_total -= 1.0

        # Compute max L dynamically
        max_L = self._expand_intra_L(site.template_id, site.pos, self.Lmax_inter, templ)  # CHG-054

        return site, max_L

    def restore_intra(self, site: IntraSite) -> None:
        """Restore one free copy of an intra site after dissociation/extension."""
        key = (site.template_id, site.pos)
        existing = self._intra_site_lookup.get(key)
        if existing is None:
            existing = IntraSite(template_id=site.template_id, pos=site.pos, free_multiplicity=0)
            self.intra_sites.append(existing)
            self._intra_site_lookup[key] = existing
            self._template_intra_sites[site.template_id].append(existing)

        self.template_has_intra.add(site.template_id)

        max_free = int(self.get_count(site.template_id))
        if max_free <= 0:
            return
        if existing.free_multiplicity >= max_free:
            return

        existing.free_multiplicity += 1
        templ_len = len(self.templates[site.template_id])
        self.intra_stats += templ_len
        self._weighted_intra_total += 1.0

    def sample_loop(self, family: str, rng) -> Optional[Tuple[LoopSite, int]]:
        """
        Sample a loop site and compute its max L (OBJ-035).

        This consumes exactly ONE free copy of the selected site (mass-action correct
        under template coalescence).

        Returns:
            (site, max_L) tuple or None if no sites available
        """
        lst = self.loop_sites.get(family, [])
        if not lst:
            return None

        weights = np.array([max(0, int(getattr(s, 'free_multiplicity', 0))) for s in lst], dtype=float)
        total = weights.sum()
        if total <= 0:
            return None

        p = weights / total
        idx = int(rng.choice(len(lst), p=p))
        site = lst[idx]
        if site.free_multiplicity <= 0:
            return None

        site.free_multiplicity -= 1
        self._weighted_loop_totals[family] -= 1.0

        # Compute max L dynamically
        primer_seq = self.primer_seqs.get(family, '')
        max_L = self._expand_site_L(site.template_id, site.pos, primer_seq,
                                     self.Lmax_inter, self.templates[site.template_id])

        return site, max_L

    def restore_loop(self, site: LoopSite) -> None:
        """Restore one free copy of a loop site after dissociation/extension."""
        self._ensure_key(site.family)

        key = (site.template_id, site.pos)
        existing = self._loop_site_lookup[site.family].get(key)
        if existing is None:
            existing = LoopSite(template_id=site.template_id, pos=site.pos, family=site.family,
                                hairpin=site.hairpin, free_multiplicity=0)
            self.loop_sites[site.family].append(existing)
            self._loop_site_lookup[site.family][key] = existing
            self._template_loop_sites[site.template_id].append(existing)

        self.template_loop_keys[site.template_id].add(site.family)

        max_free = int(self.get_count(site.template_id))
        if max_free <= 0:
            return
        if existing.free_multiplicity >= max_free:
            return

        existing.free_multiplicity += 1
        self._weighted_loop_totals[site.family] += 1.0

    # ---------- CHG-044: Product-as-Primer (Path 2) Site Indexing ----------

    def index_product_primer_kmer(self, kmer: str) -> None:
        """
        Scan ALL existing templates for RC(kmer) matches and register
        ProductInterSite entries.  Called once per unique product 3' k-mer.
        """
        if kmer in self._indexed_product_kmers:
            return
        self._indexed_product_kmers.add(kmer)

        L = len(kmer)
        rc = reverse_complement(kmer)

        for tid, templ in enumerate(self.templates):
            if not templ:
                continue
            self._scan_template_for_product_kmer(tid, templ, kmer, rc, L)

    def index_product_primer_on_template(self, template_id: int) -> None:
        """
        Scan a newly added template for all already-known product 3' k-mers.
        Called when a new product template is added to the site index.
        """
        if not (0 <= template_id < len(self.templates)):
            return
        templ = self.templates[template_id]
        if not templ:
            return

        L = self.Lmin
        for kmer in self._indexed_product_kmers:
            rc = reverse_complement(kmer)
            self._scan_template_for_product_kmer(template_id, templ, kmer, rc, L)

    def _scan_template_for_product_kmer(self, tid: int, templ: str,
                                         kmer: str, rc: str, L: int) -> None:
        """Scan a single template for RC(kmer) matches and register sites."""
        m = int(self.get_count(tid))
        start = 0
        while True:
            pos = templ.find(rc, start)
            if pos == -1:
                break
            site_pos = pos + L
            key = (tid, site_pos)
            if key not in self._product_inter_site_lookup[kmer]:
                site = ProductInterSite(
                    template_id=tid, pos=site_pos,
                    kmer=kmer, free_multiplicity=m
                )
                self._product_inter_sites[kmer].append(site)
                self._product_inter_site_lookup[kmer][key] = site
                self._template_product_inter_sites[tid].append(site)
                self._weighted_product_inter_totals[kmer] += float(m)
            start = pos + 1

    def product_inter_count_weighted(self, kmer: str) -> float:
        """Count-weighted product-primer inter sites for a given 3' k-mer."""
        return self._weighted_product_inter_totals.get(kmer, 0.0)

    def sample_product_inter(self, kmer: str, rng,
                             product_3p_seq: Optional[str] = None
                             ) -> Optional[Tuple[ProductInterSite, int]]:
        """
        Sample a product-primer inter site and compute max L.

        Consumes one free copy of the selected site.

        Args:
            kmer: The 3' k-mer (Lmin bases) for grouping.
            rng: Random number generator.
            product_3p_seq: Full 3' end of the product (up to Lmax bases)
                for accurate L expansion. If None, uses the kmer alone.

        Returns:
            (site, max_L) tuple or None if no sites available
        """
        lst = self._product_inter_sites.get(kmer, [])
        if not lst:
            return None

        weights = np.array([max(0, s.free_multiplicity) for s in lst], dtype=float)
        total = weights.sum()
        if total <= 0:
            return None

        p = weights / total
        idx = int(rng.choice(len(lst), p=p))
        site = lst[idx]
        if site.free_multiplicity <= 0:
            return None

        # Consume one free site-copy
        site.free_multiplicity -= 1
        self._weighted_product_inter_totals[kmer] -= 1.0

        # Compute max L using full product 3' sequence for accurate expansion
        templ = self.templates[site.template_id]
        expand_seq = product_3p_seq if product_3p_seq else kmer
        max_L = self._expand_site_L(site.template_id, site.pos, expand_seq,
                                     self.Lmax_inter, templ)

        return site, max_L

    def restore_product_inter(self, site: ProductInterSite) -> None:
        """Restore one free copy of a product-primer site after dissociation."""
        key = (site.template_id, site.pos)
        existing = self._product_inter_site_lookup[site.kmer].get(key)
        if existing is None:
            existing = ProductInterSite(
                template_id=site.template_id, pos=site.pos,
                kmer=site.kmer, free_multiplicity=0
            )
            self._product_inter_sites[site.kmer].append(existing)
            self._product_inter_site_lookup[site.kmer][key] = existing
            self._template_product_inter_sites[site.template_id].append(existing)

        max_free = int(self.get_count(site.template_id))
        if max_free <= 0:
            return
        if existing.free_multiplicity >= max_free:
            return

        existing.free_multiplicity += 1
        self._weighted_product_inter_totals[site.kmer] += 1.0

    def remove_template(self, template_id: int) -> None:
        """Permanently remove a template and all its indexed sites."""
        if not (0 <= template_id < len(self.templates)):
            return

        old_seq = self.templates[template_id]
        templ_len = len(old_seq) if old_seq is not None else 0

        # Remove inter sites (subtract free multiplicity, then filter lists)
        for s in self._template_inter_sites.get(template_id, []):
            m = int(getattr(s, 'free_multiplicity', 0))
            if m <= 0:
                continue
            self.inter_stats[s.family] -= templ_len * m
            self._weighted_inter_totals[s.family] -= float(m)
            self._inter_site_lookup[s.family].pop((template_id, s.pos), None)

        if template_id in self.template_inter_keys:
            for fam in self.template_inter_keys[template_id]:
                if fam in self.inter_sites:
                    self.inter_sites[fam] = [x for x in self.inter_sites[fam] if x.template_id != template_id]
            del self.template_inter_keys[template_id]
        self._template_inter_sites.pop(template_id, None)

        # Remove intra sites
        for s in self._template_intra_sites.get(template_id, []):
            m = int(getattr(s, 'free_multiplicity', 0))
            if m <= 0:
                continue
            self.intra_stats -= templ_len * m
            self._weighted_intra_total -= float(m)
            self._intra_site_lookup.pop((template_id, s.pos), None)

        self.intra_sites = [x for x in self.intra_sites if x.template_id != template_id]
        self.template_has_intra.discard(template_id)
        self._template_intra_sites.pop(template_id, None)

        # Remove loop sites
        for s in self._template_loop_sites.get(template_id, []):
            m = int(getattr(s, 'free_multiplicity', 0))
            if m <= 0:
                continue
            self._weighted_loop_totals[s.family] -= float(m)
            self._loop_site_lookup[s.family].pop((template_id, s.pos), None)

        if template_id in self.template_loop_keys:
            for fam in self.template_loop_keys[template_id]:
                if fam in self.loop_sites:
                    self.loop_sites[fam] = [x for x in self.loop_sites[fam] if x.template_id != template_id]
            del self.template_loop_keys[template_id]
        self._template_loop_sites.pop(template_id, None)

        # Remove product-primer sites (CHG-044)
        for s in self._template_product_inter_sites.get(template_id, []):
            m = int(getattr(s, 'free_multiplicity', 0))
            if m > 0:
                self._weighted_product_inter_totals[s.kmer] -= float(m)
            self._product_inter_site_lookup[s.kmer].pop((template_id, s.pos), None)
        for kmer in list(self._product_inter_sites.keys()):
            self._product_inter_sites[kmer] = [
                x for x in self._product_inter_sites[kmer] if x.template_id != template_id
            ]
        self._template_product_inter_sites.pop(template_id, None)

        # Remove per-template auxiliary structures
        self.template_duplex_positions.pop(template_id, None)
        self.template_hairpins.pop(template_id, None)

        # Remove count bookkeeping
        self.template_counts.pop(template_id, None)

        # Remove coalescence mapping for this exact sequence
        if old_seq in self.seq_to_tid and self.seq_to_tid.get(old_seq) == template_id:
            del self.seq_to_tid[old_seq]

        # Clear template data
        self.templates[template_id] = ""
        self.templates_labels[template_id] = None
