from typing import List, Dict, Tuple
import numpy as np
from ..datamodel import Segment

class Segmenter:
    def __init__(self, sequences: Dict[str, str], k: int = 5, skip: int = 1):
        self.sequences = sequences
        self.k = k
        self.skip = skip
        self.kmer_idx = self._build_kmer_index(sequences, k)

    def _build_kmer_index(self, seqs: Dict[str, str], k: int) -> Dict[str, List[Tuple[str, int]]]:
        idx: Dict[str, List[Tuple[str, int]]] = {}
        for label, seq in seqs.items():
            L = len(seq)
            for i in range(max(0, L - k + 1)):
                kmer = seq[i:i+k]
                idx.setdefault(kmer, []).append((label, i))
        return idx

    def segment(self, sequence: str) -> List[Segment]:
        # Simple matcher mirroring nanopore script behavior (no adapters here)
        segments: List[Segment] = []
        previous_end = 0
        i = 0
        L = len(sequence)
        while i + self.k <= L:
            kmer = sequence[i:i+self.k]
            matched = False
            if kmer in self.kmer_idx:
                best = None
                for label, pos in self.kmer_idx[kmer]:
                    ref = self.sequences[label]
                    match_len = self.k
                    rpos = i + self.k
                    spos = pos + self.k
                    total_mm = 0
                    cons_mm = 0
                    max_cons = 2
                    max_total = 3
                    while rpos < L and spos < len(ref):
                        if sequence[rpos] == ref[spos]:
                            match_len += 1
                            rpos += 1
                            spos += 1
                            cons_mm = 0
                        else:
                            total_mm += 1
                            cons_mm += 1
                            if cons_mm > max_cons or total_mm > max_total:
                                break
                            match_len += 1
                            rpos += 1
                            spos += 1
                    # simple upstream extension
                    up = 0
                    r_up = i - 1
                    s_up = pos - 1
                    while up < self.skip and r_up >= previous_end and s_up >= 0 and sequence[r_up] == ref[s_up]:
                        up += 1
                        r_up -= 1
                        s_up -= 1
                    seg_start = i - up
                    seg_end = i + match_len
                    length = seg_end - seg_start
                    if best is None or length > (best[1]-best[0]):
                        best = (seg_start, seg_end, label)
                if best:
                    if best[0] > previous_end:
                        segments.append(Segment(previous_end, best[0], 'Spacer'))
                    segments.append(Segment(best[0], best[1], best[2]))
                    previous_end = best[1]
                    i = best[1]
                    matched = True
            if not matched:
                i += 1
        if previous_end < L:
            segments.append(Segment(previous_end, L, 'Spacer'))
        return segments

