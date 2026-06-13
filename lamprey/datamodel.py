from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Segment:
    start: int
    end: int
    label: str

@dataclass
class ReadSegments:
    header: str
    sequence: str
    segments: List[Segment]

@dataclass
class PrimerSet:
    prefix: str
    F3: str
    B3: str
    FIP: str
    BIP: str
    LF: Optional[str] = None
    LB: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        d = {k: v for k, v in vars(self).items() if v is not None and k not in ('prefix',)}
        # add reverse complements for matcher convenience
        from .utils import reverse_complement
        add = {}
        for k, v in d.items():
            add[k + '_RC'] = reverse_complement(v)
        d.update(add)
        return d

