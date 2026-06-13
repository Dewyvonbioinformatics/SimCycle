"""
Arbitrary-format primer loader for LAMPrey.

The engine consumes a flat ``Dict[str, str]`` of uppercase sequences that includes
both each primer and its reverse complement (``<NAME>`` and ``<NAME>_RC``). This
module converts primers given in flexible formats into that canonical dict:

  - a CSV/TSV file with columns for F3/B3/FIP/BIP (LF/LB optional), one set per row
  - a Python dict ``{"F3": "...", "BIP": "...", ...}``

Column names are matched case-insensitively and ignore spaces/underscores, so
``FIP``, ``fip``, ``F.I.P`` -> ``FIP`` etc. Use as a library or a CLI:

    python tools/load_primers.py mydb.csv --name MySet
"""
from __future__ import annotations

import argparse
import csv
from typing import Dict, Optional, Mapping

from lamprey.utils import reverse_complement

# Canonical LAMP primer roles. F3/B3/FIP/BIP are required; LF/LB are optional.
REQUIRED = ("F3", "B3", "FIP", "BIP")
OPTIONAL = ("LF", "LB")
ALL_ROLES = REQUIRED + OPTIONAL

# Accepted aliases (canonicalized: uppercased, stripped of spaces/dots/underscores).
_ALIASES = {
    "F3": "F3", "B3": "B3",
    "FIP": "FIP", "BIP": "BIP",
    "LF": "LF", "LB": "LB",
    "LOOPF": "LF", "LOOPB": "LB", "LPF": "LF", "LPB": "LB",
}
_NAME_COLS = {"NAME", "PRIMERSET", "SET", "PREFIX", "ID", "INDEX"}


def _canon(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def primers_to_engine_dict(primers: Mapping[str, str]) -> Dict[str, str]:
    """Convert ``{role: sequence}`` to the engine's ``{name, name_RC}`` dict."""
    out: Dict[str, str] = {}
    missing = [r for r in REQUIRED if not primers.get(r)]
    if missing:
        raise ValueError(f"Missing required primer(s): {', '.join(missing)}")
    for role in ALL_ROLES:
        seq = primers.get(role)
        if not seq:
            continue
        s = "".join(seq.split()).upper()
        out[role] = s
        out[f"{role}_RC"] = reverse_complement(s)
    return out


def _load_table(path: str, name: Optional[str]) -> Dict[str, str]:
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path!r}")

        # Map canonical-column -> actual header for primer roles and the name col.
        role_col: Dict[str, str] = {}
        name_col: Optional[str] = None
        for col in reader.fieldnames:
            c = _canon(col)
            if c in _ALIASES:
                role_col[_ALIASES[c]] = col
            elif c in _NAME_COLS and name_col is None:
                name_col = col

        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path!r}")

    if name is not None:
        if name_col is None:
            raise ValueError(
                f"--name given but no name column found in {path!r} "
                f"(looked for {sorted(_NAME_COLS)})"
            )
        match = [r for r in rows if (r.get(name_col) or "").strip() == name]
        if not match:
            avail = sorted({(r.get(name_col) or "").strip() for r in rows})
            raise ValueError(f"Primer set {name!r} not found. Available: {avail}")
        row = match[0]
    else:
        row = rows[0]

    return {role: (row.get(col) or "").strip() for role, col in role_col.items()}


def load_primers(source, name: Optional[str] = None) -> Dict[str, str]:
    """Load primers from a CSV/TSV path or a dict into the engine's input dict.

    Args:
        source: path to a CSV/TSV file, or a ``{role: sequence}`` mapping.
        name:   when ``source`` is a multi-row file, the primer-set name to select
                (matched against a name/set/prefix column). Defaults to the first row.
    """
    if isinstance(source, Mapping):
        return primers_to_engine_dict(source)
    return primers_to_engine_dict(_load_table(str(source), name))


def main() -> None:
    ap = argparse.ArgumentParser(description="Load primers into the LAMPrey engine dict.")
    ap.add_argument("source", help="CSV/TSV file with F3/B3/FIP/BIP (LF/LB optional)")
    ap.add_argument("--name", default=None, help="primer-set name to select (multi-row files)")
    args = ap.parse_args()
    d = load_primers(args.source, args.name)
    width = max(len(k) for k in d)
    for k in sorted(d):
        print(f"{k:<{width}}  {d[k]}")


if __name__ == "__main__":
    main()
