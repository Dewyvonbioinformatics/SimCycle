# LAMPrey

**A mechanistic, template-free LAMP reaction kinetic simulator.**

LAMPrey is a Gillespie Stochastic Simulation Algorithm (SSA) engine that predicts
**spurious (no-target) amplification** in Loop-mediated isothermal AMPlification
(LAMP) reactions directly from primer sequences. Given a primer set, it simulates the
elementary biochemical events of the reaction — hairpin closure, intermolecular
binding, polymerase extension, and the products-as-primers cascade — and produces
the discrete products, amplification kinetics, and lineage that emerge with **no
target DNA present**.

---

## Table of contents
- [Background: LAMP and spurious amplification](#background-lamp-and-spurious-amplification)
- [What the engine models](#what-the-engine-models)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Primer input (arbitrary formats)](#primer-input-arbitrary-formats)
- [Simulation output schema](#simulation-output-schema)
- [Tools](#tools)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Biological glossary](#biological-glossary)
- [License & notes](#license--notes)

---

## Background: LAMP and spurious amplification

**Loop-mediated isothermal amplification (LAMP)** is a method for copying a specific DNA sequence millions-fold at a single constant temperature (around 65 °C), without the heating-and-cooling cycles that PCR requires. Because it needs only a heat block or even body heat, LAMP is widely used in point-of-care and field diagnostics — for example, rapid tests for viral or bacterial pathogens. A positive reaction is read out as a color change or a rise in fluorescence as new DNA accumulates.

LAMP achieves this with a set of **six primers** — short, single-stranded DNA oligonucleotides that recognize eight distinct regions of the target. Two outer primers (**F3, B3**) and two loop primers (**LF, LB**) are simple linear oligos; the two inner primers (**FIP, BIP**) are *composite* — each is two target-binding segments fused into one strand. This design is what lets the reaction fold its products into a characteristic **dumbbell** (stem-loop) structure. Once the first dumbbell forms, a strand-displacing polymerase repeatedly extends from the loops, and each new product becomes a template and a primer for the next round. This self-feeding **cascade** is what makes the reaction so fast and so sensitive.

That same sensitivity is also LAMP's biggest liability. The six primers are highly concentrated and partially self-complementary, so even **with no target DNA present** they can occasionally hybridize to one another, get extended, and seed the same self-amplifying cascade. The result is **template-free (spurious) amplification** — a strong signal produced from primer artifacts alone. In a diagnostic setting this is a **false positive**: the test reports a pathogen that was never there.

Understanding and predicting which primer sets are prone to this — and *why* — is therefore central to designing reliable LAMP assays. This is the problem LAMPrey addresses: it simulates the no-target reaction from the primer sequences alone and predicts whether, and how fast, spurious amplification will take off.

## What the engine models

LAMPrey is a **mechanistic stochastic simulation** of a single no-template LAMP reaction. Every molecule in the virtual tube — the six starting primers and every product they generate — is represented as a **single-stranded DNA species** defined by its sequence and a copy count. The simulator does not assume a reaction network in advance; instead it discovers products as they form, treating primers and products identically in one shared molecule pool (so a product can later act as a primer, exactly as in real LAMP).

At each step the engine enumerates the reactions physically available to the current population and the rate ("propensity") of each, then draws which reaction fires and how much time elapses, using the Gillespie stochastic simulation algorithm. The reactions it models are the elementary biochemical events of LAMP: **intramolecular hairpin closure** (a strand folding back to pair with itself, the seed of dumbbell formation), **intermolecular binding** (two separate strands annealing through a short complementary toehold), **primer extension** (a strand-displacing polymerase copying along a template from a bound 3′ end), and the **products-as-primers cascade** (the autocatalytic feedback in which an extended product anneals to and primes yet another strand). Thermodynamics (binding free energy at the reaction temperature), polymerase and nucleotide availability, and competition between hairpin closure and intermolecular binding all shape the propensities.

The simulation is judged a **success** when its predictions match experiment in three ways: the **fraction of replicates that amplify** matches the lab's observed positive rate for each primer set; the simulated **accumulation-over-time curves** resemble the measured real-time fluorescence curves; and the **discrete product sequences and the order in which they form** match the segment patterns seen in empirical **nanopore reads** of real spurious-amplification products.

---

## Installation

LAMPrey is pure Python; the engine core depends only on `numpy` (and `scipy` for batch
time-grid interpolation). The visualization tools additionally use `matplotlib` and
`networkx`.

```bash
git clone <your-repo-url> lamprey
cd lamprey

# Recommended: a fresh environment
python -m venv .venv && source .venv/bin/activate     # or: conda create -n lamprey python=3.11

# Option A — editable install (exposes `import lamprey` everywhere)
pip install -e ".[viz]"

# Option B — just the dependencies (run scripts from the repo root)
pip install -r requirements.txt
```

Python ≥ 3.9 is required. The scripts in `examples/` and `tools/` also work without an
install when run from the repo root (they bootstrap `sys.path` themselves).

## Quickstart

```bash
# Run one template-free replicate on the bundled synthetic primer set,
# then visualize the result three ways.
python examples/run_simulation.py
python tools/plot_amplification.py    examples/demo_result.pkl --out amplification.png
python tools/plot_lineage.py          examples/demo_result.pkl --out lineage.png
python tools/plot_autocat_network.py  examples/demo_result.pkl --out autocat_network.png
```

Or from Python:

```python
from lamprey import simulate_replicate_unified, DEFAULT_PARAMETERS
from tools.load_primers import load_primers

primer_seqs = load_primers("examples/data/demo_primers.csv")     # arbitrary format -> engine dict
result = simulate_replicate_unified(
    primer_seqs=primer_seqs,
    params=DEFAULT_PARAMETERS,
    t_end=90.0,        # simulated seconds
    rng_seed=42,       # fixed seed -> reproducible run
)
print(result["n_products"], result["total_nt"][-1])
```

## Primer input (arbitrary formats)

The engine consumes a flat dictionary of uppercase sequences in which every primer is
accompanied by its reverse complement (`<NAME>` and `<NAME>_RC`). You rarely build this
by hand — `tools/load_primers.py` adapts flexible inputs into it:

| Input | How |
|-------|-----|
| CSV / TSV file | `load_primers("db.csv")` — first row; or `load_primers("db.csv", name="MySet")` to pick a row by a name/set/prefix column |
| Python dict | `load_primers({"F3": "...", "B3": "...", "FIP": "...", "BIP": "...", "LF": "...", "LB": "..."})` |

Column names are matched case-insensitively and ignore spaces/underscores
(`FIP`, `fip`, `F_I_P` all map to `FIP`). **`F3`, `B3`, `FIP`, `BIP` are required;
`LF`, `LB` are optional.** Example file: [`examples/data/demo_primers.csv`](examples/data/demo_primers.csv).

```bash
# Inspect the engine dict a file produces:
python tools/load_primers.py examples/data/demo_primers.csv
```

## Simulation output schema

`simulate_replicate_unified(...)` returns a `dict`. The keys most tools use:

| Key | Type | Meaning |
|-----|------|---------|
| `time` | `list[float]` | event time points (s) |
| `total_nt` | `list[int]` | cumulative nucleotides synthesized (the amplification trace) |
| `reads` | `list[dict]` | per-product records (see below) |
| `n_products` | `int` | number of products emitted |
| `n_species` | `int` | distinct product species |
| `stop_reason` | `str` | `completed`, `hard_stop_products`, `hard_stop_species`, … |
| `count_by_source` | `dict` | event counts by mechanism (`inter`/`intra`/`loop`/`product_bind`/`product_ext`) |
| `lineage` | `list[tuple]` | `(time, info, product_id, sequence)` genealogy entries |

Each element of `reads` has: `id`, `template_id` (parent), `sequence`, `segments`
(`list[Segment(start, end, label)]`), `time_emitted`, `label` (primer family),
`kind` (`inter`/`intra`/`loop`/`product`), and `L` (toehold length). The `id` /
`template_id` pair forms the directed genealogy that `plot_lineage` draws.

> Tip: pickle the result (`examples/run_simulation.py` does this) so you can re-render
> figures without re-running the simulation.

## Tools

Standardized, schema-driven commands that turn a pickled result into a figure. Each is
both a CLI and an importable function.

| Tool | Produces |
|------|----------|
| `tools/load_primers.py` | engine-ready primer dict from CSV/TSV/dict (see above) |
| `tools/plot_amplification.py` | cumulative-nt amplification curve(s); overlays multiple results/replicates |
| `tools/plot_lineage.py` | product genealogy — each product at (emission time, length), edges parent→child, colored by mechanism |
| `tools/plot_autocat_network.py` | autocatalytic segment-transition network (nodes = primer families, edges = transition frequency; self-loops = hairpin turnbacks) |

**Additional capabilities planned for standardization** (currently in the research
codebase, not yet packaged here): batch/parallel replicate runner with live progress,
inflection-time and amplification-rate classification, empirical-vs-simulated transition
matrix comparison, product-length distributions, and edit-distance scoring against
nanopore reads.

## Project layout

```
lamprey/                 # the engine (importable package)
├── model_unified.py     # UnifiedSSAEngine, simulate_replicate_unified, SimulationRunner
├── molecule_pool.py     # MoleculePool / Molecule / BindingComplex / lineage / caps
├── sites.py             # binding-site indexing & weighted sampling
├── parameters.py        # ParameterSet + DEFAULT_PARAMETERS
├── thermo.py            # duplex ΔG
├── emergent.py          # TdT tailing
├── utils.py             # reverse complement, sequence helpers, RNG
├── datamodel.py         # PrimerSet / Segment dataclasses
└── segmentation/        # nanopore read segmenter
tools/                   # primer adapter + standardized visualizations
examples/                # runnable example + synthetic demo primer set
tests/                   # invariant-style smoke tests
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite favors **invariant properties** (determinism under a fixed seed, monotonic
amplification trace, conservation, primer-loader round-trip) over brittle recorded
"golden" numbers, so it stays meaningful as the model is retuned.

## Biological glossary

- **Primer** — a short single-stranded DNA oligo that anneals to a complementary region and gives a polymerase a 3′ end to extend from. LAMP uses six (F3, B3, FIP, BIP, LF, LB).
- **Composite (inner) primer** — FIP and BIP; each fuses two target-binding segments into one strand, which is what enables the folded dumbbell product.
- **Toehold** — a short stretch of complementary bases (here as few as ~3) where two strands first make contact before zipping up into a longer duplex.
- **Hairpin / stem-loop** — a single strand folded back on itself so part of it base-pairs internally, leaving an unpaired loop; the structural seed of the LAMP dumbbell.
- **Strand displacement** — a polymerase copying along a template while peeling off (displacing) a strand already paired there, instead of stopping; essential to isothermal LAMP.
- **Autocatalysis (cascade)** — a self-amplifying loop in which the products of the reaction catalyze production of more of themselves; the source of LAMP's exponential growth.
- **Spurious / template-free amplification** — amplification arising from primer–primer artifacts with no target DNA present; the cause of diagnostic false positives.
- **Propensity** — the instantaneous rate of a specific reaction given the current molecule counts; the quantity the stochastic algorithm samples to decide what happens next.
- **Gillespie SSA** — an exact method for simulating well-mixed chemical reactions one discrete event at a time, capturing the randomness that matters when molecule numbers are small.
- **Nanopore read** — a sequence obtained by threading a single DNA molecule through a nanopore sensor; here, used to read out the actual artifact products of a spurious LAMP reaction and reconstruct how they were assembled.

## License & notes

Released under the [MIT License](LICENSE) © 2026 Mark Knappenberger, Karen S. Anderson.

The bundled `examples/data/demo_primers.csv` is a **synthetic** primer set with no
biological provenance, included so the example runs out of the box. It does not
correspond to any real assay or organism.
