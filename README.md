# FastIso

FastIso is a fast Fourier-transform based isotope profile simulator for large
instrument-broadened molecular formulas.

The core idea is to reuse a cached log/phase isotope FT table, prune inactive
frequency bins after Gaussian broadening, and use CZT to evaluate only the mass
window requested by the caller. This makes FastIso most useful for server-side
or batch workflows where dense isotope profiles are needed repeatedly.

FastIso is a research prototype. It is designed around dense broadened profile
simulation, not small-molecule peak-list generation.

## What FastIso Is Good At

- Dense instrument-broadened isotope profiles for large formulas.
- Local m/z window simulation with CZT.
- Repeated queries from cached FT tables.
- Server-style workloads where isotope data and tables stay on the backend.
- Stress testing formulas far beyond routine high-mass MS use.

## What FastIso Is Not

- A universal isotope simulation method for every use case.
- A small-molecule-first peak-list generator.
- A substitute for method validation on the user's own formulas and instrument
  settings.

FastIso's strongest niche is cached dense profile simulation after instrument
broadening, especially when only a local mass window is needed.

## Current Status

Implemented:

- odd-length fast `irfft` helper;
- centered log/phase isotope FT table;
- Gaussian frequency pruning;
- optional Cython kernels;
- OpenMP row-level parallelism for batch calculations;
- float32 production table storage;
- CZT windowed profile evaluation;
- versioned isotope data registry and presets;
- monoisotopic element mass-shift handling;
- FastAPI prototype endpoint: `/simulate/window`;
- benchmark scripts and saved benchmark outputs.

Tested on the local development environment:

```text
99 passed, 6 skipped
```

## Installation

Create a virtual environment, then install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev,server]"
```

The pure Python path works without Cython. For the faster Cython backend on
Windows, install Microsoft C++ Build Tools, then run:

```powershell
$env:FASTISO_BUILD_CYTHON = "1"
$env:FASTISO_BUILD_OPENMP = "1"
.\.venv\Scripts\python -m pip install -e . --no-build-isolation --no-deps
```

## Quick Example

```python
from fastiso import CenteredLogPhaseTable

formula = "C500H800N125O200S10"

table = CenteredLogPhaseTable.build_for_formulas(
    [formula],
    elements=("C", "H", "N", "O", "S"),
    dm=0.002,
    resolving_power=100_000,
    storage_mode="production",
)
counts = table.counts_from_formulas([formula])

mass_axis, profile, info = table.mass_profile_many_counts(
    counts,
    method="cython_auto",
    resolving_power=100_000,
)
```

## CZT Windowed Profiles

Use `mass_profile_window_many_counts` when only a local mass window is needed or
when the requested output spacing differs from the FFT table spacing:

```python
mass_axis, profile, info = table.mass_profile_window_many_counts(
    counts,
    residual_start=-0.5,
    residual_stop=0.5,
    output_dm=0.001,
    method="cython_auto",
    resolving_power=100_000,
    workers=4,
)
```

The CZT path evaluates the inverse transform only at the requested regular
window samples. Integer table-grid samples match the full profile path.

## Isotope Data

FastIso loads versioned isotope data from packaged JSON resources:

- `src/fastiso/isotope_data/common.json`
- `src/fastiso/isotope_data/full.json`
- source metadata and version are stored with the data resource;
- presets: `bio`, `organic`, `halogen`, `adduct`, `common`, `full`.

The current `common` dataset covers:

```text
H, Li, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Fe, Ni, Cu, Zn, Se, Br, I
```

Single-isotope elements are treated as deterministic mass shifts instead of FT
table rows. For example, F, Na, Al, P, and I shift the returned mass axis but do
not increase spectral table size.

The `full` preset loads `full.json` and covers 80 real elements with at least
one stable isotope. Pseudo symbols and elements with no stable isotope, such as
Bi, Th, Pa, U, and synthetic unstable elements, are excluded. Natural-abundance
isotope rows are retained, including long-lived naturally occurring isotopes
such as K-40, Ca-48, Xe-136, and Lu-176, because they can contribute
mass-spectrometry signal. Strict stable-isotope-only datasets can be built as
custom isotope resources when needed.

```python
from fastiso import load_isotope_patterns, load_isotope_registry

registry = load_isotope_registry()
patterns = load_isotope_patterns(preset="common")
full_patterns = load_isotope_patterns(preset="full")
```

## Server Prototype

Run the FastAPI server:

```powershell
.\.venv\Scripts\python -m uvicorn fastiso.server:app --reload
```

Example request:

```json
{
  "formula": "C500H800N125O200S10",
  "preset": "common",
  "resolving_power": 100000,
  "table_dm": 0.002,
  "window": {"mode": "residual", "start": -1.0, "stop": 1.0},
  "output_dm": 0.0005,
  "method": "cython_auto",
  "workers": 4
}
```

The response contains `mass_axis`, `intensity`, summary metrics, runtime
metadata, isotope data version, and the table cache key.

## Benchmarks

Saved benchmark summaries are in `benchmark_results/`. The main overview is:

- `benchmark_results/fastiso_overall_assessment.md`

Key local observations from the current benchmark set:

- small molecules are not the target;
- 12 kDa dense profile benchmark showed a clear speed advantage over
  peak-enumeration-plus-convolution workflows in the tested environment;
- aggregated peak-list style outputs do not represent the same fine-structure
  information as dense broadened profiles;
- 240 kDa CZT local-window profile reached about 7x speedup over full dense
  profile evaluation for small windows;
- synthetic CHNOS stress tests completed windowed simulation up to 1.94 GDa,
  far beyond the expected practical range.

Practical positioning:

- routine target range: up to roughly 1 MDa;
- experimentally relevant high-mass isotope-pattern range: around hundreds of
  kDa;
- GDa-scale formulas are numerical stress tests, not routine MS targets.

Run selected benchmarks:

```powershell
.\.venv\Scripts\python benchmarks\fine_structure.py
.\.venv\Scripts\python benchmarks\external_packages.py
.\.venv\Scripts\python benchmarks\czt_window.py
.\.venv\Scripts\python benchmarks\large_scale.py
.\.venv\Scripts\python benchmarks\ultra_scale.py
```

Optional external baselines require additional packages:

```powershell
.\.venv\Scripts\python -m pip install -e ".[bench]"
```

## Development Checks

```powershell
.\.venv\Scripts\python -m pytest -q
```

The Cython parity tests are skipped when the extension is not built.

## Public API Surface

The currently intended public entry points are:

- `CenteredLogPhaseTable`
- `mass_profile_many_counts`
- `mass_profile_window_many_counts`
- `load_isotope_registry`
- `load_isotope_patterns`
- `split_formula_isotope_components`
- `/simulate/window`

This API may still change while the project is pre-1.0.

## License

FastIso is licensed under the Apache License 2.0. See `LICENSE`.
