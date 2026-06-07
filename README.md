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
- small-state exact Gaussian bin-profile backend for narrow isotope patterns;
- command-line interface;
- portable Python tkinter GUI;
- FastAPI prototype endpoint: `/simulate/window`;
- benchmark scripts and saved benchmark outputs.

Tested on the local development environment:

```text
120 passed, 6 skipped
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

## Command Line

Editable/package installs expose the `fastiso` command:

```powershell
.\.venv\Scripts\fastiso isotopes list --preset common
.\.venv\Scripts\fastiso isotopes inspect K --preset full --format json
```

Simulate a full dense profile as CSV:

```powershell
.\.venv\Scripts\fastiso simulate C500H800N125O200S10 `
  --elements C H N O S `
  --dm 0.002 `
  --rp 100000 `
  --method cython_auto `
  --workers 4 `
  --output profile.csv
```

Simulate only a local window while keeping the table spacing separate from the
output spacing. With `--start/--stop`, the default `auto` window mode treats the
range as residual mass relative to the formula mean:

```powershell
.\.venv\Scripts\fastiso window C500H800N125O200S10 `
  --elements C H N O S `
  --start -0.5 `
  --stop 0.5 `
  --output-dm 0.001 `
  --dm 0.002 `
  --rp 100000 `
  --method cython_auto `
  --workers 4 `
  --format json `
  --output window.json
```

By default, profile commands use resolving power 100,000. Pass
`--gaussian-sigma` instead when a fixed Gaussian sigma in mass units is desired.

For small or skewed formulas, a mean-centered fixed residual window can miss the
largest peak. Use adaptive mode, or omit `--start/--stop` in the default `auto`
mode, to choose a residual window from the estimated isotope support:

```powershell
.\.venv\Scripts\fastiso window S10 `
  --elements S `
  --window-mode adaptive `
  --dm 0.002 `
  --output-dm 0.002
```

For small formulas, adaptive mode first tries exact isotope-support windowing
with `--auto-window-cutoff` and falls back to sigma-based sizing when the exact
support would be too large. When the exact support is small enough, FastIso also
uses a direct Gaussian bin-profile backend instead of FT/CZT reconstruction.
This avoids ringing and keeps all meaningful peaks in view for cases such as
`Cl` through `Cl6`:

```powershell
.\.venv\Scripts\fastiso window Cl Cl2 Cl3 Cl4 Cl5 Cl6 `
  --elements Cl `
  --window-mode adaptive `
  --auto-grid
```

Adaptive mode solves the window placement problem. For large formulas where the
exact backend is not used, very narrow peaks may still need `--auto-grid`, a
smaller `--dm`, or larger `--gaussian-sigma` to avoid sinc-like ringing in the
sampled dense profile.

Use auto-grid when the desired resolving power or Gaussian width should control
the sampling grid:

```powershell
.\.venv\Scripts\fastiso window S `
  --elements S `
  --start -0.100 `
  --stop -0.085 `
  --auto-grid `
  --samples-per-fwhm 8
```

With `--auto-grid`, FastIso chooses `dm` from the narrowest requested peak width
so each FWHM has the requested number of samples. This is important for single
atoms and small molecules, where the default `dm` can be too coarse.

Bracketed formula groups are supported by the shared parser. Quote bracketed
formulas in shells:

```powershell
.\.venv\Scripts\fastiso window "(CH3OH)2(HCl)2" `
  --elements C H O Cl `
  --start -0.5 `
  --stop 0.5 `
  --output-dm 0.001
```

## Portable Python GUI

The lightweight GUI uses Python's standard `tkinter` package, so it can run
without building an `.exe`:

```powershell
.\.venv\Scripts\fastiso-gui
```

or:

```powershell
.\.venv\Scripts\python -m fastiso.gui
```

The GUI accepts the same formula syntax as the CLI, including adjacent and
nested bracketed groups such as `(CH3OH)2(HCl)2` and `K4[Fe(CN)6]`. Its default
`auto` mode chooses an adaptive local window and enables Auto grid; choose
`full` only when a complete dense profile is needed. The preview table and plot
use peak-preserving display sampling so narrow small-formula isotope peaks are
not hidden by dense output grids.

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
