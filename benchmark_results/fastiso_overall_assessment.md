# FastIso Overall Assessment

Generated on 2026-06-07 from the current benchmark CSV files in
`benchmark_results/`.

## Positioning

FastIso should be positioned as a fast dense isotope-profile simulator for
instrument-broadened fine-isotopic envelopes, especially when the target is a
local m/z window or repeated server-side evaluation from cached FT tables.

It is not currently best framed as a faster exact fine-peak enumerator. IsoSpec,
pyOpenMS, and enviPat enumerate isotope peaks first and then optionally convolve
them. Brainpy is fast, but it is an aggregated isotopic-variant model and does
not preserve fine structure in the same sense. FastIso's strongest claim is:

- reuse a precomputed log/phase FT table;
- prune inactive frequency bins from Gaussian attenuation;
- evaluate only a requested profile window with CZT;
- keep production tables compact enough for server caching.

The mass range should be described in three tiers:

- practical target range: routine use up to roughly 1 MDa;
- experimentally relevant range: above the reported high-mass region where
  resolved isotope patterns have been measured, around 466 kDa;
- numerical stress-test range: synthetic CHNOS formulas pushed far beyond
  realistic high-resolution MS use, up to 1.94 GDa in the current benchmark.

The point of the GDa-scale runs is not that such molecules are routine MS
targets. The point is that the algorithmic and numerical path remains stable
well beyond the experimentally relevant range, so for ordinary high-mass use
the limiting factor is more likely instrument resolution, data interpretation,
or table memory policy than residual-spectrum calculation.

## External Package Comparison

Batch-size 10 profile timings from `external_package_benchmark.csv`:

| family | mean mass | FastIso | IsoSpecPy convolved | IsoSpecPy peaks only | Brainpy convolved | FastIso rel L2 vs IsoSpec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 0.52 kDa | 0.0322 s | 0.0169 s | 0.00311 s | 0.00862 s | 0.180 |
| medium | 2.42 kDa | 0.0118 s | 0.0959 s | 0.00189 s | 0.0139 s | 0.00118 |
| large | 12.1 kDa | 0.00758 s | 1.448 s | 0.0105 s | 0.0175 s | 0.00107 |

Interpretation:

- Small molecules are not the target. Peak enumeration has only a few peaks and
  FastIso pays table/spectrum overhead.
- Around 2.4 kDa, FastIso is already about 8.1x faster than convolved IsoSpecPy
  for dense profiles, but peaks-only IsoSpecPy remains faster if the user only
  wants peak lists.
- Around 12 kDa, FastIso is about 191x faster than convolved IsoSpecPy in the
  batch profile benchmark and about 1.4x faster than IsoSpecPy peaks-only.
- Brainpy can be fast, but its profile has materially worse agreement with the
  fine-structure reference in the tested cases.

Fine-structure timings from `fine_structure_benchmark.csv` at R = 240,000:

| case | FastIso profile | IsoSpecPy profile | pyOpenMS profile | enviPat profile | Brainpy profile | FastIso rel L2 | Brainpy rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glucose | 0.00122 s | 0.000217 s | 0.000249 s | 0.280 s | 0.000181 s | 4.38e-4 | 1.64e-2 |
| 2.4 kDa | 0.00380 s | 0.00442 s | 0.00857 s | 0.346 s | 0.000934 s | 1.11e-3 | 1.70e-1 |
| 12 kDa | 0.00884 s | 0.204 s | 0.362 s | 0.731 s | 0.00265 s | 1.06e-3 | 4.57e-2 |

For the 12 kDa nominal +10 cluster, IsoSpecPy found 495 reference fine peaks.
Brainpy represented the same cluster with 1 comparison peak, while pyOpenMS and
enviPat had 767 and 744 comparison peaks. This is the cleanest evidence that
Brainpy is not a like-for-like fine-structure competitor.

## Internal Optimizations

Large-scale CHNOS timing from `large_scale_benchmark.csv` showed that Cython
Gaussian-pruned residual generation is the main speed path:

| case | log table | Cython pruned | speedup |
| --- | ---: | ---: | ---: |
| 12 kDa | 0.0245 s | 0.00198 s | 12.4x |
| 29 kDa | 0.0225 s | 0.00147 s | 15.3x |
| 60 kDa | 0.0423 s | 0.00176 s | 24.1x |
| 120 kDa | 0.0628 s | 0.00224 s | 28.0x |
| 240 kDa | 0.0607 s | 0.00338 s | 18.0x |

The active frequency fraction dropped from 0.079 at 12 kDa to 0.0007 at
240 kDa, so large molecules are exactly where Gaussian cutoff becomes valuable.

Float32 attenuation and production storage are primarily memory optimizations:

| variant | median memory vs research64 | median speed ratio | max rel L2 |
| --- | ---: | ---: | ---: |
| production32_auto | 0.681 | 0.976x | 2.61e-8 |
| minimal32_auto | 0.468 | 0.984x | 2.61e-8 |

OpenMP row-level parallelism is useful for batches, not for every single-formula
case:

| workers | median speedup |
| ---: | ---: |
| 2 | 1.49x |
| 4 | 1.67x |

By batch size, 4 workers gave about 1.08x for batch 1, 1.55x for batch 10,
3.02x for batch 50, and 2.55x for batch 100. All parallel comparisons had zero
relative L2 difference against the one-worker output.

## CZT Window Value

The CZT path is currently most valuable when the requested output window is
small relative to the full FFT profile. In `czt_window_benchmark.csv`, the
240 kDa case reached about 7.2x full-profile-over-CZT speedup for 0.1 to 0.2 Da
windows at 0.001 Da output spacing. Grid-matched comparisons were at numerical
precision, around 1e-15 to 1e-14 relative L2.

This supports a server-oriented use case: cache the production FT table, then
return only the m/z region requested by the frontend or downstream analysis.

## Ultra-Scale Push

The ultra-scale benchmark scales a synthetic CHNOS formula from the 240 kDa
base composition. These formulas are stress tests, not claims about chemically
typical molecules.

This section should not be interpreted as a proposed routine application range.
For product and paper positioning, the practical ceiling is closer to 1 MDa.
The ultra-scale benchmark is included to show that the current implementation
has a large safety margin beyond that practical ceiling and beyond the reported
resolved-isotope-pattern mass range around 466 kDa.

| scale vs 240 kDa | batch | mass | n_fft | table | build | active fraction | CZT window | full profile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x | 10 | 30.3 MDa | 1,240,029 | 75.7 MiB | 0.84 s | 2.42e-5 | 0.036 s | 0.058 s |
| 256x | 10 | 60.6 MDa | 2,066,715 | 126 MiB | 1.48 s | 1.45e-5 | 0.078 s | 0.099 s |
| 512x | 10 | 121 MDa | 3,720,087 | 227 MiB | 3.53 s | 8.06e-6 | 0.113 s | 0.191 s |
| 1024x | 10 | 242 MDa | 6,751,269 | 412 MiB | 8.97 s | 4.44e-6 | 0.235 s | 0.495 s |
| 2048x | 10 | 485 MDa | 13,286,025 | 811 MiB | 8.36 s | 2.26e-6 | 0.403 s | 0.677 s |
| 4096x | 1 | 969 MDa | 26,040,609 | 1.55 GiB | 30.2 s | 1.15e-6 | 0.106 s | 1.71 s |
| 8192x | 1 | 1.94 GDa | 51,667,875 | 3.08 GiB | 90.3 s | 5.81e-7 | 0.218 s | not measured |

The largest completed run was 1.94 GDa with a 3.08 GiB production table. Full
dense profile generation was intentionally skipped at that scale because the
target server use case is windowed output. A 969 MDa case did complete both
windowed and full-profile paths.

Current practical limit:

- The residual-generation kernel itself is no longer the bottleneck after
  Gaussian cutoff; active fractions become extremely small.
- Table build time and table memory become the real ceiling.
- Full dense profile output becomes less meaningful as a server result because
  it returns tens of millions of grid points.
- CZT/windowed output remains practical much longer because it returns only the
  requested local profile.

## Where This Can Go

Most defensible applications:

- server-side isotope-profile simulation where tables are cached and only
  profiles are returned;
- high-throughput screening of large biomolecules, proteoforms, glycoforms,
  polymers, or adduct variants where dense instrument-broadened envelopes are
  needed repeatedly;
- local m/z-window inspection around expected charge-state or neutral-mass
  regions;
- workflows where exact peak enumeration is too expensive or unnecessary after
  instrument broadening.

Less defensible as the primary claim:

- replacing IsoSpec as an exact fine-peak enumerator;
- outperforming peak-list algorithms for small molecules;
- claiming universal superiority over Brainpy without stating that Brainpy
  targets aggregated isotope variants rather than fine-structure preservation.

## Publication Potential

The current evidence is promising, but the strongest paper angle is narrow and
technical:

> A cached FT/log-table and CZT-window method for fast dense
> instrument-broadened isotope-profile simulation of large molecular formulas.

Before a paper-quality benchmark, the next required work is:

- validate more elements and isotope datasets beyond CHNOS;
- lock package versions, CPU information, thread settings, and exact benchmark
  commands;
- compare against external tools over more realistic formulas and charge states;
- report accuracy as profile-level error, apex shift, cluster probability, and
  fine-cluster preservation;
- separate table-build cost from cached-query cost in all benchmark tables;
- add memory scaling plots, because memory is now the limiting factor.

With those additions, FastIso has a credible niche: not as a universal isotope
algorithm replacement, but as a practical server/cache/window engine for large
instrument-broadened profile simulation.
