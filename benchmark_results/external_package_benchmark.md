# External Package Benchmark

- IsoSpecPy available: True
- brainpy available: True
- Cython backend: True
- FastIso uses isotope masses/abundances copied from IsoSpecPy for this benchmark.
- Runtime includes profile generation on the same mass grid; IsoSpecPy peaks-only rows exclude Gaussian convolution.
- With dm=0.002 and R=100000, small-mass profiles are under-sampled; medium/large rows are the relevant comparison.

| family | batch | auto | n_fft | package | method | median s | speedup vs fastiso | peaks | coverage | rel L2 vs IsoSpec |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | 1 | True | 32805 | fastiso | cython_log_pruned_profile | 0.002555 | 1.00x | 0.0 | 1.000000 | 3.21e-01 |
| small | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 0.000622 | 0.24x | 7.0 | 0.999618 | 0.00e+00 |
| small | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.000144 | 0.06x | 7.0 | 0.999618 | NA |
| small | 1 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.000294 | 0.12x | 9.0 | 1.000000 | 7.79e-03 |
| small | 10 | True | 32805 | fastiso | cython_log_pruned_profile | 0.032191 | 1.00x | 0.0 | 1.000000 | 1.80e-01 |
| small | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 0.016931 | 0.53x | 54.9 | 0.999086 | 0.00e+00 |
| small | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.003114 | 0.10x | 54.9 | 0.999086 | NA |
| small | 10 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.008625 | 0.27x | 14.2 | 1.000000 | 8.05e-02 |
| medium | 1 | True | 32805 | fastiso | cython_log_pruned_profile | 0.001192 | 1.00x | 0.0 | 1.000000 | 1.18e-03 |
| medium | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 0.002362 | 1.98x | 216.0 | 0.999009 | 0.00e+00 |
| medium | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.000348 | 0.29x | 216.0 | 0.999009 | NA |
| medium | 1 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.001296 | 1.09x | 18.0 | 1.000000 | 3.71e-02 |
| medium | 10 | True | 32805 | fastiso | cython_log_pruned_profile | 0.011810 | 1.00x | 0.0 | 1.000000 | 1.18e-03 |
| medium | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 0.095941 | 8.12x | 477.3 | 0.999003 | 0.00e+00 |
| medium | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.001887 | 0.16x | 477.3 | 0.999003 | NA |
| medium | 10 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.013910 | 1.18x | 20.6 | 1.000000 | 5.05e-02 |
| large | 1 | True | 32805 | fastiso | cython_log_pruned_profile | 0.000784 | 1.00x | 0.0 | 1.000000 | 1.07e-03 |
| large | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 0.084419 | 107.70x | 7103.0 | 0.999000 | 0.00e+00 |
| large | 1 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.001032 | 1.32x | 7103.0 | 0.999000 | NA |
| large | 1 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.001311 | 1.67x | 35.0 | 1.000000 | 1.25e-02 |
| large | 10 | True | 32805 | fastiso | cython_log_pruned_profile | 0.007580 | 1.00x | 0.0 | 1.000000 | 1.07e-03 |
| large | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_convolved | 1.448215 | 191.07x | 9995.6 | 0.999000 | 0.00e+00 |
| large | 10 | True | 32805 | IsoSpecPy | IsoTotalProb_0.999_peaks_only | 0.010544 | 1.39x | 9995.6 | 0.999000 | NA |
| large | 10 | True | 32805 | brainpy | isotopic_variants_200_convolved | 0.017464 | 2.30x | 37.0 | 1.000000 | 1.30e-02 |
