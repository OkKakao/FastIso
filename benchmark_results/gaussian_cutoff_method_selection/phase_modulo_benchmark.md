# Phase Modulo Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Measured kernel: residual spectrum generation only.
- Ratio columns above 1 mean the alternative phase reduction is faster.

| case | batch | R | n_fft | active | max active phase rad | baseline s | fmod s | cycle s | uint s | threshold s | auto s | auto selected | baseline/fmod | baseline/cycle | baseline/uint | baseline/threshold | baseline/auto | auto rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huge_120k | 10 | 0 | 59049 | 0.2052 | 1.23e+05 | 0.005576 | 0.006976 | 0.005748 | 0.005297 | 0.006022 | 0.005649 | cython_log_pruned | 0.799x | 0.970x | 1.053x | 0.926x | 0.987x | 0.00e+00 |
| huge_120k | 10 | 100000 | 59049 | 0.001998 | 983 | 0.000123 | 0.000132 | 0.000133 | 0.000145 | 0.000134 | 0.000230 | cython_log_pruned_uintphase_threshold | 0.928x | 0.919x | 0.848x | 0.916x | 0.534x | 6.48e-13 |
| huge_240k | 10 | 0 | 83349 | 0.03132 | 1.87e+05 | 0.004615 | 0.004968 | 0.004643 | 0.006278 | 0.004048 | 0.004406 | cython_log_pruned_uintphase_threshold | 0.929x | 0.994x | 0.735x | 1.140x | 1.048x | 3.73e-12 |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | 975 | 0.000114 | 0.000127 | 0.000130 | 0.000123 | 0.000129 | 0.000189 | cython_log_pruned_uintphase_threshold | 0.900x | 0.873x | 0.924x | 0.882x | 0.604x | 2.22e-12 |
| huge_60k | 10 | 0 | 42525 | 0.4183 | 6.25e+04 | 0.006620 | 0.007999 | 0.006858 | 0.006928 | 0.007837 | 0.006018 | cython_log_pruned | 0.828x | 0.965x | 0.956x | 0.845x | 1.100x | 0.00e+00 |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | 999 | 0.000161 | 0.000180 | 0.000165 | 0.000163 | 0.000177 | 0.000228 | cython_log_pruned_uintphase_threshold | 0.896x | 0.977x | 0.992x | 0.914x | 0.708x | 2.87e-13 |
| large_12k | 10 | 0 | 32805 | 1 | 1.41e+04 | 0.010057 | 0.012842 | 0.010528 | 0.009071 | 0.010955 | 0.009444 | cython_log_pruned | 0.783x | 0.955x | 1.109x | 0.918x | 1.065x | 0.00e+00 |
| large_12k | 10 | 100000 | 32805 | 0.07566 | 1.18e+03 | 0.000848 | 0.001047 | 0.000799 | 0.000749 | 0.002139 | 0.001202 | cython_log_pruned_uintphase_threshold | 0.810x | 1.061x | 1.131x | 0.396x | 0.705x | 1.99e-13 |
| xlarge_29k | 10 | 0 | 32805 | 0.7911 | 3.13e+04 | 0.011937 | 0.012255 | 0.008090 | 0.007126 | 0.010107 | 0.006987 | cython_log_pruned | 0.974x | 1.475x | 1.675x | 1.181x | 1.708x | 0.00e+00 |
| xlarge_29k | 10 | 100000 | 32805 | 0.01914 | 1.12e+03 | 0.000336 | 0.000372 | 0.000346 | 0.000338 | 0.000682 | 0.000476 | cython_log_pruned_uintphase_threshold | 0.903x | 0.970x | 0.994x | 0.493x | 0.707x | 4.03e-13 |
