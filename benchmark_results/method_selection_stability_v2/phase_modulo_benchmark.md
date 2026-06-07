# Phase Modulo Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Measured kernel: residual spectrum generation only.
- Ratio columns above 1 mean the alternative phase reduction is faster.

| case | batch | R | n_fft | active | max active phase rad | baseline s | fmod s | cycle s | uint s | threshold s | auto s | auto selected | baseline/fmod | baseline/cycle | baseline/uint | baseline/threshold | baseline/auto | auto rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huge_120k | 10 | 0 | 59049 | 0.2052 | 1.23e+05 | 0.005759 | 0.006646 | 0.006100 | 0.006854 | 0.007361 | 0.007388 | cython_log_pruned | 0.866x | 0.944x | 0.840x | 0.782x | 0.779x | 0.00e+00 |
| huge_120k | 10 | 100000 | 59049 | 0.001998 | 983 | 0.002308 | 0.002551 | 0.002382 | 0.002245 | 0.000583 | 0.000718 | cython_log_pruned_uintphase_threshold | 0.905x | 0.969x | 1.028x | 3.957x | 3.215x | 6.48e-13 |
| huge_240k | 10 | 0 | 83349 | 0.03132 | 1.87e+05 | 0.004332 | 0.004900 | 0.004350 | 0.004143 | 0.004120 | 0.003715 | cython_log_pruned_uintphase_threshold | 0.884x | 0.996x | 1.046x | 1.051x | 1.166x | 3.73e-12 |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | 975 | 0.003184 | 0.003377 | 0.003289 | 0.003147 | 0.000748 | 0.000797 | cython_log_pruned_uintphase_threshold | 0.943x | 0.968x | 1.012x | 4.255x | 3.996x | 2.22e-12 |
| huge_60k | 10 | 0 | 42525 | 0.4183 | 6.25e+04 | 0.005301 | 0.007007 | 0.006085 | 0.005587 | 0.006959 | 0.005708 | cython_log_pruned | 0.757x | 0.871x | 0.949x | 0.762x | 0.929x | 0.00e+00 |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | 999 | 0.001621 | 0.001660 | 0.001859 | 0.001700 | 0.000468 | 0.000523 | cython_log_pruned_uintphase_threshold | 0.976x | 0.872x | 0.954x | 3.464x | 3.102x | 2.87e-13 |
| large_12k | 10 | 0 | 32805 | 1 | 1.41e+04 | 0.008218 | 0.010753 | 0.011807 | 0.009601 | 0.010295 | 0.008154 | cython_log_pruned | 0.764x | 0.696x | 0.856x | 0.798x | 1.008x | 0.00e+00 |
| large_12k | 10 | 100000 | 32805 | 0.07566 | 1.18e+03 | 0.001794 | 0.002257 | 0.001954 | 0.001793 | 0.001318 | 0.001282 | cython_log_pruned_uintphase_threshold | 0.795x | 0.918x | 1.001x | 1.362x | 1.399x | 1.99e-13 |
| xlarge_29k | 10 | 0 | 32805 | 0.7911 | 3.13e+04 | 0.006829 | 0.009074 | 0.007852 | 0.008049 | 0.009105 | 0.006815 | cython_log_pruned | 0.753x | 0.870x | 0.848x | 0.750x | 1.002x | 0.00e+00 |
| xlarge_29k | 10 | 100000 | 32805 | 0.01914 | 1.12e+03 | 0.001513 | 0.001532 | 0.001378 | 0.001364 | 0.000550 | 0.000835 | cython_log_pruned_uintphase_threshold | 0.988x | 1.098x | 1.110x | 2.752x | 1.812x | 4.03e-13 |
