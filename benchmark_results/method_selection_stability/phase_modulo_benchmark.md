# Phase Modulo Benchmark

- Python: 3.13.13
- Platform: Windows-10-10.0.19045-SP0
- Measured kernel: residual spectrum generation only.
- Ratio columns above 1 mean the alternative phase reduction is faster.

| case | batch | R | n_fft | active | max active phase rad | baseline s | fmod s | cycle s | uint s | threshold s | auto s | auto selected | baseline/fmod | baseline/cycle | baseline/uint | baseline/threshold | baseline/auto | auto rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huge_120k | 10 | 0 | 59049 | 0.2052 | 1.23e+05 | 0.005083 | 0.006519 | 0.005684 | 0.005426 | 0.005679 | 0.006512 | cython_log_pruned_uintphase_threshold | 0.780x | 0.894x | 0.937x | 0.895x | 0.781x | 1.89e-12 |
| huge_120k | 10 | 100000 | 59049 | 0.001998 | 983 | 0.002466 | 0.003506 | 0.002571 | 0.002685 | 0.000578 | 0.000637 | cython_log_pruned_uintphase_threshold | 0.703x | 0.959x | 0.918x | 4.264x | 3.870x | 6.48e-13 |
| huge_240k | 10 | 0 | 83349 | 0.03132 | 1.87e+05 | 0.004510 | 0.004488 | 0.004405 | 0.004142 | 0.003753 | 0.004107 | cython_log_pruned_uintphase_threshold | 1.005x | 1.024x | 1.089x | 1.202x | 1.098x | 3.73e-12 |
| huge_240k | 10 | 100000 | 83349 | 0.0007439 | 975 | 0.003006 | 0.003289 | 0.003036 | 0.003276 | 0.000752 | 0.000757 | cython_log_pruned_uintphase_threshold | 0.914x | 0.990x | 0.917x | 3.997x | 3.971x | 2.22e-12 |
| huge_60k | 10 | 0 | 42525 | 0.4183 | 6.25e+04 | 0.005439 | 0.006986 | 0.006242 | 0.008055 | 0.006612 | 0.005442 | cython_log_pruned | 0.779x | 0.871x | 0.675x | 0.823x | 1.000x | 0.00e+00 |
| huge_60k | 10 | 100000 | 42525 | 0.005526 | 999 | 0.001776 | 0.001748 | 0.001618 | 0.001815 | 0.000516 | 0.000570 | cython_log_pruned_uintphase_threshold | 1.016x | 1.098x | 0.979x | 3.445x | 3.116x | 2.87e-13 |
| large_12k | 10 | 0 | 32805 | 1 | 1.41e+04 | 0.008252 | 0.010999 | 0.009778 | 0.008468 | 0.012112 | 0.009111 | cython_log_pruned | 0.750x | 0.844x | 0.975x | 0.681x | 0.906x | 0.00e+00 |
| large_12k | 10 | 100000 | 32805 | 0.07566 | 1.18e+03 | 0.001885 | 0.002738 | 0.001985 | 0.001838 | 0.001134 | 0.001355 | cython_log_pruned_uintphase_threshold | 0.689x | 0.950x | 1.026x | 1.663x | 1.391x | 1.99e-13 |
| xlarge_29k | 10 | 0 | 32805 | 0.7911 | 3.13e+04 | 0.006769 | 0.012906 | 0.008585 | 0.007106 | 0.009192 | 0.007015 | cython_log_pruned | 0.524x | 0.788x | 0.953x | 0.736x | 0.965x | 0.00e+00 |
| xlarge_29k | 10 | 100000 | 32805 | 0.01914 | 1.12e+03 | 0.001402 | 0.001667 | 0.001491 | 0.001432 | 0.000575 | 0.000601 | cython_log_pruned_uintphase_threshold | 0.841x | 0.940x | 0.980x | 2.439x | 2.332x | 4.03e-13 |
